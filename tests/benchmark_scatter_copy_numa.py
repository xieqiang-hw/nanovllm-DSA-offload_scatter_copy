#!/usr/bin/env python3
"""Compare Host NUMA allocation policies with fresh, synchronized NPU workers.

balance: simultaneous multi-NPU default/concentrated/balanced/interleave trials.
affinity: sequential NPU x Host-node sweep; no assumed PCIe/NUMA affinity map.
Only timing/diagnostic JSON and logs are produced. No kernel or allocator ABI
changes, no page migration, no performance assertions, no external job cleanup.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from numa_support import parse_ids, topology, worker_plans
from test_scatter_copy_multiprocess import parse_devices
from scatter_results import COLUMNS, formatted_row, normalize_case, print_result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("balance", "affinity"), default="balance")
    parser.add_argument("--devices", default="0,1", help="NPU indices in ASCEND_RT_VISIBLE_DEVICES order.")
    parser.add_argument("--nodes", default="auto", help="Host NUMA node list, e.g. 0-7 (not NPU IDs).")
    parser.add_argument("--policies", default="default,concentrated,balanced,interleave")
    parser.add_argument("--concentrated-node", type=int, default=0)
    parser.add_argument("--cpu-bind", choices=("spread", "none"), default="spread")
    parser.add_argument("--cpus-per-worker", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--hbm-slots", type=int, default=8192)
    parser.add_argument("--copy-count", type=int, default=300)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing", choices=("graph", "eager"), default="graph")
    parser.add_argument("--numa-page-samples", type=int, default=1024)
    parser.add_argument("--require-numa-placement", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.iters < 1 or args.warmup < 0 or args.numa_page_samples < 1:
        parser.error("rounds/iters/page samples must be positive; warmup must be non-negative")
    if args.batch_size < 1 or args.source_len < 1 or args.hbm_slots <= args.copy_count:
        parser.error("batch/source must be positive and HBM slots must exceed copy count")
    if not 0 <= args.copy_count <= min(args.source_len, args.copy_cap) or not 0 < args.copy_cap <= 65536:
        parser.error("invalid copy count/capacity")
    return args


def conditions(args, devices, nodes):
    if args.experiment == "affinity":
        return [(f"device{device}_node{node}", [device], "concentrated", node)
                for device in devices for node in nodes]
    policies = args.policies.split(",")
    if not policies or len(set(policies)) != len(policies) or any(
            p not in ("default", "concentrated", "balanced", "interleave") for p in policies):
        raise ValueError("--policies must be a unique list of default,concentrated,balanced,interleave.")
    return [(p, devices, p, args.concentrated_node) for p in policies]


def command_for(args, devices, nodes, policy, node, output, logs):
    command = [
        sys.executable, str(Path(__file__).with_name("test_scatter_copy_multiprocess.py")),
        "--devices", ",".join(map(str, devices)), "--same-seed",
        "--numa-policy", policy, "--numa-nodes", ",".join(map(str, nodes)),
        "--numa-node", str(node), "--cpu-bind", args.cpu_bind,
        "--cpus-per-worker", str(args.cpus_per_worker),
        "--batch-size", str(args.batch_size), "--source-len", str(args.source_len),
        "--hbm-slots", str(args.hbm_slots), "--copy-cap", str(args.copy_cap),
        "--copy-min", str(args.copy_count), "--copy-max", str(args.copy_count),
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--seed", str(args.seed), "--timing", args.timing,
        "--numa-page-samples", str(args.numa_page_samples),
        "--output", str(output), "--worker-log-dir", str(logs),
    ]
    if args.require_numa_placement:
        command.append("--require-numa-placement")
    return command


def placement_load(result):
    """Estimate allocation/read distribution, only when all sampled pages resolve."""
    if not result["summary"]["all_placements_sample_verified"]:
        return None
    resident_bytes = defaultdict(float)
    read_bytes = defaultdict(float)
    for device in result["per_device"]:
        for name, buffer in device["numa"]["buffers"].items():
            sample = buffer["allocation"]
            for node, count in sample["node_pages"].items():
                resident_bytes[node] += buffer["nbytes"] * count / sample["sampled_pages"]
            sample = buffer["active_tokens"]
            if sample:
                total = device["workload"]["copied_tokens"] * (1024 if name == "dram_ckv" else 128)
                for node, count in sample["node_pages"].items():
                    read_bytes[node] += total * count / sample["sampled_pages"]
    return {"estimated_source_bytes_per_node": dict(resident_bytes),
            "estimated_read_bytes_per_node_per_iteration": dict(read_bytes)}


def save_inventory(path):
    result = {"uname": platform.uname()._asdict(), "environment": {
        key: os.environ.get(key) for key in ("ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES",
                                            "ASCEND_LAUNCH_BLOCKING", "CPU_AFFINITY_CONF")}}
    for name, command in (("lscpu", ["lscpu"]), ("numactl", ["numactl", "--hardware"]),
                          ("npu_info", ["npu-smi", "info"]), ("npu_topo", ["npu-smi", "info", "-t", "topo"])):
        try:
            run = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
            result[name] = {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def print_failure_logs(trial: Path):
    """Print bounded evidence from this trial, without hiding its full log files."""
    paths = [trial / "launcher.log", *sorted((trial / "workers").glob("*.log"))]
    for path in paths:
        print(f"SCATTER_NUMA_FAILURE_LOG file={path}", flush=True)
        try:
            tail = deque(maxlen=8)
            with path.open(encoding="utf-8", errors="replace") as file:
                for line in file:
                    # The compact per-buffer record includes both query methods;
                    # don't flood the console with the full placement JSON again.
                    if line.startswith(("A3_SCATTER_NUMA_SETUP ", "A3_SCATTER_NUMA_ADDRESS_CAPTURE ",
                                        "A3_SCATTER_NUMA_BUFFER ")):
                        print(line.rstrip(), flush=True)
                    else:
                        tail.append(line)
            for line in tail:
                text = line.rstrip()
                print(text if len(text) <= 2000 else text[:2000] + " ... [see full log]", flush=True)
        except OSError as error:
            print(f"Cannot read failure log: {error}", flush=True)


def main():
    args = parse_args()
    if os.getenv("ASCEND_LAUNCH_BLOCKING", "0") != "0":
        raise ValueError("Unset ASCEND_LAUNCH_BLOCKING or set it to 0 for a bandwidth experiment.")
    devices = parse_devices(args.devices)
    topo = topology()
    nodes = sorted(topo["nodes"]) if args.nodes == "auto" else parse_ids(args.nodes)
    cases = conditions(args, devices, nodes)
    for label, subset, policy, node in cases:
        plans = worker_plans(len(subset), policy, nodes, topo, args.cpu_bind, args.cpus_per_worker, node)
        print("SCATTER_NUMA_PLAN " + json.dumps({"case": label, "devices": subset, "workers": plans}), flush=True)
    if args.dry_run:
        return
    root = args.output_dir or Path("results/numa") / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    save_inventory(root / "host_before.json")
    print(f"SCATTER_NUMA_BEGIN output={root} timing={args.timing} fresh_processes=1 "
          "same_workload=1 performance_assert=0", flush=True)
    rows = []
    for repeat in range(args.rounds):
        # Rotate order to reduce one-way warmup/thermal drift; seed stays fixed.
        offset = repeat % len(cases)
        for label, subset, policy, node in cases[offset:] + cases[:offset]:
            trial = root / f"round{repeat + 1}_{label}"
            trial.mkdir()
            command = command_for(args, subset, nodes, policy, node, trial / "result.json", trial / "workers")
            print(f"SCATTER_NUMA_TRIAL round={repeat + 1} case={label} devices={subset}", flush=True)
            # The child launcher performs failure cleanup of only its own workers.
            with (trial / "launcher.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    # A caught interrupt lets the launcher execute its finally cleanup.
                    import signal
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                    process.wait()
                    raise
            if code:
                print_failure_logs(trial)
                raise RuntimeError(f"NUMA trial failed ({code}); see {trial}/launcher.log and {trial}/workers/*.log")
            result = json.loads((trial / "result.json").read_text())
            summary = result["summary"]
            row = {"case": label, "round": repeat + 1, "devices": ",".join(map(str, subset)),
                   "policy": policy, "node": node if policy == "concentrated" else "",
                   **normalize_case(result),
                   "host_start_skew_us": summary["host_start_skew_us"],
                   "host_interval_overlap_fraction": summary["host_interval_overlap_fraction"],
                   "placement_verified": summary["all_placements_sample_verified"]}
            rows.append(row)
            print(f"SCATTER_NUMA_RESULT case={label} round={repeat + 1}", flush=True)
            print_result(row)
            print("SCATTER_NUMA_NODE_LOAD " + json.dumps(placement_load(result)), flush=True)
            # Checkpoint after each trial, keeping results if a later one fails.
            with (root / "trials.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(formatted_row(item) for item in rows)
            # Trial identity and placement diagnostics remain available without
            # adding columns or another bandwidth definition to the CSV.
            (root / "trials.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    aggregate = []
    for label, _, _, _ in cases:
        samples = [row for row in rows if row["case"] == label]
        worst = [row["avg_us_max"] for row in samples]
        item = {"case": label, "rank_max_us_median": statistics.median(worst),
                "rank_max_us_min": min(worst), "rank_max_us_max": max(worst),
                "placement_verified_all_rounds": all(row["placement_verified"] for row in samples)}
        aggregate.append(item)
    (root / "summary.json").write_text(json.dumps({"config": {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(args).items()}, "topology": topo,
                                     "summary": aggregate, "trials": rows}, indent=2), encoding="utf-8")
    save_inventory(root / "host_after.json")
    print(f"SCATTER_NUMA_DONE output={root} "
          "note=unverified_placement_cannot_establish_a_NUMA_effect", flush=True)


if __name__ == "__main__":
    main()
