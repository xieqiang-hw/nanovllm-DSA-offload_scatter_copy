#!/usr/bin/env python3
"""Run one A3 SCATTER worker process per visible NPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from scatter_cli import add_copy_count_arg, add_device_args, resolve_copy_count, selected_devices, validate_workload
from scatter_results import copy_label, print_result, summarize_devices

from numa_support import POLICIES, parse_ids, topology, worker_plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_args(parser, "0")
    parser.add_argument(
        "--output", type=Path, help="Case JSON path (default: branch results directory)."
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--hbm-slots", type=int, default=8192)
    add_copy_count_arg(parser)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing", choices=("eager", "graph"), default="eager")
    parser.add_argument("--numa-policy", choices=POLICIES, default="inherit")
    parser.add_argument("--numa-nodes", default="auto", help="Allowed Host NUMA nodes, e.g. 0-7.")
    parser.add_argument("--numa-node", type=int, default=0, help="Node for concentrated allocation.")
    parser.add_argument("--numa-node-map", help="One memory node per --devices entry; requires mapped policy.")
    parser.add_argument("--cpu-bind", choices=("none", "spread"), default="none")
    parser.add_argument("--cpus-per-worker", type=int, default=4)
    parser.add_argument("--numa-page-samples", type=int, default=1024)
    parser.add_argument("--require-numa-placement", action="store_true")
    parser.add_argument("--worker-log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print plans without importing Torch or starting NPUs.")
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="Use exactly the same random workload on every card.",
    )
    return resolve_copy_count(parser.parse_args())


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--devices must be comma-separated integer indices.") from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative indices.")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate indices.")
    return devices


def wait_workers(processes, ready_files=None, timeout=600):
    deadline = time.monotonic() + timeout
    while True:
        done = True
        for index, (rank, process, _, _) in enumerate(processes):
            code = process.poll()
            ready = ready_files[index].exists() if ready_files is not None else code == 0
            if code is not None and (code != 0 or not ready):
                raise RuntimeError(f"Worker rank={rank} exited with code={code}; inspect its log.")
            done = done and ready
        if done:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for SCATTER workers.")
        time.sleep(0.05)


def stop_workers(processes):
    # Only terminate processes created by this invocation, never external jobs.
    for _, process, _, _ in processes:
        if process.poll() is None:
            process.terminate()
    for _, process, _, _ in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    args = parse_args()
    validate_workload(args)
    devices = selected_devices(args)
    if args.numa_node_map is not None and args.numa_policy != "mapped":
        raise ValueError("--numa-node-map requires --numa-policy=mapped.")
    if args.numa_page_samples < 1:
        raise ValueError("--numa-page-samples must be positive.")
    numa_enabled = (args.numa_policy != "inherit" or args.cpu_bind != "none" or args.require_numa_placement)
    topo = topology() if numa_enabled else None
    if topo:
        nodes = sorted(topo["nodes"]) if args.numa_nodes == "auto" else parse_ids(args.numa_nodes)
        node_map = [int(n) for n in args.numa_node_map.split(",")] if args.numa_node_map else None
        plans = worker_plans(len(devices), args.numa_policy, nodes, topo, args.cpu_bind,
                             args.cpus_per_worker, args.numa_node, node_map)
    else:
        plans = [{"mode": "inherit", "nodes": [], "cpus": []} for _ in devices]
    print("A3_SCATTER_MULTIPROCESS_PLAN " + json.dumps({"devices": devices, "plans": plans}), flush=True)
    if args.dry_run:
        return
    worker_script = Path(__file__).with_name("test_scatter_copy.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_log_dir:
        args.worker_log_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="scatter_multiprocess_") as temp_name:
        temp_dir = Path(temp_name)
        start_file = temp_dir / "start"
        timing_start_file = temp_dir / "timing_start"
        timing_stop_file = temp_dir / "timing_stop"
        timing_ready_files: list[Path] = []
        timing_done_files: list[Path] = []
        processes: list[tuple[int, subprocess.Popen[bytes], Path, Path]] = []
        try:
            for rank, device in enumerate(devices):
                plan = plans[rank]
                result_file = temp_dir / f"result_{rank}.json"
                ready_file = temp_dir / f"ready_{rank}"
                timing_ready_file = temp_dir / f"timing_ready_{rank}"
                timing_ready_files.append(timing_ready_file)
                timing_done_file = temp_dir / f"timing_done_{rank}"
                timing_done_files.append(timing_done_file)
                seed = args.seed if args.same_seed else args.seed + rank
                command = [
                    sys.executable,
                    str(worker_script),
                    "--device", f"npu:{device}",
                    "--batch-size", str(args.batch_size),
                    "--source-len", str(args.source_len),
                    "--hbm-slots", str(args.hbm_slots),
                    "--copy-min", str(args.copy_min),
                    "--copy-max", str(args.copy_max),
                    "--copy-cap", str(args.copy_cap),
                    "--warmup", str(args.warmup),
                    "--iters", str(args.iters),
                    "--seed", str(seed),
                    "--json-output", str(result_file),
                    "--ready-file", str(ready_file),
                    "--start-file", str(start_file),
                    "--timing-ready-file", str(timing_ready_file),
                    "--timing-start-file", str(timing_start_file),
                    "--timing", args.timing,
                    "--memory-policy", plan["mode"],
                ]
                if plan["nodes"]:
                    command += ["--memory-nodes", ",".join(map(str, plan["nodes"]))]
                if plan["cpus"]:
                    command += ["--cpu-list", ",".join(map(str, plan["cpus"]))]
                if numa_enabled:
                    command += ["--numa-probe", "--numa-page-samples", str(args.numa_page_samples),
                                "--timing-done-file", str(timing_done_file),
                                "--timing-stop-file", str(timing_stop_file)]
                if args.require_numa_placement:
                    command += ["--require-numa-placement"]
                env = os.environ.copy()
                if args.cpu_bind == "spread":
                    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                env["PYTHONUNBUFFERED"] = "1"
                extension = str(Path(__file__).resolve().parent.parent / "torch_extension")
                env["PYTHONPATH"] = extension + os.pathsep + env.get("PYTHONPATH", "")
                # All workers keep the same visibility list. --devices are
                # indices in that list, not per-child npu:0 remappings.
                log = args.worker_log_dir / f"rank{rank}_device{device}.log" if args.worker_log_dir else None
                print(
                    f"A3_SCATTER_MULTIPROCESS_LAUNCH rank={rank} "
                    f"device=npu:{device} seed={seed} policy={plan['mode']} "
                    f"nodes={plan['nodes']} cpus={plan['cpus']} log={log}",
                    flush=True,
                )
                if log:
                    with log.open("wb") as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                else:
                    process = subprocess.Popen(command, env=env)
                processes.append((rank, process, result_file, ready_file))

            wait_workers(processes, [ready for _, _, _, ready in processes])
            start_file.write_text("start\n", encoding="utf-8")
            print("A3_SCATTER_MULTIPROCESS_WARMUP_START all_workers_ready=1", flush=True)
            wait_workers(processes, timing_ready_files)
            pending_signal = timing_start_file.with_suffix(".tmp")
            pending_signal.write_text(str(time.monotonic() + 1.0), encoding="utf-8")
            pending_signal.replace(timing_start_file)
            print("A3_SCATTER_MULTIPROCESS_TIMING_START all_workers_ready=1", flush=True)
            if numa_enabled:
                wait_workers(processes, timing_done_files)
                timing_stop_file.write_text("stop\n", encoding="utf-8")
            wait_workers(processes)
        finally:
            stop_workers(processes)

        per_device = []
        for rank, (device, (_, _, result_file, _)) in enumerate(zip(devices, processes)):
            result = json.loads(result_file.read_text(encoding="utf-8"))
            result["rank"] = rank
            result["logical_device_index"] = device
            per_device.append(result)

    copied_tokens = [item["workload"]["copied_tokens"] for item in per_device]
    host_starts = [item["performance"]["host_start_ns"] for item in per_device]
    host_ends = [item["performance"]["host_end_ns"] for item in per_device]
    host_span_ns = max(host_ends) - min(host_starts)
    host_common_ns = max(0, min(host_ends) - max(host_starts))
    placement = [(item.get("numa") or {}).get("placement_status", "not_requested") for item in per_device]
    output = {
        "schema_version": 2,
        "test": "kvcache_scatter_copy_multiprocess",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_count": len(devices),
        "devices": devices,
        "same_seed": args.same_seed,
        "numa_policy": args.numa_policy,
        "cpu_bind": args.cpu_bind,
        "worker_plans": plans,
        "topology": topo,
        "summary": {
            **summarize_devices(per_device),
            "copied_tokens_sum_per_iteration": sum(copied_tokens),
            "all_correct": True,
            "placement_statuses": placement,
            "all_placements_sample_verified": all(p == "sample_verified" for p in placement),
            "host_start_skew_us": (max(host_starts) - min(host_starts)) / 1000,
            "host_interval_overlap_fraction": host_common_ns / host_span_ns if host_span_ns else 0.0,
            "hold_load_until_all_timed": numa_enabled,
        },
        "per_device": per_device,
    }
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print_result({"cards": len(devices), "batch_size": args.batch_size,
                  "copy_count": copy_label(vars(args)), **summarize_devices(per_device)})
    print("A3_KVCACHE_SCATTER_COPY_MULTIPROCESS_UT_OK", flush=True)


if __name__ == "__main__":
    main()
