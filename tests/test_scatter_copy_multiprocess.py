#!/usr/bin/env python3
"""Run one A5 Scatter Copy worker process per selected NPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument(
        "--output", type=Path, default=Path("a5_scatter_multiprocess.json")
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20000)
    parser.add_argument("--hbm-slots", type=int, default=6144)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="Use exactly the same random workload on every card.",
    )
    parser.add_argument("--allow-non-a5", action="store_true")
    parser.add_argument(
        "--numa-map",
        help=(
            "Optional device-to-NUMA mapping, for example "
            "0:3,1:3,2:1,3:1. Each worker is launched with matching "
            "numactl --cpunodebind and --membind options."
        ),
    )
    return parser.parse_args()


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


def parse_numa_map(value: str | None, devices: list[int]) -> dict[int, int]:
    if value is None:
        return {}
    mapping: dict[int, int] = {}
    try:
        for item in value.split(","):
            device_text, node_text = item.strip().split(":", maxsplit=1)
            device = int(device_text)
            node = int(node_text)
            if device < 0 or node < 0 or device in mapping:
                raise ValueError
            mapping[device] = node
    except ValueError as error:
        raise ValueError(
            "--numa-map must contain unique non-negative DEVICE:NODE pairs."
        ) from error
    if set(mapping) != set(devices):
        raise ValueError(
            "--numa-map must specify exactly one NUMA node for every device."
        )
    if shutil.which("numactl") is None:
        raise RuntimeError("--numa-map requires numactl in PATH.")
    return mapping


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    devices = parse_devices(args.devices)
    numa_map = parse_numa_map(args.numa_map, devices)
    worker_script = Path(__file__).with_name("test_dram_to_hbm.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="a5_scatter_multiprocess_") as temp_name:
        temp_dir = Path(temp_name)
        start_file = temp_dir / "start"
        timing_start_file = temp_dir / "timing_start"
        timing_ready_files: list[Path] = []
        processes: list[tuple[int, subprocess.Popen[bytes], Path, Path]] = []
        for rank, device in enumerate(devices):
            result_file = temp_dir / f"result_{rank}.json"
            ready_file = temp_dir / f"ready_{rank}"
            timing_ready_file = temp_dir / f"timing_ready_{rank}"
            timing_ready_files.append(timing_ready_file)
            seed = args.seed if args.same_seed else args.seed + rank
            command = [
                sys.executable,
                str(worker_script),
                "--device", f"npu:{device}",
                "--dtype", args.dtype,
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
            ]
            if numa_map:
                numa_node = numa_map[device]
                command = [
                    "numactl",
                    f"--cpunodebind={numa_node}",
                    f"--membind={numa_node}",
                    *command,
                ]
            if args.allow_non_a5:
                command.append("--allow-non-a5")
            print(
                "A5_SCATTER_MULTIPROCESS_LAUNCH "
                f"rank={rank} device=npu:{device} seed={seed} "
                f"numa_node={numa_map.get(device, 'unbound')}",
                flush=True,
            )
            processes.append((rank, subprocess.Popen(command), result_file, ready_file))

        deadline = time.monotonic() + 600
        while not all(ready.exists() for _, _, _, ready in processes):
            failed = [
                (rank, process.returncode)
                for rank, process, _, _ in processes
                if process.poll() not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"Workers failed before the timing barrier: {failed}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for all NPU workers to become ready.")
            time.sleep(0.05)

        start_file.write_text("start\n", encoding="utf-8")
        print("A5_SCATTER_MULTIPROCESS_WARMUP_START all_workers_ready=1", flush=True)

        deadline = time.monotonic() + 600
        while not all(ready.exists() for ready in timing_ready_files):
            failed = [
                (rank, process.returncode)
                for rank, process, _, _ in processes
                if process.poll() not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"Workers failed during warmup: {failed}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for all NPU workers to finish warmup.")
            time.sleep(0.05)

        timing_start_file.write_text("start\n", encoding="utf-8")
        print("A5_SCATTER_MULTIPROCESS_TIMING_START all_workers_ready=1", flush=True)

        failures = []
        for rank, process, _, _ in processes:
            returncode = process.wait()
            if returncode != 0:
                failures.append({"rank": rank, "returncode": returncode})
        if failures:
            raise RuntimeError(f"One or more workers failed: {failures}")

        per_device = []
        for rank, (device, (_, _, result_file, _)) in enumerate(zip(devices, processes)):
            result = json.loads(result_file.read_text(encoding="utf-8"))
            result["rank"] = rank
            result["logical_device_index"] = device
            per_device.append(result)

    latencies = [float(item["performance"]["avg_us"]) for item in per_device]
    bandwidths = [float(item["performance"]["payload_gbps"]) for item in per_device]
    payload_bytes = [
        int(item["workload"]["payload_bytes_per_iteration"])
        for item in per_device
    ]
    copied_tokens = [int(item["workload"]["copied_tokens"]) for item in per_device]
    avg_us_max = max(latencies)
    total_payload_bytes = sum(payload_bytes)
    payload_gbps_sum_by_avg_us_max = (
        total_payload_bytes / (avg_us_max * 1000)
        if total_payload_bytes and avg_us_max
        else 0.0
    )
    output = {
        "schema_version": 1,
        "test": "a5_kvcache_scatter_copy_multiprocess",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_count": len(devices),
        "devices": devices,
        "same_seed": args.same_seed,
        "numa_map": {str(device): node for device, node in numa_map.items()},
        "config": {
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "source_len": args.source_len,
            "hbm_slots": args.hbm_slots,
            "copy_min": args.copy_min,
            "copy_max": args.copy_max,
            "copy_cap": args.copy_cap,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "allow_non_a5": args.allow_non_a5,
        },
        "summary": {
            "avg_us_min": min(latencies),
            "avg_us_mean": statistics.fmean(latencies),
            "avg_us_max": avg_us_max,
            "payload_gbps_sum": sum(bandwidths),
            "payload_gbps_sum_by_avg_us_max": payload_gbps_sum_by_avg_us_max,
            "total_payload_bytes_per_iteration": total_payload_bytes,
            "copied_tokens_sum_per_iteration": sum(copied_tokens),
            "all_correct": True,
        },
        "per_device": per_device,
    }
    write_json(args.output, output)
    print(
        "A5_SCATTER_MULTIPROCESS_RESULT "
        f"cards={len(devices)} avg_us_mean={statistics.fmean(latencies):.3f} "
        f"avg_us_max={avg_us_max:.3f} "
        f"payload_gbps_sum={sum(bandwidths):.3f} "
        f"payload_gbps_sum_by_avg_us_max={payload_gbps_sum_by_avg_us_max:.3f} "
        f"output={args.output}",
        flush=True,
    )
    print("A5_KVCACHE_SCATTER_COPY_MULTIPROCESS_UT_OK", flush=True)


if __name__ == "__main__":
    main()
