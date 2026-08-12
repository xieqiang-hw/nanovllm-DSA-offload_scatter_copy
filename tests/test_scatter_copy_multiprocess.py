#!/usr/bin/env python3
"""Run one A3 SCATTER worker process per visible NPU."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="0,1",
        help="Comma-separated logical NPU indices, for example 0,1,2,3.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scatter_multiprocess.json")
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20000)
    parser.add_argument("--hbm-slots", type=int, default=6144)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="Use exactly the same random workload on every card.",
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


def main() -> None:
    args = parse_args()
    devices = parse_devices(args.devices)
    worker_script = Path(__file__).with_name("test_scatter_copy.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="scatter_multiprocess_") as temp_name:
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
            print(
                f"A3_SCATTER_MULTIPROCESS_LAUNCH rank={rank} "
                f"device=npu:{device} seed={seed}",
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
        print("A3_SCATTER_MULTIPROCESS_WARMUP_START all_workers_ready=1", flush=True)

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
        print("A3_SCATTER_MULTIPROCESS_TIMING_START all_workers_ready=1", flush=True)

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

    latencies = [item["performance"]["avg_us"] for item in per_device]
    bandwidths = [item["performance"]["payload_gbps"] for item in per_device]
    copied_tokens = [item["workload"]["copied_tokens"] for item in per_device]
    payload_bytes = [
        item["workload"]["payload_bytes_per_iteration"] for item in per_device
    ]
    avg_us_max = max(latencies)
    payload_gbps_sum_by_avg_us_max = (
        sum(payload_bytes) / (avg_us_max * 1000)
        if sum(payload_bytes) and avg_us_max
        else 0.0
    )
    output = {
        "schema_version": 1,
        "test": "kvcache_scatter_copy_multiprocess",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_count": len(devices),
        "devices": devices,
        "same_seed": args.same_seed,
        "summary": {
            "avg_us_min": min(latencies),
            "avg_us_mean": statistics.fmean(latencies),
            "avg_us_max": avg_us_max,
            "payload_gbps_sum": sum(bandwidths),
            "payload_gbps_sum_by_avg_us_max": payload_gbps_sum_by_avg_us_max,
            "copied_tokens_sum_per_iteration": sum(copied_tokens),
            "all_correct": True,
        },
        "per_device": per_device,
    }
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        "A3_SCATTER_MULTIPROCESS_RESULT "
        f"cards={len(devices)} avg_us_mean={statistics.fmean(latencies):.3f} "
        f"payload_gbps_sum={sum(bandwidths):.3f} "
        f"payload_gbps_sum_by_avg_us_max={payload_gbps_sum_by_avg_us_max:.3f} "
        f"output={args.output}",
        flush=True,
    )
    print("A3_KVCACHE_SCATTER_COPY_MULTIPROCESS_UT_OK", flush=True)


if __name__ == "__main__":
    main()
