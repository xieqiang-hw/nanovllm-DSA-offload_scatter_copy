#!/usr/bin/env python3
"""Run one repository Scatter Copy worker process per selected NPU."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_args(parser, "0")
    parser.add_argument("--output", type=Path, help="Case JSON path (default: branch results directory).")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
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
    parser.add_argument("--same-seed", action="store_true")
    parser.add_argument("--allow-non-a5", action="store_true")
    return resolve_copy_count(parser.parse_args())


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--devices must be comma-separated integer indices") from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative indices")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate indices")
    return devices


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def wait_until_ready(
    ready_files: list[Path],
    processes: list[tuple[int, subprocess.Popen[bytes], Path]],
    phase: str,
) -> None:
    deadline = time.monotonic() + 600
    while not all(path.exists() for path in ready_files):
        failed = [
            (rank, process.returncode)
            for rank, process, _ in processes
            if process.poll() not in (None, 0)
        ]
        if failed:
            raise RuntimeError(f"Workers failed during {phase}: {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for workers during {phase}")
        time.sleep(0.05)


def main() -> None:
    args = parse_args()
    validate_workload(args)
    devices = selected_devices(args)
    worker_script = Path(__file__).with_name("test_kvcache_scatter_copy.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="a5_scatter_multiprocess_") as name:
        temp_dir = Path(name)
        start_file = temp_dir / "start"
        timing_start_file = temp_dir / "timing_start"
        initial_ready: list[Path] = []
        timing_ready: list[Path] = []
        processes: list[tuple[int, subprocess.Popen[bytes], Path]] = []
        for rank, device in enumerate(devices):
            result_file = temp_dir / f"result_{rank}.json"
            ready_file = temp_dir / f"ready_{rank}"
            timing_ready_file = temp_dir / f"timing_ready_{rank}"
            initial_ready.append(ready_file)
            timing_ready.append(timing_ready_file)
            seed = args.seed if args.same_seed else args.seed + rank
            command = [
                sys.executable, str(worker_script),
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
            if args.allow_non_a5:
                command.append("--allow-non-a5")
            print(
                "A5_SCATTER_MULTIPROCESS_LAUNCH "
                f"rank={rank} device=npu:{device} seed={seed}",
                flush=True,
            )
            processes.append((rank, subprocess.Popen(command), result_file))

        try:
            wait_until_ready(initial_ready, processes, "initialization")
            start_file.write_text("start\n", encoding="utf-8")
            print("A5_SCATTER_MULTIPROCESS_WARMUP_START all_workers_ready=1", flush=True)
            wait_until_ready(timing_ready, processes, "warmup")
            timing_start_file.write_text("start\n", encoding="utf-8")
            print("A5_SCATTER_MULTIPROCESS_TIMING_START all_workers_ready=1", flush=True)
            failures = []
            for rank, process, _ in processes:
                returncode = process.wait()
                if returncode:
                    failures.append({"rank": rank, "returncode": returncode})
            if failures:
                raise RuntimeError(f"One or more workers failed: {failures}")
        except BaseException:
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            raise

        per_device = []
        for rank, (device, (_, _, result_file)) in enumerate(zip(devices, processes)):
            result = json.loads(result_file.read_text(encoding="utf-8"))
            result["rank"] = rank
            result["logical_device_index"] = device
            per_device.append(result)

    payload_bytes = [int(item["workload"]["payload_bytes_per_iteration"]) for item in per_device]
    copied_tokens = [int(item["workload"]["copied_tokens"]) for item in per_device]
    total_payload_bytes = sum(payload_bytes)
    output = {
        "schema_version": 2,
        "test": "a5_kvcache_scatter_copy_multiprocess",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_count": len(devices),
        "devices": devices,
        "same_seed": args.same_seed,
        "config": {
            "dtype": args.dtype, "batch_size": args.batch_size,
            "source_len": args.source_len, "hbm_slots": args.hbm_slots,
            "copy_min": args.copy_min, "copy_max": args.copy_max,
            "copy_cap": args.copy_cap, "warmup": args.warmup,
            "iters": args.iters, "seed": args.seed,
            "allow_non_a5": args.allow_non_a5,
        },
        "summary": {
            **summarize_devices(per_device),
            "total_payload_bytes_per_iteration": total_payload_bytes,
            "copied_tokens_sum_per_iteration": sum(copied_tokens),
            "all_correct": True,
        },
        "per_device": per_device,
    }
    write_json(args.output, output)
    print_result({"cards": len(devices), "batch_size": args.batch_size,
                  "copy_count": copy_label(vars(args)), **summarize_devices(per_device)})
    print("A5_KVCACHE_SCATTER_COPY_MULTIPROCESS_UT_OK", flush=True)


if __name__ == "__main__":
    main()
