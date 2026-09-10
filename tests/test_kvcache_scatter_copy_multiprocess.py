#!/usr/bin/env python3
"""Run one independent packed-C8 DRAM -> HBM worker per selected NPU."""

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

from scatter_cli import add_device_args, resolve_copy_count, selected_devices, validate_workload
from scatter_results import copy_label, print_result, summarize_devices

from _common import ROW_BYTES, add_case_args, validate_args, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_args(parser)
    add_device_args(parser, "0")
    parser.add_argument("--output", type=Path, help="Case JSON path (default: branch results directory).")
    parser.add_argument("--same-seed", action="store_true")
    return resolve_copy_count(parser.parse_args())


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices or any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct non-negative integer indices.")
    return devices


def wait_until_ready(ready_files: list[Path], processes: list, phase: str) -> None:
    deadline = time.monotonic() + 600
    while not all(path.exists() for path in ready_files):
        failed = [
            (rank, process.returncode)
            for (rank, process, _), ready in zip(processes, ready_files)
            if process.poll() is not None and (process.returncode != 0 or not ready.exists())
        ]
        if failed:
            raise RuntimeError(f"Workers exited during {phase}: {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for workers during {phase}")
        time.sleep(0.01)


def stop_workers(processes: list) -> None:
    for _, process, _ in processes:
        if process.poll() is None:
            process.terminate()
    for _, process, _ in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_workers(args: argparse.Namespace, devices: list[int]) -> list[dict]:
    worker = Path(__file__).with_name("test_kvcache_scatter_copy.py")
    with tempfile.TemporaryDirectory(prefix="a5_scatter_c8_") as name:
        temp = Path(name)
        start_file, timing_start_file = temp / "start", temp / "timing_start"
        initial_ready, timing_ready, processes = [], [], []
        try:
            for rank, device in enumerate(devices):
                result_file = temp / f"result_{rank}.json"
                ready_file, timing_ready_file = temp / f"ready_{rank}", temp / f"timing_ready_{rank}"
                initial_ready.append(ready_file)
                timing_ready.append(timing_ready_file)
                seed = args.seed if args.same_seed else args.seed + rank
                command = [
                    sys.executable, str(worker), "--device", f"npu:{device}", "--dtype", args.dtype,
                    "--batch-size", str(args.batch_size), "--source-len", str(args.source_len),
                    "--hbm-slots", str(args.hbm_slots), "--copy-min", str(args.copy_min),
                    "--copy-max", str(args.copy_max), "--copy-cap", str(args.copy_cap),
                    "--warmup", str(args.warmup), "--iters", str(args.iters), "--seed", str(seed),
                    "--json-output", str(result_file), "--ready-file", str(ready_file),
                    "--start-file", str(start_file), "--timing-ready-file", str(timing_ready_file),
                    "--timing-start-file", str(timing_start_file),
                ]
                if args.allow_non_a5:
                    command.append("--allow-non-a5")
                print(f"A5_SCATTER_C8_MULTIPROCESS_LAUNCH rank={rank} device=npu:{device} seed={seed}", flush=True)
                processes.append((rank, subprocess.Popen(command), result_file))
            wait_until_ready(initial_ready, processes, "initialization")
            start_file.write_text("start\n", encoding="utf-8")
            print("A5_SCATTER_C8_MULTIPROCESS_WARMUP_START all_workers_ready=1", flush=True)
            wait_until_ready(timing_ready, processes, "warmup")
            timing_start_file.write_text("start\n", encoding="utf-8")
            print("A5_SCATTER_C8_MULTIPROCESS_TIMING_START all_workers_ready=1", flush=True)
            while True:
                codes = [(rank, process.poll()) for rank, process, _ in processes]
                failures = [(rank, code) for rank, code in codes if code not in (None, 0)]
                if failures:
                    raise RuntimeError(f"Workers failed: {failures}")
                if all(code is not None for _, code in codes):
                    break
                time.sleep(0.05)
            results = []
            for device, (rank, _, path) in zip(devices, processes):
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("status") != "passed" or result.get("test") != "a5_kvcache_scatter_copy_c8_singlecard":
                    raise RuntimeError(f"Worker {rank} did not produce a passing C8 result.")
                if not all(result["correctness"].values()):
                    raise RuntimeError(f"Worker {rank} failed a correctness check.")
                result.update(rank=rank, logical_device_index=device)
                results.append(result)
            return results
        finally:
            stop_workers(processes)


def summarize(args: argparse.Namespace, devices: list[int], per_device: list[dict]) -> dict:
    total_bytes = sum(int(item["workload"]["payload_bytes_per_iteration"]) for item in per_device)
    total_tokens = sum(int(item["workload"]["copied_tokens"]) for item in per_device)
    return {
        "schema_version": 2, "test": "a5_kvcache_scatter_copy_c8_multiprocess",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_count": len(devices), "devices": devices, "same_seed": args.same_seed,
        "config": {
            "dtype": args.dtype, "row_bytes": ROW_BYTES, "batch_size": args.batch_size,
            "source_len": args.source_len, "hbm_slots": args.hbm_slots,
            "copy_min": args.copy_min, "copy_max": args.copy_max, "copy_cap": args.copy_cap,
            "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
            "allow_non_a5": args.allow_non_a5,
        },
        "summary": {
            **summarize_devices(per_device),
            "total_payload_bytes_per_iteration": total_bytes,
            "copied_tokens_sum_per_iteration": total_tokens, "all_correct": True,
        },
        "per_device": per_device,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_workload(args)
    devices = selected_devices(args)
    # Invalidate a previous successful result if this rerun subsequently fails.
    write_json(args.output, {"status": "running", "devices": devices})
    try:
        output = summarize(args, devices, run_workers(args, devices))
    except BaseException as error:
        write_json(args.output, {"status": "failed", "devices": devices, "error": str(error)})
        raise
    write_json(args.output, output)
    print_result({"cards": len(devices), "batch_size": args.batch_size,
                  "copy_count": copy_label(vars(args)), **{key: output["summary"][key]
                      for key in ("avg_us_mean", "avg_us_max", "avg_bandwidth")}})
    print("A5_KVCACHE_SCATTER_COPY_C8_MULTIPROCESS_UT_OK", flush=True)


if __name__ == "__main__":
    main()
