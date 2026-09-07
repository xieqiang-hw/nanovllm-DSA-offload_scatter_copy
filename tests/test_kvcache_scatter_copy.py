#!/usr/bin/env python3
"""Ascend 950 packed-C8 scatter: real DRAM -> HBM correctness and bandwidth."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Select this checkout's custom OPP before initializing torch_npu.
import vllm_dsa_a5
import torch
import torch_npu  # type: ignore  # noqa: F401

from _common import BLOCK_SIZE, COPY_CAP, ROW_BYTES, add_case_args, validate_args, write_json
from _utils import require_a5, swapped_from_cpu

POISON = 65


@dataclass
class Case:
    device: torch.device
    device_name: str
    dram_cpu: torch.Tensor
    dram: torch.Tensor
    hbm: torch.Tensor
    dram_table_cpu: torch.Tensor
    hbm_table_cpu: torch.Tensor
    dram_table: torch.Tensor
    hbm_table: torch.Tensor
    src_ids_cpu: torch.Tensor
    dst_slots_cpu: torch.Tensor
    copy_counts_cpu: torch.Tensor
    src_ids: torch.Tensor
    dst_slots: torch.Tensor
    copy_counts: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_args(parser)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json-output", type=Path)
    for name in ("ready-file", "start-file", "timing-ready-file", "timing-start-file"):
        parser.add_argument(f"--{name}", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def make_case(args: argparse.Namespace) -> Case:
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    generator = torch.Generator().manual_seed(args.seed)
    source_blocks = (args.source_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    hbm_blocks_per_row = (args.hbm_slots + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Same source-pool policy as the BF16 branch: read-only physical blocks
    # are shared by requests, each with its own randomized block-table view.
    dram_table_cpu = torch.stack([
        torch.randperm(source_blocks, generator=generator).to(torch.int32)
        for _ in range(args.batch_size)
    ])
    mapped_blocks = args.batch_size * hbm_blocks_per_row
    # Destinations are private per request, plus one unmapped guard block.
    hbm_table_cpu = torch.randperm(mapped_blocks + 1, generator=generator)[:mapped_blocks].to(torch.int32).reshape(args.batch_size, hbm_blocks_per_row)
    dram_cpu = torch.randint(-128, 128, (source_blocks, BLOCK_SIZE, 1, ROW_BYTES), dtype=torch.int8, generator=generator)
    src_ids_cpu = torch.full((args.batch_size, 1, COPY_CAP), -1, dtype=torch.int32)
    dst_slots_cpu = torch.full_like(src_ids_cpu, -1)
    copy_counts_cpu = torch.randint(args.copy_min, args.copy_max + 1, (args.batch_size,), dtype=torch.int32, generator=generator)
    if args.batch_size > 1:
        copy_counts_cpu[0] = args.copy_min
        copy_counts_cpu[1] = args.copy_max
    for row, count in enumerate(copy_counts_cpu.tolist()):
        src_ids_cpu[row, 0, :count] = torch.randperm(args.source_len, generator=generator)[:count].to(torch.int32)
        dst_slots_cpu[row, 0, :count] = torch.randperm(args.hbm_slots, generator=generator)[:count].to(torch.int32)

    return Case(
        device=device, device_name=device_name, dram_cpu=dram_cpu,
        dram=swapped_from_cpu(dram_cpu, device),
        hbm=torch.empty((mapped_blocks + 1, BLOCK_SIZE, 1, ROW_BYTES), dtype=torch.int8, device=device),
        dram_table_cpu=dram_table_cpu, hbm_table_cpu=hbm_table_cpu,
        dram_table=dram_table_cpu.to(device), hbm_table=hbm_table_cpu.to(device),
        src_ids_cpu=src_ids_cpu, dst_slots_cpu=dst_slots_cpu,
        copy_counts_cpu=copy_counts_cpu, src_ids=src_ids_cpu.to(device),
        dst_slots=dst_slots_cpu.to(device), copy_counts=copy_counts_cpu.to(device),
    )


def active_physical_rows(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    sources, destinations = [], []
    for row, count in enumerate(case.copy_counts_cpu.tolist()):
        source = case.src_ids_cpu[row, 0, :count].long()
        destination = case.dst_slots_cpu[row, 0, :count].long()
        sources.append(case.dram_table_cpu[row].long()[source // BLOCK_SIZE] * BLOCK_SIZE + source % BLOCK_SIZE)
        destinations.append(case.hbm_table_cpu[row].long()[destination // BLOCK_SIZE] * BLOCK_SIZE + destination % BLOCK_SIZE)
    return torch.cat(sources), torch.cat(destinations)


def call_scatter(case: Case, copy_counts: torch.Tensor | None = None) -> None:
    return vllm_dsa_a5.kvcache_scatter_copy_c8(
        case.hbm, case.dram, case.hbm_table, case.dram_table,
        case.src_ids, case.dst_slots, case.copy_counts if copy_counts is None else copy_counts,
    )


def assert_copied(case: Case) -> None:
    source, destination = active_physical_rows(case)
    expected = torch.full((case.hbm.shape[0] * BLOCK_SIZE, ROW_BYTES), POISON, dtype=torch.int8)
    expected[destination] = case.dram_cpu.reshape(-1, ROW_BYTES)[source]
    # Check ALL bytes, including the scale/RoPE payload, inactive rows and
    # the unmapped guard block. No floating-point tolerance is involved.
    if not torch.equal(case.hbm.cpu().reshape(-1, ROW_BYTES), expected):
        raise AssertionError("DRAM->HBM byte mismatch or inactive HBM bytes were modified.")


def check_copy(case: Case, phase: str) -> None:
    case.hbm.fill_(POISON)
    torch.npu.synchronize()
    if not bool((case.hbm == POISON).all().item()):
        raise AssertionError("HBM destinations were not poisoned before the copy.")
    pointers = tuple(t.data_ptr() for t in (case.hbm, case.dram, case.hbm_table, case.dram_table, case.src_ids, case.dst_slots, case.copy_counts))
    result = call_scatter(case)
    torch.npu.synchronize()
    if result is not None:
        raise AssertionError("C8 scatter must return None (caller-owned HBM).")
    if pointers != tuple(t.data_ptr() for t in (case.hbm, case.dram, case.hbm_table, case.dram_table, case.src_ids, case.dst_slots, case.copy_counts)):
        raise AssertionError("An input or caller-owned cache address changed.")
    assert_copied(case)
    for actual, expected in (
        (case.src_ids, case.src_ids_cpu), (case.dst_slots, case.dst_slots_cpu),
        (case.copy_counts, case.copy_counts_cpu), (case.hbm_table, case.hbm_table_cpu),
        (case.dram_table, case.dram_table_cpu),
    ):
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError("Read-only copy metadata was modified.")
    print(
        f"A5_SCATTER_C8_DRAM_TO_HBM_CHECK phase={phase} "
        f"copied_tokens={int(case.copy_counts_cpu.sum())} row_bytes={ROW_BYTES} "
        "allocator=empty_with_swapped_memory byte_exact=1 all_inactive_rows_unchanged=1 caller_owned=1 ok=1",
        flush=True,
    )


def wait_for_start(ready_file: Path | None, start_file: Path | None) -> None:
    if ready_file is None and start_file is None:
        return
    if ready_file is None or start_file is None:
        raise ValueError("ready and start files must be specified together")
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 600
    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for multiprocess start")
        time.sleep(0.001)


def run(args: argparse.Namespace) -> dict:
    # Exercise first-fill-sized lists through the SAME kernel/ABI, outside
    # timing. Maximum capacity and exact-full HBM slots still have a guard.
    fill_args = argparse.Namespace(**vars(args))
    fill_args.batch_size = 1
    fill_args.source_len = fill_args.hbm_slots = COPY_CAP
    fill_args.copy_min = fill_args.copy_max = COPY_CAP
    fill_case = make_case(fill_args)
    check_copy(fill_case, "first_fill_capacity")
    del fill_case
    torch.npu.empty_cache()

    case = make_case(args)
    copied_tokens = int(case.copy_counts_cpu.sum())
    payload_bytes = copied_tokens * ROW_BYTES
    print(
        f"A5_SCATTER_C8_CONFIG device={case.device} device_name={case.device_name!r} "
        f"dtype=c8 row_bytes={ROW_BYTES} batch={args.batch_size} source_len={args.source_len} "
        f"hbm_slots={args.hbm_slots} copy_cap={COPY_CAP} copy_range=[{args.copy_min},{args.copy_max}] "
        f"copy_counts={case.copy_counts_cpu.tolist()} opapi={vllm_dsa_a5.local_opapi_path()} "
        f"launch_blocking={os.getenv('ASCEND_LAUNCH_BLOCKING', 'unset')}",
        flush=True,
    )
    check_copy(case, "requested_workload")
    case.hbm.fill_(POISON)
    if call_scatter(case, torch.zeros_like(case.copy_counts)) is not None:
        raise AssertionError("Zero-copy scatter must return None.")
    torch.npu.synchronize()
    if not bool((case.hbm == POISON).all().item()):
        raise AssertionError("copy_count=0 modified the HBM cache.")
    print("A5_SCATTER_C8_ZERO_COUNT_CHECK all_hbm_unchanged=1 ok=1", flush=True)

    wait_for_start(args.ready_file, args.start_file)
    for _ in range(args.warmup):
        call_scatter(case)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    wait_for_start(args.timing_ready_file, args.timing_start_file)
    # Same timer as the BF16 branch: one event pair around repeated launches.
    # Immutable copy metadata keeps the count unchanged across iterations.
    start.record()
    for _ in range(args.iters):
        call_scatter(case)
    end.record()
    end.synchronize()
    avg_us = start.elapsed_time(end) * 1000 / args.iters
    assert_copied(case)
    payload_gbps = payload_bytes / (avg_us * 1000) if payload_bytes and avg_us else 0.0
    print(
        f"A5_KVCACHE_SCATTER_COPY_C8_RESULT dtype=c8 row_bytes={ROW_BYTES} "
        f"copy_min={args.copy_min} copy_max={args.copy_max} copied_tokens={copied_tokens} "
        f"avg_us={avg_us:.3f} payload_gbps={payload_gbps:.3f} "
        f"timer=npu_event warmup={args.warmup} iters={args.iters} performance_assertion=0",
        flush=True,
    )
    result = {
        "schema_version": 1, "test": "a5_kvcache_scatter_copy_c8_singlecard",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device_count": 1,
        "config": {
            "device": str(case.device), "device_name": case.device_name,
            "dtype": "c8", "row_bytes": ROW_BYTES, "batch_size": args.batch_size,
            "source_len": args.source_len, "hbm_slots": args.hbm_slots,
            "copy_min": args.copy_min, "copy_max": args.copy_max, "copy_cap": COPY_CAP,
            "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
            "opapi": vllm_dsa_a5.local_opapi_path(),
        },
        "workload": {
            "copy_counts": case.copy_counts_cpu.tolist(), "copied_tokens": copied_tokens,
            "payload_bytes_per_iteration": payload_bytes,
        },
        "correctness": {
            "data_exact": True, "caller_owned": True, "guard_unchanged": True,
            "zero_count_unchanged": True, "first_fill_capacity": True,
        },
        "performance": {"timer": "npu_event", "avg_us": avg_us, "payload_gbps": payload_gbps},
    }
    if args.json_output is not None:
        write_json(args.json_output, result)
    print("A5_KVCACHE_SCATTER_COPY_C8_UT_OK", flush=True)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
