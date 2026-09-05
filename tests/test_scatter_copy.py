#!/usr/bin/env python3
"""A3 SCATTER correctness and NPU-event latency test."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Torch/CANN imports are deliberately deferred to main(): NUMA/CPU policies must
# be applied before either library creates background threads or host buffers.
from numa_support import (
    NumaAPI, apply_worker_policy, capture_registration_logs, inspect_buffer, parse_ids,
    placement_result, prepare_registration_logging, process_status,
)


BLOCK_SIZE = 128
KPE_DIM = 64
CKV_DIM = 512
MAX_COPY_CAP = 65536
CKV_POISON = 37.0
KPE_POISON = -29.0


@dataclass
class Case:
    device: torch.device
    device_name: str
    dram_kpe_cpu: torch.Tensor
    dram_ckv_cpu: torch.Tensor
    dram_kpe: torch.Tensor
    dram_ckv: torch.Tensor
    hbm_kpe: torch.Tensor
    hbm_ckv: torch.Tensor
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
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20000)
    parser.add_argument("--hbm-slots", type=int, default=6144)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing", choices=("eager", "graph"), default="eager")
    parser.add_argument("--memory-policy", choices=("inherit", "default", "bind", "interleave"), default="inherit")
    parser.add_argument("--memory-nodes", default="")
    parser.add_argument("--cpu-list", default="")
    parser.add_argument("--numa-probe", action="store_true")
    parser.add_argument("--numa-page-samples", type=int, default=1024)
    parser.add_argument("--require-numa-placement", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write the configuration, checks, and performance result to JSON.",
    )
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--start-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--timing-ready-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--timing-start-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--timing-done-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--timing-stop-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.source_len <= 0 or args.hbm_slots <= 0:
        raise ValueError("--source-len and --hbm-slots must be positive.")
    if not 0 <= args.copy_min <= args.copy_max:
        raise ValueError(
            "--copy-min/--copy-max must satisfy 0 <= min <= max."
        )
    if not 0 < args.copy_cap <= MAX_COPY_CAP:
        raise ValueError(
            f"--copy-cap must be in [1,{MAX_COPY_CAP}]."
        )
    if args.copy_max > args.copy_cap:
        raise ValueError("--copy-max must not exceed --copy-cap.")
    if args.copy_max > args.source_len:
        raise ValueError("--copy-max must not exceed --source-len.")
    if args.copy_max >= args.hbm_slots:
        raise ValueError(
            "--hbm-slots must exceed --copy-max so an untouched "
            "guard token exists."
        )
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")
    if args.numa_page_samples < 1:
        raise ValueError("--numa-page-samples must be positive.")
    if (args.memory_policy in ("bind", "interleave")) != bool(args.memory_nodes):
        raise ValueError("bind/interleave require --memory-nodes; inherit/default require no nodes.")


def random_block_table(
    batch_size: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_row
    table = torch.randperm(
        total_blocks,
        generator=generator,
        dtype=torch.int64,
    ).reshape(batch_size, blocks_per_row)
    return table.to(torch.int32).contiguous(), total_blocks


def swapped_from_cpu(
    cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError(
            "torch_npu.empty_with_swapped_memory is unavailable. "
            "This test refuses to replace DRAM with an HBM tensor."
        )
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape,
        dtype=cpu.dtype,
        device=device,
    )
    tensor.fill_(0)
    staging = cpu.to(device)
    tensor.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return tensor


def make_case(args: argparse.Namespace) -> Case:
    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_index = (
        device.index
        if device.index is not None
        else torch.npu.current_device()
    )
    get_device_name = getattr(torch.npu, "get_device_name", None)
    if get_device_name is None:
        get_device_name = torch_npu.npu.get_device_name
    device_name = get_device_name(device_index)
    generator = torch.Generator().manual_seed(args.seed)

    source_blocks_per_row = (
        args.source_len + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    hbm_blocks_per_row = (
        args.hbm_slots + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    dram_table_cpu, dram_blocks = random_block_table(
        args.batch_size,
        source_blocks_per_row,
        generator,
    )
    hbm_table_cpu, hbm_blocks = random_block_table(
        args.batch_size,
        hbm_blocks_per_row,
        generator,
    )

    dram_kpe_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        KPE_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    dram_ckv_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        CKV_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    src_ids_cpu = torch.full(
        (args.batch_size, args.copy_cap),
        -1,
        dtype=torch.int32,
    )
    dst_slots_cpu = torch.full_like(src_ids_cpu, -1)
    sampled_counts = torch.randint(
        args.copy_min,
        args.copy_max + 1,
        (args.batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    counts = sampled_counts.tolist()
    for row, count in enumerate(counts):
        src_ids_cpu[row, :count] = torch.randperm(
            args.source_len,
            generator=generator,
        )[:count].to(torch.int32)
        dst_slots_cpu[row, :count] = torch.randperm(
            args.hbm_slots,
            generator=generator,
        )[:count].to(torch.int32)
    copy_counts_cpu = torch.tensor(counts, dtype=torch.int32)

    return Case(
        device=device,
        device_name=device_name,
        dram_kpe_cpu=dram_kpe_cpu,
        dram_ckv_cpu=dram_ckv_cpu,
        dram_kpe=swapped_from_cpu(dram_kpe_cpu, device),
        dram_ckv=swapped_from_cpu(dram_ckv_cpu, device),
        hbm_kpe=torch.empty(
            hbm_blocks,
            BLOCK_SIZE,
            KPE_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        hbm_ckv=torch.empty(
            hbm_blocks,
            BLOCK_SIZE,
            CKV_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        dram_table_cpu=dram_table_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table=dram_table_cpu.to(device),
        hbm_table=hbm_table_cpu.to(device),
        src_ids_cpu=src_ids_cpu,
        dst_slots_cpu=dst_slots_cpu,
        copy_counts_cpu=copy_counts_cpu,
        src_ids=src_ids_cpu.to(device),
        dst_slots=dst_slots_cpu.to(device),
        copy_counts=copy_counts_cpu.to(device),
    )


def active_physical_rows(
    case: Case,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_rows: list[torch.Tensor] = []
    destination_rows: list[torch.Tensor] = []
    for row, count in enumerate(case.copy_counts_cpu.tolist()):
        source = case.src_ids_cpu[row, :count].to(torch.int64)
        destination = case.dst_slots_cpu[row, :count].to(torch.int64)
        source_rows.append(
            case.dram_table_cpu[row].to(torch.int64)[
                source // BLOCK_SIZE
            ]
            * BLOCK_SIZE
            + source % BLOCK_SIZE
        )
        destination_rows.append(
            case.hbm_table_cpu[row].to(torch.int64)[
                destination // BLOCK_SIZE
            ]
            * BLOCK_SIZE
            + destination % BLOCK_SIZE
        )
    return torch.cat(source_rows), torch.cat(destination_rows)


def poison_hbm(case: Case) -> None:
    case.hbm_ckv.fill_(CKV_POISON)
    case.hbm_kpe.fill_(KPE_POISON)


def call_scatter(
    case: Case,
    copy_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops_overlap.kvcache_scatter_copy(
        case.hbm_kpe,
        case.hbm_ckv,
        case.dram_kpe,
        case.dram_ckv,
        case.hbm_table,
        case.dram_table,
        case.src_ids,
        case.dst_slots,
        case.copy_counts if copy_counts is None else copy_counts,
    )


def assert_all_hbm_poisoned(case: Case) -> None:
    ckv_ok = bool(torch.all(case.hbm_ckv == CKV_POISON).item())
    kpe_ok = bool(torch.all(case.hbm_kpe == KPE_POISON).item())
    if not ckv_ok or not kpe_ok:
        raise AssertionError("copy_count=0 modified the HBM cache.")


def assert_copied(
    case: Case,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
) -> None:
    expected_ckv = case.dram_ckv_cpu.view(-1, CKV_DIM)[source_rows]
    expected_kpe = case.dram_kpe_cpu.view(-1, KPE_DIM)[source_rows]
    destination_rows_npu = destination_rows.to(case.device)
    actual_ckv = (
        case.hbm_ckv.view(-1, CKV_DIM)[destination_rows_npu].cpu()
    )
    actual_kpe = (
        case.hbm_kpe.view(-1, KPE_DIM)[destination_rows_npu].cpu()
    )
    if not torch.equal(actual_ckv, expected_ckv):
        max_abs = (
            actual_ckv.float() - expected_ckv.float()
        ).abs().max()
        raise AssertionError(
            f"DRAM->HBM CKV mismatch, max_abs={float(max_abs):.9f}"
        )
    if not torch.equal(actual_kpe, expected_kpe):
        max_abs = (
            actual_kpe.float() - expected_kpe.float()
        ).abs().max()
        raise AssertionError(
            f"DRAM->HBM KPE mismatch, max_abs={float(max_abs):.9f}"
        )

    active = set(destination_rows.tolist())
    total_rows = case.hbm_ckv.shape[0] * BLOCK_SIZE
    guard = next(index for index in range(total_rows) if index not in active)
    guard_ckv = case.hbm_ckv.view(-1, CKV_DIM)[guard].cpu()
    guard_kpe = case.hbm_kpe.view(-1, KPE_DIM)[guard].cpu()
    if not torch.equal(
        guard_ckv,
        torch.full_like(guard_ckv, CKV_POISON),
    ) or not torch.equal(
        guard_kpe,
        torch.full_like(guard_kpe, KPE_POISON),
    ):
        raise AssertionError("An inactive HBM guard row was modified.")


def wait_for_multiprocess_start(args: argparse.Namespace) -> None:
    if args.ready_file is None and args.start_file is None:
        return
    if args.ready_file is None or args.start_file is None:
        raise ValueError("--ready-file and --start-file must be used together.")
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 600
    while not args.start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the multi-card start signal.")
        time.sleep(0.01)


def wait_for_multiprocess_timing_start(args: argparse.Namespace) -> None:
    if args.timing_ready_file is None and args.timing_start_file is None:
        return
    if args.timing_ready_file is None or args.timing_start_file is None:
        raise ValueError(
            "--timing-ready-file and --timing-start-file must be used together."
        )
    args.timing_ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.timing_ready_file.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 600
    while not args.timing_start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the multi-card timing signal.")
        time.sleep(0.01)
    signal = args.timing_start_file.read_text(encoding="utf-8").strip()
    if signal != "start":
        # A common future CLOCK_MONOTONIC deadline removes the 10 ms polling
        # skew. Every worker has already synchronized its NPU before this gate.
        target = float(signal)
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            if remaining > 0.002:
                time.sleep(remaining - 0.001)


def write_json_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def keep_copy_load_until_all_timed(args, launch):
    if args.timing_done_file is None and args.timing_stop_file is None:
        return
    if args.timing_done_file is None or args.timing_stop_file is None:
        raise ValueError("--timing-done-file and --timing-stop-file must be used together.")
    args.timing_done_file.write_text("done\n", encoding="utf-8")
    deadline = time.monotonic() + 600
    # Fast workers must not go idle while a slower worker is still measuring:
    # otherwise its last iterations see artificially reduced link contention.
    # These idempotent copies are OUTSIDE this worker's recorded event interval.
    while not args.timing_stop_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for peers to finish measuring.")
        for _ in range(32):
            launch()
        torch.npu.synchronize()


def run(args: argparse.Namespace) -> dict[str, object]:
    host_mappings = {}
    if args.numa_probe or args.require_numa_placement:
        with capture_registration_logs() as host_mappings:
            case = make_case(args)
        print(f"A3_SCATTER_NUMA_ADDRESS_CAPTURE pid={os.getpid()} "
              f"registration_mappings={len(host_mappings)} "
              "source=current_allocation_logs allocator_unchanged=1", flush=True)
    else:
        case = make_case(args)
    numa = getattr(args, "numa_setup", None)
    if numa is not None:
        numa["policy_after_allocation"] = NumaAPI().get_policy()
        numa["allowed_after_allocation"] = process_status()
        if numa["policy_after_allocation"] != numa["policy_after"]:
            raise RuntimeError("Runtime changed the allocation thread's NUMA policy.")
        if args.cpu_list and set(os.sched_getaffinity(0)) != set(parse_ids(args.cpu_list)):
            raise RuntimeError("Runtime changed CPU affinity during NPU initialization/allocation.")
    source_rows, destination_rows = active_physical_rows(case)
    copied_tokens = int(case.copy_counts_cpu.sum())
    payload_bytes = copied_tokens * (CKV_DIM + KPE_DIM) * 2

    print(
        "A3_SCATTER_CONFIG "
        f"device={case.device} device_name={case.device_name!r} "
        f"dtype=bf16 batch={args.batch_size} "
        f"source_len={args.source_len} hbm_slots={args.hbm_slots} "
        f"copy_cap={args.copy_cap} "
        f"copy_range=[{args.copy_min},{args.copy_max}] "
        f"copy_counts={case.copy_counts_cpu.tolist()} "
        f"opapi={ops_overlap.local_opapi_path()}",
        flush=True,
    )

    poison_hbm(case)
    torch.npu.synchronize()
    assert_all_hbm_poisoned(case)
    out_kpe, out_ckv = call_scatter(case)
    torch.npu.synchronize()
    if (
        out_kpe.data_ptr() != case.hbm_kpe.data_ptr()
        or out_ckv.data_ptr() != case.hbm_ckv.data_ptr()
    ):
        raise AssertionError("HBM outputs did not alias the input caches.")
    if copied_tokens:
        assert_copied(case, source_rows, destination_rows)
    else:
        assert_all_hbm_poisoned(case)
    print(
        "A3_SCATTER_DRAM_TO_HBM_CHECK "
        "allocator=empty_with_swapped_memory "
        f"copied_tokens={copied_tokens} "
        f"payload_bytes={payload_bytes} guard_unchanged=1 ok=1",
        flush=True,
    )

    if args.numa_probe or args.require_numa_placement:
        rows = source_rows.tolist()
        buffers = {
            "dram_ckv": inspect_buffer(case.dram_ckv, args.numa_page_samples, rows, CKV_DIM * 2,
                                       host_mappings.get(case.dram_ckv.data_ptr())),
            "dram_kpe": inspect_buffer(case.dram_kpe, args.numa_page_samples, rows, KPE_DIM * 2,
                                       host_mappings.get(case.dram_kpe.data_ptr())),
        }
        nodes = parse_ids(args.memory_nodes) if args.memory_nodes else []
        status = placement_result(buffers, args.memory_policy, nodes)
        numa = {**(numa or {}), "placement_status": status, "buffers": buffers}
        for name, buffer in buffers.items():
            print("A3_SCATTER_NUMA_BUFFER " + json.dumps({
                "name": name, "tensor_ptr": buffer["tensor_ptr"], "host_ptr": buffer["host_ptr"],
                "verified": buffer["verified"],
                "node_pages": buffer.get("allocation", {}).get("node_pages", {}),
                "page_errors": buffer.get("allocation", {}).get("page_errors", {}),
                "error": buffer.get("error"),
            }, sort_keys=True), flush=True)
        print("A3_SCATTER_NUMA_PLACEMENT " + json.dumps(numa, sort_keys=True), flush=True)
        if args.require_numa_placement and status != "sample_verified":
            raise RuntimeError(f"Swapped source placement is {status}; no verified NUMA comparison is possible.")

    launch = lambda: call_scatter(case)
    graph = None
    if args.timing == "graph":
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call_scatter(case)
        launch = graph.replay

    wait_for_multiprocess_start(args)

    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    # Initialize the underlying device events before the common start barrier.
    start.record()
    end.record()
    end.synchronize()
    wait_for_multiprocess_timing_start(args)
    host_start_ns = time.monotonic_ns()
    start.record()
    for _ in range(args.iters):
        launch()
    end.record()
    end.synchronize()
    host_end_ns = time.monotonic_ns()
    avg_us = start.elapsed_time(end) * 1000 / args.iters
    keep_copy_load_until_all_timed(args, launch)

    if copied_tokens:
        assert_copied(case, source_rows, destination_rows)
    else:
        assert_all_hbm_poisoned(case)
    payload_gbps = (
        payload_bytes / (avg_us * 1000)
        if payload_bytes and avg_us
        else 0.0
    )
    print(
        "A3_SCATTER_RESULT "
        f"copy_min={args.copy_min} copy_max={args.copy_max} "
        f"copy_cap={args.copy_cap} copied_tokens={copied_tokens} "
        f"avg_us={avg_us:.3f} payload_gbps={payload_gbps:.3f} "
        f"timer=npu_event timing={args.timing} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "test": "kvcache_scatter_copy",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "device": str(case.device),
            "device_name": case.device_name,
            "dtype": "bf16",
            "batch_size": args.batch_size,
            "source_len": args.source_len,
            "hbm_slots": args.hbm_slots,
            "copy_min": args.copy_min,
            "copy_max": args.copy_max,
            "copy_cap": args.copy_cap,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "timing": args.timing,
            "hold_load_until_all_timed": args.timing_stop_file is not None,
            "opapi": ops_overlap.local_opapi_path(),
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(torch_npu.__version__),
            "torch_npu_git": getattr(getattr(torch_npu, "version", None), "git_version", "unknown"),
        },
        "workload": {
            "copy_counts": case.copy_counts_cpu.tolist(),
            "copied_tokens": copied_tokens,
            "bytes_per_token": (CKV_DIM + KPE_DIM) * 2,
            "payload_bytes_per_iteration": payload_bytes,
        },
        "correctness": {
            "allocator": "empty_with_swapped_memory",
            "data_exact": True,
            "output_alias": True,
            "guard_unchanged": True,
        },
        "performance": {
            "timer": "npu_event",
            "avg_us": avg_us,
            "payload_gbps": payload_gbps,
            "host_start_ns": host_start_ns,
            "host_end_ns": host_end_ns,
        },
        "numa": numa,
    }
    if args.json_output is not None:
        write_json_result(args.json_output, result)
        print(f"A3_SCATTER_JSON path={args.json_output}", flush=True)
    print("A3_KVCACHE_SCATTER_COPY_UT_OK", flush=True)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.memory_policy != "inherit" or args.cpu_list or args.numa_probe or args.require_numa_placement:
        args.numa_setup = apply_worker_policy(
            args.memory_policy,
            parse_ids(args.memory_nodes) if args.memory_nodes else [],
            parse_ids(args.cpu_list) if args.cpu_list else [],
        )
        print("A3_SCATTER_NUMA_SETUP " + json.dumps(args.numa_setup, sort_keys=True), flush=True)
    if args.numa_probe or args.require_numa_placement:
        args.numa_setup["registration_logging"] = prepare_registration_logging()
    global torch, ops_overlap, torch_npu
    import torch
    # Select the repository-local OPP before torch_npu is initialized.
    import ops_overlap
    import torch_npu
    if args.cpu_list:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        expected = set(parse_ids(args.cpu_list))
        if set(os.sched_getaffinity(0)) != expected:
            raise RuntimeError("Runtime changed the worker CPU affinity; controlled NUMA test aborted.")
    run(args)


if __name__ == "__main__":
    main()
