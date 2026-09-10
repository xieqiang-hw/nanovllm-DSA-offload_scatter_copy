"""CLI, workload definitions and the single terminal report (stdlib only)."""
from __future__ import annotations

import argparse
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW_BYTES = {"bf16": 1152, "c8": 656}
COLUMNS = ("data-type", "card-count", "batch-size", "copy-count",
           "avg_us_mean", "avg_us_max", "avg_bandwidth")


@dataclass(frozen=True)
class Workload:
    dtype: str
    batch_size: int = 8
    copy_count: int = 300
    source_len: int = 65536
    hbm_slots: int = 8192
    copy_cap: int = 16384
    warmup: int = 10
    iters: int = 1000
    seed: int = 7


def device_ids(text: str) -> list[int]:
    if not re.fullmatch(r"\d+(,\d+)*", text):
        raise argparse.ArgumentTypeError("Use comma-separated non-negative device IDs.")
    values = list(map(int, text.split(",")))
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Device IDs must be unique.")
    return values


def parse_args(argv=None, *, multi=False):
    parser = argparse.ArgumentParser(description="Caller-owned BF16 and packed-C8 DRAM -> HBM scatter.")
    if multi:
        parser.add_argument("--visible-devices", type=device_ids, help="Runtime device IDs; defaults to the existing visible set.")
        parser.add_argument("--cards", type=int, nargs="+", default=[1], help="Logical die counts.")
        parser.add_argument("--timeout", type=float, default=600, help="Seconds allowed for each multiprocess case.")
        parser.add_argument("--dry-run", action="store_true", help="Validate and list cases without Torch/CANN or output files.")
        parser.set_defaults(dtype=["bf16", "c8"])
    else:
        parser.add_argument("--device", default="npu:0")
        parser.add_argument("--dtype", choices=tuple(ROW_BYTES), nargs="+", default=list(ROW_BYTES))
        parser.add_argument("--graph", action="store_true", help="Also verify NPU graph replay with updated metadata.")
    parser.add_argument("--batch-size", type=int, nargs="+" if multi else None, default=[8] if multi else 8)
    parser.add_argument("--copy-count", type=int, nargs="+" if multi else None, default=[300] if multi else 300)
    for name, default in (("source-len", 65536), ("hbm-slots", 8192), ("copy-cap", 16384),
                          ("warmup", 10), ("iters", 1000), ("seed", 7)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args(argv)
    lists = [args.cards, args.batch_size, args.copy_count] if multi else [args.dtype]
    if any(len(x) != len(set(x)) for x in lists):
        parser.error("Repeated values are not allowed.")
    batches = args.batch_size if multi else [args.batch_size]
    copies = args.copy_count if multi else [args.copy_count]
    if min(batches) < 1 or max(batches) > 2**32 - 1:
        parser.error("batch-size must be in [1, 2^32-1].")
    if not (1 <= args.copy_cap <= 65536 and 1 <= args.source_len <= 2**31 and 1 <= args.hbm_slots <= 2**31):
        parser.error("Require 1 <= copy-cap <= 65536 and positive int32-addressable source/HBM capacities.")
    if min(copies) < 0 or max(copies) > min(args.copy_cap, args.source_len, args.hbm_slots):
        parser.error("copy-count must be in [0, min(copy-cap, source-len, hbm-slots)].")
    if args.warmup < 0 or args.iters < 1 or not 0 <= args.seed < 2**63:
        parser.error("Require warmup >= 0, iters >= 1 and 0 <= seed < 2^63.")
    if multi:
        if min(args.cards) < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("cards and timeout must be positive.")
        if args.visible_devices is not None and max(args.cards) > len(args.visible_devices):
            parser.error("Requested die count exceeds --visible-devices.")
    elif not re.fullmatch(r"npu:\d+", args.device):
        parser.error("--device must be npu:N.")
    return args


def workload(args, dtype, batch, count):
    return Workload(dtype, batch, count, args.source_len, args.hbm_slots,
                    args.copy_cap, args.warmup, args.iters, args.seed)


def cases(args):
    for cards, batch, count, dtype in itertools.product(args.cards, args.batch_size, args.copy_count, args.dtype):
        yield cards, workload(args, dtype, batch, count)


def summarize(config: Workload, results: list[dict]) -> tuple:
    if not results:
        raise ValueError("No die results received.")
    expected = config.batch_size * config.copy_count * ROW_BYTES[config.dtype]
    if any(not math.isfinite(r["avg_us"]) or r["avg_us"] <= 0 or r["payload_bytes"] != expected for r in results):
        raise ValueError("Invalid die timing or effective copy byte count.")
    times = [r["avg_us"] for r in results]
    bandwidth = math.fsum(r["payload_bytes"] / (r["avg_us"] * 1000) for r in results)
    return (config.dtype, len(results), config.batch_size, config.copy_count,
            math.fsum(times) / len(times), max(times), bandwidth)


def print_table(rows: list[tuple]) -> None:
    values = [COLUMNS] + [tuple(str(v) if i < 4 else f"{v:.3f}" for i, v in enumerate(row)) for row in rows]
    widths = [max(len(row[i]) for row in values) for i in range(len(COLUMNS))]
    for row in values:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)), flush=True)
