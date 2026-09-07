"""Pure-Python CLI and result helpers shared by the single/multi-card tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BLOCK_SIZE = 128
ROW_BYTES = 656
COPY_CAP = 16384
MAX_SOURCE_LEN = 1 << 18


def add_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dtype", choices=("c8",), default="c8")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--hbm-slots", type=int, default=8192)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--copy-cap", type=int, default=COPY_CAP, help="C8 ABI: fixed at 16384, not the number of copies.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true", help="Portability debugging only; not A5 validation.")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.hbm_slots <= 0:
        raise ValueError("--batch-size and --hbm-slots must be positive.")
    if not 1 <= args.source_len <= MAX_SOURCE_LEN:
        raise ValueError("--source-len must be in [1,262144].")
    if args.copy_cap != COPY_CAP:
        raise ValueError("C8 source/destination metadata must have fixed capacity 16384.")
    if not 0 <= args.copy_min <= args.copy_max <= min(COPY_CAP, args.source_len, args.hbm_slots):
        raise ValueError("Require 0 <= copy-min <= copy-max <= min(16384, source-len, hbm-slots).")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
