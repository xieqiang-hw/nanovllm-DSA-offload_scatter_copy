#!/usr/bin/env python3
"""Render C8 results without hard-coded card counts or BF16 payload sizes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def format_rows(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("CSV contains no results.")
    lines = [
        "| 卡数 | BS/卡 | 源长度 | HBM slots/请求 | copy/请求 | 总 payload (MB/轮) | 平均时延 (us) | 最慢卡时延 (us) | 总带宽估计 (GB/s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["dtype"] != "c8" or int(row["row_bytes"]) != 656:
            raise ValueError("CSV must contain only packed-C8 (656-byte) rows.")
        low, high = int(row["copy_min"]), int(row["copy_max"])
        copies = str(low) if low == high else f"{low}~{high}"
        lines.append(
            f"| {row['cards']} | {row['batch_size']} | {row['source_len']} | {row['hbm_slots']} | {copies} | "
            f"{int(row['total_payload_bytes_per_iteration']) / 1e6:.3f} | "
            f"{float(row['avg_us_mean']):.3f} | {float(row['avg_us_max']):.3f} | "
            f"{float(row['payload_gbps_sum_by_avg_us_max']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        table = format_rows(list(csv.DictReader(stream)))
    output = args.output or args.input.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
