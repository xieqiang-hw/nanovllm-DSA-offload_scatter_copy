#!/usr/bin/env python3
"""Print the scatter-copy timing CSV as compact Markdown tables."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


COPY_COUNTS = (100, 200, 300, 500, 2048)
SUPPORTED_CARDS = (1, 4, 8, 16)
PAYLOADS = {
    8: {100: 0.9, 200: 1.8, 300: 2.8, 500: 4.6, 2048: 18.8},
    32: {100: 3.6, 200: 7.4, 300: 11.1, 500: 18.4, 2048: 75.5},
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            repo_root
            / "results"
            / "kvcache_scatter_copy_multiprocess"
            / "kvcache_scatter_copy_timing_summary.csv"
        ),
        help="Timing CSV produced by analyze_kvcache_scatter_copy_results.py.",
    )
    return parser.parse_args()


def rounded_int(value: str) -> int:
    return int(Decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def load_rows(path: Path) -> tuple[list[int], dict[tuple[int, int, int], int]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {"cards", "batch_size", "copy_count"}
        if not required <= fields:
            missing = ", ".join(sorted(required - fields))
            raise ValueError(f"Missing required CSV columns: {missing}")

        if {"avg_us_min", "avg_us_mean", "avg_us_max"} <= fields:
            timing_field = "avg_us_mean"
        elif "avg_us" in fields:
            timing_field = "avg_us"
        else:
            raise ValueError(
                "CSV must contain avg_us, or avg_us_min/avg_us_mean/avg_us_max"
            )

        batch_sizes: set[int] = set()
        timings: dict[tuple[int, int, int], int] = {}
        for line_number, row in enumerate(reader, start=2):
            cards = int(row["cards"])
            copy_count = int(row["copy_count"])
            if cards not in SUPPORTED_CARDS or copy_count not in COPY_COUNTS:
                continue
            batch_size = int(row["batch_size"])
            key = (batch_size, copy_count, cards)
            if key in timings:
                raise ValueError(f"Duplicate row for {key} at CSV line {line_number}")
            timings[key] = rounded_int(row[timing_field])
            batch_sizes.add(batch_size)

    return sorted(batch_sizes), timings


def print_tables(batch_sizes: list[int], timings: dict[tuple[int, int, int], int]) -> None:
    cards = [
        card
        for card in SUPPORTED_CARDS
        if any(key_cards == card for _, _, key_cards in timings)
    ]
    if not cards:
        raise ValueError("CSV contains no 1-card, 4-card, 8-card, or 16-card data")

    timing_title = "/".join(f"{card}卡" for card in cards) + " avgtime（μs）"
    missing_messages: list[str] = []

    for batch_size in batch_sizes:
        print(f"### BS = {batch_size}\n")
        print(f"| copy_count | {timing_title} | 8卡折算带宽 |")
        print("| ---: | :--- | ---: |")
        for copy_count in COPY_COUNTS:
            values: list[str] = []
            for card in cards:
                value = timings.get((batch_size, copy_count, card))
                values.append(str(value) if value is not None else "—")
                if value is None:
                    missing_messages.append(
                        f"BS={batch_size}, copy_count={copy_count} 缺少{card}卡数据"
                    )

            avg_8_cards = timings.get((batch_size, copy_count, 8))
            payload = PAYLOADS.get(batch_size, {}).get(copy_count)
            if avg_8_cards is None:
                bandwidth = "—"
                missing_messages.append(
                    f"BS={batch_size}, copy_count={copy_count} 缺少8卡数据，"
                    "8卡折算带宽未计算"
                )
            elif payload is None:
                bandwidth = "—"
                missing_messages.append(
                    f"BS={batch_size}, copy_count={copy_count} 未配置对应数据量，"
                    "8卡折算带宽未计算"
                )
            else:
                bandwidth = f"{payload / avg_8_cards * 1000 * 8:.2f}"
            print(f"| {copy_count} | {'/'.join(values)} | {bandwidth} |")
        print()

    if missing_messages:
        print("缺失数据说明：")
        for message in dict.fromkeys(missing_messages):
            print(f"- {message}")


def main() -> None:
    args = parse_args()
    batch_sizes, timings = load_rows(args.input.resolve())
    print_tables(batch_sizes, timings)


if __name__ == "__main__":
    main()
