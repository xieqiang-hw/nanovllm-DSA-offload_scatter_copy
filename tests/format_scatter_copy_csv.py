#!/usr/bin/env python3
"""Render the six-column CSV without recalculating or rounding inputs to integers."""

import argparse
import csv
from pathlib import Path

from scatter_results import COLUMNS, format_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Expected the six-column CSV; re-analyze historical JSON files first.")
        table = format_table(list(reader), markdown=True)
    output = args.output or args.input.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
