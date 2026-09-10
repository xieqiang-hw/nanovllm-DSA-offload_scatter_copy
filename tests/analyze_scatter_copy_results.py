#!/usr/bin/env python3
"""Render scatter results using exactly the shared six columns (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scatter_config import RESULTS_SUBDIR
from scatter_results import COLUMNS, build_rows, format_table, formatted_row


def load_cases(results_dir: Path) -> tuple[list[Path], list[dict]]:
    manifest = results_dir / "scatter_copy_run_manifest.json"
    if manifest.exists():
        run = json.loads(manifest.read_text(encoding="utf-8"))
        if run.get("status") != "passed":
            raise ValueError("The latest sweep did not complete successfully; inspect its logs.")
        paths = [(results_dir / name).resolve() for name in run["case_files"]]
        if any(path.parent != results_dir.resolve() for path in paths):
            raise ValueError("Run manifest contains a path outside its results directory.")
    else:
        # Historical A3, A5 BF16 and C8 names are all accepted.
        paths = sorted(results_dir.glob("cards*.json"))
    if not paths:
        raise FileNotFoundError(f"No case JSON files found in {results_dir}")
    return paths, [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def analyze(results_dir: Path, output: Path | None = None) -> list[dict]:
    paths, cases = load_cases(results_dir)
    rows = build_rows(cases)
    output = output or results_dir / "scatter_copy_timing_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(formatted_row(row) for row in rows)
    output.with_suffix(".md").write_text(format_table(rows, markdown=True), encoding="utf-8")
    (results_dir / "scatter_copy_multiprocess_summary.json").write_text(json.dumps({
        "schema_version": 2, "columns": COLUMNS, "case_count": len(rows),
        "case_files": [str(path.name) for path in paths], "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_table(rows), end="")
    print(f"Saved: {output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / RESULTS_SUBDIR)
    parser.add_argument("--output", type=Path, help="CSV path; a six-column Markdown file is also written alongside it.")
    args = parser.parse_args()
    analyze(args.results_dir.resolve(), args.output)


if __name__ == "__main__":
    main()
