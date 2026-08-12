#!/usr/bin/env python3

"""Summarize A5 multiprocess Scatter Copy timing results."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results" / "a5_multiprocess",
        help="Directory produced by run_scatter_copy_multiprocess.sh.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: RESULTS_DIR/scatter_copy_timing_summary.csv).",
    )
    return parser.parse_args()


def normalize_case(case: dict[str, object]) -> dict[str, object]:
    config = case["config"]
    summary = case["summary"]
    return {
        "cards": case["device_count"],
        "dtype": config["dtype"],
        "batch_size": config["batch_size"],
        "copy_count": config["copy_max"],
        "avg_us_min": summary["avg_us_min"],
        "avg_us_mean": summary["avg_us_mean"],
        "avg_us_max": summary["avg_us_max"],
        "payload_gbps_sum": summary["payload_gbps_sum"],
        "payload_gbps_sum_by_avg_us_max": summary[
            "payload_gbps_sum_by_avg_us_max"
        ],
    }


def load_cases(results_dir: Path) -> list[dict[str, object]]:
    case_paths = sorted(results_dir.glob("cards*_*.json"))
    if not case_paths:
        raise FileNotFoundError(f"No case JSON files found in {results_dir}")
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in case_paths
    ]


def write_summary(
    results_dir: Path, cases: list[dict[str, object]]
) -> Path:
    summary_path = results_dir / "scatter_copy_multiprocess_summary.json"
    summary = {
        "schema_version": 1,
        "test": "a5_kvcache_scatter_copy_multiprocess_sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "cases": cases,
    }
    temporary = summary_path.with_name(
        f".{summary_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary_path


def build_timing(cases: list[dict[str, object]]) -> pd.DataFrame:
    rows = [normalize_case(case) for case in cases]
    if not rows:
        raise ValueError("No cases available for analysis.")
    timing = pd.DataFrame(rows).sort_values(
        ["cards", "dtype", "batch_size", "copy_count"]
    )
    timing = timing.reset_index(drop=True)
    for column in timing.select_dtypes(include="number").columns:
        if column not in {"cards", "batch_size", "copy_count"}:
            timing[column] = timing[column].round(3)
    return timing


def print_tables(timing: pd.DataFrame) -> None:
    print("\n===== A5 MULTIPROCESS ALL RESULTS =====")
    print(timing.to_string(index=False))
    table = timing.pivot_table(
        index=["cards", "dtype", "batch_size"],
        columns="copy_count",
        values=["avg_us_mean", "avg_us_max", "payload_gbps_sum_by_avg_us_max"],
        aggfunc="first",
    ).sort_index(axis=1, level=[0, 1])
    print("\n===== A5 MULTIPROCESS PIVOT =====")
    print(table.to_string())


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results_dir / "scatter_copy_timing_summary.csv"
    )
    cases = load_cases(results_dir)
    summary_path = write_summary(results_dir, cases)
    print(f"Wrote {len(cases)} cases to {summary_path}")
    timing = build_timing(cases)
    print_tables(timing)
    output.parent.mkdir(parents=True, exist_ok=True)
    timing.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
