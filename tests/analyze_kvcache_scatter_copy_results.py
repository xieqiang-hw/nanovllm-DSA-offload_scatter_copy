#!/usr/bin/env python3
"""Collect packed-C8 multi-card JSON results into a CSV (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from _common import ROW_BYTES, write_json


def normalize_case(case: dict) -> dict:
    if case.get("status") != "passed" or case.get("test") != "a5_kvcache_scatter_copy_c8_multiprocess":
        raise ValueError("Expected a passing packed-C8 multiprocess result.")
    config, summary = case["config"], case["summary"]
    if config["dtype"] != "c8" or config["row_bytes"] != ROW_BYTES or not summary["all_correct"]:
        raise ValueError("Wrong payload format or failed correctness checks.")
    return {
        "cards": int(case["device_count"]), "dtype": "c8", "row_bytes": ROW_BYTES,
        "batch_size": int(config["batch_size"]), "source_len": int(config["source_len"]),
        "hbm_slots": int(config["hbm_slots"]), "copy_min": int(config["copy_min"]),
        "copy_max": int(config["copy_max"]), "warmup": int(config["warmup"]), "iters": int(config["iters"]),
        "seed": int(config["seed"]), "same_seed": case["same_seed"],
        "copied_tokens_sum_per_iteration": int(summary["copied_tokens_sum_per_iteration"]),
        "total_payload_bytes_per_iteration": int(summary["total_payload_bytes_per_iteration"]),
        **{key: summary[key] for key in (
            "avg_us_min", "avg_us_mean", "avg_us_max",
            "payload_gbps_sum", "payload_gbps_sum_by_avg_us_max",
        )},
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "results/kvcache_scatter_copy_c8_multiprocess")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("cards*_*.json"))
    if not paths:
        raise FileNotFoundError(f"No cards*_*.json results in {args.results_dir}")
    cases, rows = [], []
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            rows.append(normalize_case(case))
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
        cases.append(case)
    rows.sort(key=lambda row: tuple(row[key] for key in ("cards", "batch_size", "source_len", "hbm_slots", "copy_min", "copy_max")))
    output = args.output or args.results_dir / "kvcache_scatter_copy_timing_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.results_dir / "kvcache_scatter_copy_multiprocess_summary.json", {
        "schema_version": 1, "test": "a5_kvcache_scatter_copy_c8_multiprocess_sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "case_count": len(cases), "cases": cases,
    })
    print(f"Saved {len(rows)} C8 cases: {output}")


if __name__ == "__main__":
    main()
