"""Shared six-column scatter results; all bandwidths use decimal GB/s."""

from __future__ import annotations

import math
import statistics

from scatter_config import BYTES_PER_TOKEN, DTYPE, MULTIPROCESS_TEST

COLUMNS = ("cards", "batch_size", "copy_count", "avg_us_mean", "avg_us_max", "avg_bandwidth")


def effective_bandwidth(payload_bytes: int, avg_us: float) -> float:
    if payload_bytes < 0 or not math.isfinite(avg_us) or avg_us <= 0:
        raise ValueError("Payload must be non-negative and avg_us must be finite and positive.")
    return payload_bytes / (avg_us * 1000.0)


def summarize_devices(per_device: list[dict]) -> dict:
    if not per_device:
        raise ValueError("No per-device results.")
    latencies = [float(item["performance"]["avg_us"]) for item in per_device]
    bandwidths = [effective_bandwidth(int(item["workload"]["payload_bytes_per_iteration"]), latency)
                  for item, latency in zip(per_device, latencies)]
    return {
        "avg_us_mean": statistics.fmean(latencies),
        "avg_us_max": max(latencies),
        # Sum each die's rate, before rounding; never divide by a mean/max latency.
        "avg_bandwidth": math.fsum(bandwidths),
    }


def copy_label(config: dict) -> int | str:
    low, high = int(config["copy_min"]), int(config["copy_max"])
    if not 0 <= low <= high:
        raise ValueError("Invalid copy count range.")
    return low if low == high else f"{low}..{high}"


def single_result(args, payload_bytes: int, avg_us: float) -> dict:
    return {
        "cards": 1, "batch_size": args.batch_size, "copy_count": copy_label(vars(args)),
        "avg_us_mean": avg_us, "avg_us_max": avg_us,
        "avg_bandwidth": effective_bandwidth(payload_bytes, avg_us),
    }


def normalize_case(case: dict) -> dict:
    """Recompute new AND historical files from raw per-die bytes and timings."""
    if case.get("status") != "passed" or case.get("test") != MULTIPROCESS_TEST:
        raise ValueError(f"Expected a passing {MULTIPROCESS_TEST} result.")
    workers = case["per_device"]
    count = int(case["device_count"])
    if count < 1 or len(workers) != count:
        raise ValueError("device_count does not match per_device.")
    if case.get("summary", {}).get("all_correct") is not True:
        raise ValueError("Multiprocess correctness checks did not pass.")
    config = case.get("config", workers[0]["config"])
    label = copy_label(config)
    for item in workers:
        if item.get("status") != "passed" or any(value is False for value in item.get("correctness", {}).values()):
            raise ValueError("A worker did not pass its correctness checks.")
        worker_config = item["config"]
        if worker_config.get("row_bytes", BYTES_PER_TOKEN) != BYTES_PER_TOKEN:
            raise ValueError("Incorrect payload row size.")
        if worker_config.get("dtype") != DTYPE:
            raise ValueError(f"This branch reports only {DTYPE} results.")
        for key in ("batch_size", "copy_min", "copy_max", "source_len", "hbm_slots", "copy_cap", "warmup", "iters"):
            if key in config and worker_config.get(key) != config[key]:
                raise ValueError(f"Workers have inconsistent {key} values.")
        workload = item["workload"]
        tokens = int(workload["copied_tokens"])
        payload = int(workload["payload_bytes_per_iteration"])
        if tokens < 0 or payload != tokens * BYTES_PER_TOKEN:
            raise ValueError(f"Expected {BYTES_PER_TOKEN} effective bytes per copied token.")
        if isinstance(label, int) and tokens != int(config["batch_size"]) * label:
            raise ValueError("Actual copied tokens disagree with the fixed copy count.")
    return {
        "cards": count, "batch_size": int(config["batch_size"]), "copy_count": label,
        **summarize_devices(workers),
    }


def sort_key(row: dict) -> tuple:
    low, _, high = str(row["copy_count"]).partition("..")
    return int(row["cards"]), int(row["batch_size"]), int(low), int(high or low)


def build_rows(cases: list[dict]) -> list[dict]:
    rows = [normalize_case(case) for case in cases]
    if not rows:
        raise ValueError("No cases available for analysis.")
    seen = set()
    for row in rows:
        key = sort_key(row)
        if key in seen:
            raise ValueError(f"Duplicate cards/batch_size/copy_count: {key}; use a separate results directory or run manifest.")
        seen.add(key)
    return sorted(rows, key=sort_key)


def formatted_row(row: dict) -> dict[str, str]:
    return {key: f"{float(row[key]):.3f}" if key.startswith("avg_") else str(row[key]) for key in COLUMNS}


def format_table(rows: list[dict], markdown: bool = False) -> str:
    if not rows:
        raise ValueError("No rows to format.")
    cells = [formatted_row(row) for row in rows]
    if markdown:
        lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join(["---:"] * len(COLUMNS)) + " |"]
        lines += ["| " + " | ".join(row[key] for key in COLUMNS) + " |" for row in cells]
    else:
        widths = {key: max(len(key), *(len(row[key]) for row in cells)) for key in COLUMNS}
        lines = ["  ".join(key.rjust(widths[key]) for key in COLUMNS)]
        lines += ["  ".join(row[key].rjust(widths[key]) for key in COLUMNS) for row in cells]
    return "\n".join(lines) + "\n"


def print_result(row: dict) -> None:
    print(format_table([row]), end="", flush=True)
