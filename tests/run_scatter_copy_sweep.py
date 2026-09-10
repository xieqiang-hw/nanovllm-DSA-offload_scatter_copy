#!/usr/bin/env python3
"""Sweep logical die counts, per-die batch sizes and per-request copy counts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from analyze_scatter_copy_results import analyze
from scatter_cli import parse_ids, validate_workload
from scatter_config import COPY_CAP_DEFAULT, DEFAULT_VISIBLE_DEVICES, DTYPE, LEGACY_VISIBLE_ENV, RESULTS_SUBDIR

ROOT = Path(__file__).resolve().parents[1]


def integer_list(value: str, name: str, minimum: int) -> list[int]:
    try:
        values = [int(part) for part in value.split()]
    except ValueError as error:
        raise ValueError(f"{name} must contain space-separated integers.") from error
    if not values or min(values) < minimum or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain distinct integers >= {minimum}.")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default=os.getenv("TEST_VISIBLE_DEVICES", os.getenv(LEGACY_VISIBLE_ENV,
                        os.getenv("ASCEND_RT_VISIBLE_DEVICES", DEFAULT_VISIBLE_DEVICES))))
    parser.add_argument("--cards", nargs="+", type=int, help="Logical die counts; environment: CARD_COUNTS.")
    parser.add_argument("--batch-size", nargs="+", type=int, help="Batch sizes per die; environment: BATCH_SIZES.")
    parser.add_argument("--copy-count", nargs="+", type=int, help="Fixed copies per request; environment: COPY_COUNTS.")
    for flag, env, default in (("source-len", "SOURCE_LEN", 65536), ("hbm-slots", "HBM_SLOTS", 8192),
                               ("copy-cap", "COPY_CAP", COPY_CAP_DEFAULT), ("warmup", "WARMUP", 10),
                               ("iters", "ITERS", 1000), ("seed", "SEED", 7)):
        parser.add_argument(f"--{flag}", type=int, default=os.getenv(env, str(default)))
    parser.add_argument("--results-dir", type=Path, default=Path(os.getenv("RESULTS_DIR", str(ROOT / RESULTS_SUBDIR))))
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN") == "1",
                        help="Validate and print commands without importing Torch, loading CANN or starting NPUs.")
    args = parser.parse_args(argv)
    visible = parse_ids(args.visible_devices)
    args.visible_devices = ",".join(map(str, visible))
    defaults = " ".join(str(n) for n in (1, 2, 4, 8, 16) if n <= len(visible))
    for key, env, default, minimum in (("cards", "CARD_COUNTS", defaults, 1),
                                      ("batch_size", "BATCH_SIZES", "8 16 24 32", 1),
                                      ("copy_count", "COPY_COUNTS", "0 100 200 300 500 2048", 0)):
        command_values = getattr(args, key)
        value = " ".join(map(str, command_values)) if command_values is not None else os.getenv(env, default)
        setattr(args, key, integer_list(value, env, minimum))
    if max(args.cards) > len(visible):
        raise ValueError(f"Requested {max(args.cards)} dies, but only {len(visible)} devices are visible.")
    if os.getenv("DTYPES", DTYPE).split() != [DTYPE]:
        raise ValueError(f"The unified sweep in this branch uses {DTYPE}; DTYPES must be {DTYPE} if set.")
    for batch in args.batch_size:
        for count in args.copy_count:
            validate_workload(argparse.Namespace(**{**vars(args), "batch_size": batch,
                                                    "copy_min": count, "copy_max": count}))
    args.results_dir = args.results_dir.resolve()
    return args


def cases(args):
    for cards in args.cards:
        for batch in args.batch_size:
            for copies in args.copy_count:
                name = f"cards{cards}_bs{batch}_copy{copies}"
                output = args.results_dir / f"{name}.json"
                command = [sys.executable, str(ROOT / "tests/test_scatter_copy_multiprocess.py"),
                           "--cards", str(cards), "--batch-size", str(batch), "--copy-count", str(copies),
                           "--source-len", str(args.source_len), "--hbm-slots", str(args.hbm_slots),
                           "--copy-cap", str(args.copy_cap), "--warmup", str(args.warmup),
                           "--iters", str(args.iters), "--seed", str(args.seed), "--same-seed",
                           "--output", str(output)]
                yield name, output, command


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args) -> None:
    planned = list(cases(args))
    if args.dry_run:
        print(f"ASCEND_RT_VISIBLE_DEVICES={args.visible_devices} dtype={DTYPE} cases={len(planned)}")
        for _, _, command in planned:
            print(shlex.join(command))
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / "scatter_copy_run_manifest.json"
    manifest = {"schema_version": 2, "status": "running", "dtype": DTYPE,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "visible_devices": args.visible_devices,
                "case_files": [output.name for _, output, _ in planned]}
    write_json(manifest_path, manifest)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices
    env["PYTHONPATH"] = str(ROOT / "torch_extension") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    try:
        for name, output, command in planned:
            print(f"Running {name}; log={output.with_suffix('.log')}", flush=True)
            # A failed rerun must never leave a prior successful result behind.
            write_json(output, {"status": "running"})
            with output.with_suffix(".log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                write_json(output, {"status": "failed", "returncode": completed.returncode})
                print("".join(output.with_suffix(".log").read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-30:]), end="", file=sys.stderr)
                raise RuntimeError(f"{name} failed; see {output.with_suffix('.log')}")
        manifest["status"] = "passed"
        write_json(manifest_path, manifest)
        analyze(args.results_dir)
    except BaseException:
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
