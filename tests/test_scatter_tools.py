#!/usr/bin/env python3
"""CPU-only regression tests for the public CLI, aggregation and result files."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from analyze_scatter_copy_results import analyze, load_cases
from run_scatter_copy_sweep import cases, parse_args, run
from scatter_cli import selected_devices
from scatter_config import BYTES_PER_TOKEN, COPY_CAP_DEFAULT, DTYPE, LEGACY_VISIBLE_ENV, MULTIPROCESS_TEST
from scatter_results import COLUMNS, build_rows, effective_bandwidth, format_table, normalize_case

ROOT = Path(__file__).resolve().parents[1]


def fixture(latencies=(10.0, 30.0), batch=3, copies=7):
    config = dict(dtype=DTYPE, batch_size=batch, source_len=65536, hbm_slots=8192,
                  copy_min=copies, copy_max=copies, copy_cap=COPY_CAP_DEFAULT,
                  warmup=0, iters=5, seed=7)
    workers = []
    for rank, latency in enumerate(latencies):
        workers.append({
            "status": "passed", "test": MULTIPROCESS_TEST.removesuffix("_multiprocess") + "_singlecard",
            "config": {**config, "device": f"npu:{rank}"}, "rank": rank,
            "correctness": {"data_exact": True},
            "workload": {"copied_tokens": batch * copies, "payload_bytes_per_iteration": batch * copies * BYTES_PER_TOKEN},
            "performance": {"avg_us": latency, "host_start_ns": 1000000 + rank,
                            "host_end_ns": 1000000 + rank + int(latency * 5000)},
        })
    return {"status": "passed", "test": MULTIPROCESS_TEST, "device_count": len(workers),
            "config": config, "summary": {"all_correct": True}, "per_device": workers}


class ResultsTests(unittest.TestCase):
    def test_unequal_die_latencies_sum_individual_rates(self):
        result = normalize_case(fixture())
        # For 21 tokens per die and 10/30 us: sum is 2.8 * bytes_per_token / 1000.
        self.assertAlmostEqual(result["avg_bandwidth"], BYTES_PER_TOKEN * 0.0028)
        self.assertEqual(result["avg_us_mean"], 20)
        self.assertEqual(result["avg_us_max"], 30)
        self.assertNotAlmostEqual(result["avg_bandwidth"], 42 * BYTES_PER_TOKEN / 20000)
        self.assertNotAlmostEqual(result["avg_bandwidth"], 42 * BYTES_PER_TOKEN / 30000)
        self.assertEqual(tuple(result), COLUMNS)

    def test_zero_copy_and_single_die(self):
        self.assertEqual(normalize_case(fixture(copies=0))["avg_bandwidth"], 0)
        row = normalize_case(fixture(latencies=(20,)))
        self.assertEqual(row["cards"], 1)
        self.assertEqual(row["avg_us_mean"], row["avg_us_max"])
        self.assertAlmostEqual(row["avg_bandwidth"], 21 * BYTES_PER_TOKEN / 20000)

    def test_actual_bytes_can_differ_between_dies_for_legacy_random_counts(self):
        case = fixture()
        case["config"]["copy_min"] = 0
        for worker in case["per_device"]:
            worker["config"]["copy_min"] = 0
        case["per_device"][0]["workload"] = {"copied_tokens": 1, "payload_bytes_per_iteration": BYTES_PER_TOKEN}
        row = normalize_case(case)
        self.assertEqual(row["copy_count"], "0..7")
        self.assertAlmostEqual(row["avg_bandwidth"], BYTES_PER_TOKEN / 10000 + 21 * BYTES_PER_TOKEN / 30000)

    def test_old_summary_and_per_die_rates_are_not_used_or_re_emitted(self):
        case = fixture()
        case["summary"].update(payload_gbps_sum=999, payload_gbps_sum_by_avg_us_max=888)
        for worker in case["per_device"]:
            worker["performance"]["payload_gbps"] = 777
        self.assertAlmostEqual(normalize_case(case)["avg_bandwidth"], BYTES_PER_TOKEN * 0.0028)

    def test_a3_legacy_config_inside_per_device(self):
        case = fixture()
        del case["config"]
        self.assertEqual(normalize_case(case)["copy_count"], 7)

    def test_invalid_time_never_produces_a_plausible_rate(self):
        for time in (0, -1, float("inf"), float("nan")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                effective_bandwidth(1000, time)

    def test_bad_dtype_payload_failed_case_and_count_mismatch_rejected(self):
        bad = []
        case = fixture(); case["status"] = "failed"; bad.append(case)
        case = fixture(); case["device_count"] = 3; bad.append(case)
        case = fixture(); case["per_device"][0]["config"]["dtype"] = "wrong"; bad.append(case)
        case = fixture(); case["per_device"][0]["workload"]["payload_bytes_per_iteration"] += 1; bad.append(case)
        case = fixture(); case["per_device"][0]["correctness"]["data_exact"] = False; bad.append(case)
        for case in bad:
            with self.subTest(case=case), self.assertRaises(ValueError):
                normalize_case(case)

    def test_six_columns_and_arbitrary_counts_in_text_and_markdown(self):
        rows = build_rows([fixture(latencies=(12,), batch=17, copies=321), fixture(batch=5, copies=0)])
        self.assertEqual(format_table(rows).splitlines()[0].split(), list(COLUMNS))
        markdown = format_table(rows, markdown=True)
        self.assertEqual([part.strip() for part in markdown.splitlines()[0].strip("|").split("|")], list(COLUMNS))
        self.assertIn("321", markdown)
        self.assertIn("17", markdown)

    def test_duplicate_coordinates_are_not_silently_merged(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_rows([fixture(), fixture()])

    def test_csv_markdown_json_share_six_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = fixture()
            old["summary"].update(payload_gbps_sum=999, payload_gbps_sum_by_avg_us_max=888)
            (root / "cards2_bf16_bs3_copy7.json").write_text(json.dumps(old))
            with contextlib.redirect_stdout(io.StringIO()):
                analyze(root)
            output = root / "scatter_copy_timing_summary.csv"
            with output.open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), COLUMNS)
                self.assertEqual(len(list(reader)), 1)
            summary = json.loads((root / "scatter_copy_multiprocess_summary.json").read_text())
            self.assertEqual(tuple(summary["rows"][0]), COLUMNS)
            self.assertNotIn("payload_gbps", json.dumps(summary))
            formatted = subprocess.run([sys.executable, str(ROOT / "tests/format_scatter_copy_csv.py"), "--input", str(output)], capture_output=True, text=True)
            self.assertEqual(formatted.returncode, 0, formatted.stderr)
            self.assertEqual(formatted.stdout, output.with_suffix(".md").read_text())

    def test_manifest_excludes_stale_files_and_failed_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cards_current.json").write_text(json.dumps(fixture()))
            (root / "cards_stale.json").write_text(json.dumps({"status": "failed"}))
            manifest = root / "scatter_copy_run_manifest.json"
            manifest.write_text(json.dumps({"status": "passed", "case_files": ["cards_current.json"]}))
            self.assertEqual(len(load_cases(root)[1]), 1)
            manifest.write_text(json.dumps({"status": "failed", "case_files": ["cards_current.json"]}))
            with self.assertRaises(ValueError):
                load_cases(root)


class CliTests(unittest.TestCase):
    def test_environment_sweep_and_logical_device_mapping(self):
        with patch.dict(os.environ, {"TEST_VISIBLE_DEVICES": "3,7", "CARD_COUNTS": "1 2",
                                    "BATCH_SIZES": "5 17", "COPY_COUNTS": "0 321"}, clear=True):
            args = parse_args(["--dry-run"])
        planned = list(cases(args))
        self.assertEqual(len(planned), 8)
        self.assertEqual(args.visible_devices, "3,7")
        self.assertEqual(args.copy_cap, COPY_CAP_DEFAULT)
        self.assertIn("321", planned[-1][2])
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "3,7"}, clear=True):
            self.assertEqual(selected_devices(argparse.Namespace(cards=2, devices=None)), [0, 1])

    def test_command_line_overrides_environment(self):
        with patch.dict(os.environ, {"CARD_COUNTS": "999", "BATCH_SIZES": "0", "COPY_COUNTS": "-1"}, clear=True):
            args = parse_args(["--visible-devices", "4,9", "--cards", "2", "--batch-size", "8", "16", "--copy-count", "200", "300"])
        self.assertEqual((args.cards, args.batch_size, args.copy_count), ([2], [8, 16], [200, 300]))

    def test_legacy_visibility_alias(self):
        with patch.dict(os.environ, {LEGACY_VISIBLE_ENV: "2,5"}, clear=True):
            self.assertEqual(parse_args([]).visible_devices, "2,5")

    def test_invalid_requests_fail_before_launch(self):
        for env in ({"CARD_COUNTS": "3"}, {"CARD_COUNTS": "0"}, {"CARD_COUNTS": "1 1"},
                    {"BATCH_SIZES": "0"}, {"COPY_COUNTS": "-1"}, {"COPY_COUNTS": "20000"},
                    {"COPY_COUNTS": ""}, {"ITERS": "0"}, {"TEST_VISIBLE_DEVICES": "1,,2"},
                    {"TEST_VISIBLE_DEVICES": "1,1"}, {"DTYPES": "bf16 fp16"}):
            with self.subTest(env=env), patch.dict(os.environ, {"TEST_VISIBLE_DEVICES": "0,1", **env}, clear=True):
                with self.assertRaises(ValueError):
                    parse_args([])

    def test_zero_copy_and_c8_capacity(self):
        with patch.dict(os.environ, {}, clear=True):
            args = parse_args(["--cards", "1", "--copy-count", "0"])
            self.assertEqual(args.copy_count, [0])
            if DTYPE == "c8":
                with self.assertRaises(ValueError):
                    parse_args(["--copy-cap", "2048"])

    def test_shell_and_legacy_entrypoints_work_outside_repo_without_cann(self):
        scripts = [ROOT / "tests/run_scatter_copy_multiprocess.sh"]
        legacy = ROOT / "tests/run_kvcache_scatter_copy_multiprocess.sh"
        if legacy.exists():
            scripts.append(legacy)
        with tempfile.TemporaryDirectory() as directory:
            for script in scripts:
                env = {"PATH": os.environ["PATH"], "PYTHON_BIN": sys.executable,
                       "ASCEND_HOME_PATH": "/does/not/exist", "TEST_VISIBLE_DEVICES": "4,9",
                       "CARD_COUNTS": "1 2", "BATCH_SIZES": "8 16", "COPY_COUNTS": "0 300"}
                result = subprocess.run(["bash", str(script), "--dry-run", "--results-dir", directory], env=env, cwd=directory, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("cases=8", result.stdout)
                self.assertEqual(len(list(Path(directory).iterdir())), 0)

    def test_failed_rerun_invalidates_previous_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = parse_args(["--visible-devices", "0", "--cards", "1", "--batch-size", "3", "--copy-count", "7", "--results-dir", directory])
            output = next(cases(args))[1]
            output.write_text(json.dumps(fixture(latencies=(10,))))
            with patch("run_scatter_copy_sweep.subprocess.run", return_value=subprocess.CompletedProcess([], 1)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    run(args)
            self.assertEqual(json.loads(output.read_text())["status"], "failed")
            self.assertEqual(json.loads((Path(directory) / "scatter_copy_run_manifest.json").read_text())["status"], "failed")

    def test_successful_sweep_only_reports_selected_cases(self):
        def simulate(command, **kwargs):
            values = dict(zip(command[2::2], command[3::2]))
            # --same-seed has no value, so read --output by index instead.
            output = Path(command[command.index("--output") + 1])
            cards = int(values["--cards"])
            data = fixture(latencies=(10, 30)[:cards], batch=int(values["--batch-size"]), copies=int(values["--copy-count"]))
            output.write_text(json.dumps(data))
            self.assertEqual(kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4,9")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = parse_args(["--visible-devices", "4,9", "--cards", "1", "2", "--batch-size", "3", "--copy-count", "0", "7", "--results-dir", directory])
            (Path(directory) / "cards_stale.json").write_text('{"status": "failed"}')
            with patch("run_scatter_copy_sweep.subprocess.run", side_effect=simulate), contextlib.redirect_stdout(io.StringIO()):
                run(args)
            self.assertEqual(len(load_cases(Path(directory))[1]), 4)
            with (Path(directory) / "scatter_copy_timing_summary.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), COLUMNS)
                self.assertEqual(len(list(reader)), 4)

    def test_python_launcher_default_output_and_worker_cli_without_torch(self):
        launcher_name = "test_scatter_copy_multiprocess" if MULTIPROCESS_TEST == "kvcache_scatter_copy_multiprocess" else "test_kvcache_scatter_copy_multiprocess"
        launcher = importlib.import_module(launcher_name)
        with patch.object(sys, "argv", ["test", "--cards", "2", "--batch-size", "17", "--copy-count", "321"]):
            args = launcher.parse_args()
        self.assertEqual(args.output.name, "cards2_bs17_copy321.json")
        worker_name = "test_scatter_copy" if MULTIPROCESS_TEST == "kvcache_scatter_copy_multiprocess" else "test_kvcache_scatter_copy"
        worker = importlib.import_module(worker_name)
        with patch.object(sys, "argv", ["test", "--copy-count", "200"]):
            args = worker.parse_args()
        self.assertEqual((args.copy_min, args.copy_max), (200, 200))
        self.assertEqual((args.batch_size, args.source_len, args.hbm_slots, args.warmup, args.iters), (24, 65536, 8192, 10, 1000))

    def test_real_launcher_aggregates_simulated_worker_results(self):
        module = importlib.import_module("test_scatter_copy_multiprocess" if MULTIPROCESS_TEST == "kvcache_scatter_copy_multiprocess" else "test_kvcache_scatter_copy_multiprocess")
        commands = []

        class FakeProcess:
            returncode = 0

            def __init__(self, command, **kwargs):
                commands.append(command)
                values = dict(zip(command[2::2], command[3::2]))
                rank = int(values["--device"].split(":")[1])
                data = fixture()["per_device"][rank]
                for key in ("batch-size", "source-len", "hbm-slots", "copy-min", "copy-max", "copy-cap", "warmup", "iters", "seed"):
                    data["config"][key.replace("-", "_")] = int(values[f"--{key}"])
                for name in ("--ready-file", "--timing-ready-file"):
                    Path(values[name]).write_text("ready\n")
                Path(values["--json-output"]).write_text(json.dumps(data))

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case.json"
            argv = ["test", "--cards", "2", "--batch-size", "3", "--copy-count", "7", "--warmup", "0", "--iters", "5", "--same-seed", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "4,9"}), patch.object(module.subprocess, "Popen", FakeProcess), contextlib.redirect_stdout(io.StringIO()) as log:
                module.main()
            result = json.loads(output.read_text())
            self.assertEqual(result["devices"], [0, 1])
            self.assertEqual(result["summary"]["avg_us_mean"], 20)
            self.assertEqual(result["summary"]["avg_us_max"], 30)
            self.assertAlmostEqual(result["summary"]["avg_bandwidth"], BYTES_PER_TOKEN * 0.0028)
            self.assertNotIn("payload_gbps", json.dumps(result))
            self.assertNotIn("avg_us_min", json.dumps(result))
            self.assertEqual(tuple(normalize_case(result)), COLUMNS)
            self.assertIn("avg_bandwidth", log.getvalue())
            self.assertEqual(len(commands), 2)


if __name__ == "__main__":
    unittest.main()
