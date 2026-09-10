"""CPU checks for the CLI/report, process failures, generated ABI and kernel logic."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from _common import ROOT, Workload, cases, parse_args, print_table, summarize
from check_build import check_abi, check_definition
from run_scatter_copy_multiprocess import main, run_workers

spec = importlib.util.spec_from_file_location("scatter_soc", ROOT / "torch_extension/kvcache_ops/_soc.py")
soc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soc)


@contextlib.contextmanager
def working_directory(directory):
    original = Path.cwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original)


def successful_worker(config, device, barrier, connection):
    barrier.wait()
    connection.send({"result": {"avg_us": 10.0 * (device + 1), "payload_bytes": 1152}})
    connection.close()


def failed_worker(config, device, barrier, connection):
    if device == 0:
        connection.send({"error": "intentional failure"})
        connection.close()
    else:
        barrier.wait()


def exited_worker(config, device, barrier, connection):
    connection.close()


def stalled_worker(config, device, barrier, connection):
    time.sleep(20)


class CliTests(unittest.TestCase):
    def test_defaults_ignore_legacy_environment(self):
        with patch.dict(os.environ, {"TEST_VISIBLE_DEVICES": "garbage", "CARD_COUNTS": "16",
                                     "BATCH_SIZES": "99", "COPY_COUNTS": "-1"}):
            args = parse_args([], multi=True)
        self.assertEqual((args.cards, args.batch_size, args.copy_count), ([1], [8], [300]))
        self.assertIsNone(args.visible_devices)

    def test_cartesian_product_and_order(self):
        args = parse_args(["--cards", "1", "2", "--batch-size", "8", "16", "--copy-count", "0", "300"], multi=True)
        planned = list(cases(args))
        self.assertEqual(len(planned), 16)
        for left, right in zip(planned[::2], planned[1::2]):
            self.assertEqual((left[1].dtype, right[1].dtype), ("bf16", "c8"))
            self.assertEqual(left[1].batch_size, right[1].batch_size)
            self.assertEqual(left[1].copy_count, right[1].copy_count)

    def test_reject_invalid_cli(self):
        invalid = (["--cards", "0"], ["--batch-size", "0"], ["--copy-count", "-1"],
                   ["--copy-count", "16385"], ["--copy-cap", "65537"], ["--iters", "0"],
                   ["--warmup", "-1"], ["--cards", "1", "1"], ["--visible-devices", "0,0"],
                   ["--visible-devices", "0,1", "--cards", "4"], ["--timeout", "nan"],
                   ["--results-dir", "unused"], ["--json-output", "unused"])
        for args in invalid:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(args, multi=True)

    def test_exact_full_capacity_allowed(self):
        args = parse_args(["--copy-count", "65536", "--copy-cap", "65536", "--hbm-slots", "65536"], multi=True)
        self.assertEqual(args.copy_count, [65536])

    def test_single_device_and_dtype(self):
        args = parse_args(["--device", "npu:3", "--dtype", "c8", "--graph"])
        self.assertEqual(args.device, "npu:3")
        self.assertEqual(args.dtype, ["c8"])
        self.assertTrue(args.graph)

    def test_dry_run_needs_no_torch_or_files(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, working_directory(directory), contextlib.redirect_stdout(output):
            with patch.dict(sys.modules, {"_case": None, "torch": None, "torch_npu": None}):
                main(["--dry-run", "--cards", "1", "2", "--copy-count", "0", "100"])
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(output.getvalue().count("CASE "), 8)

    def test_shell_is_cli_only_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(["bash", str(ROOT / "tests/run_scatter_copy_multiprocess.sh"), "--dry-run"],
                                    cwd=directory, capture_output=True, text=True, check=True)
            self.assertIn("dtype='bf16'", result.stdout)
            self.assertIn("dtype='c8'", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


class ReportTests(unittest.TestCase):
    def test_sum_per_die_bandwidth_not_total_over_mean(self):
        row = summarize(Workload("bf16", 1, 1), [{"avg_us": 10, "payload_bytes": 1152}, {"avg_us": 30, "payload_bytes": 1152}])
        self.assertEqual(row[:6], ("bf16", 2, 1, 1, 20, 30))
        self.assertAlmostEqual(row[6], 0.1536)

    def test_c8_effective_bytes(self):
        row = summarize(Workload("c8", 2, 100), [{"avg_us": 10, "payload_bytes": 131200}])
        self.assertAlmostEqual(row[-1], 13.12)

    def test_zero_copy(self):
        self.assertEqual(summarize(Workload("bf16", 8, 0), [{"avg_us": 10, "payload_bytes": 0}])[-1], 0)

    def test_invalid_worker_measurements(self):
        for time_us, size in ((0, 1152), (-1, 1152), (float("nan"), 1152), (float("inf"), 1152), (1, 2304)):
            with self.subTest(time=time_us, size=size), self.assertRaises(ValueError):
                summarize(Workload("bf16", 1, 1), [{"avg_us": time_us, "payload_bytes": size}])
        with self.assertRaises(ValueError):
            summarize(Workload("bf16"), [])

    def test_exactly_seven_columns(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_table([("bf16", 2, 8, 300, 10.12345, 20, 30), ("c8", 2, 8, 0, 1, 2, 0)])
        rows = output.getvalue().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row.split()) == 7 for row in rows))
        self.assertEqual(rows[0].split(), ["data-type", "card-count", "batch-size", "copy-count", "avg_us_mean", "avg_us_max", "avg_bandwidth"])


class ProcessTests(unittest.TestCase):
    def test_collect_all_dies_in_device_order(self):
        result = run_workers(Workload("bf16", 1, 1), [1, 0], 10, target=successful_worker)
        self.assertEqual([r["avg_us"] for r in result], [20, 10])

    def test_failure_does_not_hang_at_barrier(self):
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            run_workers(Workload("bf16"), [0, 1], 10, target=failed_worker)

    def test_exit_without_result(self):
        with self.assertRaisesRegex(RuntimeError, "without a result"):
            run_workers(Workload("bf16"), [0], 10, target=exited_worker)

    def test_timeout_terminates_workers(self):
        with self.assertRaises(TimeoutError):
            run_workers(Workload("bf16"), [0], 0.2, target=stalled_worker)


class BuildTests(unittest.TestCase):
    def test_supported_soc_names(self):
        for name in ("Ascend910_9391", "ascend910_93", "Ascend910C"):
            self.assertEqual(soc.normalize_soc(name), "ascend910_93")
        for name in ("ascend950", "Ascend950PR_9599", "Ascend950DT"):
            self.assertEqual(soc.normalize_soc(name), "ascend950")
        with self.assertRaises(ValueError):
            soc.normalize_soc("ascend910b")

    def test_reference_abi_and_order(self):
        names = ["hbmKvRef", "dramKv", "hbmKpeRef", "dramKpe", "hbmBlockTable",
                 "dramBlockTable", "sourceTokenIds", "destinationSlots", "copyCounts"]
        signature = "aclnnKvcacheScatterCopyGetWorkspaceSize(" + ",".join("aclTensor *" + name for name in names) + ", uint64_t *size, aclOpExecutor **executor);"
        check_abi(signature)
        with self.assertRaises(RuntimeError):
            check_abi(signature.replace("uint64_t *size", "aclTensor *hbmKvOut, uint64_t *size"))
        with self.assertRaises(RuntimeError):
            check_abi(signature.replace("hbmKpeRef", "copyCounts2"))
        with self.assertRaises(RuntimeError):
            check_abi(signature.replace("hbmKpeRef", "hbmKpeRefOptional"))
        # Regression for the reported CANN header: an optional inout appears twice.
        duplicate = signature.replace("hbmKpeRef", "hbmKpeRefOptional").replace(
            "uint64_t *size", "const aclTensor *hbmKpeRefOptional, uint64_t *size")
        with self.assertRaises(RuntimeError):
            check_abi(duplicate)

    def test_reject_unsupported_reference_definitions(self):
        operators = json.loads((ROOT / "csrc/ops.json").read_text())
        check_definition(operators)
        for field, index in (("input_desc", 2), ("output_desc", 1)):
            invalid = copy.deepcopy(operators)
            invalid[0][field][index]["param_type"] = "optional"
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "must be required"):
                check_definition(invalid)
        invalid = copy.deepcopy(operators)
        invalid[0]["output_desc"][1]["name"] = "separate_output"
        with self.assertRaisesRegex(RuntimeError, "caller-owned"):
            check_definition(invalid)
        invalid = copy.deepcopy(operators)
        invalid[0]["output_desc"][1]["type"] = ["bfloat16", "bfloat16"]
        with self.assertRaisesRegex(RuntimeError, "must match"):
            check_definition(invalid)

    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the CPU kernel model")
    def test_actual_kernel_with_cpu_memory_and_queue_model(self):
        with tempfile.TemporaryDirectory() as directory:
            for a5 in (0, 1):
                Path(directory, "scatter_target.h").write_text(f"#define SCATTER_A5 {a5}\n")
                binary = str(Path(directory, f"kernel_model_{a5}"))
                compile = ["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Wno-unused-parameter",
                           "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-I", directory,
                           "-I", str(ROOT / "tests/native"), str(ROOT / "tests/native/kernel_model.cpp"), "-o", binary]
                compiled = subprocess.run(compile, capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                completed = subprocess.run([binary], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertIn("kernel model OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
