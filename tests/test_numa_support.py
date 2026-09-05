#!/usr/bin/env python3
"""CPU-only regression tests for the NUMA experiment (no CANN/NPU required)."""

import ctypes
import errno
import sys
import tempfile
import unittest
import contextlib
import io
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numa_support as numa
import benchmark_scatter_copy_numa as bench
import test_scatter_copy as worker
import test_scatter_copy_multiprocess as multi


def fake_topology():
    return {"nodes": {n: {"cpus": list(range(n * 80, (n + 1) * 80))} for n in range(8)}}


class PlanTests(unittest.TestCase):
    def plan(self, policy, **kw):
        args = dict(count=16, policy=policy, nodes=list(range(8)), topo=fake_topology(),
                    cpu_bind="spread", cpus_per_worker=4, concentrated_node=0)
        return numa.worker_plans(**(args | kw))

    def test_cpu_placement_is_independent_of_memory_policy(self):
        baseline = [p["cpus"] for p in self.plan("default")]
        for policy in ("concentrated", "balanced", "interleave"):
            self.assertEqual([p["cpus"] for p in self.plan(policy)], baseline)
        self.assertEqual(len(set(cpu for cpus in baseline for cpu in cpus)), 16 * 4)

    def test_16_die_8_node_balance(self):
        self.assertEqual(Counter(p["nodes"][0] for p in self.plan("balanced")), {n: 2 for n in range(8)})
        self.assertTrue(all(p["nodes"] == [0] for p in self.plan("concentrated")))
        self.assertTrue(all(p["nodes"] == list(range(8)) for p in self.plan("interleave")))

    def test_noncontiguous_nodes_and_explicit_mapping(self):
        self.assertEqual([p["nodes"] for p in self.plan("balanced", nodes=[2, 5], count=3)], [[2], [5], [2]])
        self.assertEqual([p["nodes"] for p in self.plan("mapped", count=3, node_map=[5, 5, 2])], [[5], [5], [2]])

    def test_bad_node_maps_and_cpu_exhaustion(self):
        for kwargs in ({"nodes": [0, 9]}, {"nodes": [0, 0]}, {"cpus_per_worker": 100}):
            with self.assertRaises(ValueError):
                self.plan("balanced", **kwargs)
        for mapping in ([0], [0] * 15 + [8], None):
            with self.assertRaises(ValueError):
                self.plan("mapped", node_map=mapping)
        with self.assertRaises(ValueError):
            self.plan("concentrated", concentrated_node=9)

    def test_cpuset_cpu_restrictions(self):
        topo = {"nodes": {2: {"cpus": [180, 184, 185, 189]}, 7: {"cpus": []}}}
        plans = self.plan("balanced", count=2, nodes=[2, 7], topo=topo, cpus_per_worker=2)
        self.assertEqual([p["cpus"] for p in plans], [[180, 184], [185, 189]])

    def test_parse_ids(self):
        self.assertEqual(numa.parse_ids("0-3,8,10-11"), [0, 1, 2, 3, 8, 10, 11])
        for bad in ("", "1,", "-1", "3-1", "1,1", "a", "0-2,2"):
            with self.assertRaises(ValueError):
                numa.parse_ids(bad)


class ResidencyTests(unittest.TestCase):
    def test_sample_bounds_small_and_unaligned(self):
        self.assertEqual(numa.sample_addresses(4097, 4, 64, 4096), [4096])
        self.assertEqual(numa.sample_addresses(8191, 4, 64, 4096), [4096, 8192])
        pages = numa.sample_addresses(4096, 10**9, 1024, 4096)
        self.assertEqual(len(pages), len(set(pages)))
        self.assertTrue(all(p % 4096 == 0 and 4096 <= p < 4096 + 10**9 for p in pages))
        self.assertEqual(set((p // 4096) % 8 for p in pages), set(range(8)))

    def test_mask_handles_multiple_words(self):
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask = numa.NumaAPI.mask([0, bits + 1, 3 * bits - 1], 3 * bits)
        self.assertEqual(list(mask), [1, 2, 1 << (bits - 1)])

    def test_move_pages_is_query_only(self):
        def query(pid, count, pages, nodes, status, flags):
            self.assertEqual(pid, 0)
            self.assertIsNone(nodes)
            self.assertEqual(flags, 0)
            self.assertEqual(list(pages), [4096, 8192])
            status[0], status[1] = 3, -errno.EFAULT
            return 0
        api = numa.NumaAPI.__new__(numa.NumaAPI)
        api.lib = SimpleNamespace(move_pages=query)
        self.assertEqual(api.query_pages([4096, 8192]), [3, -errno.EFAULT])

    def test_set_policy_error_is_not_silent(self):
        def fail(*args):
            ctypes.set_errno(errno.EPERM)
            return -1
        api = numa.NumaAPI.__new__(numa.NumaAPI)
        api.lib = SimpleNamespace(set_mempolicy=fail)
        with self.assertRaises(OSError):
            api.set_policy("bind", [0])

    def test_set_policy_keeps_highest_bit_on_linux_510(self):
        for nodes in ([0], [7], [63], [0, 7], [0, 64]):
            def set_policy(mode, mask, maxnode):
                # Model Linux 5.10 get_nodes(): --maxnode, then endmask.
                word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
                observed = [n for n in range(maxnode - 1)
                            if mask[n // word_bits] & (1 << (n % word_bits))]
                self.assertEqual(observed, list(nodes))
                return 0
            api = numa.NumaAPI.__new__(numa.NumaAPI)
            api.lib = SimpleNamespace(set_mempolicy=set_policy)
            api.get_policy = lambda: {"mode": 2, "nodes": list(nodes)}
            api.set_policy("bind", list(nodes))

    def test_unknown_residency_is_not_a_success(self):
        report = numa.summarize_pages([0, 0, -errno.EFAULT])
        self.assertFalse(report["verified"])
        self.assertEqual(report["node_pages"], {"0": 2})
        self.assertEqual(report["page_errors"], {"EFAULT": 1})
        self.assertFalse(numa.summarize_pages([])["verified"])
        self.assertEqual(numa.placement_result({"a": {"verified": False}}, "bind", [0]), "unverified")

    def test_bound_wrong_node_and_interleave(self):
        a = {"verified": True, "allocation": numa.summarize_pages([0, 1]), "active_tokens": None}
        self.assertEqual(numa.placement_result({"a": a}, "bind", [0]), "policy_mismatch")
        self.assertEqual(numa.placement_result({"a": a}, "interleave", [0, 1]), "sample_verified")
        self.assertEqual(numa.placement_result({"a": a}, "interleave", [0, 1, 2]), "interleave_not_observed")
        a["active_tokens"] = numa.summarize_pages([2])
        self.assertEqual(numa.placement_result({"a": a}, "interleave", [0, 1]), "policy_mismatch")

    def test_svm_query_failure_is_reported_without_cpu_read(self):
        tensor = SimpleNamespace(data_ptr=lambda: 4096, numel=lambda: 2048, element_size=lambda: 2)
        with patch.object(numa, "mapping_lines", return_value={}), patch.object(numa, "NumaAPI") as api:
            api.return_value.query_pages.side_effect = OSError(errno.EPERM, "query blocked")
            result = numa.inspect_buffer(tensor, 16, [0], 1024)
        self.assertFalse(result["verified"])
        self.assertIn("query blocked", result["error"])


class HarnessTests(unittest.TestCase):
    def test_imports_do_not_initialize_torch(self):
        # These test/plan modules must work even with no Torch installation.
        self.assertNotIn("torch_npu", sys.modules)
        self.assertNotIn("ops_overlap", sys.modules)

    def test_legacy_worker_cli_defaults(self):
        with patch.object(sys, "argv", ["test_scatter_copy.py"]):
            args = worker.parse_args()
        worker.validate_args(args)
        self.assertEqual(args.memory_policy, "inherit")
        self.assertEqual(args.timing, "eager")
        self.assertFalse(args.numa_probe)

    def test_full_16_worker_plan_dry_run_without_npu(self):
        argv = ["multi", "--devices", ",".join(map(str, range(16))),
                "--numa-policy", "balanced", "--numa-nodes", "0-7", "--cpu-bind", "spread", "--dry-run"]
        with patch.object(sys, "argv", argv), patch.object(multi, "topology", return_value=fake_topology()), \
                patch.object(multi.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            multi.main()
            popen.assert_not_called()

    def test_benchmark_dry_run_never_initializes_npu(self):
        argv = ["bench", "--devices", "0,1,2,3", "--nodes", "0-7", "--dry-run"]
        with patch.object(sys, "argv", argv), patch.dict("os.environ", {"ASCEND_LAUNCH_BLOCKING": "0"}), \
                patch.object(bench, "topology", return_value=fake_topology()), \
                patch.object(bench.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            bench.main()
            popen.assert_not_called()

    def test_worker_memory_policy_arguments(self):
        for argv in (["--memory-policy", "bind"], ["--memory-nodes", "0"], ["--numa-page-samples", "0"]):
            with patch.object(sys, "argv", ["test"] + argv):
                args = worker.parse_args()
            with self.assertRaises(ValueError):
                worker.validate_args(args)

    def test_separate_affinity_sweep(self):
        args = SimpleNamespace(experiment="affinity")
        self.assertEqual(bench.conditions(args, [2, 7], [1, 5]), [
            ("device2_node1", [2], "concentrated", 1), ("device2_node5", [2], "concentrated", 5),
            ("device7_node1", [7], "concentrated", 1), ("device7_node5", [7], "concentrated", 5)])

    def test_dead_worker_does_not_leave_peers_waiting(self):
        process = Mock()
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready"
            with self.assertRaises(RuntimeError):
                multi.wait_workers([(0, process, None, path)], [path])
        # Cleanup touches only subprocess handles given to this invocation.
        alive = Mock()
        alive.poll.return_value = None
        multi.stop_workers([(0, alive, None, None), (1, process, None, None)])
        alive.terminate.assert_called_once()
        process.terminate.assert_not_called()

    def test_unverified_results_have_no_fake_node_load(self):
        result = {"summary": {"all_placements_sample_verified": False}}
        self.assertIsNone(bench.placement_load(result))

    def test_fast_worker_holds_copy_load_outside_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(timing_done_file=Path(directory) / "done",
                                   timing_stop_file=Path(directory) / "stop")
            count = 0
            def launch():
                nonlocal count
                count += 1
                if count == 32:
                    args.timing_stop_file.touch()
            runtime = SimpleNamespace(npu=SimpleNamespace(synchronize=Mock()))
            with patch.object(worker, "torch", runtime, create=True):
                worker.keep_copy_load_until_all_timed(args, launch)
            self.assertTrue(args.timing_done_file.exists())
            self.assertEqual(count, 32)
            runtime.npu.synchronize.assert_called_once()

    def test_node_load_is_byte_weighted(self):
        def buffer(size, node):
            return {"nbytes": size, "allocation": numa.summarize_pages([node]),
                    "active_tokens": numa.summarize_pages([node])}
        result = {"summary": {"all_placements_sample_verified": True}, "per_device": [{
            "numa": {"buffers": {"dram_ckv": buffer(8192, 0), "dram_kpe": buffer(1024, 1)}},
            "workload": {"copied_tokens": 4}}]}
        result = bench.placement_load(result)
        self.assertEqual(result["estimated_source_bytes_per_node"], {"0": 8192, "1": 1024})
        self.assertEqual(result["estimated_read_bytes_per_node_per_iteration"], {"0": 4096, "1": 512})


if __name__ == "__main__":
    unittest.main()
