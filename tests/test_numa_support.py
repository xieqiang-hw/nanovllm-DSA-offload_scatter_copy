#!/usr/bin/env python3
"""CPU-only regression tests for the NUMA experiment (no CANN/NPU required)."""

import ctypes
import errno
import os
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
import numa_pagemap as pagemap
import benchmark_scatter_copy_numa as bench
import test_scatter_copy as worker
import test_scatter_copy_multiprocess as multi


def fake_topology():
    return {"nodes": {n: {"cpus": list(range(n * 80, (n + 1) * 80))} for n in range(8)}}


def registration_line(svm=0xDFFF91C00000, host=0x12C180000000, pid=None):
    pid = os.getpid() if pid is None else pid
    return (f'[WARNING] APP({pid},python3):2026-09-05-23:04:51.750.583 '
            '[log_inner.cpp:81]1841 build/CMakeFiles/torch_npu.dir/compiler_depend.ts:'
            f'registerSvmMem:70: "[PTA]:"The svmPtr({hex(svm)}) is not equel to '
            f'alignedPtr({hex(host)}), then the memory pointed by svmPtr can not be printed '
            'directly on host ""\n')


def fake_maps(start=4096, end=8192):
    return {"overlapping_maps": [f"{start:x}-{end:x} rw-p 00000000 00:00 0"],
            "overlapping_numa_maps": []}


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
        with patch.object(numa, "mapping_lines", return_value=fake_maps()), patch.object(numa, "NumaAPI") as api, \
                patch.object(numa, "query_pagemap_pages", return_value=pagemap.summarize(["PFN_ZERO_OR_REDACTED"])):
            api.return_value.query_pages.side_effect = OSError(errno.EPERM, "query blocked")
            result = numa.inspect_buffer(tensor, 16, [0], 1024)
        self.assertFalse(result["verified"])
        self.assertIn("query blocked", result["allocation"]["move_pages"]["error"])
        self.assertIn("PFN_ZERO_OR_REDACTED", result["allocation"]["page_errors"])


class HostMappingTests(unittest.TestCase):
    def mapping(self, svm=0xDFFF91C00000, host=0x12C180000000):
        mappings = {}
        numa.record_registration(mappings, registration_line(svm, host), os.getpid())
        return mappings[svm]

    def test_exact_warning_format_and_pid_filter(self):
        mappings = {}
        numa.record_registration(mappings, registration_line(pid=os.getpid() + 1), os.getpid())
        numa.record_registration(mappings, registration_line().replace("registerSvmMem", "other"), os.getpid())
        self.assertEqual(mappings, {})
        line = registration_line()
        numa.record_registration(mappings, line, os.getpid())
        numa.record_registration(mappings, line, os.getpid())
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0xDFFF91C00000]["host_ptr"], "0x12c180000000")
        self.assertEqual(mappings[0xDFFF91C00000]["log_line"], line.rstrip())

    def test_conflicting_mapping_never_becomes_valid_again(self):
        mappings = {}
        for host in (0x10000, 0x20000, 0x10000):
            numa.record_registration(mappings, registration_line(host=host), os.getpid())
            if host == 0x20000:
                self.assertIsNone(mappings[0xDFFF91C00000]["host_ptr"])
        self.assertIn("error", mappings[0xDFFF91C00000])
        self.assertIsNone(mappings[0xDFFF91C00000]["host_ptr"])

    def test_logging_configuration_before_torch_import(self):
        with patch.dict(os.environ, {"ASCEND_GLOBAL_LOG_LEVEL": "4", "ASCEND_SLOG_PRINT_TO_STDOUT": "0"}):
            report = numa.prepare_registration_logging()
            self.assertEqual(report["previous"]["ASCEND_GLOBAL_LOG_LEVEL"], "4")
            self.assertEqual(os.environ["ASCEND_GLOBAL_LOG_LEVEL"], "2")
            self.assertEqual(os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"], "1")
        with patch.dict(sys.modules, {"torch_npu": Mock()}):
            with self.assertRaisesRegex(RuntimeError, "before importing"):
                numa.prepare_registration_logging()

    @unittest.skipUnless(sys.platform.startswith("linux"), "native fd capture requires Linux")
    def test_native_stdout_and_stderr_are_captured_and_replayed(self):
        output = io.StringIO()
        before = [os.fstat(fd) for fd in (1, 2)]
        with contextlib.redirect_stdout(output):
            with numa.capture_registration_logs() as mappings:
                os.write(1, registration_line().encode())
                os.write(2, registration_line(svm=0xDFFFC1E00000, host=0x12C001600000).encode())
                os.write(2, b"unrelated runtime warning is preserved\n")
        self.assertEqual(len(mappings), 2)
        self.assertIn("unrelated runtime warning", output.getvalue())
        self.assertEqual(output.getvalue().count("[WARNING]"), 2)
        self.assertEqual([(s.st_dev, s.st_ino) for s in before],
                         [(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2)])

    @unittest.skipUnless(sys.platform.startswith("linux"), "native fd capture requires Linux")
    def test_allocation_exception_restores_output_and_keeps_warning(self):
        output = io.StringIO()
        before = [(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2)]
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                with numa.capture_registration_logs() as mappings:
                    os.write(2, registration_line().encode())
                    raise RuntimeError("allocation failed")
        self.assertEqual(len(mappings), 1)
        self.assertIn("alignedPtr", output.getvalue())
        self.assertEqual(before, [(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2)])

    def test_queries_host_pages_and_active_token_offsets_not_svm(self):
        svm, host = 0xDFFF91C00000, 0x12C180000000
        tensor = SimpleNamespace(data_ptr=lambda: svm, numel=lambda: 4096, element_size=lambda: 2)
        def maps(ptr, nbytes):
            self.assertEqual(nbytes, 8192)
            return fake_maps(host, host + 8192) if ptr == host else {
                "overlapping_maps": [], "overlapping_numa_maps": []}
        with patch.object(numa, "mapping_lines", side_effect=maps), patch.object(numa, "NumaAPI") as api:
            api.return_value.query_pages.side_effect = lambda addresses: [4] * len(addresses)
            result = numa.inspect_buffer(tensor, 1024, [1, 7], 1024, self.mapping())
        self.assertTrue(result["verified"])
        self.assertEqual(result["tensor_ptr"], hex(svm))
        self.assertEqual(result["host_ptr"], hex(host))
        self.assertEqual(result["allocation"]["node_pages"], {"4": 2})
        calls = api.return_value.query_pages.call_args_list
        self.assertEqual(calls[0].args[0], [host, host + 4096])
        self.assertEqual(set(calls[1].args[0]), {host, host + 4096})
        self.assertEqual(numa.placement_result({"a": result}, "bind", [0]), "policy_mismatch")

    def test_host_mapping_does_not_hide_driver_page_query_errors(self):
        svm, host = 0xDFFF91C00000, 0x12C180000000
        tensor = SimpleNamespace(data_ptr=lambda: svm, numel=lambda: 2048, element_size=lambda: 2)
        maps = fake_maps(host, host + 4096) | {"smaps_vmflags": [{"flags": ["io", "pf"]}]}
        with patch.object(numa, "mapping_lines", return_value=maps), \
                patch.object(numa, "NumaAPI") as api, \
                patch.object(numa, "query_pagemap_pages", return_value=pagemap.summarize(["PAGEMAP_NOT_PRESENT_OR_HIDDEN"])):
            api.return_value.query_pages.return_value = [-errno.EFAULT]
            result = numa.inspect_buffer(tensor, 16, [0], 1024, self.mapping())
        self.assertFalse(result["verified"])
        self.assertEqual(result["host_ptr"], hex(host))
        self.assertEqual(result["allocation"]["move_pages"]["page_errors"], {"EFAULT": 1})
        self.assertIn("Host address resolved", result["note"])
        self.assertIn("pfnmap_hidden_from_page_query", result["diagnosis"])

    def test_missing_mapping_and_unmapped_svm_is_not_queried(self):
        tensor = SimpleNamespace(data_ptr=lambda: 0xDFFF91C00000, numel=lambda: 2048, element_size=lambda: 2)
        with patch.object(numa, "mapping_lines", return_value={"overlapping_maps": [], "overlapping_numa_maps": []}), \
                patch.object(numa, "NumaAPI") as api:
            result = numa.inspect_buffer(tensor, 16, [0], 1024)
            api.assert_not_called()
        self.assertFalse(result["verified"])
        self.assertIn("registration mapping", result["error"])

    def test_stale_or_conflicting_mapping_is_rejected(self):
        svm, host = 0xDFFF91C00000, 0x12C180000000
        tensor = SimpleNamespace(data_ptr=lambda: svm, numel=lambda: 2048, element_size=lambda: 2)
        changes = ({"pid": os.getpid() + 1}, {"tensor_ptr": hex(svm + 4096)},
                   {"host_ptr": None, "error": "conflict"})
        for change in changes:
            with patch.object(numa, "mapping_lines", return_value=fake_maps(host, host + 4096)), \
                    patch.object(numa, "NumaAPI") as api:
                result = numa.inspect_buffer(tensor, 16, [0], 1024, self.mapping() | change)
                api.assert_not_called()
            self.assertFalse(result["verified"])
            self.assertIn("stale", result["error"])

    def test_same_va_runtime_remains_supported(self):
        tensor = SimpleNamespace(data_ptr=lambda: 4096, numel=lambda: 2048, element_size=lambda: 2)
        with patch.object(numa, "mapping_lines", return_value=fake_maps()), patch.object(numa, "NumaAPI") as api:
            api.return_value.query_pages.return_value = [0]
            result = numa.inspect_buffer(tensor, 16, [], 1024)
        self.assertTrue(result["verified"])
        self.assertEqual(result["host_ptr"], "0x1000")
        self.assertIsNone(result["active_tokens"])

    def test_host_vma_coverage_requires_entire_allocation(self):
        self.assertFalse(numa.maps_cover_buffer([], 4096, 4096))
        self.assertFalse(numa.maps_cover_buffer(fake_maps()["overlapping_maps"], 4096, 8192))
        maps = fake_maps(4096, 8192)["overlapping_maps"] + fake_maps(12288, 16384)["overlapping_maps"]
        self.assertFalse(numa.maps_cover_buffer(maps, 4096, 12288))
        maps += fake_maps(8192, 12288)["overlapping_maps"]
        self.assertTrue(numa.maps_cover_buffer(maps, 4096, 12288))

    def test_invalid_active_row_never_queries_an_unrelated_page(self):
        tensor = SimpleNamespace(data_ptr=lambda: 4096, numel=lambda: 2048, element_size=lambda: 2)
        for row in (-1, 4):
            with patch.object(numa, "NumaAPI") as api:
                result = numa.inspect_buffer(tensor, 16, [row], 1024)
                api.assert_not_called()
            self.assertFalse(result["verified"])
            self.assertIn("outside", result["error"])


class PagemapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = self.root / "memory"
        self.memory.mkdir()
        (self.memory / "block_size_bytes").write_text("10000\n")
        self.iomem = self.root / "iomem"
        self.iomem.write_text("00000000-000bffff : System RAM\n")
        self.pagemap = self.root / "pagemap"
        self.pagemap.touch()
        for block, node in ((0, 0), (1, 4), (10, 7)):
            root = self.memory / f"memory{block}"
            root.mkdir()
            (root / "phys_index").write_text(f"{block:08x}\n")
            (root / "state").write_text("online\n")
            (root / f"node{node}").mkdir()

    def entries(self, *entries):
        # Virtual page 0 is unused; all reads must seek by VA/page_size * 8.
        self.pagemap.write_bytes(bytes(8) + b"".join(e.to_bytes(8, sys.byteorder) for e in entries))

    def query(self, count, **kwargs):
        page_size = kwargs.pop("page_size", 4096)
        addresses = kwargs.pop("addresses", [page_size * (i + 1) for i in range(count)])
        return pagemap.query_pages(addresses, page_size, pagemap_path=self.pagemap,
                                   memory_root=self.memory, iomem_path=self.iomem, **kwargs)

    def test_visible_pfns_resolve_nodes_including_hex_block_index(self):
        self.entries(pagemap.PRESENT | 1, pagemap.PRESENT | 16, pagemap.PRESENT | 160)
        result = self.query(3)
        self.assertTrue(result["verified"])
        self.assertEqual(result["node_pages"], {"0": 1, "4": 1, "7": 1})
        self.assertEqual(result["pagemap_observations"], {"present": 3, "visible_pfn": 3})
        buffer = {"verified": True, "allocation": result, "active_tokens": None}
        self.assertEqual(numa.placement_result({"a": buffer}, "bind", [0]), "policy_mismatch")

    def test_hidden_zero_pfns_are_not_node_zero(self):
        self.entries(pagemap.PRESENT, 0, pagemap.SWAPPED | 16)
        with patch.object(pagemap, "PhysicalNodes") as resolver:
            result = self.query(3)
            resolver.assert_not_called()
        self.assertFalse(result["verified"])
        self.assertEqual(result["node_pages"], {})
        self.assertEqual(result["page_errors"], {"PFN_ZERO_OR_REDACTED": 1,
                         "PAGEMAP_NOT_PRESENT_OR_HIDDEN": 1, "PAGEMAP_SWAPPED": 1})

    def test_physical_addresses_must_be_host_system_ram(self):
        self.entries(pagemap.PRESENT | 1, pagemap.PRESENT | 16)
        self.iomem.write_text("00000000-00000fff : System RAM\n00001000-0001ffff : device MMIO\n")
        self.assertEqual(self.query(2)["page_errors"], {"PFN_OUTSIDE_SYSTEM_RAM": 2})

    def test_redacted_or_missing_ram_ranges_fail_closed(self):
        self.entries(pagemap.PRESENT | 1)
        for text in ("", "00000000-00000000 : System RAM\n"):
            self.iomem.write_text(text)
            result = self.query(1)
            self.assertFalse(result["verified"])
            self.assertIn("redacted", result["error"])
        self.iomem.unlink()
        self.assertFalse(self.query(1)["verified"])

    def test_offline_or_ambiguous_memory_blocks_are_not_accepted(self):
        self.entries(pagemap.PRESENT | 16)
        root = self.memory / "memory1"
        (root / "state").write_text("offline\n")
        self.assertEqual(self.query(1)["page_errors"], {"MEMORY_BLOCK_NOT_ONLINE": 1})
        (root / "state").write_text("online\n")
        (root / "node5").mkdir()
        self.assertEqual(self.query(1)["page_errors"], {"MEMORY_BLOCK_NODE_AMBIGUOUS": 1})
        (root / "phys_index").write_text("00000002\n")
        self.assertEqual(self.query(1)["page_errors"], {"MEMORY_BLOCK_INDEX_MISMATCH": 1})

    def test_missing_block_and_node_are_not_guessed_from_policy(self):
        self.entries(pagemap.PRESENT | 32, pagemap.PRESENT | 16)
        (self.memory / "memory1" / "node4").rmdir()
        result = self.query(2)
        self.assertEqual(result["resident_pages"], 0)
        self.assertEqual(result["sampled_pages"], 2)
        self.assertIn("MEMORY_BLOCK_NODE_AMBIGUOUS", result["page_errors"])
        self.assertTrue(any(k.startswith("MEMORY_BLOCK_UNAVAILABLE") for k in result["page_errors"]))

    def test_base_page_size_is_not_hardcoded_4k(self):
        (self.memory / "block_size_bytes").write_text("100000\n")
        self.iomem.write_text("00000000-001fffff : System RAM\n")
        self.entries(pagemap.PRESENT | 1, pagemap.PRESENT | 16)
        result = self.query(2, page_size=65536)
        self.assertTrue(result["verified"])
        self.assertEqual(result["node_pages"], {"0": 1, "4": 1})

    def test_duplicate_active_pages_retain_token_weighting_and_reads_are_read_only(self):
        self.entries(pagemap.PRESENT | 16)
        original_open = os.open
        def read_only(path, flags, *args, **kwargs):
            self.assertEqual(flags & os.O_ACCMODE, os.O_RDONLY)
            self.assertEqual(Path(path), self.pagemap)
            return original_open(path, flags, *args, **kwargs)
        with patch.object(pagemap.os, "open", side_effect=read_only), \
                patch.object(pagemap.os, "pread", wraps=os.pread) as pread:
            result = self.query(3, addresses=[4096] * 3)
            self.assertEqual(pread.call_count, 1)
            self.assertEqual(pread.call_args.args[1:], (8, 8))
        self.assertEqual(result["node_pages"], {"4": 3})
        self.assertEqual(result["sampled_pages"], 3)

    def test_short_read_and_permissions_preserve_partial_evidence(self):
        self.entries(pagemap.PRESENT | 1)
        result = self.query(2)
        self.assertFalse(result["verified"])
        self.assertEqual(result["node_pages"], {"0": 1})
        self.assertEqual(result["page_errors"], {"PAGEMAP_SHORT_READ": 1})
        with patch.object(pagemap.os, "open", side_effect=PermissionError(errno.EACCES, "denied")):
            result = self.query(2)
        self.assertFalse(result["verified"])
        self.assertEqual(result["sampled_pages"], 2)
        self.assertIn("denied", result["error"])

    def test_cross_check_disagreement_fails_closed(self):
        self.entries(pagemap.PRESENT | 1, pagemap.PRESENT | 16)
        result = self.query(2, expected_nodes=[0, 0])
        self.assertFalse(result["verified"])
        self.assertEqual(result["page_errors"], {"RESIDENCY_CHANGED_OR_INCONSISTENT": 1})
        self.assertTrue(self.query(2, expected_nodes=[0, -errno.EFAULT])["verified"])

    def test_invalid_sample_inputs(self):
        for addresses in ([0], [-4096], [4097]):
            with self.assertRaises(ValueError):
                self.query(1, addresses=addresses)
        with self.assertRaises(ValueError):
            self.query(1, expected_nodes=[])
        self.assertFalse(self.query(0)["verified"])

    def test_fallback_only_when_primary_cannot_resolve(self):
        api = Mock()
        api.query_pages.return_value = [0, 4]
        with patch.object(numa, "query_pagemap_pages") as fallback:
            result = numa.query_residency(api, [4096, 8192], 4096)
            fallback.assert_not_called()
        self.assertTrue(result["verified"])
        api.query_pages.return_value = [0, -errno.EFAULT]
        with patch.object(numa, "query_pagemap_pages", return_value=pagemap.summarize([0, 4])) as fallback:
            result = numa.query_residency(api, [4096, 8192], 4096)
            fallback.assert_called_once_with([4096, 8192], 4096, [0, -errno.EFAULT])
        self.assertTrue(result["verified"])
        self.assertEqual(result["move_pages"]["page_errors"], {"EFAULT": 1})

    def test_smaps_vmflags_are_diagnostic_not_node_evidence(self):
        path = self.root / "smaps"
        path.write_text("1000-2000 rw-p 00000000 00:00 0\nSize: 4 kB\nVmFlags: rd wr\n"
                        "100000000000-180000000000 rw-s 00000000 00:8c 103 /dev/davinci_manager\n"
                        "Size: 8589934592 kB\nVmFlags: rd wr sh io pf\n"
                        "dfff00000000-e00000000000 rw-p 00000000 00:00 0\nVmFlags: rd wr\n")
        result = numa.smaps_vmflags({0x100000000000}, path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["flags"], ["rd", "wr", "sh", "io", "pf"])
        self.assertNotIn("node_pages", result[0])


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

    def test_trial_failure_prints_buffer_diagnostics_and_worker_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workers").mkdir()
            (root / "launcher.log").write_text("RuntimeError: inspect worker log\n")
            (root / "workers" / "rank0_device0.log").write_text(
                'A3_SCATTER_NUMA_BUFFER {"verified": false, "page_errors": {"PFN_ZERO_OR_REDACTED": 1}}\n'
                'A3_SCATTER_NUMA_PLACEMENT ' + 'x' * 10000 + '\n'
                'RuntimeError: Swapped source placement is unverified\n')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                bench.print_failure_logs(root)
        self.assertIn("PFN_ZERO_OR_REDACTED", output.getvalue())
        self.assertIn("Swapped source placement is unverified", output.getvalue())
        self.assertLess(len(output.getvalue()), 4000)


if __name__ == "__main__":
    unittest.main()
