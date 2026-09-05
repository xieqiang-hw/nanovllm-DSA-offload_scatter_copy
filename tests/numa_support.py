"""Linux NUMA experiment helpers. No Torch import and no page migration.

Policies must be installed before importing Torch/CANN so subsequently created
runtime threads inherit them. move_pages is used with nodes=NULL (query only);
never dereference an NPU/SVM address on the CPU or move registered DMA pages.
"""

from __future__ import annotations

import ctypes
import errno
import os
import random
import sys
from collections import Counter
from pathlib import Path


POLICIES = ("inherit", "default", "concentrated", "balanced", "interleave", "mapped")
MODES = {"default": 0, "bind": 2, "interleave": 3}


def parse_ids(value: str) -> list[int]:
    """Parse Linux cpulist syntax, preserving order and rejecting duplicates."""
    result = []
    for part in value.strip().split(","):
        if not part:
            raise ValueError(f"Empty element in ID list: {value!r}")
        ends = part.strip().split("-")
        if len(ends) == 1:
            start = end = int(ends[0])
        elif len(ends) == 2:
            start, end = map(int, ends)
        else:
            raise ValueError(f"Invalid ID range: {part!r}")
        if start < 0 or end < start:
            raise ValueError(f"Invalid ID range: {part!r}")
        result.extend(range(start, end + 1))
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate ID in {value!r}")
    return result


def process_status() -> dict:
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("Cpus_allowed_list", "Mems_allowed_list"):
            fields[key] = parse_ids(value.strip())
    return fields


def topology() -> dict:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("NUMA experiments require Linux.")
    status = process_status()
    allowed_cpus = set(os.sched_getaffinity(0))
    nodes = {}
    for node in status["Mems_allowed_list"]:
        root = Path(f"/sys/devices/system/node/node{node}")
        if not root.is_dir():
            continue
        meminfo = root.joinpath("meminfo").read_text()
        total = next(int(line.split()[-2]) for line in meminfo.splitlines()
                     if "MemTotal:" in line)
        if not total:
            continue
        cpulist = root.joinpath("cpulist").read_text().strip()
        nodes[node] = {
            "cpus": sorted(allowed_cpus & set(parse_ids(cpulist) if cpulist else [])),
            "meminfo": meminfo,
            "distance": root.joinpath("distance").read_text().strip(),
        }
    if not nodes:
        raise RuntimeError("No allowed NUMA nodes with memory were found.")
    return {"nodes": nodes, "allowed": status}


def worker_plans(count: int, policy: str, nodes: list[int], topo: dict,
                 cpu_bind: str, cpus_per_worker: int, concentrated_node: int,
                 node_map: list[int] | None = None) -> list[dict]:
    available = topo["nodes"]
    if policy not in POLICIES or count < 1 or not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("Invalid NUMA policy, worker count, or node list.")
    if any(node not in available for node in nodes):
        raise ValueError("Requested NUMA nodes are outside allowed memory nodes.")
    if policy == "concentrated" and concentrated_node not in nodes:
        raise ValueError("--numa-node must belong to --numa-nodes.")
    if policy == "mapped":
        if node_map is None or len(node_map) != count or any(n not in nodes for n in node_map):
            raise ValueError("--numa-node-map needs one allowed node per worker (duplicates allowed).")
    elif node_map is not None:
        raise ValueError("--numa-node-map requires --numa-policy=mapped.")
    if cpu_bind not in ("none", "spread") or cpus_per_worker < 1:
        raise ValueError("Invalid CPU binding options.")
    # Independent of memory policy, including concentrated/mapped: never change
    # CPU placement at the same time as the allocation policy being compared.
    cpu_nodes = sorted(node for node, info in available.items() if info["cpus"])
    if cpu_bind == "spread" and not cpu_nodes:
        raise ValueError("No allowed CPUs on the available NUMA nodes.")
    plans = []
    for rank in range(count):
        cpus = []
        if cpu_bind == "spread":
            cpu_node = cpu_nodes[rank % len(cpu_nodes)]
            offset = (rank // len(cpu_nodes)) * cpus_per_worker
            cpus = available[cpu_node]["cpus"][offset:offset + cpus_per_worker]
            if len(cpus) != cpus_per_worker:
                raise ValueError("Not enough disjoint allowed CPUs; reduce --cpus-per-worker.")
        memory_nodes = []
        mode = policy
        if policy in ("concentrated", "balanced", "mapped"):
            mode = "bind"
            memory_nodes = [concentrated_node if policy == "concentrated" else
                            nodes[rank % len(nodes)] if policy == "balanced" else node_map[rank]]
        elif policy == "interleave":
            memory_nodes = list(nodes)
        plans.append({"policy": policy, "mode": mode, "nodes": memory_nodes, "cpus": cpus})
    return plans


class NumaAPI:
    def __init__(self):
        self.lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
        ulongp = ctypes.POINTER(ctypes.c_ulong)
        self.lib.set_mempolicy.argtypes = [ctypes.c_int, ulongp, ctypes.c_ulong]
        self.lib.set_mempolicy.restype = ctypes.c_long
        self.lib.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ulongp,
                                          ctypes.c_ulong, ctypes.c_void_p, ctypes.c_uint]
        self.lib.get_mempolicy.restype = ctypes.c_long
        self.lib.move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong,
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_int),
                                       ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.move_pages.restype = ctypes.c_long

    @staticmethod
    def mask(nodes: list[int], bits: int):
        word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask = (ctypes.c_ulong * ((bits + word_bits - 1) // word_bits))()
        for node in nodes:
            mask[node // word_bits] |= 1 << (node % word_bits)
        return mask

    @staticmethod
    def check(result: int, operation: str):
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, f"{operation}: {os.strerror(error)}")

    def get_policy(self) -> dict:
        # Kernel masks may be bigger than the online node set. Grow on EINVAL.
        for bits in (1024, 4096, 16384):
            mask = self.mask([], bits)
            mode = ctypes.c_int()
            result = self.lib.get_mempolicy(ctypes.byref(mode), mask, bits + 1, None, 0)
            if result == 0:
                word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
                nodes = [n for n in range(bits) if mask[n // word_bits] & (1 << (n % word_bits))]
                return {"mode": mode.value, "nodes": nodes}
            if ctypes.get_errno() != errno.EINVAL:
                break
        self.check(result, "get_mempolicy")
        raise AssertionError("unreachable")

    def set_policy(self, mode: str, nodes: list[int]):
        bits = max(nodes) + 1 if nodes else 0
        mask = self.mask(nodes, bits) if nodes else None
        # Like libnuma's bitmask wrappers, pass allocated mask bits + 1:
        # Linux 5.10 get_nodes() decrements maxnode before reading the mask.
        # Passing max(nodes)+1 directly would lose its highest set bit (and
        # binding node 0 alone would turn into an invalid empty nodemask).
        maxnode = ctypes.sizeof(mask) * 8 + 1 if mask is not None else 0
        self.check(self.lib.set_mempolicy(MODES[mode], mask, maxnode), "set_mempolicy")
        actual = self.get_policy()
        if actual != {"mode": MODES[mode], "nodes": sorted(nodes)}:
            raise RuntimeError(f"NUMA policy readback differs: requested={mode}/{nodes}, actual={actual}")

    def query_pages(self, addresses: list[int]) -> list[int]:
        pages = (ctypes.c_void_p * len(addresses))(*addresses)
        status = (ctypes.c_int * len(addresses))(*([-errno.EIO] * len(addresses)))
        # NULL nodes and flags=0: obtain residency, NEVER migrate a page.
        self.check(self.lib.move_pages(0, len(addresses), pages, None, status, 0), "move_pages(query)")
        return list(status)


def apply_worker_policy(mode: str, nodes: list[int], cpus: list[int]) -> dict:
    if cpus:
        if not set(cpus).issubset(os.sched_getaffinity(0)):
            raise ValueError("Requested CPUs are outside the process's allowed affinity.")
        os.sched_setaffinity(0, cpus)
    api = NumaAPI()
    before = api.get_policy()
    if mode != "inherit":
        if not set(nodes).issubset(process_status()["Mems_allowed_list"]):
            raise ValueError("Requested memory nodes are outside the cpuset.")
        api.set_policy(mode, nodes)
    return {"pid": os.getpid(), "requested_mode": mode, "requested_nodes": nodes,
            "requested_cpus": cpus, "policy_before": before,
            "policy_after": api.get_policy(), "allowed_after": process_status()}


def sample_addresses(ptr: int, nbytes: int, count: int, page_size: int) -> list[int]:
    if ptr <= 0 or nbytes <= 0 or count < 1:
        raise ValueError("Invalid buffer address, size, or page sample count.")
    first = ptr // page_size * page_size
    pages = (ptr + nbytes - 1 - first) // page_size + 1
    # Random, reproducible sampling avoids stride aliasing with page interleave.
    indices = sorted(random.Random(7).sample(range(pages), min(pages, count)))
    return [first + i * page_size for i in indices]


def summarize_pages(status: list[int]) -> dict:
    counts = Counter(str(n) for n in status if n >= 0)
    errors = Counter(errno.errorcode.get(-n, str(n)) for n in status if n < 0)
    return {"sampled_pages": len(status), "resident_pages": sum(counts.values()),
            "node_pages": dict(sorted(counts.items())), "page_errors": dict(errors),
            "verified": bool(status) and not errors}


def mapping_lines(ptr: int, nbytes: int) -> dict:
    """Supporting diagnostics only; VMA totals need not describe this buffer."""
    maps = []
    starts = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        lo, hi = (int(x, 16) for x in line.split()[0].split("-"))
        if lo < ptr + nbytes and hi > ptr:
            maps.append(line)
            starts.add(lo)
    numa = [line for line in Path("/proc/self/numa_maps").read_text().splitlines()
            if int(line.split()[0], 16) in starts]
    return {"overlapping_maps": maps, "overlapping_numa_maps": numa}


def inspect_buffer(tensor, sample_count: int, active_rows: list[int], row_bytes: int) -> dict:
    ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    page_size = os.sysconf("SC_PAGE_SIZE")
    report = {"tensor_ptr": hex(ptr), "nbytes": nbytes, "page_size": page_size,
              "method": "move_pages(nodes=NULL); tensor/SVM VA; no CPU dereference"}
    try:
        report.update(mapping_lines(ptr, nbytes))
        api = NumaAPI()
        report["allocation"] = summarize_pages(api.query_pages(
            sample_addresses(ptr, nbytes, sample_count, page_size)))
        # Query the pages the copy workload actually reads, not just the full
        # allocation. Duplicate pages are retained: counts are token-weighted.
        rows = random.Random(11).sample(active_rows, min(len(active_rows), sample_count))
        report["active_tokens"] = summarize_pages(api.query_pages(
            [(ptr + row * row_bytes) // page_size * page_size for row in rows])) if rows else None
        report["verified"] = report["allocation"]["verified"] and (
            report["active_tokens"] is None or report["active_tokens"]["verified"])
    except (OSError, ValueError) as error:
        report.update(verified=False, error=str(error))
    if not report["verified"]:
        report["note"] = ("Unverified: SVM VA may differ from Host VA, registered pages may not "
                          "support residency queries, or permissions may block move_pages. "
                          "Process totals/memory policy alone do NOT prove buffer placement.")
    return report


def placement_result(buffers: dict, mode: str, nodes: list[int]) -> str:
    if not buffers or not all(item["verified"] for item in buffers.values()):
        return "unverified"
    if mode in ("bind", "interleave"):
        wanted = set(map(str, nodes))
        for item in buffers.values():
            for sample in (item["allocation"], item["active_tokens"]):
                if sample is not None and not set(sample["node_pages"]).issubset(wanted):
                    return "policy_mismatch"
            if mode == "interleave" and set(item["allocation"]["node_pages"]) != wanted:
                return "interleave_not_observed"
    return "sample_verified"
