"""Read-only fallback for Host mappings unsupported by move_pages.

No memory access through the supplied VA, page faults, migration, or driver
calls. A visible, present PFN is accepted only inside System RAM and an online
sysfs memory block belonging to exactly one NUMA node. Requested policy is
never used to infer placement. Restricted/ambiguous evidence fails closed.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path


METHOD = "pagemap PFN -> System RAM -> online memory block -> NUMA node (read-only)"
PRESENT = 1 << 63
SWAPPED = 1 << 62
PFN_MASK = (1 << 55) - 1


def summarize(status: list[int | str]) -> dict:
    nodes = Counter(str(n) for n in status if isinstance(n, int))
    errors = Counter(n for n in status if isinstance(n, str))
    return {"sampled_pages": len(status), "resident_pages": sum(nodes.values()),
            "node_pages": dict(sorted(nodes.items())), "page_errors": dict(errors),
            "verified": bool(status) and not errors}


class PhysicalNodes:
    """Resolve Linux physical frames, NOT device addresses or VMA offsets."""

    def __init__(self, page_size: int, memory_root: Path, iomem_path: Path):
        self.page_size = page_size
        self.root = memory_root
        self.block_size = int((memory_root / "block_size_bytes").read_text().strip(), 16)
        if self.block_size < page_size or self.block_size % page_size:
            raise ValueError("Invalid sysfs memory block size.")
        self.ram = []
        for line in iomem_path.read_text().splitlines():
            match = re.fullmatch(r"\s*([0-9a-fA-F]+)-([0-9a-fA-F]+)\s*:\s*System RAM\s*", line)
            if match:
                lo, hi = int(match[1], 16), int(match[2], 16) + 1
                # Container kernels may redact all physical addresses to zero.
                if hi - lo >= page_size:
                    self.ram.append((lo, hi))
        if not self.ram:
            raise ValueError("No visible System RAM ranges in /proc/iomem (missing or redacted).")
        self.blocks = {}

    def node(self, pfn: int) -> int | str:
        physical = pfn * self.page_size
        if not any(lo <= physical and physical + self.page_size <= hi for lo, hi in self.ram):
            return "PFN_OUTSIDE_SYSTEM_RAM"
        block_id = physical // self.block_size
        if block_id not in self.blocks:
            root = self.root / f"memory{block_id}"
            try:
                # memoryXXX is decimal; phys_index and block_size_bytes are hex.
                if int((root / "phys_index").read_text().strip(), 16) != block_id:
                    value = "MEMORY_BLOCK_INDEX_MISMATCH"
                elif (root / "state").read_text().strip() != "online":
                    value = "MEMORY_BLOCK_NOT_ONLINE"
                else:
                    # Do not filter by the process's requested/allowed nodes:
                    # discovering allocation on an unexpected node is the point.
                    nodes = [int(p.name[4:]) for p in root.glob("node[0-9]*")
                             if re.fullmatch(r"node\d+", p.name) and p.is_dir()]
                    value = nodes[0] if len(nodes) == 1 else "MEMORY_BLOCK_NODE_AMBIGUOUS"
            except (OSError, ValueError) as error:
                value = f"MEMORY_BLOCK_UNAVAILABLE:{type(error).__name__}"
            self.blocks[block_id] = value
        return self.blocks[block_id]


def query_pages(addresses: list[int], page_size: int,
                expected_nodes: list[int] | None = None, *,
                pagemap_path: Path = Path("/proc/self/pagemap"),
                memory_root: Path = Path("/sys/devices/system/memory"),
                iomem_path: Path = Path("/proc/iomem")) -> dict:
    """Query live sampled Host pages. Internal path overrides are for CPU UTs.

    `expected_nodes` optionally cross-checks pages move_pages could resolve.
    PFNs are read as metadata only; they are never mapped/dereferenced. Zero
    PFNs are not treated as node 0: Linux can hide PFNs without CAP_SYS_ADMIN.
    """
    if page_size <= 0 or any(addr <= 0 or addr % page_size for addr in addresses):
        raise ValueError("pagemap requires positive, page-aligned Host addresses.")
    if expected_nodes is not None and len(expected_nodes) != len(addresses):
        raise ValueError("move_pages and pagemap samples must match.")
    report = {"method": METHOD, "read_only": True}
    observations = Counter()
    status = []
    resolver = None
    resolver_error = None
    entries = {}
    try:
        # Kernel pagemap entries use native endianness, 64 bits per base page.
        # This opens a proc metadata file, NOT /proc/self/mem or /dev/mem.
        fd = os.open(pagemap_path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            for index, addr in enumerate(addresses):
                if addr not in entries:
                    entries[addr] = os.pread(fd, 8, (addr // page_size) * 8)
                data = entries[addr]
                if len(data) != 8:
                    status.append("PAGEMAP_SHORT_READ")
                    continue
                entry = int.from_bytes(data, sys.byteorder)
                if entry & SWAPPED:
                    observations["swapped"] += 1
                    status.append("PAGEMAP_SWAPPED")
                    continue
                if not entry & PRESENT:
                    observations["not_present"] += 1
                    # Linux 5.10 can skip PFNMAP VMAs entirely. This result is
                    # NOT proof that a live driver buffer has no backing pages.
                    status.append("PAGEMAP_NOT_PRESENT_OR_HIDDEN")
                    continue
                observations["present"] += 1
                pfn = entry & PFN_MASK
                if not pfn:
                    observations["zero_or_redacted_pfn"] += 1
                    status.append("PFN_ZERO_OR_REDACTED")
                    continue
                observations["visible_pfn"] += 1
                if resolver is None and resolver_error is None:
                    try:
                        resolver = PhysicalNodes(page_size, memory_root, iomem_path)
                        report["memory_block_size_bytes"] = resolver.block_size
                    except (OSError, ValueError) as error:
                        resolver_error = str(error)
                        report["error"] = resolver_error
                node = resolver.node(pfn) if resolver is not None else "PHYSICAL_TO_NODE_UNAVAILABLE"
                if (isinstance(node, int) and expected_nodes is not None and
                        expected_nodes[index] >= 0 and node != expected_nodes[index]):
                    node = "RESIDENCY_CHANGED_OR_INCONSISTENT"
                status.append(node)
        finally:
            os.close(fd)
    except OSError as error:
        report["error"] = str(error)
        status.extend([f"PAGEMAP_READ_ERROR:{error.errno}"] * (len(addresses) - len(status)))
    report.update(summarize(status), pagemap_observations=dict(observations))
    if not report["verified"]:
        report["note"] = ("Only resolved Host RAM PFNs count as node evidence. Missing/hidden PFNs, "
                          "inaccessible topology, non-RAM PFNs or ambiguous nodes remain unverified. "
                          "Linux 5.10 may hide PFNMAP VMAs even from root; "
                          "do not infer placement from NUMA policy or grant privileges automatically.")
    return report
