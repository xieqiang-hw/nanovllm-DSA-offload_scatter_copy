"""Shared hardware fixture: private DRAM pools, byte-exact oracle, event timing."""
from __future__ import annotations

import sys
from _common import ROOT, ROW_BYTES, Workload

sys.path.insert(0, str(ROOT / "torch_extension"))
import kvcache_ops  # Configure this checkout's OPP before importing torch_npu.
import torch
import torch_npu

POISON = 65


def swapped_from_cpu(cpu, device):
    allocator = getattr(torch_npu, "empty_with_swapped_memory", None)
    if allocator is None:
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required for real DRAM sources.")
    result = allocator(cpu.shape, dtype=cpu.dtype, device=device)
    # Initialize raw bytes: BF16 NaNs/signed zeros must not undergo arithmetic.
    raw = result.view(torch.int8)
    raw.zero_()
    staging = cpu.view(torch.int8).to(device)
    raw.add_(staging)
    torch.npu.synchronize()
    return result


class Case:
    def __init__(self, config: Workload, device: str, *, counts=None, edges=False):
        self.config, self.device = config, torch.device(device)
        torch.npu.set_device(self.device)
        torch.set_num_threads(1)
        torch.npu.config.allow_internal_format = False
        batch, cap = config.batch_size, config.copy_cap
        source_blocks = (config.source_len + 127) // 128
        target_blocks = (config.hbm_slots + 127) // 128
        rng = torch.Generator().manual_seed(config.seed)
        # Physical source blocks are disjoint across requests on every platform.
        self.dram_table_cpu = torch.randperm(batch * source_blocks, generator=rng).to(torch.int32).reshape(batch, source_blocks)
        self.hbm_table_cpu = torch.randperm(batch * target_blocks + 1, generator=rng)[:-1].to(torch.int32).reshape(batch, target_blocks)
        self.src_cpu = torch.full((batch, 1, cap), -1, dtype=torch.int32)
        self.dst_cpu = torch.full_like(self.src_cpu, -1)
        self.counts_cpu = torch.tensor(counts if counts is not None else [config.copy_count] * batch, dtype=torch.int32)
        for b, count in enumerate(self.counts_cpu.tolist()):
            source = torch.randperm(config.source_len, generator=rng).tolist()
            dest = torch.randperm(config.hbm_slots, generator=rng).tolist()
            if edges:
                prefix = [0, 1, 127, 128, 129, config.hbm_slots - 1]
                prefix = list(dict.fromkeys(v for v in prefix if v < config.hbm_slots))
                dest = prefix + [v for v in dest if v not in set(prefix)]
                source[:6] = [0, 128, 128, config.source_len - 1, 127, 1]
            self.src_cpu[b, 0, :count] = torch.tensor(source[:count], dtype=torch.int32)
            self.dst_cpu[b, 0, :count] = torch.tensor(dest[:count], dtype=torch.int32)
        # A separate RNG keeps metadata identical between BF16 and packed-C8.
        payload_rng = torch.Generator().manual_seed(config.seed ^ 0x5A17)
        dtype = torch.bfloat16 if config.dtype == "bf16" else torch.int8
        widths = (1024, 128) if config.dtype == "bf16" else (656,)
        self.sources_cpu, self.sources, self.targets = [], [], []
        for width in widths:
            cpu = torch.randint(-128, 128, (batch * source_blocks, 128, 1, width),
                                dtype=torch.int8, generator=payload_rng).view(dtype)
            self.sources_cpu.append(cpu)
            self.sources.append(swapped_from_cpu(cpu, self.device))
            self.targets.append(torch.empty((batch * target_blocks + 1, 128, 1, width // cpu.element_size()),
                                            dtype=dtype, device=self.device))
        self.metadata_cpu = (self.hbm_table_cpu, self.dram_table_cpu, self.src_cpu, self.dst_cpu, self.counts_cpu)
        self.metadata = tuple(t.to(self.device) for t in self.metadata_cpu)
        self.inputs = (self.targets[0], self.sources[0],
                       self.targets[1] if config.dtype == "bf16" else None,
                       self.sources[1] if config.dtype == "bf16" else None, *self.metadata)
        self.pointers = tuple(t.data_ptr() for t in self.inputs if t is not None)
        self.expected = self.reference()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def reference(self):
        # CPU gather/scatter oracle is independent of kernel core ownership/tiling.
        sources, destinations = [], []
        for b, count in enumerate(self.counts_cpu.tolist()):
            src, dst = self.src_cpu[b, 0, :count].long(), self.dst_cpu[b, 0, :count].long()
            sources.append(self.dram_table_cpu[b].long()[src // 128] * 128 + src % 128)
            destinations.append(self.hbm_table_cpu[b].long()[dst // 128] * 128 + dst % 128)
        src_rows, dst_rows = torch.cat(sources), torch.cat(destinations)
        result = []
        for cpu, target in zip(self.sources_cpu, self.targets):
            width = cpu.size(-1) * cpu.element_size()
            expected = torch.full((target.size(0) * 128, width), POISON, dtype=torch.int8)
            expected[dst_rows] = cpu.view(torch.int8).reshape(-1, width)[src_rows]
            result.append(expected)
        return result

    def reset(self):
        for target in self.targets:
            target.view(torch.int8).fill_(POISON)

    def set_counts(self, values):
        self.counts_cpu.copy_(torch.tensor(values, dtype=torch.int32))
        self.metadata[-1].copy_(self.counts_cpu)
        self.expected = self.reference()

    def verify(self):
        if self.pointers != tuple(t.data_ptr() for t in self.inputs if t is not None):
            raise AssertionError("A caller-owned tensor address changed.")
        for target, expected in zip(self.targets, self.expected):
            if not torch.equal(target.view(torch.int8).cpu().reshape_as(expected), expected):
                raise AssertionError("Cache bytes differ, or inactive/guard bytes were modified.")
        for tensor, cpu in zip((*self.sources, *self.metadata), (*self.sources_cpu, *self.metadata_cpu)):
            if not torch.equal(tensor.view(torch.int8).cpu(), cpu.view(torch.int8)):
                raise AssertionError("A read-only source or metadata tensor was modified.")

    def check(self):
        self.reset()
        result = kvcache_ops.kvcache_scatter_copy(*self.inputs)
        torch.npu.synchronize()
        if result is not None:
            raise AssertionError("Scatter must return None.")
        self.verify()

    def measure(self, barrier=None):
        if barrier is not None:
            barrier.wait()  # Every die finishes initialization/correctness before warmup.
        call, inputs = kvcache_ops.kvcache_scatter_copy, self.inputs
        for _ in range(self.config.warmup):
            call(*inputs)
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        if barrier is not None:
            barrier.wait()
        start.record()
        for _ in range(self.config.iters):
            call(*inputs)
        end.record()
        end.synchronize()
        avg_us = start.elapsed_time(end) * 1000 / self.config.iters
        if barrier is not None:
            barrier.wait()  # No D2H verification traffic while another die is timing.
        self.verify()
        return {"avg_us": avg_us, "payload_bytes": int(self.counts_cpu.sum()) * ROW_BYTES[self.config.dtype]}
