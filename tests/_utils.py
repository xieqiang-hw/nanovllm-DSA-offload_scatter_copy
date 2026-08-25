"""Small shared helpers for standalone Ascend 950 operator tests."""

from __future__ import annotations

import torch
import torch_npu  # type: ignore


def require_a5(device: torch.device, allow_non_a5: bool) -> str:
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower() and not allow_non_a5:
        raise RuntimeError(
            f"expected Ascend 950, got {name!r}; "
            "use --allow-non-a5 only for debugging"
        )
    return name


def swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create real swapped-memory storage and initialize it from a CPU tensor."""

    allocator = getattr(torch_npu, "empty_with_swapped_memory", None)
    if allocator is None:
        raise RuntimeError(
            "torch_npu.empty_with_swapped_memory is unavailable; "
            "the test refuses to replace DRAM with an HBM tensor"
        )
    tensor = allocator(cpu.shape, dtype=cpu.dtype, device=device)
    tensor.fill_(0)
    staging = cpu.to(device)
    tensor.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return tensor
