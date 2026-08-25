from __future__ import annotations

import os
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OPP = _ROOT / "_custom_opp_bf16"
_EXPLICIT_OPAPI = os.getenv("NANOVLLM_CUST_OPAPI_LIB")
if _EXPLICIT_OPAPI:
    _OPAPI = Path(_EXPLICIT_OPAPI).expanduser().resolve()
    if not _OPAPI.is_file():
        raise RuntimeError(
            f"NANOVLLM_CUST_OPAPI_LIB does not exist: {_OPAPI}"
        )
else:
    _OPAPI_LIBS = tuple(
        path
        path
        for path in (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
    )
    if len(_OPAPI_LIBS) != 1:
        raise RuntimeError(
            "Build the Scatter Copy operator, or set NANOVLLM_CUST_OPAPI_LIB."
        )
    _OPAPI = _OPAPI_LIBS[0].resolve()

_VENDOR = _OPAPI.parents[2]
_INSTALL_OPP = _OPAPI.parents[4]

_existing = [
    value
    for value in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if value
]
_ordered_vendors = []
for _vendor in [_VENDOR, *_existing]:
    _vendor_str = str(_vendor)
    if _vendor_str not in _ordered_vendors:
        _ordered_vendors.append(_vendor_str)
os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(_ordered_vendors)
os.environ.setdefault("NANOVLLM_A5_INSTALL_OPP_PATH", str(_INSTALL_OPP))
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401
from .ops import kvcache_scatter_copy  # noqa: E402


def local_opapi_path() -> str:
    return str(_OPAPI)


__all__ = [
    "kvcache_scatter_copy",
    "local_opapi_path",
]
