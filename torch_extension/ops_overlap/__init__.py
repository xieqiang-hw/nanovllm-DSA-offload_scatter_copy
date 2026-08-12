import os
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OPP = _ROOT / "_custom_opp"
_VENDOR = _LOCAL_OPP / "vendors" / "ops-overlap"
_OPAPI = _VENDOR / "op_api" / "lib" / "libcust_opapi.so"

if not _OPAPI.is_file():
    raise RuntimeError(
        f"ops_overlap custom OPP is missing: {_OPAPI}. "
        "Run bash build.sh before importing ops_overlap."
    )

_existing = [
    item
    for item in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if item
]
_vendor_str = str(_VENDOR)
if _vendor_str not in _existing:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        [_vendor_str, *_existing]
    )
os.environ.setdefault("OPS_OVERLAP_INSTALL_OPP_PATH", str(_LOCAL_OPP))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401


kvcache_scatter_copy = torch.ops.ops_overlap.kvcache_scatter_copy


def local_opapi_path() -> str:
    """Return the repository-local custom-op API selected by this package."""
    return str(_OPAPI)


__all__ = [
    "kvcache_scatter_copy",
    "local_opapi_path",
]
