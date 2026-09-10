"""Caller-owned BF16 / packed-C8 KV scatter on Ascend A3 and A5."""
import os
from pathlib import Path
from ._soc import detect_soc

_ROOT = Path(__file__).resolve().parents[2]
_SOC = detect_soc()
_LIBS = list((_ROOT / "build" / _SOC / "opp" / "vendors").glob("*/op_api/lib/libcust_opapi.so"))
if len(_LIBS) != 1:
    raise RuntimeError(f"Run bash build.sh on this {_SOC} machine first.")
_OPAPI = _LIBS[0].resolve()
_VENDOR = _OPAPI.parents[2]
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(_VENDOR)
os.environ["SCATTER_OPAPI_LIB"] = str(_OPAPI)

# Select the local OPP before torch_npu initializes its operator registry.
import torch
import torch_npu
from . import _C

kvcache_scatter_copy = torch.ops.kvcache_ops.kvcache_scatter_copy.default
__all__ = ["kvcache_scatter_copy"]
