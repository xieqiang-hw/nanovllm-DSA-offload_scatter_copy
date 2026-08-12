import multiprocessing
import os
from pathlib import Path

import torch_npu
from setuptools import find_packages, setup
from torch_npu.utils.cpp_extension import NpuExtension
from torch.utils.cpp_extension import BuildExtension, verify_ninja_availability


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
USE_NINJA = os.getenv("USE_NINJA") == "1"
MAX_JOBS = int(os.getenv("MAX_JOBS", multiprocessing.cpu_count()))

if USE_NINJA:
    verify_ninja_availability()

setup(
    name="ops_overlap",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="ops_overlap._C",
            sources=[
                str(CSRC / "ops_common.cpp"),
                str(CSRC / "ops_registration.cpp"),
                str(CSRC / "npu_kvcache_scatter_copy.cpp"),
            ],
            include_dirs=[str(CSRC)],
            extra_compile_args=[
                "-I"
                + str(
                    TORCH_NPU_ROOT
                    / "include"
                    / "third_party"
                    / "acl"
                    / "inc"
                ),
                "-O3",
                "-std=c++17",
                "-fvisibility=hidden",
            ],
            extra_link_args=["-ldl"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=USE_NINJA,
            parallel=MAX_JOBS,
        )
    },
)
