"""Compile actual registrations against libtorch; no Python binding or NPU needed."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from _common import ROOT


TORCH = importlib.util.find_spec("torch")


@unittest.skipUnless(TORCH and shutil.which("g++"), "Torch headers/libraries and g++ are required")
class DispatchRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        torch_dir = Path(next(iter(TORCH.submodule_search_locations)))
        config = (torch_dir / "share/cmake/Torch/TorchConfig.cmake").read_text()
        abi = re.search(r"-D_GLIBCXX_USE_CXX11_ABI=([01])", config)
        if abi is None:
            raise unittest.SkipTest("Cannot determine libtorch C++ ABI from TorchConfig.cmake")
        # Keep the implementation and signatures verbatim; only replace the NPU
        # dependencies and unused Python module macro with small test headers.
        headers = {
            "torch/extension.h": '#pragma once\n#include <ATen/ATen.h>\n#define PYBIND11_MODULE(name, m) void unused_python_module()\n',
            "op_api_common.h": '#pragma once\n#define EXEC_NPU_CMD_ORDERED(...) TORCH_CHECK(false, "NPU launch is unavailable in the dispatcher test")\n',
            "torch_npu/csrc/core/npu/NPUGuard.h": '#pragma once\nnamespace c10_npu { struct NPUGuard { explicit NPUGuard(c10::Device) {} }; }\n',
        }
        for name, contents in headers.items():
            path = cls.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        for name in ("npu_kvcache_scatter_copy.cpp", "ops_registration.cpp"):
            shutil.copy2(ROOT / "torch_extension/csrc" / name, cls.directory / name)
        cls.source = cls.directory / "npu_kvcache_scatter_copy.cpp"
        cls.original = cls.source.read_text()
        cls.compile = ["g++", "-std=c++17", "-O0", abi[0], "-I", str(cls.directory),
                       "-I", str(torch_dir / "include"), "-I", str(torch_dir / "include/torch/csrc/api/include"),
                       str(ROOT / "tests/native/dispatch_registration.cpp")]
        cls.libraries = ["-L", str(torch_dir / "lib"), "-Wl,-rpath," + str(torch_dir / "lib"), "-ltorch_cpu", "-lc10"]

    def test_dispatcher_registration_and_meta_inputs(self):
        binary = str(self.directory / "dispatch_registration")
        self.source.write_text(self.original)
        build = subprocess.run(self.compile + self.libraries + ["-o", binary], capture_output=True, text=True, timeout=120)
        self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
        run = subprocess.run([binary], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("dispatcher OK", run.stdout)

    def test_signature_mismatch_is_rejected_at_compile_time(self):
        invalid = self.original.replace("void Scatter(\n    const at::Tensor& hbm", "void Scatter(\n    at::Tensor hbm", 1)
        self.assertNotEqual(invalid, self.original)
        self.source.write_text(invalid)
        try:
            build = subprocess.run(self.compile + ["-fsyntax-only"], capture_output=True, text=True, timeout=120)
            self.assertNotEqual(build.returncode, 0)
            self.assertIn("PrivateUse1 and Meta kernels must have identical C++ signatures", build.stderr)
        finally:
            self.source.write_text(self.original)


if __name__ == "__main__":
    unittest.main()
