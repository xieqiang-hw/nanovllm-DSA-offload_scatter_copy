"""Exercise build.sh through installation with a fake SDK, without CANN/NPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from _common import ROOT


# Reproduce the toolchain distinction: aclnn emits op_api only; tf emits a RUN
# package. The real build.sh must select, configure, install and validate it.
SDK_STUB = r'''
import json
import os
from pathlib import Path
import shlex
import sys

stage, args = sys.argv[1], sys.argv[2:]
names = ["hbmKvRef", "dramKv", "hbmKpeRef", "dramKpe", "hbmBlockTable",
         "dramBlockTable", "sourceTokenIds", "destinationSlots", "copyCounts"]
header = "aclnnStatus aclnnKvcacheScatterCopyGetWorkspaceSize(" + ",".join(
    "aclTensor *" + name for name in names) + ",uint64_t *size,aclOpExecutor **executor);"

def opapi(root):
    for path, text in (("op_api/lib/libcust_opapi.so", "fake library"),
                       ("op_api/include/aclnn_kvcache_scatter_copy.h", header)):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)

def launcher(stage):
    return "#!/bin/sh\nexec " + shlex.join([sys.executable, str(Path(__file__).resolve()), stage]) + ' "$@"\n'

if stage == "gen":
    root = Path(args[args.index("-out") + 1])
    for directory in ("op_host", "op_kernel", "framework"):
        (root / directory).mkdir(parents=True)
    (root / "framework/CMakeLists.txt").write_text("unwanted TensorFlow plugin")
    (root / "mode").write_text(args[args.index("-f") + 1])
    (root / "CMakePresets.json").write_text(json.dumps({"configurePresets": [{
        "name": "default", "cacheVariables": {
            "ENABLE_BINARY_PACKAGE": {"type": "BOOL", "value": "False"},
            "ASCEND_PACK_SHARED_LIBRARY": {"type": "BOOL", "value": "True"},
            "ASCEND_COMPUTE_UNIT": {"type": "STRING", "value": args[args.index("-c") + 1]}}}]}))
    (root / "build.sh").write_text(launcher("build"))
elif stage == "build":
    root = Path.cwd()
    output = root / "build_out"
    opapi(output)
    if (root / "mode").read_text() == "tf":
        cache = json.loads((root / "CMakePresets.json").read_text())["configurePresets"][0]["cacheVariables"]
        assert cache["ENABLE_BINARY_PACKAGE"]["value"] == "True"
        assert cache["ENABLE_SOURCE_PACKAGE"]["value"] == "True"
        assert cache["ASCEND_PACK_SHARED_LIBRARY"]["value"] == "False"
        assert (root / "framework/CMakeLists.txt").read_text() == ""
        assert args == ["package"]
        assert (root / "op_kernel/kvcache_scatter_copy.cpp").is_file()
        for index in range(int(os.getenv("STUB_PACKAGES", "1"))):
            (output / f"custom_opp_{index}.run").write_text(launcher("install"))
elif stage == "install":
    root = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--install-path=")))
    vendor = root / "vendors/customize"
    opapi(vendor)
    metadata = vendor / "op_impl/ai_core/tbe/kernel/config/binary_info_config.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"KvcacheScatterCopy": {}}')
else:
    raise ValueError(stage)
'''


class BuildPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "scatter project"
        for name in ("csrc", "torch_extension/kvcache_ops"):
            shutil.copytree(ROOT / name, self.root / name, ignore=shutil.ignore_patterns("__pycache__", "*.so"))
        (self.root / "tests").mkdir()
        shutil.copy2(ROOT / "tests/check_build.py", self.root / "tests/check_build.py")
        shutil.copy2(ROOT / "build.sh", self.root / "build.sh")
        (self.root / "torch_extension/setup.py").write_text(
            "import os\nfrom pathlib import Path\n"
            "if not os.getenv('STUB_SKIP_EXTENSION'):\n"
            "    Path('kvcache_ops/_C.stub.so').write_text('fake extension')\n")
        binary = Path(temporary.name) / "bin"
        binary.mkdir()
        generator = binary / "msopgen"
        generator.write_text(f"#!{sys.executable}\n" + SDK_STUB)
        generator.chmod(0o755)
        self.environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                                PYTHON=sys.executable, MAX_JOBS="1", SOC_VERSION="ascend910_93")

    def run_build(self, **environment):
        return subprocess.run(["bash", "build.sh"], cwd=self.root,
                              env=self.environment | environment, capture_output=True, text=True, timeout=30)

    def test_full_package_install_and_extension_for_both_socs(self):
        for soc in ("ascend910_93", "ascend950"):
            with self.subTest(soc=soc):
                result = self.run_build(SOC_VERSION=soc)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Build complete:", result.stdout)
                self.assertTrue((self.root / f"build/{soc}/opp/vendors/customize/op_api/lib/libcust_opapi.so").is_file())
                self.assertTrue((self.root / "torch_extension/kvcache_ops/_C.stub.so").is_file())

    def test_standalone_aclnn_layout_is_not_a_complete_opp(self):
        script = self.root / "build.sh"
        script.write_text(script.read_text().replace("-f tf", "-f aclnn"))
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("found 0. Build is incomplete", result.stderr)
        self.assertTrue((self.root / "build/ascend910_93/custom_op/build_out/op_api/lib/libcust_opapi.so").is_file())
        self.assertNotIn("Build complete:", result.stdout)

    def test_missing_or_ambiguous_run_package_stops_before_extension(self):
        for count in ("0", "2"):
            with self.subTest(count=count):
                result = self.run_build(STUB_PACKAGES=count)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"found {count}. Build is incomplete", result.stderr)
                self.assertFalse((self.root / "torch_extension/kvcache_ops/_C.stub.so").exists())
                self.assertNotIn("Build complete:", result.stdout)

    def test_missing_extension_is_not_reported_as_success(self):
        result = self.run_build(STUB_SKIP_EXTENSION="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Torch extension was not produced", result.stderr)
        self.assertNotIn("Build complete:", result.stdout)


if __name__ == "__main__":
    unittest.main()
