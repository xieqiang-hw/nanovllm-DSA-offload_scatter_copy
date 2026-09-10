"""CPU storage model: swapped DRAM forbids direct .cpu(), even through views."""
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from _common import ROOT


class Bytes:
    def __init__(self, values, memory="cpu"):
        self.values, self.memory = list(values), memory
        self.shape, self.dtype = (len(self.values),), "int8"
        self.device = "cpu" if memory == "cpu" else "npu:0"

    def view(self, dtype):
        assert dtype == "int8", "The copy must operate on raw bytes."
        return self

    def data_ptr(self):
        return id(self.values)

    def fill_(self, value):
        self.values[:] = [value] * len(self.values)
        return self

    def mul_(self, other):
        self.values[:] = [a * b for a, b in zip(self.values, other.values)]
        return self

    def to(self, device):
        assert self.memory == "cpu" and device == "npu:0"
        return Bytes(self.values, "hbm")

    def cpu(self):
        assert self.memory != "dram", "Direct swapped DRAM .cpu() is unsupported."
        return Bytes(self.values)


class DramReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch = SimpleNamespace(
            int8="int8", npu=SimpleNamespace(synchronize=lambda: None),
            empty=lambda shape, **kwargs: Bytes([0] * shape[0], "hbm"),
            equal=lambda a, b: a.values == b.values,
        )
        torch_npu = SimpleNamespace(empty_with_swapped_memory=lambda shape, **kwargs: Bytes([0] * shape[0], "dram"))
        spec = importlib.util.spec_from_file_location("scatter_fixture_model", ROOT / "tests/_case.py")
        cls.fixture = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch, "torch_npu": torch_npu, "kvcache_ops": SimpleNamespace()}), \
                patch.object(sys, "path", sys.path.copy()):
            spec.loader.exec_module(cls.fixture)

    def test_all_byte_values_round_trip_and_source_stays_read_only(self):
        cpu = Bytes(range(-128, 128))
        dram = self.fixture.swapped_from_cpu(cpu, "npu:0")
        self.assertEqual(dram.memory, "dram")
        actual = self.fixture.swapped_to_cpu_bytes(dram)
        self.assertEqual(actual.values, cpu.values)
        self.assertEqual(dram.values, cpu.values)
        self.assertNotEqual(actual.data_ptr(), dram.data_ptr())

    def test_verification_preserves_source_and_metadata_checks(self):
        case = self.fixture.Case.__new__(self.fixture.Case)
        case.debug = False
        case.sources_cpu, case.sources = [Bytes(range(-128, 128))], [Bytes(range(-128, 128), "dram")]
        case.metadata_cpu, case.metadata = [Bytes([1, 2, 3])], [Bytes([1, 2, 3], "hbm")]
        case.targets, case.expected = [], []
        case.inputs = (*case.sources, *case.metadata)
        case.pointers = tuple(t.data_ptr() for t in case.inputs)
        case.verify()  # The old source.cpu() path raises in this storage model.
        case.sources[0].values[0] = 0
        with self.assertRaisesRegex(AssertionError, "DRAM source 0 was modified"):
            case.verify()
        case.sources[0].values[0] = -128
        case.metadata[0].values[0] = 0
        with self.assertRaisesRegex(AssertionError, "metadata 0 was modified"):
            case.verify()


if __name__ == "__main__":
    unittest.main()
