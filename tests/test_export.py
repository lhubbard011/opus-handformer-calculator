"""Weight export: the .safetensors file holds every weight bit-exactly, standard tools read it,
and the numpy-only runner computes the right answers from the file alone."""

import json
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handformer import arith, export

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent

try:
    import safetensors.numpy as st_numpy  # optional: the official reader
except ImportError:
    st_numpy = None


class TestSafetensors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="handformer-export-"))
        cls.path = cls.tmp / "w.safetensors"
        export.export(cls.path)
        cls.arrays, cls.index = export.tensors()

    def test_round_trip_is_bit_exact(self):
        got, meta = export.read_safetensors(self.path)
        self.assertEqual(set(got), set(self.arrays))
        for name, a in self.arrays.items():
            self.assertEqual(got[name].dtype, np.float64, name)
            self.assertTrue(np.array_equal(got[name], a), name)  # exact, not approximately equal
        self.assertEqual(json.loads(meta["residual_dims"]), self.index["residual_dims"])
        self.assertEqual(int(meta["nonzero_weights"]), arith.build().nonzero_params())
        log.info("%d tensors round-trip exactly", len(got))

    @unittest.skipIf(st_numpy is None, "safetensors package not installed")
    def test_official_reader_agrees(self):
        got = st_numpy.load_file(str(self.path))
        self.assertEqual(set(got), set(self.arrays))
        for name, a in self.arrays.items():
            self.assertTrue(np.array_equal(got[name], a), name)

    def test_numpy_only_runner_computes_from_the_file(self):
        out = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "run_weights.py"), str(self.path),
                              "2+3*(4-1)", "355/113~6", "-2^3"], capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr)
        answers = [line.split("->")[-1].strip() for line in out.stderr.splitlines() if "->" in line]
        self.assertEqual(answers, ["11", "3.141592", "-8"], out.stderr)

    def test_unknown_suffix_is_refused(self):
        with self.assertRaises(ValueError):
            export.export(self.tmp / "w.bin")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
