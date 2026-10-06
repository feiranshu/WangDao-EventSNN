import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from snn_accel.export import LayerSpec, export_model, verify_manifest


class ExportTests(unittest.TestCase):
    def test_export_and_verify(self):
        layer = LayerSpec(
            name="layer0",
            input_channels=2,
            output_channels=2,
            kernel=1,
            stride=1,
            threshold=32,
            leak_shift=2,
            accumulator_bits=16,
            approx_lsb_bits=2,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = export_model({"layer0": np.array([0.0, 0.5, -1.0, 2.0])}, [layer], root)
            verify_manifest(root)
            self.assertEqual((root / "weights.bin").stat().st_size, 4)
            self.assertEqual(manifest["layers"][0]["weight_offset"], 0)
            loaded = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["input_shape"], [2, 256, 320])

    def test_detects_corruption(self):
        layer = LayerSpec("x", 1, 1, 1, 1, 1, 0, 16, 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_model({"x": np.array([1.0])}, [layer], root)
            (root / "weights.bin").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                verify_manifest(root)


if __name__ == "__main__":
    unittest.main()

