"""Third-layer no-fire readout, exact pooling, and frozen export contracts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
from host_tools.rtl_verification.generate_readout_vectors import generate, readout_trace
from host_tools.rtl_verification.generate_rtl_vectors import frozen_first_layer, lif_step
from host_tools.rtl_verification.run_rtl_simulation import validate_manifest, sha256
from wangdao.fixed import fixed_balanced_add_tree, signed_round_divide


class ReadoutRTLTests(unittest.TestCase):
    def test_trace_matches_scalar_tree_no_fire_and_exact_mean(self) -> None:
        exact, approximate = frozen_first_layer()
        inputs = np.random.default_rng(520).integers(0, 2, (3, 4, 24), dtype=np.uint8)
        inputs[0] = 0
        inputs[1] = 1
        for model, k in ((exact, 0), (approximate, 2)):
            trace, final = readout_trace(inputs, model, k)
            saw_above_threshold = False
            for fixture in range(3):
                for channel in range(16):
                    membrane, total, tree_total, mem_total = 0, 0, 0, 0
                    for step in range(4):
                        products = inputs[fixture, step].astype(np.int64) * model.weights[2][channel]
                        current, tree, _ = fixed_balanced_add_tree(products, 24, k)
                        aligned = int(current) << (12 - int(model.weight_fraction_bits[2][channel]))
                        membrane, spike, saturated = lif_step(membrane, aligned, False, k)
                        self.assertEqual(spike, 0)
                        saw_above_threshold |= membrane >= model.threshold_q
                        total += membrane
                        tree_total += tree
                        mem_total += saturated
                        self.assertEqual(trace[fixture, step, channel].tolist(),
                            [membrane & 0xFFFFFF, total & 0x3FFFFFF, tree, saturated])
                    expected = int(signed_round_divide(np.asarray(total, dtype=np.int64), 4))
                    self.assertEqual(final[fixture, channel].tolist(), [expected & 0xFFFFFFFF, tree_total, mem_total])
            self.assertTrue(saw_above_threshold)
        self.assertTrue(np.any(readout_trace(inputs, exact, 0)[1] != readout_trace(inputs, approximate, 2)[1]))

    def test_grouped_24_input_tree_preserves_odd_tail_and_saturation(self) -> None:
        values = np.random.default_rng(521).integers(-(1 << 23), 1 << 23, (8, 24), dtype=np.int64)
        for k in (0, 2):
            full, full_sats, _ = fixed_balanced_add_tree(values, 24, k)
            partial, group_sats, _ = fixed_balanced_add_tree(values.reshape(8, 3, 8), 24, k)
            grouped, tail_sats, _ = fixed_balanced_add_tree(partial, 24, k)
            np.testing.assert_array_equal(full, grouped)
            self.assertEqual(full_sats, group_sats + tail_sats)

    def test_bad_binary_shape_and_readout_contract_rejected(self) -> None:
        exact, _ = frozen_first_layer()
        for inputs in (np.zeros((1, 4, 23)), np.zeros((1, 24)), np.full((1, 4, 24), 2), np.full((1, 4, 24), -1)):
            with self.assertRaises(ValueError):
                readout_trace(inputs, exact, 0)
        exact.readout = "last_membrane"
        with self.assertRaisesRegex(ValueError, "mean_membrane"):
            readout_trace(np.zeros((1, 4, 24)), exact, 0)

    def test_frozen_third_layer_mismatch_rejected(self) -> None:
        exact, approximate = frozen_first_layer()
        approximate.weights = tuple(w.copy() for w in approximate.weights)
        approximate.weights[2][0, 0] ^= 1
        with tempfile.TemporaryDirectory() as temporary:
            with patch("host_tools.rtl_verification.generate_readout_vectors.frozen_first_layer", return_value=(exact, approximate)):
                with self.assertRaisesRegex(ValueError, "weights or scaling differ"):
                    generate(Path(temporary), 0, 42)

    def test_existing_manifest_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proposal_manifest.json"
            path.write_text("{}", encoding="ascii")
            with self.assertRaises(FileExistsError):
                generate(path.parent, 0, 42)
            self.assertEqual(path.read_text(), "{}")

    def test_proposal_manifest_and_corrupt_readout_rejected(self) -> None:
        manifest, files = validate_manifest(ROOT / "pl_design/test_vectors", "proposal")
        self.assertTrue(manifest["configuration"]["complete_window"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for source in files:
                shutil.copyfile(source, directory / source.name)
            path = directory / "proposal_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            for name in ("layer3_bank0.txt", "layer3_shifts.txt", "proposal_trace.txt"):
                vector = directory / name
                original = vector.read_bytes()
                vector.write_text("00\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    validate_manifest(directory, "proposal")
                vector.write_bytes(original)
            vector = directory / "readout_trace.txt"
            rows = vector.read_text(encoding="ascii").splitlines()
            columns = rows[0].split()
            columns[1] = "4000000"  # 27th bit must be rejected even with a matching hash.
            rows[0] = " ".join(columns)
            vector.write_text("\n".join(rows) + "\n", encoding="ascii")
            manifest["files"]["readout_trace"]["sha256"] = sha256(vector)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid field"):
                validate_manifest(directory, "proposal")


if __name__ == "__main__":
    unittest.main()
