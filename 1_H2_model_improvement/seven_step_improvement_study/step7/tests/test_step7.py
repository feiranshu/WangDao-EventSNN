from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from step7_common import (  # noqa: E402
    FIXED_EPOCH, FIXED_MODEL, FIXED_SEED, FIXED_THRESHOLD, PROTOCOL,
    acceptance_gates, acceptance_role_paths, paired_bootstrap, stratified_weights,
)


def group(iou: float, recall: float, pd: float, fa: float):
    return {
        "event": {"tp": 10, "fp": 1, "fn": 1, "tn": 100, "iou": iou, "precision": 0.9, "recall": recall, "f1": 0.9},
        "localization": {"targets": 10, "correct_targets": 9, "false_components": 1, "frame_denominator": 1, "pd": pd, "fa": fa},
    }


def view(ap: float, iou: float, recall: float, pd: float, fa: float):
    return {
        "threshold": 0.4,
        "average_precision": ap,
        "overall": group(iou, recall, pd, fa),
        "high_density": group(iou, recall, pd, fa),
        "ordinary_density": group(iou, recall, pd, fa),
        "per_sequence": {},
    }


class Step7Tests(unittest.TestCase):
    def test_protocol_candidate_is_frozen(self):
        self.assertEqual(PROTOCOL, "h2-ann-step7-supplemental-independent-acceptance-v1")
        self.assertEqual((FIXED_MODEL, FIXED_SEED, FIXED_EPOCH, FIXED_THRESHOLD), ("A4-2", 42, 8, 0.40))

    def test_only_validation_and_test_are_reachable(self):
        with self.assertRaises(ValueError):
            acceptance_role_paths("train_dev", limit=1)

    def test_stratified_bootstrap_preserves_group_sizes(self):
        high = np.asarray([True, True, False, False, False])
        first, metadata = stratified_weights(high, 64, 9)
        second, _ = stratified_weights(high, 64, 9)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first[:, :2].sum(axis=1) == 2))
        self.assertTrue(np.all(first[:, 2:].sum(axis=1) == 3))
        self.assertIn("density-stratified", metadata["method"])

    def test_acceptance_preserves_fa_guards(self):
        h2_r = view(0.92, 0.78, 0.85, 0.84, 1.0e-5)
        h2_f = view(0.92, 0.78, 0.85, 0.84, 1.0e-5)
        candidate = view(0.93, 0.79, 0.86, 0.85, 1.049e-5)
        self.assertTrue(acceptance_gates(candidate, h2_f, h2_r)["passes"])
        candidate["ordinary_density"]["localization"]["fa"] = 1.051e-5
        result = acceptance_gates(candidate, h2_f, h2_r)
        self.assertFalse(result["passes"])
        self.assertFalse(result["checks"]["ordinary_fa_rise_within_5pct"])

    def test_missing_density_group_never_passes(self):
        h2 = view(0.92, 0.78, 0.85, 0.84, 1.0e-5)
        candidate = view(0.93, 0.79, 0.86, 0.85, 1.0e-5)
        candidate["high_density"] = None
        self.assertFalse(acceptance_gates(candidate, h2, h2)["passes"])

    def test_paired_bootstrap_recomputes_event_metrics(self):
        files = np.asarray(["a.npz", "b.npz"])
        labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
        sequence = np.asarray([0, 0, 1, 1], dtype=np.int16)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            np.savez_compressed(root / "candidate.npz", labels=labels, probability=np.asarray([0.9, 0.2, 0.8, 0.1], dtype=np.float32), sequence=sequence, sequence_files=files)
            np.savez_compressed(root / "h2.npz", labels=labels, probability=np.asarray([0.8, 0.3, 0.7, 0.2], dtype=np.float32), sequence=sequence, sequence_files=files)
            candidate = view(1.0, 1.0, 1.0, 1.0, 0.0)
            h2 = view(1.0, 1.0, 1.0, 1.0, 0.0)
            candidate["per_sequence"] = {"a.npz": group(1.0, 1.0, 1.0, 0.0), "b.npz": group(1.0, 1.0, 1.0, 0.0)}
            h2["per_sequence"] = candidate["per_sequence"]
            result = paired_bootstrap(root / "candidate.npz", root / "h2.npz", candidate, h2, [True, False], iterations=16, seed=3)
        self.assertEqual(result["iterations"], 16)
        self.assertIn("delta_event_iou", result)
        self.assertIn("delta_average_precision", result)

    def test_readmes_have_single_line_commands(self):
        for name in ("README.md", "README_SERVER.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertFalse(any(line.rstrip().endswith("\\") for line in text.splitlines()))


if __name__ == "__main__":
    unittest.main()

