from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step6_common import (  # noqa: E402
    _ap_replicates, aggregate_decision, exact_pr, matched_attribution,
    stratified_weights,
)
from run_step6 import density_compatibility  # noqa: E402
from step4_common import DensityTransform  # noqa: E402


def seed_row(seed: int, diou: float, dap: float, noninferior: bool, revised: bool, attribution: bool):
    return {
        "seed": seed,
        "delta_vs_h2": {"overall_iou": diou, "average_precision": dap},
        "noninferiority": {"passes": noninferior},
        "revised_candidate_gates": {"passes": revised},
        "matched_attribution": {"passes": attribution},
    }


class Step6Tests(unittest.TestCase):
    def test_stratified_weights_keep_density_counts(self):
        files = ["train_089.npz", "train_092.npz", "train_002.npz", "train_020.npz"]
        first = stratified_weights(files, iterations=32, seed=7)
        second = stratified_weights(files, iterations=32, seed=7)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first[:, :2].sum(axis=1) == 2))
        self.assertTrue(np.all(first[:, 2:].sum(axis=1) == 2))

    def test_grouped_ap_bootstrap_matches_explicit_duplication(self):
        labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
        probability = np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float32)
        sequence = np.asarray([0, 0, 1, 1], dtype=np.int64)
        weights = np.asarray([[1, 1], [2, 0], [0, 2]], dtype=np.int16)
        observed = _ap_replicates(labels, probability, sequence, weights)
        expected = []
        for row in weights:
            indices = np.concatenate([
                np.tile(np.flatnonzero(sequence == index), int(row[index]))
                for index in range(2) if row[index]
            ])
            expected.append(exact_pr(labels[indices], probability[indices])["average_precision"])
        self.assertTrue(np.allclose(observed, expected, atol=1e-12))

    def test_matched_attribution_requires_both_metrics_nonlower(self):
        candidate = {"average_precision": 0.93, "overall": {"event": {"iou": 0.79}}}
        control = {"average_precision": 0.92, "overall": {"event": {"iou": 0.791}}}
        self.assertFalse(matched_attribution(candidate, control)["passes"])
        candidate["overall"]["event"]["iou"] = 0.792
        self.assertTrue(matched_attribution(candidate, control)["passes"])

    def test_aggregate_decision_passes_frozen_rule(self):
        rows = [
            seed_row(42, 0.006, 0.004, True, True, True),
            seed_row(43, 0.002, 0.001, True, False, True),
            seed_row(44, -0.001, -0.0005, True, False, False),
        ]
        result = aggregate_decision(rows)
        self.assertTrue(result["passes"])
        self.assertAlmostEqual(result["mean_delta_iou"], 0.007 / 3)

    def test_aggregate_decision_rejects_failed_seed_noninferiority(self):
        rows = [
            seed_row(42, 0.006, 0.004, True, True, True),
            seed_row(43, 0.002, 0.001, False, False, True),
            seed_row(44, 0.001, 0.001, True, False, True),
        ]
        self.assertFalse(aggregate_decision(rows)["passes"])

    def test_density_check_accepts_float_tail_when_uint8_codes_match(self):
        records = [
            {"file": "a.npz", "window_rate_eps": np.asarray([500.0, 1000.0], dtype=np.float32)},
            {"file": "b.npz", "window_rate_eps": np.asarray([2000.0], dtype=np.float32)},
        ]
        actual_min = float(np.log1p(500.0))
        actual_max = float(np.log1p(2000.0))
        frozen = DensityTransform(actual_min + 1e-14, actual_max - 1e-14, 3, 2)
        report = density_compatibility(records, frozen)
        self.assertTrue(report["passes"])
        self.assertFalse(report["exact_dataclass_match"])

    def test_density_check_rejects_material_encoding_change(self):
        records = [
            {"file": "a.npz", "window_rate_eps": np.asarray([500.0, 1000.0], dtype=np.float32)},
            {"file": "b.npz", "window_rate_eps": np.asarray([2000.0], dtype=np.float32)},
        ]
        frozen = DensityTransform(float(np.log1p(1.0)), float(np.log1p(20_000.0)), 3, 2)
        report = density_compatibility(records, frozen)
        self.assertFalse(report["passes"])
        self.assertFalse(report["checks"]["uint8_density_codes_identical"])

    def test_density_check_rejects_candidate_count_change(self):
        records = [
            {"file": "a.npz", "window_rate_eps": np.asarray([500.0, 1000.0], dtype=np.float32)},
        ]
        frozen = DensityTransform(float(np.log1p(500.0)), float(np.log1p(1000.0)), 3, 1)
        report = density_compatibility(records, frozen)
        self.assertFalse(report["passes"])
        self.assertFalse(report["checks"]["train_candidates_exact"])

    def test_readmes_have_single_line_commands(self):
        for name in ("README.md", "README_SERVER.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertFalse(any(line.rstrip().endswith("\\") for line in text.splitlines()))


if __name__ == "__main__":
    unittest.main()
