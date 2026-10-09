"""Tests for the H2-first implementations of plan steps 1-3."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao.h2.numpy_model import load_export, run_pipeline
from wangdao.h2_research import (
    FrozenH2Pipeline,
    augment_refiner_features,
    build_h2_refiner,
    hard_random_sample,
    previous_window_features,
    select_fair_working_point,
)
from wangdao.phase1_common import ROOT


def synthetic_events(count=300, stop=199.0):
    rng = np.random.default_rng(17)
    events = np.zeros(count, dtype=[("x", "i4"), ("y", "i4"), ("t", "f8"), ("p", "i4")])
    events["x"] = rng.integers(0, 346, count)
    events["y"] = rng.integers(0, 260, count)
    events["t"] = np.linspace(0.0, stop, count)
    events["p"] = rng.integers(0, 2, count)
    return events


class H2ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weight_path = ROOT / "software/weights/H2_D1_seed42.npz"
        cls.exported = load_export(cls.weight_path)

    def test_frozen_pipeline_reproduces_h2(self):
        events = synthetic_events()
        expected = run_pipeline(events, self.exported, start_ms=0.0, duration_ms=200.0)
        pipeline = FrozenH2Pipeline(self.weight_path)
        prepared = pipeline.prepare(events, windows=[0, 1], include_all=True)
        actual = pipeline.infer_prepared(prepared, len(events))
        np.testing.assert_array_equal(prepared.indices, expected["candidate_indices"])
        np.testing.assert_allclose(actual["probabilities"], expected["probabilities"], rtol=0, atol=0)
        self.assertEqual(prepared.all_features.shape, (len(events), 32))
        self.assertEqual(pipeline.ungated_probability(prepared, len(events)).shape, (len(events),))

    def test_history_uses_only_previous_windows(self):
        events = np.zeros(3, dtype=[("x", "i4"), ("y", "i4"), ("t", "f8"), ("p", "i4")])
        events["x"], events["y"], events["t"] = 16, 24, [10.0, 110.0, 210.0]
        heatmaps = np.stack([
            np.full((34, 44), 0.8, np.float32),
            np.full((34, 44), 0.4, np.float32),
            np.full((34, 44), 0.9, np.float32),
        ])
        features = previous_window_features(events, np.arange(3), [0, 1, 2], heatmaps)
        np.testing.assert_array_equal(features[0], np.zeros(4, np.uint8))
        np.testing.assert_array_equal(features[1], np.full(4, round(255 * 0.8), np.uint8))
        np.testing.assert_array_equal(
            features[2], np.asarray([round(255 * 0.4), round(255 * 0.4),
                                     round(255 * 0.8), round(255 * 0.8)], dtype=np.uint8)
        )
        changed_future = heatmaps.copy()
        changed_future[2] = 0.0
        np.testing.assert_array_equal(
            features, previous_window_features(events, np.arange(3), [0, 1, 2], changed_future)
        )

    def test_a3_null_starts_identically_to_a1(self):
        rng = np.random.default_rng(11)
        base = rng.integers(0, 256, (13, 32), dtype=np.uint8)
        history = np.zeros((13, 4), dtype=np.uint8)
        a1 = build_h2_refiner("A1", self.exported, 42)
        null = build_h2_refiner("A3-null", self.exported, 42)
        with torch.no_grad():
            first = a1(torch.from_numpy(base))
            second = null(torch.from_numpy(augment_refiner_features(base, history)))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_hard_random_sampling_keeps_all_positives_and_is_deterministic(self):
        labels = np.asarray([1, 0, 0, 1, 0, 0, 0, 0])
        probability = np.asarray([.2, .9, .1, .3, .8, .7, .05, .01], np.float32)
        first = hard_random_sample(labels, probability, seed=8, negative_ratio=2)
        second = hard_random_sample(labels, probability, seed=8, negative_ratio=2)
        np.testing.assert_array_equal(first, second)
        self.assertTrue({0, 3}.issubset(set(first.tolist())))
        self.assertIn(1, first)
        self.assertIn(4, first)
        self.assertEqual(len(first), 6)

    def test_fair_selection_uses_iou_fa_recall_tie_break(self):
        def row(iou, recall, pd, fa):
            return {"event": {"iou": iou, "recall": recall},
                    "localization": {"pd": pd, "fa": fa}}
        curve = {
            "0.2": row(.82, .91, .90, .011),
            "0.3": row(.82, .90, .90, .009),
            "0.4": row(.82, .92, .90, .009),
            "0.5": row(.90, .70, .70, .001),
        }
        selected = select_fair_working_point(curve, .90, .01)
        self.assertTrue(selected["eligible"])
        self.assertEqual(selected["threshold"], .4)
        self.assertFalse(select_fair_working_point(curve, 1.0, 0.0)["eligible"])


if __name__ == "__main__":
    unittest.main()
