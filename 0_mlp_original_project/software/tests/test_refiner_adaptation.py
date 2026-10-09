"""The faster offline S1 path must reproduce the frozen integer reference."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao.phase1_common import ROOT
from wangdao.refiner_adaptation import (FrozenS1Pipeline, refiner_features_shared,
                                       error_decomposition, full_probability, build_refiner)
from wangdao.h2.representation import build_refiner_features


class RefinerAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weights = ROOT / "software/weights"
        cls.pipeline = FrozenS1Pipeline(weights / "H2_D1_seed42.npz",
                                       weights / "S1P_INT8_ACC24_MEM24_Q12.npz")

    def test_logits_only_exact_to_original_s1(self):
        rng = np.random.default_rng(20)
        for n in (0, 1, 17):
            features = rng.integers(0, 256, (4, n, 128), dtype=np.uint8)
            np.testing.assert_array_equal(self.pipeline.s1.forward(features).logits,
                                          self.pipeline.fast_s1(features))

    def test_preconverted_fixed_weights_exact(self):
        rng = np.random.default_rng(3)
        for stage, width in (("proposal", 128), ("refiner", 32)):
            for n in (0, 19):
                features = rng.integers(0, 256, (n, width), dtype=np.uint8)
                reference = getattr(self.pipeline.fixed, stage)(features).logits
                np.testing.assert_array_equal(reference, self.pipeline.fast_fixed.logits(features, stage))

    def test_shared_features_match_original(self):
        rng = np.random.default_rng(21)
        ev = np.zeros(300, dtype=[("x", "i4"), ("y", "i4"), ("t", "f8"), ("p", "i4")])
        ev["x"] = rng.integers(0, 346, len(ev))
        ev["y"] = rng.integers(0, 260, len(ev))
        ev["t"] = np.linspace(0, 199, len(ev))
        ev["p"] = rng.integers(0, 2, len(ev))
        heatmaps = rng.random((2, 34, 44)).astype(np.float32)
        indices = np.arange(0, len(ev), 3)
        expected = build_refiner_features(ev, indices, [0, 1], heatmaps, .51)
        actual = refiner_features_shared(ev, indices, [0, 1], heatmaps, .51)
        np.testing.assert_array_equal(expected, actual)

    def test_rejected_events_are_counted_as_false_negatives(self):
        probability = full_probability(np.array([1, 2]), np.array([.1, .8]), 4)
        result = error_decomposition(np.array([1, 1, 0, 0]), np.array([1, 2]), probability)
        self.assertEqual(result["proposal_false_negatives"], 1)
        self.assertEqual(result["refiner_false_negatives"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 2)

    def test_new_refiners_are_explicit_floating_variants(self):
        b1 = build_refiner("B1", self.pipeline.h2)
        b2 = build_refiner("B2", self.pipeline.h2)
        self.assertEqual(b1.dimensions, (32, 64, 32, 4))
        self.assertEqual(b2.dimensions, (32, 128, 64, 4))
        self.assertFalse(b1.quantized)
        self.assertFalse(b2.quantized)


if __name__ == "__main__":
    unittest.main()
