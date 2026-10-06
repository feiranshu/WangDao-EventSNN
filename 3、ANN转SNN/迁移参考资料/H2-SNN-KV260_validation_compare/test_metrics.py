"""Boundary and differential tests; synthetic data only, no Test dataset access."""
import unittest

import numpy as np
import torch

from metrics import (event_counts, event_summary, connected_components_8,
                     localization_counts, localization_summary, sum_counts)
from reference_check import ReferenceEvaluator, union_find_components


class MetricTests(unittest.TestCase):
    def test_acc_is_target_recall_not_overall(self):
        result = event_summary(event_counts(np.array([1, 1, 0, 0, 0]), np.array([.9, .1, 0, 0, 0]), .5))
        self.assertEqual(result["official_acc"], .5)
        self.assertEqual(result["overall_accuracy"], .8)
        self.assertEqual(result["iou"], .5)

    def test_threshold_is_inclusive(self):
        result = event_counts(np.array([1, 0]), np.array([.5, .5]), .5)
        self.assertEqual(result, {"tp": 1, "fp": 1, "fn": 0, "tn": 0})

    def test_full_event_scoring_penalizes_rejected_targets(self):
        result = event_counts(np.array([1, 1, 0]), np.array([.8, 0, 0]), .5)
        self.assertEqual(result["fn"], 1)

    def test_micro_counts_not_macro_iou(self):
        rows = [{"tp": 1, "fp": 0, "fn": 0, "tn": 0},
                {"tp": 0, "fp": 0, "fn": 9, "tn": 0}]
        self.assertEqual(event_summary(sum_counts(rows))["iou"], .1)

    def test_undefined_metrics_are_null(self):
        result = event_summary(event_counts(np.zeros(3), np.zeros(3), .5))
        self.assertIsNone(result["iou"])
        self.assertIsNone(result["official_acc"])

    def test_invalid_predictions_rejected(self):
        for p in (np.array([np.nan]), np.array([2]), np.ones((1, 1))):
            with self.assertRaises(ValueError):
                event_counts(np.array([1]), p, .5)

    def test_diagonal_is_connected(self):
        mask = np.eye(4, dtype=np.uint8)
        self.assertEqual(connected_components_8(mask), 1)
        self.assertEqual(union_find_components(mask), 1)

    def test_independent_connected_component_algorithms(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            mask = rng.random((15, 21)) < rng.uniform(.01, .7)
            self.assertEqual(connected_components_8(mask), union_find_components(mask))

    def test_official_open_boundaries_and_integer_ms(self):
        loc = np.array([[1, 1, 0], [1, 1, 1.9], [1, 1, 50], [1, 1, 50.9],
                        [1, 1, 51], [1, 1, 100], [1, 1, 101]])
        result = localization_counts(loc, np.ones(7), np.ones(7), np.ones(7), .5)
        self.assertEqual(result["scored_localization_events"], 3)
        self.assertEqual(result["frame_denominator"], 2)
        self.assertEqual(result["targets"], 3)

    def test_distinct_target_ids_count_once_each_frame(self):
        loc = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 2], [2, 2, 2], [1, 1, 101]])
        result = localization_counts(loc, np.ones(5), np.array([1, 1, 1, 2, 1]), np.ones(5), .5)
        self.assertEqual(result["targets"], 3)

    def test_official_uint8_wrap_preserved(self):
        loc = np.tile([5, 5, 1], (258, 1))
        loc[0, 2], loc[-1, 2] = 0, 101
        result = localization_counts(loc, np.zeros(258), np.zeros(258), np.ones(258), .5)
        self.assertEqual(result["wrapped_false_pixels"], 1)
        self.assertEqual(result["false_components"], 1)

    def test_pinned_official_reference_parity(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(57)
        labels = rng.integers(0, 2, 1000)
        identities = np.where(labels == 1, rng.integers(1, 4, 1000), 0)
        loc = np.column_stack((rng.integers(0, 346, 1000), rng.integers(0, 260, 1000),
                               np.sort(rng.uniform(0, 350, 1000))))
        probability = rng.random(1000).astype(np.float32)
        for threshold in (.28, .5, .9):
            reference = ReferenceEvaluator()
            reference.add("synthetic", loc, labels, identities, probability, threshold)
            oracle = reference.summary(threshold)
            event = event_summary(event_counts(labels, probability, threshold))
            local = localization_summary(localization_counts(loc, labels, identities, probability, threshold))
            for key in ("iou", "official_acc"):
                self.assertAlmostEqual(event[key], oracle[key], delta=1e-7)
            for key in ("targets", "correct_targets", "false_components", "frame_denominator"):
                self.assertEqual(local[key], oracle[key])
            self.assertAlmostEqual(local["pd"], oracle["pd"], delta=1e-7)
            self.assertAlmostEqual(local["fa"], oracle["fa"], delta=1e-12)


if __name__ == "__main__":
    unittest.main()
