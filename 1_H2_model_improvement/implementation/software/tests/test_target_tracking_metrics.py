from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.target_tracking_metrics import (  # noqa: E402
    Detection,
    bbox_iou,
    frame_indices,
    hota_metrics,
    hota_raw,
    linear_sum_assignment,
    longest_miss_per_track_ms,
    predicted_components,
)


def event_array(rows):
    dtype = np.dtype([
        ("x", "i4"), ("y", "i4"), ("t", "f8"), ("p", "i1"),
        ("label", "i1"), ("name", "i4"),
    ])
    return np.asarray(rows, dtype=dtype)


class TargetTrackingMetricTests(unittest.TestCase):
    def test_half_open_window_boundaries(self):
        events = event_array([
            (1, 1, 100.0, 1, 1, 1),
            (1, 1, 149.999, 1, 1, 1),
            (1, 1, 150.0, 1, 1, 1),
            (1, 1, 200.0, 1, 1, 1),
        ])
        self.assertEqual(frame_indices(events).tolist(), [0, 0, 1, 2])

    def test_single_pixel_box_has_area(self):
        item = Detection(0, 1, (5.0, 7.0, 6.0, 8.0))
        self.assertEqual(bbox_iou(item, item), 1.0)

    def test_prediction_components_do_not_read_labels(self):
        first = event_array([
            (2, 2, 0.0, 1, 1, 9),
            (3, 2, 1.0, 1, 0, 0),
            (20, 20, 2.0, 1, 0, 0),
        ])
        second = first.copy()
        second["label"] = 1 - second["label"]
        second["name"] = np.asarray([100, 101, 102])
        probability = np.asarray([0.8, 0.9, 0.95])
        left = predicted_components(first, probability, 0.5, 1)
        right = predicted_components(second, probability, 0.5, 1)
        self.assertEqual([(x.frame, x.bbox) for x in left], [(x.frame, x.bbox) for x in right])

    def test_perfect_tracks_score_one(self):
        gt = [
            Detection(0, 7, (0, 0, 2, 2)),
            Detection(1, 7, (1, 0, 3, 2)),
        ]
        pred = [
            Detection(0, 12, (0, 0, 2, 2)),
            Detection(1, 12, (1, 0, 3, 2)),
        ]
        values = hota_metrics(hota_raw(gt, pred))["averaged"]
        for key in ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "AssRe", "AssPr", "LocA"):
            self.assertAlmostEqual(values[key], 1.0)

    def test_all_missed_and_all_false_are_not_perfect(self):
        gt = [Detection(0, 1, (0, 0, 2, 2))]
        missed = hota_metrics(hota_raw(gt, []))["averaged"]
        false = hota_metrics(hota_raw([], [Detection(0, 1, (0, 0, 2, 2))]))["averaged"]
        self.assertEqual(missed["HOTA"], 0.0)
        self.assertEqual(missed["DetA"], 0.0)
        self.assertEqual(false["HOTA"], 0.0)
        self.assertEqual(false["DetA"], 0.0)

    def test_identity_switch_reduces_association_not_detection(self):
        gt = [Detection(frame, 1, (frame, 0, frame + 2, 2)) for frame in range(4)]
        pred = [
            Detection(0, 10, (0, 0, 2, 2)),
            Detection(1, 10, (1, 0, 3, 2)),
            Detection(2, 11, (2, 0, 4, 2)),
            Detection(3, 11, (3, 0, 5, 2)),
        ]
        values = hota_metrics(hota_raw(gt, pred))["averaged"]
        self.assertAlmostEqual(values["DetA"], 1.0)
        self.assertLess(values["AssA"], 1.0)
        self.assertLess(values["HOTA"], 1.0)

    def test_duplicate_prediction_reduces_detection_accuracy(self):
        gt = [Detection(0, 1, (0, 0, 2, 2))]
        pred = [
            Detection(0, 10, (0, 0, 2, 2)),
            Detection(0, 11, (0, 0, 2, 2)),
        ]
        values = hota_metrics(hota_raw(gt, pred))["averaged"]
        self.assertLess(values["DetA"], 1.0)
        self.assertEqual(values["DetRe"], 1.0)

    def test_silent_gt_gap_breaks_miss_run(self):
        gt = [
            Detection(0, 1, (0, 0, 2, 2)),
            Detection(2, 1, (2, 0, 4, 2)),
        ]
        self.assertEqual(longest_miss_per_track_ms(gt, []), [50.0])

    def test_hungarian_matches_bruteforce(self):
        rng = np.random.default_rng(4)
        for rows in range(1, 4):
            for cols in range(rows, 5):
                matrix = rng.uniform(0.0, 10.0, size=(rows, cols))
                actual_rows, actual_cols = linear_sum_assignment(matrix)
                actual = float(matrix[actual_rows, actual_cols].sum())
                expected = min(
                    sum(matrix[row, columns[row]] for row in range(rows))
                    for columns in permutations(range(cols), rows)
                )
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
