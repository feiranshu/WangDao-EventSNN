"""Frozen inputs, reserved-split exclusion and diagnostic attribution."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from wangdao.h2pro_snn.foundation import (ANN_ROOT, PACKAGE, audit, development_index,
    role_records, verified_sequence, verify_baseline, write_report)
from wangdao.h2pro_snn.costs import OperationLedger, profile_exact_mlp
from wangdao.h2pro_snn.diagnostics import (aggregate_counts, error_counts, feature_collisions,
                                          summarize_counts, temporal_support)


class FoundationTests(unittest.TestCase):
    def test_frozen_inputs_and_training_manual_only(self):
        result = audit()
        self.assertEqual(result['baseline']['model_id'], 'H2-pro-Causal-ANN-v1')
        self.assertEqual(result['baseline']['threshold'], 0.35)
        self.assertTrue(result['protocol']['training_enabled'])
        self.assertEqual(result['protocol']['training_policy']['launcher'], 'user_only')
        self.assertFalse(result['protocol']['training_policy']['automatic_training'])
        self.assertFalse(result['training_started'])
        self.assertEqual(result['reserved_sequences_read'], 0)

    def test_changed_teacher_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').mkdir()
            (root/'data/index.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_baseline(root)

    def test_only_development_roles_are_exposed(self):
        self.assertEqual(len(role_records('train_main')), 79)
        self.assertEqual(len(role_records('train_dev')), 10)
        for role in ('test', 'validation', 'val', 'hw_calibration', 'all', '../train'):
            with self.subTest(role=role), self.assertRaises(ValueError):
                role_records(role)

    def test_invalid_limits(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                role_records('train_dev', limit)

    def test_calibration_record_cannot_bypass_role_guard(self):
        index = development_index()
        name = 'train/' + index['roles']['hw_calibration'][0]
        record = next(r for r in index['files'] if r['path'] == name)
        with self.assertRaisesRegex(ValueError, 'development record'):
            verified_sequence(Path('does-not-exist'), record)

    def test_reports_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            write_report(path, {'status': 'one'})
            with self.assertRaises(FileExistsError):
                write_report(path, {'status': 'two'})
            self.assertEqual(json.loads(path.read_text())['status'], 'one')

    def test_full_event_false_negative_partition(self):
        counts = error_counts([1, 1, 1, 0, 0], [0., .2, .9, .8, 0.], np.array([0, 1, 1, 1, 0], bool))
        result = summarize_counts(counts)
        self.assertEqual((result['proposal_fn'], result['refiner_fn'], result['fn']), (1, 1, 2))
        self.assertEqual(result['proposal_target_recall'], 2/3)
        self.assertEqual(result['refiner_conditional_recall'], .5)
        self.assertEqual(result['iou'], .25)
        self.assertEqual(aggregate_counts([counts, counts])['iou'], result['iou'])

    def test_empty_metrics_are_unknown_not_perfect(self):
        result = aggregate_counts([])
        self.assertIsNone(result['iou'])
        self.assertIsNone(result['proposal_target_recall'])

    def test_rejected_target_is_not_omitted(self):
        result = summarize_counts(error_counts([1], [0.], np.array([False])))
        self.assertEqual(result['iou'], 0.)
        self.assertEqual(result['proposal_fn'], 1)
        with self.assertRaises(ValueError):
            error_counts([1], [.9], np.array([False]))

    def test_invalid_probabilities(self):
        for p in (np.nan, np.inf, -.1, 1.1):
            with self.subTest(p=p), self.assertRaises(ValueError):
                error_counts([1], [p], np.array([True]))

    def test_mixed_label_feature_collisions(self):
        features = np.array([[0, 0], [0, 0], [1, 2], [1, 2], [2, 3]], np.uint8)
        result = feature_collisions(features, np.array([0, 1, 1, 1, 0]))
        self.assertEqual(result['conflicting_codes'], 1)
        self.assertEqual(result['events_in_conflicting_codes'], 2)

    def test_neural_mac_counts_include_zero_weights(self):
        features = np.array([[0, 2], [3, 0]], np.uint8)
        weights = np.array([[1, 0, 1], [0, -1, 0]], np.int8)
        scales = np.full(3, 255., np.float32)
        logits, cost = profile_exact_mlp(features, ((weights, scales),))
        np.testing.assert_array_equal(logits, [[0, -2, 0], [3, 0, 3]])
        self.assertEqual(cost.dense_neural_mac, 12)
        self.assertEqual(cost.zero_skip_neural_mac_proxy, 3)
        self.assertFalse(cost.report('neural', ['features'])['end_to_end_complete'])

    def test_multilayer_quantized_replay(self):
        from wangdao.h2.numpy_model import quantized_mlp
        rng = np.random.default_rng(1)
        features = rng.integers(0, 256, (17, 5), dtype=np.uint8)
        layers = tuple((rng.integers(-127, 128, shape, dtype=np.int8),
                        rng.uniform(.001, .2, shape[1]).astype(np.float32))
                       for shape in ((5, 7), (7, 4)))
        actual, _ = profile_exact_mlp(features, layers)
        np.testing.assert_array_equal(actual, quantized_mlp(features, layers))

    def test_ledger_rejects_invalid_counts_atomically(self):
        ledger = OperationLedger()
        for bad in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                ledger.add(ac=2, exp=bad)
        self.assertEqual(ledger.ac, 0)

    def test_temporal_support_uses_complete_bins(self):
        events = np.array([(0.,), (9.,), (10.,), (99.,)], dtype=[('t', 'f8')])
        result = temporal_support(events, np.ones(4), 0., 100.)
        self.assertEqual(result['10ms']['target_occupied_bins'], 3)
        self.assertEqual(result['100ms']['target_occupied_bins'], 1)


if __name__ == '__main__':
    unittest.main()
