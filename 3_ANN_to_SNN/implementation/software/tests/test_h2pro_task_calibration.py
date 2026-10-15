"""Task selection, sample isolation, unchanged arithmetic and restart evidence."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / 'software'))
from wangdao.h2pro_calibration_v1.core import Encoder, forward, load_layers, probabilities
from wangdao.h2pro_calibration_v1.common import protocol as previous_protocol
from wangdao.h2pro_task_calibration_v1.common import identity, inputs, protocol
from wangdao.h2pro_task_calibration_v1.calibration import (
    PreparedForward, sample_data, search_point, select_candidates, task_key, task_metrics,
)
from wangdao.h2pro_task_calibration_v1.workflow import evaluate, validate_selection


def logits(prediction):
    return np.array([[0., 0., 2. if p else -2., 2. if p else -2.] for p in prediction], np.float32)


def candidate(key, iou, ac, harm=0., mse=0.):
    return {'key': key, 'balanced_iou': iou, 'equivalent_ac': ac,
            'balanced_teacher_harm_rate': harm, 'balanced_margin_mse': mse}


class TaskCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers = load_layers()
        cls.x = np.random.default_rng(2026).integers(0, 256, (31, 36), dtype=np.uint8)

    def test_same_encoding_search_scope(self):
        for key in ('steps', 'stages', 'threshold_grid', 'initial_phase_quarters', 'coordinate_passes',
                    'refinement_passes', 'threshold_refinement_offsets', 'membrane_refinement_offsets',
                    'sample_seed', 'candidate_rows_per_sequence', 'decision_threshold'):
            self.assertEqual(protocol()[key], previous_protocol()[key])

    def test_cached_forward_exact_and_weights_unchanged(self):
        before = [a.copy() for pair in self.layers for a in pair]
        for t in (4, 8, 16):
            runner = PreparedForward(self.x, self.layers, t)
            for theta in (128, 247, 255, 271, 510):
                encoders = (Encoder(theta, theta - 1), Encoder(271, 134))
                for stage in ('r1', 'r2'):
                    raw, cost = forward(self.x, self.layers, stage, t, encoders, 'spike_add')
                    actual, stats = runner(stage, encoders)
                    np.testing.assert_array_equal(raw, actual)
                    self.assertEqual(cost.unit_spikes, stats['unit_spikes'])
                    self.assertEqual(cost.executed_synaptic_ac, stats['equivalent_ac'])
                    self.assertEqual(stats['equivalent_ac'], 32 * stats['layer1_spikes'] + 4 * stats['layer2_spikes'])
        for old, new in zip(before, [a for pair in self.layers for a in pair]):
            np.testing.assert_array_equal(old, new)

    def test_prepared_cache_bounded(self):
        runner = PreparedForward(self.x, self.layers, 4)
        for theta in range(200, 220):
            runner('r2', (Encoder(theta, 0), Encoder()))
            self.assertLessEqual(len(runner.second_cache), 8)

    def test_task_objective_takes_precedence_over_mse(self):
        low_mse = candidate('mse', .5, 50, mse=.01)
        high_iou = candidate('task', .8, 70, mse=100.)
        self.assertEqual(min([low_mse, high_iou], key=task_key), high_iou)

    def test_teacher_correct_harm_breaks_task_ties(self):
        good = candidate('good', .8, 100, harm=.01, mse=10)
        bad = candidate('bad', .8, 50, harm=.02, mse=0)
        self.assertEqual(min([good, bad], key=task_key), good)

    def test_ac_budget_selection_and_fallback(self):
        base = candidate('base', .5, 100)
        accurate = candidate('accurate', .9, 120)
        eligible = candidate('matched', .8, 99)
        best, matched = select_candidates([base, accurate, eligible], base)
        self.assertEqual(best, accurate)
        self.assertEqual(matched, eligible)
        self.assertEqual(select_candidates([base, accurate], base)[1], base)

    def test_proposal_misses_are_not_dropped(self):
        raw = logits([True, False, True, False])
        y = np.array([1, 0, 1, 0], np.uint8)
        receipts = [{'sampled_candidates': 2, 'available_candidates': 2, 'events': 4,
                     'proposal_fn': 1, 'proposal_tn': 1}] * 2
        metrics = task_metrics(raw, raw, y, np.array([0, 0, 1, 1], bool),
                               np.array([0, 0, 1, 1]), receipts)
        self.assertEqual(metrics['balanced_iou'], .5)
        for group in metrics['groups'].values():
            self.assertEqual(group['recall'], .5)
            self.assertEqual(group['estimated_normalized_confusion']['fn'], .25)

    def test_inverse_sampling_weight_applied_before_proposal_counts(self):
        raw = logits([True, False, True, False])
        y = np.array([1, 0, 1, 0], np.uint8)
        receipts = [{'sampled_candidates': 2, 'available_candidates': 20, 'events': 24,
                     'proposal_fn': 2, 'proposal_tn': 2}] * 2
        metrics = task_metrics(raw, raw, y, np.array([0, 0, 1, 1], bool),
                               np.array([0, 0, 1, 1]), receipts)
        self.assertAlmostEqual(metrics['balanced_iou'], 10 / 12)

    def test_labels_change_task_score_without_changing_inference(self):
        raw = logits([True, False, True, False])
        receipts = [{'sampled_candidates': 2, 'available_candidates': 2, 'events': 2,
                     'proposal_fn': 0, 'proposal_tn': 0}] * 2
        groups = np.array([0, 0, 1, 1], bool)
        seqs = np.array([0, 0, 1, 1])
        a = task_metrics(raw, raw, np.array([1, 0, 1, 0]), groups, seqs, receipts)
        b = task_metrics(raw, raw, np.array([0, 1, 0, 1]), groups, seqs, receipts)
        self.assertEqual(a['balanced_iou'], 1.)
        self.assertEqual(b['balanced_iou'], 0.)

    def test_sampling_reuses_old_rows_and_aligns_candidate_labels(self):
        c = np.array([False, True, False, True, True], bool)
        y = np.array([1, 0, 0, 1, 0], np.uint8)
        data = {'features': self.x[:3], 'candidate': c, 'label': y}
        old = [{'sequence': 'train/a.npz', 'high_density': False, 'available_candidates': 3,
                'sampled_candidates': 3,
                'selected_row_indices_sha256': hashlib.sha256(np.arange(3, dtype='<i8').tobytes()).hexdigest()}]
        with patch('wangdao.h2pro_task_calibration_v1.calibration.load_sequence', return_value=data):
            x, labels, _, _, receipts = sample_data('.', {'sequences': old}, old)
            np.testing.assert_array_equal(x, self.x[:3])
            np.testing.assert_array_equal(labels, y[c])
            self.assertEqual(receipts[0]['proposal_fn'], 1)
            wrong = copy.deepcopy(old)
            wrong[0]['selected_row_indices_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'exactly'):
                sample_data('.', {'sequences': old}, wrong)

    def test_selection_rejects_dev_role(self):
        with patch('wangdao.h2pro_task_calibration_v1.common.calibration_manifest', side_effect=ValueError('role')):
            with self.assertRaises(ValueError):
                inputs('.', '.')

    def test_small_search_keeps_old_pool_control_and_sample_ac_budget(self):
        settings = {**protocol(), 'threshold_grid': [192, 255, 320], 'initial_phase_quarters': [1, 2],
                    'coordinate_passes': 1, 'refinement_passes': 1,
                    'threshold_refinement_offsets': [-1, 0, 1], 'membrane_refinement_offsets': [-1, 0, 1]}
        encoders = [{'threshold': 255, 'initial_membrane': 127}] * 2
        previous = {'stage': 'r2', 'steps': 4, 'candidates': [{'encoders': encoders}],
                    'accuracy': {'encoders': encoders}}
        groups = np.arange(len(self.x)) >= 15
        seqs = groups.astype(np.int32)
        receipts = [{'sampled_candidates': n, 'available_candidates': n, 'events': n + 1,
                     'proposal_fn': 1, 'proposal_tn': 0} for n in (15, 16)]
        with patch('wangdao.h2pro_task_calibration_v1.calibration.protocol', return_value=settings), contextlib.redirect_stdout(io.StringIO()):
            args = (self.x, (np.arange(len(self.x)) % 3 == 0).astype(np.uint8), groups, seqs,
                    receipts, self.layers, previous)
            first, second = search_point(*args), search_point(*args)
        first.pop('calibration_seconds')
        second.pop('calibration_seconds')
        self.assertEqual(first, second)
        self.assertEqual(first['old_pool_candidates'], 1)
        self.assertEqual(first['task_rescore'], first['uncalibrated'])
        self.assertLessEqual(first['task_matched_ac']['equivalent_ac'], first['uncalibrated']['equivalent_ac'])
        self.assertTrue(first['prepared_reference_exact'])

    def selection(self):
        points = []
        for t in protocol()['steps']:
            for stage in protocol()['stages']:
                value = {'encoders': [{'threshold': 255, 'initial_membrane': 127}] * 2, 'equivalent_ac': 1}
                points.append({'stage': stage, 'steps': t, 'prepared_reference_exact': True,
                               **{m: copy.deepcopy(value) for m in protocol()['modes']}})
        return {'status': 'complete', 'data_role': 'train_main', 'labels_used': True,
                'weights_updated': False, 'dev_used_for_selection': False,
                'identity': identity('task_encoder_calibration'), 'points': points, 'sample': []}

    def test_dev_evaluation_requires_frozen_complete_selection(self):
        original = self.selection()
        validate_selection(original)
        for key, value in (('labels_used', False), ('data_role', 'train_dev'), ('dev_used_for_selection', True)):
            bad = {**original, key: value}
            with self.assertRaises(ValueError):
                validate_selection(bad)
        bad = copy.deepcopy(original)
        bad['points'].pop()
        with self.assertRaises(ValueError):
            validate_selection(bad)
        bad = copy.deepcopy(original)
        bad['points'][0]['task_matched_ac']['equivalent_ac'] = 2
        with self.assertRaises(ValueError):
            validate_selection(bad)

    def test_full_workflow_resume_does_not_repeat_inference(self):
        raw, _ = forward(self.x, self.layers, 'ann', 255)
        arrays = {'features': self.x, 'candidate': np.ones(len(self.x), bool),
                  'label': (np.arange(len(self.x)) % 3 == 0).astype(np.uint8),
                  'teacher': probabilities(raw), 'logits': raw}
        rows = [{'sequence': 'train/a.npz', 'cache_sha256': 'test', 'high_density': False,
                 'proposal_synaptic_mac': 100},
                {'sequence': 'train/b.npz', 'cache_sha256': 'test', 'high_density': True,
                 'proposal_synaptic_mac': 100}]
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/runs') as name:
            base = Path(name)
            (base / 'fit').mkdir()
            (base / 'cache').mkdir()
            (base / 'cache/manifest.json').write_text('{}', encoding='utf-8')
            (base / 'fit/calibration.json').write_text(json.dumps(self.selection()), encoding='utf-8')
            with patch('wangdao.h2pro_task_calibration_v1.workflow.checks', return_value={'status': 'passed'}), \
                 patch('wangdao.h2pro_task_calibration_v1.workflow.cache_manifest', return_value={'sequences': rows}), \
                 patch('wangdao.h2pro_calibration_v1.workflow.load_sequence', return_value=arrays), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = evaluate(base / 'cache', base / 'fit', base / 'dev', 13)
            self.assertEqual(len(first['results']), 31)
            self.assertTrue(all(r['reference_logits_exact'] for r in first['results']))
            with patch('wangdao.h2pro_task_calibration_v1.workflow.cache_manifest', return_value={'sequences': rows}), \
                 patch('wangdao.h2pro_task_calibration_v1.workflow.evaluate_point', side_effect=AssertionError('already complete')):
                second = evaluate(base / 'cache', base / 'fit', base / 'dev', 13)
            self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
