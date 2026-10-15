"""Direction-7 sparse arithmetic, lineage, adapter and recovery regression tests."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / 'software'))
from wangdao.fixed import load_fixed_proposal_snn
from wangdao.h2.numpy_model import load_export, quantized_mlp
from wangdao.h2pro_proposal_transfer_v1.common import lineage, lock_directory
from wangdao.h2pro_proposal_transfer_v1.core import Cost, EventFixedProposal, sparse_ann, sparse_dot
from wangdao.h2pro_proposal_transfer_v1.workflow import context_delta, point, summarize, transition


class ProposalTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed, _, _ = load_fixed_proposal_snn(ROOT / 'software/weights/S1P_INT8_ACC24_MEM24_Q12.npz')
        cls.h2 = load_export(ROOT / 'software/weights/H2_D1_seed42.npz')

    def test_lineage_same_ann_proposal_and_pretrained_snn(self):
        a = lineage()
        self.assertTrue(a['same_original_ann_proposal'])
        self.assertEqual(a['legacy_s0p_trained_epochs'], 20)
        self.assertFalse(a['new_weights_updated'])

    def test_sparse_signed_mac_matches_dense_and_counts_nonzeros(self):
        rng = np.random.default_rng(7)
        x = rng.integers(0, 256, (19, 128), dtype=np.uint8)
        x[rng.random(x.shape) < .8] = 0
        w = rng.integers(-128, 128, (128, 48), dtype=np.int8)
        a, count = sparse_dot(x, w)
        np.testing.assert_array_equal(a, x.astype(np.int64) @ w.astype(np.int64))
        self.assertEqual(count, np.count_nonzero(x) * 48)

    def test_binary_spike_add_has_exact_signed_fanout(self):
        x = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], np.uint8)
        w = np.array([[-128, 127], [127, -128], [0, 3]], np.int8)
        a, count = sparse_dot(x, w, binary=True)
        np.testing.assert_array_equal(a, x.astype(np.int64) @ w.astype(np.int64))
        self.assertEqual(count, 6)

    def test_nonbinary_values_cannot_be_reported_as_ac(self):
        with self.assertRaisesRegex(ValueError, 'binary'):
            sparse_dot(np.array([[2, 0]], np.uint8), np.ones((2, 1), np.int8), binary=True)

    def test_sparse_input_contract_and_empty_batch(self):
        for x, w in [(np.zeros((3, 4), np.float32), np.zeros((4, 2), np.int8)),
                     (np.zeros((3, 4), np.uint8), np.zeros((3, 2), np.int8))]:
            with self.assertRaises(ValueError):
                sparse_dot(x, w)
        out, count = sparse_dot(np.empty((0, 4), np.uint8), np.ones((4, 2), np.int8))
        self.assertEqual(out.shape, (0, 2))
        self.assertEqual(count, 0)

    def test_ann_execution_control_is_bit_exact(self):
        rng = np.random.default_rng(8)
        for x in (np.zeros((9, 128), np.uint8), np.full((9, 128), 255, np.uint8),
                  rng.integers(0, 256, (27, 128), dtype=np.uint8)):
            actual, count = sparse_ann(x, self.h2.proposal_layers)
            np.testing.assert_array_equal(actual, quantized_mlp(x, self.h2.proposal_layers))
            self.assertLessEqual(count, len(x) * 7680)

    def test_s1_golden_rounding_reset_and_saturation_exact(self):
        rng = np.random.default_rng(71)
        model = EventFixedProposal(self.fixed)
        for x in (np.zeros((4, 13, 128), np.uint8), np.full((4, 13, 128), 255, np.uint8),
                  rng.integers(0, 256, (4, 37, 128), dtype=np.uint8)):
            actual, expected = model.forward(x), self.fixed.forward(x)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(model.mac, 48 * np.count_nonzero(x))
            self.assertEqual(model.ac, 24 * actual.spike_counts[0] + 16 * actual.spike_counts[1])

    def test_s1_batch_partition_and_state_reset(self):
        x = np.random.default_rng(9).integers(0, 256, (4, 17, 128), dtype=np.uint8)
        model = EventFixedProposal(self.fixed)
        full = model.forward(x)
        cost = (model.mac, model.ac)
        first = model.forward(x[:, :7])
        partial = (model.mac, model.ac)
        second = model.forward(x[:, 7:])
        np.testing.assert_array_equal(full.logits_q, np.concatenate([first.logits_q, second.logits_q]))
        self.assertEqual(cost, (partial[0] + model.mac, partial[1] + model.ac))
        np.testing.assert_array_equal(full.logits_q, model.forward(x).logits_q)

    def test_s1_rejects_float_input(self):
        with self.assertRaises(ValueError):
            EventFixedProposal(self.fixed).forward(np.zeros((4, 1, 128), np.float32))

    @staticmethod
    def fixture():
        candidate = np.array([True, False, True, True], bool)
        output = {'candidate': candidate, 'probabilities': np.array([.8, 0., .2, .6], np.float32),
                  'features': np.zeros((3, 36), np.uint8), 'heatmaps': np.zeros((1, 34, 44), np.float32),
                  'window_events': np.array([4], np.int64), 'window_candidates': np.array([3], np.int64)}
        arrays = {'label': np.array([1, 1, 0, 0], np.uint8), 'candidate': candidate,
                  'teacher': output['probabilities']}
        row = {'sequence': 'train/train_002.npz', 'cache_sha256': 'fixture',
               'source_sha256': 'source', 'high_density': False}
        meta = {'closed_windows': 1, 'trailing_events': 2}
        return output, arrays, row, meta

    def test_sequence_resume_reuses_payload_without_inference(self):
        output, arrays, row, meta = self.fixture()
        executor = SimpleNamespace(cost=Cost(proposal_mac=100, refiner_mac=3 * 4480),
                                   saturations={'accumulator': [0, 0, 0], 'membrane': [0, 0, 0, 0]})
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/runs') as name:
            first, _ = point(name, row, arrays, meta, 'ann_dense', None,
                             inference=lambda *_: (output, executor))
            def unexpected(*_):
                self.fail('completed sequence must not rerun inference')
            second, saved = point(name, row, arrays, meta, 'ann_dense', None, inference=unexpected)
            self.assertEqual(first, second)
            np.testing.assert_array_equal(saved['features'], output['features'])
            self.assertEqual(second['counts']['proposal_fn'], 1)
            self.assertEqual(second['counts']['events'], 4)

    def test_resume_detects_payload_corruption(self):
        output, arrays, row, meta = self.fixture()
        executor = SimpleNamespace(cost=Cost(), saturations={'accumulator': [0, 0, 0], 'membrane': [0, 0, 0, 0]})
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/runs') as name:
            point(name, row, arrays, meta, 'ann_dense', None, inference=lambda *_: (output, executor))
            payload = Path(name) / 'ann_dense/train_002.npz'
            payload.write_bytes(payload.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                point(name, row, arrays, meta, 'ann_dense', None)

    def test_cost_aggregation_does_not_mix_reference_or_ideal_ac(self):
        output, arrays, row, meta = self.fixture()
        executor = SimpleNamespace(cost=Cost(proposal_mac=100, refiner_mac=200, proposal_ac=40,
                                             verification_dense_proposal_mac=9999, hidden1_spikes=2),
                                   saturations={'accumulator': [0, 0, 0], 'membrane': [0, 0, 0, 0]})
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/runs') as name:
            saved, _ = point(name, row, arrays, meta, 'ann_dense', None, inference=lambda *_: (output, executor))
        result = summarize('s1_event', [saved])
        self.assertEqual(result['executed_neural_mac'], 300)
        self.assertEqual(result['executed_neural_mac_plus_ac'], 340)
        self.assertEqual(result['ideal_snn_ac_if_sparse'], 48)
        self.assertEqual(result['cost']['verification_dense_proposal_mac'], 9999)

    def test_context_checks_common_event_mapping_and_base_invariance(self):
        output, _, _, _ = self.fixture()
        other = {k: v.copy() for k, v in output.items()}
        other['features'][1, 32] = 9
        self.assertEqual(context_delta(other, output)['history_context_rows_changed'], 1)
        other['features'][1, 3] = 9
        with self.assertRaises(AssertionError):
            context_delta(other, output)

    def test_target_candidate_losses_are_counted(self):
        output, arrays, _, _ = self.fixture()
        changed = {k: v.copy() for k, v in output.items()}
        changed['candidate'][0] = False
        changed['probabilities'][0] = 0
        delta = transition(arrays['label'], changed, output)
        self.assertEqual(delta['target_candidates_dropped'], 1)
        self.assertEqual(delta['teacher_tp_lost'], 1)

    def test_identity_change_cannot_overwrite_old_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/runs') as name:
            with lock_directory(name, {'version': 'one'}):
                pass
            with self.assertRaises(ValueError):
                with lock_directory(name, {'version': 'two'}):
                    pass

    def test_isolated_causal_adapter_and_legacy_encoder(self):
        run = subprocess.run([sys.executable, '-B', str(ROOT / 'software/scripts/h2pro_proposal_transfer.py'), 'check'],
                             cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                             env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout[-3000:])
        report = json.loads(run.stdout)
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['s1_rederived_from_s0p_exact'])
        self.assertEqual(report['legacy_microbin_encoding_exact_windows'], 3)
        self.assertFalse(report['training_started'])


if __name__ == '__main__':
    unittest.main()
