"""Matched-state control and readout diagnostic invariants."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/'software'))
from wangdao.h2pro_snn_v2.neurons import lif
from wangdao.h2pro_snn_v2_2.refiner import RefinerConfig as SNNConfig, StreamRefiner as SNN
from wangdao.h2pro_snn_v2_2.training import task_loss as snn_loss
from wangdao.h2pro_ann_state_v1.refiner import RefinerConfig, StreamRefiner, analog_cell
from wangdao.h2pro_ann_state_v1.training import cost_report, load_profile, task_loss
from wangdao.h2pro_ann_state_v1.probe import ReadoutProbe, pool_prototypes
from wangdao.h2pro_ann_state_v1.audit import step


class AnalogStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_initial_parameters_buffers_and_state_size(self):
        torch.manual_seed(42); a = SNN()
        torch.manual_seed(42); b = StreamRefiner()
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        self.assertEqual(a.architecture()['parameters'], b.architecture()['parameters'])
        self.assertEqual(a.architecture()['state_scalars_per_stream'], b.architecture()['state_scalars_per_stream'])

    def test_only_rounding_changes_the_cell_forward(self):
        v = torch.tensor([-1., -.05, 0., .1, .25, .45, 75.125])
        n, sr = lif(v, .25)
        a, ar = analog_cell(v, .25)
        torch.testing.assert_close(n, a.floor(), rtol=0, atol=0)
        torch.testing.assert_close(ar, v.clamp_max(0), rtol=0, atol=0)
        torch.testing.assert_close(sr-ar, .25*(a-n), rtol=0, atol=0)
        self.assertGreater(float(a[-1]), 256)  # No hidden activation clamp.

    def test_continuous_cell_preserves_surrogate_and_detached_reset_gradient(self):
        v = torch.tensor([-1., 0., .1, .45], requires_grad=True)
        n, r = lif(v, .25)
        snn = torch.autograd.grad((n*.25+r*.7).sum(), v)[0]
        a, r = analog_cell(v, .25)
        ann = torch.autograd.grad((a*.25+r*.7).sum(), v)[0]
        torch.testing.assert_close(snn, ann, rtol=0, atol=0)
        torch.testing.assert_close(ann, torch.tensor([.7, .7, 1.7, 1.7]))

    def test_signed_negative_state_and_positive_reset_are_explicit(self):
        decay = .5
        v = torch.tensor([-1., 1.])
        state = torch.zeros_like(v)
        for _ in range(20):
            _, state = analog_cell(decay*state+v, .25)
        self.assertAlmostEqual(float(state[0]), -2., places=5)
        self.assertEqual(float(state[1]), 0.)

    def test_chunk_equivalence_and_reset(self):
        torch.manual_seed(42)
        m = StreamRefiner(RefinerConfig(rows=2, columns=3))
        x = torch.rand(4, 2, 64, 2, 3)
        whole, state, _, _ = m(x)
        a, s, _, _ = m(x[:2]); b, s, _, _ = m(x[2:], s)
        torch.testing.assert_close(torch.cat((a, b)), whole, rtol=0, atol=0)
        for u, v in zip(s, state): torch.testing.assert_close(u, v, rtol=0, atol=0)
        cleared = m.reset_slots(s, torch.tensor([True, False]))
        for u, v in zip(s, cleared):
            self.assertTrue((v[0] == 0).all())
            torch.testing.assert_close(u[1], v[1], rtol=0, atol=0)

    def test_loss_numeric_parity_and_honest_activity_label(self):
        cfg = load_profile()['training']
        z = torch.tensor([.1, -.2], requires_grad=True)
        args = (z, torch.tensor([1., 0.]), torch.tensor([.8, .1]), torch.ones(2, 1, 2), cfg, torch.ones(2, 1, 2))
        a, ap = snn_loss(*args); b, bp = task_loss(*args)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertNotIn('spike', bp)
        torch.testing.assert_close(ap['spike'], bp['activation'])

    def test_no_spike_addition_or_energy_claim_for_continuous_activations(self):
        m = StreamRefiner()
        r = cost_report(m, np.ones(5), 10)
        self.assertNotIn('ideal_unit_spike_ac_upper_bound', r)
        self.assertIsNone(r['continuous_activation_ac_estimate'])
        self.assertFalse(r['energy_measured'])

    def test_invalid_control_version_is_rejected(self):
        with self.assertRaises(ValueError): RefinerConfig(version='phase_grid_refiner_v2_2')
        with self.assertRaises(ValueError): RefinerConfig(threshold=0)

    def test_probe_exact_original_forward_and_state_for_both_cells(self):
        torch.manual_seed(42)
        x = torch.rand(3, 1, 64, 2, 3)
        maps = np.array([[0, 1, 2], [0, 0, 0], [0, 7, 31], [0, 1, 0], [0, 2, 1]])
        for analog, original in ((False, SNN(SNNConfig(rows=2, columns=3))),
                                  (True, StreamRefiner(RefinerConfig(rows=2, columns=3)))):
            probe = ReadoutProbe(original.config, .1, analog=analog)
            probe.load_state_dict(original.state_dict())
            with torch.no_grad():
                expected, state, _, _ = original(x)
                scores, ps, _, _, _, d, _ = probe.probe(x, maps)
            torch.testing.assert_close(expected[tuple(maps)], scores['original'], rtol=0, atol=0)
            for u, v in zip(state, ps): torch.testing.assert_close(u, v, rtol=0, atol=0)
            torch.testing.assert_close(scores['original'], d['current_margin']+d['carry_margin']+d['bias_margin'])

    def test_counterfactuals_preserve_bias_once_and_common_mode_cancels(self):
        v = torch.tensor([[-8., -4., 1., 2.]])
        b = torch.tensor([[0., 0., -2., -2.]])
        score, winners = pool_prototypes(v, b)
        self.assertEqual(float(score[0]), 4.)
        torch.testing.assert_close(pool_prototypes(v+10, b)[0], score)
        self.assertEqual(winners.tolist(), [[1, 3]])

    def test_probe_auxiliary_state_resets_with_sequence_boundaries(self):
        config = SNNConfig(rows=2, columns=3)
        m = ReadoutProbe(config, .1)
        batch = {'inputs': np.ones((2, 2, 64, 2, 3), np.float32), 'reset': np.array([True, True]),
                 'maps': np.array([[0, 1], [0, 1], [1, 3], [0, 1], [1, 2]])}
        with torch.no_grad():
            first = step(m, batch, None, None, torch.device('cpu'))
            reset = step(m, batch, first[1], first[2]+100., torch.device('cpu'))
        for k in first[0]: torch.testing.assert_close(first[0][k], reset[0][k], rtol=0, atol=0)

    def test_probe_chunked_readouts_match_whole_and_future_does_not_leak(self):
        m = ReadoutProbe(SNNConfig(rows=2, columns=3), .1)
        x = torch.rand(4, 1, 64, 2, 3)
        maps = np.array([[0, 1, 2, 3], [0]*4, [1]*4, [0]*4, [1]*4])
        whole = m.probe(x, maps)
        a = m.probe(x[:2], maps[:, :2])
        rest = maps[:, 2:].copy(); rest[0] -= 2
        b = m.probe(x[2:], rest, a[1], a[2])
        for k in whole[0]: torch.testing.assert_close(torch.cat((a[0][k], b[0][k])), whole[0][k], rtol=0, atol=0)
        changed = x.clone(); changed[2:] *= 10
        other = m.probe(changed, maps)
        for k in whole[0]: torch.testing.assert_close(other[0][k][:2], whole[0][k][:2], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
