"""Signed-peak loss contracts, isolation and causal-gradient checks."""
import copy
from pathlib import Path
import sys
import unittest
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from wangdao.h2pro_snn.training import task_loss as old_loss
from wangdao.h2pro_snn_v2.refiner import RefinerConfig as OldConfig, StreamRefiner as OldRefiner
from wangdao.h2pro_snn_v2_1.foundation import research_sources as old_sources
from wangdao.h2pro_snn_v2_2.foundation import research_sources
from wangdao.h2pro_snn_v2_2.refiner import RefinerConfig, StreamRefiner, signed_voltage_tail
from wangdao.h2pro_snn_v2_2.training import load_profile, task_loss


class SignedHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_signs_have_correct_inward_gradients_and_zero_dead_zone(self):
        v = torch.tensor([[-32., -16., 0., 16., 64.]], requires_grad=True)
        tails = signed_voltage_tail(v, 16.)
        torch.testing.assert_close(tails, torch.tensor([[9., 1.]]))
        tails.sum().backward()
        torch.testing.assert_close(v.grad, torch.tensor([[-.125, 0., 0., 0., .375]]))

    def test_empty_cells_do_not_dilute_one_extreme(self):
        compact = torch.tensor([[32., -64.]])
        sparse = torch.cat((compact, torch.zeros(1, 10000)), 1)
        torch.testing.assert_close(signed_voltage_tail(compact, 16.), signed_voltage_tail(sparse, 16.))

    def test_streams_are_separate_and_ties_share_gradient(self):
        v = torch.tensor([[32., 32.], [-64., 0.]], requires_grad=True)
        t = signed_voltage_tail(v, 16.)
        torch.testing.assert_close(t, torch.tensor([[1., 0.], [0., 9.]]))
        t[0].sum().backward()
        torch.testing.assert_close(v.grad, torch.tensor([[.0625, .0625], [0., 0.]]))

    def test_invalid_limits_and_nonfinite_values_are_not_silently_clipped(self):
        for value in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError): signed_voltage_tail(torch.ones(1, 2), value)
        self.assertFalse(torch.isfinite(signed_voltage_tail(torch.tensor([[float('inf')]]), 16.)).all())
        self.assertEqual(float(signed_voltage_tail(torch.tensor([[1600.]]), 16.)[0, 0]), 9801.)

    def test_new_forward_state_initialization_and_base_gradients_match_v2_exactly(self):
        args: dict[str, Any] = dict(rows=2, columns=3, stem_channels=4, hidden_channels=6, local_channels=3)
        torch.manual_seed(42); a = OldRefiner(OldConfig(**args))
        torch.manual_seed(42); b = StreamRefiner(RefinerConfig(**args))
        for k, v in a.state_dict().items(): torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        x = torch.rand(4, 2, 64, 2, 3)
        ya, sa, ra, ca = a(x); yb, sb, rb, cb = b(x)
        for u, v in [(ya, yb), (ra, rb), (ca, cb), *zip(sa, sb)]:
            torch.testing.assert_close(u, v, rtol=0, atol=0)
        ya.square().mean().backward(); yb.square().mean().backward()
        for u, v in zip(a.parameters(), b.parameters()): torch.testing.assert_close(u.grad, v.grad, rtol=0, atol=0)

    def test_continuous_chunks_and_future_isolation_include_regularizer(self):
        torch.manual_seed(42)
        model = StreamRefiner(RefinerConfig(rows=2, columns=3, head_voltage_soft_limit=.1))
        x = torch.rand(4, 1, 64, 2, 3)
        whole, _, _, _ = model(x); tail = model.voltage_tail().clone()
        first, state, _, _ = model(x[:2]); t1 = model.voltage_tail().clone()
        second, _, _, _ = model(x[2:], state); t2 = model.voltage_tail().clone()
        torch.testing.assert_close(torch.cat((first, second)), whole, rtol=0, atol=0)
        torch.testing.assert_close(torch.cat((t1, t2)), tail, rtol=0, atol=0)
        changed = x.clone(); changed[2:] *= 10
        model(changed)
        torch.testing.assert_close(model.voltage_tail()[:2], tail[:2], rtol=0, atol=0)

    def test_extra_loss_is_exactly_weighted_sign_sum_with_inactive_stream_excluded(self):
        cfg = load_profile()['training']
        z = torch.tensor([-.3, .6], requires_grad=True)
        labels = torch.tensor([0., 1.]); teacher = torch.tensor([.1, .8]); rates = torch.ones(2, 1, 2)*.1
        tails = torch.tensor([[[2., 3.], [1e6, 1e6]], [[4., 5.], [1e6, 1e6]]], requires_grad=True)
        active = np.array([True, False])
        base, _ = old_loss(z, labels, teacher, rates, cfg)
        total, parts = task_loss(z, labels, teacher, rates, cfg, tails[:, active])
        torch.testing.assert_close(total, base + cfg['head_voltage_weight']*7.)
        self.assertEqual(float(parts['head_voltage_positive']), 3.)
        self.assertEqual(float(parts['head_voltage_negative']), 4.)
        total.backward()
        assert tails.grad is not None
        self.assertTrue((tails.grad[:, 1] == 0).all())
        zero = copy.deepcopy(cfg); zero['head_voltage_weight'] = 0
        torch.testing.assert_close(task_loss(z, labels, teacher, rates, zero, tails[:, active])[0], base, rtol=0, atol=0)

    def test_regularizer_reaches_head_and_both_prior_layers(self):
        torch.manual_seed(42)
        model = StreamRefiner(RefinerConfig(rows=2, columns=3, head_voltage_soft_limit=.1))
        model(torch.rand(3, 1, 64, 2, 3))
        model.voltage_tail().sum().backward()
        for layer in (model.stem, model.hidden, model.head):
            assert layer.weight.grad is not None
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.norm()), 0.)
        self.assertIsNone(model.bias.grad)  # One-time readout bias is not membrane state.

    def test_new_version_retains_v21_source_identity_and_all_hard_limits(self):
        previous, current = old_sources(), research_sources()
        self.assertTrue(all(current[k] == v for k, v in previous.items()))
        self.assertTrue(all('v2_2' not in k for k in previous))
        cfg = load_profile()
        self.assertEqual(cfg['training']['stability_limits']['max_unit_burst'], 256.)
        self.assertEqual(cfg['model']['head_voltage_soft_limit'], 16.)
        self.assertEqual(StreamRefiner().architecture()['parameters'], 9856)


if __name__ == '__main__':
    unittest.main()
