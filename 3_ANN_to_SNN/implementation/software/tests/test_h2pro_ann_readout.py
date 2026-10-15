"""Causal and numerical contracts of the independently trained linear head."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/'software'))
from wangdao.h2pro_ann_state_v1.refiner import RefinerConfig as OldConfig, StreamRefiner as OldModel
from wangdao.h2pro_ann_state_v1.training import load_profile as old_profile, task_loss as old_loss, cost_report as old_cost
from wangdao.h2pro_ann_state_v1.probe import ReadoutProbe
from wangdao.h2pro_ann_readout_v1.refiner import RefinerConfig, StreamRefiner, normalized_linear_state
from wangdao.h2pro_ann_readout_v1.training import load_profile, task_loss, cost_report
from wangdao.h2pro_ann_readout_v1.monitor import HealthMonitor


class LinearReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def models(self):
        torch.manual_seed(42); old = OldModel(OldConfig(rows=2, columns=3))
        torch.manual_seed(42); new = StreamRefiner(RefinerConfig(rows=2, columns=3))
        return old, new

    def test_all_training_options_guards_and_initial_parameters_match(self):
        old, new = old_profile(), load_profile()
        self.assertEqual(old['training'], new['training'])
        self.assertEqual(old['execution'], new['execution'])
        om, nm = old['model'].copy(), new['model'].copy()
        om.pop('version'); nm.pop('version'); self.assertEqual(om, nm)
        a, b = self.models()
        for k, v in a.state_dict().items(): torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        self.assertEqual(a.architecture()['parameters'], b.architecture()['parameters'])
        self.assertEqual(a.architecture()['state_scalars_per_stream'], b.architecture()['state_scalars_per_stream'])

    def test_linear_update_preserves_both_signs_and_has_no_clamp(self):
        previous = torch.tensor([-1000., -2., 0., 2., 1000.])
        current = torch.tensor([-3000., 3., 0., -3., 3000.])
        z = normalized_linear_state(previous, current, .6)
        torch.testing.assert_close(normalized_linear_state(-previous, -current, .6), -z, rtol=0, atol=0)
        torch.testing.assert_close(z, previous*.6+current*.4, rtol=0, atol=0)
        self.assertGreater(float(z.abs().max()), 1000.)

    def test_constant_current_has_unit_steady_gain_for_either_sign(self):
        d = math.exp(-.5)
        current = torch.tensor([-3., 2.])
        z = torch.zeros_like(current)
        for _ in range(100): z = normalized_linear_state(z, current, d)
        torch.testing.assert_close(z, current, rtol=1e-6, atol=1e-6)

    def test_linear_gradient_has_correct_gain_and_temporal_decay_for_both_signs(self):
        d = .6
        initial = torch.tensor([-.7, .7], requires_grad=True)
        currents = torch.tensor([[-2., 2.], [3., -3.], [-1., 1.]], requires_grad=True)
        z = initial
        for current in currents: z = normalized_linear_state(z, current, d)
        z.sum().backward()
        torch.testing.assert_close(initial.grad, torch.full((2,), d**3))
        torch.testing.assert_close(currents.grad, torch.tensor([[(1-d)*d**2]*2, [(1-d)*d]*2, [1-d]*2]))

    def test_ema_stays_in_convex_hull_of_initial_state_and_input_history(self):
        torch.manual_seed(42)
        x = torch.randn(12, 7)*20
        z = torch.randn(7)
        low, high = z.clone(), z.clone()
        for current in x:
            low, high = torch.minimum(low, current), torch.maximum(high, current)
            z = normalized_linear_state(z, current, .6065306597)
            self.assertTrue(((z >= low-1e-5) & (z <= high+1e-5)).all())

    def test_stem_hidden_states_rates_costs_and_currents_are_exactly_unchanged(self):
        old, new = self.models()
        x = torch.rand(4, 2, 64, 2, 3)
        old_currents, new_currents = [], []
        h1 = old.head.register_forward_hook(lambda m, args, output: old_currents.append(output.detach().clone()))
        h2 = new.head.register_forward_hook(lambda m, args, output: new_currents.append(output.detach().clone()))
        _, a, ar, ac = old(x); _, b, br, bc = new(x)
        h1.remove(); h2.remove()
        for u, v in [(a[0], b[0]), (a[1], b[1]), (a[3], b[3]), (ar, br), (ac, bc), *zip(old_currents, new_currents)]:
            torch.testing.assert_close(u, v, rtol=0, atol=0)

    def test_readout_matches_independent_manual_recurrence_bias_and_class_pool(self):
        _, model = self.models()
        x = torch.rand(4, 2, 64, 2, 3)
        currents = []
        hook = model.head.register_forward_hook(lambda m, args, output: currents.append(output.detach().clone()))
        output, state, _, _ = model(x); hook.remove()
        z = torch.zeros_like(state[2]); expected = []
        extrema = []
        for current in currents:
            z = z*model.head_decay + current*(1-model.head_decay)
            raw = z.reshape(2, 32, 4, 2, 3)+model.bias[None, :, :, None, None]
            classes = raw.reshape(2, 32, 2, 2, 2, 3).amax(3)
            expected.append(classes[:, :, 1]-classes[:, :, 0])
            extrema.append(torch.stack((z.amin((1,2,3)), z.amax((1,2,3))), -1))
        torch.testing.assert_close(output, torch.stack(expected), rtol=0, atol=0)
        torch.testing.assert_close(state[2], z, rtol=0, atol=0)
        torch.testing.assert_close(model.last_readout_extrema, torch.stack(extrema), rtol=0, atol=0)

    def test_frozen_counterfactual_matches_up_to_documented_fp32_operation_order(self):
        old, new = self.models()
        probe = ReadoutProbe(old.config, .1, analog=True); probe.load_state_dict(old.state_dict())
        x = torch.rand(5, 1, 64, 2, 3)
        maps = np.array([[0,1,2,3,4], [0]*5, [0,1,4,16,31], [0,1,0,1,0], [0,1,2,0,1]])
        with torch.no_grad():
            expected = probe.probe(x, maps)[0]['normalized_leaky_no_reset']
            output = new(x)[0][tuple(maps)]
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)

    def test_chunking_future_isolation_sequence_resets_and_tbptt_detach(self):
        _, m = self.models()
        x = torch.rand(4, 2, 64, 2, 3)
        whole, state, _, _ = m(x)
        a, s, _, _ = m(x[:2]); b, sb, _, _ = m(x[2:], s)
        torch.testing.assert_close(torch.cat((a,b)), whole, rtol=0, atol=0)
        for u,v in zip(sb,state): torch.testing.assert_close(u,v,rtol=0,atol=0)
        changed = x.clone(); changed[2:] *= 10
        torch.testing.assert_close(m(changed)[0][:2], whole[:2], rtol=0, atol=0)
        self.assertTrue(all(not v.requires_grad for v in m.detach_state(state)))
        cleared = m.reset_slots(state, torch.tensor([True,False]))
        for u,v in zip(state,cleared):
            self.assertTrue((v[0] == 0).all()); torch.testing.assert_close(v[1],u[1],rtol=0,atol=0)

    def test_bias_is_applied_once_and_zero_input_does_not_integrate_it(self):
        _, m = self.models()
        with torch.no_grad(): m.bias[:,2:] = -2.
        output, state, _, _ = m(torch.zeros(20, 1, 64, 2, 3))
        self.assertTrue((output == -2).all())
        self.assertTrue((state[2] == 0).all())

    def test_losses_match_numerically_given_identical_inputs(self):
        cfg = load_profile()['training']
        args = (torch.tensor([.1,-.2]),torch.tensor([1.,0.]),torch.tensor([.8,.1]),
                torch.ones(2,1,2),cfg,torch.ones(2,1,2))
        a, ap = old_loss(*args); b, bp = task_loss(*args)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(set(ap),set(bp))
        for k in ap: torch.testing.assert_close(ap[k],bp[k],rtol=0,atol=0)

    def test_both_head_signs_remain_in_tail_loss_and_gradients_are_finite(self):
        _, m = self.models()
        m.config = RefinerConfig(rows=2,columns=3,head_voltage_soft_limit=.001)
        m(torch.rand(3,1,64,2,3))
        tails = m.voltage_tail()
        for sign in range(2):
            gradient = torch.autograd.grad(tails[...,sign].mean(),m.head.weight,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()),0.)

    def test_positive_and_negative_amplitude_guards_are_not_removed(self):
        _, model = self.models()
        model.last_diagnostics = torch.zeros(1,1,3,7)
        model.last_readout_extrema = torch.tensor([[[-3.,65.]]])
        model.last_diagnostics[0,0,2,4] = 260.
        with self.assertRaisesRegex(FloatingPointError,'max_activation'):
            HealthMonitor().observe(model,torch.tensor([0.]),[True],[True])
        model.last_diagnostics.zero_(); model.last_diagnostics[0,0,2,2] = 1025.
        model.last_readout_extrema = torch.tensor([[[-1025.,3.]]])
        monitor = HealthMonitor()
        with self.assertRaisesRegex(FloatingPointError,'max_abs_voltage'):
            monitor.observe(model,torch.tensor([0.]),[True],[True])
        self.assertEqual(monitor.summary()['signed_head_state']['min'],-1025.)

    def test_same_dense_mac_and_explicit_extra_normalization_cost(self):
        old,new = self.models(); counts = np.ones(5)
        a,b = old_cost(old,counts,10),cost_report(new,counts,10)
        self.assertEqual(a['dense_backend_synaptic_mac_estimate'],b['dense_backend_synaptic_mac_estimate'])
        self.assertEqual(b['reference_readout_current_normalization_multiplications'],10*2*3*128)
        self.assertIsNone(b['continuous_activation_ac_estimate'])

    def test_old_version_is_rejected_by_new_config(self):
        with self.assertRaises(ValueError): RefinerConfig(version='phase_grid_ann_state_v1')


if __name__ == '__main__':
    unittest.main()
