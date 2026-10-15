"""Numerical contracts and recovery checks; no real optimizer update or fitting."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from test_h2pro_refiner import FakeStore, sequence
from wangdao.h2pro_snn.foundation import research_sources as v1_sources, sha256
from wangdao.h2pro_snn.stream_data import atomic_json, stream_batches
from wangdao.h2pro_snn.training import checkpoint_load, cpu_weights
from wangdao.h2pro_snn_v2.foundation import research_sources
from wangdao.h2pro_snn_v2.monitor import HealthMonitor
from wangdao.h2pro_snn_v2.neurons import burst, lif
from wangdao.h2pro_snn_v2.refiner import RefinerConfig, StreamRefiner
from wangdao.h2pro_snn_v2.training import (cost_report, evaluate, evaluate_checkpoint, load_profile,
                                        logit_counts, train, train_epoch)


class V2NumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(42)
        return StreamRefiner(RefinerConfig(rows=2, columns=3, stem_channels=4,
                                          hidden_channels=6, local_channels=3))

    def test_burst_forward_conserves_signed_charge_without_capping(self):
        v = torch.tensor([-6., -.01, 0., .12, .25, .7, 6., 100.125])
        count, residual = lif(v, .25)
        torch.testing.assert_close(count, torch.tensor([0., 0., 0., 0., 1., 2., 24., 400.]))
        torch.testing.assert_close(residual + .25*count, v, rtol=0, atol=0)
        self.assertTrue((residual[v >= 0] < .25).all())
        torch.testing.assert_close(residual[v < 0], v[v < 0])

    def test_positive_burst_gradient_does_not_collapse_at_large_counts(self):
        v = torch.tensor([-.1, 0., .1, .25, 2., 6., 100.], requires_grad=True)
        burst(v, .25).sum().backward()
        torch.testing.assert_close(v.grad, torch.tensor([0., 0., 4., 4., 4., 4., 4.]))

    def test_transmitted_charge_has_unit_surrogate_gain(self):
        for theta in (.125, .25, 1.):
            v = torch.tensor([.15, 2., 24.], requires_grad=True)
            (theta*burst(v, theta)).sum().backward()
            torch.testing.assert_close(v.grad, torch.ones_like(v))

    def test_detached_reset_retains_subthreshold_temporal_gradient(self):
        first = torch.tensor(.15, requires_grad=True)
        _, residual = lif(first, .25)
        count, _ = lif(residual+.15, .25)
        self.assertEqual(float(count), 1.)
        count.backward()
        self.assertEqual(float(first.grad), 4.)

    def test_reset_derivative_is_one_for_both_signs_and_multispikes(self):
        v = torch.tensor([-2., .1, .25, 6.], requires_grad=True)
        lif(v, .25)[1].sum().backward()
        torch.testing.assert_close(v.grad, torch.ones_like(v))

    def test_hidden_and_head_receive_charge_not_unit_counts(self):
        model = self.model()
        captured = {}
        def hook(name):
            def capture(module, args):
                captured[name] = args[0].detach().clone()
            return capture
        a = model.hidden.register_forward_pre_hook(hook("hidden"))
        b = model.head.register_forward_pre_hook(hook("head"))
        x = torch.rand(1, 1, 64, 2, 3)
        model(x)
        stem, _ = lif(model.stem(x[0]), .25)
        hidden, _ = lif(model.hidden(stem*.25), .25)
        torch.testing.assert_close(captured["hidden"], stem*.25)
        torch.testing.assert_close(captured["head"], hidden*.25)
        a.remove(); b.remove()

    def test_chunked_state_future_isolation_and_finite_gradients(self):
        model = self.model()
        before = cpu_weights(model)
        x = torch.rand(4, 2, 64, 2, 3)
        whole, state, rates, _ = model(x)
        first, middle, _, _ = model(x[:2])
        last, tail, _, _ = model(x[2:], middle)
        torch.testing.assert_close(whole, torch.cat((first, last)), rtol=0, atol=0)
        for a, b in zip(state, tail):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        changed = x.clone(); changed[2:] = 20
        torch.testing.assert_close(model(changed)[0][:2], whole[:2], rtol=0, atol=0)
        (whole.square().mean()+rates.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in cpu_weights(model).items()))

    def test_sequence_reset_and_empty_input_preserve_stream_contract(self):
        model = self.model()
        _, state, _, _ = model(torch.rand(2, 2, 64, 2, 3))
        reset = model.reset_slots(state, torch.tensor([True, False]))
        for previous, current in zip(state, reset):
            self.assertEqual(float(current[0].abs().sum()), 0.)
            torch.testing.assert_close(current[1], previous[1])
        empty = torch.zeros(1, 1, 64, 2, 3)
        _, state, rates, counts = model(empty)
        self.assertTrue(all(float(v.abs().sum()) == 0 for v in state))
        self.assertEqual(float(rates.sum()+counts.sum()), 0.)

    def test_v2_keeps_capacity_and_rejects_unvalidated_modes(self):
        self.assertEqual(StreamRefiner().architecture()["parameters"], 9856)
        self.assertEqual(StreamRefiner().architecture()["state_scalars_per_stream"], 1059168)
        for changes in ({"mode": "membrane_route"}, {"version": "phase_grid_refiner_v1"}, {"threshold": 0}):
            with self.assertRaises(ValueError):
                RefinerConfig(**changes)

    def test_logit_decisions_preserve_order_after_sigmoid_saturates(self):
        self.assertTrue((torch.tensor([20., 40.]).sigmoid() == 1).all())
        counts = logit_counts(np.array([1, 0, 1]), np.array([80., 20., 40.]), np.array([False, True, True]), 30.)
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"], counts["proposal_fn"]), (1, 0, 1, 1))
        with self.assertRaises(ValueError):
            logit_counts(np.array([1]), np.array([float("nan")]), np.array([True]), 0)

    def test_monitor_ignores_padded_slots_and_rejected_event_logits(self):
        model = self.model()
        model.last_diagnostics = torch.zeros(1, 2, 3, 7)
        model.last_diagnostics[:, 1] = 1e6
        monitor = HealthMonitor()
        monitor.observe(model, torch.tensor([1000., 1.]), np.array([False, True]), np.array([True, False]))
        self.assertEqual(monitor.summary()["active_stream_steps"], 1)
        self.assertEqual(monitor.summary()["candidate_logits"]["absmax"], 1.)

    def test_monitor_stops_excessive_bursts_logits_loss_and_gradients(self):
        model = self.model()
        model.last_diagnostics = torch.zeros(1, 1, 3, 7)
        model.last_diagnostics[0, 0, 1, 4] = 300
        with self.assertRaisesRegex(FloatingPointError, "max_unit_burst"):
            HealthMonitor().observe(model, torch.tensor([0.]), np.array([True]), np.array([True]))
        model.last_diagnostics.zero_()
        with self.assertRaisesRegex(FloatingPointError, "max_abs_logit"):
            HealthMonitor().observe(model, torch.tensor([81.]), np.array([True]), np.array([True]))
        with self.assertRaisesRegex(FloatingPointError, "max_loss"):
            HealthMonitor().observe_loss(torch.tensor(101.))
        with self.assertRaisesRegex(FloatingPointError, "max_gradient_norm"):
            HealthMonitor().observe_gradients(10001., {}, 5.)

    def test_profile_has_one_revision_and_lower_learning_rate(self):
        profile = load_profile()
        self.assertEqual(profile["training"]["learning_rate"], 1e-4)
        self.assertEqual(profile["model"]["version"], "phase_grid_refiner_v2")
        self.assertEqual(profile["execution"]["threshold_domain"], "logit")
        self.assertFalse(profile["execution"]["automatic_training"])
        with self.assertRaises(ValueError): load_profile(mode="spike_route")
        with self.assertRaises(ValueError): load_profile(batch_size=0)

    def test_v1_source_identity_excludes_v2_but_v2_tracks_both(self):
        old, new = v1_sources(), research_sources()
        self.assertTrue(all("v2" not in key for key in old))
        self.assertTrue(all(new[k] == v for k, v in old.items()))
        self.assertIn("train_local_4060_v2.ps1", new)
        self.assertIn("software/wangdao/h2pro_snn_v2/neurons.py", new)

    def test_dense_cost_includes_charge_scaling_and_padding(self):
        model = self.model()
        a = cost_report(model, np.ones(5), 10)
        b = cost_report(model, np.ones(5), 10, 20)
        self.assertEqual(b["reference_charge_scale_scalar_multiplications"], 2*a["reference_charge_scale_scalar_multiplications"])
        self.assertFalse(b["end_to_end_complete"])
        self.assertFalse(b["spike_convolution_zero_skip_implemented"])

    def test_evaluation_keeps_rejections_coverage_and_logit_domain(self):
        record = sequence((1., 2., 3.), 1)
        record.arrays.update(label=np.array([1, 0, 1]), candidate=np.array([False, True, True]), teacher=np.array([0., .1, .9]))
        store = FakeStore({"a": record})
        batch = next(stream_batches(store, "train_dev", 1))
        model = self.model()
        model.last_diagnostics = torch.zeros(10, 1, 3, 7)
        def forward(*args): return torch.tensor([80., 20., 40.]), None, None, torch.zeros(10, 1, 5)
        with patch("wangdao.h2pro_snn_v2.training.forward_batch", side_effect=forward), \
                patch("wangdao.h2pro_snn_v2.training.stream_batches", return_value=iter([batch])):
            result = evaluate(model, store, torch.device("cpu"), 1, [30.])
        self.assertEqual(result["selected"]["overall"]["iou"], .5)
        self.assertEqual(result["selected"]["threshold_domain"], "logit")
        with patch("wangdao.h2pro_snn_v2.training.forward_batch", side_effect=forward), \
                patch("wangdao.h2pro_snn_v2.training.stream_batches", return_value=iter([batch, batch])):
            with self.assertRaisesRegex(AssertionError, "Duplicate"):
                evaluate(model, store, torch.device("cpu"), 1, [30.])

    def test_epoch_monitor_and_optimizer_contract_without_parameter_updates(self):
        model = self.model(); before = cpu_weights(model)
        class NoUpdate:
            calls = 0
            def zero_grad(self, **kwargs): model.zero_grad(**kwargs)
            def step(self): self.calls += 1
        batch = {"inputs": np.ones((2, 1, 64, 2, 3), np.float32), "reset": np.array([True]),
                 "maps": np.array([[0, 1], [0, 0], [0, 1], [0, 1], [0, 2]]),
                 "labels": np.array([0, 1]), "candidate": np.array([True, True]),
                 "teacher": np.array([.1, .8], np.float32), "active": np.array([True])}
        store = FakeStore({"a": sequence((1., 2.), 1)})
        optimizer = NoUpdate()
        with patch("wangdao.h2pro_snn_v2.training.stream_batches", return_value=iter([batch])):
            result = train_epoch(model, store, optimizer, torch.device("cpu"), load_profile()["training"], 1)
        self.assertEqual(optimizer.calls, 1)
        self.assertEqual(result["stability"]["pre_clip_gradients"]["backward_passes"], 1)
        self.assertEqual(result["parameter_norms_before"], result["parameter_norms_after"])
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in cpu_weights(model).items()))
        batch["inputs"] *= 1e6
        with patch("wangdao.h2pro_snn_v2.training.stream_batches", return_value=iter([batch])):
            with self.assertRaises(FloatingPointError):
                train_epoch(model, store, optimizer, torch.device("cpu"), load_profile()["training"], 1)
        self.assertEqual(optimizer.calls, 1)

    def test_resume_and_failure_preserve_committed_checkpoint_without_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"results/runs") as folder:
            folder = Path(folder); store = FakeStore({"a": sequence()})
            store.directory = folder/"cache"; store.directory.mkdir()
            atomic_json(store.directory/"manifest.json", {"unit_test": True})
            for row in store.manifest["roles"]["train_main"]:
                row.update(candidate_events=6, candidate_targets=3)
            profile = load_profile(batch_size=1); profile["model"].update(rows=2, columns=3)
            selected = {"threshold": 0., "threshold_domain": "logit", "overall": {"iou": .5}, "high_density": {"recall": .6}}
            evaluation = {"selected": selected, "seconds": 0., "stability": {"warnings": []}, "selected_at_scan_boundary": False}
            def no_training(model, store, optimizer, device, config, epoch):
                optimizer._opt_called = True
                return {"epoch": epoch, "loss": 1., "train_s": 0., "mean_support_spikes_per_neuron_step": [0., 0.], "stability": {"warnings": []}}
            run = folder/"run"
            with patch("wangdao.h2pro_snn_v2.training.SequenceStore", return_value=store), \
                    patch("wangdao.h2pro_snn_v2.training.train_epoch", side_effect=no_training), \
                    patch("wangdao.h2pro_snn_v2.training.evaluate", return_value=evaluation), patch("builtins.print"), \
                    patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Real optimizer.step forbidden in tests")):
                train(store.directory, run, profile, "cpu", 1, pilot=True)
                first = checkpoint_load(run/"last.pt")
                second = train(store.directory, run, profile, "cpu", 2, resume=True, pilot=True)
                self.assertEqual(second["best"]["epoch"], 1)
                self.assertEqual(second["completed_epochs"], 2)
                self.assertTrue(all(torch.equal(v, first["model"][k]) for k, v in checkpoint_load(run/"last.pt")["model"].items()))
                changed = copy.deepcopy(profile); changed["training"]["learning_rate"] *= .5
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    train(store.directory, run, changed, "cpu", 3, resume=True, pilot=True)
                previous = checkpoint_load(run/"last.pt")
                with patch("wangdao.h2pro_snn_v2.training.train_epoch", side_effect=FloatingPointError("test failure")):
                    with self.assertRaises(FloatingPointError):
                        train(store.directory, run, profile, "cpu", 3, resume=True, pilot=True)
                failure = json.loads((run/"stability_failure.json").read_text())
                self.assertEqual(failure["last_committed_epoch"], 2)
                self.assertEqual(checkpoint_load(run/"last.pt")["report"], previous["report"])
                self.assertFalse((run/".training.lock").exists())

    def test_cli_defaults_to_one_epoch_and_preserves_v1_entrypoint(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT/"software/scripts/train_h2pro_refiner_v2.py"), "train", "--help"], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", result.stdout)
        self.assertIn("health check is one", " ".join(result.stdout.split()))
        wrapper = (ROOT/"train_local_4060_v2.ps1").read_text(encoding="ascii")
        self.assertIn("$Action = 'check'", wrapper)
        self.assertIn("$Epochs = 1", wrapper)
        self.assertIn("h2pro_refiner_fixed_v2_s42", wrapper)


if __name__ == "__main__":
    unittest.main()
