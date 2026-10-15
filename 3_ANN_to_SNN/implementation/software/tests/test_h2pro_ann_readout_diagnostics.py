"""Guard, diagnosis and recovery regressions. No real optimizer update/fitting."""
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
from test_h2pro_refiner import FakeStore as BaseFakeStore, sequence
from wangdao.h2pro_snn.foundation import sha256
from wangdao.h2pro_snn.stream_data import atomic_json, stream_batches
from wangdao.h2pro_snn.training import checkpoint_load, cpu_weights, forward_batch as original_forward
from wangdao.h2pro_snn_v2.foundation import research_sources as v2_sources
from wangdao.h2pro_ann_readout_v1.refiner import RefinerConfig, StreamRefiner
from wangdao.h2pro_snn_v2.training import load_profile as v2_profile
from wangdao.h2pro_ann_readout_v1.diagnostics import RunDiagnostics, batch_summary, distribution, json_safe, write_json
from wangdao.h2pro_ann_readout_v1.foundation import research_sources
from wangdao.h2pro_ann_readout_v1.monitor import HealthMonitor
from wangdao.h2pro_ann_readout_v1.training import evaluate, forward_batch, load_profile, train, train_epoch

MODULE = "wangdao.h2pro_ann_readout_v1.training."


class FakeStore(BaseFakeStore):
    directory: Path


class ANNReadoutDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(42)
        model = StreamRefiner(RefinerConfig(rows=2, columns=3, stem_channels=4,
                                          hidden_channels=6, local_channels=3))
        model.last_diagnostics = torch.zeros(2, 1, 3, 7)
        return model

    def batch(self):
        return {"inputs": np.ones((2, 1, 64, 2, 3), np.float32), "reset": np.array([True]),
                "maps": np.array([[0, 1], [0, 0], [0, 1], [0, 1], [0, 2]]),
                "labels": np.array([0, 1]), "candidate": np.array([True, True]),
                "teacher": np.array([.1, .8], np.float32), "active": np.array([True]),
                "refs": [{"sequence": "synthetic", "slot": 0, "window": 0, "ids": np.array([6, 9])}]}

    def recorder(self, folder, model=None, batch=None):
        model, batch = model or self.model(), batch or self.batch()
        recorder = RunDiagnostics(folder, {"test_only": True})
        monitor = HealthMonitor()
        recorder.begin(model, batch, None, monitor, torch.device("cpu"), phase="train_main", epoch=1, index=17)
        return recorder, model, batch, monitor

    def test_finite_positive_and_negative_logits_warn_without_clamping(self):
        values = torch.tensor([-82.2119, 83.])
        before = values.clone()
        monitor = HealthMonitor()
        self.assertTrue(monitor.observe(self.model(), values, np.ones(2, bool), np.ones(1, bool)))
        torch.testing.assert_close(values, before, rtol=0, atol=0)
        summary = monitor.summary()
        self.assertEqual(summary["candidate_logits"]["above_warning_count"], 2)
        self.assertEqual(summary["candidate_logits"]["warning_batches"], 1)
        self.assertNotIn("max_abs_logit", summary["failure_limits"])
        self.assertEqual(summary["warning_limits"]["max_abs_logit"], 80.)

    def test_candidate_padding_and_warning_boundary(self):
        model = self.model()
        model.last_diagnostics = torch.zeros(2, 2, 3, 7)
        model.last_diagnostics[:, 1] = float("nan")
        monitor = HealthMonitor()
        self.assertEqual(monitor.observe(model, torch.tensor([1000., -80., 80.]),
                                        np.array([False, True, True]), np.array([True, False])), [])
        self.assertEqual(monitor.summary()["candidate_logits"]["min"], -80.)
        self.assertEqual(monitor.summary()["active_stream_steps"], 2)

    def test_nan_and_infinite_candidate_logits_still_stop(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(FloatingPointError, "Nonfinite candidate"):
                HealthMonitor().observe(self.model(), torch.tensor([value]), np.array([True]), np.array([True]))

    def test_layer_nonfinite_still_stops(self):
        model = self.model()
        assert model.last_diagnostics is not None
        model.last_diagnostics[0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite layer"):
            HealthMonitor().observe(model, torch.tensor([0.]), np.array([True]), np.array([True]))

    def test_burst_and_voltage_budgets_preserve_offending_statistics(self):
        for stat, value, message in ((4, 300., "max_activation_in_threshold_units"), (2, 1025., "max_abs_voltage")):
            model = self.model()
            assert model.last_diagnostics is not None
            model.last_diagnostics[0, 0, 1, stat] = value
            monitor = HealthMonitor()
            with self.assertRaisesRegex(FloatingPointError, message):
                monitor.observe(model, torch.tensor([0.]), np.array([True]), np.array([True]))
            self.assertEqual(monitor.summary()["active_stream_steps"], 2)
            self.assertGreater(max(monitor.summary()["layers"]["hidden"]["max_over_active_steps"].values()), 0)

    def test_loss_and_raw_gradient_guards_remain_hard(self):
        for value in (101., float("nan"), float("inf")):
            monitor = HealthMonitor()
            with self.assertRaises(FloatingPointError): monitor.observe_loss(torch.tensor(value))
            self.assertIsNotNone(monitor.summary()["last_observed_loss"])
        for value in (10001., float("nan"), float("inf")):
            monitor = HealthMonitor()
            with self.assertRaises(FloatingPointError): monitor.observe_gradients(value, {}, 5.)
            self.assertIsNotNone(monitor.summary()["pre_clip_gradients"]["last_observed_norm"])

    def test_wrong_configuration_cannot_silently_disable_a_hard_guard(self):
        for limits in ({}, {"max_abs_logit": 80.}):
            with self.assertRaises(ValueError): HealthMonitor(limits)
        with self.assertRaises(ValueError): HealthMonitor(warnings={"max_abs_logit": float("inf")})

    def test_signed_class_diagnostics_map_back_to_sequence_event(self):
        batch = self.batch()
        result = batch_summary(batch, torch.tensor([-82.2119, -90.]), 80., {"synthetic": {"high_density": True}})
        counts = result["diagnostic_at_logit_zero"]
        self.assertEqual((counts["above_warning_negative"], counts["above_warning_correct"], counts["above_warning_wrong"]), (2, 1, 1))
        self.assertEqual(result["target_logits"]["min"], -90.)
        self.assertEqual(result["extreme_candidates"][0]["event_id"], 9)
        self.assertEqual(result["extreme_candidates"][0]["sequence"], "synthetic")
        self.assertTrue(result["sequences"][0]["high_density"])

    def test_diagnostic_handles_empty_and_nonfinite_data_as_valid_json(self):
        self.assertEqual(distribution([])["finite_count"], 0)
        value = distribution([float("nan"), float("inf"), -float("inf"), 1.])
        self.assertEqual((value["nan_count"], value["positive_inf_count"], value["negative_inf_count"]), (1, 1, 1))
        record = batch_summary(self.batch(), torch.tensor([float("nan"), float("inf")]), 80.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"failure.json"
            write_json(path, record)
            parsed = json.loads(path.read_text(), parse_constant=lambda x: self.fail("Invalid JSON token " + x))
            self.assertEqual(parsed["candidate_logits"]["nan_count"], 1)
        json.dumps(json_safe({"v": [float("nan"), float("inf")]}), allow_nan=False)

    def test_original_forward_values_gradients_and_state_are_unchanged(self):
        a, b, batch = self.model(), self.model(), self.batch()
        out_a = original_forward(a, batch, None, torch.device("cpu"))
        out_b = forward_batch(b, batch, None, torch.device("cpu"))
        for i in (0, 2, 3): torch.testing.assert_close(out_a[i], out_b[i], rtol=0, atol=0)
        for x, y in zip(out_a[1], out_b[1]): torch.testing.assert_close(x, y, rtol=0, atol=0)
        out_a[0].square().sum().backward(); out_b[0].square().sum().backward()
        for a_p, b_p in zip(a.parameters(), b.parameters()): torch.testing.assert_close(a_p.grad, b_p.grad, rtol=0, atol=0)

    def test_full_warning_snapshot_replays_without_parameter_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder, model, batch, monitor = self.recorder(folder)
            with torch.no_grad(): model.bias[:, 2:] += 100.
            before = cpu_weights(model)
            logits, state, _, _ = forward_batch(model, batch, None, torch.device("cpu"), diagnostics=recorder)
            alerts = monitor.observe(model, logits, batch["candidate"], batch["active"])
            with patch("builtins.print"): recorder.warnings(alerts)
            snapshot = checkpoint_load(next(recorder.folder.glob("*.pt")))
            self.assertEqual(snapshot["purpose"], "diagnostic_only_not_resumable")
            self.assertTrue(snapshot["forward_replay_weights_match"])
            self.assertNotIn("optimizer", snapshot)
            replay_batch = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in snapshot["batch"].items()}
            replay = StreamRefiner(RefinerConfig(**snapshot["model_config"]))
            replay.load_state_dict(snapshot["model"])
            recovered = forward_batch(replay, replay_batch, snapshot["state_before"], torch.device("cpu"))
            torch.testing.assert_close(recovered[0], snapshot["logits"], rtol=0, atol=0)
            for x, y in zip(recovered[1], state): torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in cpu_weights(model).items()))

    def test_warning_snapshot_frequency_is_bounded_but_counts_continue(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder, model, batch, monitor = self.recorder(folder)
            model.last_diagnostics = torch.zeros(2, 1, 3, 7)
            recorder.observation(torch.tensor([81., -82.]), model.initial_state(1))
            with patch("builtins.print"):
                for _ in range(101): recorder.warnings(["finite_candidate_logit_magnitude"])
            self.assertEqual(len(list(recorder.folder.glob("*.pt"))), 1)
            self.assertEqual(len((Path(folder)/"diagnostics.jsonl").read_text().splitlines()), 3)
            self.assertEqual(recorder.warning_counts["train_main"], 101)

    def test_forward_nonfinite_captured_before_guard_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder, model, batch, _ = self.recorder(folder)
            def nonfinite(*args):
                model.last_diagnostics = torch.zeros(2, 1, 3, 7)
                return torch.full((2, 1, 32, 2, 3), float("nan")), model.initial_state(1), torch.zeros(2, 1, 2), torch.zeros(2, 1, 5)
            with patch.object(model, "forward", side_effect=nonfinite):
                with self.assertRaises(FloatingPointError): forward_batch(model, batch, None, torch.device("cpu"), diagnostics=recorder)
            result = recorder.failure(FloatingPointError("injected nonfinite"))
            self.assertEqual(result["batch"]["candidate_logits"]["nan_count"], 2)
            saved = checkpoint_load(Path(folder)/result["snapshot"])
            self.assertTrue(torch.isnan(saved["logits"]).all())
            self.assertEqual(result["capture_errors"], [])

    def test_secondary_snapshot_error_does_not_mask_original_cause(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder, _, _, _ = self.recorder(folder)
            with patch.object(recorder, "snapshot", side_effect=OSError("disk test")):
                result = recorder.failure(FloatingPointError("original cause"))
            self.assertEqual(result["message"], "original cause")
            self.assertIn("disk test", result["capture_errors"][0])
            self.assertTrue((recorder.folder/"failure.json").exists())

    def test_warning_train_path_runs_with_no_update_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            model, batch = self.model(), self.batch()
            with torch.no_grad(): model.bias[:, 2:] += 85.
            before = cpu_weights(model)
            class NoUpdate:
                calls = 0
                def zero_grad(self, **kwargs): model.zero_grad(**kwargs)
                def step(self): self.calls += 1
            optimizer = NoUpdate()
            store = FakeStore({"synthetic": sequence((1., 2.), 1)})
            recorder = RunDiagnostics(folder, {"test_only": True})
            with patch(MODULE+"stream_batches", return_value=iter([batch])), patch("builtins.print"):
                result = train_epoch(model, store, optimizer, torch.device("cpu"), load_profile()["training"], 1, diagnostics=recorder)
            self.assertEqual(optimizer.calls, 1)
            self.assertEqual(result["stability"]["candidate_logits"]["warning_batches"], 1)
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in cpu_weights(model).items()))
            progress = json.loads((Path(folder)/"progress.json").read_text())
            self.assertEqual(progress["status"], "phase_complete")
            self.assertIn("focal", progress["loss_components"])
            self.assertIn("peak_allocated_bytes", progress)

    def test_evaluation_allows_large_finite_scores_with_same_event_coverage(self):
        record = sequence((1., 2., 3.), 1)
        record.arrays.update(label=np.array([1, 0, 1]), candidate=np.array([False, True, True]), teacher=np.array([0., .1, .9]))
        store, model = FakeStore({"a": record}), self.model()
        batch = next(stream_batches(store, "train_dev", 1))
        def fake_forward(*args, **kwargs): return torch.tensor([90., 82., 84.]), None, None, torch.zeros(10, 1, 5)
        with patch(MODULE+"forward_batch", side_effect=fake_forward), patch(MODULE+"stream_batches", return_value=iter([batch])):
            result = evaluate(model, store, torch.device("cpu"), 1, [83.])
        self.assertEqual(result["selected"]["overall"]["iou"], .5)
        self.assertEqual(result["selected"]["overall"]["proposal_fn"], 1)
        self.assertEqual(result["stability"]["candidate_logits"]["above_warning_count"], 2)

    def test_evaluation_persists_warning_and_phase_completion_without_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            model, batch = self.model(), self.batch()
            with torch.no_grad(): model.bias[:, 2:] += 85.
            before = cpu_weights(model)
            batch["refs"][0]["ids"] = np.array([0, 1])
            store = FakeStore({"synthetic": sequence((1., 2.), 1)})
            recorder = RunDiagnostics(folder, {"test_only": True})
            with patch(MODULE+"stream_batches", return_value=iter([batch])), patch("builtins.print"):
                result = evaluate(model, store, torch.device("cpu"), 1, [0.], diagnostics=recorder, epoch=1)
            self.assertEqual(result["stability"]["candidate_logits"]["warning_batches"], 1)
            progress = json.loads((Path(folder)/"progress.json").read_text())
            self.assertEqual((progress["phase"], progress["status"]), ("train_dev", "phase_complete"))
            self.assertTrue((recorder.folder/"first_warning_train_dev.pt").exists())
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in cpu_weights(model).items()))

    def test_completed_checkpoint_resume_and_failure_status_without_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"results/runs") as folder:
            folder = Path(folder); store = FakeStore({"a": sequence()})
            store.directory = folder/"cache"; store.directory.mkdir()
            atomic_json(store.directory/"manifest.json", {"unit_test": True})
            for row in store.manifest["roles"]["train_main"]: row.update(candidate_events=6, candidate_targets=3)
            profile = load_profile(batch_size=1); profile["model"].update(rows=2, columns=3)
            selected = {"threshold": 0., "threshold_domain": "logit", "overall": {"iou": .5}, "high_density": {"recall": .6}}
            evaluation = {"selected": selected, "seconds": 0., "stability": {"warnings": []}, "selected_at_scan_boundary": False}
            def no_training(model, store, optimizer, device, config, epoch, **kwargs):
                optimizer._opt_called = True
                return {"epoch": epoch, "loss": 1., "train_s": 0., "mean_support_activation_in_threshold_units": [0., 0.], "stability": {"warnings": []}}
            run = folder/"run"
            with patch(MODULE+"SequenceStore", return_value=store), patch(MODULE+"train_epoch", side_effect=no_training), \
                    patch(MODULE+"evaluate", return_value=evaluation), patch("builtins.print"), \
                    patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No real optimizer step allowed")):
                train(store.directory, run, profile, "cpu", 1, pilot=True)
                train(store.directory, run, profile, "cpu", 2, resume=True, pilot=True)
                checkpoint_hash = sha256(run/"last.pt")
                for error in (FloatingPointError("failure test"), KeyboardInterrupt()):
                    with patch(MODULE+"train_epoch", side_effect=error), self.assertRaises(type(error)):
                        train(store.directory, run, profile, "cpu", 3, resume=True, pilot=True)
                    self.assertEqual(sha256(run/"last.pt"), checkpoint_hash)
                    report = json.loads((run/"training.json").read_text())
                    self.assertEqual((report["status"], report["completed_epochs"]), ("stopped", 2))
                    self.assertEqual(report["failure"]["last_committed_epoch"], 2)
                    self.assertFalse((run/".training.lock").exists())
                changed = copy.deepcopy(profile); changed["training"]["learning_rate"] *= .5
                report_hash = sha256(run/"training.json")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    train(store.directory, run, changed, "cpu", 3, resume=True, pilot=True)
                self.assertEqual(sha256(run/"training.json"), report_hash)

    def test_initial_failure_captures_batch_and_keeps_epoch_zero(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"results/runs") as folder:
            folder = Path(folder); store = FakeStore({"a": sequence()})
            store.directory = folder/"cache"; store.directory.mkdir()
            atomic_json(store.directory/"manifest.json", {"unit_test": True})
            for row in store.manifest["roles"]["train_main"]: row.update(candidate_events=6, candidate_targets=3)
            profile = load_profile(batch_size=1); profile["model"].update(rows=2, columns=3)
            def fail(model, store, optimizer, device, config, epoch, *, diagnostics):
                batch, monitor = self.batch(), HealthMonitor()
                diagnostics.begin(model, batch, None, monitor, device, phase="train_main", epoch=epoch, index=17)
                logits, _, _, _ = forward_batch(model, batch, None, device, diagnostics=diagnostics)
                diagnostics.stage("loss_guard", loss=101.)
                monitor.observe_loss(torch.tensor(101.))
            run = folder/"run"
            with patch(MODULE+"SequenceStore", return_value=store), patch(MODULE+"train_epoch", side_effect=fail), \
                    patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No actual training")):
                with self.assertRaisesRegex(FloatingPointError, "max_loss"):
                    train(store.directory, run, profile, "cpu", 1, pilot=True)
            failure = json.loads((run/"stability_failure.json").read_text())
            self.assertEqual((failure["last_committed_epoch"], failure["batch_index"], failure["stage"]), (0, 17, "loss_guard"))
            self.assertEqual(failure["loss"], 101.)
            self.assertIn("monitor.observe_loss", failure["traceback"])
            self.assertTrue((run/failure["snapshot"]).exists())
            self.assertEqual(checkpoint_load(run/"last.pt")["optimizer"]["state"], {})
            self.assertFalse((run/"best.pt").exists())

    def test_control_is_independently_versioned_with_old_sources_preserved(self):
        from wangdao.h2pro_ann_state_v1.foundation import research_sources as reference_sources
        from wangdao.h2pro_ann_state_v1.training import load_profile as reference_profile
        old, new = reference_profile(), load_profile()
        self.assertEqual(old['training'], new['training'])
        self.assertEqual(old['execution'], new['execution'])
        self.assertEqual(new['model']['version'], 'phase_grid_ann_readout_v1')
        previous, current = reference_sources(), research_sources()
        self.assertTrue(all(current[k] == v for k, v in previous.items()))
        self.assertIn('train_local_4060_ann_readout.ps1', current)

    def test_launcher_is_safe_default_with_fresh_run_directory(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT/"software/scripts/train_h2pro_ann_readout.py"), "train", "--help"], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        wrapper = (ROOT/"train_local_4060_ann_readout.ps1").read_text(encoding="ascii")
        self.assertIn("$Action = 'check'", wrapper)
        self.assertIn("$Epochs = 1", wrapper)
        self.assertIn("h2pro_ann_readout_v1_s42", wrapper)


if __name__ == "__main__":
    unittest.main()
