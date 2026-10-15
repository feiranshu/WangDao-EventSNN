"""Finite-advisory policy, exact migration and continued mid-epoch recovery."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/"software"))
import test_h2pro_longtrain as previous_tests
from test_h2pro_longtrain import Data, tiny_profile, fake_evaluate, same
from wangdao.h2pro_longtrain_v1 import runtime as old_runtime
from wangdao.h2pro_longtrain_v1.storage import load, save_atomic, PauseControl, PauseRequested
from wangdao.h2pro_longtrain_v1.models import ARMS
from wangdao.h2pro_longtrain_v1_1 import runtime, migration
from wangdao.h2pro_longtrain_v1_1.models import profile_for
from wangdao.h2pro_longtrain_v1_1.monitor import (StateHealthMonitor, LinearHealthMonitor,
                                                monitor_for, emit_advisories)


def new_tiny_profile(arm):
    p = profile_for(arm)
    p["model"].update(rows=2, columns=3)
    return p


class AdvisoryTrainingTests(unittest.TestCase):
    temp = previous_tests.LongTrainingTests.temp

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_former_finite_cutoffs_only_record_advisories(self):
        for cls in (StateHealthMonitor, LinearHealthMonitor):
            m = cls()
            for key, value in m.limits.items():
                m._limit(key, value*100)
                self.assertEqual(m.advisory_counts[key], 1)
                self.assertEqual(m.advisory_maxima[key], value*100)
            self.assertEqual(m.summary()["failure_limits"], {})
            self.assertEqual(len(m.drain()), 4)

    def test_nonfinite_limits_still_stop(self):
        for cls in (StateHealthMonitor, LinearHealthMonitor):
            m = cls()
            for key in m.limits:
                for value in (float("nan"), float("inf"), -float("inf")):
                    with self.assertRaises(FloatingPointError):
                        m._limit(key, value)

    def test_observe_accepts_large_finite_head_without_changing_values(self):
        for arm in ARMS:
            p = profile_for(arm)
            m = monitor_for(arm, p["training"])
            stats = torch.zeros(1, 1, 3, 7)
            stats[0, 0, 2, 4] = 293.64
            stats[0, 0, 2, 2] = 1200.  # also crosses the old 4096 threshold-unit level
            logits = torch.tensor([90., -90.])
            model = SimpleNamespace(last_diagnostics=stats, last_readout_extrema=None,
                                    config=SimpleNamespace(threshold=.25))
            before = stats.clone(), logits.clone()
            m.observe(model, logits, np.ones(2, bool), np.ones(1, bool))
            self.assertTrue(torch.equal(before[0], stats) and torch.equal(before[1], logits))
            self.assertEqual(m.advisory_counts["max_activation_in_threshold_units"], 1)
            self.assertEqual(m.advisory_counts["max_abs_voltage_in_thresholds"], 1)

    def test_nonfinite_logits_and_layer_stats_still_stop(self):
        for arm in ARMS:
            for invalid_logits in (True, False):
                m = monitor_for(arm, profile_for(arm)["training"])
                stats = torch.zeros(1, 1, 3, 7)
                logits = torch.zeros(1)
                (logits if invalid_logits else stats).fill_(float("nan"))
                model = SimpleNamespace(last_diagnostics=stats, last_readout_extrema=None,
                                        config=SimpleNamespace(threshold=.25))
                with self.assertRaises(FloatingPointError):
                    m.observe(model, logits, np.ones(1, bool), np.ones(1, bool))

    def test_large_loss_and_gradient_are_advisory_but_nonfinite_are_not(self):
        m = LinearHealthMonitor()
        m.observe_loss(torch.tensor(200.))
        m.observe_gradients(torch.tensor(20000.), {"weight": 20000.}, 5.)
        self.assertEqual(m.advisory_counts["max_loss"], 1)
        self.assertEqual(m.advisory_counts["max_gradient_norm"], 1)
        with self.assertRaises(FloatingPointError): m.observe_loss(torch.tensor(float("nan")))
        with self.assertRaises(FloatingPointError): m.observe_gradients(torch.tensor(float("inf")), {}, 5.)

    def test_warning_io_bounded_while_all_occurrences_counted(self):
        m = LinearHealthMonitor()
        for _ in range(2500): m._limit("max_activation_in_threshold_units", 300.)
        self.assertEqual(m.advisory_counts["max_activation_in_threshold_units"], 2500)
        self.assertEqual(len(m.drain()), 3)

    def test_monitor_state_restores_old_statistics_and_new_advisories(self):
        p = profile_for("ann_linear")["training"]
        m = monitor_for("ann_linear", p)
        m._limit("max_loss", 120.)
        m.drain()
        m.steps = 20
        path = self.temp()/"monitor.pt"
        save_atomic(path, m.__dict__)
        restored = monitor_for("ann_linear", p, load(path))
        self.assertEqual(m.summary(), restored.summary())

    def fixture(self, arm):
        root = self.temp(); data = Data(root)
        data.encoded = {r["sequence"]: {"cuts": np.arange(r["closed_windows"]+1)*2}
                        for r in data.store.manifest["roles"]["train_main"]}
        with patch.object(old_runtime, "profile_for", tiny_profile):
            with PauseControl(root) as pause:
                old_runtime.run_arm(data, root/"old", arm, 2, pause, stop_after_batches=1, import_original=False)
        saved = load(root/"old/last.pt")
        # Fixture models formal provenance; all files live in the disposable test directory.
        saved["report"]["disposable_check"] = False
        return root, data, saved

    def test_exact_migration_then_full_training_matches_old_mathematics_all_arms(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                root, data, saved = self.fixture(arm)
                with patch.object(migration, "old_profile", tiny_profile), \
                     patch.object(migration, "legacy_origin", return_value=(None, None)):
                    migration.validate_source(saved, data, arm)
                identity = migration.make_identity(data, arm, new_tiny_profile(arm), {"fixture": True})
                migrated = migration.migrate_payload(saved, identity)
                for key in migration.STATE_KEYS: self.assertTrue(same(saved[key], migrated[key]), key)
                save_atomic(root/"new/last.pt", migrated)
                save_atomic(root/"split/last.pt", migrated)
                with patch.object(old_runtime, "profile_for", tiny_profile), \
                     patch.object(old_runtime, "evaluate", fake_evaluate), \
                     patch.object(old_runtime, "final_health", return_value={"status": "passed"}), \
                     patch.object(runtime, "profile_for", new_tiny_profile), \
                     patch.object(runtime, "evaluate", fake_evaluate), \
                     patch.object(runtime, "final_health", return_value={"status": "passed"}):
                    with PauseControl(root) as pause:
                        old_runtime.run_arm(data, root/"old", arm, 2, pause, import_original=False)
                        runtime.run_arm(data, root/"new", arm, 2, pause)
                        runtime.run_arm(data, root/"split", arm, 2, pause, stop_after_batches=2)
                        runtime.run_arm(data, root/"split", arm, 2, pause)
                a, b, c = [load(root/n/"last.pt") for n in ("old", "new", "split")]
                for key in ("model", "optimizer", "scheduler", "rng", "cursor", "best_model"):
                    self.assertTrue(same(a[key], b[key]) and same(b[key], c[key]), key)
                for ae, be, ce in zip(a["report"]["history"], b["report"]["history"], c["report"]["history"]):
                    for k in ("loss", "loss_components", "evaluation", "events", "windows", "updates"):
                        self.assertEqual(ae[k], be[k]); self.assertEqual(be[k], ce[k])

    def test_migration_rejects_incomplete_nonfinite_and_mismatched_checkpoints(self):
        _, data, saved = self.fixture("snn_linear")
        variants = []
        bad = copy.deepcopy(saved); bad.pop("optimizer"); variants.append(bad)
        bad = copy.deepcopy(saved); bad["identity"]["arm"] = "ann_state"; variants.append(bad)
        bad = copy.deepcopy(saved); bad["cursor"]["batch"] += 1; variants.append(bad)
        bad = copy.deepcopy(saved); bad["stream_state"] = None; variants.append(bad)
        bad = copy.deepcopy(saved); next(iter(bad["model"].values())).fill_(float("nan")); variants.append(bad)
        bad = copy.deepcopy(saved); bad["report"]["disposable_check"] = True; variants.append(bad)
        with patch.object(migration, "old_profile", tiny_profile), \
             patch.object(migration, "legacy_origin", return_value=(None, None)):
            for value in variants:
                with self.assertRaises((ValueError, FloatingPointError)):
                    migration.validate_source(value, data, "snn_linear")

    def test_native_checkpoint_identity_is_not_bypassed(self):
        root, data, saved = self.fixture("snn_linear")
        identity = migration.make_identity(data, "snn_linear", new_tiny_profile("snn_linear"), {})
        migrated = migration.migrate_payload(saved, identity)
        migrated["identity"]["sources"] = {}
        save_atomic(root/"new/last.pt", migrated)
        with patch.object(runtime, "profile_for", new_tiny_profile), PauseControl(root) as pause:
            with self.assertRaisesRegex(ValueError, "Resume code"):
                runtime.run_arm(data, root/"new", "snn_linear", 2, pause)

    def test_native_pause_during_dev_resumes_without_repeating_updates(self):
        root, data, saved = self.fixture("ann_linear")
        identity = migration.make_identity(data, "ann_linear", new_tiny_profile("ann_linear"), {})
        save_atomic(root/"new/last.pt", migration.migrate_payload(saved, identity))
        with patch.object(runtime, "profile_for", new_tiny_profile), \
             patch.object(runtime, "final_health", return_value={"status": "passed"}), PauseControl(root) as pause:
            with patch.object(runtime, "evaluate", side_effect=PauseRequested):
                report = runtime.run_arm(data, root/"new", "ann_linear", 1, pause)
            before = load(root/"new/last.pt")
            self.assertEqual(report["status"], "paused")
            self.assertEqual(before["cursor"]["phase"], "evaluate")
            with patch.object(runtime, "evaluate", fake_evaluate), \
                 patch.object(runtime, "update_step", side_effect=AssertionError("Repeated update")):
                report = runtime.run_arm(data, root/"new", "ann_linear", 1, pause)
            after = load(root/"new/last.pt")
        self.assertEqual(report["status"], "complete")
        self.assertTrue(same(before["model"], after["model"]))
        self.assertTrue(same(before["optimizer"]["state"], after["optimizer"]["state"]))
        self.assertEqual(after["scheduler"]["last_epoch"], before["scheduler"]["last_epoch"]+1)

    def test_native_nonfinite_failure_preserves_checkpoint_and_does_not_auto_retry(self):
        root, data, saved = self.fixture("ann_linear")
        identity = migration.make_identity(data, "ann_linear", new_tiny_profile("ann_linear"), {})
        save_atomic(root/"new/last.pt", migration.migrate_payload(saved, identity))
        with patch.object(runtime, "profile_for", new_tiny_profile), \
             patch.object(runtime, "update_step", side_effect=FloatingPointError("Nonfinite loss")) as step, \
             PauseControl(root) as pause:
            report = runtime.run_arm(data, root/"new", "ann_linear", 2, pause)
            self.assertEqual(report["status"], "failed")
            preserved = load(root/"new/last.pt")
            for key in migration.STATE_KEYS:
                self.assertTrue(same(saved[key], preserved[key]), key)
            self.assertTrue((root/"new/failure_context.pt").exists())
            calls = step.call_count
            report = runtime.run_arm(data, root/"new", "ann_linear", 2, pause)
            self.assertEqual(report["status"], "failed_requires_review")
            self.assertEqual(step.call_count, calls)


if __name__ == "__main__":
    unittest.main()
