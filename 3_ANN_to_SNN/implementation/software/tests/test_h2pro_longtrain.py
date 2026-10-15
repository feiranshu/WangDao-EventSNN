"""Recovery invariants across sequence replacement, empty windows and corruption."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/"software"))
from wangdao.h2pro_longtrain_v1 import runtime
from wangdao.h2pro_longtrain_v1.data import schedule
from wangdao.h2pro_longtrain_v1.models import profile_for, make_model, ARMS
from wangdao.h2pro_longtrain_v1.storage import (save_atomic, load, all_rng, set_rng, process_lock, PauseControl, PauseRequested)
from wangdao.h2pro_snn.stream_data import atomic_json


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def tiny_profile(arm):
    p = profile_for(arm)
    p["model"].update(rows=2, columns=3)
    return p


class Store:
    def __init__(self, directory):
        self.directory = directory
        self.manifest = {"roles": {"train_main": [
            {"sequence": "a", "closed_windows": 3, "published_events": 6, "candidate_targets": 3, "candidate_events": 6},
            {"sequence": "b", "closed_windows": 1, "published_events": 2, "candidate_targets": 1, "candidate_events": 2},
            {"sequence": "c", "closed_windows": 2, "published_events": 4, "candidate_targets": 2, "candidate_events": 4},
            {"sequence": "empty", "closed_windows": 0, "published_events": 0, "candidate_targets": 0, "candidate_events": 0}]}}
        atomic_json(directory/"manifest.json", self.manifest)

    def names(self, role):
        return [r["sequence"] for r in self.manifest["roles"][role]]


class Data:
    def __init__(self, directory):
        self.directory = directory
        self.store = Store(directory)
        self.device = torch.device("cpu")
        self.stats = {}

    def batch(self, spec):
        slots, reset = spec
        active = np.asarray([s is not None for s in slots])
        inputs = torch.zeros(2, 2, 64, 2, 3)
        maps, labels, refs = [], [], []
        for slot, s in enumerate(slots):
            if s is not None:
                # Include a truly empty input window that still carries temporal state.
                if s != ("a", 1):
                    inputs[:, slot] = (ord(s[0])+s[1])*.001
                maps.extend([[0, slot, 0, 0, 0], [1, slot, 1, 1, 1]])
                labels.extend([0., 1.])
                refs.append({"sequence": s[0], "slot": slot, "window": s[1], "ids": np.arange(2)})
        y = np.asarray(labels, np.float32)
        candidate = np.ones(len(y), bool)
        maps = np.asarray(maps, np.int64).T
        return {"inputs": inputs, "maps": maps, "labels": y, "candidate": candidate, "teacher": y*.8+.1,
                "reset": np.asarray(reset), "active": active, "refs": refs,
                "device": {"maps": torch.from_numpy(maps), "labels": torch.from_numpy(y),
                           "candidate": torch.from_numpy(candidate), "teacher": torch.from_numpy(y*.8+.1)}}


def fake_evaluate(model, *args, **kwargs):
    # Stable metric sufficient to exercise best/last checkpoint separation.
    selected = {"threshold": 0., "overall": {"iou": .3, "recall": .4}, "high_density": {"recall": .2}}
    return {"selected": selected, "cost": {}, "seconds": 0.}


class LongTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def temp(self):
        base = ROOT/"results/runs/longtrain_unit_tests"
        base.mkdir(exist_ok=True)
        context = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(context.cleanup)
        return Path(context.name)

    def test_paired_model_preserves_all_initial_parameters_and_hyperparameters(self):
        a, b = tiny_profile("ann_linear"), tiny_profile("snn_linear")
        self.assertEqual(a["training"], b["training"])
        torch.manual_seed(42); am = make_model("ann_linear", a, .1)
        torch.manual_seed(42); bm = make_model("snn_linear", b, .1)
        self.assertTrue(same(am.state_dict(), bm.state_dict()))
        voltage = torch.tensor([-.3, 0., .1, .4, 1.])
        spike, residual = bm._lif(voltage)
        self.assertTrue(torch.equal(spike, spike.floor()))
        self.assertFalse(torch.equal(spike, am._lif(voltage)[0]))
        self.assertTrue(torch.equal(residual, voltage-bm.config.threshold*spike))

    def test_schedule_covers_windows_in_order_and_resets_replacement(self):
        s = Store(self.temp())
        plan = schedule(s, "train_main", 2, epoch=3)
        seen = {n: [] for n in s.names("train_main")}
        for slots, resets in plan:
            for i, entry in enumerate(slots):
                if entry:
                    seen[entry[0]].append(entry[1])
                    self.assertEqual(resets[i], entry[1] == 0)
        self.assertEqual(seen, {"a": [0, 1, 2], "b": [0], "c": [0, 1], "empty": []})

    def test_mid_epoch_resume_exact_model_optimizer_rng_and_metrics_all_arms(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                root = self.temp()
                data = Data(root)
                with patch.object(runtime, "profile_for", tiny_profile), patch.object(runtime, "evaluate", fake_evaluate), \
                     patch.object(runtime, "final_health", return_value={"status": "passed"}):
                    with PauseControl(root) as pause:
                        runtime.run_arm(data, root/"uninterrupted", arm, 2, pause, import_original=False)
                        r = runtime.run_arm(data, root/"interrupted", arm, 2, pause,
                                            stop_after_batches=2, import_original=False)
                        self.assertEqual(r["status"], "paused")
                        saved = load(root/"interrupted/last.pt")
                        self.assertEqual(saved["cursor"]["batch"], 2)
                        self.assertIsNotNone(saved["stream_state"])
                        runtime.run_arm(data, root/"interrupted", arm, 2, pause, import_original=False)
                a, b = [load(root/p/"last.pt") for p in ("uninterrupted", "interrupted")]
                for key in ("model", "optimizer", "scheduler", "rng", "cursor", "best_model"):
                    self.assertTrue(same(a[key], b[key]), key)
                for ae, be in zip(a["report"]["history"], b["report"]["history"]):
                    for key in ("loss", "loss_components", "events", "windows", "updates", "stability", "evaluation"):
                        self.assertTrue(same(ae[key], be[key]), key)

    def test_atomic_failure_preserves_previous_checkpoint(self):
        path = self.temp()/"last.pt"
        save_atomic(path, {"step": 3})
        with patch("os.replace", side_effect=OSError("simulated interrupted replacement")):
            with self.assertRaises(OSError):
                save_atomic(path, {"step": 4})
        self.assertEqual(load(path)["step"], 3)

    def test_lock_released_and_existing_lock_file_is_not_stale_owner(self):
        path = self.temp()/"lock"
        with process_lock(path):
            with self.assertRaises(RuntimeError):
                with process_lock(path):
                    pass
        with process_lock(path):
            self.assertTrue(path.exists())

    def test_random_states_survive_safe_serialization(self):
        path = self.temp()/"rng.pt"
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        save_atomic(path, all_rng())
        expected = (random.random(), np.random.rand(), torch.rand(5))
        set_rng(load(path))
        self.assertTrue(same(expected, (random.random(), np.random.rand(), torch.rand(5))))

    def test_failed_step_does_not_replace_committed_checkpoint(self):
        root = self.temp(); data = Data(root)
        with patch.object(runtime, "profile_for", tiny_profile), \
             patch.object(runtime, "update_step", side_effect=FloatingPointError("synthetic guard")):
            with PauseControl(root) as pause:
                result = runtime.run_arm(data, root/"run", "ann_linear", 1, pause, import_original=False)
        self.assertEqual(result["status"], "failed")
        saved = load(root/"run/last.pt")
        self.assertEqual(saved["cursor"]["batch"], 0)
        self.assertEqual(saved["report"]["completed_epochs"], 0)
        self.assertTrue((root/"run/failure_context.pt").exists())

    def test_changed_identity_refuses_resume(self):
        root = self.temp(); data = Data(root)
        with patch.object(runtime, "profile_for", tiny_profile):
            with PauseControl(root) as pause:
                runtime.run_arm(data, root/"run", "ann_linear", 1, pause, stop_after_batches=1, import_original=False)
                with patch.object(runtime, "sources", return_value={"changed": "yes"}):
                    with self.assertRaisesRegex(ValueError, "Resume code"):
                        runtime.run_arm(data, root/"run", "ann_linear", 1, pause, import_original=False)

    def test_pause_file_is_observed_at_safe_boundary(self):
        root = self.temp(); data = Data(root)
        (root/"PAUSE").touch()
        with patch.object(runtime, "profile_for", tiny_profile):
            with PauseControl(root) as pause:
                r = runtime.run_arm(data, root/"run", "ann_linear", 1, pause, import_original=False)
        self.assertEqual(r["status"], "paused")
        self.assertEqual(load(root/"run/last.pt")["cursor"]["batch"], 0)

    def test_pause_during_dev_does_not_repeat_training_or_scheduler_step(self):
        root = self.temp(); data = Data(root)
        with patch.object(runtime, "profile_for", tiny_profile), \
             patch.object(runtime, "final_health", return_value={"status": "passed"}):
            with PauseControl(root) as pause:
                with patch.object(runtime, "evaluate", side_effect=PauseRequested):
                    r = runtime.run_arm(data, root/"run", "ann_linear", 1, pause, import_original=False)
                before = load(root/"run/last.pt")
                self.assertEqual(before["cursor"]["phase"], "evaluate")
                with patch.object(runtime, "evaluate", fake_evaluate), \
                     patch.object(runtime, "update_step", side_effect=AssertionError("Training repeated")):
                    r = runtime.run_arm(data, root/"run", "ann_linear", 1, pause, import_original=False)
                after = load(root/"run/last.pt")
        self.assertEqual(r["status"], "complete")
        self.assertTrue(same(before["model"], after["model"]))
        self.assertEqual(after["scheduler"]["last_epoch"], before["scheduler"]["last_epoch"]+1)
        for key in before["optimizer"]["state"]:
            self.assertTrue(same(before["optimizer"]["state"][key], after["optimizer"]["state"][key]))

    def test_failure_requires_review_instead_of_endless_auto_retry(self):
        root = self.temp(); data = Data(root)
        with patch.object(runtime, "profile_for", tiny_profile), \
             patch.object(runtime, "update_step", side_effect=FloatingPointError("synthetic guard")) as step:
            with PauseControl(root) as pause:
                runtime.run_arm(data, root/"run", "ann_linear", 1, pause, import_original=False)
                n = step.call_count
                result = runtime.run_arm(data, root/"run", "ann_linear", 1, pause, import_original=False)
                self.assertEqual(step.call_count, n)
        self.assertEqual(result["status"], "failed_requires_review")

    def test_os_lock_released_after_process_is_killed(self):
        root = self.temp()
        code = ("import sys,time; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                "from wangdao.h2pro_longtrain_v1.storage import process_lock; "
                "c=process_lock(Path(sys.argv[2])/'lock'); c.__enter__(); "
                "(Path(sys.argv[2])/'ready').touch(); time.sleep(60)")
        child = subprocess.Popen([sys.executable, "-B", "-c", code, str(ROOT/"software"), str(root)],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic()+20
            while not (root/"ready").exists() and time.monotonic() < deadline and child.poll() is None:
                time.sleep(.05)
            self.assertTrue((root/"ready").exists())
            with self.assertRaises(RuntimeError):
                with process_lock(root/"lock"):
                    pass
            child.kill()
            child.wait(timeout=5)
            with process_lock(root/"lock"):
                pass
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_comparison_handles_no_positive_predictions_and_partial_queue(self):
        selected = {"overall": {"iou": 0., "precision": None, "recall": 0.}, "high_density": {"recall": 0.}}
        evaluation = {"selected": selected, "cost": {},
                      "frozen_teacher": {"overall": {"iou": .8, "precision": .9, "recall": .85}}}
        report = {"status": "complete", "completed_epochs": 30, "history": [],
                  "best": {"epoch": 1, "evaluation": evaluation}}
        r = runtime.comparison(self.temp(), [{**report, "arm": a} for a in ("ann_linear", "snn_linear")])
        self.assertEqual(r["status"], "incomplete")
        self.assertIsNone(r["models"]["ann_linear"]["delta_vs_frozen_teacher"]["precision"])
        self.assertIsNone(r["snn_minus_ann_linear_at_independently_selected_dev_best"]["precision"])


if __name__ == "__main__":
    unittest.main()
