"""Bounded selection must preserve the paused control and existing checkpoint identity."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/"software"))
import test_h2pro_longtrain as previous
import test_h2pro_longtrain_v1_1 as advisory
from wangdao import h2pro_shorttrain as short
from wangdao.h2pro_longtrain_v1_1 import runtime, migration
from wangdao.h2pro_longtrain_v1.storage import load, save_atomic, PauseControl, process_lock
from wangdao.h2pro_snn.stream_data import atomic_json


def evaluation(model, *args, **kwargs):
    result = previous.fake_evaluate(model)
    result["selected"]["overall"]["precision"] = .5
    return result


class ShortTrainingTests(unittest.TestCase):
    temp = previous.LongTrainingTests.temp

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        root, data, saved = advisory.AdvisoryTrainingTests.fixture(self, "ann_linear")
        suite = root/"short"
        for arm in short.CONTROLS:
            if arm == "snn_linear":
                _, _, saved = advisory.AdvisoryTrainingTests.fixture(self, arm)
            identity = migration.make_identity(data, arm, advisory.new_tiny_profile(arm), {"fixture": True})
            migrated = migration.migrate_payload(saved, identity)
            save_atomic(suite/arm/"last.pt", migrated)
            atomic_json(suite/arm/"training.json", migrated["report"])
        hold = suite/"ann_state"
        hold.mkdir()
        (hold/"last.pt").write_bytes(b"held checkpoint must not even be loaded")
        atomic_json(hold/"training.json", {"arm": "ann_state", "status": "paused", "completed_epochs": 21,
                                           "best": None, "history": []})
        return root, suite, data

    def train(self, suite, data, epochs=2, arms=short.CONTROLS):
        with patch.object(short, "profile_for", advisory.new_tiny_profile), \
             patch.object(runtime, "profile_for", advisory.new_tiny_profile), \
             patch.object(runtime, "evaluate", evaluation), \
             patch.object(runtime, "final_health", return_value={"status": "passed"}), PauseControl(suite) as pause:
            return short.run_queue(data, suite, epochs, pause, arms)

    def test_only_selected_controls_reach_budget_and_held_files_never_change(self):
        _, suite, data = self.fixture()
        held = {p: p.read_bytes() for p in (suite/"ann_state").iterdir()}
        result = self.train(suite, data)
        self.assertEqual(result["status"], "short_budget_complete")
        self.assertFalse(result["automatic_extension"])
        for arm in short.CONTROLS:
            saved = load(suite/arm/"last.pt")
            self.assertEqual(saved["report"]["completed_epochs"], 2)
            self.assertEqual(saved["cursor"], {"epoch": 3, "batch": 0, "phase": "complete"})
        self.assertTrue(all(p.read_bytes() == b for p, b in held.items()))
        status = json.loads((suite/"progress.json").read_text())
        self.assertEqual(status["excluded_controls"], ["ann_state"])
        self.assertTrue(status["review_required"])

    def test_completed_short_budget_does_not_repeat_updates(self):
        _, suite, data = self.fixture()
        self.train(suite, data)
        before = {a: (suite/a/"last.pt").read_bytes() for a in short.CONTROLS}
        with patch.object(runtime, "update_step", side_effect=AssertionError("Unexpected extra update")):
            result = self.train(suite, data)
        self.assertEqual(result["status"], "short_budget_complete")
        self.assertTrue(all((suite/a/"last.pt").read_bytes() == b for a, b in before.items()))

    def test_single_control_selection_preserves_the_other_linear_control(self):
        _, suite, data = self.fixture()
        before = (suite/"ann_linear/last.pt").read_bytes()
        result = self.train(suite, data, epochs=1, arms=("snn_linear",))
        self.assertEqual(result["selected_controls"], ["snn_linear"])
        self.assertEqual(result["status"], "short_budget_complete")
        self.assertEqual((suite/"ann_linear/last.pt").read_bytes(), before)

    def test_pause_does_not_start_the_next_control(self):
        _, suite, data = self.fixture()
        before = {a: (suite/a/"last.pt").read_bytes() for a in short.CONTROLS}
        with patch.object(short, "profile_for", advisory.new_tiny_profile), PauseControl(suite) as pause, \
             patch.object(short, "run_arm", side_effect=AssertionError("Should remain paused")):
            pause.requested = True
            result = short.run_queue(data, suite, 2, pause)
        self.assertEqual(result["status"], "paused")
        self.assertTrue(all((suite/a/"last.pt").read_bytes() == b for a, b in before.items()))

    def test_all_selected_identities_checked_before_starting_any_control(self):
        _, suite, data = self.fixture()
        damaged = load(suite/"snn_linear/last.pt")
        damaged["identity"]["sources"] = {}
        save_atomic(suite/"snn_linear/last.pt", damaged)
        with patch.object(short, "profile_for", advisory.new_tiny_profile), PauseControl(suite) as pause, \
             patch.object(short, "run_arm", side_effect=AssertionError("No updates before validation")):
            with self.assertRaisesRegex(ValueError, "checkpoint source"):
                short.run_queue(data, suite, 2, pause)
            for arms in ((), ("ann_state",), ("ann_linear", "ann_linear")):
                with self.assertRaises(ValueError):
                    short.validate_plan(data, suite, 2, arms)

    def test_check_is_read_only_and_does_not_clear_pause(self):
        _, suite, data = self.fixture()
        (suite/"PAUSE").write_text("keep paused")
        before = {p: p.read_bytes() for p in suite.rglob("*") if p.is_file()}
        with patch.object(short, "profile_for", advisory.new_tiny_profile), \
             patch.object(short, "setup_device", return_value=torch.device("cpu")), \
             patch.object(short, "ResidentData", return_value=data), \
             patch.object(short, "run_arm", side_effect=AssertionError("Read-only check")):
            result = short.execute("unused", "unused", suite, 2, check_only=True)
            self.assertEqual(result["status"], "ready")
            self.assertFalse(result["formal_training_started"])
            with process_lock(suite/".suite.lock"):
                with self.assertRaises(RuntimeError):
                    short.execute("unused", "unused", suite, 2, check_only=True)
        self.assertTrue(all(p.read_bytes() == b for p, b in before.items()))


if __name__ == "__main__":
    unittest.main()
