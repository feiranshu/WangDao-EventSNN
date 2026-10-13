"""Readout study audit rejects incomplete records and keeps accuracy admission separate from timing."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.select_snn_readout import accuracy_decision, audit_history, selection_key, timing_summary
from scripts.train_snn_readout import PAIR_PROTOCOL, VARIANTS


def training_record():
    event = {"iou": .6, "f1": .75, "true_positive": 6, "false_negative": 4,
             "false_positive": 0, "true_negative": 90}
    selection = {"event": event, "threshold": .3, "files": ["dev"], "seconds": 1.,
                 "scan": {"best_event": event, "best_threshold": .3, "grid_min": .01,
                          "grid_max": .99, "grid_step": .01, "evaluated_thresholds": 99}}
    history = [{"epoch": i, "blocks": 17, "events": 50, "updates": 3, "loss": .4,
                "train_s": 2., "input_wait_s": .5,
                "lr": 1e-5 + (1e-3 - 1e-5) * (1 + math.cos(math.pi * (i - 1) / 30)) / 2,
                "selection": copy.deepcopy(selection)} for i in range(1, 31)]
    return {"status": "ok", "pilot": False, "epochs": 30, "variant": "baseline",
            "protocol": PAIR_PROTOCOL, "identity": {}, "model_kwargs": VARIANTS["baseline"],
            "priors": [.1, .2], "history": history, "primary_metrics": event,
            "evaluation": {"event": event},
            "best": {"epoch": 1, "threshold": .3, "event": event, "key": selection_key(history[0])}}


class ReadoutSelectionTests(unittest.TestCase):
    def test_history_selection_budget_and_schedule(self):
        def audit(record):
            return audit_history(record, "baseline", {}, 17, 50, (.1, .2), ["dev"], 100, 10)
        original = training_record()
        self.assertEqual(audit(original)["epoch"], 1)
        for key, value in (("epoch", 8), ("updates", 4), ("events", 49), ("lr", .01)):
            broken = copy.deepcopy(original)
            broken["history"][2][key] = value
            with self.assertRaises(ValueError):
                audit(broken)
        broken = copy.deepcopy(original)
        broken["history"].pop()
        with self.assertRaises(ValueError):
            audit(broken)
        broken = copy.deepcopy(original)
        broken["best"]["epoch"] = 30
        with self.assertRaises(ValueError):
            audit(broken)
        broken = copy.deepcopy(original)
        broken["history"][3]["selection"]["files"] = ["test"]
        with self.assertRaises(ValueError):
            audit(broken)

    def test_accuracy_exclusion_does_not_require_or_invent_speed(self):
        def row(iou, fa=0.):
            return {"event": {"iou": iou}, "localization": {"pd": .8, "fa": fa}}
        excluded = accuracy_decision({"baseline": row(.59), "membrane": row(.57)}, row(.73))
        self.assertTrue(excluded["both_paths_excluded_by_iou"])
        self.assertEqual(excluded["next"], "stop_readout_pair_retain_snn_mainline")
        surviving = accuracy_decision({"baseline": row(.73)}, row(.73))
        self.assertFalse(surviving["both_paths_excluded_by_iou"])
        self.assertFalse(surviving["quantization_admitted"])
        failing_fa = accuracy_decision({"baseline": row(.73, 1e-12)}, row(.73))
        self.assertFalse(failing_fa["checks_vs_frozen_hybrid"]["baseline"]["localization_pass"])

    def test_timings_preserve_loader_groups_and_completed_work(self):
        history = training_record()["history"]
        for row in history[20:]:
            row.update(loader="packed_scatter_v1", train_s=1., input_wait_s=.1)
        summary = timing_summary(history)
        self.assertEqual(summary["sum_completed_epoch_train_and_selection_s"], 80.)
        self.assertEqual(len(summary["by_loader"]["original"]["epochs"]), 20)
        self.assertEqual(summary["by_loader"]["packed_scatter_v1"]["median_input_wait_s"], .1)


if __name__ == "__main__":
    unittest.main()
