"""Synthetic release guards; never access Validation or Test event files."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wangdao import phase1_common as common
from wangdao import causal_training as training
from wangdao.causal import CausalModel, CausalConfig
from freeze_phase1 import freeze, validate


class ReleaseTests(unittest.TestCase):
    def test_configuration_budget_counts_recipes_not_seeds(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)):
            first = common.register_experiment({"variant": 0, "seed": 42})
            self.assertEqual(first, common.register_experiment({"variant": 0, "seed": 43}))
            self.assertIsNone(common.register_experiment({"variant": 100}, smoke=True))
            for variant in range(1, 12):
                common.register_experiment({"variant": variant})
            with self.assertRaises(ValueError):
                common.register_experiment({"variant": 12})
            self.assertFalse((Path(directory) / "results/runs/phase1_registry.lock").exists())

    def test_empty_pr_curve_is_undefined_not_perfect(self):
        import numpy as np
        result = common.pr_curve(np.array([], dtype=int), np.array([], dtype=float))
        self.assertEqual(result["thresholds"], [])
        self.assertIsNone(result["average_precision"])

    def test_smoke_without_positives_still_roundtrips_not_frozen(self):
        config = training.TrainConfig(smoke=True)
        dev = {"curve": {"0.28": {"event": {"iou": None}}}}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "smoke"
            with patch.object(training, "role_paths", return_value=[Path("synthetic.npz")]), \
                 patch.object(training, "prefetch_sequences", return_value=iter([
                     (Path("synthetic.npz"), (np.array([[1., 1., 1., 0.]]), None, np.zeros(1), None), 0.)])), \
                 patch.object(training, "evaluate_paths", return_value=dev):
                report = training.train_experiment(run, CausalConfig(), config, torch.device("cpu"))
            self.assertEqual(report["decision"], "smoke_only")
            self.assertIsNone(report["primary_metrics"]["event"]["iou"])
            self.assertTrue((run / report["best_checkpoint"]).is_file())

    def report_fixture(self, checkpoint):
        index_path = common.ROOT / "data/index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        counts = {r["path"]: r["events"] for r in index["files"]}
        cfg = CausalConfig()
        training.save_checkpoint(checkpoint, CausalModel(cfg), cfg,
                                 {"smoke": False, "data_index_sha256": common.sha256(index_path)})
        return {"status": "complete", "data_role": "train_dev", "smoke": False,
                "metric_protocol": common.PROTOCOL_ID, "data_index_sha256": common.sha256(index_path),
                "checkpoint": str(checkpoint), "checkpoint_sha256": common.sha256(checkpoint),
                "model_config": cfg.to_dict(),
                "selection": {"eligible": True, "threshold": .28, "metrics": {"synthetic_test": True}},
                "evaluation": {"partial": False, "sequence_count": 10,
                               "per_sequence": [{"sequence": name, "events": counts["train/" + name],
                                                 "partial": False} for name in index["roles"]["train_dev"]]}}

    def test_freeze_rejects_smoke_partial_and_mutated_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            report = self.report_fixture(base / "model.pt")
            for kind in ("smoke", "partial", "duplicates"):
                changed = copy.deepcopy(report)
                if kind == "smoke":
                    changed["smoke"] = True
                elif kind == "partial":
                    changed["evaluation"]["per_sequence"][0]["events"] -= 1
                else:
                    changed["evaluation"]["per_sequence"][-1] = changed["evaluation"]["per_sequence"][0]
                path = base / (kind + ".json")
                common.write_json(path, changed)
                with self.assertRaises(ValueError):
                    freeze(path, base / (kind + "_frozen.json"))
            path = base / "valid.json"
            common.write_json(path, report)
            manifest = base / "frozen.json"
            self.assertEqual(freeze(path, manifest)["status"], "frozen")
            # A different valid checkpoint is sufficient to trigger the integrity guard.
            other = base / "other.pt"
            training.save_checkpoint(other, CausalModel(CausalConfig()), CausalConfig(), {"smoke": True})
            changed = json.loads(manifest.read_text(encoding="utf-8"))
            changed["checkpoint"] = "other.pt"
            bad = base / "changed.json"
            common.write_json(bad, changed)
            with patch("freeze_phase1.evaluate_paths") as scoring:
                with self.assertRaises(ValueError):
                    validate(bad, base / "validation.json", torch.device("cpu"))
                scoring.assert_not_called()

    def test_wrong_baseline_role_or_protocol_rejected(self):
        with self.assertRaises(ValueError):
            training.select_evaluation({"partial": False, "sequence_count": 10},
                                       {"selection": {"metrics": {"event": {"iou": 1}}},
                                        "data_role": "validation"}, "C2")

    def test_baseline_requires_exact_sequences_not_just_same_count(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.report_fixture(Path(directory) / "model.pt")
            metrics = {"event": {"iou": .8}, "localization": {"pd": .9, "fa": .001}}
            report["selection"]["metrics"] = metrics
            result = copy.deepcopy(report["evaluation"])
            result["curve"] = {"0.28": metrics}
            self.assertTrue(training.select_evaluation(result, report, "C2")["eligible"])
            result["per_sequence"][0]["sequence"] = "train_999.npz"
            with self.assertRaises(ValueError):
                training.select_evaluation(result, report, "C2")


if __name__ == "__main__":
    unittest.main()
