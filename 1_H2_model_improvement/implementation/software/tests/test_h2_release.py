"""Release guards for Steps 4-7; never read Validation or Test event files."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao import h2_release as release
from wangdao.phase1_common import PROTOCOL_ID, sha256


class H2ReleaseTests(unittest.TestCase):
    def make_index(self, root: Path):
        names = [f"train_{value:03d}.npz" for value in range(10)]
        index = {"roles": {"train_dev": names, "validation": [f"val_{i:03d}.npz" for i in range(24)],
                           "test": [f"test_{i:03d}.npz" for i in range(24)]},
                 "files": [{"path": "train/" + name, "events": 100 + i,
                            "sha256": "unused"} for i, name in enumerate(names)]}
        path = root / "data/index.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(index), encoding="utf-8")
        return names, path

    def make_accuracy_summary(self, root: Path, seed: int, delta: float,
                              names, index_path: Path) -> Path:
        run = root / f"run_{seed}"
        run.mkdir()
        checkpoint = run / "model.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        report = {
            "status": "complete", "smoke": False, "data_role": "train_main+train_dev",
            "test_used": False, "validation_used": False, "metric_protocol": PROTOCOL_ID,
            "seed": seed, "recipe_id": "same-recipe", "recipe": {"variant": "A1"},
            "provenance": {"index_sha256": sha256(index_path)},
            "final_evaluation": {"partial": False, "sequence_count": 10,
                                 "per_sequence": [{"file": name, "events": 100 + i}
                                                  for i, name in enumerate(names)]},
            "candidate_fair_working_point": {"eligible": True, "threshold": .4,
                                             "metrics": {"event": {"iou": .8 + delta}}},
            "h2_fair_working_point": {"eligible": True, "threshold": .3,
                                      "metrics": {"event": {"iou": .8}}},
            "selected_iou_delta_vs_h2_r": delta,
            "best_checkpoint": checkpoint.name,
            "best_checkpoint_sha256": sha256(checkpoint),
        }
        path = run / "summary.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_accuracy_freeze_gate_requires_three_consistent_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names, index_path = self.make_index(root)
            paths = [self.make_accuracy_summary(root, seed, delta, names, index_path)
                     for seed, delta in ((42, .006), (43, .004), (44, -.001))]
            with patch.object(release, "ROOT", root):
                selected = release.validate_accuracy_summaries(paths)
            self.assertEqual(selected["variant"], "A1")
            self.assertEqual(selected["seeds"], [42, 43, 44])
            self.assertGreater(selected["mean_delta_iou_vs_h2_r"], 0)
            with patch.object(release, "ROOT", root):
                with self.assertRaises(ValueError):
                    release.validate_accuracy_summaries(paths[:2])

    def test_paired_interval_resamples_sequences_not_events(self):
        rows = []
        for offset in range(4):
            rows.append({"models": {
                "A*": {"event": {"tp": 80 + offset, "fp": 10, "fn": 10}},
                "H2-R": {"event": {"tp": 75 + offset, "fp": 12, "fn": 13}},
            }})
        first = release.paired_iou_interval(rows, "A*", "H2-R", repeats=500)
        second = release.paired_iou_interval(rows, "A*", "H2-R", repeats=500)
        self.assertEqual(first, second)
        self.assertGreater(first["lower_95"], 0)
        self.assertIn("Sequences", first["limitation"])

    def test_test_role_requires_matching_validation_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _names, index_path = self.make_index(root)
            manifest = root / "frozen.json"
            manifest.write_text(json.dumps({"status": "frozen", "metric_protocol": PROTOCOL_ID,
                                            "data_index_sha256": sha256(index_path)}), encoding="utf-8")
            with patch.object(release, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Validation"):
                    release.evaluate_frozen(manifest, "test", root / "test.json",
                                            torch.device("cpu"))
            self.assertFalse((root / "frozen_test_started.json").exists())

    def test_protected_role_rejects_development_name_before_access(self):
        with self.assertRaises(ValueError):
            release.protected_role_paths("train_dev")


if __name__ == "__main__":
    unittest.main()
