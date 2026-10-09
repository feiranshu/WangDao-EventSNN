from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step2_common import CandidateCorpus, FEATURE_GROUPS, export_metrics, make_float_refiner  # noqa: E402
from run_step2 import parser as step2_parser  # noqa: E402
from step2_training import (  # noqa: E402
    Step2Sampler,
    _append_progress,
    _progress_row,
    _save_checkpoint,
    candidate_loss,
    load_checkpoint_package,
    stc_weights_sparse,
)


def synthetic_record():
    labels = np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    high = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    probability = np.asarray([0.1, 0.8, 0.9, 0.2, 0.7, 0.8, 0.6, 0.1], dtype=np.float32)
    return {
        "file": "train_synthetic.npz",
        "features": np.tile(np.arange(32, dtype=np.uint8), (8, 1)),
        "labels": labels,
        "baseline_probability": probability,
        "windows": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
        "high_density": high,
        "hard_negative": np.asarray([0, 0, 1, 0, 0, 1, 0, 0], dtype=np.uint8),
        "hard_positive": np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
        "x": np.arange(8, dtype=np.uint16),
        "y": np.zeros(8, dtype=np.uint16),
        "t_ms": np.arange(8, dtype=np.int32),
    }


class Step2Tests(unittest.TestCase):
    def test_feature_groups_partition_32_inputs(self):
        columns = [value for group in FEATURE_GROUPS.values() for value in group]
        self.assertEqual(sorted(columns), list(range(32)))
        self.assertEqual(len(columns), len(set(columns)))

    def test_sampler_has_exact_protocol_ratios_and_is_deterministic(self):
        corpus = CandidateCorpus([synthetic_record()])
        first = Step2Sampler(corpus, 42)
        second = Step2Sampler(corpus, 42)
        first.begin_epoch(1)
        second.begin_epoch(1)
        a = first.batch(0)
        b = second.batch(0)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 4096)
        report = first.report()
        self.assertEqual(report["stratum_counts"]["high_hard_positive"], 256)
        self.assertEqual(report["stratum_counts"]["ordinary_positive"], 512)
        self.assertEqual(report["stratum_counts"]["high_negative"], 768)
        self.assertEqual(report["stratum_counts"]["ordinary_negative"], 768)

    def test_null_loss_is_exactly_a2_1_loss(self):
        raw = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.4, 0.1, 0.2, 0.6]])
        labels = torch.tensor([1.0, 0.0])
        high = torch.tensor([True, False])
        first = candidate_loss("A2-1", raw, labels, high)
        null = candidate_loss("A2-1", raw, labels, high)
        torch.testing.assert_close(first["total"], null["total"], rtol=0, atol=0)

    def test_density_weighting_changes_focal_contribution(self):
        raw = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.4, 0.1, 0.2, 0.6]])
        labels = torch.tensor([1.0, 0.0])
        high = torch.tensor([True, False])
        base = candidate_loss("A2-1", raw, labels, high)
        weighted = candidate_loss("A2-2", raw, labels, high)
        self.assertNotEqual(float(base["classification"]), float(weighted["classification"]))

    def test_stc_assigns_more_weight_to_supported_events(self):
        probability = np.asarray([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        x = np.asarray([10, 10, 10, 100])
        y = np.asarray([10, 10, 10, 100])
        t = np.asarray([10, 11, 12, 100])
        result = stc_weights_sparse(probability, x, y, t)
        self.assertGreater(float(result["released"][:3].mean()), float(result["released"][3]))
        self.assertTrue(np.all(np.isfinite(result["released"])))

    def test_metrics_export_handles_no_eligible_working_point(self):
        def score(iou, precision, recall):
            return {
                "event": {"iou": iou, "precision": precision, "recall": recall},
                "localization": {"pd": recall, "fa": 0.1},
            }

        evaluation = {
            "curve": {
                "0.2": score(0.40, 0.50, 0.60),
                "0.3": score(0.45, 0.55, 0.65),
            },
            "density_curves": {},
            "per_sequence": [
                {
                    "file": "train_089.npz",
                    "high_density": True,
                    "events": 10,
                    "candidate_events": 5,
                    "scores": {
                        "0.2": score(0.40, 0.50, 0.60),
                        "0.3": score(0.45, 0.55, 0.65),
                    },
                }
            ],
            "average_precision": 0.5,
            "target_events": 3,
            "sequence_count": 1,
            "metric_protocol": "test",
            "_pr_points": (
                np.asarray([0.9, 0.3]),
                np.asarray([1.0, 0.5]),
                np.asarray([0.5, 1.0]),
            ),
        }
        view = {"eligible": False, "threshold": None, "reason": "fairness constraints"}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "metrics"
            export_metrics(target, evaluation, view)
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reporting_threshold"], 0.3)
            self.assertEqual(
                summary["reporting_threshold_kind"],
                "diagnostic_max_iou_not_an_eligible_working_point",
            )
            self.assertTrue((target / "per_sequence.csv").is_file())

    def test_every_epoch_progress_is_persisted(self):
        epoch_row = {
            "epoch": 3,
            "loss": 0.4,
            "classification_loss": 0.2,
            "soft_iou_loss": 0.4,
            "learning_rate": 1.0e-4,
            "wall_s": 2.5,
            "sampling": {
                "coverage": 0.25,
                "repeat_fraction": 0.75,
                "unique_candidates": 100,
            },
        }
        row = _progress_row(epoch_row, 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_progress.csv"
            _append_progress(path, row)
            _append_progress(path, dict(row, epoch=4))
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("classification_loss", lines[0])
            self.assertTrue(lines[1].startswith("3,20,"))
            self.assertTrue(lines[2].startswith("4,20,"))

    def test_continuation_checkpoint_and_cli_contract(self):
        model = make_float_refiner(42)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch_020.pt"
            _save_checkpoint(
                checkpoint,
                model,
                {"model_id": "A2-2", "seed": 42, "epoch": 20},
                optimizer=optimizer,
                scheduler=scheduler,
            )
            package = load_checkpoint_package(checkpoint)
            self.assertIn("optimizer_state_dict", package)
            self.assertIn("scheduler_state_dict", package)
            self.assertEqual(package["metadata"]["epoch"], 20)
            args = step2_parser().parse_args(
                [
                    "train", "--model", "A2-2", "--output", "continued",
                    "--resume-checkpoint", str(checkpoint), "--epoch-offset", "20",
                    "--epochs", "10", "--run",
                ]
            )
            self.assertEqual(args.epoch_offset, 20)
            self.assertEqual(args.epochs, 10)
            self.assertEqual(args.resume_checkpoint, checkpoint)


if __name__ == "__main__":
    unittest.main()

