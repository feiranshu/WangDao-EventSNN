from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step4_common import (  # noqa: E402
    DensityTransform,
    REGISTERED_CONTINUATION,
    REGISTERED_INITIAL,
    build_model,
    continue10_allowed,
    feature_parts,
    high_density_rank,
    primary_rank,
    quantized_layers,
)
from step4_training import _activate_full_finetune, _optimizer, _print_progress  # noqa: E402
from run_step4 import _drop_time_entry  # noqa: E402


def record(rows: int = 4) -> dict:
    return {
        "features": np.arange(rows * 32, dtype=np.uint8).reshape(rows, 32),
        "history": np.arange(rows * 4, dtype=np.uint8).reshape(rows, 4),
        "window_rate_eps": np.asarray([0, 1000, 20000, 100000][:rows], dtype=np.float32),
    }


def view(iou: float, ap: float, recall: float, fa: float = 1.0) -> dict:
    return {
        "eligible": True, "threshold": 0.3, "average_precision": ap,
        "overall": {"event": {"iou": iou}, "localization": {"fa": fa}},
        "high_density": {"event": {"recall": recall, "iou": iou - 0.2}},
    }


def evaluation(epoch: int, level: int, iou: float, ap: float, recall: float) -> dict:
    return {
        "epoch": epoch, "int8_working_point": view(iou, ap, recall),
        "replacement_gates": {"passes": level == 2},
        "noninferiority": {"passes": level >= 1},
    }


class Step4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.density = DensityTransform(0.0, float(np.log1p(100000)), 4, 1)

    def test_feature_shapes_and_nulls(self) -> None:
        row = record()
        expected = {
            "A4-1": (32, 1), "A4-1-null": (32, 1),
            "A4-2": (32, 4), "A4-2-null": (32, 4),
            "A4-3": (31, 0), "A4-3-control": (32, 0),
            "A4-4": (32, 5), "A4-4-null": (32, 5),
        }
        for model_id, widths in expected.items():
            old, new = feature_parts(row, model_id, self.density)
            self.assertEqual((old.shape[1], new.shape[1]), widths)
            if model_id.endswith("-null"):
                self.assertFalse(np.any(new))
        old, _ = feature_parts(row, "A4-3", self.density)
        np.testing.assert_array_equal(old, np.delete(row["features"], 2, axis=1))

    def test_density_is_clipped_uint8(self) -> None:
        values = self.density.apply(np.asarray([-1, 0, 100000, 1e9]))
        self.assertEqual(values.dtype, np.uint8)
        self.assertEqual(values.shape, (4, 1))
        self.assertEqual(int(values[1, 0]), 0)
        self.assertEqual(int(values[-1, 0]), 255)

    def test_new_columns_are_zero_initialized_and_quantizable(self) -> None:
        for model_id, width in (("A4-1", 33), ("A4-2", 36), ("A4-4", 37)):
            model = build_model(model_id)
            self.assertEqual(model.dimensions[0], width)
            self.assertEqual(float(model.new_input.weight.detach().abs().max()), 0.0)
            layers = quantized_layers(model)
            self.assertEqual(layers[0][0].shape, (width, 64))

    def test_phase_change_preserves_new_optimizer_state(self) -> None:
        model = build_model("A4-1")
        optimizer = _optimizer(model, continuation=False)
        self.assertEqual(len(optimizer.param_groups), 1)
        new_parameter = model.new_input.weight
        x = torch.ones((2, 33), dtype=torch.uint8)
        model(x).sum().backward()
        optimizer.step()
        state_before = optimizer.state[new_parameter]["step"].clone()
        _activate_full_finetune(optimizer, model)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertTrue(torch.equal(state_before, optimizer.state[new_parameter]["step"]))
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0e-5)

    def test_registered_epochs_are_frozen(self) -> None:
        self.assertEqual(REGISTERED_INITIAL, (0, 1, 2, 5, 8, 10))
        self.assertEqual(REGISTERED_CONTINUATION, (11, 12, 15, 18, 20))

    def test_eligibility_precedes_metric_ranking(self) -> None:
        protected = evaluation(10, 1, 0.70, 0.90, 0.50)
        diagnostic = evaluation(1, 0, 0.99, 0.99, 0.99)
        self.assertGreater(primary_rank(protected), primary_rank(diagnostic))
        self.assertGreater(high_density_rank(protected), high_density_rank(diagnostic))

    def test_continue10_requires_three_epoch_trend(self) -> None:
        h2_sequence = {
            "train_089.npz": {"event": {"iou": 0.2}},
            "train_092.npz": {"event": {"iou": 0.5}},
        }
        rows = []
        for epoch, iou, ap in ((8, 0.780, 0.9270), (9, 0.7806, 0.9271), (10, 0.7812, 0.9272)):
            value = view(iou, ap, 0.57)
            value["per_sequence"] = {
                "train_089.npz": {"event": {"iou": 0.2}},
                "train_092.npz": {"event": {"iou": 0.5}},
            }
            rows.append({"epoch": epoch, "int8_working_point": value, "noninferiority": {"passes": True}})
        summary = {"evaluations": rows, "h2_r": {"average_precision": 0.9274, "per_sequence": h2_sequence}}
        self.assertTrue(continue10_allowed(summary)["passes"])

    def test_progress_print_accepts_missing_metrics(self) -> None:
        row = {
            "epoch": 1, "final_epoch": 10, "phase": "new_columns_only", "loss": 0.1,
            "learning_rate_new": 0.001, "learning_rate_old": None,
            "gradient_norm_mean": 1.0, "wall_s": 2.0, "new_input_norm": 0.0,
            "old_input_norm": 1.0, "int8_ap": None, "int8_iou": None,
            "int8_high_recall": None,
        }
        with contextlib.redirect_stdout(io.StringIO()) as output:
            _print_progress("A4-1", 42, row)
        self.assertIn("int8_ap=NA", output.getvalue())

    def test_drop_time_entry_handles_missing_working_point(self) -> None:
        candidate = {"final": {"primary": {"int8_working_point": {"eligible": False}}}}
        control = {"final": {"primary": {"int8_working_point": view(0.78, 0.92, 0.57)}}}
        self.assertFalse(_drop_time_entry(candidate, control)["passes"])

    def test_readme_commands_have_no_backslash_continuations(self) -> None:
        for name in ("README.md", "README_SERVER.md"):
            path = ROOT / name
            if path.exists():
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"\\\s*\n")


if __name__ == "__main__":
    unittest.main()
