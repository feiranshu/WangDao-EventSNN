"""数据指标与 H2 基线的最小自动测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))

from wangdao.h2.constants import PROPOSAL_DIMENSIONS, REFINER_DIMENSIONS
from wangdao.h2.numpy_model import load_export, quantized_mlp, run_pipeline
from wangdao.h2.torch_model import load_exported_models
from wangdao.metrics import (
    event_metrics,
    evuav_paper_metric_view,
    localization_metrics,
)


WEIGHT = ROOT / "model_runtime" / "weights" / "H2_D1_seed42.npz"
WEIGHT_SHA256 = "21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_events(count: int = 1000) -> np.ndarray:
    generator = np.random.default_rng(2026)
    dtype = np.dtype([
        ("x", "<i2"), ("y", "<i2"), ("t", "<f8"), ("p", "u1"),
        ("label", "u1"), ("name", "<i2"),
    ])
    events = np.empty(count, dtype=dtype)
    events["x"] = generator.integers(0, 346, count, dtype=np.int16)
    events["y"] = generator.integers(0, 260, count, dtype=np.int16)
    events["t"] = np.linspace(0.0, 7999.0, count)
    events["p"] = generator.integers(0, 2, count, dtype=np.uint8)
    events["label"] = 0
    events["name"] = 0
    return events


class MetricTests(unittest.TestCase):
    def test_event_metrics(self) -> None:
        perfect = event_metrics(
            np.asarray([0, 1, 1, 0], dtype=np.uint8),
            np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
            0.5,
        )
        self.assertEqual(perfect["iou"], 1.0)
        self.assertEqual(perfect["f1"], 1.0)

        wrong = event_metrics(
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([0.9, 0.1], dtype=np.float32),
            0.5,
        )
        self.assertEqual(wrong["iou"], 0.0)
        self.assertEqual(wrong["f1"], 0.0)

    def test_localization_metrics(self) -> None:
        result = localization_metrics(
            np.asarray([1.0, 2.0, 51.0, 52.0]),
            np.asarray([10, 11, 20, 21]),
            np.asarray([10, 11, 20, 21]),
            np.asarray([1, 1, 0, 0], dtype=np.uint8),
            np.asarray([1, 1, 0, 0]),
            np.asarray([0.9, 0.9, 0.1, 0.1]),
            threshold=0.5,
        )
        self.assertEqual(result["pd"], 1.0)
        self.assertEqual(result["fa"], 0.0)

    def test_evuav_paper_metric_mapping_and_fa_unit(self) -> None:
        event = event_metrics(
            np.asarray([1, 1, 0, 0], dtype=np.uint8),
            np.asarray([0.9, 0.1, 0.2, 0.2], dtype=np.float32),
            0.5,
        )
        view = evuav_paper_metric_view(event, {"pd": 0.75, "fa": 0.000163})
        self.assertEqual(view["paper_acc_target_recall"], event["recall"])
        self.assertNotEqual(view["paper_acc_target_recall"], event["accuracy"])
        self.assertAlmostEqual(view["fa_x_1e4"], 1.63)


class H2Tests(unittest.TestCase):
    def test_locked_weight_and_quantized_forward(self) -> None:
        self.assertEqual(sha256_file(WEIGHT), WEIGHT_SHA256)
        model = load_export(WEIGHT)
        proposal_dimensions = tuple(
            [model.proposal_layers[0][0].shape[0]]
            + [layer[0].shape[1] for layer in model.proposal_layers]
        )
        refiner_dimensions = tuple(
            [model.refiner_layers[0][0].shape[0]]
            + [layer[0].shape[1] for layer in model.refiner_layers]
        )
        self.assertEqual(proposal_dimensions, PROPOSAL_DIMENSIONS)
        self.assertEqual(refiner_dimensions, REFINER_DIMENSIONS)

        generator = np.random.default_rng(42)
        features = generator.integers(0, 256, (17, 128), dtype=np.uint8)
        numpy_output = quantized_mlp(features, model.proposal_layers)
        proposal, _ = load_exported_models(WEIGHT)
        with torch.no_grad():
            torch_output = proposal(torch.from_numpy(features)).numpy()
        numpy_probability = 1.0 / (1.0 + np.exp(-numpy_output))
        torch_probability = 1.0 / (1.0 + np.exp(-torch_output))
        self.assertLess(float(np.max(np.abs(numpy_probability - torch_probability))), 0.005)

    def test_end_to_end_shape(self) -> None:
        result = run_pipeline(synthetic_events(), load_export(WEIGHT))
        self.assertEqual(result["probabilities"].shape, (1000,))
        self.assertEqual(result["heatmaps"].shape, (80, 34, 44))
        self.assertTrue(np.all(np.isfinite(result["probabilities"])))


if __name__ == "__main__":
    unittest.main()
