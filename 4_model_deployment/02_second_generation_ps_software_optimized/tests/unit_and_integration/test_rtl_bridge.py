"""Actual transfer integrity, event routing, and frozen CPU Refiner replay."""

from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
from host_tools.rtl_verification.integrate_rtl_proposal import checked_simulation, prepare_window, run
from host_tools.rtl_verification.run_rtl_simulation import sha256
from wangdao.fixed import FixedH2, FixedH2Config
from wangdao.h2.numpy_model import _refiner_probability, load_export
from wangdao.h2.representation import build_refiner_features, candidate_indices, patch_scores_to_heatmaps
from wangdao.rtl_bridge import ProposalCapture, assert_pipeline_equal, read_proposal_capture, route_window


def capture_text() -> str:
    rows = ["WD_CAPTURE version=1 patches=2 channels=16"]
    words = ("FF800000", "FFFFFFFF", "00000000", "007FFFFF")
    for index in range(32):
        rows.append(f"{index // 16} {index % 16} {100 + index * 3} {words[index % 4]} 0 0 00000001 0 0")
    return "\n".join(rows + ["WD_CAPTURE_END records=32", ""])


class CaptureTests(unittest.TestCase):
    def test_signed_words_and_actual_mode_order(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actual.txt"
            path.write_text(capture_text(), encoding="ascii")
            captured = read_proposal_capture(path, 2)
            self.assertEqual(captured.logits_q[0, 0, :4].tolist(), [-8388608, -1, 0, 8388607])
            self.assertTrue(np.all(captured.logits_q[1] == 1))
            self.assertEqual(captured.cycles.shape, (2, 16))

    def test_corrupt_transfers_fail_closed(self) -> None:
        good = capture_text()
        lines = good.splitlines()
        cases = {
            "empty": "", "truncated": "\n".join(lines[:-2]),
            "no_footer": "\n".join(lines[:-1]), "extra": good + lines[1] + "\n",
            "duplicate": good.replace(lines[2], lines[1]),
            "wrong_channel": good.replace("0 0 100", "0 1 100"),
            "wrong_cycle": good.replace("0 1 103", "0 1 100"),
            "unknown": good.replace("FFFFFFFF", "FFFFFFFX", 1),
            "non_sign_extended": good.replace("FF800000", "00800000", 1),
            "short_word": good.replace("FF800000", "800000", 1),
            "missing_field": good.replace(" 0 0 00000001", " 0 00000001", 1),
            "excess_saturation": good.replace("FF800000 0 0", "FF800000 93 0", 1),
            "wrong_count": good.replace("records=32", "records=31"),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actual.txt"
            for name, content in cases.items():
                with self.subTest(case=name):
                    path.write_text(content, encoding="ascii")
                    with self.assertRaises(ValueError):
                        read_proposal_capture(path, 2)

    def test_simulation_receipt_inputs_and_success_marker_required(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work, vectors = root / "work", root / "vectors"
            work.mkdir()
            vectors.mkdir()
            manifest = vectors / "proposal_manifest.json"
            manifest.write_text("{}", encoding="ascii")
            inputs = vectors / "proposal_inputs.txt"
            inputs.write_text("00", encoding="ascii")
            (work / inputs.name).write_bytes(inputs.read_bytes())
            capture = work / "proposal_actual_passed.txt"
            capture.write_text(capture_text(), encoding="ascii")
            log = work / "tb_proposal_engine_simulate.log"
            log.write_text("WD_PASS top=tb_proposal_engine records=32\n", encoding="ascii")
            receipt = {"status": "ok", "decision": "behavioral_alignment_passed",
                       "source": {"vector_manifest_sha256": sha256(manifest)},
                       "simulation": {"work_directory": "work", "tests": [
                           {"testbench": "tb_proposal_engine", "records": 32, "status": "passed"}]},
                       "actual_output": {"path": "work/proposal_actual_passed.txt", "sha256": sha256(capture),
                           "origin": "DUT_ready_valid_transfers", "format": "WD_CAPTURE_v1", "records_per_mode": 32}}
            with patch("host_tools.rtl_verification.integrate_rtl_proposal.ROOT", root), patch(
                "host_tools.rtl_verification.integrate_rtl_proposal.validate_manifest",
                return_value=({"configuration": {"patches": 2}}, [inputs]),
            ):
                checked_simulation(receipt, vectors)
                for key, value in (("status", "failed"), ("decision", "simulation_failed")):
                    damaged = {**receipt, key: value}
                    with self.assertRaises(ValueError):
                        checked_simulation(damaged, vectors)
                damaged = copy.deepcopy(receipt)
                damaged["actual_output"]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    checked_simulation(damaged, vectors)
                capture.write_text(capture_text().replace("00000001", "00000002", 1), encoding="ascii")
                with self.assertRaises(ValueError):
                    checked_simulation(receipt, vectors)
                capture.write_text(capture_text(), encoding="ascii")
                (work / inputs.name).write_text("01", encoding="ascii")
                with self.assertRaises(ValueError):
                    checked_simulation(receipt, vectors)
                (work / inputs.name).write_bytes(inputs.read_bytes())
                log.write_text("Simulation interrupted\n", encoding="ascii")
                with self.assertRaises(RuntimeError):
                    checked_simulation(receipt, vectors)

    def test_partial_or_other_window_cannot_be_called_complete(self) -> None:
        config = {"window": 0, "patches": 346, "real_patches": 336, "complete_window": True,
                  "synthetic_cascade_patches": 10, "time_steps": 4, "channels": [48, 24, 16]}
        for key, value in (("window", 1), ("real_patches", 2), ("complete_window", False),
                           ("synthetic_cascade_patches", 0)):
            with self.subTest(key=key), patch("host_tools.rtl_verification.integrate_rtl_proposal.validate_manifest",
                return_value=({"configuration": {**config, key: value}, "source": {}}, [])), patch(
                    "host_tools.rtl_verification.integrate_rtl_proposal.hardware_calibration_path") as data:
                with self.assertRaises(ValueError):
                    prepare_window(Path("unused"))
                data.assert_not_called()

    def test_other_data_roles_rejected_before_loading_events(self) -> None:
        config = {"window": 0, "patches": 346, "real_patches": 336, "complete_window": True,
                  "synthetic_cascade_patches": 10, "time_steps": 4, "channels": [48, 24, 16]}
        for role in ("train_main", "validation", "test"):
            with self.subTest(role=role), patch("host_tools.rtl_verification.integrate_rtl_proposal.validate_manifest",
                return_value=({"configuration": config, "source": {"dataset_role": role,
                    "dataset_sequence": "dataset/EV-UAV-dataset/train/train_022.npz"}}, [])), patch(
                    "host_tools.rtl_verification.integrate_rtl_proposal.hardware_calibration_path") as data:
                with self.assertRaisesRegex(ValueError, "HW-calibration"):
                    prepare_window(Path("unused"))
                data.assert_not_called()

    def test_integrator_cannot_replace_corrupt_actual_words_with_golden(self) -> None:
        captured = np.zeros((2, 346, 16), dtype=np.int64)
        captured[0, 10, 0] = 1
        model = Mock()
        model.forward.return_value.logits_q = np.zeros((336, 16), dtype=np.int64)
        metadata = {"proposal_threshold": 0.51, "event_threshold": 0.28, "h2_fixed_configuration": {}}
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / "receipt.json"
            receipt.write_text(json.dumps({"actual_output": {}, "simulation": {}}), encoding="utf-8")
            args = argparse.Namespace(experiment_id="fault", vector_directory=Path("unused"),
                                      simulation_result=receipt, batch_size=4096)
            with patch("host_tools.rtl_verification.integrate_rtl_proposal.prepare_window", return_value=(
                {"source": {}}, [model, model], None, None, None, metadata)), patch(
                    "host_tools.rtl_verification.integrate_rtl_proposal.checked_simulation", return_value=(
                        ProposalCapture(captured, np.zeros_like(captured), np.zeros_like(captured), np.zeros((346, 16))), {})), patch(
                            "host_tools.rtl_verification.integrate_rtl_proposal.route_window") as route:
                result = run(args)
                self.assertEqual(result["status"], "failed")
                self.assertIn("actual Proposal logits differ", result["error"])
                route.assert_not_called()


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.refiner = FixedH2(load_export(ROOT / "model_runtime/weights/H2_D1_seed42.npz"), FixedH2Config())
        cls.events = np.zeros(64, dtype=[("x", "u2"), ("y", "u2"), ("t", "f8"), ("p", "u1")])
        cls.events["t"] = np.arange(64) * 3.0
        cls.events["x"] = np.arange(64) % 32
        cls.events["y"] = np.arange(64) % 24
        cls.events["p"] = np.arange(64) % 2

    def test_q12_route_matches_existing_host_functions_and_batching(self) -> None:
        q = np.full((336, 16), 4096, dtype=np.int64)
        actual = route_window(self.events, 0, q, self.refiner, 0.51, 0.28, 3)
        heatmaps = patch_scores_to_heatmaps(np.full((1, 336, 16), 1 / (1 + np.exp(-1.0)), dtype=np.float32))
        candidates = candidate_indices(self.events, [0], heatmaps, 0.51)
        features = build_refiner_features(self.events, candidates, [0], heatmaps, 0.51)
        output = self.refiner.refiner(features)
        np.testing.assert_array_equal(actual["heatmaps"], heatmaps)
        np.testing.assert_array_equal(actual["candidates"], candidates)
        np.testing.assert_array_equal(actual["features"], features)
        np.testing.assert_array_equal(actual["refiner_logits_q"], output.logits_q)
        np.testing.assert_array_equal(actual["probabilities"], _refiner_probability(output.logits, 2))
        self.assertEqual(len(actual["event_indices"]), 34)
        self.assertEqual(list(actual["refiner_saturations"]), list(output.accumulator_saturations))
        assert_pipeline_equal(actual, route_window(self.events, 0, q, self.refiner, 0.51, 0.28, 1000))

    def test_empty_candidates_excluded_events_and_state_isolation(self) -> None:
        high = np.full((336, 16), 4096, dtype=np.int64)
        first = route_window(self.events, 1, high, self.refiner, 0.51, 0.28)
        empty = route_window(self.events, 0, -high, self.refiner, 0.51, 0.28)
        self.assertEqual(np.asarray(empty["features"]).shape, (0, 32))
        self.assertEqual(np.asarray(empty["refiner_logits_q"]).shape, (0, 4))
        self.assertEqual(len(empty["probabilities"]), 34)
        self.assertFalse(np.any(empty["probabilities"]))
        self.assertFalse(np.any(empty["predictions"]))
        self.assertEqual(empty["refiner_saturations"], (0, 0, 0))
        assert_pipeline_equal(first, route_window(self.events, 1, high, self.refiner, 0.51, 0.28, 1))

    def test_malformed_q12_and_one_bit_mismatch_are_rejected(self) -> None:
        good = np.zeros((336, 16), dtype=np.int64)
        for invalid in (good[:-1], good.astype(np.float32), good + (1 << 23), good - (1 << 23) - 1):
            with self.assertRaises(ValueError):
                route_window(self.events, 0, invalid, self.refiner, 0.51, 0.28)
        damaged = good.copy()
        damaged[0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "logits_q"):
            assert_pipeline_equal({"logits_q": damaged}, {"logits_q": good})


if __name__ == "__main__":
    unittest.main()
