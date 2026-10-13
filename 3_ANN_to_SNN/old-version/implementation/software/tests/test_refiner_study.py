"""Study controls, data boundaries, selection gates and inference equivalence."""

from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.h2.numpy_model import _refiner_probability, load_export, quantized_mlp
from wangdao.h2.representation import build_refiner_features, candidate_indices
from wangdao.h2.torch_model import QuantizedMLP
from wangdao.metrics import event_metrics
from wangdao.refiner_study import (
    SequenceCache, balanced_indices, complete_evaluation, decide, evaluate_probabilities,
    execute, final_decision, flatten, frozen_inputs, full_probabilities, infer, load_cache,
    make_model, profile_cost, prune_refiner, replication_arms, save_cache,
    scan_thresholds, study_paths, train_epoch, weight_path,
)
from wangdao.s0r import build_s0r_candidates
from wangdao.snn import RefinerSNN


def events(count: int = 40) -> np.ndarray:
    value = np.zeros(count, dtype=[("x", "i2"), ("y", "i2"), ("t", "f8"),
                                   ("p", "u1"), ("label", "u1"), ("name", "i2")])
    value["t"] = np.linspace(0, 150, count)
    value["x"] = np.arange(count) % 32
    value["y"] = 3
    value["label"] = np.arange(count) % 2
    value["name"] = value["label"]
    return value


def record() -> SequenceCache:
    e = events()
    indices = np.arange(0, len(e), 3, dtype=np.int64)
    value = SequenceCache("synthetic.npz", e, indices,
                          np.random.default_rng(123).integers(0, 256, (len(indices), 32), dtype=np.uint8),
                          e["label"][indices].astype(np.float32), np.full(len(indices), 0.4, dtype=np.float32))
    value.validate()
    return value


def metric_record(iou: float, pd: float = 0.8, fa: float = 1e-5, cost: float = 10) -> dict:
    values = {k: cost for k in ("nonzero_multiplication_proxy", "nonzero_accumulation_proxy",
                                "membrane_updates", "parameter_bytes_in_backend", "median_inference_s")}
    return {"selected": {"scan": {"best_event": {"iou": iou}}, "localization": {"pd": pd, "fa": fa}},
            "cost": {"uncached": values, "cached": values}}


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.threads)

    def test_roles_exclude_validation_test_and_calibration(self) -> None:
        for role in ("test", "validation", "hw_calibration", "anything"):
            with self.assertRaises(ValueError):
                study_paths(role)

    def test_source_escape_is_rejected(self) -> None:
        with patch("wangdao.refiner_study.role_file_paths", return_value=[ROOT / "README.md"]):
            with self.assertRaises(ValueError):
                study_paths("train_main")

    def test_frozen_weight_hashes(self) -> None:
        self.assertEqual(set(frozen_inputs()), {"index", "h2", "proposal", "refiner"})

    def test_cache_roundtrip_identity_and_immutability(self) -> None:
        r = record()
        identity = {"file": r.name, "source": "snn", "role": "train_main"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.npz"
            save_cache(path, identity, r)
            loaded = load_cache(path, identity)
            np.testing.assert_array_equal(loaded.features, r.features)
            self.assertFalse(loaded.features.flags.writeable)
            with self.assertRaises(ValueError):
                load_cache(path, {**identity, "source": "h2"})

    def test_cache_rejects_misalignment_duplicate_and_invalid_teacher(self) -> None:
        for mutation in ("labels", "indices", "teacher"):
            r = record()
            if mutation == "labels":
                r.labels = 1 - r.labels
            elif mutation == "indices":
                r.candidates = np.zeros_like(r.candidates)
            else:
                r.teacher = np.full(len(r.labels), np.nan)
            with self.assertRaises(ValueError):
                r.validate()

    def test_sampling_equal_counts_shared_indices_and_seed_changes(self) -> None:
        labels = np.arange(1000) % 11 == 0
        indices = balanced_indices(labels, 1200, 42, 1)
        self.assertEqual(len(indices), 1200)
        self.assertEqual(int(labels[indices].sum()), 120)
        np.testing.assert_array_equal(indices, balanced_indices(labels, 1200, 42, 1))
        self.assertFalse(np.array_equal(indices, balanced_indices(labels, 1200, 43, 1)))
        self.assertFalse(indices.flags.writeable)
        with self.assertRaises(ValueError):
            balanced_indices(np.zeros(100), 100, 42, 1)

    def test_pruning_sizes_ties_and_submatrix(self) -> None:
        teacher = QuantizedMLP((32, 64, 32, 4), quantized=True)
        first, second = teacher.layers[0], teacher.layers[1]
        assert isinstance(first, torch.nn.Linear) and isinstance(second, torch.nn.Linear)
        with torch.no_grad():
            for layer in teacher.layers:
                assert isinstance(layer, torch.nn.Linear)
                layer.weight.fill_(1)
        model, channels = prune_refiner(teacher)
        self.assertEqual(channels, {"hidden1": list(range(32)), "hidden2": list(range(16))})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1600)
        with torch.no_grad():
            first.weight[63].fill_(10)
            second.weight[31].fill_(10)
        model, channels = prune_refiner(teacher)
        self.assertIn(63, channels["hidden1"])
        self.assertIn(31, channels["hidden2"])
        torch.testing.assert_close(model.layers[1].weight, second.weight[channels["hidden2"]][:, channels["hidden1"]])

    def test_ab_share_exact_initial_weights(self) -> None:
        a, _ = make_model("A")
        b, _ = make_model("B")
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]))

    def test_static_current_raw_logits_and_spike_rates_are_exact(self) -> None:
        model, _ = make_model("B")
        model.eval()
        features = torch.from_numpy(record().features.copy()).unsqueeze(0).expand(4, -1, -1)
        with torch.no_grad():
            original = model(features)
            cached = model(features, cache_static_input=True)
        self.assertTrue(torch.equal(original.raw_logits, cached.raw_logits))
        self.assertTrue(torch.equal(original.spike_rates, cached.spike_rates))

    def test_static_cache_rejects_training_grad_and_temporal_input(self) -> None:
        model = RefinerSNN()
        x = torch.zeros((4, 2, 32), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            model(x, cache_static_input=True)
        model.eval()
        with self.assertRaises(ValueError):
            model(x, cache_static_input=True)
        x[2, 0, 0] = 1
        with torch.no_grad(), self.assertRaises(ValueError):
            model(x, cache_static_input=True)

    def test_inference_batch_sizes_and_state_isolation(self) -> None:
        model, _ = make_model("B")
        x = record().features
        reference = infer(model, x, torch.device("cpu"), batch_size=7)
        infer(model, np.full_like(x, 255), torch.device("cpu"))
        actual = infer(model, x, torch.device("cpu"), True, batch_size=3)
        np.testing.assert_allclose(reference, actual, rtol=0, atol=1e-6)
        self.assertEqual(infer(model, x[:0], torch.device("cpu")).shape, (0,))

    def test_non_candidates_are_false_negatives(self) -> None:
        r = record()
        p = np.ones(len(r.candidates), dtype=np.float32)
        full = full_probabilities([r], p)
        metrics = evaluate_probabilities([r], p, 0.64, True)
        self.assertEqual(len(full), len(r.events))
        self.assertEqual(metrics["scan"]["best_event"]["false_negative"], int(r.events["label"].sum() - r.labels.sum()))
        with self.assertRaises(ValueError):
            full_probabilities([r], p[:-1])

    def test_empty_candidates_keep_all_events(self) -> None:
        e = events()
        r = SequenceCache("empty", e, np.empty(0, dtype=np.int64), np.empty((0, 32), dtype=np.uint8),
                          np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
        r.validate()
        result = evaluate_probabilities([r], np.empty(0, dtype=np.float32), 0.28, True)
        self.assertEqual(result["scan"]["best_event"]["recall"], 0)
        self.assertEqual(result["localization"]["fa"], 0)

    def test_threshold_scan_matches_original_at_float32_boundaries(self) -> None:
        thresholds = np.arange(0.01, 0.995, 0.01)
        p = np.concatenate([thresholds.astype(np.float32), np.nextafter(thresholds.astype(np.float32), np.float32(0)),
                            np.random.default_rng(2).random(300).astype(np.float32)])
        labels = np.arange(len(p)) % 3 == 0
        expected = max((event_metrics(labels, p, float(t)) for t in thresholds), key=lambda r: (r["iou"], r["f1"]))
        self.assertEqual(scan_thresholds(labels, p)["best_event"], expected)

    def test_teacher_uses_actual_snn_context(self) -> None:
        h2 = load_export(weight_path("h2"))
        e = events()
        heatmaps = np.ones((80, 33, 44), dtype=np.float32) * 0.6
        candidates, features, labels, teacher = build_s0r_candidates(e, h2, heatmaps=heatmaps, proposal_threshold=0.51)
        expected_indices = candidate_indices(e, list(range(80)), heatmaps, 0.51)
        expected_features = build_refiner_features(e, expected_indices, list(range(80)), heatmaps, 0.51)
        np.testing.assert_array_equal(candidates, expected_indices)
        np.testing.assert_array_equal(features, expected_features)
        np.testing.assert_array_equal(labels, e["label"][candidates])
        np.testing.assert_array_equal(teacher, _refiner_probability(quantized_mlp(features, h2.refiner_layers), h2.prototypes))
        with self.assertRaises(ValueError):
            build_s0r_candidates(e, h2, heatmaps=heatmaps)

    def test_static_current_cost_and_compact_ann_cost(self) -> None:
        model, _ = make_model("B")
        features = record().features
        uncached = profile_cost(model, features, torch.device("cpu"))
        cached = profile_cost(model, features, torch.device("cpu"), True)
        self.assertEqual(uncached["scheduled_multiplications"], cached["scheduled_multiplications"] * 4)
        self.assertEqual(uncached["spike_accumulation_proxy"], cached["spike_accumulation_proxy"])
        self.assertEqual(uncached["membrane_updates"], 4 * 104 * len(features))
        ann, _ = make_model("D")
        self.assertEqual(profile_cost(ann, features, torch.device("cpu"))["scheduled_multiplications"], 1600 * len(features))

    def test_training_updates_student_only(self) -> None:
        r = record()
        data = flatten([r])
        teacher_before = data.teacher.copy()
        indices = balanced_indices(data.labels, 20, 42, 1)
        for arm in "ABCD":
            model, _ = make_model(arm)
            state = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            result = train_epoch(model, data, indices, optimizer, torch.device("cpu"))
            self.assertEqual(result["samples"], 20)
            self.assertTrue(any(not torch.equal(value, state[key]) for key, value in model.state_dict().items()))
        np.testing.assert_array_equal(data.teacher, teacher_before)

    def test_gates_require_accuracy_localization_and_replication(self) -> None:
        baselines = {"frozen_ann": metric_record(0.75)}
        arms = {"A": metric_record(0.744), "B": metric_record(0.752),
                "C": metric_record(0.753), "D": metric_record(0.751, cost=9)}
        decision = decide(arms, baselines)
        self.assertTrue(decision["adaptation_pass"])
        self.assertTrue(decision["full_snn_accuracy_pass"])
        self.assertTrue(decision["compact_ann_pass"])
        self.assertEqual(replication_arms(decision), ["B", "C", "D"])
        self.assertLessEqual(4 + 2 * len(replication_arms(decision)), 10)
        self.assertFalse(final_decision(decision, {}, baselines)["full_snn_ready_for_quantization_research"])
        replicas = {str(s): {k: arms[k] for k in "BCD"} for s in (43, 44)}
        self.assertTrue(final_decision(decision, replicas, baselines)["full_snn_ready_for_quantization_research"])
        bad = copy.deepcopy(arms)
        bad["B"]["selected"]["localization"]["fa"] = 1.11e-5
        self.assertFalse(decide(bad, baselines)["full_snn_accuracy_pass"])

    def test_dominated_snn_stops_replication(self) -> None:
        baselines = {"frozen_ann": metric_record(0.75)}
        arms = {"A": metric_record(0.744), "B": metric_record(0.752),
                "C": metric_record(0.752), "D": metric_record(0.753, cost=5)}
        decision = decide(arms, baselines)
        self.assertTrue(decision["full_snn_dominated_by_compact_ann"])
        self.assertEqual(replication_arms(decision), ["C", "D"])

    def test_formal_scheduler_caps_training_at_ten_and_reuses_samples(self) -> None:
        base = metric_record(0.75)
        trained = {"A": metric_record(0.744), "B": metric_record(0.752),
                   "C": metric_record(0.753), "D": metric_record(0.751, cost=9)}
        calls = []
        samples = {}
        def fake_train(arm, seed, epochs, data, count, dev, features, directory, device, shared, hashes, deadline):
            self.assertEqual(epochs, 20)
            if arm in "BCD":
                if seed in samples:
                    self.assertIs(samples[seed], shared)
                samples[seed] = shared
            calls.append((seed, arm))
            return copy.deepcopy(trained[arm])
        with tempfile.TemporaryDirectory(dir=ROOT / "results" / "runs") as temp:
            args = argparse.Namespace(run_directory=Path(temp) / "study", output=None,
                                      cache_directory=None, experiment_id="TEST-MOCK-SCHEDULER",
                                      mode="formal", device="cpu", memory_mb=1024, wall_time_limit=0)
            with patch("wangdao.refiner_study.prepare_sequences", return_value=[record()]), \
                 patch("wangdao.refiner_study.complete_evaluation", return_value=base), \
                 patch("wangdao.refiner_study.train_arm", side_effect=fake_train):
                result = execute(args)
        self.assertEqual(len(calls), 10)
        self.assertEqual(calls, [(42, a) for a in "ABCD"] + [(s, a) for s in (43, 44) for a in "BCD"])
        self.assertTrue(result["decision"]["full_snn_ready_for_quantization_research"])
        self.assertFalse(result["test_used"])


if __name__ == "__main__":
    unittest.main()
