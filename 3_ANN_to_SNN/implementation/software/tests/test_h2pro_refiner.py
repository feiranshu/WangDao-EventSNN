"""Stream semantics, event coverage and gradient checks; no optimizer updates."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/"software"))
from wangdao.h2pro_snn.foundation import role_records, sha256, verify_baseline
from wangdao.h2pro_snn.refiner import (RefinerConfig, StreamRefiner, activity_routes,
                                       transport_grid)
from wangdao.h2pro_snn.stream_data import (CACHE_VERSION, CachedSequence, SequenceStore,
    atomic_json, encode_window, prefetch, run_path, stream_batches, validate_sequence)
from wangdao.h2pro_snn.training import (checkpoint_load, checkpoint_save, cost_report,
    cpu_weights, evaluate, forward_batch, load_profile, restore_rng, rng_state,
    selection_key, task_loss, train, train_epoch)


def sequence(times=(0., 9.99, 10., 99.9, 100., 199.99), windows=2):
    n = len(times)
    arrays = {"x": np.arange(n, dtype=np.int16) % 4,
              "y": np.zeros(n, np.int16), "p": np.arange(n, dtype=np.uint8) % 2,
              "t": np.array(times), "label": np.arange(n, dtype=np.uint8) % 2,
              "candidate": np.ones(n, bool), "teacher": np.full(n, .5, np.float32)}
    meta = {"t0_ms": 0., "closed_windows": windows, "published_events": n, "trailing_events": 0,
            "input_events": n, "original_events": n, "beyond_prefix_events": 0, "high_density": False}
    return CachedSequence(arrays, meta)


class FakeStore:
    def __init__(self, records):
        self.records = records
        self.stats = {}
        rows = [{"sequence": n, "published_events": len(r.arrays["t"]),
                 "closed_windows": r.metadata["closed_windows"]} for n, r in records.items()]
        self.manifest = {"roles": {"train_main": rows, "train_dev": rows}, "complete_development_roles": False}

    def names(self, role):
        return list(self.records)

    def get(self, role, name):
        return self.records[name]


class RefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, mode="fixed"):
        torch.manual_seed(42)
        return StreamRefiner(RefinerConfig(rows=2, columns=3, stem_channels=4,
                                          hidden_channels=6, local_channels=3, mode=mode))

    def test_three_modes_gradients_and_chunk_state_without_updates(self):
        x = torch.rand(4, 2, 64, 2, 3)
        for mode in ("fixed", "spike_route", "membrane_route"):
            with self.subTest(mode=mode):
                model = self.model(mode)
                before = cpu_weights(model)
                full, state, _, _ = model(x)
                first, split, _, _ = model(x[:2])
                rest, split, _, _ = model(x[2:], split)
                torch.testing.assert_close(full, torch.cat((first, rest)), rtol=0, atol=0)
                for a, b in zip(state, split):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                full.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(all(torch.equal(before[k], v) for k, v in cpu_weights(model).items()))

    def test_reset_and_batch_slots_are_independent(self):
        model = self.model()
        x = torch.rand(3, 2, 64, 2, 3)
        _, state, _, _ = model(x)
        reset = model.reset_slots(state, torch.tensor([True, False]))
        actual, _, _, _ = model(x[:1], reset)
        fresh, _, _, _ = model(x[:1, :1])
        torch.testing.assert_close(actual[:, :1], fresh, rtol=1e-6, atol=1e-6)
        for original, changed in zip(state, reset):
            self.assertEqual(changed[0].abs().sum().item(), 0)
            self.assertTrue(torch.equal(original[1], changed[1]))

    def test_future_steps_do_not_change_previous_logits(self):
        for mode in ("fixed", "spike_route", "membrane_route"):
            model = self.model(mode)
            inputs = torch.rand(4, 1, 64, 2, 3)
            modified = inputs.clone()
            modified[2:] = 8
            torch.testing.assert_close(model(inputs)[0][:2], model(modified)[0][:2], rtol=0, atol=0)

    def test_router_ties_empty_sources_and_boundaries_stay(self):
        previous = torch.ones(1, 1, 2, 3)
        current = torch.ones_like(previous)
        self.assertTrue((activity_routes(previous, current) == 4).all())
        current.zero_()
        current[..., 0, 1] = 2
        routes = activity_routes(previous, current)
        self.assertEqual(routes[0, 0, 0, 0].item(), 5)
        self.assertTrue((activity_routes(previous*0, current) == 4).all())
        moved = transport_grid(previous, routes)
        self.assertEqual(moved.sum().item(), previous.sum().item())

    def test_transport_consumes_each_source_once_and_preserves_signed_mass(self):
        values = torch.tensor([[[[.4, -.2, .8]]]], requires_grad=True)
        routes = torch.tensor([[[[5, 3, 3]]]])
        moved = transport_grid(values, routes)
        torch.testing.assert_close(moved, torch.tensor([[[[-.2, 1.2, 0.]]]]))
        moved.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))

    def test_active_cell_stem_matches_dense_projection(self):
        model = self.model()
        x = torch.zeros(1, 1, 64, 2, 3)
        x[0, 0, 3, 0, 1] = .7
        _, state, _, counts = model(x)
        spikes, reference = model._lif(model.stem(x[0]))
        torch.testing.assert_close(state[0], reference)
        self.assertEqual(counts[0, 0, 0].item(), 1)
        self.assertEqual(counts[0, 0, 1].item(), 1)

    def test_default_architecture_has_phase_specific_readout_and_bounded_parameters(self):
        a = StreamRefiner().architecture()
        self.assertEqual(a["parameters"], 9856)
        self.assertEqual(a["state_scalars_per_stream"], 68*88*177)

    def test_cost_does_not_claim_dense_cuda_is_ac_only(self):
        model = self.model()
        c = cost_report(model, np.array([2, 7, 3, 4, 1.]), 10)
        self.assertFalse(c["end_to_end_complete"])
        self.assertFalse(c["spike_convolution_zero_skip_implemented"])
        self.assertEqual(c["active_cell_stem_mac_estimate"], 2*64*4)
        self.assertEqual(c["ideal_unit_spike_ac_upper_bound"], 3*9*6+4*128)

    def test_invalid_config_and_shape(self):
        for kwargs in ({"mode": "oracle"}, {"threshold": 0}, {"tau_long_ms": float("nan")}, {"micro_ms": 1}):
            with self.assertRaises(ValueError):
                RefinerConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.model()(torch.zeros(1, 1, 63, 2, 3))


class StreamDataTests(unittest.TestCase):
    def test_direct_strided_batch_encoding_matches_contiguous_result(self):
        record = sequence()
        expected = encode_window(record, 0)[0]
        packed = np.zeros((10, 2, 64, 68, 88), np.float32)
        actual = encode_window(record, 0, out=packed[:, 1])[0]
        self.assertFalse(actual.flags.c_contiguous)
        self.assertTrue(np.shares_memory(actual, packed))
        np.testing.assert_array_equal(expected, actual)
        self.assertFalse(packed[:, 0].any())

    def test_cache_roles_hash_corruption_and_bounded_residency(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"results/runs") as directory:
            directory = Path(directory)
            baseline = verify_baseline()
            manifest = {"version": CACHE_VERSION, "status": "complete", "sequence_limit": 1,
                        "max_windows": 2, "complete_development_roles": False,
                        "baseline_lock_sha256": baseline["baseline_lock_sha256"], "roles": {}}
            for role in ("train_main", "train_dev"):
                source = role_records(role, 1)[0]
                r = sequence()
                r.metadata.update(sequence=source["path"], role=role, source_sha256=source["sha256"],
                                  version=CACHE_VERSION, max_windows=2,
                                  baseline_lock_sha256=baseline["baseline_lock_sha256"])
                path = directory/(role+".npz")
                np.savez(path, **r.arrays, metadata=np.array(json.dumps(r.metadata)))
                manifest["roles"][role] = [{"sequence": source["path"], "source_sha256": source["sha256"],
                    "cache_file": path.name, "cache_sha256": sha256(path), "published_events": 6,
                    "closed_windows": 2, "candidate_events": 6, "candidate_targets": 3}]
            atomic_json(directory/"manifest.json", manifest)
            store = SequenceStore(directory, cache_mib=0)
            name = store.names("train_main")[0]
            store.get("train_main", name)
            self.assertEqual(store.stats["resident_bytes"], 0)
            with self.assertRaises(ValueError):
                store.get("train_dev", name)
            for role in ("validation", "test", "hw_calibration"):
                with self.assertRaises(ValueError):
                    store.names(role)
            with (directory/"train_main.npz").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                SequenceStore(directory)

    def test_encoding_has_no_label_teacher_or_candidate_dependency(self):
        record = sequence()
        baseline = encode_window(record, 0)
        altered = copy.deepcopy(record)
        altered.arrays["label"] ^= 1
        altered.arrays["candidate"][:] = False
        altered.arrays["teacher"][:] = 0
        np.testing.assert_array_equal(baseline[0], encode_window(altered, 0)[0])
        self.assertEqual(baseline[1][0].tolist(), [0, 0, 1, 9])
        self.assertEqual(baseline[2].tolist(), [0, 1, 2, 3])
        self.assertEqual(encode_window(record, 1)[2].tolist(), [4, 5])

    def test_all_pixel_phases_and_polarities_are_distinct(self):
        record = sequence(tuple(np.arange(32, dtype=float)), 1)
        record.arrays["x"] = np.tile(np.arange(4, dtype=np.int16), 8)
        record.arrays["y"] = np.tile(np.repeat(np.arange(4, dtype=np.int16), 4), 2)
        record.arrays["p"] = np.repeat(np.arange(2, dtype=np.uint8), 16)
        _, mapping, _ = encode_window(record, 0)
        np.testing.assert_array_equal(mapping[1], np.arange(32))

    def test_duplicate_events_are_scored_individually_and_future_bin_isolation(self):
        record = sequence((1., 1., 12.), 1)
        record.arrays["x"][:] = 0
        record.arrays["p"][:] = 0
        encoded, _, ids = encode_window(record, 0)
        self.assertEqual(len(ids), 3)
        self.assertAlmostEqual(encoded[0, 0, 0, 0], 1-np.exp(-2), places=6)
        record.arrays["x"][2] = 20
        np.testing.assert_array_equal(encoded[0], encode_window(record, 0)[0][0])

    def test_empty_windows_and_last_slot_keep_chronological_coverage(self):
        store = FakeStore({"a": sequence((1., 201.), 3), "b": sequence((1.,), 1), "c": sequence((1., 100.), 2)})
        rows = list(stream_batches(store, "train_dev", 2))
        observed = {name: [] for name in store.records}
        windows = {name: [] for name in store.records}
        for batch in rows:
            for ref in batch["refs"]:
                observed[ref["sequence"]].extend(ref["ids"].tolist())
                windows[ref["sequence"]].append(ref["window"])
                if ref["window"] == 0:
                    self.assertTrue(batch["reset"][ref["slot"]])
        for name, record in store.records.items():
            self.assertEqual(observed[name], list(range(len(record.arrays["t"]))))
            self.assertEqual(windows[name], list(range(record.metadata["closed_windows"])))

    def test_prefetch_preserves_exact_order_and_values(self):
        expected = [np.arange(i+1) for i in range(5)]
        actual = [value for value, _ in prefetch(iter(expected))]
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_cache_numeric_and_prefix_validation(self):
        record = sequence()
        validate_sequence(record.arrays, record.metadata)
        for key, value in (("t", float("nan")), ("label", 2), ("x", 346), ("teacher", 2)):
            bad = copy.deepcopy(record)
            bad.arrays[key][0] = value
            with self.assertRaises(ValueError):
                validate_sequence(bad.arrays, bad.metadata)
        bad = copy.deepcopy(record)
        bad.metadata["trailing_events"] = 1
        with self.assertRaises(ValueError):
            validate_sequence(bad.arrays, bad.metadata)

    def test_output_paths_cannot_escape_runs(self):
        for p in (ROOT/"results", ROOT/"results/runs/../wrong", ROOT/"software/test"):
            with self.assertRaises(ValueError):
                run_path(p)


class TrainingContractTests(unittest.TestCase):
    def test_epoch_resume_identity_and_best_checkpoint_recovery_with_mocked_training(self):
        # The lifecycle is exercised, but neither fitting nor optimizer.step runs.
        with tempfile.TemporaryDirectory(dir=ROOT/"results/runs") as directory:
            directory = Path(directory)
            store = FakeStore({"a": sequence()})
            store.directory = directory/"cache"
            store.directory.mkdir()
            atomic_json(store.directory/"manifest.json", {"unit_test": True})
            for row in store.manifest["roles"]["train_main"]:
                row.update(candidate_events=6, candidate_targets=3)
            profile = load_profile(batch_size=1)
            profile["model"].update(rows=2, columns=3)
            selected = {"threshold": .35, "overall": {"iou": .5}, "high_density": {"recall": .6}}
            evaluation = {"selected": selected, "seconds": 0.}
            calls = []
            def no_training(model, store, optimizer, device, config, epoch):
                calls.append(epoch)
                optimizer._opt_called = True  # scheduler sequencing only; no update
                return {"epoch": epoch, "loss": 1., "train_s": 0.}
            with patch("wangdao.h2pro_snn.training.SequenceStore", return_value=store), \
                    patch("wangdao.h2pro_snn.training.train_epoch", side_effect=no_training), \
                    patch("wangdao.h2pro_snn.training.evaluate", return_value=evaluation), patch("builtins.print"):
                run = directory/"run"
                first = train(store.directory, run, profile, "cpu", 1, pilot=True)
                first_weights = checkpoint_load(run/"last.pt")["model"]
                # best.pt is a derived artifact; recovery can regenerate it.
                (run/"best.pt").unlink()
                second = train(store.directory, run, profile, "cpu", 2, resume=True, pilot=True)
                self.assertEqual(calls, [1, 2])
                self.assertEqual(second["best"]["epoch"], 1)
                self.assertTrue((run/"best.pt").exists())
                self.assertEqual(len(second["history"]), 2)
                for key, value in checkpoint_load(run/"last.pt")["model"].items():
                    torch.testing.assert_close(value, first_weights[key], rtol=0, atol=0)
                changed = copy.deepcopy(profile)
                changed["training"]["batch_size"] = 2
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    train(store.directory, run, changed, "cpu", 3, resume=True, pilot=True)
                self.assertFalse((run/".training.lock").exists())
                with self.assertRaisesRegex(ValueError, "Partial cache"):
                    train(store.directory, directory/"not_pilot", profile, "cpu", 1)

    def test_evaluation_keeps_proposal_rejections_and_unique_event_coverage(self):
        r = sequence((1., 2., 3.), 1)
        r.arrays["label"] = np.array([1, 1, 0])
        r.arrays["candidate"] = np.array([False, True, True])
        r.arrays["teacher"] = np.array([0., .9, .1], np.float32)
        store = FakeStore({"a": r})
        batch = next(stream_batches(store, "train_dev", 1))
        model = StreamRefiner(RefinerConfig(rows=2, columns=3))
        def fake_forward(*args):
            return torch.tensor([8., 8., -8.]), None, None, torch.zeros(10, 1, 5)
        with patch("wangdao.h2pro_snn.training.forward_batch", side_effect=fake_forward), \
                patch("wangdao.h2pro_snn.training.stream_batches", return_value=iter([batch])):
            result = evaluate(model, store, torch.device("cpu"), 1, [.35])
        metric = result["selected"]["overall"]
        self.assertEqual((metric["events"], metric["tp"], metric["fn"]), (3, 1, 1))
        self.assertEqual(metric["proposal_fn"], 1)
        self.assertEqual(metric["iou"], .5)
        self.assertEqual(result["delta_vs_teacher"]["iou"], 0)
        with patch("wangdao.h2pro_snn.training.forward_batch", side_effect=fake_forward), \
                patch("wangdao.h2pro_snn.training.stream_batches", return_value=iter([batch, batch])):
            with self.assertRaisesRegex(AssertionError, "Duplicate"):
                evaluate(model, store, torch.device("cpu"), 1, [.35])

    def test_batch_padding_is_in_actual_mac_but_not_logical_spike_proxy(self):
        model = StreamRefiner(RefinerConfig(rows=2, columns=3))
        logical = cost_report(model, np.array([1, 1, 1, 1, 0.]), 10)
        padded = cost_report(model, np.array([1, 1, 1, 1, 0.]), 10, 20)
        self.assertGreater(padded["dense_backend_synaptic_mac_estimate"], logical["dense_backend_synaptic_mac_estimate"])
        self.assertEqual(padded["ideal_unit_spike_ac_upper_bound"], logical["ideal_unit_spike_ac_upper_bound"])

    def test_profile_requires_explicit_numerical_contract(self):
        config = load_profile(mode="membrane_route", batch_size=1)
        self.assertEqual(config["model"]["mode"], "membrane_route")
        self.assertFalse(config["execution"]["automatic_training"])
        self.assertEqual(config["execution"]["training_launcher"], "user_only")
        with self.assertRaises(ValueError):
            load_profile(batch_size=0)

    def test_loss_gradients_are_finite_for_all_background_and_target_cases(self):
        config = load_profile()["training"]
        for values in ([0., 0.], [1., 1.], [0., 1.]):
            logits = torch.tensor([-.5, .2], requires_grad=True)
            rates = torch.tensor([.1, .4], requires_grad=True)
            loss, _ = task_loss(logits, torch.tensor(values), torch.tensor([.2, .7]), rates, config)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue(torch.isfinite(rates.grad).all())

    def test_epoch_flow_and_optimizer_call_contract_with_no_parameter_updates(self):
        model = StreamRefiner(RefinerConfig(rows=2, columns=3, stem_channels=4, hidden_channels=6, local_channels=3))
        before = cpu_weights(model)
        class NoUpdateOptimizer:
            calls = 0
            def zero_grad(self, **kwargs):
                model.zero_grad(**kwargs)
            def step(self):
                self.calls += 1  # intentionally never changes any parameter
        batch = {"inputs": np.ones((2, 1, 64, 2, 3), np.float32), "reset": np.array([True]),
                 "maps": np.array([[0, 1], [0, 0], [0, 1], [0, 1], [0, 2]]),
                 "labels": np.array([0, 1]), "candidate": np.array([True, True]),
                 "teacher": np.array([.1, .8], np.float32), "active": np.array([True])}
        store = FakeStore({"a": sequence((1., 2.), 1)})
        optimizer = NoUpdateOptimizer()
        with patch("wangdao.h2pro_snn.training.stream_batches", return_value=iter([batch])):
            result = train_epoch(model, store, optimizer, torch.device("cpu"), load_profile()["training"], 1)
        self.assertEqual(result["updates"], 1)
        self.assertEqual(optimizer.calls, 1)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in cpu_weights(model).items()))

    def test_checkpoint_roundtrip_and_rng_restore_without_optimizer_step(self):
        model = StreamRefiner(RefinerConfig(rows=2, columns=3))
        optimizer = torch.optim.AdamW(model.parameters())
        parameter = next(model.parameters())
        optimizer.state[parameter] = {"step": torch.tensor(7.), "exp_avg": torch.full_like(parameter, .02),
                                      "exp_avg_sq": torch.full_like(parameter, .03)}
        saved_rng = rng_state()
        with tempfile.TemporaryDirectory(dir=ROOT/"results/runs") as directory:
            path = Path(directory)/"last.pt"
            checkpoint_save(path, {"model": cpu_weights(model), "optimizer": optimizer.state_dict(),
                                   "rng": saved_rng, "history": [{"epoch": 0}]})
            loaded = checkpoint_load(path)
            self.assertEqual(loaded["history"], [{"epoch": 0}])
            restored = StreamRefiner(model.config)
            restored.load_state_dict(loaded["model"])
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            restored_optimizer.load_state_dict(loaded["optimizer"])
            restored_parameter = next(restored.parameters())
            for key, value in optimizer.state[parameter].items():
                torch.testing.assert_close(value, restored_optimizer.state[restored_parameter][key], rtol=0, atol=0)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            restore_rng(saved_rng)
            expected = torch.rand(8)
            restore_rng(loaded["rng"])
            torch.testing.assert_close(expected, torch.rand(8), atol=0, rtol=0)

    def test_help_has_separate_prepare_check_train_evaluate(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT/"software/scripts/train_h2pro_refiner.py"), "--help"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare,check,train,evaluate", result.stdout)


if __name__ == "__main__":
    unittest.main()
