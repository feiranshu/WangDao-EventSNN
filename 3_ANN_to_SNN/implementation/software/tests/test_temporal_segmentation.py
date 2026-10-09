"""Boundary, causality, supervision and protocol tests without official evaluation data."""
import itertools
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wangdao.temporal_segmentation import (ARMS, TemporalSegmenter, collision_counts,
                                           collision_summary, event_loss, represent, structural_cost)
from wangdao.temporal_study import (PROTOCOL, SequenceStore, batches, checkpoint_load, collate, decide, epoch_order,
                                    evaluate, gates, infer_record, read_json, train, update, write_json)


def events():
    e = np.zeros(10, dtype=[("x", "i4"), ("y", "i4"), ("t", "f8"),
                            ("p", "i4"), ("label", "i4"), ("name", "i4")])
    e["t"] = [10, 10, 34.999, 35, 35, 60, 110, 210, 210, 234]
    e["x"] = [345, 345, 0, 1, 1, 3, 2, 4, 4, 0]
    e["y"] = [259, 259, 0, 1, 1, 3, 2, 4, 4, 0]
    e["p"] = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
    e["label"] = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    e["name"] = e["label"]
    return e


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_boundaries_padding_counts_and_event_indices(self):
        r = represent(events())
        np.testing.assert_array_equal(r.bins, [0, 0, 0, 1, 1, 2, 4, 8, 8, 8])
        self.assertEqual(r.blocks.tolist(), [0, 1, 2])
        packed, index = r.block(0)
        dense = torch.nn.functional.pixel_shuffle(torch.from_numpy(packed), 4).numpy()
        np.testing.assert_array_equal(index, np.arange(6))
        self.assertEqual(float(dense[0, 0, 259, 345]), np.float32(round(32 * np.log1p(2)) / 255))
        self.assertAlmostEqual(float(dense[0, 3, 0, 0]), 24.999 / 25, places=6)
        self.assertTrue(np.all(dense[..., 346:] == 0))
        self.assertTrue(np.all(dense[3] == 0))
        self.assertGreater(dense[1, 0, 1, 1], 0)
        self.assertGreater(dense[1, 1, 1, 1], 0)

    def test_feature_labels_identity_and_future_independence(self):
        e = events()
        before, _ = represent(e).block(0)
        e["label"] = 1 - e["label"]
        e["name"] = 7 * e["label"]
        e["x"][6:] = 30
        np.testing.assert_array_equal(before, represent(e).block(0)[0])
        prefix = represent(e[:6]).block(0)[0]
        np.testing.assert_array_equal(prefix, before)

    def test_oracle_matches_exhaustive_subsets_not_majority(self):
        p, n = np.array([3, 1, 0, 2]), np.array([0, 2, 7, 1])
        best = max(p[list(mask)].sum() / (p.sum() + n[list(mask)].sum())
                   for mask in itertools.product([False, True], repeat=4))
        self.assertAlmostEqual(collision_summary(p, n)["oracle_event_iou"], best)
        p, n = collision_counts(represent(events()))
        self.assertEqual(collision_summary(p, n)["mixed_groups"], 2)

    def test_pair_initialization_parameters_and_prior_pixel_order(self):
        for spatial in ("Local", "Dilated"):
            a = TemporalSegmenter(spatial + "-ANN", 42, (0.1, 0.2))
            b = TemporalSegmenter(spatial + "-SNN", 42, (0.1, 0.2))
            self.assertEqual(sum(p.numel() for p in a.parameters()), 1952)
            for left, right in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            output = a.step(torch.zeros(1, 64, 65, 87)).logits.sigmoid()
            torch.testing.assert_close(output[:, 0], torch.full_like(output[:, 0], 0.1))
            torch.testing.assert_close(output[:, 1], torch.full_like(output[:, 1], 0.2))
        self.assertEqual(structural_cost(True)["dense_products_per_100ms"], 43430400)

    def test_step_block_batch_state_isolation_and_empty_advancement(self):
        x = torch.from_numpy(represent(events()).block(0)[0]).unsqueeze(1)
        for arm in ARMS:
            with self.subTest(arm=arm), torch.no_grad():
                model = TemporalSegmenter(arm)
                block = model(x)
                state, values = None, []
                for value in x:
                    out = model.step(value, state)
                    state = out.state
                    values.append(out.logits)
                torch.testing.assert_close(block.logits, torch.stack(values), rtol=0, atol=0)
                other = model(torch.ones_like(x))
                torch.testing.assert_close(model(x).logits, block.logits, rtol=0, atol=0)
                batched = model(x.expand(-1, 2, -1, -1, -1))
                torch.testing.assert_close(batched.logits[:, :1], block.logits, rtol=1e-5, atol=1e-6)
                zero = torch.zeros(1, 64, 65, 87)
                state = tuple(torch.full((1, 8, 65, 87), 0.1) for _ in range(3))
                advanced = model.step(zero, state)
                torch.testing.assert_close(advanced.state[0], torch.full_like(state[0], 0.0875))
                self.assertTrue(torch.isfinite(other.logits).all())
                with self.assertRaises(ValueError):
                    model.step(zero.expand(2, -1, -1, -1), state)

    def test_predictions_restore_duplicates_last_partial_and_stream_causality(self):
        e = events()
        for arm in ("Local-ANN", "Local-SNN"):
            model = TemporalSegmenter(arm)
            p = infer_record(model, represent(e), torch.device("cpu"), 1)
            q = infer_record(model, represent(e), torch.device("cpu"), 3)
            np.testing.assert_allclose(p, q, rtol=1e-5, atol=1e-6)
            self.assertEqual(p[0], p[1])
            self.assertEqual(p[7], p[8])
            np.testing.assert_array_equal(p[:6], infer_record(model, represent(e[:6]), torch.device("cpu"), 1))

    def test_mixed_labels_individual_loss_gradient_and_metrics(self):
        value = torch.tensor([0.2], requires_grad=True)
        loss = event_loss(value.expand(3), torch.tensor([1., 0., 0.]), torch.zeros(3), False)
        loss.backward()
        assert value.grad is not None
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertNotEqual(float(loss), float(event_loss(value, torch.tensor([0.]), torch.zeros(3), False)))
        e = events()
        metric = evaluate([e], [e["label"].astype(np.float32)], 0.5)
        self.assertEqual(metric["event"]["iou"], 1)
        self.assertEqual(metric["localization"]["fa"], 0)

    def test_invalid_events(self):
        e = events()
        e["t"][0] = np.nan
        with self.assertRaises(ValueError):
            represent(e)

    def test_cache_identity_corruption_roles_and_prefetch_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "train_fake.npz"
            np.savez(path, ev=events())
            with patch("wangdao.temporal_study.study_paths", return_value=[path]):
                store = SequenceStore(root / "cache", 0)
                first = store.get("train_main", path.name)
                second = store.get("train_main", path.name)
                np.testing.assert_array_equal(first.keys, second.keys)
                for role in ("validation", "test", "hw_calibration"):
                    with self.assertRaises(ValueError):
                        store.get(role, path.name)
                order = [(path.name, 0), (path.name, 1), (path.name, 2)] * 3
                self.assertEqual(epoch_order(order, 42, 1), epoch_order(order, 42, 1))
                for a, b in zip(batches(store, order, True), batches(store, order, False)):
                    for x, y in zip(a, b):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
                batch = collate(store, "train_main", [(path.name, 0)])
                self.assertEqual(len(batch[2]), 6)
                cache = root / "cache" / ("train_main-" + path.name)
                with np.load(cache) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                arrays["counts"][0] += 1
                np.savez(cache, **arrays)
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    store.get("train_main", path.name)

    def test_gates_replication_stop_and_dominance(self):
        def record(iou, seconds, fa=0.0):
            return {"evaluation": {"event": {"iou": iou}, "localization": {"pd": .9, "fa": fa}}, "cpu": {"median_s": seconds}}
        reference = record(.8, 10)
        self.assertFalse(gates(record(.8, 5, 1e-10), reference)["cost_pass"])
        failed = {"42": {a: record(.7, 10) for a in ARMS}}
        self.assertEqual(decide(failed, reference)["next"], "retain_mainline_stop_search")
        runs = {str(s): {a: record(.8, 7 if a.endswith("ANN") else 8) for a in ARMS} for s in (42, 43, 44)}
        decision = decide(runs, reference)
        self.assertEqual(decision["fixed_point_research_candidates"], ["Local-ANN", "Dilated-ANN"])
        runs["44"]["Local-ANN"] = record(.7, 7)
        self.assertNotIn("Local-ANN", decide(runs, reference)["fixed_point_research_candidates"])

    def test_interrupted_epoch_resume_matches_uninterrupted_and_ties_keep_first(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            path = root / "train_fake.npz"
            np.savez(path, ev=events())
            evaluation = {"event": {"iou": .5, "f1": 2 / 3}, "threshold": .2, "seconds": 0.0}
            hashes = {"synthetic_test": "not_a_formal_run"}
            with patch("wangdao.temporal_study.study_paths", return_value=[path]), \
                 patch("wangdao.temporal_study.frozen_inputs", return_value=hashes), \
                 patch("wangdao.temporal_study.training_inventory", return_value=([(path.name, 0)], (.1, .1), {})), \
                 patch("wangdao.temporal_study.evaluate_model", return_value=evaluation):
                store = SequenceStore(root / "cache", 1)
                for name in ("resumed", "continuous"):
                    run = root / name
                    write_json(run / "preflight.json", {"protocol": PROTOCOL, "frozen_inputs": hashes,
                                                       "decision": {"formal_training_allowed": True}})
                args = SimpleNamespace(run_directory=root / "resumed", seed=42, arms=["Local-ANN"])
                count = [0]
                def interrupt(*values):
                    count[0] += 1
                    if count[0] == 3:
                        raise RuntimeError("simulated process interruption")
                    return update(*values)
                with patch("wangdao.temporal_study.update", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        train(args, store, torch.device("cpu"))
                train(args, store, torch.device("cpu"))
                args.run_directory = root / "continuous"
                train(args, store, torch.device("cpu"))
                a = checkpoint_load(root / "resumed/Local-ANN-seed42/last.pt", torch.device("cpu"))
                b = checkpoint_load(root / "continuous/Local-ANN-seed42/last.pt", torch.device("cpu"))
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
                result = read_json(root / "resumed/Local-ANN-seed42/training.json")
                self.assertEqual(len(result["history"]), 30)
                self.assertEqual(result["best"]["epoch"], 1)
                self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
