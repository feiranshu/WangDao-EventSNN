"""Causality, ablation and temporal-training invariants on synthetic events."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao.causal import CausalConfig, CausalState, FeatureStream, CausalModel
from wangdao.causal_training import (predict_sequence, train_sequence, block_losses,
                                    save_checkpoint, load_checkpoint, select_evaluation)


class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def events(self):
        return np.array([[4, 8, 1, 1], [4, 8, 24, 0], [9, 8, 25, 1],
                         [9, 9, 25, 1], [10, 10, 60, 0], [11, 9, 220, 1]], dtype=float)

    def test_chunk_invariance_boundaries_and_empty_blocks(self):
        events = self.events()
        bulk = FeatureStream().push(events, 225, final=True)
        stream = FeatureStream()
        chunks = []
        for part, cutoff in ((events[:2], 25), (events[2:3], 25), (events[3:4], 50),
                             (events[4:5], 100), (events[5:], 225)):
            chunks += stream.push(part, cutoff)
        self.assertEqual(len(chunks), 9)
        for expected, actual in zip(bulk, chunks):
            np.testing.assert_array_equal(expected.spatial, actual.spatial)
            np.testing.assert_array_equal(expected.event_features, actual.event_features)
            np.testing.assert_array_equal(expected.event_indices, actual.event_indices)
        np.testing.assert_array_equal(np.concatenate([b.event_indices for b in chunks]), np.arange(6))
        self.assertEqual(len(chunks[3].events), 0)

    def test_future_append_does_not_change_published_scores(self):
        config = CausalConfig(variant="C2", proposal_threshold=0.)
        torch.manual_seed(4)
        model = CausalModel(config).eval()
        full = predict_sequence(model, self.events(), torch.device("cpu"), config)
        prefix = predict_sequence(model, self.events()[:2], torch.device("cpu"), config)
        np.testing.assert_array_equal(full["probabilities"][:2], prefix["probabilities"])

    def test_late_events_and_label_fields_rejected(self):
        stream = FeatureStream()
        stream.push(self.events()[:2], 25)
        with self.assertRaises(ValueError):
            stream.push(np.array([[1, 1, 24, 1.]]), 50)
        with self.assertRaises(ValueError):
            FeatureStream().push(np.ones((2, 6)), 25)
        with self.assertRaises(ValueError):
            FeatureStream().push(np.array([[1, 1, 26, 1.]]), 25)

    def test_count_horizons_expire_and_state_is_bounded(self):
        stream = FeatureStream()
        first = stream.push(np.array([[4, 8, 1, 1.]]), 25)[0]
        self.assertGreater(first.spatial[:6].sum(), 0)
        before = stream.state_bytes
        later = stream.push(np.empty((0, 4)), 200)
        self.assertEqual(float(later[-1].spatial.sum()), 0.)
        self.assertEqual(stream.state_bytes, before)
        stream.reset()
        self.assertEqual(stream.pending_events, 0)

    def test_all_variants_forward_backward_and_reset(self):
        for variant in ("C0", "C1", "C2", "C3", "C4"):
            config = CausalConfig(variant=variant, proposal_threshold=0.)
            model = CausalModel(config)
            block = FeatureStream(config).push(self.events()[:2], 25)[0]
            output = model.forward_block(block, force_deep=True)
            self.assertEqual(tuple(output["logits"].shape), (2, 4))
            self.assertEqual(tuple(output["proposal_logits"].shape), (34, 44))
            losses = block_losses(output, block, np.array([1, 0]), variant, torch.device("cpu"))
            losses["total"].backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            initial = output["probabilities"].detach().clone()
            again = model.forward_block(block, model.new_state(), force_deep=True)["probabilities"]
            torch.testing.assert_close(initial, again)
            detached = output["state"].detach()
            self.assertTrue(detached.short is None or detached.short.grad_fn is None)

    def test_rejected_events_still_receive_zero_probability(self):
        config = CausalConfig(variant="C0", proposal_threshold=1.)
        output = predict_sequence(CausalModel(config), self.events(), torch.device("cpu"), config)
        self.assertEqual(len(output["probabilities"]), len(self.events()))
        self.assertTrue(np.all(output["probabilities"] == 0))

    def test_force_deep_exposes_explicit_ungated_probabilities(self):
        config = CausalConfig(variant="C0", proposal_threshold=1.)
        block = FeatureStream(config).push(self.events()[:2], 25)[0]
        output = CausalModel(config).eval().forward_block(block, force_deep=True)
        self.assertTrue(torch.all(output["probabilities"] == 0))
        self.assertIsNotNone(output["ungated_probabilities"])
        self.assertEqual(tuple(output["ungated_probabilities"].shape), (2,))
        self.assertTrue(torch.all(output["ungated_probabilities"] > 0))
        replay = predict_sequence(CausalModel(config), self.events(), torch.device("cpu"), config,
                                  ungated_diagnostic=True)
        self.assertTrue(np.all(replay["probabilities"] == 0))
        self.assertTrue(np.all(replay["ungated_probabilities"] > 0))

    def test_no_memory_keeps_fusion_parameters_and_ignores_history_state(self):
        memory_config = CausalConfig(variant="C2", proposal_threshold=0.)
        null_config = CausalConfig(variant="C2", proposal_threshold=0., no_memory=True)
        torch.manual_seed(7)
        memory_model = CausalModel(memory_config)
        torch.manual_seed(7)
        null_model = CausalModel(null_config).eval()
        self.assertEqual(memory_model.state_dict().keys(), null_model.state_dict().keys())
        self.assertEqual(sum(p.numel() for p in memory_model.parameters()),
                         sum(p.numel() for p in null_model.parameters()))
        block = FeatureStream(null_config).push(self.events()[:2], 25)[0]
        empty = CausalState()
        populated = CausalState(short=torch.ones(1, 32, 34, 44),
                                long=torch.full((1, 32, 34, 44), 2.0))
        without_history = null_model.forward_block(block, empty)
        with_history = null_model.forward_block(block, populated)
        torch.testing.assert_close(without_history["logits"], with_history["logits"])
        self.assertIsNone(with_history["state"].short)
        self.assertIsNone(with_history["state"].long)

    def test_tbptt_updates_and_roundtrip(self):
        config = CausalConfig(variant="C2", proposal_threshold=0.)
        model = CausalModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = next(model.parameters()).detach().clone()
        result = train_sequence(model, self.events(), np.array([1, 0, 1, 1, 0, 1]),
                                optimizer, torch.device("cpu"), config)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
        expected = predict_sequence(model, self.events(), torch.device("cpu"), config)["probabilities"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, config, {"smoke": True})
            restored, restored_cfg, metadata = load_checkpoint(path, torch.device("cpu"))
            actual = predict_sequence(restored, self.events(), torch.device("cpu"), restored_cfg)["probabilities"]
            np.testing.assert_array_equal(expected, actual)
            self.assertTrue(metadata["smoke"])

    def test_partial_evaluation_cannot_be_selected(self):
        selected = select_evaluation({"partial": True}, None, "C0")
        self.assertFalse(selected["eligible"])


if __name__ == "__main__":
    unittest.main()
