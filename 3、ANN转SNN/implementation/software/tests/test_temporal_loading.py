"""Compare packed loading with the original dense pixel-unshuffle path."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao.temporal_segmentation import CELLS, HEIGHT, PAD_WIDTH, TemporalSegmenter, represent
from wangdao.temporal_study import SequenceStore, batches, collate, epoch_order, infer_record, update


def dense_reference_block(record, block):
    dense = np.zeros((4, 4, HEIGHT, PAD_WIDTH), dtype=np.float32)
    low, high = np.searchsorted(record.keys, [block * 8 * CELLS, (block + 1) * 8 * CELLS])
    micro, remainder = np.divmod(record.keys[low:high], 2 * CELLS)
    polarity, pixel = np.divmod(remainder, CELLS)
    yy, xx = np.divmod(pixel, PAD_WIDTH)
    dense[micro % 4, polarity, yy, xx] = record.counts[low:high].astype(np.float32) / 255.0
    dense[micro % 4, polarity + 2, yy, xx] = record.last[low:high]
    packed = dense.reshape(4, 4, 65, 4, 87, 4).transpose(0, 1, 3, 5, 2, 4).reshape(4, 64, 65, 87)
    begin, end = np.searchsorted(record.bins, [block * 4, (block + 1) * 4])
    return np.ascontiguousarray(packed), np.arange(begin, end, dtype=np.int64)


def dense_reference_collate(store, role, selected):
    features, indices, labels = [], [], []
    for batch, (name, block) in enumerate(selected):
        record = store.get(role, name)
        value, event_indices = dense_reference_block(record, block)
        events = record.events[event_indices]
        features.append(value)
        indices.append(np.column_stack((record.bins[event_indices] % 4, np.full(len(events), batch),
                                       events["p"], events["y"], events["x"])).astype(np.int64))
        labels.append(events["label"].astype(np.float32))
    return (torch.from_numpy(np.stack(features, axis=1)), torch.from_numpy(np.concatenate(indices)),
            torch.from_numpy(np.concatenate(labels)))


def dense_reference_pack(blocks):
    parts = [dense_reference_block(record, block) for record, block in blocks]
    return np.stack([p[0] for p in parts], axis=1), [p[1] for p in parts]


def synthetic_record():
    rng = np.random.default_rng(735)
    events = np.zeros(512, dtype=[("x", "i4"), ("y", "i4"), ("t", "f8"),
                                 ("p", "i4"), ("label", "i4"), ("name", "i4")])
    events["x"] = rng.integers(0, 346, len(events))
    events["y"] = rng.integers(0, 260, len(events))
    events["p"] = rng.integers(0, 2, len(events))
    events["label"] = rng.integers(0, 2, len(events))
    events["name"] = events["label"]
    events["t"] = np.arange(len(events)) * .4
    # Every subpixel phase and both polarities, then duplicates with mixed labels at the edge.
    events["x"][:32] = np.tile(np.arange(16) % 4, 2)
    events["y"][:32] = np.tile(np.arange(16) // 4, 2)
    events["p"][:32] = np.repeat([0, 1], 16)
    events["x"][32:36], events["y"][32:36] = 345, 259
    events["t"][32:36] = events["t"][32]
    events["p"][32:36], events["label"][32:36] = [0, 0, 1, 1], [0, 1, 0, 1]
    return represent(events)


class MemoryStore(SequenceStore):
    def __init__(self, record):
        self.record = record

    def get(self, role, name):
        assert role == "train_main" and name == "synthetic"
        return self.record


class TemporalLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scatter_matches_dense_with_strided_output_and_empty_blocks(self):
        record = synthetic_record()
        backing = np.full((4, 3, 64, 65, 87), np.nan, dtype=np.float32)
        for block in (0, 1, 2, 8):
            expected, mapping = dense_reference_block(record, block)
            actual, index = record.block(block, out=backing[:, 1])
            self.assertTrue(np.shares_memory(actual, backing))
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(index, mapping)
            self.assertTrue(np.isnan(backing[:, 0]).all())
            self.assertTrue(np.isnan(backing[:, 2]).all())
        for output in (np.empty((1,), dtype=np.float32), np.empty((4, 64, 65, 87), dtype=np.float64)):
            with self.assertRaises(ValueError):
                record.block(0, out=output)

    def test_batch_order_tail_prefetch_ownership_and_rng_are_unchanged(self):
        store = MemoryStore(synthetic_record())
        order = epoch_order([("synthetic", i % 3) for i in range(19)], 42, 3)
        rng = torch.get_rng_state().clone()
        saved = []
        for batch, start in zip(batches(store, order), range(0, len(order), 8)):
            expected = dense_reference_collate(store, "train_main", order[start:start + 8])
            for actual, reference in zip(batch, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertEqual(actual.stride(), reference.stride())
            saved.append((batch, tuple(t.clone() for t in batch)))
        self.assertEqual([b[0][0].shape[1] for b in saved], [8, 8, 3])
        for batch, snapshot in saved:
            for actual, reference in zip(batch, snapshot):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_logits_gradients_optimizer_updates_and_inference_are_identical(self):
        record = synthetic_record()
        store = MemoryStore(record)
        order = [("synthetic", 0), ("synthetic", 1)]
        old = dense_reference_collate(store, "train_main", order)
        new = collate(store, "train_main", order)
        for readout in ("instant", "membrane"):
            a = TemporalSegmenter("Local-SNN", readout=readout)
            b = copy.deepcopy(a)
            oa = torch.optim.AdamW(a.parameters(), lr=1e-3)
            ob = torch.optim.AdamW(b.parameters(), lr=1e-3)
            for _ in range(2):
                self.assertEqual(update(a, old, oa, torch.device("cpu")), update(b, new, ob, torch.device("cpu")))
                for pa, pb in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(oa.state[pa][key], ob.state[pb][key], rtol=0, atol=0)
            with patch("wangdao.temporal_study.pack_blocks", side_effect=dense_reference_pack):
                expected = infer_record(a, record, torch.device("cpu"), batch_size=2)
            np.testing.assert_array_equal(infer_record(b, record, torch.device("cpu"), batch_size=2), expected)


if __name__ == "__main__":
    unittest.main()
