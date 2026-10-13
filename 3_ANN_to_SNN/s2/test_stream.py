import unittest
from pathlib import Path

import numpy as np
import torch

from core import ResearchANN, ResearchSNN
from stream import EVENT_DTYPE, ProtocolAPipeline


ROOT = Path(__file__).absolute().parent.parent.parent
SOURCE = ROOT / "1_H2_model_improvement" / "implementation" / "data" / "EV-UAV-dataset" / "train" / "train_002.npz"
CACHE = ROOT / "1_H2_model_improvement" / "seven_step_improvement_study" / "H2-pro" / "cache" / "train_dev" / "train_002_C1-proposal-motion.npz"


@unittest.skipUnless(SOURCE.exists() and CACHE.exists(), "local Train-dev source and cache are required")
class ProtocolATests(unittest.TestCase):
    def test_empty_windows_silence_and_reset(self):
        torch.manual_seed(42)
        pipe = ProtocolAPipeline(ResearchSNN(), "snn")
        events = np.zeros(2, dtype=EVENT_DTYPE)
        events["x"] = [0, 345]
        events["y"] = [0, 259]
        events["t"] = [0.0, 350.0]
        events["p"] = [0, 1]
        pipe.reset(0.0)
        pipe.push(events, 400.0)
        first = pipe.flush_completed_windows()
        self.assertEqual([len(row.event_indices) for row in first], [1, 0, 0, 1])
        self.assertEqual([float(row.available_time_ms[0]) for row in first if len(row.event_indices)],
                         [100.0, 400.0])
        pipe.reset(0.0)
        pipe.push(events, 400.0)
        replay = pipe.flush_completed_windows()
        for a, b in zip(first, replay):
            np.testing.assert_allclose(a.probability, b.probability, atol=1e-7)

    def test_publication_chunking_and_tail(self):
        with np.load(SOURCE, allow_pickle=False) as archive:
            all_events = archive["ev"]
            t0 = float(all_events["t"][0])
            first = all_events[all_events["t"] < t0 + 100]
            tail = all_events[(all_events["t"] >= t0 + 100) & (all_events["t"] < t0 + 150)]
        self.assertGreater(len(first), 0)
        for kind, cls in (("ann", ResearchANN), ("snn", ResearchSNN)):
            torch.manual_seed(42)
            model = cls().eval()
            full = ProtocolAPipeline(model, kind)
            full.reset(t0)
            full.push(first, t0 + 100)
            result = full.flush_completed_windows()
            self.assertEqual(len(result), 1)
            with np.load(CACHE, allow_pickle=False) as cache:
                expected = cache["indices"]
                expected = expected[expected < len(first)]
            np.testing.assert_array_equal(np.flatnonzero(result[0].candidate), expected)
            split = ProtocolAPipeline(model, kind)
            split.reset(t0)
            split.push(first[first["t"] < t0 + 50], t0 + 50)
            self.assertEqual(split.flush_completed_windows(), [])
            split.push(first[first["t"] >= t0 + 50], t0 + 100)
            other = split.flush_completed_windows()
            self.assertEqual(len(other), 1)
            np.testing.assert_array_equal(result[0].event_indices, other[0].event_indices)
            np.testing.assert_array_equal(result[0].candidate, other[0].candidate)
            np.testing.assert_allclose(result[0].probability, other[0].probability, atol=1e-6)
            np.testing.assert_array_equal(result[0].available_time_ms,
                                          np.full(len(first), t0 + 100))
            self.assertTrue(np.all(result[0].available_time_ms > result[0].event_time_ms))
            split.push(tail, t0 + 150)
            ready, leftover = split.finalize()
            self.assertEqual(ready, [])
            self.assertEqual(leftover, len(tail))


if __name__ == "__main__":
    unittest.main()
