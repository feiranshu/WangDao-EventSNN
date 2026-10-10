"""Frozen H2-pro identity and closed-window publishing contract."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from wangdao.h2_pro.stream_state import CausalPipeline, EVENT_DTYPE  # noqa: E402


class H2ProRuntimeTests(unittest.TestCase):
    def test_frozen_weight_and_causal_window(self) -> None:
        proposal = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
        refiner = ROOT / "software" / "weights" / "H2-pro-Causal-ANN-v1_int8.npz"
        self.assertEqual(hashlib.sha256(refiner.read_bytes()).hexdigest(),
                         "6e9221829459825413c58604a40da3582137a555aaa508d84d9da8325aa5368d")
        pipeline = CausalPipeline(proposal, refiner, variant="C1-proposal-motion",
                                  threshold=0.35, initial_refiner=False)
        events = np.zeros(2, dtype=EVENT_DTYPE)
        events["t"] = [0.0, 110.0]
        events["x"] = [10, 11]
        events["y"] = [10, 11]
        pipeline.reset(0.0)
        pipeline.push(events[:1], 99.0)
        self.assertEqual(len(pipeline.flush_completed_windows()), 0)
        pipeline.push(events[1:], 110.000001)
        windows, trailing = pipeline.finalize()
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0].probabilities), 1)
        self.assertEqual(windows[0].right_boundary_ms, 100.0)
        self.assertEqual(trailing, 1)


if __name__ == "__main__":
    unittest.main()
