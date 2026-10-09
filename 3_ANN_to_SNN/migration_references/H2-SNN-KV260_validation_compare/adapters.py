"""Inference-only adapters around unchanged, frozen project source snapshots."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline/software"))
sys.path.insert(0, str(ROOT / "baseline/software/scripts"))

from wangdao.fixed import FixedH2, FixedH2Config, load_fixed_proposal_snn
from wangdao.h2.numpy_model import load_export, run_pipeline, _refiner_probability
from wangdao.h2.representation import (build_patch_features, build_temporal_patch_features,
                                      build_refiner_features, candidate_indices)
from evaluate_s1 import fixed_mlp_batched, proposal_heatmaps
from prepare import sha256


class Models:
    def __init__(self):
        weights = ROOT / "baseline/software/weights"
        self.h2 = load_export(weights / "H2_D1_seed42.npz")
        self.s1, self.s1_config, self.s1_meta = load_fixed_proposal_snn(weights / "S1P_INT8_ACC24_MEM24_Q12.npz")
        if self.s1_meta["dataset_index_sha256"] != sha256(ROOT / "baseline/data/index.json"):
            raise ValueError("S1 index mismatch")
        self.a1 = FixedH2(self.h2, FixedH2Config(**self.s1_meta["h2_fixed_configuration"]))
        self.threshold = float(self.h2.event_threshold)
        if self.threshold != float(self.s1_meta["event_threshold"]):
            raise ValueError("Unexpected frozen threshold disagreement")

    def predict(self, events):
        # Do not expose labels or target identities to the inference path.
        inputs = events[["x", "y", "t", "p"]].copy()
        h2_result = run_pipeline(inputs, self.h2)
        yield "h2", h2_result["probabilities"], h2_result["candidate_indices"]
        windows = list(range(80))
        ann = build_patch_features(inputs, windows).reshape(-1, 128)
        a1_out = fixed_mlp_batched(self.a1.proposal, ann, 4096)
        temporal = build_temporal_patch_features(inputs, windows, self.s1.time_steps)
        temporal = temporal.transpose(1, 0, 2, 3).reshape(self.s1.time_steps, -1, 128)
        s1_logits = np.concatenate([self.s1.forward(temporal[:, i:i + 4096]).logits
                                    for i in range(0, temporal.shape[1], 4096)], axis=0)
        for name, logits, threshold in (
            ("a1_h2", a1_out.logits, self.h2.proposal_threshold),
            ("s1_mixed", s1_logits, float(self.s1_meta["proposal_threshold"])),
        ):
            heatmaps = proposal_heatmaps(logits)
            candidates = candidate_indices(inputs, windows, heatmaps, threshold)
            probabilities = np.zeros(len(inputs), dtype=np.float32)
            if len(candidates):
                features = build_refiner_features(inputs, candidates, windows, heatmaps, threshold)
                logits = fixed_mlp_batched(self.a1.refiner, features, 4096).logits
                probabilities[candidates] = _refiner_probability(logits, self.h2.prototypes)
            yield name, probabilities, candidates
