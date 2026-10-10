#!/usr/bin/env python3
"""Run the frozen causal H2-pro exact-int8 model on one EV-UAV sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events  # noqa: E402
from wangdao.h2_pro.stream_state import CausalPipeline  # noqa: E402

DEFAULT_H2 = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
DEFAULT_PRO = ROOT / "software" / "weights" / "H2-pro-Causal-ANN-v1_int8.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="EV-UAV NPZ containing an ev array")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h2-weights", type=Path, default=DEFAULT_H2)
    parser.add_argument("--pro-weights", type=Path, default=DEFAULT_PRO)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    events = load_events(args.input, require_labels=False)
    if not len(events):
        raise ValueError("the sequence must contain at least one event")
    pipeline = CausalPipeline(args.h2_weights, args.pro_weights,
                              variant="C1-proposal-motion", threshold=0.35,
                              initial_refiner=False)
    pipeline.reset(float(events["t"][0]))
    # Only complete 100 ms windows are published. The trailing partial window is
    # intentionally excluded from the formal same-prefix evaluation protocol.
    pipeline.push(events, float(events["t"][-1]) + 1e-6)
    windows, trailing_events = pipeline.finalize()
    probabilities = np.concatenate([w.probabilities for w in windows]) if windows else np.empty(0, dtype=np.float32)
    predictions = np.concatenate([w.predictions for w in windows]) if windows else np.empty(0, dtype=bool)
    candidate_indices = np.concatenate([w.event_indices[w.candidate_mask] for w in windows]) if windows else np.empty(0, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, probabilities=probabilities, predictions=predictions,
                            candidate_indices=candidate_indices, threshold=np.float32(0.35),
                            model_id=np.asarray("H2-pro-Causal-ANN-v1"),
                            trailing_events=np.int64(trailing_events))
    print(json.dumps({"model_id": "H2-pro-Causal-ANN-v1", "threshold": 0.35,
                      "published_events": len(probabilities),
                      "trailing_events": trailing_events, "closed_windows": len(windows),
                      "candidate_events": len(candidate_indices),
                      "positive_predictions": int(predictions.sum()),
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
