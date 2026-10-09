#!/usr/bin/env python3
"""Run the final A4-2 exact-int8 detector on one EV-UAV NPZ sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.a4_2_runtime import A42Pipeline  # noqa: E402
from wangdao.data import load_events  # noqa: E402


DEFAULT_H2 = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
DEFAULT_A4_2 = ROOT / "software" / "weights" / "A4-2_seed42_epoch008_int8.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="EV-UAV NPZ containing the ev structured array")
    parser.add_argument("--output", type=Path, required=True, help="new NPZ for probabilities and predictions")
    parser.add_argument("--h2-weights", type=Path, default=DEFAULT_H2)
    parser.add_argument("--a4-2-weights", type=Path, default=DEFAULT_A4_2)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    events = load_events(args.input, require_labels=False)
    result = A42Pipeline(args.h2_weights, args.a4_2_weights, args.batch_size).infer(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=result["probabilities"],
            predictions=result["predictions"],
            candidate_indices=result["candidate_indices"],
            threshold=np.asarray(result["threshold"], dtype=np.float32),
            model_id=np.asarray(result["model_id"]),
        )
    summary = {
        "model_id": result["model_id"],
        "threshold": result["threshold"],
        "events": len(events),
        "candidate_events": len(result["candidate_indices"]),
        "positive_predictions": int(np.count_nonzero(result["predictions"])),
        "output": str(args.output),
        "timings": result["timings"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

