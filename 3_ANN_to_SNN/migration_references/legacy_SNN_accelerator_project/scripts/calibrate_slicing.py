from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software"))

from snn_accel.calibration import event_count_candidates
from snn_accel.events import SliceConfig, load_ev_uav_npz
from snn_accel.slicing import slice_events


def main() -> None:
    parser = argparse.ArgumentParser(description="derive P25/P50/P75 event-count candidates from training traces")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--glob", default="*.npz")
    parser.add_argument("--key", default="ev")
    parser.add_argument("--timestamp-unit", choices=("ns", "us", "ms", "s"), required=True)
    parser.add_argument("--fixed-duration-us", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "generated" / "slice_candidates.json")
    args = parser.parse_args()

    files = sorted(args.data_root.rglob(args.glob))
    if not files:
        raise SystemExit(f"no files matching {args.glob!r} under {args.data_root}")
    config = SliceConfig(mode="fixed_time", fixed_duration_us=args.fixed_duration_us)
    counts: list[int] = []
    for path in files:
        events = load_ev_uav_npz(path, key=args.key, timestamp_unit=args.timestamp_unit)
        counts.extend(item.stop - item.start for item in slice_events(events, config))
    result = {
        "timestamp_unit": args.timestamp_unit,
        "fixed_duration_us": args.fixed_duration_us,
        "files": len(files),
        "windows": len(counts),
        "candidates": event_count_candidates(np.asarray(counts)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

