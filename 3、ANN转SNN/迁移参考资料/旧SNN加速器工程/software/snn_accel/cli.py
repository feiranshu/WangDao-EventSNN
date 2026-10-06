from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .events import SliceConfig, load_ev_uav_npz
from .export import verify_manifest
from .pipeline import preprocess


def _preprocess(args: argparse.Namespace) -> None:
    config_values = json.loads(Path(args.config).read_text(encoding="utf-8"))
    timestamp_unit = args.timestamp_unit or config_values.get("source_timestamp_unit", "us")
    events = load_ev_uav_npz(args.input, key=args.key, timestamp_unit=timestamp_unit)
    result = preprocess(events, SliceConfig.from_mapping(config_values))
    targets = [
        {
            "slice_id": item.slice_id,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "targets": [box.__dict__ for box in item.targets],
        }
        for item in result.slices
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "aer_packets.npy", result.packets)
    np.save(output / "aer_tlast.npy", result.tlast)
    (output / "targets.json").write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="snn-accel")
    subparsers = parser.add_subparsers(required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="convert EV-UAV .npz to AER packets")
    preprocess_parser.add_argument("input")
    preprocess_parser.add_argument("output")
    preprocess_parser.add_argument("--key", default="ev")
    preprocess_parser.add_argument("--timestamp-unit", choices=("ns", "us", "ms", "s"))
    preprocess_parser.add_argument("--config", default="configs/slice_config.json")
    preprocess_parser.set_defaults(func=_preprocess)

    verify_parser = subparsers.add_parser("verify-manifest", help="verify exported artifact checksums")
    verify_parser.add_argument("directory")
    verify_parser.set_defaults(func=lambda args: verify_manifest(args.directory))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
