"""Match multiple board captures to the 24 byte-lane candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import read_hex_words, sha256_file, write_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated" / "lane_probe",
    )
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    probe_dir = arguments.probe_dir.resolve()
    manifest_path = probe_dir / "lane_probe_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    matches = []
    differences = {}
    actual_records = []
    order_names = sorted(manifest["probes"][0]["expected_files"])
    for order in order_names:
        total_difference = 0
        for probe in manifest["probes"]:
            stem = Path(probe["input_words"]).name.removesuffix("_words.txt")
            actual_path = probe_dir / f"{stem}_actual.txt"
            actual = read_hex_words(actual_path, expected_count=16)
            expected = read_hex_words(
                probe_dir / probe["expected_files"][order], expected_count=16
            )
            total_difference += int(np.count_nonzero(actual != expected))
            if not any(record["file"] == actual_path.name for record in actual_records):
                actual_records.append(
                    {"file": actual_path.name, "sha256": sha256_file(actual_path)}
                )
        differences[order] = total_difference
        if total_difference == 0:
            matches.append(order)

    result = {
        "probe_manifest": str(manifest_path),
        "actual_captures": actual_records,
        "matches": matches,
        "total_word_difference_count_by_order": differences,
    }
    result_path = arguments.result or probe_dir / "lane_probe_match.json"
    write_json(result_path, result)

    if len(matches) != 1:
        print(f"LANE_PROBE_FAIL matches={matches}")
        closest = sorted(differences.items(), key=lambda item: item[1])[:5]
        print(f"closest candidates: {closest}")
        return 1

    order = matches[0]
    print(f"LANE_PROBE_PASS time_to_lane={order}")
    for time_step, lane in enumerate(order):
        print(f"t{time_step} -> lane{lane} bits[{8 * int(lane) + 7}:{8 * int(lane)}]")
    print(f"result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

