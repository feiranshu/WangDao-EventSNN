#!/usr/bin/env python3
"""Recount frozen event predictions, without inference or threshold selection."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from wangdao.localization_v2 import PROTOCOL_ID, localization_counts, summarize_counts
from evaluation_contract import PREFIX_SCOPE, SCHEMA, read_json, sha256, verify_record


def recount(manifest_path, evaluation_root, output):
    manifest_path, evaluation_root, output = map(Path, (manifest_path, evaluation_root, output))
    manifest = read_json(manifest_path)
    data = copy.deepcopy(manifest["record"])
    policy = read_json(evaluation_root / "comparison_contract.json")
    split = data["split"]
    with (evaluation_root / "model_registry.csv").open(encoding="utf-8-sig", newline="") as stream:
        model = next(row for row in csv.DictReader(stream) if row["evaluation_id"] == data["evaluation_id"])
    entries = manifest["sequences"]
    expected = policy["sequence_ids"][split]
    if sorted(row["sequence_id"] for row in entries) != sorted(expected):
        raise ValueError("Manifest must include each frozen split sequence exactly once")
    data.update(record_schema=SCHEMA, protocol_id=PROTOCOL_ID)
    if data["comparison"]["scope_id"] != PREFIX_SCOPE:
        raise ValueError("Only the shared closed 100 ms prefix is supported")
    data["comparison"]["sequence_ids"] = expected
    detail = output.with_name(output.stem + "_per_sequence.json")
    if output.exists() or detail.exists():
        raise ValueError("Use a new output location; frozen records are never overwritten")
    rows = []
    for entry in entries:
        name = entry["sequence_id"]
        source = manifest_path.parent / entry["events_file"]
        prediction = manifest_path.parent / entry["probabilities_file"]
        source_hash, prediction_hash = sha256(source), sha256(prediction)
        if source_hash != policy["sequence_files"][split][name]["sha256"]:
            raise ValueError("Frozen event hash mismatch: " + name)
        if prediction_hash != entry["probabilities_sha256"]:
            raise ValueError("Frozen prediction hash mismatch: " + name)
        with np.load(source, allow_pickle=False) as archive:
            events = archive["ev"].copy()
        if not len(events) or np.any(np.diff(events["t"].astype(np.float64)) < 0):
            raise ValueError("Expected nonempty events in timestamp order")
        start = float(events["t"][0])
        windows = int((float(events["t"][-1]) - start) // 100.0)
        end = start + windows * 100.0
        n = int(np.searchsorted(events["t"], end, side="left"))
        with np.load(prediction, allow_pickle=False) as archive:
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
        if probabilities.ndim != 1 or len(probabilities) not in {n, len(events)}:
            raise ValueError("Scores must align with all input events or exactly the closed prefix")
        # Tail fillers are excluded by the explicit exposure and never scored.
        complete = np.zeros(len(events), dtype=np.float64)
        complete[:n] = probabilities[:n]
        counts = localization_counts(events["t"], events["x"], events["y"], events["label"],
                                     events["name"], complete, data["comparison"]["threshold"],
                                     start_ms=start, end_ms=end)
        rows.append({"sequence_id": name, "start_ms": start, "end_ms": end,
                     "source_event_sha256": source_hash, "prediction_sha256": prediction_hash,
                     "counts": counts})
    data["summary"] = summarize_counts([r["counts"] for r in rows])
    data["localization_counts_file"] = detail.name
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data["provenance"]["artifacts"] = [{"path": detail.name, "sha256": sha256(detail)}]
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = verify_record(evaluation_root, output, model, split, "localization")
    if errors:
        # Keep the failed output inspectable; it cannot pass table admission.
        raise ValueError("Recount record is unverified: " + "; ".join(errors))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evaluation-root", type=Path, default=Path(__file__).absolute().parents[4] / "5_model_evaluation")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(recount(args.manifest, args.evaluation_root, args.output))


if __name__ == "__main__":
    main()
