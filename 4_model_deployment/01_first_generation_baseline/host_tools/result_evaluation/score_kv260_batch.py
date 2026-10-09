"""Validate KV260 Proposal-SNN captures and run the frozen host-side pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))

from wangdao.data import load_events
from wangdao.fixed import FixedH2, FixedH2Config, load_fixed_proposal_snn
from wangdao.h2.numpy_model import _refiner_probability, load_export
from wangdao.h2.representation import (
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
)
from wangdao.h2.constants import SENSOR_HEIGHT, SENSOR_WIDTH
from wangdao.metrics import event_metrics, localization_metrics


H2_WEIGHT = ROOT / "model_runtime" / "weights" / "H2_D1_seed42.npz"
OUTPUT_DEFAULT = ROOT / "experiments" / "board_runs" / "KV260-BATCH-RESULT" / "evaluation" / "board_result.json"
PATCHES_PER_WINDOW = 336
OUTPUT_DIM = 16
Q12_SCALE = 4096.0


def portable_path(path: Path, root: Path = ROOT) -> str:
    """Use a relative provenance path only when both paths share a root."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_json(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent,
                                      mode="w", encoding="utf-8", newline="\n", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def signed_q12_to_probability(logits_q: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits_q, dtype=np.float64) / Q12_SCALE
    clipped = np.clip(logits, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def empty_totals() -> Dict[str, Any]:
    return {
        "probabilities": [],
        "labels": [],
        "candidate_events": 0,
        "retained_targets": 0,
        "correct_targets": 0.0,
        "localization_targets": 0.0,
        "false_components": 0.0,
        "frame_denominator": 0.0,
        "events": 0,
        "target_events": 0,
    }


def aggregate_metrics(totals: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    labels = np.concatenate(totals["labels"]) if totals["labels"] else np.empty(0, dtype=np.uint8)
    probabilities = np.concatenate(totals["probabilities"]) if totals["probabilities"] else np.empty(0, dtype=np.float32)
    event = event_metrics(labels, probabilities, threshold)
    return {
        "event": event,
        "candidate_events": int(totals["candidate_events"]),
        "candidate_fraction": (totals["candidate_events"] / totals["events"]
                                if totals["events"] else 0.0),
        "proposal_recall": (totals["retained_targets"] / totals["target_events"]
                             if totals["target_events"] else 1.0),
        "pd": (totals["correct_targets"] / totals["localization_targets"]
               if totals["localization_targets"] else 1.0),
        "fa": (totals["false_components"]
               / (totals["frame_denominator"] * SENSOR_WIDTH * SENSOR_HEIGHT)
               if totals["frame_denominator"] else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path,
                        help="directory containing <sequence>.capture.bin files")
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--experiment-id", default="KV260-BATCH-BOARD-v1")
    arguments = parser.parse_args()

    manifest_path = arguments.manifest.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("status") != "ok" or manifest.get("schema_version") != 1:
        raise ValueError("manifest is not a successful KV260 batch manifest")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("manifest contains no sequence records")
    package_root = manifest_path.parent
    capture_dir = (arguments.capture_dir.resolve() if arguments.capture_dir
                   else package_root / "captures")
    h2_path = arguments.h2_weight.resolve()
    s1_name = manifest.get("source", {}).get("s1_weight")
    if not isinstance(s1_name, str):
        raise ValueError("manifest has no S1 weight identity")
    s1_path = (ROOT / s1_name).resolve()
    if not s1_path.is_file() or not h2_path.is_file():
        raise FileNotFoundError("frozen S1/H2 weight file is missing")
    if manifest["source"].get("s1_weight_sha256") != sha256_file(s1_path):
        raise ValueError("S1 weight hash does not match the batch manifest")

    _s1, _s1_configuration, s1_metadata = load_fixed_proposal_snn(s1_path)
    h2_configuration = s1_metadata.get("h2_fixed_configuration")
    if not isinstance(h2_configuration, dict):
        raise ValueError("S1 metadata has no frozen H2 configuration")
    refiner = FixedH2(load_export(h2_path), FixedH2Config(
        accumulator_bits=int(h2_configuration["accumulator_bits"]),
        scale_fraction_bits=int(h2_configuration["scale_fraction_bits"]),
        logit_fraction_bits=int(h2_configuration["logit_fraction_bits"]),
    ))
    proposal_threshold = float(s1_metadata["proposal_threshold"])
    event_threshold = float(s1_metadata["event_threshold"])

    totals = empty_totals()
    per_sequence: List[Dict[str, Any]] = []
    mismatches = 0
    actual_words = 0
    expected_words = 0
    started = time.perf_counter()
    for number, record in enumerate(sequences, start=1):
        if not isinstance(record, dict):
            raise ValueError("manifest sequence record is not an object")
        name = str(record["name"])
        windows = [int(value) for value in record["windows"]]
        patches = len(windows) * PATCHES_PER_WINDOW
        if int(record["patches"]) != patches:
            raise ValueError(f"manifest patch count mismatch for {name}")
        stem = Path(name).stem
        capture_path = capture_dir / f"{stem}.capture.bin"
        golden_path = package_root / str(record["golden_file"])
        expected_size = patches * OUTPUT_DIM * 4
        if not capture_path.is_file():
            raise FileNotFoundError(f"capture is missing: {capture_path}")
        if not golden_path.is_file():
            raise FileNotFoundError(f"golden output is missing: {golden_path}")
        if capture_path.stat().st_size != expected_size or golden_path.stat().st_size != expected_size:
            raise ValueError(f"capture/golden size mismatch for {name}")
        actual = np.fromfile(capture_path, dtype="<i4").reshape(patches, OUTPUT_DIM)
        golden = np.fromfile(golden_path, dtype="<i4").reshape(patches, OUTPUT_DIM)
        difference = actual.astype(np.int64) - golden.astype(np.int64)
        mismatch_mask = difference != 0
        mismatch_count = int(np.count_nonzero(mismatch_mask))
        mismatches += mismatch_count
        actual_words += int(actual.size)
        expected_words += int(golden.size)

        events = load_events((ROOT / str(record["path"])).resolve(), require_labels=True)
        scores = signed_q12_to_probability(actual).reshape(len(windows), PATCHES_PER_WINDOW, OUTPUT_DIM)
        heatmaps = patch_scores_to_heatmaps(scores)
        candidates = candidate_indices(events, windows, heatmaps, proposal_threshold)
        features = build_refiner_features(events, candidates, windows, heatmaps, proposal_threshold)
        if len(features):
            refiner_output = refiner.refiner(features)
            candidate_probability = _refiner_probability(refiner_output.logits, refiner.source.prototypes)
        else:
            candidate_probability = np.empty(0, dtype=np.float32)
        probability = np.zeros(len(events), dtype=np.float32)
        probability[candidates] = candidate_probability
        labels = events["label"].astype(np.uint8, copy=False)
        localization = localization_metrics(
            events["t"], events["x"], events["y"], labels, events["name"],
            probability, threshold=event_threshold,
        )
        retained = int(labels[candidates].sum())
        totals["probabilities"].append(probability)
        totals["labels"].append(labels)
        totals["candidate_events"] += len(candidates)
        totals["retained_targets"] += retained
        totals["correct_targets"] += localization["correct_targets"]
        totals["localization_targets"] += localization["targets"]
        totals["false_components"] += localization["false_components"]
        totals["frame_denominator"] += localization["frame_denominator"]
        totals["events"] += len(events)
        totals["target_events"] += int(labels.sum())
        per_sequence.append({
            "name": name,
            "role": record.get("role"),
            "windows": windows,
            "patches": patches,
            "capture_file": str(capture_path.relative_to(capture_dir.parent).as_posix()),
            "capture_sha256": sha256_file(capture_path),
            "golden_sha256": sha256_file(golden_path),
            "actual_words": int(actual.size),
            "mismatch_words": mismatch_count,
            "max_abs_logit_error": int(np.max(np.abs(difference))) if difference.size else 0,
            "events": int(len(events)),
            "target_events": int(labels.sum()),
            "candidate_events": int(len(candidates)),
            "event": event_metrics(labels, probability, event_threshold),
            "proposal_recall": retained / int(labels.sum()) if int(labels.sum()) else 1.0,
        })
        print(f"[{number:03d}/{len(sequences):03d}] {name}: mismatches={mismatch_count} candidates={len(candidates)}",
              flush=True)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "status": "ok" if mismatches == 0 else "failed",
        "experiment_id": arguments.experiment_id,
        "stage": "P5-kv260-board-batch-and-host-integration",
        "data_role": manifest.get("data_role"),
        "primary_metrics": {
            "sequences": len(sequences),
            "patches": int(manifest["primary_metrics"]["patches"]),
            "logit_words": actual_words,
            "mismatch_words": mismatches,
            "max_abs_logit_error": max((item["max_abs_logit_error"] for item in per_sequence), default=0),
            "events": totals["events"],
            "target_events": totals["target_events"],
        },
        "decision": "board_exact_and_host_pipeline_passed" if mismatches == 0 else "board_output_mismatch",
        "conclusion": ("All board Proposal-SNN logits match the frozen exact golden model; "
                       "candidate routing and A1-H2 Refiner were evaluated on the board outputs."
                       if mismatches == 0 else
                       "Board output differs from the frozen exact golden model; host metrics are diagnostic only."),
        "test_used": bool(manifest.get("source", {}).get("test_used", False)),
        "validation_used": any(record.get("role") == "validation" for record in sequences),
        "configuration": manifest.get("configuration"),
        "model": {
            "s1_weight": portable_path(s1_path),
            "s1_weight_sha256": sha256_file(s1_path),
            "h2_weight": portable_path(h2_path),
            "h2_weight_sha256": sha256_file(h2_path),
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
            "h2_fixed_configuration": h2_configuration,
        },
        "board": {
            "base_address": "0xA0000000",
            "capture_format": "little-endian signed int32, 16 words per patch",
            "golden_words": expected_words,
            "actual_words": actual_words,
        },
        "evaluation": aggregate_metrics(totals, event_threshold),
        "per_sequence": per_sequence,
        "reproducibility": {
            "manifest": portable_path(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "elapsed_s": time.perf_counter() - started,
        },
    }
    atomic_json(result, arguments.output.resolve())
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "mismatch_words": mismatches,
                      "output": str(arguments.output.resolve())}, ensure_ascii=False))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
