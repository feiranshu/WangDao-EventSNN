"""Organize and evaluate the KV260 Train-dev board results in-place.

The script keeps raw board artifacts untouched.  Direct board metrics use the
two-byte-per-event PS output.  PR/AP are reconstructed from the bit-exact PL
capture with the frozen fixed-point H2 refiner and are labelled as replayed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
OUTPUT = SCRIPT.parent
RESULT_ROOT = OUTPUT.parent
DEPLOY_ROOT = SCRIPT.parents[4]
PROJECT_ROOT = DEPLOY_ROOT.parents[1]
BATCH_ROOT = DEPLOY_ROOT / "archive" / "legacy_generated_packages" / "kv260_batch" / "train_dev10"
FULL_ROOT = DEPLOY_ROOT / "archive" / "legacy_generated_packages" / "kv260_full" / "train_dev10"
CAPTURE_ROOT = RESULT_ROOT / "raw" / "captures"
PREDICTION_ROOT = RESULT_ROOT / "raw" / "predictions"
SOFTWARE_ROOT = DEPLOY_ROOT / "model_runtime"

sys.path.insert(0, str(SOFTWARE_ROOT))

from wangdao.fixed import FixedH2, FixedH2Config, load_fixed_proposal_snn  # noqa: E402
from wangdao.h2.constants import SENSOR_HEIGHT, SENSOR_WIDTH  # noqa: E402
from wangdao.h2.numpy_model import (  # noqa: E402
    _refiner_probability,
    load_export,
    run_pipeline,
)
from wangdao.h2.representation import (  # noqa: E402
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
)


PATCHES_PER_WINDOW = 336
OUTPUT_DIM = 16
Q12_SCALE = 4096.0
DENSITY_THRESHOLD_EPS = 20_000.0
EVENT_THRESHOLD = 0.2800000011920929
TECHNICAL_BINARY_THRESHOLD = 0.5
THRESHOLDS = tuple(sorted(set([round(index / 100, 2) for index in range(5, 96, 5)] + [EVENT_THRESHOLD])))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OFFICIAL = load_module(
    "wangdao_board_phase1_metrics",
    PROJECT_ROOT / "3_ANN_to_SNN" / "implementation" / "software" / "wangdao" / "phase1_metrics.py",
)
TRACK = load_module(
    "wangdao_board_target_tracking_metrics",
    PROJECT_ROOT / "1_H2_model_improvement" / "implementation" / "software" / "wangdao" / "target_tracking_metrics.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signed_q12_to_probability(logits_q: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits_q, dtype=np.float64) / Q12_SCALE
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))).astype(np.float32)


def load_evaluation_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        events = archive["ev"].copy()
        loc = archive["ev_loc"].copy()
        norm = archive["evs_norm"]
        labels = norm[:, 4].astype(np.uint8, copy=True)
        identities = norm[:, 5].astype(np.int32, copy=True)
    if len(events) == 0 or loc.shape != (len(events), 3):
        raise ValueError(f"invalid evaluation arrays: {path}")
    if not np.array_equal(events["label"], labels) or not np.array_equal(events["name"], identities):
        raise ValueError(f"label/identity mismatch: {path}")
    return events, loc, labels, identities


def aggregate_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    event_counts = {
        key: sum(int(row["event"][key]) for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }
    localization_counts = {
        key: sum(int(row["localization"][key]) for row in rows)
        for key in (
            "targets",
            "correct_targets",
            "false_components",
            "frame_denominator",
            "scored_localization_events",
            "wrapped_false_pixels",
        )
    }
    return {
        "event": OFFICIAL.event_summary(event_counts),
        "localization": OFFICIAL.localization_summary(localization_counts),
    }


def score(loc: np.ndarray, labels: np.ndarray, identities: np.ndarray,
          probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "event": OFFICIAL.event_summary(OFFICIAL.event_counts(labels, probabilities, threshold)),
        "localization": OFFICIAL.localization_summary(
            OFFICIAL.localization_counts(loc, labels, identities, probabilities, threshold)
        ),
    }


def exact_pr(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-probabilities, kind="stable")
    ordered_probability = probabilities[order]
    ordered_labels = labels[order] == 1
    ends = np.r_[np.flatnonzero(ordered_probability[1:] != ordered_probability[:-1]), len(order) - 1]
    tp = np.cumsum(ordered_labels, dtype=np.int64)[ends]
    precision = tp / (ends + 1)
    positives = int(ordered_labels.sum())
    recall = tp / positives if positives else np.zeros(len(ends), dtype=np.float64)
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision)) if positives else None
    return {
        "average_precision": average_precision,
        "target_events": positives,
        "thresholds": ordered_probability[ends],
        "precision": precision,
        "recall": recall,
    }


def sequence_rate(events: np.ndarray) -> float:
    duration_s = max((float(events["t"][-1]) - float(events["t"][0])) / 1000.0, 1.0e-9)
    return float(len(events) / duration_s)


def track_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = TRACK.combine_hota_raw([item["raw"] for item in items])
    misses = [value for item in items for value in item["longest_miss_per_track_ms"]]
    return {
        "protocol": TRACK.PROTOCOL_ID,
        "metrics": TRACK.hota_metrics(raw),
        "longest_continuous_miss_p95_ms": TRACK.percentile95(misses),
    }


def static_profile(layers: tuple[tuple[np.ndarray, np.ndarray], ...]) -> dict[str, Any]:
    return {
        "weight_parameters": sum(int(weight.size) for weight, _ in layers),
        "nonzero_weights": sum(int(np.count_nonzero(weight)) for weight, _ in layers),
        "weight_bytes": sum(int(weight.nbytes) for weight, _ in layers),
        "scale_bytes": sum(int(scale.nbytes) for _, scale in layers),
        "macs_per_call": sum(int(weight.size) for weight, _ in layers),
        "dimensions": [int(layers[0][0].shape[0])] + [int(weight.shape[1]) for weight, _ in layers],
    }


def metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("event", "localization"):
        result[group] = {}
        for key in candidate[group]:
            left = candidate[group][key]
            right = baseline[group].get(key)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                result[group][key] = float(left - right)
    return result


def make_plots(board_pr: Mapping[str, Any], ann_pr: Mapping[str, Any], thresholds: Sequence[Mapping[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - optional presentation artifact
        return [f"plots not generated: {error}"]
    generated: list[str] = []
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.plot(board_pr["recall"], board_pr["precision"], label=f"Board capture replay AP={board_pr['average_precision']:.4f}")
    axis.plot(ann_pr["recall"], ann_pr["precision"], label=f"ANN baseline AP={ann_pr['average_precision']:.4f}", alpha=0.85)
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.01), title="Train-dev10 event PR curve")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "pr_curve.png", dpi=170)
    plt.close(fig)
    generated.append("pr_curve.png")

    values = np.asarray([row["threshold"] for row in thresholds], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for field, label in (("iou", "Event IoU"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        axis.plot(values, [row[field] for row in thresholds], marker="o", markersize=3, label=label)
    axis.axvline(EVENT_THRESHOLD, color="black", linestyle="--", linewidth=1, label="Frozen threshold 0.28")
    axis.set(xlabel="Threshold", ylabel="Metric", ylim=(0, 1.01), title="Train-dev10 threshold metrics")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "threshold_metrics.png", dpi=170)
    plt.close(fig)
    generated.append("threshold_metrics.png")
    return generated


def main() -> None:
    batch_manifest_path = BATCH_ROOT / "manifest.json"
    ps_manifest_path = RESULT_ROOT / "manifests" / "ps_manifest.json"
    batch = read_json(batch_manifest_path)
    ps_manifest = read_json(ps_manifest_path)
    if sha256(batch_manifest_path) != ps_manifest["source"]["s1_manifest_sha256"]:
        raise ValueError("PS manifest is not bound to the local S1 batch manifest")

    h2_path = DEPLOY_ROOT / "model_runtime" / "weights" / "H2_D1_seed42.npz"
    s1_path = DEPLOY_ROOT / batch["source"]["s1_weight"]
    h2_export = load_export(h2_path)
    _, _, s1_metadata = load_fixed_proposal_snn(s1_path)
    h2_configuration = s1_metadata["h2_fixed_configuration"]
    fixed_h2 = FixedH2(
        h2_export,
        FixedH2Config(
            accumulator_bits=int(h2_configuration["accumulator_bits"]),
            scale_fraction_bits=int(h2_configuration["scale_fraction_bits"]),
            logit_fraction_bits=int(h2_configuration["logit_fraction_bits"]),
        ),
    )
    event_threshold = float(s1_metadata["event_threshold"])
    proposal_threshold = float(s1_metadata["proposal_threshold"])
    if event_threshold != EVENT_THRESHOLD:
        raise ValueError(f"unexpected frozen event threshold: {event_threshold}")

    protocol = read_json(
        PROJECT_ROOT / "5_model_evaluation" / "A_model_accuracy" / "03_rules_and_protocols"
        / "protocols_and_sources" / "target_track_protocol.json"
    )
    adapter = protocol["selected"]
    adapter_args = {
        "cluster_radius": int(adapter["cluster_radius_px"]),
        "association_distance": float(adapter["association_distance_px_per_50ms"]),
        "max_missed_windows": int(adapter["max_missed_windows"]),
    }

    probability_dir = OUTPUT / "replay_probabilities"
    probability_dir.mkdir(exist_ok=True)
    board_rows: list[dict[str, Any]] = []
    ann_rows: list[dict[str, Any]] = []
    sequence_csv: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    board_track_items: list[dict[str, Any]] = []
    ann_track_items: list[dict[str, Any]] = []
    all_labels: list[np.ndarray] = []
    all_board_probability: list[np.ndarray] = []
    all_ann_probability: list[np.ndarray] = []
    raw_artifacts: list[dict[str, Any]] = []
    total_candidates = 0
    total_targets_retained = 0

    for number, record in enumerate(batch["sequences"], start=1):
        name = str(record["name"])
        stem = Path(name).stem
        dataset_path = Path(str(record["path"]))
        capture = CAPTURE_ROOT / f"{stem}.capture.bin"
        s1_golden = BATCH_ROOT / str(record["golden_file"])
        prediction = PREDICTION_ROOT / f"{stem}.pred.bin"
        h2_golden = FULL_ROOT / "h2_golden" / f"{stem}.pred.bin"
        for artifact in (dataset_path, capture, s1_golden, prediction, h2_golden):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)

        capture_hash = sha256(capture)
        s1_golden_hash = sha256(s1_golden)
        prediction_hash = sha256(prediction)
        h2_golden_hash = sha256(h2_golden)
        pl_exact = capture_hash == s1_golden_hash
        ps_exact = prediction_hash == h2_golden_hash
        if not pl_exact or not ps_exact:
            raise ValueError(f"board/golden mismatch: {name}")

        events, loc, labels, identities = load_evaluation_arrays(dataset_path)
        raw_prediction = np.fromfile(prediction, dtype=np.uint8)
        if raw_prediction.size != 2 * len(events):
            raise ValueError(f"prediction size mismatch: {name}")
        raw_prediction = raw_prediction.reshape(-1, 2)
        if not np.all(np.isin(raw_prediction, (0, 1))):
            raise ValueError(f"non-binary PS output: {name}")
        board_candidate = raw_prediction[:, 0].astype(bool)
        board_prediction = raw_prediction[:, 1].astype(bool)
        if np.any(board_prediction & ~board_candidate):
            raise ValueError(f"PS prediction escaped proposal gate: {name}")

        windows = [int(value) for value in record["windows"]]
        patches = len(windows) * PATCHES_PER_WINDOW
        capture_logits = np.fromfile(capture, dtype="<i4")
        if capture_logits.size != patches * OUTPUT_DIM:
            raise ValueError(f"capture size mismatch: {name}")
        capture_logits = capture_logits.reshape(patches, OUTPUT_DIM)
        scores = signed_q12_to_probability(capture_logits).reshape(len(windows), PATCHES_PER_WINDOW, OUTPUT_DIM)
        heatmaps = patch_scores_to_heatmaps(scores)
        replay_candidates = candidate_indices(events, windows, heatmaps, proposal_threshold)
        replay_probability = np.zeros(len(events), dtype=np.float32)
        if len(replay_candidates):
            features = build_refiner_features(events, replay_candidates, windows, heatmaps, proposal_threshold)
            fixed_output = fixed_h2.refiner(features)
            replay_probability[replay_candidates] = _refiner_probability(fixed_output.logits, h2_export.prototypes)
        replay_candidate_mask = np.zeros(len(events), dtype=bool)
        replay_candidate_mask[replay_candidates] = True
        if not np.array_equal(replay_candidate_mask, board_candidate):
            raise ValueError(f"replayed candidate mask differs from PS output: {name}")
        if not np.array_equal(replay_probability >= event_threshold, board_prediction):
            raise ValueError(f"replayed prediction differs from PS output: {name}")

        ann_output = run_pipeline(events, h2_export)
        ann_probability = ann_output["probabilities"].astype(np.float32, copy=False)
        np.savez_compressed(
            probability_dir / f"{stem}.npz",
            board_capture_replay_probability=replay_probability,
            ann_h2_baseline_probability=ann_probability,
        )

        board_score = score(loc, labels, identities, board_prediction.astype(np.float32), TECHNICAL_BINARY_THRESHOLD)
        ann_score = score(loc, labels, identities, ann_probability, event_threshold)
        board_rows.append(board_score)
        ann_rows.append(ann_score)
        rate = sequence_rate(events)
        high_density = rate >= DENSITY_THRESHOLD_EPS
        retained_targets = int(labels[board_candidate].sum())
        total_candidates += int(board_candidate.sum())
        total_targets_retained += retained_targets

        board_track = TRACK.evaluate_sequence(
            events, board_prediction.astype(np.float32), TECHNICAL_BINARY_THRESHOLD, **adapter_args
        )
        ann_track = TRACK.evaluate_sequence(events, ann_probability, event_threshold, **adapter_args)
        board_track_items.append(board_track)
        ann_track_items.append(ann_track)
        board_hota = board_track["metrics"]["averaged"]
        ann_hota = ann_track["metrics"]["averaged"]

        event = board_score["event"]
        localization = board_score["localization"]
        sequence_csv.append({
            "sequence": name,
            "high_density": high_density,
            "event_rate_eps": rate,
            "events": len(events),
            "target_events": int(labels.sum()),
            "candidate_events": int(board_candidate.sum()),
            "candidate_fraction": float(board_candidate.mean()),
            "proposal_recall": retained_targets / int(labels.sum()) if int(labels.sum()) else None,
            "tp": event["tp"], "fp": event["fp"], "fn": event["fn"], "tn": event["tn"],
            "event_iou": event["iou"], "precision": event["precision"], "recall_official_acc": event["recall"],
            "f1": event["f1"], "overall_accuracy": event["overall_accuracy"],
            "pd": localization["pd"], "fa": localization["fa"], "fa_times_1e4": localization["fa_times_1e4"],
            "HOTA": board_hota["HOTA"], "DetA": board_hota["DetA"], "AssA": board_hota["AssA"], "LocA": board_hota["LocA"],
            "ann_HOTA": ann_hota["HOTA"],
            "pl_exact_match": pl_exact, "ps_exact_match": ps_exact,
        })
        complexity_rows.append({
            "sequence": name,
            "events": len(events),
            "patches": patches,
            "candidate_events": int(board_candidate.sum()),
            "proposal_scheduled_dense_synaptic_products": int(4 * patches * 7680),
            "proposal_membrane_updates": int(4 * patches * (48 + 24 + 2 * 16)),
            "refiner_dense_macs": int(board_candidate.sum()) * 4224,
        })
        for kind, path in (("pl_capture", capture), ("ps_prediction", prediction)):
            raw_artifacts.append({
                "sequence": name,
                "kind": kind,
                "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        all_labels.append(labels)
        all_board_probability.append(replay_probability)
        all_ann_probability.append(ann_probability)
        print(f"[{number:02d}/{len(batch['sequences']):02d}] {name}: events={len(events)} candidates={int(board_candidate.sum())}", flush=True)

    labels = np.concatenate(all_labels)
    board_probability = np.concatenate(all_board_probability)
    ann_probability = np.concatenate(all_ann_probability)
    board_overall = aggregate_scores(board_rows)
    ann_overall = aggregate_scores(ann_rows)
    board_pr = exact_pr(labels, board_probability)
    ann_pr = exact_pr(labels, ann_probability)

    high_indices = [index for index, row in enumerate(sequence_csv) if row["high_density"]]
    ordinary_indices = [index for index, row in enumerate(sequence_csv) if not row["high_density"]]
    density = {
        "threshold_events_per_second": DENSITY_THRESHOLD_EPS,
        "high_density": aggregate_scores([board_rows[index] for index in high_indices]) if high_indices else None,
        "ordinary_density": aggregate_scores([board_rows[index] for index in ordinary_indices]) if ordinary_indices else None,
        "high_density_sequences": [sequence_csv[index]["sequence"] for index in high_indices],
        "ordinary_density_sequences": [sequence_csv[index]["sequence"] for index in ordinary_indices],
    }

    threshold_rows: list[dict[str, Any]] = []
    offsets = np.cumsum([0] + [len(part) for part in all_labels])
    for threshold in THRESHOLDS:
        rows = []
        for index, record in enumerate(batch["sequences"]):
            dataset_path = Path(str(record["path"]))
            _, loc, sequence_labels, identities = load_evaluation_arrays(dataset_path)
            probability = all_board_probability[index]
            rows.append(score(loc, sequence_labels, identities, probability, float(threshold)))
        combined = aggregate_scores(rows)
        threshold_rows.append({
            "threshold": float(threshold),
            "iou": combined["event"]["iou"],
            "precision": combined["event"]["precision"],
            "recall": combined["event"]["recall"],
            "f1": combined["event"]["f1"],
            "pd": combined["localization"]["pd"],
            "fa": combined["localization"]["fa"],
        })

    board_track_summary = track_summary(board_track_items)
    ann_track_summary = track_summary(ann_track_items)
    board_hota_avg = board_track_summary["metrics"]["averaged"]
    ann_hota_avg = ann_track_summary["metrics"]["averaged"]

    proposal_profile = {
        "weight_parameters": 7680,
        "nonzero_weights": sum(int(np.count_nonzero(np.load(s1_path, allow_pickle=False)[f"weight_{index}"])) for index in range(3)),
        "weight_bytes": 7680,
        "time_steps": 4,
        "persistent_membrane_state_bytes_per_patch": int((48 + 24 + 2 * 16) * 3),
    }
    refiner_profile = static_profile(h2_export.refiner_layers)
    complexity_summary = {
        "protocol_relation": "B_model_complexity-compatible diagnostic for the first-generation hybrid SNN; Train-dev only",
        "scope": "S1 fixed Proposal-SNN (T=4) in PL + conditional fixed H2 Refiner in PS",
        "proposal": proposal_profile,
        "refiner": refiner_profile,
        "total_weight_parameters": proposal_profile["weight_parameters"] + refiner_profile["weight_parameters"],
        "total_nonzero_weights": proposal_profile["nonzero_weights"] + refiner_profile["nonzero_weights"],
        "total_logical_int8_weight_bytes": proposal_profile["weight_bytes"] + refiner_profile["weight_bytes"],
        "sequences": len(sequence_csv),
        "events": int(len(labels)),
        "patches": int(sum(row["patches"] for row in complexity_rows)),
        "candidate_events": int(total_candidates),
        "candidate_fraction": float(total_candidates / len(labels)),
        "proposal_scheduled_dense_synaptic_products": int(sum(row["proposal_scheduled_dense_synaptic_products"] for row in complexity_rows)),
        "proposal_membrane_updates": int(sum(row["proposal_membrane_updates"] for row in complexity_rows)),
        "refiner_dense_macs": int(sum(row["refiner_dense_macs"] for row in complexity_rows)),
        "event_driven_synaptic_accumulations": None,
        "event_driven_status": "not measurable from final logits; internal spike counters were not captured",
        "interpretation": "algorithmic operation counts, not measured FPGA latency or energy",
    }

    implementation_path = DEPLOY_ROOT / "verification_evidence/implementation" / "kv260_synthesis_implementation.json"
    implementation = read_json(implementation_path)
    deployment = {
        "protocol_relation": "C_deployment_efficiency_and_hardware_cost",
        "B01_model_kernel_latency": {"status": "not_measured", "reason": "no cycle counter or runtime log in the returned archive"},
        "B02_end_to_end_latency": {"status": "not_measured", "reason": "no monotonic-clock timing samples in the returned archive"},
        "B03_throughput_realtime": {"status": "not_measured", "reason": "file timestamps are not a valid per-run timing boundary"},
        "B04_runtime_power": {"status": "not_measured", "reason": "no board rail or external meter samples were returned"},
        "B05_energy_per_task": {"status": "not_measured", "reason": "requires both runtime and active/idle power measurements"},
        "B07_hardware_resources": {
            "status": "available_from_matching_exact_bitstream_implementation_report",
            "source": str(implementation_path),
            "source_sha256": sha256(implementation_path),
            **implementation["primary_metrics"],
            "power_note": "vectorless_total_on_chip_power_w is a Vivado estimate, not board runtime or PL-only measured power",
        },
    }

    delta = metric_delta(board_overall, ann_overall)
    ann_snn = {
        "protocol_relation": "D_ANN_and_SNN_differences; diagnostic Train-dev comparison",
        "comparison": "first-generation S1 fixed Proposal-SNN T=4 + fixed H2 versus frozen H2 exact-INT8 ANN pipeline",
        "threshold": event_threshold,
        "board_snn": {"working_point": board_overall, "average_precision": board_pr["average_precision"], "HOTA": board_hota_avg},
        "ann_baseline": {"working_point": ann_overall, "average_precision": ann_pr["average_precision"], "HOTA": ann_hota_avg},
        "snn_minus_ann": {
            **delta,
            "average_precision": float(board_pr["average_precision"] - ann_pr["average_precision"]),
            "HOTA": float(board_hota_avg["HOTA"] - ann_hota_avg["HOTA"]),
        },
        "limitations": [
            "The delta combines Proposal ANN-to-SNN conversion and fixed-point deployment effects; it cannot isolate each intervention.",
            "The board result contains one T=4 point, so a multi-time-step C01 curve cannot be measured.",
        ],
    }

    summary = {
        "schema_version": 1,
        "status": "complete_with_unmeasured_runtime_fields",
        "experiment_id": "KV260-TRAIN-DEV10-BOARD-EVAL-v1",
        "data_role": "train_dev",
        "formal_evaluation_destination": False,
        "sequence_count": len(sequence_csv),
        "events": int(len(labels)),
        "target_events": int(labels.sum()),
        "board_correctness": {
            "pl_capture_exact_sequences": sum(bool(row["pl_exact_match"]) for row in sequence_csv),
            "ps_prediction_exact_sequences": sum(bool(row["ps_exact_match"]) for row in sequence_csv),
            "pl_mismatch_words": 0,
            "ps_mismatch_bytes": 0,
        },
        "working_point": {
            "source": "direct PS board prediction bytes",
            "frozen_event_threshold": event_threshold,
            **board_overall,
            "candidate_events": int(total_candidates),
            "candidate_fraction": float(total_candidates / len(labels)),
            "proposal_recall": float(total_targets_retained / labels.sum()),
        },
        "event_pr": {
            "source": "bit-exact PL capture plus frozen fixed H2 replay on PC",
            "average_precision": board_pr["average_precision"],
            "target_events": board_pr["target_events"],
            "direct_board_probability_recorded": False,
        },
        "density_groups": density,
        "target_track": {
            "source": "direct PS board prediction bytes with frozen shared adapter",
            "adapter": adapter_args,
            **board_track_summary,
        },
        "model_complexity": complexity_summary,
        "deployment": deployment,
        "ann_snn_difference": ann_snn,
        "provenance": {
            "batch_manifest": str(batch_manifest_path),
            "batch_manifest_sha256": sha256(batch_manifest_path),
            "ps_manifest": str(ps_manifest_path),
            "ps_manifest_sha256": sha256(ps_manifest_path),
            "s1_weight": str(s1_path),
            "s1_weight_sha256": sha256(s1_path),
            "h2_weight": str(h2_path),
            "h2_weight_sha256": sha256(h2_path),
            "evaluation_script_sha256": sha256(SCRIPT),
        },
    }

    write_json(OUTPUT / "summary.json", summary)
    write_json(OUTPUT / "hota_summary.json", {"board_snn": board_track_summary, "ann_baseline": ann_track_summary, "adapter": adapter_args})
    write_json(OUTPUT / "model_complexity.json", complexity_summary)
    write_json(OUTPUT / "deployment_metrics.json", deployment)
    write_json(OUTPUT / "ann_snn_difference.json", ann_snn)
    write_csv(OUTPUT / "per_sequence.csv", sequence_csv)
    write_csv(OUTPUT / "complexity_per_sequence.csv", complexity_rows)
    write_csv(OUTPUT / "threshold_curve.csv", threshold_rows)
    write_csv(OUTPUT / "pr_curve.csv", [
        {"threshold": float(threshold), "precision": float(precision), "recall": float(recall)}
        for threshold, precision, recall in zip(board_pr["thresholds"], board_pr["precision"], board_pr["recall"])
    ])
    write_csv(OUTPUT / "ann_baseline_pr_curve.csv", [
        {"threshold": float(threshold), "precision": float(precision), "recall": float(recall)}
        for threshold, precision, recall in zip(ann_pr["thresholds"], ann_pr["precision"], ann_pr["recall"])
    ])
    raw_artifacts.extend([
        {"sequence": "", "kind": "ps_manifest", "relative_path": "manifests/ps_manifest.json", "bytes": ps_manifest_path.stat().st_size, "sha256": sha256(ps_manifest_path)},
        {"sequence": "", "kind": "source_batch_manifest", "relative_path": str(batch_manifest_path), "bytes": batch_manifest_path.stat().st_size, "sha256": sha256(batch_manifest_path)},
    ])
    write_csv(OUTPUT / "artifact_checksums.csv", raw_artifacts)
    plots = make_plots(board_pr, ann_pr, threshold_rows)

    event = board_overall["event"]
    localization = board_overall["localization"]
    readme = f"""# KV260 Train-dev10 板端结果整理与评测

本目录只对应 `train_dev10` 开发集，不写入根目录 `5_model_evaluation`，不能作为 Validation/Test 正式结果。

## 核心结论

- PL Proposal-SNN：10/10 条序列与 Python 定点 golden 逐字一致。
- PS H2：10/10 条序列与 Python fixed golden 逐字节一致。
- 事件数：{len(labels):,}；目标事件：{int(labels.sum()):,}；候选事件：{total_candidates:,}。
- Event IoU：{event['iou']:.6f}；Precision：{event['precision']:.6f}；Recall/官方兼容 ACC：{event['recall']:.6f}；F1：{event['f1']:.6f}。
- Pd：{localization['pd']:.6f}；Fa：{localization['fa']:.9g}；Fa×10⁴：{localization['fa_times_1e4']:.6f}。
- Event AP：{board_pr['average_precision']:.6f}。该值由位级一致的 PL capture 在 PC 上重放冻结 fixed H2 概率得到，不是板端直接保存的概率。
- HOTA：{board_hota_avg['HOTA']:.6f}；DetA：{board_hota_avg['DetA']:.6f}；AssA：{board_hota_avg['AssA']:.6f}；LocA：{board_hota_avg['LocA']:.6f}。

## 文件说明

- `summary.json`：完整总览与证据边界。
- `per_sequence.csv`：逐序列事件指标、Pd/Fa、HOTA 与一致性。
- `threshold_curve.csv`、`pr_curve.csv`：基于 PL capture + fixed H2 重放的阈值曲线和 PR 曲线。
- `hota_summary.json`：板端预测与 ANN 基线的冻结轨迹协议结果。
- `model_complexity.json`、`complexity_per_sequence.csv`：可从本次输入规模确定的 B 类复杂度。
- `deployment_metrics.json`：C 类可用面积/时序，以及无法事后补算的运行时字段。
- `ann_snn_difference.json`：同一 Train-dev10 上的 T=4 SNN 与冻结 ANN 基线差值。
- `artifact_checksums.csv`：原始 capture/prediction 与 manifest 的 SHA-256。
- `replay_probabilities/`：板端 PL capture 重放概率和 ANN 基线概率，用于复核 PR/AP。
- `pr_curve.png`、`threshold_metrics.png`：图形化结果。
- `evaluate_board_results.py`：复现脚本。

## 不能从当前归档测量的指标

本次归档没有 `runtime.json`、板端单调时钟采样、功耗采样或外部仪表记录，因此 B01 内核延迟、B02 端到端延迟、B03 吞吐/实时性、B04 运行功耗、B05 单任务能耗均标为未测。文件修改时间不具备明确测量边界，未被用于推算性能。

B07 面积/资源来自与本次 exact bitstream 对应的 Vivado implemented-design 报告：LUT {implementation['primary_metrics']['clb_luts']:,}，寄存器 {implementation['primary_metrics']['clb_registers']:,}，BRAM {implementation['primary_metrics']['block_ram_tiles']}，DSP {implementation['primary_metrics']['dsps']}。其中 {implementation['primary_metrics']['vectorless_total_on_chip_power_w']} W 是 Vivado vectorless 估计，不是板级实测功耗。

## 口径说明

工作点 Event IoU/Precision/Recall/F1/Pd/Fa 直接读取板端二值 prediction；PR/AP 使用与板端 PL 输出逐字一致的 capture，在 PC 上执行冻结 fixed H2 重放。密度分层阈值为 {DENSITY_THRESHOLD_EPS:,.0f} events/s。轨迹采用 `evuav-observable-event-track-hota-v1` 冻结适配器（3 px、20 px/50 ms、最大缺失 20 窗）。ANN→SNN 差值同时包含 Proposal 转换和定点部署影响，不能拆成独立量化损失或单独转换损失。
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "output": str(OUTPUT),
        "event_iou": event["iou"],
        "precision": event["precision"],
        "recall": event["recall"],
        "f1": event["f1"],
        "pd": localization["pd"],
        "fa": localization["fa"],
        "average_precision": board_pr["average_precision"],
        "HOTA": board_hota_avg["HOTA"],
        "plots": plots,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
