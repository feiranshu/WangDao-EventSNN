"""Freeze and evaluate H2-family candidates without development-role leakage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .h2.torch_model import QuantizedMLP
from .h2_research import FrozenH2Pipeline, augment_refiner_features
from .h2_students import load_student_artifact
from .phase1_common import PROTOCOL_ID, ROOT, aggregate_scores, load_sequence, score_predictions, sha256, write_json
from .h2_refiner import full_probability, predict_refiner


REQUIRED_SEEDS = (42, 43, 44)


def candidate_features(data: Dict[str, np.ndarray], feature_variant: str) -> np.ndarray:
    if feature_variant == "A3":
        return augment_refiner_features(data["features"], data["history"])
    if feature_variant == "A3-null":
        return augment_refiner_features(data["features"], np.zeros_like(data["history"]))
    if feature_variant not in ("A1", "A2"):
        raise ValueError("unknown H2 feature variant")
    return data["features"]


def _dev_names_and_counts() -> Tuple[List[str], Dict[str, int], str]:
    index_path = ROOT / "data/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = list(index["roles"]["train_dev"])
    records = {record["path"]: int(record["events"]) for record in index["files"]}
    return names, {name: records["train/" + name] for name in names}, sha256(index_path)


def _validate_common_summary(path: Path, report: Dict[str, Any]) -> None:
    if report.get("status") != "complete" or report.get("smoke", True):
        raise ValueError(f"{path}: expected complete non-smoke summary")
    if report.get("data_role") != "train_main+train_dev" or report.get("test_used") or report.get("validation_used"):
        raise ValueError(f"{path}: candidate used a protected role")
    if report.get("metric_protocol") != PROTOCOL_ID:
        raise ValueError(f"{path}: metric protocol mismatch")
    names, counts, index_hash = _dev_names_and_counts()
    provenance_hash = report.get("provenance", {}).get("index_sha256")
    if provenance_hash != index_hash:
        raise ValueError(f"{path}: data index mismatch")
    evaluation = report.get("final_evaluation", {})
    rows = evaluation.get("per_sequence", [])
    actual = [row.get("file") for row in rows]
    if evaluation.get("partial", True) or evaluation.get("sequence_count") != len(names) or actual != names:
        raise ValueError(f"{path}: full ordered Train-dev evidence is required")
    if any(int(row.get("events", -1)) != counts[row["file"]] for row in rows):
        raise ValueError(f"{path}: Train-dev event counts changed")


def validate_accuracy_summaries(paths: Sequence[Path]) -> Dict[str, Any]:
    if len(paths) != len(REQUIRED_SEEDS):
        raise ValueError("A* freeze requires exactly seed42/43/44 summaries")
    rows = []
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate_common_summary(path, report)
        selected = report.get("candidate_fair_working_point", {})
        if not selected.get("eligible") or selected.get("metrics") is None:
            raise ValueError(f"{path}: candidate has no eligible working point")
        checkpoint = path.parent / report["best_checkpoint"]
        if sha256(checkpoint) != report.get("best_checkpoint_sha256"):
            raise ValueError(f"{path}: checkpoint hash mismatch")
        rows.append((path, report, checkpoint))
    variants = {row[1]["recipe"]["variant"] for row in rows}
    seeds = {int(row[1]["seed"]) for row in rows}
    recipes = {row[1]["recipe_id"] for row in rows}
    if len(variants) != 1 or seeds != set(REQUIRED_SEEDS) or len(recipes) != 1:
        raise ValueError("A* summaries must share one recipe and cover seeds 42/43/44")
    ordered = sorted(rows, key=lambda row: int(row[1]["seed"]))
    deltas = [float(row[1]["selected_iou_delta_vs_h2_r"]) for row in ordered]
    if deltas[0] < 0.005 or float(np.mean(deltas)) <= 0 or sum(value > 0 for value in deltas) < 2:
        raise ValueError("A* does not pass the preregistered development gate")
    primary_path, primary, checkpoint = ordered[0]
    return {
        "variant": primary["recipe"]["variant"],
        "seeds": list(REQUIRED_SEEDS),
        "delta_iou_vs_h2_r": deltas,
        "mean_delta_iou_vs_h2_r": float(np.mean(deltas)),
        "std_delta_iou_vs_h2_r": float(np.std(deltas)),
        "primary_summary": str(primary_path.resolve()),
        "primary_summary_sha256": sha256(primary_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": float(primary["candidate_fair_working_point"]["threshold"]),
        "h2_r_threshold": float(primary["h2_fair_working_point"]["threshold"]),
        "primary_metrics": primary["candidate_fair_working_point"]["metrics"],
    }


def validate_student_summaries(paths: Sequence[Path], accuracy: Dict[str, Any]) -> Dict[str, Any]:
    if len(paths) != len(REQUIRED_SEEDS):
        raise ValueError("S* freeze requires exactly seed42/43/44 summaries")
    rows = []
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate_common_summary(path, report)
        selected = report.get("fair_working_points", {}).get("student", {})
        if not selected.get("eligible") or selected.get("metrics") is None:
            raise ValueError(f"{path}: student has no eligible working point")
        checkpoint = path.parent / report["best_checkpoint"]
        artifact = path.parent / report["artifact"]
        if sha256(checkpoint) != report.get("best_checkpoint_sha256") or sha256(artifact) != report.get("artifact_sha256"):
            raise ValueError(f"{path}: student checkpoint/artifact hash mismatch")
        rows.append((path, report, checkpoint, artifact))
    signatures = {(row[1]["recipe"]["student_variant"], row[1]["recipe"]["mode"],
                   row[1]["recipe"]["feature_variant"], row[1]["recipe_id"]) for row in rows}
    seeds = {int(row[1]["seed"]) for row in rows}
    if len(signatures) != 1 or seeds != set(REQUIRED_SEEDS):
        raise ValueError("S* summaries must share one recipe and cover seeds 42/43/44")
    ordered = sorted(rows, key=lambda row: int(row[1]["seed"]))
    vs_h2 = [float(row[1]["selected_iou_delta_vs_h2_r"]) for row in ordered]
    vs_teacher = [float(row[1]["selected_iou_delta_vs_teacher"]) for row in ordered]
    if vs_h2[0] <= 0 or vs_teacher[0] < -0.003 or float(np.mean(vs_h2)) <= 0:
        raise ValueError("S* does not remain above H2 and within 0.3pp of A*")
    primary_path, primary, checkpoint, artifact = ordered[0]
    if primary["recipe"]["feature_variant"] != accuracy["variant"]:
        raise ValueError("S* information range does not match A*")
    return {
        "variant": primary["recipe"]["student_variant"],
        "mode": primary["recipe"]["mode"],
        "feature_variant": primary["recipe"]["feature_variant"],
        "seeds": list(REQUIRED_SEEDS),
        "delta_iou_vs_h2_r": vs_h2,
        "delta_iou_vs_teacher": vs_teacher,
        "primary_summary": str(primary_path.resolve()),
        "primary_summary_sha256": sha256(primary_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256(artifact),
        "threshold": float(primary["fair_working_points"]["student"]["threshold"]),
        "primary_metrics": primary["fair_working_points"]["student"]["metrics"],
        "complexity": primary["complexity"],
    }


def freeze_manifest(accuracy_paths: Sequence[Path], output: Path,
                    student_paths: Optional[Sequence[Path]] = None) -> Dict[str, Any]:
    output = Path(output)
    accuracy = validate_accuracy_summaries(accuracy_paths)
    student = validate_student_summaries(student_paths, accuracy) if student_paths else None
    h2_path = ROOT / "software/weights/H2_D1_seed42.npz"
    pipeline = FrozenH2Pipeline(h2_path)
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": output.stem,
        "stage": "h2_family_release",
        "data_role": "train_dev",
        "primary_metrics": accuracy["primary_metrics"],
        "decision": "ready_for_single_validation_then_test",
        "conclusion": "Model identities and thresholds are frozen; no Validation/Test result exists yet.",
        "metric_protocol": PROTOCOL_ID,
        "data_index_sha256": sha256(ROOT / "data/index.json"),
        "test_used": False,
        "validation_used": False,
        "post_historical_test_development": True,
        "models": {
            "H2-F": {"kind": "frozen_h2", "weights": str(h2_path.resolve()),
                     "weights_sha256": sha256(h2_path), "threshold": pipeline.h2.event_threshold,
                     "layers": 6, "matrix_weights": 11904},
            "H2-R": {"kind": "frozen_h2", "weights": str(h2_path.resolve()),
                     "weights_sha256": sha256(h2_path), "threshold": accuracy["h2_r_threshold"],
                     "layers": 6, "matrix_weights": 11904},
            "A*": dict(accuracy, kind="adapted_refiner"),
        },
        "primary_comparison": "A* minus H2-R",
        "secondary_comparisons": [] if student is None else ["S* minus H2-R", "S* minus A*"],
        "failure_policy": "Report without retuning. Protocol defects require a new invalidated release version.",
        "independent_generalization": "pending_new_sealed_dataset",
    }
    if student is not None:
        manifest["models"]["S*"] = dict(student, kind="shallow_refiner")
    write_json(output, manifest)
    return manifest


def protected_role_paths(role: str) -> List[Path]:
    if role not in ("validation", "test"):
        raise ValueError("protected evaluation role must be validation or test")
    index_path = ROOT / "data/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = list(index["roles"][role])
    prefix = "val" if role == "validation" else "test"
    expected_prefix = "val_" if role == "validation" else "test_"
    records = {record["path"]: record for record in index["files"]}
    paths = []
    for name in names:
        if Path(name).name != name or not name.startswith(expected_prefix) or not name.endswith(".npz"):
            raise ValueError("invalid protected-role filename")
        path = ROOT / "data/EV-UAV-dataset" / prefix / name
        if sha256(path) != records[prefix + "/" + name]["sha256"]:
            raise ValueError("protected-role data hash mismatch: " + name)
        paths.append(path)
    if len(paths) != 24 or len(set(paths)) != 24:
        raise ValueError("protected evaluation requires exactly 24 unique sequences")
    return paths


def _load_adapted(checkpoint: Path, feature_variant: str, device: torch.device) -> QuantizedMLP:
    package = torch.load(checkpoint, map_location=device, weights_only=True)
    dimensions = tuple(int(value) for value in package["dimensions"])
    model = QuantizedMLP(dimensions, quantized=False).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    if package.get("variant") != feature_variant or package.get("metadata", {}).get("smoke", True):
        raise ValueError("frozen adapted Refiner checkpoint metadata mismatch")
    return model.eval()


def paired_iou_interval(per_sequence: Sequence[Dict[str, Any]], left: str, right: str,
                        seed: int = 20260916, repeats: int = 10000) -> Dict[str, Any]:
    if repeats < 100:
        raise ValueError("bootstrap repeats must be at least 100")
    rows = list(per_sequence)
    if not rows:
        raise ValueError("paired interval requires sequence rows")
    counts = {}
    for model in (left, right):
        counts[model] = np.asarray([[row["models"][model]["event"][key]
                                    for key in ("tp", "fp", "fn")] for row in rows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    observed = []
    for model in (left, right):
        tp, fp, fn = counts[model].sum(axis=0)
        denominator = tp + fp + fn
        observed.append(float(tp / denominator) if denominator else np.nan)
    for index in range(repeats):
        chosen = rng.integers(0, len(rows), size=len(rows))
        values = []
        for model in (left, right):
            tp, fp, fn = counts[model][chosen].sum(axis=0)
            denominator = tp + fp + fn
            values.append(float(tp / denominator) if denominator else np.nan)
        samples[index] = values[0] - values[1]
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0 or not np.isfinite(observed).all():
        raise ValueError("paired interval has no defined IoU difference")
    return {"left": left, "right": right, "unit": "sequence bootstrap IoU difference",
            "seed": seed, "repeats": repeats, "estimate": observed[0] - observed[1],
            "lower_95": float(np.quantile(finite, 0.025)),
            "upper_95": float(np.quantile(finite, 0.975)),
            "limitation": "Sequences approximate acquisition groups; correlated events are not independent samples."}


def evaluate_frozen(manifest_path: Path, role: str, output: Path, device: torch.device,
                    validation_result: Optional[Path] = None,
                    batch_size: int = 4096) -> Dict[str, Any]:
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("metric_protocol") != PROTOCOL_ID:
        raise ValueError("expected a frozen H2-family manifest")
    if sha256(ROOT / "data/index.json") != manifest.get("data_index_sha256"):
        raise ValueError("dataset index changed after freeze")
    if role == "test":
        if validation_result is None:
            raise ValueError("Test requires the one completed Validation result")
        validation = json.loads(Path(validation_result).read_text(encoding="utf-8"))
        if (validation.get("status") != "complete" or validation.get("data_role") != "validation"
                or validation.get("manifest_sha256") != sha256(manifest_path)):
            raise ValueError("Validation evidence does not match this manifest")
    elif role != "validation":
        raise ValueError("role must be validation or test")
    if output.exists():
        raise FileExistsError(output)
    claim = manifest_path.with_name(manifest_path.stem + f"_{role}_started.json")
    write_json(claim, {"manifest_sha256": sha256(manifest_path), "role": role,
                       "output": str(output), "status": "started",
                       "note": "Do not silently rerun after interruption."})
    paths = protected_role_paths(role)
    h2_path = Path(manifest["models"]["H2-F"]["weights"])
    if sha256(h2_path) != manifest["models"]["H2-F"]["weights_sha256"]:
        raise ValueError("frozen H2 weights changed")
    pipeline = FrozenH2Pipeline(h2_path, batch_size)
    networks: Dict[str, Tuple[Any, str]] = {}
    accuracy = manifest["models"]["A*"]
    accuracy_path = Path(accuracy["checkpoint"])
    if sha256(accuracy_path) != accuracy["checkpoint_sha256"]:
        raise ValueError("A* checkpoint changed")
    networks["A*"] = (_load_adapted(accuracy_path, accuracy["variant"], device), accuracy["variant"])
    if "S*" in manifest["models"]:
        student = manifest["models"]["S*"]
        artifact_path = Path(student["artifact"])
        if sha256(artifact_path) != student["artifact_sha256"]:
            raise ValueError("S* artifact changed")
        artifact = load_student_artifact(artifact_path)
        networks["S*"] = (artifact.model.to(device).eval(), student["feature_variant"])
    model_names = list(manifest["models"])
    aggregate: Dict[str, List[Dict[str, Any]]] = {name: [] for name in model_names}
    per_sequence = []
    prediction_root = output.with_name(output.stem + "_predictions")
    prediction_root.mkdir(parents=True, exist_ok=False)
    for path in paths:
        events, loc, labels, identities = load_sequence(path)
        prepared = pipeline.prepare(events[["x", "y", "t", "p"]].copy())
        baseline = pipeline.infer_prepared(prepared, len(events))["probabilities"]
        probabilities = {"H2-F": baseline, "H2-R": baseline}
        cache_data = {"features": prepared.features, "history": prepared.history}
        for name, (network, feature_variant) in networks.items():
            candidate = predict_refiner(network, candidate_features(cache_data, feature_variant),
                                        batch_size, str(device))
            probabilities[name] = full_probability(prepared.indices, candidate, len(events))
        row = {"file": path.name, "events": len(events), "models": {}}
        for name in model_names:
            threshold = float(manifest["models"][name]["threshold"])
            score = score_predictions(loc, labels, identities, probabilities[name], threshold)
            aggregate[name].append(score)
            row["models"][name] = score
            destination = prediction_root / name.replace("*", "star")
            destination.mkdir(exist_ok=True)
            with (destination / path.name).open("xb") as stream:
                np.savez_compressed(stream, probabilities=probabilities[name],
                                    event_index=np.arange(len(events), dtype=np.int64),
                                    source_sha256=np.asarray(sha256(path)), threshold=np.asarray(threshold))
        per_sequence.append(row)
    metrics = {name: aggregate_scores(rows) for name, rows in aggregate.items()}
    intervals = {"A*_minus_H2-R": paired_iou_interval(per_sequence, "A*", "H2-R")}
    if "S*" in model_names:
        intervals["S*_minus_H2-R"] = paired_iou_interval(per_sequence, "S*", "H2-R")
        intervals["S*_minus_A*"] = paired_iou_interval(per_sequence, "S*", "A*")
    report = {
        "schema_version": 1, "status": "complete", "experiment_id": output.stem,
        "stage": "h2_family_frozen_evaluation", "data_role": role,
        "primary_metrics": metrics["A*"], "decision": "report_without_retuning",
        "conclusion": "Frozen one-pass evaluation; thresholds and model identities were not searched.",
        "manifest_sha256": sha256(manifest_path), "test_used": role == "test",
        "validation_used": role in ("validation", "test"), "models": metrics,
        "paired_intervals": intervals, "per_sequence": per_sequence,
        "prediction_directory": str(prediction_root.resolve()),
        "independent_generalization": "pending_new_sealed_dataset",
    }
    write_json(output, report)
    return report
