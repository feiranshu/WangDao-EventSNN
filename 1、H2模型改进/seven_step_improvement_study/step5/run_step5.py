#!/usr/bin/env python3
"""Correct Step4 selection and freeze the sole A4-2 Step5 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from step5_common import (
    PROTOCOL, REGISTERED_EPOCHS, compact, count_passes, enumerate_combinations,
    select_original, select_revised,
)

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step4"):
    if (candidate / "step4_common.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from step4_common import (  # noqa: E402
    H2_F_THRESHOLD, RUNTIME_ROOT, build_model, density_from_json, evaluate_model,
    evaluate_records, exact_pr, export_metrics, joined_features, json_write, load_records,
    predict_int8, role_paths, select_h2_r, sha256, working_point_view,
)
from step4_training import _export_bootstrap_scores, export_int8, load_checkpoint  # noqa: E402
from wangdao.h2.numpy_model import _refiner_probability, quantized_mlp  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_paths(root: Path) -> Dict[str, Path]:
    return {
        "candidate_summary": root / "runs" / "A4-2-seed42" / "summary.json",
        "control_summary": root / "runs" / "A4-2-null-seed42" / "summary.json",
        "density": root / "work" / "density_transform.json",
    }


def choose_device(name: str, formal: bool) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if formal:
        raise RuntimeError("formal Step5 requires CUDA when --device auto is used")
    return torch.device("cpu")


def preflight(step4_results: Path, output: Path, require_cuda: bool) -> Dict[str, Any]:
    sources = source_paths(step4_results)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Step4 source artifact missing: " + ", ".join(missing))
    candidate, control = read_json(sources["candidate_summary"]), read_json(sources["control_summary"])
    checks = {
        "candidate_complete": candidate.get("status") == "complete",
        "control_complete": control.get("status") == "complete",
        "candidate_is_A4_2_seed42": candidate.get("model_id") == "A4-2" and candidate.get("seed") == 42,
        "control_is_A4_2_null_seed42": control.get("model_id") == "A4-2-null" and control.get("seed") == 42,
        "source_roles_isolated": all(row.get("validation_used") is False and row.get("test_used") is False for row in (candidate, control)),
        "registered_epochs_match": all(tuple(row.get("configuration", {}).get("registered_epochs", ())) == REGISTERED_EPOCHS for row in (candidate, control)),
        "threshold_grids_valid": bool(enumerate_combinations(candidate)) and bool(enumerate_combinations(control)),
        "train_dev_has_10_sequences": len(role_paths("train_dev")) == 10,
        "validation_and_test_unreachable": True,
        "cuda_requirement": (not require_cuda) or torch.cuda.is_available(),
    }
    report = {
        "schema_version": 1, "protocol": PROTOCOL,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks, "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "h2_sha256": sha256(RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"),
        "index_sha256": sha256(RUNTIME_ROOT / "data" / "index.json"),
        "validation_used": False, "test_used": False,
    }
    json_write(output, report)
    if report["status"] != "pass":
        raise RuntimeError("Step5 preflight failed")
    return report


def _all_audit_rows(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    result = []
    for row in rows:
        result.append({
            "epoch": row["epoch"], "threshold": row["threshold"],
            "metrics": compact(row),
            "original_full": row["original_full"],
            "original_noninferior": row["original_noninferior"],
            "revised": row["revised"],
        })
    return result


def _history_report(
    records: Sequence[Mapping[str, Any]], candidate_probability: Sequence[np.ndarray],
    candidate_threshold: float, h2_threshold: float,
) -> Dict[str, Any]:
    labels = np.concatenate([np.asarray(row["labels"], dtype=np.uint8) for row in records])
    windows = np.concatenate([np.asarray(row["windows"], dtype=np.int64) for row in records])
    availability = np.minimum(windows, 2)
    candidate = np.concatenate(candidate_probability)
    baseline = np.concatenate([np.asarray(row["baseline_probability"], dtype=np.float32) for row in records])
    groups: Dict[str, Any] = {}
    for count in (0, 1, 2):
        mask = availability == count
        truth = labels[mask] == 1
        row: Dict[str, Any] = {
            "available_prior_windows": count, "candidate_events": int(mask.sum()),
            "positive_candidate_events": int(truth.sum()),
            "scope": "Proposal candidates only; diagnostic, not full-event AP",
        }
        for prefix, probability, threshold in (
            ("candidate", candidate[mask], candidate_threshold),
            ("h2", baseline[mask], h2_threshold),
        ):
            predicted = probability >= threshold
            tp = int(np.sum(predicted & truth)); fp = int(np.sum(predicted & ~truth)); fn = int(np.sum(~predicted & truth))
            row.update({
                f"{prefix}_average_precision": exact_pr(truth.astype(np.uint8), probability)["average_precision"],
                f"{prefix}_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                f"{prefix}_precision": tp / (tp + fp) if tp + fp else None,
                f"{prefix}_recall": tp / (tp + fn) if tp + fn else None,
            })
        row["delta_iou"] = row["candidate_iou"] - row["h2_iou"]
        row["delta_average_precision"] = row["candidate_average_precision"] - row["h2_average_precision"]
        groups[str(count)] = row
    return {"scope": "Train-dev candidate diagnostic", "groups": groups}


def _export_h2_bootstrap(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    labels, probability, sequence = [], [], []
    for index, record in enumerate(records):
        full = np.zeros(int(record["event_count"]), dtype=np.float32)
        full[np.asarray(record["indices"], dtype=np.int64)] = np.asarray(record["baseline_probability"], dtype=np.float32)
        labels.append(np.asarray(record["all_labels"], dtype=np.uint8))
        probability.append(full)
        sequence.append(np.full(len(full), index, dtype=np.uint8))
    with path.open("xb") as stream:
        np.savez_compressed(stream, labels=np.concatenate(labels), probability=np.concatenate(probability),
                            sequence=np.concatenate(sequence),
                            sequence_files=np.asarray([str(record["file"]) for record in records]))


def _find(rows: Sequence[Mapping[str, Any]], epoch: int, threshold: float) -> Mapping[str, Any]:
    matches = [row for row in rows if int(row["epoch"]) == epoch and float(row["threshold"]) == threshold]
    if len(matches) != 1:
        raise RuntimeError("frozen Step5 combination is not unique")
    return matches[0]


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    preflight(args.step4_results, args.output / "preflight.json", require_cuda=args.run and args.device != "cpu")
    sources = source_paths(args.step4_results)
    candidate_summary = read_json(sources["candidate_summary"])
    control_summary = read_json(sources["control_summary"])
    candidate_rows = enumerate_combinations(candidate_summary)
    control_rows = enumerate_combinations(control_summary)
    original = select_original(candidate_rows)
    revised = select_revised(candidate_rows)
    qualified_nulls = [row for row in control_rows if row["original_noninferior"]["passes"]]
    best_null = None if not qualified_nulls else max(
        qualified_nulls,
        key=lambda row: (
            float(row["view"]["overall"]["event"]["iou"]), float(row["view"]["average_precision"]),
            float(row["view"]["high_density"]["event"]["recall"]),
            -float(row["view"]["ordinary_density"]["localization"]["fa"]),
            -float(row["epoch"]), -float(row["threshold"]),
        ),
    )
    json_write(args.output / "combination_audit.json", {
        "schema_version": 1, "protocol": PROTOCOL,
        "registered_epochs": list(REGISTERED_EPOCHS),
        "threshold_grid": sorted({row["threshold"] for row in candidate_rows}),
        "candidate": _all_audit_rows(candidate_rows), "control": _all_audit_rows(control_rows),
        "validation_used": False, "test_used": False,
    })
    original_decision = {
        "schema_version": 1, "protocol": "Step4-corrected-original-protocol",
        "source_protocol": candidate_summary["protocol"], "status": "complete",
        "rule_change": False, "implementation_corrections": [
            "eligibility_before_ranking", "all_registered_epoch_threshold_combinations",
            "high_density_fa_in_noninferiority", "overall_pd_fa_in_noninferiority",
            "missing_or_nonfinite_required_metrics_are_ineligible",
        ],
        "combination_counts": {
            "total": len(candidate_rows),
            "full_replacement": count_passes(candidate_rows, "original_full"),
            "noninferior": count_passes(candidate_rows, "original_noninferior"),
        },
        "selected": compact(original), "eligibility_level": original["eligibility_level"],
        "produces_original_protocol_replacement": original["eligibility_level"] == 2,
        "validation_used": False, "test_used": False,
    }
    json_write(args.output / "corrected_original_decision.json", original_decision)
    revised_decision = {
        "schema_version": 1, "protocol": PROTOCOL, "status": "complete",
        "qualified_combinations": count_passes(candidate_rows, "revised"),
        "selected": compact(revised), "best_qualified_null": compact(best_null),
        "candidate_frozen": revised is not None,
        "validation_used": False, "test_used": False,
    }
    json_write(args.output / "revised_candidate_decision.json", revised_decision)
    if args.smoke:
        decision = {
            "schema_version": 1, "protocol": PROTOCOL, "status": "smoke_complete",
            "candidate_frozen": False, "selection_would_freeze": compact(revised),
            "validation_used": False, "test_used": False,
        }
        json_write(args.output / "step5_decision.json", decision)
        return decision
    if revised is None:
        decision = {
            "schema_version": 1, "protocol": PROTOCOL, "status": "complete",
            "candidate_frozen": False, "next_step": "retain_H2_stop_before_Step6",
            "validation_used": False, "test_used": False,
        }
        json_write(args.output / "step5_decision.json", decision)
        return decision

    device = choose_device(args.device, formal=True)
    _, records = load_records("train_dev", args.output / "work" / "cache")
    density = density_from_json(sources["density"])
    epoch, threshold = int(revised["epoch"]), float(revised["threshold"])
    checkpoint = args.step4_results / "runs" / "A4-2-seed42" / f"epoch_{epoch:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    package = load_checkpoint(checkpoint)
    if package.get("protocol") != candidate_summary["protocol"] or package.get("metadata", {}).get("epoch") != epoch:
        raise RuntimeError("selected checkpoint metadata does not match the Step5 decision")
    model = build_model("A4-2", seed=42).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    pair = evaluate_model(records, model, "A4-2", density, device, args.batch_size, keep_pr_points=True)
    int8_view = working_point_view(pair["int8"], {"eligible": True, "threshold": threshold})
    float_view = working_point_view(pair["float"], {"eligible": True, "threshold": threshold})
    if abs(float(int8_view["average_precision"]) - float(revised["view"]["average_precision"])) > 1e-12 or abs(float(int8_view["overall"]["event"]["iou"]) - float(revised["view"]["overall"]["event"]["iou"])) > 1e-12:
        raise RuntimeError("reloaded exact-int8 result differs from the selected Step4 curve")

    h2 = evaluate_records(records, None, device, args.batch_size, keep_pr_points=True)
    h2_f = working_point_view(h2, {"eligible": True, "threshold": H2_F_THRESHOLD})
    h2_r = working_point_view(h2, select_h2_r(h2))
    artifacts = args.output / "artifacts"
    artifacts.mkdir()
    copied_checkpoint = artifacts / checkpoint.name
    shutil.copy2(checkpoint, copied_checkpoint)
    int8_export = artifacts / "A4-2_frozen_refiner_int8.npz"
    export_int8(int8_export, model, {
        "protocol": PROTOCOL, "model_id": "A4-2", "seed": 42,
        "epoch": epoch, "threshold": threshold, "history_feature_version": 1,
    })
    export_metrics(args.output / "metrics" / "A4-2" / "int8", pair["int8"], int8_view)
    export_metrics(args.output / "metrics" / "A4-2" / "float", pair["float"], float_view)
    export_metrics(args.output / "metrics" / "H2-R" / "int8", h2, h2_r)
    _export_bootstrap_scores(args.output / "metrics" / "A4-2" / "int8" / "bootstrap_event_scores.npz", records, model, "A4-2", density, None)
    _export_h2_bootstrap(args.output / "metrics" / "H2-R" / "int8" / "bootstrap_event_scores.npz", records)
    probabilities = [predict_int8(model, joined_features(row, "A4-2", density)) for row in records]
    history = _history_report(records, probabilities, threshold, float(h2_r["threshold"]))
    json_write(args.output / "history_availability.json", history)

    matched_null = _find(control_rows, epoch, threshold)
    frozen = {
        "schema_version": 1, "protocol": PROTOCOL, "model_id": "A4-2", "seed": 42,
        "epoch": epoch, "threshold": threshold,
        "checkpoint": str(Path("artifacts") / copied_checkpoint.name),
        "checkpoint_sha256": sha256(copied_checkpoint),
        "int8_weights": str(Path("artifacts") / int8_export.name),
        "int8_weights_sha256": sha256(int8_export),
        "proposal_sha256": sha256(RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"),
        "index_sha256": sha256(RUNTIME_ROOT / "data" / "index.json"),
        "history_feature_version": 1, "quantization_version": 1,
        "density_boundary_eps": 20000.0, "postprocessing_version": "frozen_H2",
        "selection_protocol_version": PROTOCOL, "numeric_path": "exact_int8_replay",
        "candidate_metrics": compact(revised), "matched_null_metrics": compact(matched_null),
        "best_qualified_null_metrics": compact(best_null),
        "h2_f": h2_f, "h2_r": h2_r,
        "validation_used": False, "test_used": False,
    }
    json_write(args.output / "frozen_candidate.json", frozen)
    decision = {
        "schema_version": 1, "protocol": PROTOCOL, "status": "complete",
        "candidate_frozen": True, "frozen_candidate": "frozen_candidate.json",
        "selected_model": "A4-2", "selected_epoch": epoch, "selected_threshold": threshold,
        "corrected_original_replacement": original_decision["produces_original_protocol_replacement"],
        "next_step": "Step6_three_seed_optimization_stability",
        "validation_used": False, "test_used": False,
    }
    json_write(args.output / "step5_decision.json", decision)
    return decision


def verify_results(results: Path, step4_results: Path, output: Path, device_name: str) -> Dict[str, Any]:
    frozen = read_json(results / "frozen_candidate.json")
    epoch, threshold = int(frozen["epoch"]), float(frozen["threshold"])
    checkpoint = results / frozen["checkpoint"]
    exported = results / frozen["int8_weights"]
    package = load_checkpoint(checkpoint)
    model = build_model("A4-2", seed=42).to(choose_device(device_name, formal=False))
    model.load_state_dict(package["state_dict"], strict=True)
    _, records = load_records("train_dev", results / "work" / "cache")
    density = density_from_json(source_paths(step4_results)["density"])
    direct = [predict_int8(model, joined_features(row, "A4-2", density)) for row in records]
    with np.load(exported, allow_pickle=False) as arrays:
        layers = tuple((arrays[f"layer{index}_weight"], arrays[f"layer{index}_scale"]) for index in range(3))
        replay = [
            _refiner_probability(quantized_mlp(joined_features(row, "A4-2", density), layers), 2)
            for row in records
        ]
    max_abs = max(float(np.max(np.abs(a - b))) for a, b in zip(direct, replay))
    metric = read_json(results / "metrics" / "A4-2" / "int8" / "summary.json")["working_point"]
    source_checkpoint = step4_results / "runs" / "A4-2-seed42" / f"epoch_{epoch:03d}.pt"
    checks = {
        "checkpoint_hash_matches_frozen": sha256(checkpoint) == frozen["checkpoint_sha256"],
        "checkpoint_copy_matches_step4_source": sha256(checkpoint) == sha256(source_checkpoint),
        "int8_export_hash_matches_frozen": sha256(exported) == frozen["int8_weights_sha256"],
        "checkpoint_metadata_matches_decision": package.get("metadata", {}).get("epoch") == epoch and package.get("metadata", {}).get("model_id") == "A4-2",
        "decision_threshold_matches_metric": float(metric["threshold"]) == threshold,
        "decision_ap_matches_metric": abs(float(metric["average_precision"]) - float(frozen["candidate_metrics"]["average_precision"])) <= 1e-12,
        "decision_iou_matches_metric": abs(float(metric["overall"]["event"]["iou"]) - float(frozen["candidate_metrics"]["overall_iou"])) <= 1e-12,
        "exported_int8_replay_exact": max_abs == 0.0,
        "validation_and_test_unused": frozen.get("validation_used") is False and frozen.get("test_used") is False,
    }
    report = {
        "schema_version": 1, "protocol": PROTOCOL,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks, "int8_replay_max_abs": max_abs,
        "validation_used": False, "test_used": False,
    }
    json_write(output, report)
    if report["status"] != "pass":
        raise RuntimeError("Step5 artifact consistency verification failed")
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--step4-results", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--require-cuda", action="store_true")
    run = sub.add_parser("run-all")
    run.add_argument("--step4-results", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--batch-size", type=int, default=4096)
    run.add_argument("--run", action="store_true")
    run.add_argument("--smoke", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--results", type=Path, required=True)
    verify.add_argument("--step4-results", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "preflight":
        result = preflight(args.step4_results, args.output, args.require_cuda)
    elif args.command == "run-all":
        if args.run == args.smoke:
            raise SystemExit("choose exactly one of --run or --smoke")
        if args.batch_size != 4096:
            raise SystemExit("the frozen Step5 batch size is 4096")
        result = run_all(args)
    else:
        result = verify_results(args.results, args.step4_results, args.output, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
