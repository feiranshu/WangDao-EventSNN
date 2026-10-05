#!/usr/bin/env python3
"""Run the frozen Validation/Test acceptance of A4-2 without retraining."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from step7_common import (
    BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, FIXED_EPOCH, FIXED_MODEL, FIXED_SEED,
    FIXED_THRESHOLD, H2_F_THRESHOLD, H2_R_THRESHOLD, PROTOCOL, ROLES,
    acceptance_gates, acceptance_role_paths, compact_view, comparison_svg,
    csv_write, delta_report, export_score_archive, json_write,
    load_acceptance_records, paired_bootstrap, per_sequence_rows, read_json,
)


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step4"):
    try:
        if (candidate / "step4_common.py").is_file():
            sys.path.insert(0, str(candidate))
            break
    except (OSError, PermissionError):
        continue
else:  # pragma: no cover
    raise RuntimeError("Cannot locate bundled step4_common.py")

from step3_common import evaluate_records, export_metrics, remove_private, sha256, svg_curve, working_point_view  # noqa: E402
from step4_common import (  # noqa: E402
    RUNTIME_ROOT, Step4Refiner, density_from_json, joined_features, predict_float,
    predict_int8,
)
from step4_training import load_checkpoint  # noqa: E402


def sources(step5_results: Path, step6_results: Path, step4_results: Path) -> Dict[str, Path]:
    return {
        "frozen_candidate": step5_results / "frozen_candidate.json",
        "candidate_checkpoint": step5_results / "artifacts" / "epoch_008.pt",
        "candidate_int8": step5_results / "artifacts" / "A4-2_frozen_refiner_int8.npz",
        "density_transform": step4_results / "work" / "density_transform.json",
        "step6_decision": step6_results / "step6_decision.json",
        "step6_protocol": step6_results / "step6_protocol.json",
        "h2_weights": RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz",
        "index": RUNTIME_ROOT / "data" / "index.json",
        "step7_protocol": HERE / "step7_protocol.json",
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
        raise RuntimeError("formal Step7 requires CUDA when --device auto is used")
    return torch.device("cpu")


def _index_role_check(index: Mapping[str, Any], role: str) -> bool:
    names = index.get("roles", {}).get(role, [])
    prefix = "val_" if role == "validation" else "test_"
    paths = {row.get("path") for row in index.get("files", [])}
    directory = "val" if role == "validation" else "test"
    return (
        len(names) == 24
        and len(set(names)) == 24
        and all(Path(name).name == name and name.startswith(prefix) and name.endswith(".npz") for name in names)
        and all(f"{directory}/{name}" in paths for name in names)
    )


def preflight(
    step5_results: Path, step6_results: Path, step4_results: Path,
    output: Path, require_cuda: bool,
) -> Dict[str, Any]:
    paths = sources(step5_results, step6_results, step4_results)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Step7 source artifact missing: " + ", ".join(missing))
    frozen = read_json(paths["frozen_candidate"])
    decision = read_json(paths["step6_decision"])
    protocol = read_json(paths["step7_protocol"])
    index = read_json(paths["index"])
    checks = {
        "step7_protocol_frozen": protocol.get("protocol") == PROTOCOL and protocol.get("frozen_before_validation_or_test_access") is True,
        "candidate_identity_exact": frozen.get("model_id") == FIXED_MODEL and frozen.get("seed") == FIXED_SEED and frozen.get("epoch") == FIXED_EPOCH and float(frozen.get("threshold")) == FIXED_THRESHOLD,
        "checkpoint_hash_matches_step5": sha256(paths["candidate_checkpoint"]) == frozen.get("checkpoint_sha256") == protocol.get("candidate", {}).get("checkpoint_sha256"),
        "int8_export_hash_matches_step5": sha256(paths["candidate_int8"]) == frozen.get("int8_weights_sha256"),
        "step6_complete_and_same_candidate": decision.get("status") == "complete" and decision.get("fixed_model") == FIXED_MODEL and decision.get("deployment_seed") == FIXED_SEED and decision.get("fixed_epoch") == FIXED_EPOCH and float(decision.get("fixed_threshold")) == FIXED_THRESHOLD,
        "step6_failure_preserved": decision.get("step6_passes") is False,
        "step6_validation_test_unused": decision.get("validation_used") is False and decision.get("test_used") is False,
        "validation_index_exact_24": _index_role_check(index, "validation"),
        "test_index_exact_24": _index_role_check(index, "test"),
        "dataset_root_present": (RUNTIME_ROOT / "data" / "EV-UAV-dataset").is_dir(),
        "cuda_requirement": (not require_cuda) or torch.cuda.is_available(),
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "protected_data_loaded": False,
    }
    json_write(output, report, overwrite=output.exists())
    if report["status"] != "pass":
        raise RuntimeError("Step7 preflight failed; inspect the generated JSON")
    return report


def load_candidate(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    package = load_checkpoint(checkpoint)
    metadata = package.get("metadata", {})
    if metadata.get("model_id") != FIXED_MODEL or int(metadata.get("epoch", -1)) != FIXED_EPOCH:
        raise RuntimeError("candidate checkpoint metadata differs from the frozen Step7 protocol")
    # A4-2 is exactly 32 frozen H2 inputs plus four history inputs.  Construct
    # that architecture directly and load every tensor strictly; do not rebuild
    # an initialization that is immediately overwritten by the checkpoint.
    model = Step4Refiner(old_width=32, new_width=4, seed=FIXED_SEED).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    model.eval()
    return model


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    preflight(args.step5_results, args.step6_results, args.step4_results, output / "preflight.json", require_cuda=False)
    device = choose_device(args.device, formal=False)
    model = load_candidate(sources(args.step5_results, args.step6_results, args.step4_results)["candidate_checkpoint"], device)
    synthetic = np.zeros((7, 36), dtype=np.uint8)
    synthetic[:, :4] = np.arange(7, dtype=np.uint8)[:, None]
    probabilities = predict_int8(model, synthetic)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "smoke_complete",
        "synthetic_rows": len(synthetic),
        "probabilities_finite": bool(np.isfinite(probabilities).all()),
        "probabilities_in_unit_interval": bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0))),
        "validation_used": False,
        "test_used": False,
        "note": "Synthetic checkpoint/int8 replay only; no Validation/Test file was opened.",
    }
    json_write(output / "smoke.json", report)
    return report


def _model_row(name: str, view: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model": name,
        "threshold": view["threshold"],
        "average_precision": view["average_precision"],
        "overall_iou": view["overall"]["event"]["iou"],
        "overall_precision": view["overall"]["event"]["precision"],
        "overall_recall": view["overall"]["event"]["recall"],
        "overall_f1": view["overall"]["event"]["f1"],
        "overall_pd": view["overall"]["localization"]["pd"],
        "overall_fa": view["overall"]["localization"]["fa"],
        "high_iou": None if view["high_density"] is None else view["high_density"]["event"]["iou"],
        "high_recall": None if view["high_density"] is None else view["high_density"]["event"]["recall"],
        "high_pd": None if view["high_density"] is None else view["high_density"]["localization"]["pd"],
        "high_fa": None if view["high_density"] is None else view["high_density"]["localization"]["fa"],
        "ordinary_iou": None if view["ordinary_density"] is None else view["ordinary_density"]["event"]["iou"],
        "ordinary_recall": None if view["ordinary_density"] is None else view["ordinary_density"]["event"]["recall"],
        "ordinary_pd": None if view["ordinary_density"] is None else view["ordinary_density"]["localization"]["pd"],
        "ordinary_fa": None if view["ordinary_density"] is None else view["ordinary_density"]["localization"]["fa"],
    }


def _overlay_pr(path: Path, title: str, evaluations: Sequence[tuple[str, Mapping[str, Any], str]]) -> None:
    series = []
    for name, evaluation, color in evaluations:
        _, precision, recall = evaluation["_pr_points"]
        series.append((name, np.asarray(recall), np.asarray(precision), color))
    svg_curve(path, title, "Recall", "Precision", series)


def _write_scores(path: Path, records: Sequence[Mapping[str, Any]], field: str) -> None:
    export_score_archive(path, records, [np.asarray(record[field], dtype=np.float32) for record in records])


def evaluate_split(
    role: str, output: Path, model: torch.nn.Module, density: Any,
    device: torch.device, batch_size: int, cache_dir: Path,
    bootstrap_iterations: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    _, records = load_acceptance_records(role, cache_dir)
    blocks = [joined_features(record, FIXED_MODEL, density) for record in records]
    int8_probabilities = [predict_int8(model, block) for block in blocks]
    float_probabilities = [predict_float(model, block, device, batch_size) for block in blocks]

    h2_evaluation = evaluate_records(records, None, device, batch_size, keep_pr_points=True)
    int8_evaluation = evaluate_records(records, None, device, batch_size, candidate_probabilities=int8_probabilities, keep_pr_points=True)
    float_evaluation = evaluate_records(records, None, device, batch_size, candidate_probabilities=float_probabilities, keep_pr_points=True)
    h2_f = working_point_view(h2_evaluation, {"eligible": True, "threshold": H2_F_THRESHOLD})
    h2_r = working_point_view(h2_evaluation, {"eligible": True, "threshold": H2_R_THRESHOLD})
    candidate_int8 = working_point_view(int8_evaluation, {"eligible": True, "threshold": FIXED_THRESHOLD})
    candidate_float = working_point_view(float_evaluation, {"eligible": True, "threshold": FIXED_THRESHOLD})

    comparison = [_model_row("H2-F", h2_f), _model_row("H2-R", h2_r), _model_row("A4-2-int8", candidate_int8)]
    csv_write(output / "comparison.csv", comparison, tuple(comparison[0]))
    comparison_svg(output / "comparison.svg", f"Step7 {role}: fixed working points", comparison)
    _overlay_pr(
        output / "pr_comparison.svg", f"Step7 {role}: PR curves",
        (("H2", h2_evaluation, "#2563eb"), ("A4-2-int8", int8_evaluation, "#dc2626"), ("A4-2-float", float_evaluation, "#16a34a")),
    )

    rates = {str(record["file"]): float(record["sequence_rate_eps"]) for record in records}
    sequence_rows = per_sequence_rows(candidate_int8, h2_f, h2_r, rates)
    csv_write(output / "per_sequence_comparison.csv", sequence_rows, tuple(sequence_rows[0]))
    high_mask = [rates[str(record["file"])] >= 20_000.0 for record in records]

    score_dir = output / "bootstrap_scores"
    candidate_scores = score_dir / "A4-2-int8.npz"
    h2_scores = score_dir / "H2.npz"
    export_score_archive(candidate_scores, records, int8_probabilities)
    _write_scores(h2_scores, records, "baseline_probability")
    bootstrap = paired_bootstrap(
        candidate_scores, h2_scores, candidate_int8, h2_r, high_mask,
        iterations=bootstrap_iterations, seed=BOOTSTRAP_SEED,
    )
    json_write(output / "paired_bootstrap.json", bootstrap)

    gates = acceptance_gates(candidate_int8, h2_f, h2_r)
    delta = delta_report(candidate_int8, h2_r)
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "role": role,
        "candidate": {"model_id": FIXED_MODEL, "seed": FIXED_SEED, "epoch": FIXED_EPOCH, "threshold": FIXED_THRESHOLD, "numeric_path": "exact_int8_replay"},
        "sequence_count": len(records),
        "high_density_sequence_count": int(sum(high_mask)),
        "ordinary_density_sequence_count": int(len(high_mask) - sum(high_mask)),
        "metrics": {
            "H2-F": compact_view(h2_f),
            "H2-R": compact_view(h2_r),
            "A4-2-int8": compact_view(candidate_int8),
            "A4-2-float": compact_view(candidate_float),
        },
        "delta_A4-2-int8_vs_H2-R": delta,
        "acceptance": gates,
        "paired_bootstrap": bootstrap,
        "bootstrap_event_iou_lower_95_positive": bootstrap["delta_event_iou"]["lower_95"] is not None and bootstrap["delta_event_iou"]["lower_95"] > 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "threshold_search_performed": False,
        "training_performed": False,
    }
    json_write(output / "split_summary.json", summary)

    export_metrics(output / "metrics" / "H2-F", copy.deepcopy(h2_evaluation), h2_f)
    export_metrics(output / "metrics" / "H2-R", h2_evaluation, h2_r)
    export_metrics(output / "metrics" / "A4-2-int8", int8_evaluation, candidate_int8)
    export_metrics(output / "metrics" / "A4-2-float", float_evaluation, candidate_float)
    return summary


def _archive_interrupted_work(output: Path, role: str) -> None:
    working = output / f".{role}.working"
    if not working.exists():
        return
    destination = output / "failed_attempts" / f"{role}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(working), str(destination))


def _write_manifest(output: Path) -> Dict[str, Any]:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or "work" in path.relative_to(output).parts or path.name == "RESULT_MANIFEST.json":
            continue
        rows.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "files": rows, "cache_excluded": True}
    json_write(output / "RESULT_MANIFEST.json", manifest)
    return manifest


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.run:
        raise RuntimeError("formal Step7 requires the explicit --run flag")
    output = args.output
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}; use --resume only after an infrastructure interruption")
    output.mkdir(parents=True, exist_ok=True)
    protocol_lock = output / "step7_protocol.lock.json"
    source_protocol = read_json(HERE / "step7_protocol.json")
    if protocol_lock.exists():
        if read_json(protocol_lock) != source_protocol:
            raise RuntimeError("existing Step7 protocol lock differs from the packaged protocol")
    else:
        json_write(protocol_lock, source_protocol)
    preflight(args.step5_results, args.step6_results, args.step4_results, output / "preflight.json", require_cuda=args.device != "cpu")
    device = choose_device(args.device, formal=True)
    source = sources(args.step5_results, args.step6_results, args.step4_results)
    model = load_candidate(source["candidate_checkpoint"], device)
    density = density_from_json(source["density_transform"])
    summaries: Dict[str, Any] = {}
    for role in ROLES:
        final_dir = output / role
        if final_dir.exists():
            summary_path = final_dir / "split_summary.json"
            if not args.resume or not summary_path.is_file():
                raise RuntimeError(f"existing incomplete or unauthorized role result: {final_dir}")
            summaries[role] = read_json(summary_path)
            print(f"[Step7] {role}: reusing completed immutable result", flush=True)
            continue
        _archive_interrupted_work(output, role)
        working = output / f".{role}.working"
        working.mkdir()
        print(f"[Step7] {role}: loading 24 indexed sequences and evaluating fixed models", flush=True)
        summaries[role] = evaluate_split(
            role, working, model, density, device, args.batch_size,
            output / "work" / "cache", args.bootstrap_iterations,
        )
        working.replace(final_dir)
        d = summaries[role]["delta_A4-2-int8_vs_H2-R"]
        print(
            f"[Step7] {role}: AP delta={d['average_precision']:+.6f} "
            f"IoU delta={d['overall_iou']:+.6f} gates={summaries[role]['acceptance']['passes']}",
            flush=True,
        )

    validation = summaries["validation"]
    test = summaries["test"]
    test_delta = test["delta_A4-2-int8_vs_H2-R"]
    checks = {
        "validation_fixed_protocol_complete": validation.get("status") == "complete" and validation.get("threshold_search_performed") is False,
        "test_fixed_protocol_complete": test.get("status") == "complete" and test.get("threshold_search_performed") is False,
        "validation_safety_gates_pass": bool(validation["acceptance"]["passes"]),
        "test_safety_gates_pass": bool(test["acceptance"]["passes"]),
        "test_iou_improves": test_delta["overall_iou"] is not None and test_delta["overall_iou"] > 0.0,
        "test_ap_not_lower": test_delta["average_precision"] is not None and test_delta["average_precision"] >= 0.0,
        "test_iou_bootstrap_lower_95_positive": bool(test["bootstrap_event_iou_lower_95_positive"]),
    }
    if all(checks.values()):
        conclusion = "independently_confirmed_replacement_candidate"
    elif checks["test_iou_improves"] and checks["test_ap_not_lower"]:
        conclusion = "test_improvement_observed_but_one_or_more_guardrails_or_confidence_checks_failed"
    else:
        conclusion = "independent_test_does_not_confirm_replacement"
    decision = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "candidate": {"model_id": FIXED_MODEL, "seed": FIXED_SEED, "epoch": FIXED_EPOCH, "threshold": FIXED_THRESHOLD},
        "conclusion": conclusion,
        "passes_all_independent_acceptance_checks": all(checks.values()),
        "checks": checks,
        "validation": "validation/split_summary.json",
        "test": "test/split_summary.json",
        "original_step6_passes": False,
        "original_step6_result_rewritten": False,
        "interpretation": "Step7 is a pre-frozen supplemental independent acceptance requested after Step6; it does not retroactively change the Step6 decision.",
        "validation_used": True,
        "test_used": True,
        "retraining_or_threshold_search": False,
    }
    json_write(output / "step7_decision.json", decision)
    _write_manifest(output)
    return decision


def verify_results(results: Path) -> Dict[str, Any]:
    manifest = read_json(results / "RESULT_MANIFEST.json")
    failures = []
    for row in manifest["files"]:
        path = results / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            failures.append(row["path"])
    decision = read_json(results / "step7_decision.json")
    checks = {
        "manifest_complete": not failures,
        "decision_complete": decision.get("status") == "complete",
        "candidate_frozen": decision.get("candidate") == {"model_id": FIXED_MODEL, "seed": FIXED_SEED, "epoch": FIXED_EPOCH, "threshold": FIXED_THRESHOLD},
        "both_roles_used": decision.get("validation_used") is True and decision.get("test_used") is True,
        "no_retraining_or_threshold_search": decision.get("retraining_or_threshold_search") is False,
        "step6_result_preserved": decision.get("original_step6_passes") is False and decision.get("original_step6_result_rewritten") is False,
    }
    report = {"schema_version": 1, "protocol": PROTOCOL, "status": "pass" if all(checks.values()) else "fail", "checks": checks, "manifest_failures": failures}
    if report["status"] != "pass":
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--step5-results", type=Path, default=Path("input/step5_results"))
    common.add_argument("--step6-results", type=Path, default=Path("input/step6_results"))
    common.add_argument("--step4-results", type=Path, default=Path("input/step4_results"))
    pre = sub.add_parser("preflight", parents=[common])
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--require-cuda", action="store_true")
    sm = sub.add_parser("smoke", parents=[common])
    sm.add_argument("--output", type=Path, required=True)
    sm.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    run = sub.add_parser("run-all", parents=[common])
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--batch-size", type=int, default=65536)
    run.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    run.add_argument("--run", action="store_true")
    run.add_argument("--resume", action="store_true")
    verify = sub.add_parser("verify-results")
    verify.add_argument("--results", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "preflight":
        result = preflight(args.step5_results, args.step6_results, args.step4_results, args.output, args.require_cuda)
    elif args.command == "smoke":
        result = smoke(args)
    elif args.command == "run-all":
        if args.bootstrap_iterations < 10000:
            raise ValueError("formal Step7 requires at least 10,000 bootstrap iterations")
        result = run_all(args)
    else:
        result = verify_results(args.results)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

