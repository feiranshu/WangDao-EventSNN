"""Execute the evidence-gated H2 ANN step-2 workflow.

Examples:
  python run_step2.py preflight --output work/preflight.json
  python run_step2.py audit --output work/feature_audit
  python run_step2.py stc-diagnose --output work/stc_diagnostic.json
  python run_step2.py train --model A2-1 --output runs/A2-1-seed42 --run
  python run_step2.py run-all --output step2_run --run

No command can load Validation or Test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from step2_audit import run_feature_audit, run_stc_diagnostic
from step2_common import (
    FEATURE_GROUPS,
    RUNTIME_ROOT,
    json_write,
    load_records,
    role_paths,
    sha256,
)
from step2_training import (
    MODEL_DESCRIPTIONS,
    load_checkpoint_package,
    repair_candidate_export,
    train_candidate,
)


def choose_device(name: str, formal: bool) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        if formal:
            raise RuntimeError("formal step-2 training requires CUDA; use --smoke for CPU verification")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if formal:
        raise RuntimeError("formal step-2 training requires a CUDA server")
    return torch.device("cpu")


def preflight(output: Path, require_cuda: bool) -> Dict[str, Any]:
    h2_path = RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"
    index_path = RUNTIME_ROOT / "data" / "index.json"
    train = role_paths("train_main")
    dev = role_paths("train_dev")
    report = {
        "schema_version": 1,
        "status": "pass",
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "require_cuda": bool(require_cuda),
        "train_main_sequences": len(train),
        "train_dev_sequences": len(dev),
        "validation_used": False,
        "test_used": False,
        "h2_sha256": sha256(h2_path),
        "index_sha256": sha256(index_path),
        "runtime_root": str(RUNTIME_ROOT),
    }
    if len(train) != 79 or len(dev) != 10:
        report.update(status="fail", error="frozen role counts must be 79 Train-main and 10 Train-dev")
    if require_cuda and not torch.cuda.is_available():
        report.update(status="fail", error="formal run requires CUDA")
    json_write(output, report)
    if report["status"] != "pass":
        raise RuntimeError(report["error"])
    return report


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_inputs(cache: Path, smoke: bool) -> tuple[Any, Any]:
    train_limit = 20 if smoke else None
    _, train_records = load_records("train_main", cache, limit=train_limit)
    _, dev_records = load_records("train_dev", cache)
    return train_records, dev_records


def _iou_ap(report: Mapping[str, Any]) -> tuple[float, float]:
    view = report["final_working_point"]
    return float(view["overall"]["event"]["iou"]), float(view["average_precision"])


def _meaningful_gain(candidate: Mapping[str, Any], a2_1: Mapping[str, Any]) -> bool:
    if not candidate.get("final_working_point", {}).get("eligible"):
        return False
    if not a2_1.get("final_working_point", {}).get("eligible"):
        return False
    candidate_iou, candidate_ap = _iou_ap(candidate)
    a2_iou, a2_ap = _iou_ap(a2_1)
    return candidate_iou >= a2_iou + 0.002 and candidate_ap >= a2_ap


def _state_dict_equal(first: Path, second: Path) -> bool:
    first_package = load_checkpoint_package(first)
    second_package = load_checkpoint_package(second)
    if set(first_package["state_dict"]) != set(second_package["state_dict"]):
        return False
    return all(
        torch.equal(first_package["state_dict"][key], second_package["state_dict"][key])
        for key in first_package["state_dict"]
    )


def _train(
    model_id: str,
    output: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    selected_group: Optional[str],
) -> Dict[str, Any]:
    return train_candidate(
        model_id=model_id,
        output_dir=output,
        train_records=train_records,
        dev_records=dev_records,
        device=device,
        seed=seed,
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        selected_feature_group=selected_group,
        smoke=args.smoke,
        resume_checkpoint=getattr(args, "resume_checkpoint", None),
        epoch_offset=getattr(args, "epoch_offset", 0),
    )


def _train_or_resume(
    model_id: str,
    output: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    selected_group: Optional[str],
) -> Dict[str, Any]:
    if not output.exists():
        return _train(
            model_id, output, train_records, dev_records, device, args, seed, selected_group
        )
    if not getattr(args, "resume", False):
        raise FileExistsError(f"refusing to overwrite {output}")
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"resume candidate has no summary.json: {output}")
    report = read_json(summary_path)
    if report.get("model_id") != model_id or int(report.get("seed", -1)) != int(seed):
        raise RuntimeError(f"resume candidate identity mismatch: {output}")
    if report.get("status") == "complete":
        return report
    return repair_candidate_export(output, dev_records, device, args.batch_size)


def _candidate_priority(audit: Mapping[str, Any], stc: Mapping[str, Any]) -> list[str]:
    result = []
    if audit.get("a2_3_eligible"):
        result.append("A2-3")
    if stc.get("passes"):
        result.append("A2-4")
    result.append("A2-2")
    return result


def _rank_report(report: Mapping[str, Any]) -> tuple[float, float, float, float]:
    view = report["final_working_point"]
    return (
        float(view["overall"]["event"]["iou"]),
        float(view["average_precision"]),
        float(view["high_density"]["event"]["recall"]),
        -float(view["overall"]["localization"]["fa"]),
    )


def _copy_metrics(run_root: Path, reports: Mapping[str, Mapping[str, Any]]) -> None:
    destination = run_root / "metrics"
    destination.mkdir(exist_ok=False)
    for name, report in reports.items():
        if name.endswith("-null"):
            continue
        source = run_root / "runs" / name / "metrics"
        if source.is_dir():
            shutil.copytree(source, destination / name)


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    device = choose_device(args.device, formal=not args.smoke)
    preflight_path = args.output / "preflight.json"
    if not (args.resume and preflight_path.is_file()):
        preflight(preflight_path, require_cuda=not args.smoke)
    cache = args.output / "work" / "cache"
    train_records, dev_records = _load_inputs(cache, args.smoke)
    audit_path = args.output / "work" / "feature_audit" / "summary.json"
    audit = (
        read_json(audit_path) if args.resume and audit_path.is_file()
        else run_feature_audit(
            train_records, dev_records, audit_path.parent, device, args.batch_size
        )
    )
    stc_path = args.output / "work" / "stc_diagnostic.json"
    stc = (
        read_json(stc_path) if args.resume and stc_path.is_file()
        else run_stc_diagnostic(dev_records, stc_path)
    )
    runs = args.output / "runs"
    runs.mkdir(exist_ok=args.resume)
    reports: Dict[str, Dict[str, Any]] = {}
    a2_1_name = "A2-1-seed42"
    reports[a2_1_name] = _train_or_resume(
        "A2-1", runs / a2_1_name, train_records, dev_records, device, args, 42, None
    )
    attempted = ["A2-1"]
    skipped = []
    priority = _candidate_priority(audit, stc)
    for model_id in priority:
        if len(attempted) >= 3:
            skipped.append({"model_id": model_id, "reason": "three-candidate seed42 cap reached"})
            continue
        if model_id == "A2-4":
            null_name = "A2-4-null-seed42"
            null_report = _train_or_resume(
                "A2-4-null", runs / null_name, train_records, dev_records, device, args, 42, None
            )
            reports[null_name] = null_report
            first_checkpoint = runs / a2_1_name / reports[a2_1_name]["best_checkpoint"]
            null_checkpoint = runs / null_name / null_report["best_checkpoint"]
            if not _state_dict_equal(first_checkpoint, null_checkpoint):
                skipped.append({"model_id": "A2-4", "reason": "A2-4-null did not reproduce A2-1 exactly"})
                continue
        name = f"{model_id}-seed42"
        selected_group = audit.get("selected_feature_group") if model_id == "A2-3" else None
        reports[name] = _train_or_resume(
            model_id, runs / name, train_records, dev_records, device, args, 42, selected_group
        )
        attempted.append(model_id)
        if _meaningful_gain(reports[name], reports[a2_1_name]) and reports[name]["final_gates"]["passes"]:
            skipped.extend(
                {"model_id": remaining, "reason": "a candidate passed, gained >=0.2pp IoU over A2-1, and did not lower AP"}
                for remaining in priority if remaining not in attempted
            )
            break

    seed42_reports = {
        name: report for name, report in reports.items()
        if name.endswith("seed42") and "null" not in name
    }
    passing = {name: report for name, report in seed42_reports.items() if report["final_gates"]["passes"]}
    multiseed = None
    if passing and not args.smoke:
        best_name, best_report = max(passing.items(), key=lambda item: _rank_report(item[1]))
        model_id = best_name.split("-seed", 1)[0]
        selected_group = audit.get("selected_feature_group") if model_id == "A2-3" else None
        seed_reports = {42: best_report}
        for seed in (43, 44):
            name = f"{model_id}-seed{seed}"
            reports[name] = _train_or_resume(
                model_id, runs / name, train_records, dev_records, device, args, seed, selected_group
            )
            seed_reports[seed] = reports[name]
        baseline_iou = float(best_report["h2_r"]["overall"]["event"]["iou"])
        baseline_ap = float(best_report["h2_r"]["average_precision"])
        ious = [float(value["final_working_point"]["overall"]["event"]["iou"]) for value in seed_reports.values()]
        aps = [float(value["final_working_point"]["average_precision"]) for value in seed_reports.values()]
        all_fair_and_ordinary = all(
            value["final_gates"]["checks"]["overall_pd_loss_within_0_5pp"]
            and value["final_gates"]["checks"]["overall_fa_rise_within_10pct"]
            and value["final_gates"]["checks"]["ordinary_iou_loss_within_0_3pp"]
            and value["final_gates"]["checks"]["ordinary_recall_loss_within_0_3pp"]
            and value["final_gates"]["checks"]["ordinary_pd_loss_within_0_3pp"]
            and value["final_gates"]["checks"]["ordinary_fa_rise_within_5pct"]
            for value in seed_reports.values()
        )
        full_passes = sum(bool(value["final_gates"]["passes"]) for value in seed_reports.values())
        checks = {
            "all_seeds_fair_and_ordinary_protected": all_fair_and_ordinary,
            "at_least_two_full_passes": full_passes >= 2,
            "mean_iou_gain_at_least_0_5pp": float(np.mean(ious)) >= baseline_iou + 0.005,
            "mean_ap_not_lower": float(np.mean(aps)) >= baseline_ap,
        }
        multiseed = {
            "model_id": model_id,
            "seeds": [42, 43, 44],
            "mean_iou": float(np.mean(ious)),
            "mean_iou_gain": float(np.mean(ious) - baseline_iou),
            "mean_ap": float(np.mean(aps)),
            "full_passes": full_passes,
            "checks": checks,
            "passes": all(checks.values()),
        }
    decision = {
        "schema_version": 1,
        "status": "smoke_complete" if args.smoke else "complete",
        "validation_used": False,
        "test_used": False,
        "attempted_seed42_models": attempted,
        "skipped": skipped,
        "feature_audit_decision": audit["decision"],
        "selected_feature_group": audit.get("selected_feature_group"),
        "stc_decision": stc["decision"],
        "seed42_passing_models": list(passing),
        "multiseed": multiseed,
        "step2_passes": bool(multiseed and multiseed["passes"]),
        "next_step": (
            "enter_step3_evidence_review" if multiseed and multiseed["passes"]
            else ("smoke_only" if args.smoke else "stop_step2_without_promoting_a_model")
        ),
        "runs": {name: str((runs / name / "summary.json").relative_to(args.output)) for name in reports},
    }
    decision_path = args.output / "step2_decision.json"
    json_write(decision_path, decision, overwrite=args.resume and decision_path.exists())
    _copy_metrics(args.output, reports)
    return decision


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--require-cuda", action="store_true")

    audit = sub.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--cache", type=Path, default=Path("work/cache"))
    audit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    audit.add_argument("--batch-size", type=int, default=4096)

    stc = sub.add_parser("stc-diagnose")
    stc.add_argument("--output", type=Path, required=True)
    stc.add_argument("--cache", type=Path, default=Path("work/cache"))

    train = sub.add_parser("train")
    train.add_argument("--model", choices=tuple(MODEL_DESCRIPTIONS), required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--cache", type=Path, default=Path("work/cache"))
    train.add_argument("--audit-summary", type=Path)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--learning-rate", type=float, default=1.0e-4)
    train.add_argument("--resume-checkpoint", type=Path)
    train.add_argument("--epoch-offset", type=int, default=0)
    train.add_argument("--run", action="store_true")
    train.add_argument("--smoke", action="store_true")

    all_parser = sub.add_parser("run-all")
    all_parser.add_argument("--output", type=Path, required=True)
    all_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    all_parser.add_argument("--epochs", type=int, default=20)
    all_parser.add_argument("--batch-size", type=int, default=4096)
    all_parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    all_parser.add_argument("--run", action="store_true")
    all_parser.add_argument("--smoke", action="store_true")
    all_parser.add_argument("--resume", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "run") and args.run == args.smoke:
        raise SystemExit("choose exactly one of --run or --smoke")
    if hasattr(args, "epochs") and not 1 <= args.epochs <= 20:
        raise SystemExit("epochs must be 1..20")
    if hasattr(args, "batch_size") and args.batch_size != 4096:
        raise SystemExit("the frozen step-2 protocol requires batch-size 4096")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise SystemExit("learning rate must be positive")
    if hasattr(args, "epoch_offset") and args.epoch_offset < 0:
        raise SystemExit("epoch-offset must be non-negative")
    if getattr(args, "epoch_offset", 0) and getattr(args, "resume_checkpoint", None) is None:
        raise SystemExit("epoch-offset requires --resume-checkpoint")

    if args.command == "preflight":
        result = preflight(args.output, args.require_cuda)
    elif args.command == "audit":
        device = choose_device(args.device, formal=False)
        train_records, dev_records = _load_inputs(args.cache, smoke=False)
        result = run_feature_audit(train_records, dev_records, args.output, device, args.batch_size)
    elif args.command == "stc-diagnose":
        _, dev_records = load_records("train_dev", args.cache)
        result = run_stc_diagnostic(dev_records, args.output)
    elif args.command == "train":
        device = choose_device(args.device, formal=args.run)
        train_records, dev_records = _load_inputs(args.cache, args.smoke)
        selected_group = None
        if args.model == "A2-3":
            if args.audit_summary is None:
                raise SystemExit("A2-3 requires --audit-summary")
            selected_group = read_json(args.audit_summary).get("selected_feature_group")
            if selected_group not in FEATURE_GROUPS:
                raise SystemExit("feature audit did not qualify an A2-3 group")
        result = _train(
            args.model, args.output, train_records, dev_records, device, args, args.seed, selected_group
        )
    else:
        result = run_all(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

