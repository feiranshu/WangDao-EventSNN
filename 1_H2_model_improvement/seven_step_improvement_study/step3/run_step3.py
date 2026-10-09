"""Run H2 ANN Step 3 without accessing Validation or Test.

All README commands are single-line invocations.  Run ``preflight`` and unit
tests first; ``run-all`` then performs zero-update, D3-0, A3-1 and A3-2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Mapping

import numpy as np
import torch

from step3_common import (
    RUNTIME_ROOT,
    build_sequence_record,
    json_write,
    load_records,
    load_sequence,
    role_paths,
    sequence_rate,
    sha256,
)
from step3_diagnostics import run_d3_diagnostics
from step3_training import MODEL_DESCRIPTIONS, train_candidate, zero_update_control


def choose_device(name: str, formal: bool) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        if formal:
            raise RuntimeError("formal step-3 work requires CUDA; CPU is allowed only with --smoke")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if formal:
        raise RuntimeError("formal step-3 work requires a CUDA server")
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
        "sequence_density_boundary_eps": 20000.0,
    }
    if len(train) != 79 or len(dev) != 10:
        report.update(status="fail", error="frozen role counts must be 79 Train-main and 10 Train-dev")
    if require_cuda and not torch.cuda.is_available():
        report.update(status="fail", error="formal run requires CUDA")
    json_write(output, report)
    if report["status"] != "pass":
        raise RuntimeError(report["error"])
    return report


def _load(cache: Path, smoke: bool) -> tuple[Any, Any]:
    if smoke:
        pipeline, train = load_records("train_main", cache, limit=18)
        high_paths = []
        for path in role_paths("train_main"):
            if path.name in {record["file"] for record in train}:
                continue
            events = load_sequence(path)[0]
            if sequence_rate(events) >= 20000.0:
                high_paths.append(path)
            if len(high_paths) == 2:
                break
        if len(high_paths) != 2:
            raise RuntimeError("smoke could not find two indexed high-density Train-main sequences")
        train.extend(
            build_sequence_record(path, pipeline, cache / "train_main", False)
            for path in high_paths
        )
    else:
        _, train = load_records("train_main", cache)
    _, dev = load_records("train_dev", cache)
    return train, dev


def _noninferiority(report: Mapping[str, Any]) -> Dict[str, Any]:
    view = report["final"]["int8_working_point"]
    baseline = report["h2_r"]
    if not view.get("eligible"):
        return {"passes": False, "checks": {"working_point_exists": False}}

    def at_least(candidate: float, reference: float, delta: float) -> bool:
        return float(candidate) >= float(reference) + delta

    def ratio(candidate: float, reference: float, limit: float) -> bool:
        return float(candidate) <= (0.0 if float(reference) == 0.0 else float(reference) * limit)

    checks = {
        "working_point_exists": True,
        "ap_loss_within_0_1pp": at_least(view["average_precision"], baseline["average_precision"], -0.001),
        "overall_iou_loss_within_0_3pp": at_least(view["overall"]["event"]["iou"], baseline["overall"]["event"]["iou"], -0.003),
        "high_iou_loss_within_0_3pp": at_least(view["high_density"]["event"]["iou"], baseline["high_density"]["event"]["iou"], -0.003),
        "high_recall_loss_within_0_3pp": at_least(view["high_density"]["event"]["recall"], baseline["high_density"]["event"]["recall"], -0.003),
        "ordinary_iou_loss_within_0_3pp": at_least(view["ordinary_density"]["event"]["iou"], baseline["ordinary_density"]["event"]["iou"], -0.003),
        "ordinary_recall_loss_within_0_3pp": at_least(view["ordinary_density"]["event"]["recall"], baseline["ordinary_density"]["event"]["recall"], -0.003),
        "ordinary_pd_loss_within_0_3pp": at_least(view["ordinary_density"]["localization"]["pd"], baseline["ordinary_density"]["localization"]["pd"], -0.003),
        "ordinary_fa_rise_within_5pct": ratio(view["ordinary_density"]["localization"]["fa"], baseline["ordinary_density"]["localization"]["fa"], 1.05),
    }
    return {"passes": all(checks.values()), "checks": checks}


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_reused_stage(
    path: Path,
    stage: str,
    expected_status: str,
    expected_smoke: bool | None = None,
) -> Dict[str, Any]:
    """Read a completed stage without crossing smoke/formal boundaries."""
    if not path.is_file():
        raise RuntimeError(
            f"cannot resume {stage}: missing {path}; preserve the incomplete directory and rerun in a new output directory"
        )
    report = _read(path)
    if report.get("status") != expected_status:
        raise RuntimeError(
            f"cannot resume {stage}: expected status={expected_status}, got {report.get('status')!r}"
        )
    if expected_smoke is not None:
        if "smoke" not in report or bool(report["smoke"]) != bool(expected_smoke):
            mode = "smoke" if expected_smoke else "formal"
            raise RuntimeError(f"cannot resume {stage}: existing result does not belong to {mode} mode")
    if report.get("validation_used") is not False or report.get("test_used") is not False:
        raise RuntimeError(f"cannot resume {stage}: Validation/Test isolation cannot be verified")
    return report


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    device = choose_device(args.device, formal=not args.smoke)
    preflight_path = args.output / "preflight.json"
    if preflight_path.exists():
        existing_preflight = _read_reused_stage(preflight_path, "preflight", "pass")
        if bool(existing_preflight.get("require_cuda")) != bool(not args.smoke):
            raise RuntimeError("cannot resume preflight across smoke/formal modes")
    else:
        preflight(preflight_path, require_cuda=not args.smoke)
    train_records, dev_records = _load(args.output / "work" / "cache", args.smoke)

    zero_dir = args.output / "preflight" / "zero_update"
    if zero_dir.exists():
        zero = _read_reused_stage(
            zero_dir / "zero_update_control.json", "zero-update", "pass", args.smoke
        )
    else:
        zero = zero_update_control(zero_dir, train_records, dev_records, device, args.batch_size, args.smoke)

    diagnostic_dir = args.output / "diagnostics" / "D3-0"
    if diagnostic_dir.exists():
        diagnostics = _read_reused_stage(
            diagnostic_dir / "summary.json",
            "D3-0",
            "smoke_complete" if args.smoke else "complete",
            args.smoke,
        )
    else:
        diagnostics = run_d3_diagnostics(
            diagnostic_dir, train_records, dev_records, device, args.batch_size,
            args.probe_epochs, args.smoke,
        )

    runs = args.output / "runs"
    runs.mkdir(exist_ok=True)
    reports = {}
    for model_id in ("A3-1", "A3-2"):
        name = f"{model_id}-seed42"
        run_dir = runs / name
        if run_dir.exists():
            report = _read_reused_stage(
                run_dir / "summary.json", name, "complete", args.smoke
            )
            configuration = report.get("configuration", {})
            expected_epochs = 1 if args.smoke else args.epochs
            if (
                int(configuration.get("epochs", -1)) != int(expected_epochs)
                or int(configuration.get("epoch_offset", -1)) != 0
                or int(configuration.get("batch_size", -1)) != int(args.batch_size)
                or float(configuration.get("learning_rate", -1.0)) != float(args.learning_rate)
            ):
                raise RuntimeError(f"cannot resume {name}: configuration differs from the current run")
        else:
            report = train_candidate(
                model_id, run_dir, train_records, dev_records, device, 42,
                1 if args.smoke else args.epochs, args.batch_size, args.learning_rate,
                args.smoke,
            )
        reports[name] = report

    noninferiority = {name: _noninferiority(report) for name, report in reports.items()}
    passing = [name for name, value in noninferiority.items() if value["passes"]]
    decision = {
        "schema_version": 1,
        "status": "smoke_complete" if args.smoke else "complete",
        "validation_used": False,
        "test_used": False,
        "zero_update_status": zero["status"],
        "d3_0_status": diagnostics["status"],
        "models": {name: str(Path("runs") / name / "summary.json") for name in reports},
        "noninferiority": noninferiority,
        "noninferiority_passing_models": passing,
        "step3_produces_replacement": any(report["final"]["gates"]["passes"] for report in reports.values()),
        "next_step": (
            "smoke_only" if args.smoke else
            ("review_D3-0_and_enter_step4_or_capacity_branch" if not passing else "review_passing_A3_before_step4")
        ),
    }
    decision_path = args.output / "step3_decision.json"
    json_write(decision_path, decision, overwrite=decision_path.exists())
    metrics = args.output / "metrics"
    if not metrics.exists():
        metrics.mkdir()
        for name in reports:
            shutil.copytree(runs / name / "metrics", metrics / name)
    return decision


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    check = sub.add_parser("preflight")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--require-cuda", action="store_true")

    zero = sub.add_parser("zero-update")
    zero.add_argument("--output", type=Path, required=True)
    zero.add_argument("--cache", type=Path, default=Path("work/cache"))
    zero.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    zero.add_argument("--batch-size", type=int, default=4096)
    zero.add_argument("--smoke", action="store_true")

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--cache", type=Path, default=Path("work/cache"))
    diagnose.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    diagnose.add_argument("--batch-size", type=int, default=4096)
    diagnose.add_argument("--probe-epochs", type=int, default=30)
    diagnose.add_argument("--smoke", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--model", choices=tuple(MODEL_DESCRIPTIONS), required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--cache", type=Path, default=Path("work/cache"))
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--learning-rate", type=float, default=1.0e-5)
    train.add_argument("--resume-checkpoint", type=Path)
    train.add_argument("--epoch-offset", type=int, default=0)
    train.add_argument("--run", action="store_true")
    train.add_argument("--smoke", action="store_true")

    all_parser = sub.add_parser("run-all")
    all_parser.add_argument("--output", type=Path, required=True)
    all_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    all_parser.add_argument("--epochs", type=int, default=20)
    all_parser.add_argument("--probe-epochs", type=int, default=30)
    all_parser.add_argument("--batch-size", type=int, default=4096)
    all_parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    all_parser.add_argument("--run", action="store_true")
    all_parser.add_argument("--smoke", action="store_true")
    all_parser.add_argument("--resume", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "run") and args.run == args.smoke:
        raise SystemExit("choose exactly one of --run or --smoke")
    if hasattr(args, "epochs") and not 1 <= args.epochs <= 20:
        raise SystemExit("epochs must be 1..20; continuation uses a new 10-epoch run")
    if hasattr(args, "probe_epochs") and not 1 <= args.probe_epochs <= 30:
        raise SystemExit("probe-epochs must be 1..30")
    if hasattr(args, "batch_size") and args.batch_size != 4096:
        raise SystemExit("the frozen step-3 protocol requires batch-size 4096")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise SystemExit("learning rate must be positive")
    if getattr(args, "epoch_offset", 0) < 0:
        raise SystemExit("epoch-offset must be non-negative")
    if getattr(args, "epoch_offset", 0) and getattr(args, "resume_checkpoint", None) is None:
        raise SystemExit("epoch-offset requires --resume-checkpoint")
    if getattr(args, "resume_checkpoint", None) is not None and args.epochs != 10:
        raise SystemExit("a continuation run must add exactly 10 epochs")

    if args.command == "preflight":
        result = preflight(args.output, args.require_cuda)
    elif args.command == "zero-update":
        device = choose_device(args.device, formal=not args.smoke)
        train, dev = _load(args.cache, args.smoke)
        result = zero_update_control(args.output, train, dev, device, args.batch_size, args.smoke)
    elif args.command == "diagnose":
        device = choose_device(args.device, formal=not args.smoke)
        train, dev = _load(args.cache, args.smoke)
        result = run_d3_diagnostics(args.output, train, dev, device, args.batch_size, args.probe_epochs, args.smoke)
    elif args.command == "train":
        device = choose_device(args.device, formal=args.run)
        train, dev = _load(args.cache, args.smoke)
        result = train_candidate(
            args.model, args.output, train, dev, device, args.seed, args.epochs,
            args.batch_size, args.learning_rate, args.smoke,
            args.resume_checkpoint, args.epoch_offset,
        )
    else:
        result = run_all(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
