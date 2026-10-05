#!/usr/bin/env python3
"""Run Step4 feature experiments without exposing Validation or Test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from step4_common import (
    MODEL_DESCRIPTIONS,
    PROTOCOL,
    RUNTIME_ROOT,
    density_from_json,
    fit_density,
    json_write,
    load_records,
    pair_entry,
    primary_rank,
    role_paths,
    sha256,
)
from step4_training import equivalence_audit, train_candidate


def choose_device(name: str, formal: bool) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        if formal:
            raise RuntimeError("formal Step4 requires CUDA; CPU is smoke-only")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if formal:
        raise RuntimeError("formal Step4 requires CUDA")
    return torch.device("cpu")


def _load(cache: Path, smoke: bool) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    limit = 4 if smoke else None
    _, train = load_records("train_main", cache, limit=limit)
    _, dev = load_records("train_dev", cache)
    return train, dev


def preflight(output: Path, require_cuda: bool, require_gpus: int) -> Dict[str, Any]:
    h2 = RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"
    index = RUNTIME_ROOT / "data" / "index.json"
    train_paths = role_paths("train_main")
    dev_paths = role_paths("train_dev")
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "status": "pass",
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "require_cuda": bool(require_cuda), "require_gpus": int(require_gpus),
        "h2_sha256": sha256(h2), "index_sha256": sha256(index),
        "runtime_root": str(RUNTIME_ROOT),
        "train_main_sequences": len(train_paths), "train_dev_sequences": len(dev_paths),
        "validation_used": False, "test_used": False,
        "step3_frozen_reference": {
            "H2_R_int8_ap": 0.927410633080494,
            "H2_R_iou": 0.7805328724,
            "A3_1_selected_ap": 0.9276521875,
            "A3_1_selected_iou": 0.7823043790,
            "decision": "no Step3 replacement; Step4 starts from original H2",
        },
    }
    if len(train_paths) != 79 or len(dev_paths) != 10:
        report.update(status="fail", error="frozen roles must contain 79 Train-main and 10 Train-dev sequences")
    elif require_cuda and not torch.cuda.is_available():
        report.update(status="fail", error="CUDA is required")
    elif require_gpus and torch.cuda.device_count() < require_gpus:
        report.update(status="fail", error=f"at least {require_gpus} visible GPUs are required")
    json_write(output, report)
    if report["status"] != "pass":
        raise RuntimeError(report["error"])
    return report


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(path: Path, smoke: bool) -> Optional[Dict[str, Any]]:
    summary = path / "summary.json"
    if not summary.is_file():
        return None
    value = _read(summary)
    if value.get("status") != "complete" or bool(value.get("smoke")) != bool(smoke):
        raise RuntimeError(f"existing run is incomplete or mode-mismatched: {path}")
    if value.get("validation_used") is not False or value.get("test_used") is not False:
        raise RuntimeError(f"data-role isolation cannot be verified: {path}")
    return value


def _stream(pipe: Any, log: Any, prefix: str) -> None:
    for line in iter(pipe.readline, ""):
        log.write(line)
        log.flush()
        print(f"[{prefix}] {line}", end="", flush=True)
    pipe.close()


def launch_wave(
    jobs: Sequence[tuple[str, Optional[str]]], output: Path, cache: Path,
    density_path: Path, gpu_ids: Sequence[str], smoke: bool,
) -> Dict[str, Dict[str, Any]]:
    if not jobs:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    pending = []
    for index, (model_id, selected_information) in enumerate(jobs):
        run_dir = output / "runs" / f"{model_id}-seed42"
        existing = _complete(run_dir, smoke) if run_dir.exists() else None
        if existing is not None:
            results[model_id] = existing
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), "train", "--model", model_id,
            "--output", str(run_dir), "--cache", str(cache), "--density", str(density_path),
            "--device", "cuda" if not smoke else "cpu", "--seed", "42",
            "--epochs", "1" if smoke else "10", "--batch-size", "4096",
            "--smoke" if smoke else "--run",
        ]
        if selected_information:
            command.extend(("--selected-information", selected_information))
        environment = os.environ.copy()
        if not smoke:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[index % len(gpu_ids)])
        log_path = output / "logs" / f"{model_id}-seed42.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8", newline="\n")
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parent, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        thread = threading.Thread(target=_stream, args=(process.stdout, log, model_id), daemon=True)
        thread.start()
        pending.append((model_id, run_dir, process, thread, log))
        if smoke:  # CPU smoke is intentionally serial and small.
            process.wait()
            thread.join()
            log.close()
            if process.returncode:
                raise RuntimeError(f"{model_id} smoke failed; inspect {log_path}")
            results[model_id] = _complete(run_dir, smoke)  # type: ignore[assignment]
            pending.clear()
    failures = []
    for model_id, run_dir, process, thread, log in pending:
        process.wait()
        thread.join()
        log.close()
        if process.returncode:
            failures.append(model_id)
        else:
            results[model_id] = _complete(run_dir, smoke)  # type: ignore[assignment]
    if failures:
        raise RuntimeError("parallel Step4 jobs failed: " + ", ".join(failures))
    return results


def _drop_time_entry(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> Dict[str, Any]:
    c = candidate["final"]["primary"]
    n = control["final"]["primary"]
    cv, nv = c["int8_working_point"], n["int8_working_point"]
    if not cv.get("eligible") or not nv.get("eligible"):
        return {"passes": False, "checks": {"working_points_exist": False}}
    checks = {
        "candidate_noninferior": c["noninferiority"]["passes"],
        "control_noninferior": n["noninferiority"]["passes"],
        "ap_not_lower": cv.get("average_precision", -1) >= nv.get("average_precision", 0),
        "iou_not_lower": cv.get("overall", {}).get("event", {}).get("iou", -1) >= nv.get("overall", {}).get("event", {}).get("iou", 0),
        "one_strict": (
            cv.get("average_precision", -1) > nv.get("average_precision", 0)
            or cv.get("overall", {}).get("event", {}).get("iou", -1) > nv.get("overall", {}).get("event", {}).get("iou", 0)
        ),
        "high_recall_or_iou_strict_other_not_lower": (
            (cv["high_density"]["event"]["recall"] > nv["high_density"]["event"]["recall"] and cv["high_density"]["event"]["iou"] >= nv["high_density"]["event"]["iou"])
            or (cv["high_density"]["event"]["iou"] > nv["high_density"]["event"]["iou"] and cv["high_density"]["event"]["recall"] >= nv["high_density"]["event"]["recall"])
        ),
    }
    h2 = candidate["h2_r"]
    for file_name in ("train_089.npz", "train_092.npz"):
        checks[f"{file_name}_within_0_3pp"] = (
            cv["per_sequence"][file_name]["event"]["iou"]
            >= h2["per_sequence"][file_name]["event"]["iou"] - 0.003
        )
    return {"passes": all(checks.values()), "checks": checks}


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    preflight_path = args.output / "preflight.json"
    if not preflight_path.exists():
        preflight(preflight_path, require_cuda=not args.smoke, require_gpus=0 if args.smoke else 4)
    train, dev = _load(args.output / "work" / "cache", args.smoke)
    density = fit_density(train)
    density_path = args.output / "work" / "density_transform.json"
    if density_path.exists():
        existing = density_from_json(density_path)
        if existing != density:
            raise RuntimeError("existing Train-main density transform does not match current cache")
    else:
        json_write(density_path, density.as_dict())
    audit_path = args.output / "preflight" / "step4_equivalence.json"
    if not audit_path.exists():
        equivalence_audit(audit_path, train, dev, density, choose_device(args.device, not args.smoke), args.batch_size, args.smoke)
    elif _read(audit_path).get("status") != "pass":
        raise RuntimeError("existing Step4 equivalence audit did not pass")

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    first_jobs = [("A4-1", None), ("A4-1-null", None), ("A4-2", None), ("A4-2-null", None)]
    reports = launch_wave(first_jobs, args.output, args.output / "work" / "cache", density_path, gpu_ids, args.smoke)
    entries = {
        "A4-1": pair_entry(reports["A4-1"], reports["A4-1-null"]),
        "A4-2": pair_entry(reports["A4-2"], reports["A4-2-null"]),
    }

    second_jobs: list[tuple[str, Optional[str]]] = [("A4-3", None), ("A4-3-control", None)]
    if entries["A4-1"]["passes"] and entries["A4-2"]["passes"]:
        second_jobs.extend((("A4-4", None), ("A4-4-null", None)))
    reports.update(launch_wave(second_jobs, args.output, args.output / "work" / "cache", density_path, gpu_ids, args.smoke))
    entries["A4-3"] = _drop_time_entry(reports["A4-3"], reports["A4-3-control"])
    if "A4-4" in reports:
        entries["A4-4"] = pair_entry(reports["A4-4"], reports["A4-4-null"])

    selected_information = None
    eligible_information = [name for name in ("A4-1", "A4-2", "A4-4") if entries.get(name, {}).get("passes")]
    if entries["A4-3"]["passes"] and eligible_information:
        selected_model = max(eligible_information, key=lambda name: primary_rank(reports[name]["final"]["primary"]))
        selected_information = {"A4-1": "density", "A4-2": "history", "A4-4": "both"}[selected_model]
        reports.update(launch_wave(
            [("A4-5", selected_information), ("A4-5-control", selected_information)],
            args.output, args.output / "work" / "cache", density_path, gpu_ids, args.smoke,
        ))
        entries["A4-5"] = pair_entry(reports["A4-5"], reports["A4-5-control"])

    metrics = args.output / "metrics"
    metrics.mkdir(exist_ok=True)
    for name in reports:
        source, target = args.output / "runs" / f"{name}-seed42" / "metrics", metrics / name
        if source.is_dir() and not target.exists():
            shutil.copytree(source, target)
    decision = {
        "schema_version": 1, "protocol": PROTOCOL,
        "status": "smoke_complete" if args.smoke else "complete",
        "validation_used": False, "test_used": False,
        "four_gpu_task_parallelism": not args.smoke,
        "gpu_ids": [] if args.smoke else gpu_ids,
        "runs": {name: str(Path("runs") / f"{name}-seed42" / "summary.json") for name in reports},
        "pair_entry": entries,
        "selected_information_for_A4_5": selected_information,
        "skipped": {
            "A4-4": "requires both A4-1 and A4-2 pair-entry checks" if "A4-4" not in reports else None,
            "A4-5": "requires an effective information feature and A4-3 pair-entry" if "A4-5" not in reports else None,
        },
        "replacement_candidates": [name for name, row in reports.items() if row["final"]["primary"]["replacement_gates"]["passes"]],
        "next_step": "smoke_only" if args.smoke else "review_Step4_before_Step5",
    }
    json_write(args.output / "step4_decision.json", decision, overwrite=(args.output / "step4_decision.json").exists())
    return decision


def continue_pair(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = args.run_root
    density_path = run_root / "work" / "density_transform.json"
    first = _read(run_root / "runs" / f"{args.model}-seed42" / "summary.json")
    control_id = args.model + "-null" if args.model in ("A4-1", "A4-2", "A4-4") else ("A4-3-control" if args.model == "A4-3" else "A4-5-control")
    second = _read(run_root / "runs" / f"{control_id}-seed42" / "summary.json")
    if not first["continue10"]["passes"] or not second["continue10"]["passes"]:
        raise RuntimeError("candidate and matched control must both satisfy the frozen continue10 boundary")
    selected = first.get("selected_information")
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    jobs = []
    for model_id, gpu in ((args.model, gpu_ids[0]), (control_id, gpu_ids[1])):
        source = run_root / "runs" / f"{model_id}-seed42"
        target = run_root / "continuations" / f"{model_id}-seed42-continue10"
        command = [
            sys.executable, str(Path(__file__).resolve()), "train", "--model", model_id,
            "--output", str(target), "--cache", str(run_root / "work" / "cache"),
            "--density", str(density_path), "--device", "cuda", "--seed", "42", "--epochs", "10",
            "--batch-size", "4096", "--resume-checkpoint", str(source / "epoch_010.pt"),
            "--epoch-offset", "10", "--run",
        ]
        if selected:
            command.extend(("--selected-information", selected))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        jobs.append((model_id, subprocess.Popen(command, cwd=Path(__file__).resolve().parent, env=environment)))
    failures = [name for name, process in jobs if process.wait()]
    if failures:
        raise RuntimeError("continuation failed: " + ", ".join(failures))
    result = {"status": "complete", "models": [name for name, _ in jobs], "validation_used": False, "test_used": False}
    json_write(run_root / "continuations" / f"{args.model}_pair_decision.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--require-cuda", action="store_true")
    check.add_argument("--require-gpus", type=int, default=0)
    train = sub.add_parser("train")
    train.add_argument("--model", choices=tuple(MODEL_DESCRIPTIONS), required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--cache", type=Path, default=Path("work/cache"))
    train.add_argument("--density", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--selected-information", choices=("density", "history", "both"))
    train.add_argument("--resume-checkpoint", type=Path)
    train.add_argument("--epoch-offset", type=int, default=0)
    train.add_argument("--run", action="store_true")
    train.add_argument("--smoke", action="store_true")
    all_parser = sub.add_parser("run-all")
    all_parser.add_argument("--output", type=Path, required=True)
    all_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    all_parser.add_argument("--gpus", default="0,1,2,3")
    all_parser.add_argument("--batch-size", type=int, default=4096)
    all_parser.add_argument("--run", action="store_true")
    all_parser.add_argument("--smoke", action="store_true")
    all_parser.add_argument("--resume", action="store_true")
    continuation = sub.add_parser("continue-pair")
    continuation.add_argument("--run-root", type=Path, required=True)
    continuation.add_argument("--model", choices=("A4-1", "A4-2", "A4-3", "A4-4", "A4-5"), required=True)
    continuation.add_argument("--gpus", default="0,1")
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "run") and args.run == args.smoke:
        raise SystemExit("choose exactly one of --run or --smoke")
    if hasattr(args, "batch_size") and args.batch_size != 4096:
        raise SystemExit("the frozen Step4 batch size is 4096")
    if args.command == "preflight":
        result = preflight(args.output, args.require_cuda, args.require_gpus)
    elif args.command == "train":
        train, dev = _load(args.cache, args.smoke)
        result = train_candidate(
            args.model, args.output, train, dev, density_from_json(args.density),
            choose_device(args.device, args.run), args.seed, args.epochs, args.batch_size,
            args.smoke, args.selected_information, args.resume_checkpoint, args.epoch_offset,
        )
    elif args.command == "continue-pair":
        result = continue_pair(args)
    else:
        result = run_all(args)
    if args.command == "train":
        result = {
            "status": result["status"], "model_id": result["model_id"],
            "decision": result.get("decision"), "primary_checkpoint": result.get("primary_checkpoint"),
            "continue10": result.get("continue10"), "validation_used": False, "test_used": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
