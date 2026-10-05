#!/usr/bin/env python3
"""Run Step6 three-seed stability without exposing Validation or Test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step4"):
    if (candidate / "step4_common.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from step4_common import (  # noqa: E402
    RUNTIME_ROOT, build_model, density_from_json, fit_density, joined_features,
    json_write, load_records, predict_int8, role_paths, sha256, working_point_view,
)
from step4_training import load_checkpoint  # noqa: E402
from step6_common import (  # noqa: E402
    BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, FIXED_EPOCH, FIXED_THRESHOLD,
    PROTOCOL, TRAIN_SEEDS, aggregate_decision, paired_bootstrap,
    seed_report, sensitivity_report,
)
from step6_training import train_fixed_seed  # noqa: E402
from wangdao.h2_research import FrozenH2Pipeline, augment_refiner_features  # noqa: E402
from wangdao.phase1_common import load_sequence  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_paths(step5_results: Path, step4_results: Path) -> Dict[str, Path]:
    return {
        "frozen": step5_results / "frozen_candidate.json",
        "candidate_summary": step5_results / "metrics" / "A4-2" / "int8" / "summary.json",
        "candidate_scores": step5_results / "metrics" / "A4-2" / "int8" / "bootstrap_event_scores.npz",
        "h2_summary": step5_results / "metrics" / "H2-R" / "int8" / "summary.json",
        "h2_scores": step5_results / "metrics" / "H2-R" / "int8" / "bootstrap_event_scores.npz",
        "seed42_checkpoint": step5_results / "artifacts" / "epoch_008.pt",
        "null_summary": step4_results / "runs" / "A4-2-null-seed42" / "summary.json",
        "density": step4_results / "work" / "density_transform.json",
    }


def choose_device(name: str, formal: bool) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        if formal:
            raise RuntimeError("formal Step6 training requires CUDA")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if formal:
        raise RuntimeError("formal Step6 training requires CUDA")
    return torch.device("cpu")


def _null_seed42_view(summary: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in summary["evaluations"] if int(row["epoch"]) == FIXED_EPOCH]
    if len(rows) != 1:
        raise RuntimeError("A4-2-null seed42 epoch8 is not unique")
    return working_point_view(rows[0]["int8"], {"eligible": True, "threshold": FIXED_THRESHOLD})


def preflight(
    step5_results: Path, step4_results: Path, output: Path,
    require_cuda: bool, require_gpus: int,
) -> Dict[str, Any]:
    sources = source_paths(step5_results, step4_results)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Step6 source artifact missing: " + ", ".join(missing))
    frozen = read_json(sources["frozen"])
    null = _null_seed42_view(read_json(sources["null_summary"]))
    checks = {
        "frozen_protocol": frozen.get("protocol") == "h2-ann-step5-history-freeze-v1",
        "frozen_identity": frozen.get("model_id") == "A4-2" and frozen.get("seed") == 42,
        "fixed_epoch": int(frozen.get("epoch", -1)) == FIXED_EPOCH,
        "fixed_threshold": float(frozen.get("threshold", -1)) == FIXED_THRESHOLD,
        "checkpoint_hash": sha256(sources["seed42_checkpoint"]) == frozen.get("checkpoint_sha256"),
        "h2_hash": sha256(RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz") == frozen.get("proposal_sha256"),
        "index_hash": sha256(RUNTIME_ROOT / "data" / "index.json") == frozen.get("index_sha256"),
        "matched_null_epoch8_threshold040": abs(null["average_precision"] - frozen["matched_null_metrics"]["average_precision"]) < 1e-15,
        "train_main_count": len(role_paths("train_main")) == 79,
        "train_dev_count": len(role_paths("train_dev")) == 10,
        "cuda_requirement": (not require_cuda) or torch.cuda.is_available(),
        "gpu_requirement": (not require_gpus) or (torch.cuda.is_available() and torch.cuda.device_count() >= require_gpus),
        "validation_and_test_unreachable": True,
    }
    report = {
        "schema_version": 1, "protocol": PROTOCOL,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks, "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "validation_used": False, "test_used": False,
    }
    json_write(output, report)
    if report["status"] != "pass":
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("Step6 preflight failed: " + ", ".join(failed))
    return report


def _load(cache: Path, smoke: bool) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    _, train = load_records("train_main", cache, limit=4 if smoke else None)
    _, dev = load_records("train_dev", cache)
    return train, dev


def density_compatibility(
    records: Sequence[Mapping[str, Any]], frozen: Any,
) -> Dict[str, Any]:
    """Compare the recomputed transform by its effective uint8 encoding.

    Exact dataclass equality is not portable across NumPy/libm versions because
    the log1p extrema can differ in the last floating-point bits.  Step6 trains
    only the history and history-null models, but the frozen density metadata is
    still checked as a data/cache integrity sentinel.  Counts must match exactly
    and every Train-main candidate must receive the same uint8 density code.
    """
    actual = fit_density(records)
    count_match = actual.train_candidates == frozen.train_candidates
    sequence_match = actual.train_sequences == frozen.train_sequences
    quantized_equal = True
    differing_sequences = []
    for record in records:
        rates = np.asarray(record["window_rate_eps"], dtype=np.float64)
        if not np.array_equal(actual.apply(rates), frozen.apply(rates)):
            quantized_equal = False
            differing_sequences.append(str(record.get("file", "unknown")))
    checks = {
        "train_candidates_exact": bool(count_match),
        "train_sequences_exact": bool(sequence_match),
        "uint8_density_codes_identical": bool(quantized_equal),
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "passes": bool(all(checks.values())),
        "checks": checks,
        "exact_dataclass_match": bool(actual == frozen),
        "actual": actual.as_dict(),
        "frozen": frozen.as_dict(),
        "delta_log1p_min": float(actual.log1p_min - frozen.log1p_min),
        "delta_log1p_max": float(actual.log1p_max - frozen.log1p_max),
        "differing_sequence_count": len(differing_sequences),
        "differing_sequences": differing_sequences,
        "note": "A4-2 and A4-2-null use history inputs, not the density column.",
    }


def _complete(path: Path, smoke: bool) -> Optional[Dict[str, Any]]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return None
    report = read_json(summary_path)
    expected = "smoke_complete" if smoke else "complete"
    if report.get("step6_status") != expected or bool(report.get("smoke")) != bool(smoke):
        raise RuntimeError(f"existing Step6 run is incomplete or mode-mismatched: {path}")
    if report.get("validation_used") is not False or report.get("test_used") is not False:
        raise RuntimeError(f"data-role isolation cannot be verified: {path}")
    fixed = report.get("step6_fixed", {})
    if not smoke and (int(fixed.get("evaluated_epoch", -1)) != FIXED_EPOCH or float(fixed.get("fixed_threshold", -1)) != FIXED_THRESHOLD):
        raise RuntimeError(f"existing run violates frozen epoch/threshold: {path}")
    return report


def _stream(pipe: Any, log: Any, prefix: str) -> None:
    for line in iter(pipe.readline, ""):
        log.write(line); log.flush()
        print(f"[{prefix}] {line}", end="", flush=True)
    pipe.close()


def launch_jobs(
    output: Path, cache: Path, density: Path, gpu_ids: Sequence[str],
    batch_size: int, smoke: bool, resume: bool,
) -> Dict[str, Dict[str, Any]]:
    tasks = (("A4-2", 43), ("A4-2-null", 43), ("A4-2", 44), ("A4-2-null", 44))
    reports: Dict[str, Dict[str, Any]] = {}
    pending = []
    for index, (model_id, seed) in enumerate(tasks):
        key = f"{model_id}-seed{seed}"
        run_dir = output / "runs" / key
        existing = None
        if run_dir.exists():
            try:
                existing = _complete(run_dir, smoke)
            except RuntimeError:
                if not resume:
                    raise
            if existing is None and not resume:
                raise RuntimeError(f"existing Step6 run is incomplete: {run_dir}")
            if existing is None and resume:
                failed_root = output / "failed_attempts"
                failed_root.mkdir(exist_ok=True)
                index_retry = 1
                archived = failed_root / f"{key}-attempt{index_retry}"
                while archived.exists():
                    index_retry += 1
                    archived = failed_root / f"{key}-attempt{index_retry}"
                shutil.move(str(run_dir), str(archived))
                print(f"[{key}] preserved incomplete attempt at {archived}", flush=True)
        if existing is not None:
            reports[key] = existing
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), "train",
            "--model", model_id, "--seed", str(seed), "--output", str(run_dir),
            "--cache", str(cache), "--density", str(density),
            "--device", "cpu" if smoke else "cuda", "--batch-size", str(batch_size),
            "--smoke" if smoke else "--run",
        ]
        environment = os.environ.copy()
        if not smoke:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[index])
        log_path = output / "logs" / f"{key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8", newline="\n")
        process = subprocess.Popen(
            command, cwd=HERE, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        thread = threading.Thread(target=_stream, args=(process.stdout, log, key), daemon=True)
        thread.start()
        pending.append((key, run_dir, process, thread, log, log_path))
        if smoke:
            process.wait(); thread.join(); log.close()
            if process.returncode:
                raise RuntimeError(f"{key} smoke failed; inspect {log_path}")
            reports[key] = _complete(run_dir, smoke)  # type: ignore[assignment]
            pending.clear()
    failures = []
    for key, run_dir, process, thread, log, log_path in pending:
        process.wait(); thread.join(); log.close()
        if process.returncode:
            failures.append(f"{key} ({log_path})")
        else:
            reports[key] = _complete(run_dir, smoke)  # type: ignore[assignment]
    if failures:
        raise RuntimeError("parallel Step6 jobs failed: " + ", ".join(failures))
    return reports


def _full_pipeline_latency(checkpoint: Path, device: torch.device, batch_size: int) -> Dict[str, Any]:
    package = load_checkpoint(checkpoint)
    model = build_model("A4-2", 42).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    pipeline = FrozenH2Pipeline(RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz", batch_size=batch_size)
    rows = []
    for path in role_paths("train_dev"):
        started_io = time.perf_counter()
        events, _, _, _ = load_sequence(path)
        inputs = events[["x", "y", "t", "p"]].copy()
        io_s = time.perf_counter() - started_io
        started = time.perf_counter()
        prepared = pipeline.prepare(inputs)
        features = augment_refiner_features(prepared.features, prepared.history)
        refiner_started = time.perf_counter()
        probability = predict_int8(model, features)
        refiner_s = time.perf_counter() - refiner_started
        compute_s = time.perf_counter() - started
        rows.append({
            "file": path.name, "events": len(events), "candidates": len(probability),
            "io_s": io_s, "compute_s": compute_s, "total_with_io_s": io_s + compute_s,
            "new_refiner_int8_s": refiner_s, "pipeline_stages_s": prepared.timings,
        })
    compute = np.asarray([row["compute_s"] for row in rows])
    total = np.asarray([row["total_with_io_s"] for row in rows])
    return {
        "scope": "complete in-memory H2 Proposal, routing, causal history and A4-2 int8 Refiner; file I/O also reported separately",
        "sequences": rows,
        "compute_mean_s": float(np.mean(compute)), "compute_p95_s": float(np.percentile(compute, 95)),
        "with_io_mean_s": float(np.mean(total)), "with_io_p95_s": float(np.percentile(total, 95)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history_buffer": {
            "two_34x44_heatmaps": True,
            "software_float32_bytes": 2 * 34 * 44 * 4,
            "deployment_uint8_target_bytes": 2 * 34 * 44,
        },
    }


def _formal_analysis(
    output: Path, step5_results: Path, step4_results: Path,
    reports: Mapping[str, Mapping[str, Any]], device: torch.device, batch_size: int,
) -> Dict[str, Any]:
    sources = source_paths(step5_results, step4_results)
    frozen = read_json(sources["frozen"])
    candidate42 = read_json(sources["candidate_summary"])["working_point"]
    h2 = read_json(sources["h2_summary"])["working_point"]
    h2_f, h2_r = frozen["h2_f"], frozen["h2_r"]
    null42 = _null_seed42_view(read_json(sources["null_summary"]))
    rows = [seed_report(42, candidate42, null42, h2_f, h2_r)]
    statistics = output / "statistics"
    statistics.mkdir(exist_ok=True)
    for seed in TRAIN_SEEDS:
        candidate_run = reports[f"A4-2-seed{seed}"]
        null_run = reports[f"A4-2-null-seed{seed}"]
        candidate = candidate_run["step6_fixed"]["int8_working_point"]
        control = null_run["step6_fixed"]["int8_working_point"]
        rows.append(seed_report(seed, candidate, control, h2_f, h2_r))
    for row in rows:
        seed = int(row["seed"])
        candidate_view = candidate42 if seed == 42 else reports[f"A4-2-seed{seed}"]["step6_fixed"]["int8_working_point"]
        score_path = sources["candidate_scores"] if seed == 42 else output / "runs" / f"A4-2-seed{seed}" / "fixed_epoch_008_threshold_040" / "metrics" / "int8" / "bootstrap_event_scores.npz"
        bootstrap = paired_bootstrap(
            score_path, sources["h2_scores"], candidate_view, h2,
            BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED,
        )
        sensitivity = sensitivity_report(score_path, sources["h2_scores"], candidate_view, h2)
        json_write(statistics / f"seed{seed}_bootstrap.json", bootstrap)
        json_write(statistics / f"seed{seed}_sensitivity.json", sensitivity)
        row["bootstrap"] = str(Path("statistics") / f"seed{seed}_bootstrap.json")
        row["sensitivity"] = str(Path("statistics") / f"seed{seed}_sensitivity.json")
    decision = aggregate_decision(rows)
    latency = _full_pipeline_latency(sources["seed42_checkpoint"], device, batch_size)
    json_write(output / "latency_and_footprint.json", latency)
    return {
        "schema_version": 1, "protocol": PROTOCOL, "status": "complete",
        "fixed_model": "A4-2", "deployment_seed": 42,
        "fixed_epoch": FIXED_EPOCH, "fixed_threshold": FIXED_THRESHOLD,
        "seed_reports": rows, "aggregate_decision": decision,
        "step6_passes": bool(decision["passes"]),
        "next_step": "Step7_validation_once" if decision["passes"] else "stop_and_keep_H2",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
        "latency_and_footprint": "latency_and_footprint.json",
        "validation_used": False, "test_used": False,
    }


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    preflight_path = args.output / "preflight.json"
    if not preflight_path.exists():
        preflight(
            args.step5_results, args.step4_results, preflight_path,
            require_cuda=not args.smoke, require_gpus=0 if args.smoke else 4,
        )
    train, dev = _load(args.output / "work" / "cache", args.smoke)
    density_path = source_paths(args.step5_results, args.step4_results)["density"]
    density = density_from_json(density_path)
    density_report = density_compatibility(train, density)
    density_report_path = args.output / "density_compatibility.json"
    json_write(density_report_path, density_report, overwrite=density_report_path.exists())
    if not density_report["passes"] and not args.smoke:
        raise RuntimeError(
            "Train-main density transform is not uint8-equivalent to the frozen "
            "Step4 transform; see density_compatibility.json"
        )
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not args.smoke and (len(gpu_ids) != 4 or len(set(gpu_ids)) != 4):
        raise ValueError("formal Step6 requires exactly four distinct GPU ids")
    reports = launch_jobs(
        args.output, args.output / "work" / "cache", density_path,
        gpu_ids, args.batch_size, args.smoke, args.resume,
    )
    if args.smoke:
        decision = {
            "schema_version": 1, "protocol": PROTOCOL, "status": "smoke_complete",
            "runs": {name: str(Path("runs") / name / "summary.json") for name in reports},
            "formal_decision_performed": False, "validation_used": False,
            "test_used": False, "next_step": "run_formal_Step6",
        }
    else:
        decision = _formal_analysis(
            args.output, args.step5_results, args.step4_results, reports,
            choose_device(args.device, True), args.batch_size,
        )
        metrics = args.output / "metrics"
        metrics.mkdir(exist_ok=True)
        for name in reports:
            source = args.output / "runs" / name / "fixed_epoch_008_threshold_040" / "metrics"
            target = metrics / name
            if not target.exists():
                shutil.copytree(source, target)
    json_write(args.output / "step6_decision.json", decision, overwrite=(args.output / "step6_decision.json").exists())
    return decision


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--step5-results", type=Path, required=True)
    check.add_argument("--step4-results", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--require-cuda", action="store_true")
    check.add_argument("--require-gpus", type=int, default=0)
    train = sub.add_parser("train")
    train.add_argument("--model", choices=("A4-2", "A4-2-null"), required=True)
    train.add_argument("--seed", type=int, choices=TRAIN_SEEDS, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--density", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--run", action="store_true")
    train.add_argument("--smoke", action="store_true")
    all_parser = sub.add_parser("run-all")
    all_parser.add_argument("--step5-results", type=Path, required=True)
    all_parser.add_argument("--step4-results", type=Path, required=True)
    all_parser.add_argument("--output", type=Path, required=True)
    all_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    all_parser.add_argument("--gpus", default="0,1,2,3")
    all_parser.add_argument("--batch-size", type=int, default=4096)
    all_parser.add_argument("--run", action="store_true")
    all_parser.add_argument("--smoke", action="store_true")
    all_parser.add_argument("--resume", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command in ("train", "run-all") and getattr(args, "run", False) == getattr(args, "smoke", False):
        raise SystemExit("choose exactly one of --run or --smoke")
    if args.command == "preflight":
        result = preflight(args.step5_results, args.step4_results, args.output, args.require_cuda, args.require_gpus)
    elif args.command == "train":
        device = choose_device(args.device, args.run)
        train, dev = _load(args.cache, args.smoke)
        result = train_fixed_seed(
            args.model, args.output, train, dev, density_from_json(args.density),
            device, args.seed, args.batch_size, args.smoke,
        )
    else:
        result = run_all(args)
    if args.command == "train":
        fixed = result.get("step6_fixed", {})
        view = fixed.get("int8_working_point", {})
        result = {
            "status": result.get("step6_status"), "model_id": result.get("model_id"),
            "seed": result.get("seed"),
            "evaluated_epoch": fixed.get("evaluated_epoch"),
            "fixed_threshold": fixed.get("fixed_threshold"),
            "int8_ap": view.get("average_precision"),
            "int8_iou": view.get("overall", {}).get("event", {}).get("iou"),
            "int8_high_recall": view.get("high_density", {}).get("event", {}).get("recall"),
            "noninferiority_pass": fixed.get("noninferiority", {}).get("passes"),
            "validation_used": False, "test_used": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
