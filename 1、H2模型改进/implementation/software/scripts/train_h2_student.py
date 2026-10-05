"""Step 4: train direct or logit-distilled shallow H2 Refiner students."""

from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from wangdao.runtime import choose_device, environment, set_seed
from wangdao.h2.torch_model import QuantizedMLP
from wangdao.h2_research import (
    H2_RESEARCH_PROTOCOL,
    FrozenH2Pipeline,
    augment_refiner_features,
    build_h2_refiner,
    hard_random_sample,
    select_fair_working_point,
)
from wangdao.h2_students import (
    build_student,
    complexity_summary,
    export_student_artifact,
    save_training_checkpoint,
    student_objective,
)
from wangdao.phase1_common import (
    ROOT,
    PROTOCOL_ID,
    THRESHOLD_GRID,
    aggregate_scores,
    load_sequence,
    register_experiment,
    role_paths,
    score_predictions,
    sha256,
    write_json,
)
from wangdao.h2_refiner import CandidateDiskCache, full_probability, predict_refiner


REFERENCE_THRESHOLDS = (0.2800000011920929, 0.9)


def model_features(data, feature_variant):
    if feature_variant == "A3":
        return augment_refiner_features(data["features"], data["history"])
    if feature_variant == "A3-null":
        return augment_refiner_features(data["features"], np.zeros_like(data["history"]))
    return data["features"]


def load_teacher(summary_path: Path, device: torch.device, expected_variant: str):
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("smoke", True):
        raise ValueError("formal teacher must be a complete non-smoke experiment")
    if report.get("data_role") != "train_main+train_dev" or report.get("test_used") or report.get("validation_used"):
        raise ValueError("teacher must be selected only from Train-main/Train-dev")
    if report.get("metric_protocol") != PROTOCOL_ID or report.get("recipe", {}).get("variant") != expected_variant:
        raise ValueError("teacher protocol or feature variant mismatch")
    selected = report.get("candidate_fair_working_point", {})
    if not selected.get("eligible") or selected.get("metrics") is None:
        raise ValueError("teacher has no eligible common-constraint working point")
    checkpoint = summary_path.parent / report["best_checkpoint"]
    if sha256(checkpoint) != report.get("best_checkpoint_sha256"):
        raise ValueError("teacher checkpoint hash mismatch")
    package = torch.load(checkpoint, map_location=device, weights_only=True)
    dimensions = tuple(int(value) for value in package["dimensions"])
    model = QuantizedMLP(dimensions, quantized=False).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    metadata = package.get("metadata", {})
    if metadata.get("smoke", True) or package.get("variant") != expected_variant:
        raise ValueError("teacher checkpoint metadata mismatch")
    model.eval().requires_grad_(False)
    return model, report, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=["S-R1", "S-R2"], required=True)
    parser.add_argument("--mode", choices=["direct", "kd"], required=True)
    parser.add_argument("--feature-variant", choices=["A1", "A2", "A3", "A3-null"], required=True)
    parser.add_argument("--teacher-summary", type=Path,
                        help="Selected non-smoke A* summary; required for every formal run")
    parser.add_argument("--output-dir", type=Path, required=True)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--run", action="store_true")
    run_mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=0.2)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/runs/h2_candidate_cache_v2")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    smoke = not args.run
    if not smoke and args.teacher_summary is None:
        parser.error("formal Step-4 runs require --teacher-summary for the selected A*")
    if not 1 <= args.epochs <= 20 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs must be 1..20; batch size and threads must be positive")
    if args.learning_rate <= 0 or args.temperature <= 0 or not 0 <= args.kd_weight <= 1:
        parser.error("learning rate/temperature must be positive and kd-weight must be in [0,1]")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = choose_device(args.device)
    train_paths = role_paths("train_main", limit=1 if smoke else None)
    dev_paths = role_paths("train_dev", limit=1 if smoke else None)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    h2_path = ROOT / "software/weights/H2_D1_seed42.npz"
    pipeline = FrozenH2Pipeline(h2_path)
    protocol = {
        "protocol": H2_RESEARCH_PROTOCOL,
        "h2_sha256": sha256(h2_path),
        "index_sha256": sha256(ROOT / "data/index.json"),
        "proposal_threshold": pipeline.h2.proposal_threshold,
        "feature_variant": args.feature_variant,
    }
    cache = CandidateDiskCache(args.cache_dir, protocol)
    model = build_student(args.student, args.feature_variant, args.seed).to(device)
    teacher_report = None
    teacher_path = None
    if args.teacher_summary is not None:
        teacher, teacher_report, teacher_path = load_teacher(
            args.teacher_summary, device, args.feature_variant
        )
        teacher_status = "qualified_train_dev_teacher"
    else:
        teacher = build_h2_refiner(args.feature_variant, pipeline.h2, args.seed).to(device)
        teacher.eval().requires_grad_(False)
        teacher_status = "engineering_smoke_only_unqualified_h2_reconstruction"
    recipe = {
        "student_variant": args.student,
        "mode": args.mode,
        "feature_variant": args.feature_variant,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": 1.0e-4,
        "temperature": args.temperature,
        "kd_weight": args.kd_weight if args.mode == "kd" else 0.0,
        "sampling": "A2 hard/random" if args.feature_variant == "A2" else "all_candidates",
        "candidate_protocol": protocol,
    }
    recipe_id = register_experiment(recipe, smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    epochs = 1 if smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    report = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": args.output_dir.name,
        "stage": "step4_shallow_refiner",
        "data_role": "train_main+train_dev",
        "primary_metrics": {},
        "decision": "training",
        "conclusion": "No measured student result yet.",
        "smoke": smoke,
        "test_used": False,
        "validation_used": False,
        "recipe": recipe,
        "recipe_id": recipe_id,
        "seed": args.seed,
        "environment": environment(device),
        "metric_protocol": PROTOCOL_ID,
        "provenance": protocol,
        "teacher": {
            "status": teacher_status,
            "summary": None if args.teacher_summary is None else str(args.teacher_summary.resolve()),
            "summary_sha256": None if args.teacher_summary is None else sha256(args.teacher_summary),
            "checkpoint": None if teacher_path is None else str(teacher_path.resolve()),
            "checkpoint_sha256": None if teacher_path is None else sha256(teacher_path),
        },
        "complexity": complexity_summary(model),
        "train_sequences": [path.name for path in train_paths],
        "dev_sequences": [path.name for path in dev_paths],
        "selection_protocol": "checkpoint: full-event Train-dev IoU@0.28; final baseline/teacher/student share the Pd/Fa-constrained grid",
        "numeric_mode": "floating_refiner_with_frozen_exact_H2_proposal",
        "history": [],
    }
    write_json(args.output_dir / "summary.json", report)

    def build(path):
        events, loc, labels, identities = load_sequence(path)
        inputs = events[["x", "y", "t", "p"]].copy()
        prepared = pipeline.prepare(inputs)
        baseline = pipeline.infer_prepared(prepared, len(events))["candidate_probabilities"]
        return {"features": prepared.features, "history": prepared.history,
                "indices": prepared.indices, "baseline_probability": baseline,
                "labels": labels, "identities": identities, "loc": loc}

    def probabilities(candidate_data, network):
        return predict_refiner(network, model_features(candidate_data, args.feature_variant),
                               args.batch_size, str(device))

    def evaluate(thresholds, network, save=False):
        rows = {str(threshold): [] for threshold in thresholds}
        details = []
        destination = args.output_dir / "dev_predictions"
        if save:
            destination.mkdir(exist_ok=False)
        for path in dev_paths:
            data = cache.get(path, build)
            candidate_probability = probabilities(data, network)
            full = full_probability(data["indices"], candidate_probability, len(data["labels"]))
            scores = {}
            for threshold in thresholds:
                score = score_predictions(data["loc"], data["labels"], data["identities"], full, threshold)
                rows[str(threshold)].append(score)
                scores[str(threshold)] = score
            details.append({"file": path.name, "events": len(full),
                            "candidate_events": len(candidate_probability), "scores": scores})
            if save:
                with (destination / path.name).open("xb") as stream:
                    np.savez_compressed(stream, probabilities=full,
                                        event_index=np.arange(len(full), dtype=np.int64),
                                        source_sha256=np.asarray(sha256(path)))
        return {"curve": {key: aggregate_scores(value) for key, value in rows.items()},
                "per_sequence": details, "sequence_count": len(dev_paths), "partial": smoke}

    try:
        report["teacher_reference"] = evaluate(REFERENCE_THRESHOLDS, teacher)
        best_iou = -1.0
        best_path = None
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            order = np.random.default_rng(args.seed + epoch - 1).permutation(len(train_paths))
            totals = {"loss": 0.0, "task": 0.0, "kd": 0.0, "batches": 0,
                      "samples": 0, "positive": 0, "negative": 0}
            model.train()
            for position in order:
                data = cache.get(train_paths[int(position)], build)
                features = model_features(data, args.feature_variant)
                labels = data["labels"][data["indices"]].astype(np.int64)
                if args.feature_variant == "A2":
                    permutation = hard_random_sample(labels, data["baseline_probability"],
                        seed=args.seed * 100000 + epoch * 100 + int(position))
                else:
                    permutation = np.random.default_rng(
                        args.seed * 100000 + epoch * 100 + int(position)).permutation(len(features))
                if smoke:
                    permutation = permutation[:min(len(permutation), 8192)]
                totals["positive"] += int(np.sum(labels[permutation] == 1))
                totals["negative"] += int(np.sum(labels[permutation] == 0))
                for first in range(0, len(permutation), args.batch_size):
                    selected = permutation[first:first + args.batch_size]
                    x = torch.from_numpy(features[selected].copy()).to(device)
                    y = torch.as_tensor(labels[selected].copy(), dtype=torch.float32, device=device)
                    raw = model(x)
                    with torch.no_grad():
                        teacher_raw = teacher(x) if args.mode == "kd" else None
                    losses = student_objective(raw, y, teacher_raw, args.temperature,
                                               args.kd_weight if args.mode == "kd" else 0.0)
                    if not torch.isfinite(losses["total"]):
                        raise FloatingPointError("nonfinite student loss")
                    optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    totals["loss"] += float(losses["total"].detach().cpu())
                    totals["task"] += float(losses["task"].detach().cpu())
                    totals["kd"] += float(losses["kd"].detach().cpu())
                    totals["batches"] += 1
                    totals["samples"] += len(selected)
                print(f"[{args.student}/{args.mode} {epoch}/{epochs}] {train_paths[int(position)].name}", flush=True)
            dev = evaluate(REFERENCE_THRESHOLDS, model)
            iou = dev["curve"][str(0.2800000011920929)]["event"]["iou"]
            selected_checkpoint = iou is not None and iou > best_iou
            if selected_checkpoint:
                best_iou = float(iou)
                best_path = args.output_dir / f"epoch_{epoch:03d}.pt"
                save_training_checkpoint(best_path, model, {
                    "student_variant": args.student, "feature_variant": args.feature_variant,
                    "mode": args.mode, "seed": args.seed, "epoch": epoch, "smoke": smoke,
                    "data_index_sha256": protocol["index_sha256"],
                    "h2_sha256": protocol["h2_sha256"],
                    "teacher_checkpoint_sha256": report["teacher"]["checkpoint_sha256"],
                })
                report.update(best_checkpoint=best_path.name, best_epoch=epoch)
            report["history"].append({
                "epoch": epoch, "loss": totals["loss"] / max(totals["batches"], 1),
                "task_loss": totals["task"] / max(totals["batches"], 1),
                "kd_loss": totals["kd"] / max(totals["batches"], 1),
                "batches": totals["batches"], "samples": totals["samples"],
                "selected_positive": totals["positive"], "selected_negative": totals["negative"],
                "dev": dev, "selected": selected_checkpoint,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "wall_s": time.perf_counter() - started, "cache": dict(cache.statistics),
            })
            scheduler.step()
            write_json(args.output_dir / "summary.json", report, overwrite=True)
        if best_path is None:
            raise RuntimeError("no finite Train-dev checkpoint score")
        package = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(package["state_dict"], strict=True)
        baseline_rows = {str(value): [] for value in THRESHOLD_GRID}
        for path in dev_paths:
            data = cache.get(path, build)
            full = full_probability(data["indices"], data["baseline_probability"], len(data["labels"]))
            for threshold in THRESHOLD_GRID:
                baseline_rows[str(threshold)].append(score_predictions(
                    data["loc"], data["labels"], data["identities"], full, threshold))
        baseline_final = {"curve": {key: aggregate_scores(rows) for key, rows in baseline_rows.items()}}
        teacher_final = evaluate(THRESHOLD_GRID, teacher)
        student_final = evaluate(THRESHOLD_GRID, model, save=True)
        anchor = baseline_final["curve"][str(0.2800000011920929)]
        selections = {
            "h2": select_fair_working_point(baseline_final["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]),
            "teacher": select_fair_working_point(teacher_final["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]),
            "student": select_fair_working_point(student_final["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]),
        }
        metadata = dict(package["metadata"], checkpoint_sha256=sha256(best_path),
                        event_threshold=selections["student"]["threshold"],
                        metric_protocol=PROTOCOL_ID)
        artifact_path = args.output_dir / "student_float32_v1.npz"
        export_student_artifact(model, artifact_path, metadata)
        student_iou = selections["student"]["metrics"]["event"]["iou"] if selections["student"]["eligible"] else None
        h2_iou = selections["h2"]["metrics"]["event"]["iou"] if selections["h2"]["eligible"] else None
        teacher_iou = selections["teacher"]["metrics"]["event"]["iou"] if selections["teacher"]["eligible"] else None
        report.update(
            status="complete", decision="smoke_only" if smoke else "development_experiment_complete",
            conclusion="Smoke verifies execution only; no accuracy or compression claim." if smoke else
                       "Train-dev student experiment complete; paired direct/KD and multiseed evidence remain required.",
            h2_baseline_final=baseline_final, teacher_final=teacher_final,
            final_evaluation=student_final, fair_working_points=selections,
            selected_iou_delta_vs_h2_r=None if student_iou is None or h2_iou is None else student_iou - h2_iou,
            selected_iou_delta_vs_teacher=None if student_iou is None or teacher_iou is None else student_iou - teacher_iou,
            primary_metrics=student_final["curve"][str(0.2800000011920929)],
            best_checkpoint_sha256=sha256(best_path), artifact=artifact_path.name,
            artifact_sha256=sha256(artifact_path), cache=dict(cache.statistics),
        )
        write_json(args.output_dir / "summary.json", report, overwrite=True)
        print("Saved:", args.output_dir / "summary.json", flush=True)
    except Exception as error:
        report.update(status="failed", decision="stop", error=type(error).__name__ + ": " + str(error))
        write_json(args.output_dir / "summary.json", report, overwrite=True)
        raise


if __name__ == "__main__":
    main()
