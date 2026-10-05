"""Steps 2-3: train H2 A1/A2 or paired A3/A3-null Refiner variants."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from wangdao.runtime import choose_device, environment, set_seed
from wangdao.h2.losses import refiner_loss
from wangdao.h2.torch_model import prototype_logits
from wangdao.h2_research import (
    H2_RESEARCH_PROTOCOL,
    FrozenH2Pipeline,
    augment_refiner_features,
    build_h2_refiner,
    hard_random_sample,
    select_fair_working_point,
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


def model_features(data, variant):
    if variant == "A3":
        return augment_refiner_features(data["features"], data["history"])
    if variant == "A3-null":
        return augment_refiner_features(data["features"], np.zeros_like(data["history"]))
    return data["features"]


def class_focal(raw_logits, labels):
    probability = torch.softmax(prototype_logits(raw_logits), dim=1)[:, 1].clamp(1.0e-6, 1.0 - 1.0e-6)
    target = labels.to(torch.float32)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    alpha = torch.where(target > 0.5, 0.75, 0.25)
    values = -alpha * torch.pow(1.0 - pt, 2.0) * torch.log(pt)
    positive = target > 0.5
    negative = ~positive
    return {
        "positive_sum": float(values[positive].sum().detach().cpu()),
        "positive_count": int(positive.sum().detach().cpu()),
        "negative_sum": float(values[negative].sum().detach().cpu()),
        "negative_count": int(negative.sum().detach().cpu()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["A1", "A2", "A3", "A3-null"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/runs/h2_candidate_cache_v2")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs must be 1..20; batch size and threads must be positive")
    if args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    torch.set_num_threads(args.threads)
    smoke = not args.run
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
        "history": "previous 1/2 legacy windows; same-cell and 3x3-mean scores",
    }
    cache = CandidateDiskCache(args.cache_dir, protocol)
    model = build_h2_refiner(args.variant, pipeline.h2, args.seed).to(device)
    recipe = {
        "variant": args.variant,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": 1.0e-4,
        "loss": "focal+0.5soft_iou",
        "sampling": "all_candidates" if args.variant != "A2" else "all_positive+3x_negative_half_hard_half_random",
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
        "stage": "step2_h2_adaptation" if args.variant in ("A1", "A2") else "step3_minimal_history",
        "data_role": "train_main+train_dev",
        "primary_metrics": {},
        "decision": "training",
        "conclusion": "No measured gain yet.",
        "smoke": smoke,
        "test_used": False,
        "validation_used": False,
        "recipe": recipe,
        "recipe_id": recipe_id,
        "environment": environment(device),
        "seed": args.seed,
        "provenance": protocol,
        "metric_protocol": PROTOCOL_ID,
        "history": [],
        "train_sequences": [path.name for path in train_paths],
        "dev_sequences": [path.name for path in dev_paths],
        "selection_protocol": "checkpoint: full-event Train-dev IoU@0.28; baseline and candidate use the same Pd/Fa-constrained grid",
        "numeric_mode": "floating_refiner_with_frozen_exact_H2_proposal",
        "formal_configuration_budget_used": not smoke,
    }
    write_json(args.output_dir / "summary.json", report)

    def build(path):
        events, loc, labels, identities = load_sequence(path)
        inputs = events[["x", "y", "t", "p"]].copy()
        prepared = pipeline.prepare(inputs)
        baseline = pipeline.infer_prepared(prepared, len(events))["candidate_probabilities"]
        return {
            "features": prepared.features,
            "history": prepared.history,
            "indices": prepared.indices,
            "baseline_probability": baseline,
            "labels": labels,
            "identities": identities,
            "loc": loc,
        }

    def evaluate(thresholds, baseline=False, save=False):
        rows = {str(threshold): [] for threshold in thresholds}
        details = []
        destination = args.output_dir / "dev_predictions"
        if save:
            destination.mkdir(exist_ok=False)
        for path in dev_paths:
            data = cache.get(path, build)
            candidate_probability = (
                data["baseline_probability"]
                if baseline
                else predict_refiner(model, model_features(data, args.variant), args.batch_size, str(device))
            )
            full = full_probability(data["indices"], candidate_probability, len(data["labels"]))
            scores = {}
            for threshold in thresholds:
                score = score_predictions(data["loc"], data["labels"], data["identities"], full, threshold)
                rows[str(threshold)].append(score)
                scores[str(threshold)] = score
            details.append({
                "file": path.name,
                "events": len(full),
                "candidate_events": len(candidate_probability),
                "scores": scores,
            })
            if save:
                with (destination / path.name).open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        probabilities=full,
                        event_index=np.arange(len(full), dtype=np.int64),
                        source_sha256=np.asarray(sha256(path)),
                    )
        return {
            "curve": {key: aggregate_scores(value) for key, value in rows.items()},
            "per_sequence": details,
            "sequence_count": len(dev_paths),
            "partial": smoke,
        }

    try:
        report["h2_frozen_reference"] = evaluate(REFERENCE_THRESHOLDS, baseline=True)
        report["initial_float_refiner"] = evaluate(REFERENCE_THRESHOLDS)
        best_iou = -1.0
        best_path = None
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            order = np.random.default_rng(args.seed + epoch - 1).permutation(len(train_paths))
            total_loss = 0.0
            batches = 0
            samples = 0
            selected_positive = 0
            selected_negative = 0
            focal = {"positive_sum": 0.0, "positive_count": 0,
                     "negative_sum": 0.0, "negative_count": 0}
            model.train()
            for position in order:
                data = cache.get(train_paths[int(position)], build)
                features = model_features(data, args.variant)
                labels = data["labels"][data["indices"]].astype(np.int64)
                if args.variant == "A2":
                    permutation = hard_random_sample(
                        labels,
                        data["baseline_probability"],
                        seed=args.seed * 100000 + epoch * 100 + int(position),
                    )
                else:
                    permutation = np.random.default_rng(
                        args.seed * 100000 + epoch * 100 + int(position)
                    ).permutation(len(features))
                if smoke:
                    permutation = permutation[:min(len(permutation), 8192)]
                selected_positive += int(np.sum(labels[permutation] == 1))
                selected_negative += int(np.sum(labels[permutation] == 0))
                for first in range(0, len(permutation), args.batch_size):
                    selected = permutation[first:first + args.batch_size]
                    x = torch.from_numpy(features[selected].copy()).to(device)
                    y = torch.as_tensor(labels[selected].copy(), dtype=torch.float32, device=device)
                    raw = model(x)
                    zeros = torch.zeros_like(y)
                    loss = refiner_loss(raw, y, zeros, zeros, distillation_weight=0.0)["total"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite Refiner loss")
                    diagnostics = class_focal(raw, y)
                    for key in focal:
                        focal[key] += diagnostics[key]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    batches += 1
                    samples += len(selected)
                print(f"[{args.variant} train {epoch}/{epochs}] {train_paths[int(position)].name}", flush=True)
            dev = evaluate(REFERENCE_THRESHOLDS)
            iou = dev["curve"][str(0.2800000011920929)]["event"]["iou"]
            selected_checkpoint = iou is not None and iou > best_iou
            if selected_checkpoint:
                best_iou = float(iou)
                best_path = args.output_dir / f"epoch_{epoch:03d}.pt"
                with best_path.open("xb") as stream:
                    torch.save({
                        "format_version": 2,
                        "variant": args.variant,
                        "dimensions": tuple(model.dimensions),
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "metadata": {"smoke": smoke, "seed": args.seed, "epoch": epoch, **protocol},
                    }, stream)
                report.update(best_checkpoint=best_path.name, best_epoch=epoch)
            history_gradient = None
            if args.variant in ("A3", "A3-null") and model.layers[0].weight.grad is not None:
                history_gradient = float(model.layers[0].weight.grad[:, 32:].norm().detach().cpu())
            report["history"].append({
                "epoch": epoch,
                "loss": total_loss / max(batches, 1),
                "batches": batches,
                "samples": samples,
                "selected_positive": selected_positive,
                "selected_negative": selected_negative,
                "positive_focal_mean": focal["positive_sum"] / focal["positive_count"] if focal["positive_count"] else None,
                "negative_focal_mean": focal["negative_sum"] / focal["negative_count"] if focal["negative_count"] else None,
                "history_input_gradient_norm_last_batch": history_gradient,
                "dev": dev,
                "selected": selected_checkpoint,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "wall_s": time.perf_counter() - started,
                "cache": dict(cache.statistics),
            })
            scheduler.step()
            write_json(args.output_dir / "summary.json", report, overwrite=True)
        if best_path is None:
            raise RuntimeError("no finite Train-dev checkpoint score")
        package = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(package["state_dict"], strict=True)
        baseline_final = evaluate(THRESHOLD_GRID, baseline=True)
        candidate_final = evaluate(THRESHOLD_GRID, save=True)
        anchor = baseline_final["curve"][str(0.2800000011920929)]
        baseline_selection = select_fair_working_point(
            baseline_final["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]
        )
        candidate_selection = select_fair_working_point(
            candidate_final["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]
        )
        selected_delta = None
        if baseline_selection["eligible"] and candidate_selection["eligible"]:
            selected_delta = (
                candidate_selection["metrics"]["event"]["iou"]
                - baseline_selection["metrics"]["event"]["iou"]
            )
        report.update(
            status="complete",
            decision="smoke_only" if smoke else "development_experiment_complete",
            conclusion=(
                "Smoke verifies execution only; no accuracy claim."
                if smoke else
                "Train-dev experiment complete; multiseed and frozen evaluation are still required."
            ),
            final_evaluation=candidate_final,
            h2_baseline_final=baseline_final,
            h2_fair_working_point=baseline_selection,
            candidate_fair_working_point=candidate_selection,
            selected_iou_delta_vs_h2_r=selected_delta,
            primary_metrics=candidate_final["curve"][str(0.2800000011920929)],
            best_checkpoint_sha256=sha256(best_path),
            cache=dict(cache.statistics),
        )
        write_json(args.output_dir / "summary.json", report, overwrite=True)
        print("Saved:", args.output_dir / "summary.json", flush=True)
    except Exception as error:
        report.update(status="failed", decision="stop", error=type(error).__name__ + ": " + str(error))
        write_json(args.output_dir / "summary.json", report, overwrite=True)
        raise


if __name__ == "__main__":
    main()
