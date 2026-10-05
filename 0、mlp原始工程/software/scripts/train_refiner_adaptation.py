"""B1/B2 floating Refiner training on candidates produced by the frozen S1."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wangdao.phase1_common import (ROOT, PROTOCOL_ID, role_paths, load_sequence, write_json,
                                  sha256, register_experiment, score_predictions, aggregate_scores,
                                  select_working_point, THRESHOLD_GRID)
from wangdao.refiner_adaptation import (FrozenS1Pipeline, CandidateDiskCache, build_refiner,
                                       predict_refiner, full_probability)
from wangdao.h2.losses import refiner_loss
from wangdao.causal_training import choose_device, environment, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["B1", "B2"], default="B1")
    p.add_argument("--output-dir", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "results/runs/s1_candidate_cache")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if not 1 <= a.epochs <= 20 or a.batch_size < 1 or a.threads < 1:
        p.error("epochs must be 1..20; batch size and threads positive")
    if a.learning_rate is not None and a.learning_rate <= 0:
        p.error("learning rate must be positive")
    torch.set_num_threads(a.threads)
    smoke = not a.run
    set_seed(a.seed)
    device = choose_device(a.device)
    paths = role_paths("train_main", limit=1 if smoke else None)
    dev_paths = role_paths("train_dev", limit=1 if smoke else None)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    weights = ROOT / "software/weights"
    h2_path, s1_path = weights / "H2_D1_seed42.npz", weights / "S1P_INT8_ACC24_MEM24_Q12.npz"
    pipeline = FrozenS1Pipeline(h2_path, s1_path)
    protocol = {"h2_sha256": sha256(h2_path), "s1_sha256": sha256(s1_path),
                "index_sha256": sha256(ROOT / "data/index.json"), "proposal_threshold": .51}
    cache = CandidateDiskCache(a.cache_dir, protocol)
    model = build_refiner(a.variant, pipeline.h2, a.seed).to(device)
    lr = a.learning_rate if a.learning_rate is not None else (1e-4 if a.variant == "B1" else 1e-3)
    recipe = {"variant": a.variant, "epochs": a.epochs, "lr": lr, "batch_size": a.batch_size,
              "loss": "focal+0.5soft_iou", "candidate_protocol": protocol}
    identity = register_experiment(recipe, smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs = 1 if smoke else a.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    report = {"schema_version": 1, "status": "running", "experiment_id": a.output_dir.name,
              "stage": "phase1_refiner_adaptation", "data_role": "train_main+train_dev",
              "primary_metrics": {}, "decision": "training", "conclusion": "No measured gain yet.",
              "smoke": smoke, "test_used": False, "validation_used": False, "recipe": recipe,
              "recipe_id": identity, "environment": environment(device), "seed": a.seed,
              "provenance": protocol, "metric_protocol": PROTOCOL_ID, "history": [],
              "train_sequences": [path.name for path in paths], "dev_sequences": [path.name for path in dev_paths],
              "numeric_mode": "floating_refiner_with_frozen_integer_S1_frontend",
              "selection_protocol": "checkpoint: full-event Train-dev IoU@0.28; final Pd/Fa-constrained grid",
              "formal_configuration_budget_used": not smoke}
    write_json(a.output_dir / "summary.json", report)

    def build(path):
        events, loc, labels, ids = load_sequence(path)
        inputs = events[["x", "y", "t", "p"]].copy()
        prepared = pipeline.prepare(inputs)
        return {"features": prepared.features, "indices": prepared.indices,
                "labels": labels, "identities": ids, "loc": loc}

    def evaluate(thresholds=(.28, .9), baseline=False, save=False):
        rows = {str(t): [] for t in thresholds}
        details = []
        destination = a.output_dir / "dev_predictions"
        if save:
            destination.mkdir(exist_ok=False)
        for path in dev_paths:
            data = cache.get(path, build)
            probability = (pipeline.fixed_probability(data["features"]) if baseline else
                           predict_refiner(model, data["features"], a.batch_size, str(device)))
            full = full_probability(data["indices"], probability, len(data["labels"]))
            scores = {}
            for threshold in thresholds:
                score = score_predictions(data["loc"], data["labels"], data["identities"], full, threshold)
                rows[str(threshold)].append(score)
                scores[str(threshold)] = score
            details.append({"file": path.name, "events": len(full), "candidate_events": len(probability),
                            "scores": scores})
            if save:
                with (destination / path.name).open("xb") as stream:
                    np.savez_compressed(stream, probabilities=full, event_index=np.arange(len(full)),
                                        source_sha256=np.asarray(sha256(path)))
        return {"curve": {key: aggregate_scores(value) for key, value in rows.items()},
                "per_sequence": details, "sequence_count": len(dev_paths), "partial": smoke}

    try:
        report["baseline"] = evaluate(baseline=True)
        # A floating reconstruction is separately measured; it is not the integer B0.
        report["initial_float_refiner"] = evaluate()
        best = -1.0
        best_path = None
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            order = np.random.default_rng(a.seed + epoch - 1).permutation(len(paths))
            losses, batches, samples = 0., 0, 0
            model.train()
            for position in order:
                data = cache.get(paths[int(position)], build)
                x, y = data["features"], data["labels"][data["indices"]]
                permutation = np.random.default_rng(a.seed * 100000 + epoch * 100 + int(position)).permutation(len(x))
                if smoke:
                    permutation = permutation[:min(len(permutation), 8192)]
                for start in range(0, len(permutation), a.batch_size):
                    selected = permutation[start:start + a.batch_size]
                    features = torch.from_numpy(x[selected].copy()).to(device)
                    labels = torch.as_tensor(y[selected].copy(), dtype=torch.float32, device=device)
                    zeros = torch.zeros_like(labels)
                    loss = refiner_loss(model(features), labels, zeros, zeros, distillation_weight=0.)["total"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite Refiner loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    losses += float(loss.detach().cpu())
                    batches += 1
                    samples += len(selected)
                print(f"[B train {epoch}/{epochs}] {paths[int(position)].name}", flush=True)
            dev = evaluate()
            iou = dev["curve"]["0.28"]["event"]["iou"]
            selected = iou is not None and iou > best
            if selected:
                best = float(iou)
                best_path = a.output_dir / f"epoch_{epoch:03d}.pt"
                with best_path.open("xb") as stream:
                    torch.save({"format_version": 1, "variant": a.variant,
                                "dimensions": tuple(model.dimensions),
                                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                "metadata": {"smoke": smoke, "seed": a.seed, "epoch": epoch, **protocol}}, stream)
                report.update(best_checkpoint=best_path.name, best_epoch=epoch)
            report["history"].append({"epoch": epoch, "loss": losses / max(batches, 1),
                                      "batches": batches, "samples": samples, "dev": dev, "selected": selected,
                                      "learning_rate": optimizer.param_groups[0]["lr"],
                                      "wall_s": time.perf_counter() - started, "cache": dict(cache.statistics)})
            scheduler.step()
            write_json(a.output_dir / "summary.json", report, overwrite=True)
        if best_path is None:
            raise RuntimeError("No finite Train-dev checkpoint score")
        package = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(package["state_dict"], strict=True)
        final = evaluate(THRESHOLD_GRID, save=True)
        baseline = report["baseline"]["curve"]["0.28"]["localization"]
        selection = select_working_point(final["curve"], baseline["pd"], baseline["fa"])
        if smoke:
            selection = {"eligible": False, "threshold": None, "metrics": None, "reason": "smoke_only"}
        report.update(status="complete", decision="smoke_only" if smoke else "refiner_experiment_complete",
                      conclusion="Offline Refiner experiment; not causal or integer-deployment evidence.",
                      final_evaluation=final, selection=selection,
                      primary_metrics=final["curve"][str(0.2800000011920929)],
                      best_checkpoint_sha256=sha256(best_path), cache=dict(cache.statistics))
        write_json(a.output_dir / "summary.json", report, overwrite=True)
        print("Saved:", a.output_dir / "summary.json", flush=True)
    except Exception as error:
        report.update(status="failed", decision="stop", error=type(error).__name__ + ": " + str(error))
        write_json(a.output_dir / "summary.json", report, overwrite=True)
        raise


if __name__ == "__main__":
    main()
