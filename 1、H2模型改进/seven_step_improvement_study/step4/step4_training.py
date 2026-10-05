"""Natural-pass Step4 training with per-epoch float/int8 evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from step4_common import (
    INT8_REFERENCE_AP,
    MODEL_DESCRIPTIONS,
    PROTOCOL,
    REGISTERED_CONTINUATION,
    REGISTERED_INITIAL,
    RUNTIME_ROOT,
    CandidateCorpus,
    DensityTransform,
    Step4Refiner,
    activation_saturation,
    baseline_views,
    build_model,
    continue10_allowed,
    eligibility,
    evaluate_model,
    evaluation_row,
    exact_pr,
    export_metrics,
    feature_parts,
    high_density_rank,
    joined_features,
    json_write,
    predict_float,
    predict_int8,
    primary_rank,
    quantized_layers,
    remove_private,
    sha256,
    weight_norms,
)
from wangdao.h2.torch_model import prototype_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_batches(size: int, epoch: int, seed: int, batch_size: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed * 1_000_003 + epoch)
    order = rng.permutation(size).astype(np.int64, copy=False)
    return [order[first:first + batch_size] for first in range(0, size, batch_size)]


def candidate_bce(raw: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(prototype_logits(raw), dim=1)[:, 1]
    return F.binary_cross_entropy(probability, labels)


def _clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _optimizer(model: Step4Refiner, continuation: bool) -> torch.optim.AdamW:
    if model.new_width:
        groups = [{"params": model.new_parameters(), "lr": 1.0e-5 if continuation else 1.0e-3, "name": "new_input"}]
        if continuation:
            groups.append({"params": model.old_parameters(), "lr": 1.0e-5, "name": "old_h2"})
        return torch.optim.AdamW(
            groups, lr=1.0e-5 if continuation else 1.0e-3, weight_decay=1.0e-4,
        )
    parameters = model.old_parameters() + model.new_parameters()
    return torch.optim.AdamW(
        [{"params": parameters, "lr": 1.0e-5, "name": "all"}],
        lr=1.0e-5, weight_decay=1.0e-4,
    )


def _activate_full_finetune(optimizer: torch.optim.AdamW, model: Step4Refiner) -> None:
    if any(group.get("name") == "old_h2" for group in optimizer.param_groups):
        return
    for group in optimizer.param_groups:
        group["lr"] = 1.0e-5
    optimizer.add_param_group({"params": model.old_parameters(), "lr": 1.0e-5, "name": "old_h2"})


def _save_checkpoint(
    path: Path, model: Step4Refiner, optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
) -> None:
    package = {
        "format_version": 1,
        "protocol": PROTOCOL,
        "dimensions": model.dimensions,
        "old_width": model.old_width,
        "new_width": model.new_width,
        "state_dict": _clone_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "metadata": dict(metadata),
    }
    with path.open("xb") as stream:
        torch.save(package, stream)


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def export_int8(path: Path, model: Step4Refiner, metadata: Mapping[str, Any]) -> None:
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "dimensions": np.asarray(model.dimensions, dtype=np.int64),
        "old_width": np.asarray(model.old_width, dtype=np.int64),
        "new_width": np.asarray(model.new_width, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    for index, (weight, scale) in enumerate(quantized_layers(model)):
        arrays[f"layer{index}_weight"] = weight
        arrays[f"layer{index}_scale"] = scale
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


PROGRESS_FIELDS = (
    "epoch", "final_epoch", "phase", "loss", "learning_rate_new", "learning_rate_old",
    "gradient_norm_mean", "gradient_norm_max", "coverage", "repeat_fraction",
    "new_input_norm", "old_input_norm", "new_to_old_norm",
    "hidden1_zero_fraction", "hidden1_one_fraction", "hidden1_linear_fraction",
    "hidden2_zero_fraction", "hidden2_one_fraction", "hidden2_linear_fraction",
    "train_s", "float_evaluation_s", "int8_evaluation_s", "checkpoint_s", "wall_s",
    "float_ap", "int8_ap", "int8_iou", "int8_high_recall",
    "eligibility_level", "replacement_pass", "noninferiority_pass",
)


def _write_progress(path: Path, row: Mapping[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def _fmt(value: Any, digits: int = 6) -> str:
    return "NA" if value in (None, "") else f"{float(value):.{digits}f}"


def _sci(value: Any) -> str:
    return "NA" if value in (None, "") else f"{float(value):.2e}"


def _print_progress(model_id: str, seed: int, row: Mapping[str, Any]) -> None:
    print(
        f"[{model_id} seed{seed}] epoch {int(row['epoch']):02d}/{int(row['final_epoch']):02d} "
        f"phase={row['phase']} loss={_fmt(row['loss'])} "
        f"lr(new/old)={_sci(row.get('learning_rate_new'))}/{_sci(row.get('learning_rate_old'))} "
        f"grad={_fmt(row['gradient_norm_mean'], 4)} wall={_fmt(row['wall_s'], 1)}s "
        f"norm(new/old)={_fmt(row['new_input_norm'], 4)}/{_fmt(row['old_input_norm'], 4)} "
        f"int8_ap={_fmt(row.get('int8_ap'))} iou={_fmt(row.get('int8_iou'))} "
        f"high_r={_fmt(row.get('int8_high_recall'))} level={row.get('eligibility_level', 'NA')}",
        flush=True,
    )


def _learning_rates(optimizer: torch.optim.Optimizer) -> tuple[Optional[float], Optional[float]]:
    new = old = None
    for group in optimizer.param_groups:
        name = group.get("name")
        if name in ("new_input", "all"):
            new = float(group["lr"])
        if name in ("old_h2", "all"):
            old = float(group["lr"])
    return new, old


def _registered(epoch_offset: int) -> set[int]:
    return set(REGISTERED_CONTINUATION if epoch_offset else REGISTERED_INITIAL)


def _history_availability_report(
    records: Sequence[Mapping[str, Any]], model: Step4Refiner, model_id: str,
    density: DensityTransform, device: torch.device, batch_size: int,
    threshold: Optional[float], selected_information: Optional[str],
) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    all_labels = np.concatenate([np.asarray(record["labels"], dtype=np.uint8) for record in records])
    all_windows = np.concatenate([np.asarray(record["windows"], dtype=np.int64) for record in records])
    availability = np.minimum(all_windows, 2)
    probabilities = np.concatenate([
        predict_int8(model, joined_features(record, model_id, density, selected_information))
        for record in records
    ])
    for count in (0, 1, 2):
        mask = availability == count
        labels = all_labels[mask]
        probability = probabilities[mask]
        pr = exact_pr(labels, probability)
        row: Dict[str, Any] = {
            "available_prior_windows": count,
            "candidate_events": int(mask.sum()),
            "positive_candidate_events": int(labels.sum()),
            "candidate_average_precision": pr["average_precision"],
            "scope": "Proposal candidates only; diagnostic, not full-event AP",
        }
        if threshold is None or not len(labels):
            row.update(candidate_iou=None, candidate_precision=None, candidate_recall=None)
        else:
            predicted = probability >= float(threshold)
            positive = labels == 1
            tp = int(np.sum(predicted & positive))
            fp = int(np.sum(predicted & ~positive))
            fn = int(np.sum(~predicted & positive))
            row.update(
                candidate_iou=(tp / (tp + fp + fn) if tp + fp + fn else None),
                candidate_precision=(tp / (tp + fp) if tp + fp else None),
                candidate_recall=(tp / (tp + fn) if tp + fn else None),
            )
        rows[str(count)] = row
    return {"scope": "Train-dev candidate diagnostic", "groups": rows}


def _export_bootstrap_scores(
    path: Path, records: Sequence[Mapping[str, Any]], model: Step4Refiner,
    model_id: str, density: DensityTransform, selected_information: Optional[str],
) -> None:
    labels, probabilities, sequences = [], [], []
    for sequence, record in enumerate(records):
        candidate = predict_int8(model, joined_features(record, model_id, density, selected_information))
        full = np.zeros(int(record["event_count"]), dtype=np.float32)
        full[np.asarray(record["indices"], dtype=np.int64)] = candidate
        labels.append(np.asarray(record["all_labels"], dtype=np.uint8))
        probabilities.append(full)
        sequences.append(np.full(len(full), sequence, dtype=np.uint8))
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, labels=np.concatenate(labels), probability=np.concatenate(probabilities),
            sequence=np.concatenate(sequences),
            sequence_files=np.asarray([str(record["file"]) for record in records]),
        )


def train_candidate(
    model_id: str,
    output_dir: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    density: DensityTransform,
    device: torch.device,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 4096,
    smoke: bool = False,
    selected_information: Optional[str] = None,
    resume_checkpoint: Optional[Path] = None,
    epoch_offset: int = 0,
) -> Dict[str, Any]:
    if model_id not in MODEL_DESCRIPTIONS:
        raise ValueError("unknown Step4 model")
    if epoch_offset not in (0, 10):
        raise ValueError("Step4 supports only initial 0..10 or one continuation 10..20")
    if epochs != (1 if smoke else 10):
        raise ValueError("formal Step4 stages are fixed at 10 epochs; smoke is one epoch")
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    corpus = CandidateCorpus(train_records)
    train_features = np.concatenate([
        joined_features(row, model_id, density, selected_information) for row in train_records
    ])
    model = build_model(model_id, seed, selected_information).to(device)
    optimizer = _optimizer(model, continuation=bool(epoch_offset))
    if resume_checkpoint is not None:
        package = load_checkpoint(Path(resume_checkpoint))
        metadata = package["metadata"]
        if metadata.get("model_id") != model_id or int(metadata.get("seed", -1)) != seed:
            raise ValueError("resume checkpoint identity mismatch")
        if metadata.get("selected_information") != selected_information:
            raise ValueError("resume checkpoint selected-information mismatch")
        if int(metadata.get("epoch", -1)) != epoch_offset:
            raise ValueError("resume checkpoint epoch mismatch")
        model.load_state_dict(package["state_dict"], strict=True)
        optimizer.load_state_dict(package["optimizer_state_dict"])
        torch.set_rng_state(package["torch_rng_state"])
        if torch.cuda.is_available() and package.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(package["cuda_rng_state"])

    baseline, h2_f, h2_r = baseline_views(dev_records, device, batch_size)
    initial_pair = evaluate_model(
        dev_records, model, model_id, density, device, batch_size, selected_information
    )
    initial = evaluation_row(epoch_offset, initial_pair, baseline, h2_f, h2_r)
    fixed_indices = np.linspace(0, len(corpus) - 1, min(65536, len(corpus)), dtype=np.int64)
    uses_history = model_id in ("A4-2", "A4-2-null", "A4-4", "A4-4-null") or selected_information in ("history", "both")
    train_history_counts = None
    if uses_history:
        values, counts = np.unique(np.minimum(corpus.windows.astype(np.int64), 2), return_counts=True)
        train_history_counts = {str(int(value)): int(count) for value, count in zip(values, counts)}
    report: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "running",
        "model_id": model_id,
        "description": MODEL_DESCRIPTIONS[model_id],
        "selected_information": selected_information,
        "seed": seed,
        "smoke": bool(smoke),
        "validation_used": False,
        "test_used": False,
        "configuration": {
            "epochs": epochs, "epoch_offset": epoch_offset, "final_epoch": epoch_offset + epochs,
            "batch_size": batch_size, "sampling": "natural_full_pass", "loss": "BCE",
            "new_only_epochs": [] if epoch_offset else ([1, 2] if model.new_width else []),
            "new_only_lr": 1.0e-3 if model.new_width and not epoch_offset else None,
            "full_finetune_lr": 1.0e-5, "weight_decay": 1.0e-4,
            "gradient_clip": 5.0, "ema": False, "qat": False,
            "evaluate_every_epoch": True,
            "registered_epochs": sorted(_registered(epoch_offset)),
            "formal_selection_numeric_path": "exact_int8_replay",
        },
        "density_transform": density.as_dict(),
        "provenance": {
            "h2": sha256(RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"),
            "index": sha256(RUNTIME_ROOT / "data" / "index.json"),
        },
        "candidate_count": len(corpus),
        "train_main_history_availability_candidate_counts": train_history_counts,
        "h2_f": h2_f,
        "h2_r": h2_r,
        "initial": initial,
        "evaluations": [initial],
        "history": [],
    }
    json_write(output_dir / "summary.json", report)
    initial_checkpoint = output_dir / f"epoch_{epoch_offset:03d}.pt"
    _save_checkpoint(initial_checkpoint, model, optimizer, {
        "model_id": model_id, "selected_information": selected_information, "seed": seed,
        "epoch": epoch_offset, "update_count": 0,
    })
    checkpoint_by_epoch: Dict[int, Path] = {epoch_offset: initial_checkpoint}

    final_epoch = epoch_offset + epochs
    try:
        for epoch in range(epoch_offset + 1, final_epoch + 1):
            if model.new_width and not epoch_offset and epoch == 3:
                _activate_full_finetune(optimizer, model)
            phase = "new_columns_only" if model.new_width and not epoch_offset and epoch <= 2 else "full_refiner"
            batches = natural_batches(len(corpus), epoch, seed, batch_size)
            if smoke:
                batches = batches[:1]
            started = time.perf_counter()
            model.train()
            losses, gradients = [], []
            for selected in batches:
                x = torch.from_numpy(train_features[selected]).to(device)
                y = torch.as_tensor(corpus.labels[selected], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = candidate_bce(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Step4 BCE")
                loss.backward()
                trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
                norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradients.append(float(norm.detach().cpu()))
            train_s = time.perf_counter() - started
            saturation = activation_saturation(model, train_features[fixed_indices], device)
            norms = weight_norms(model)
            pair = evaluate_model(
                dev_records, model, model_id, density, device, batch_size,
                selected_information,
            )
            evaluated = evaluation_row(epoch, pair, baseline, h2_f, h2_r)
            report["evaluations"].append(evaluated)
            checkpoint = output_dir / f"epoch_{epoch:03d}.pt"
            checkpoint_started = time.perf_counter()
            _save_checkpoint(checkpoint, model, optimizer, {
                "model_id": model_id, "selected_information": selected_information, "seed": seed,
                "epoch": epoch, "update_count": len(batches), "phase": phase,
            })
            checkpoint_s = time.perf_counter() - checkpoint_started
            checkpoint_by_epoch[epoch] = checkpoint
            learning_rate_new, learning_rate_old = _learning_rates(optimizer)
            row = {
                "epoch": epoch, "final_epoch": final_epoch, "phase": phase,
                "loss": float(np.mean(losses)), "learning_rate_new": learning_rate_new,
                "learning_rate_old": learning_rate_old,
                "gradient_norm_mean": float(np.mean(gradients)), "gradient_norm_max": float(np.max(gradients)),
                "coverage": len(np.unique(np.concatenate(batches))) / len(corpus),
                "repeat_fraction": 1.0 - len(np.unique(np.concatenate(batches))) / len(np.concatenate(batches)),
                "old_input_norm": norms["old_input"], "new_input_norm": norms["new_input"],
                "new_to_old_norm": norms["new_to_old"],
                "hidden1_zero_fraction": saturation[0]["zero_fraction"],
                "hidden1_one_fraction": saturation[0]["one_fraction"],
                "hidden1_linear_fraction": saturation[0]["linear_fraction"],
                "hidden2_zero_fraction": saturation[1]["zero_fraction"],
                "hidden2_one_fraction": saturation[1]["one_fraction"],
                "hidden2_linear_fraction": saturation[1]["linear_fraction"],
                "train_s": train_s,
                "float_evaluation_s": pair["_timings"]["float_evaluation_s"],
                "int8_evaluation_s": pair["_timings"]["int8_evaluation_s"],
                "checkpoint_s": checkpoint_s,
                "wall_s": time.perf_counter() - started,
                "float_ap": pair["float"]["average_precision"],
                "int8_ap": pair["int8"]["average_precision"],
                "int8_iou": evaluated["int8_working_point"].get("overall", {}).get("event", {}).get("iou"),
                "int8_high_recall": evaluated["int8_working_point"].get("high_density", {}).get("event", {}).get("recall"),
                "eligibility_level": eligibility(evaluated),
                "replacement_pass": evaluated["replacement_gates"]["passes"],
                "noninferiority_pass": evaluated["noninferiority"]["passes"],
                "activation_saturation": saturation,
                "weight_norms": norms,
                "checkpoint": checkpoint.name,
            }
            report["history"].append(row)
            _write_progress(output_dir / "training_progress.csv", row)
            _print_progress(model_id, seed, row)
            json_write(output_dir / "summary.json", report, overwrite=True)

        registered = [row for row in report["evaluations"] if int(row["epoch"]) in _registered(epoch_offset)]
        primary = max(registered, key=primary_rank)
        high_density = max(registered, key=high_density_rank)
        primary_checkpoint = checkpoint_by_epoch[int(primary["epoch"])]
        high_checkpoint = checkpoint_by_epoch[int(high_density["epoch"])]
        package = load_checkpoint(primary_checkpoint)
        model.load_state_dict(package["state_dict"], strict=True)
        export_started = time.perf_counter()
        final_pair = evaluate_model(
            dev_records, model, model_id, density, device, batch_size,
            selected_information, keep_pr_points=True,
        )
        primary_final = evaluation_row(int(primary["epoch"]), final_pair, baseline, h2_f, h2_r)
        export_path = output_dir / "primary_refiner_int8.npz"
        export_int8(export_path, model, {
            "model_id": model_id, "selected_information": selected_information,
            "epoch": primary["epoch"], "density_transform": density.as_dict(),
        })
        export_metrics(output_dir / "metrics" / "int8", final_pair["int8"], primary_final["int8_working_point"])
        export_metrics(output_dir / "metrics" / "float", final_pair["float"], primary_final["float_working_point"])
        _export_bootstrap_scores(
            output_dir / "metrics" / "int8" / "bootstrap_event_scores.npz",
            dev_records, model, model_id, density, selected_information,
        )
        history_report = None
        if uses_history:
            history_report = _history_availability_report(
                dev_records, model, model_id, density, device, batch_size,
                primary_final["int8_working_point"].get("threshold"), selected_information,
            )
        report.update(
            status="complete",
            decision="smoke_only" if smoke else ("replacement" if eligibility(primary) == 2 else ("noninferior" if eligibility(primary) == 1 else "diagnostic_only")),
            final={"primary": primary_final, "high_density": high_density},
            primary_checkpoint=primary_checkpoint.name,
            primary_checkpoint_sha256=sha256(primary_checkpoint),
            high_density_checkpoint=high_checkpoint.name,
            high_density_checkpoint_sha256=sha256(high_checkpoint),
            int8_export=export_path.name,
            int8_export_sha256=sha256(export_path),
            history_availability=history_report,
            final_export_s=time.perf_counter() - export_started,
        )
        report["continue10"] = {"passes": False, "reason": "smoke or continuation already used"} if (smoke or epoch_offset) else continue10_allowed(report)
        json_write(output_dir / "summary.json", report, overwrite=True)
        print(f"[{model_id} seed{seed}] complete decision={report['decision']} primary=epoch_{int(primary['epoch']):03d}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        json_write(output_dir / "summary.json", report, overwrite=True)
        raise


def equivalence_audit(
    output: Path, train_records: Sequence[Mapping[str, Any]], dev_records: Sequence[Mapping[str, Any]],
    density: DensityTransform, device: torch.device, batch_size: int, smoke: bool,
) -> Dict[str, Any]:
    """Prove epoch-0 and null one-step equivalence without touching formal runs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = build_model("A4-3-control", 42).to(device)
    baseline_float = [predict_float(reference, np.asarray(row["features"], dtype=np.uint8), device, batch_size) for row in dev_records]
    baseline_int8 = [predict_int8(reference, np.asarray(row["features"], dtype=np.uint8)) for row in dev_records]
    checks: Dict[str, Any] = {}
    for model_id in ("A4-1", "A4-1-null", "A4-2", "A4-2-null", "A4-4", "A4-4-null"):
        model = build_model(model_id, 42).to(device)
        blocks = [joined_features(row, model_id, density) for row in dev_records]
        values_float = [predict_float(model, block, device, batch_size) for block in blocks]
        values_int8 = [predict_int8(model, block) for block in blocks]
        max_float = max(float(np.max(np.abs(a - b))) for a, b in zip(values_float, baseline_float))
        max_int8 = max(float(np.max(np.abs(a - b))) for a, b in zip(values_int8, baseline_int8))
        checks[f"{model_id}_epoch0"] = {
            "float_max_abs": max_float, "int8_max_abs": max_int8,
            "passes": max_float <= 1.0e-7 and max_int8 <= 1.0e-7,
        }

    corpus = CandidateCorpus(train_records)
    selected = natural_batches(len(corpus), 1, 42, batch_size)[0]
    reference = build_model("A4-3-control", 42).to(device)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1.0e-5, weight_decay=1.0e-4)
    x_ref = torch.from_numpy(corpus.features[selected]).to(device)
    y = torch.as_tensor(corpus.labels[selected], dtype=torch.float32, device=device)
    ref_optimizer.zero_grad(set_to_none=True)
    candidate_bce(reference(x_ref), y).backward()
    ref_optimizer.step()
    for model_id in ("A4-1-null", "A4-2-null", "A4-4-null"):
        model = build_model(model_id, 42).to(device)
        optimizer = torch.optim.AdamW(model.old_parameters(), lr=1.0e-5, weight_decay=1.0e-4)
        x = torch.from_numpy(np.concatenate([
            joined_features(row, model_id, density) for row in train_records
        ])[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        candidate_bce(model(x), y).backward()
        optimizer.step()
        differences = [
            float(torch.max(torch.abs(a.detach() - b.detach())).cpu())
            for a, b in zip(model.old_parameters(), reference.parameters())
        ]
        new_unchanged = float(torch.max(torch.abs(model.new_input.weight.detach())).cpu()) == 0.0
        checks[f"{model_id}_one_step"] = {
            "old_parameter_max_abs": max(differences), "new_weights_remain_zero": new_unchanged,
            "passes": max(differences) <= 1.0e-6 and new_unchanged,
        }
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "status": "pass" if all(row["passes"] for row in checks.values()) else "fail",
        "smoke": bool(smoke), "validation_used": False, "test_used": False,
        "checks": checks, "reference_int8_ap": INT8_REFERENCE_AP,
    }
    json_write(output, report)
    if report["status"] != "pass":
        raise RuntimeError("Step4 equivalence audit failed")
    return report
