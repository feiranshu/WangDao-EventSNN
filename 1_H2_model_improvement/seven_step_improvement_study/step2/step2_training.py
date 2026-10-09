"""Sampling, losses, STC support, and training for H2 step-2 candidates."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from step2_common import (
    CandidateCorpus,
    FEATURE_GROUPS,
    H2_F_THRESHOLD,
    RUNTIME_ROOT,
    candidate_gates,
    checkpoint_protection,
    checkpoint_rank,
    evaluate_records,
    export_metrics,
    json_write,
    make_float_refiner,
    predict_model,
    remove_private,
    select_h2_r,
    select_fair_working_point,
    sha256,
    working_point_view,
)

from wangdao.h2.torch_model import prototype_logits
from wangdao.runtime import environment, set_seed


MODEL_DESCRIPTIONS = {
    "A2-1": "density-stratified sampling with hard/random positives and negatives",
    "A2-2": "A2-1 plus density-dependent focal contribution weights",
    "A2-3": "A2-1 plus one evidence-qualified feature-group mask",
    "A2-4": "A2-1 with released-code-style STC classification plus Soft-IoU",
    "A2-4-null": "A2-1 loss through the identical STC context path; strict control",
}


PROGRESS_FIELDS = (
    "epoch", "epochs", "loss", "classification_loss", "soft_iou_loss",
    "learning_rate", "wall_s", "coverage", "repeat_fraction", "unique_candidates",
    "working_point_eligible", "threshold", "average_precision", "overall_iou",
    "overall_precision", "overall_recall", "high_iou", "high_recall",
    "ordinary_iou", "ordinary_recall", "overall_pd", "overall_fa", "gates_pass",
)


def _progress_row(epoch_row: Mapping[str, Any], epochs: int) -> Dict[str, Any]:
    sampling = epoch_row["sampling"]
    row: Dict[str, Any] = {
        "epoch": int(epoch_row["epoch"]),
        "epochs": int(epochs),
        "loss": epoch_row["loss"],
        "classification_loss": epoch_row["classification_loss"],
        "soft_iou_loss": epoch_row["soft_iou_loss"],
        "learning_rate": epoch_row["learning_rate"],
        "wall_s": epoch_row["wall_s"],
        "coverage": sampling["coverage"],
        "repeat_fraction": sampling["repeat_fraction"],
        "unique_candidates": sampling["unique_candidates"],
        "working_point_eligible": "",
        "threshold": "",
        "average_precision": "",
        "overall_iou": "",
        "overall_precision": "",
        "overall_recall": "",
        "high_iou": "",
        "high_recall": "",
        "ordinary_iou": "",
        "ordinary_recall": "",
        "overall_pd": "",
        "overall_fa": "",
        "gates_pass": "",
    }
    view = epoch_row.get("working_point")
    if view is not None:
        row["working_point_eligible"] = bool(view.get("eligible"))
        row["average_precision"] = epoch_row["dev"]["average_precision"]
        row["gates_pass"] = bool(epoch_row["gates"]["passes"])
        if view.get("eligible"):
            row.update(
                threshold=view["threshold"],
                overall_iou=view["overall"]["event"]["iou"],
                overall_precision=view["overall"]["event"]["precision"],
                overall_recall=view["overall"]["event"]["recall"],
                high_iou=view["high_density"]["event"]["iou"],
                high_recall=view["high_density"]["event"]["recall"],
                ordinary_iou=view["ordinary_density"]["event"]["iou"],
                ordinary_recall=view["ordinary_density"]["event"]["recall"],
                overall_pd=view["overall"]["localization"]["pd"],
                overall_fa=view["overall"]["localization"]["fa"],
            )
    return row


def _append_progress(path: Path, row: Mapping[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def _print_progress(model_id: str, seed: int, row: Mapping[str, Any]) -> None:
    message = (
        f"[{model_id} seed{seed}] epoch {int(row['epoch']):02d}/{int(row['epochs']):02d} "
        f"loss={float(row['loss']):.6f} cls={float(row['classification_loss']):.6f} "
        f"iou_loss={float(row['soft_iou_loss']):.6f} lr={float(row['learning_rate']):.3e} "
        f"coverage={float(row['coverage']):.4f} repeat={float(row['repeat_fraction']):.4f} "
        f"wall={float(row['wall_s']):.1f}s"
    )
    if row["working_point_eligible"] != "":
        if row["working_point_eligible"]:
            message += (
                f" dev_ap={float(row['average_precision']):.6f} "
                f"dev_iou={float(row['overall_iou']):.6f} "
                f"dev_p={float(row['overall_precision']):.6f} "
                f"dev_r={float(row['overall_recall']):.6f} "
                f"high_r={float(row['high_recall']):.6f} "
                f"ordinary_iou={float(row['ordinary_iou']):.6f} "
                f"gates={bool(row['gates_pass'])}"
            )
        else:
            message += (
                f" dev_ap={float(row['average_precision']):.6f} "
                "working_point=none gates=False"
            )
    print(message, flush=True)


class HierarchicalCyclePool:
    """Uniform sequence/window selection with shuffled cyclic event buckets."""

    def __init__(self, corpus: CandidateCorpus, mask: np.ndarray, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.buckets: Dict[tuple[int, int], np.ndarray] = {}
        self.positions: Dict[tuple[int, int], int] = {}
        selected = np.flatnonzero(mask)
        for sequence in np.unique(corpus.sequence[selected]):
            seq_selected = selected[corpus.sequence[selected] == sequence]
            for window in np.unique(corpus.windows[seq_selected]):
                values = seq_selected[corpus.windows[seq_selected] == window].copy()
                self.rng.shuffle(values)
                key = (int(sequence), int(window))
                self.buckets[key] = values
                self.positions[key] = 0
        self.sequences = tuple(sorted({key[0] for key in self.buckets}))
        self.windows = {
            sequence: tuple(sorted(key[1] for key in self.buckets if key[0] == sequence))
            for sequence in self.sequences
        }

    def __len__(self) -> int:
        return sum(len(value) for value in self.buckets.values())

    def _take_bucket(self, key: tuple[int, int], count: int) -> np.ndarray:
        pieces = []
        remaining = int(count)
        while remaining:
            values = self.buckets[key]
            position = self.positions[key]
            available = len(values) - position
            take = min(remaining, available)
            pieces.append(values[position:position + take])
            position += take
            remaining -= take
            if position == len(values):
                values = values.copy()
                self.rng.shuffle(values)
                self.buckets[key] = values
                position = 0
            self.positions[key] = position
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 0 or not self.sequences:
            raise ValueError("cannot draw from an empty hierarchical pool")
        sequence_choices = rng.choice(np.asarray(self.sequences), size=count, replace=True)
        output = np.empty(count, dtype=np.int64)
        for sequence in np.unique(sequence_choices):
            positions = np.flatnonzero(sequence_choices == sequence)
            window_choices = rng.choice(np.asarray(self.windows[int(sequence)]), size=len(positions), replace=True)
            for window in np.unique(window_choices):
                local_positions = positions[window_choices == window]
                output[local_positions] = self._take_bucket(
                    (int(sequence), int(window)), len(local_positions)
                )
        return output


class Step2Sampler:
    QUOTAS = {
        "high_hard_positive": 256,
        "high_positive": 256,
        "ordinary_positive": 512,
        "high_hard_negative": 768,
        "high_negative": 768,
        "ordinary_hard_negative": 768,
        "ordinary_negative": 768,
    }

    def __init__(self, corpus: CandidateCorpus, seed: int) -> None:
        self.corpus = corpus
        self.seed = int(seed)
        self.pools = {
            name: HierarchicalCyclePool(corpus, corpus.mask_for_group(name), seed + 7919 * index)
            for index, name in enumerate(self.QUOTAS)
        }
        fallbacks = {
            "high_hard_positive": "high_positive",
            "high_hard_negative": "high_negative",
            "ordinary_hard_negative": "ordinary_negative",
        }
        self.fallbacks = fallbacks
        for name, pool in self.pools.items():
            if len(pool) == 0 and name not in fallbacks:
                raise ValueError(f"required A2 sampling pool is empty: {name}")
            if len(pool) == 0 and len(self.pools[fallbacks[name]]) == 0:
                raise ValueError(f"sampling pool and fallback are empty: {name}")
        self.epoch = 0
        self.selected = np.zeros(len(corpus), dtype=np.uint32)
        self.sequence_contribution = np.zeros(len(corpus.files), dtype=np.int64)
        self.window_contribution = np.zeros((len(corpus.files), 80), dtype=np.int64)
        self.stratum_counts = {name: 0 for name in self.QUOTAS}
        self.fallback_counts = {name: 0 for name in self.QUOTAS}

    def begin_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.selected.fill(0)
        self.sequence_contribution.fill(0)
        self.window_contribution.fill(0)
        self.stratum_counts = {name: 0 for name in self.QUOTAS}
        self.fallback_counts = {name: 0 for name in self.QUOTAS}

    def batch(self, update: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 1_000_003 + self.epoch * 10_007 + int(update))
        pieces = []
        for name, quota in self.QUOTAS.items():
            pool = self.pools[name]
            if len(pool) == 0:
                pool = self.pools[self.fallbacks[name]]
                self.fallback_counts[name] += quota
            values = pool.draw(quota, rng)
            pieces.append(values)
            self.stratum_counts[name] += quota
        selected = np.concatenate(pieces)
        selected = selected[rng.permutation(len(selected))]
        np.add.at(self.selected, selected, 1)
        np.add.at(self.sequence_contribution, self.corpus.sequence[selected], 1)
        np.add.at(
            self.window_contribution,
            (self.corpus.sequence[selected], self.corpus.windows[selected]),
            1,
        )
        return selected

    def report(self) -> Dict[str, Any]:
        positive = self.corpus.labels == 1
        selected_unique = self.selected > 0
        return {
            "draws": int(self.selected.sum()),
            "unique_candidates": int(selected_unique.sum()),
            "coverage": float(selected_unique.mean()),
            "positive_coverage": float(selected_unique[positive].mean()) if positive.any() else None,
            "repeat_fraction": float(1.0 - selected_unique.sum() / max(self.selected.sum(), 1)),
            "stratum_counts": self.stratum_counts,
            "fallback_counts": self.fallback_counts,
            "sequence_contribution": {
                file_name: int(value)
                for file_name, value in zip(self.corpus.files, self.sequence_contribution)
            },
            "nonzero_window_contribution": int(np.count_nonzero(self.window_contribution)),
        }


def class_probability(raw_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(prototype_logits(raw_logits), dim=1)[:, 1].clamp(1.0e-6, 1.0 - 1.0e-6)


def soft_iou_loss(probability: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = labels.to(torch.float32)
    intersection = torch.sum(probability * target)
    union = torch.sum(probability) + torch.sum(target) - intersection
    return 1.0 - (intersection + 1.0) / (union + 1.0)


def focal_values(probability: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = labels.to(torch.float32)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    alpha = torch.where(target > 0.5, 0.75, 0.25)
    return -alpha * torch.pow(1.0 - pt, 2.0) * torch.log(pt)


def candidate_loss(
    model_id: str,
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    high_density: torch.Tensor,
    stc_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    probability = class_probability(raw_logits)
    focal = focal_values(probability, labels)
    if model_id == "A2-2":
        contribution = torch.ones_like(probability)
        contribution = torch.where((labels > 0.5) & high_density, contribution * 2.0, contribution)
        contribution = torch.where((labels <= 0.5) & ~high_density, contribution * 1.5, contribution)
        classification = torch.sum(focal * contribution) / torch.sum(contribution)
    elif model_id == "A2-4":
        if stc_weight is None:
            raise ValueError("A2-4 requires detached STC weights")
        target = labels.to(torch.float32)
        classification = torch.mean(
            target * stc_weight * (-torch.log(probability))
            + (1.0 - target) * (1.0 - stc_weight) * (-torch.log(1.0 - probability))
        )
    else:
        classification = torch.mean(focal)
    soft_iou = soft_iou_loss(probability, labels)
    total = classification + 0.5 * soft_iou
    return {"total": total, "classification": classification, "soft_iou": soft_iou}


def stc_weights_sparse(
    probability: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    spatial_kernel: int = 3,
    temporal_kernel: int = 5,
) -> Dict[str, np.ndarray]:
    """Released-code-style detached STC weights on a sparse 1px/1ms grid."""
    if spatial_kernel % 2 != 1 or temporal_kernel % 2 != 1:
        raise ValueError("STC kernels must be odd")
    p = np.asarray(probability, dtype=np.float64)
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    t = np.asarray(t_ms, dtype=np.int64)
    if not (p.shape == x.shape == y.shape == t.shape):
        raise ValueError("STC arrays must have matching shapes")
    width, height = 346, 260
    keys = ((t * height + y) * width + x).astype(np.int64)
    unique, inverse = np.unique(keys, return_inverse=True)
    voxel_probability = np.zeros(len(unique), dtype=np.float64)
    np.maximum.at(voxel_probability, inverse, p)
    support = np.zeros(len(p), dtype=np.float64)
    spatial_radius = spatial_kernel // 2
    temporal_radius = temporal_kernel // 2
    for dt in range(-temporal_radius, temporal_radius + 1):
        qt = t + dt
        for dy in range(-spatial_radius, spatial_radius + 1):
            qy = y + dy
            for dx in range(-spatial_radius, spatial_radius + 1):
                qx = x + dx
                valid = (qt >= 0) & (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
                query = ((qt * height + qy) * width + qx).astype(np.int64)
                position = np.searchsorted(unique, query)
                safe = np.minimum(position, max(len(unique) - 1, 0))
                matched = valid & (position < len(unique))
                if len(unique):
                    matched &= unique[safe] == query
                    support[matched] += voxel_probability[safe[matched]]
    centered = support - float(support.mean()) if len(support) else support
    released = 1.0 / (1.0 + np.exp(-np.clip(centered, -60.0, 60.0)))
    paper = 1.0 / (1.0 + np.exp(-np.clip(support, -60.0, 60.0)))
    return {
        "support": support.astype(np.float32),
        "released": released.astype(np.float32),
        "paper": paper.astype(np.float32),
    }


def corpus_stc_weights(
    model: torch.nn.Module,
    corpus: CandidateCorpus,
    batch_size: int,
    device: torch.device,
    masked_columns: Sequence[int] = (),
) -> tuple[np.ndarray, Dict[str, Any]]:
    probability = predict_model(model, corpus.features, batch_size, device, masked_columns)
    weights = np.empty(len(corpus), dtype=np.float32)
    diagnostics = []
    for file_name, (start, stop) in zip(corpus.files, corpus.ranges):
        result = stc_weights_sparse(
            probability[start:stop], corpus.x[start:stop], corpus.y[start:stop], corpus.t_ms[start:stop]
        )
        weights[start:stop] = result["released"]
        diagnostics.append(
            {
                "file": file_name,
                "mean": float(result["released"].mean()),
                "min": float(result["released"].min()),
                "max": float(result["released"].max()),
                "saturated_fraction": float(np.mean((result["released"] < 0.01) | (result["released"] > 0.99))),
            }
        )
    return weights, {"sequences": diagnostics, "mean": float(weights.mean())}


def _baseline_views(baseline: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    h2_f_selection = {
        "eligible": True,
        "threshold": H2_F_THRESHOLD,
        "metrics": baseline["curve"][str(float(H2_F_THRESHOLD))],
    }
    h2_r_selection = select_h2_r(baseline)
    return (
        working_point_view(baseline, h2_f_selection),
        working_point_view(baseline, h2_r_selection),
        h2_r_selection,
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    with path.open("xb") as stream:
        package = {
            "format_version": 2,
            "dimensions": tuple(model.dimensions),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": dict(metadata),
        }
        if optimizer is not None:
            package["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            package["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(package, stream)


def load_checkpoint_package(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 1.9 compatibility
        return torch.load(path, map_location="cpu")


def _load_checkpoint(path: Path, model: torch.nn.Module) -> Mapping[str, Any]:
    package = load_checkpoint_package(path)
    model.load_state_dict(package["state_dict"], strict=True)
    return package


def train_candidate(
    model_id: str,
    output_dir: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    selected_feature_group: Optional[str] = None,
    smoke: bool = False,
    resume_checkpoint: Optional[Path] = None,
    epoch_offset: int = 0,
) -> Dict[str, Any]:
    if model_id not in MODEL_DESCRIPTIONS:
        raise ValueError("unknown step-2 model")
    if model_id == "A2-3" and selected_feature_group not in FEATURE_GROUPS:
        raise ValueError("A2-3 requires an evidence-qualified feature group")
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    corpus = CandidateCorpus(train_records)
    sampler = Step2Sampler(corpus, seed)
    masked_columns = FEATURE_GROUPS[selected_feature_group] if model_id == "A2-3" else ()
    model = make_float_refiner(seed).to(device)
    resume_package = None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        resume_package = _load_checkpoint(resume_checkpoint, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    if resume_package is not None and "optimizer_state_dict" in resume_package:
        optimizer.load_state_dict(resume_package["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    baseline = evaluate_records(dev_records, None, device, batch_size)
    h2_f_view, h2_r_view, h2_r_selection = _baseline_views(baseline)
    initial = evaluate_records(dev_records, model, device, batch_size, masked_columns)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "model_id": model_id,
        "description": MODEL_DESCRIPTIONS[model_id],
        "seed": int(seed),
        "smoke": bool(smoke),
        "data_roles": {"training": "train_main", "selection": "train_dev"},
        "validation_used": False,
        "test_used": False,
        "environment": environment(device),
        "configuration": {
            "epochs": int(epochs),
            "epoch_offset": int(epoch_offset),
            "final_epoch": int(epoch_offset + epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": 1.0e-4,
            "gradient_clip": 5.0,
            "scheduler": "cosine",
            "updates_per_epoch": 1 if smoke else int(math.ceil(len(corpus) / batch_size)),
            "feature_group_mask": selected_feature_group,
            "loss": (
                "released-centered-detached-STC+0.5soft_iou" if model_id == "A2-4"
                else "focal+0.5soft_iou"
            ),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "optimizer_state_restored": bool(
                resume_package is not None and "optimizer_state_dict" in resume_package
            ),
        },
        "provenance": {
            "h2": sha256(RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"),
            "index": sha256(RUNTIME_ROOT / "data" / "index.json"),
        },
        "candidate_count": len(corpus),
        "h2_f": h2_f_view,
        "h2_r": h2_r_view,
        "h2_r_selection": h2_r_selection,
        "initial_float_refiner": remove_private(initial),
        "history": [],
    }
    json_write(output_dir / "summary.json", report)
    evaluation_epochs = {
        epoch_offset + value for value in range(5, epochs + 1, 5)
    } | {epoch_offset + epochs}
    if smoke:
        evaluation_epochs = {epoch_offset + epochs}
    best_protected: Optional[tuple[tuple[float, ...], Path, Dict[str, Any]]] = None
    best_any: Optional[tuple[tuple[float, ...], Path, Dict[str, Any]]] = None
    updates = 1 if smoke else int(math.ceil(len(corpus) / batch_size))
    try:
        for epoch in range(1 + epoch_offset, epochs + 1 + epoch_offset):
            started = time.perf_counter()
            sampler.begin_epoch(epoch)
            stc_weights = None
            stc_report = None
            if model_id in ("A2-4", "A2-4-null"):
                stc_weights, stc_report = corpus_stc_weights(
                    model, corpus, batch_size, device, masked_columns
                )
            model.train()
            total_loss = 0.0
            total_classification = 0.0
            total_iou = 0.0
            for update in range(updates):
                selected = sampler.batch(update)
                block = corpus.features[selected].copy()
                if masked_columns:
                    block[:, masked_columns] = 0
                x = torch.from_numpy(block).to(device)
                y = torch.as_tensor(corpus.labels[selected], dtype=torch.float32, device=device)
                density = torch.as_tensor(
                    corpus.high_density[selected].astype(bool), dtype=torch.bool, device=device
                )
                weight = (
                    torch.as_tensor(stc_weights[selected], dtype=torch.float32, device=device)
                    if model_id == "A2-4" else None
                )
                raw = model(x)
                loss = candidate_loss(
                    "A2-1" if model_id == "A2-4-null" else model_id,
                    raw,
                    y,
                    density,
                    weight,
                )
                if not torch.isfinite(loss["total"]):
                    raise FloatingPointError("non-finite step-2 loss")
                optimizer.zero_grad(set_to_none=True)
                loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss["total"].detach().cpu())
                total_classification += float(loss["classification"].detach().cpu())
                total_iou += float(loss["soft_iou"].detach().cpu())
            scheduler.step()
            epoch_row: Dict[str, Any] = {
                "epoch": epoch,
                "loss": total_loss / updates,
                "classification_loss": total_classification / updates,
                "soft_iou_loss": total_iou / updates,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "sampling": sampler.report(),
                "stc": stc_report,
                "wall_s": time.perf_counter() - started,
            }
            if epoch in evaluation_epochs:
                evaluation = evaluate_records(
                    dev_records, model, device, batch_size, masked_columns
                )
                anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
                selection = select_fair_working_point(
                    evaluation["curve"],
                    anchor["localization"]["pd"],
                    anchor["localization"]["fa"],
                )
                view = working_point_view(evaluation, selection)
                gates = candidate_gates(view, h2_f_view, h2_r_view)
                protected = checkpoint_protection(view, h2_f_view, h2_r_view)
                checkpoint = output_dir / f"epoch_{epoch:03d}.pt"
                _save_checkpoint(
                    checkpoint,
                    model,
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "epoch": epoch,
                        "feature_group_mask": selected_feature_group,
                        "smoke": smoke,
                        "epoch_offset": epoch_offset,
                        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
                    },
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                rank = checkpoint_rank(view, epoch) if view.get("eligible") else (-math.inf,) * 5
                item = (rank, checkpoint, {"evaluation": evaluation, "selection": selection, "view": view, "gates": gates})
                if best_any is None or item[0] > best_any[0]:
                    best_any = item
                if protected and (best_protected is None or item[0] > best_protected[0]):
                    best_protected = item
                epoch_row.update(
                    dev=remove_private(evaluation),
                    working_point=view,
                    gates=gates,
                    checkpoint_protected=protected,
                    checkpoint=checkpoint.name,
                )
            report["history"].append(epoch_row)
            progress = _progress_row(epoch_row, epoch_offset + epochs)
            _append_progress(output_dir / "training_progress.csv", progress)
            _print_progress(model_id, seed, progress)
            json_write(output_dir / "summary.json", report, overwrite=True)
        chosen = best_protected or best_any
        if chosen is None:
            raise RuntimeError("no finite evaluated checkpoint")
        _, chosen_path, chosen_state = chosen
        _load_checkpoint(chosen_path, model)
        final = evaluate_records(
            dev_records, model, device, batch_size, masked_columns, keep_pr_points=True
        )
        anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
        final_selection = select_fair_working_point(
            final["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]
        )
        final_view = working_point_view(final, final_selection)
        gates = candidate_gates(final_view, h2_f_view, h2_r_view)
        metrics_dir = output_dir / "metrics"
        export_metrics(metrics_dir, final, final_view)
        report.update(
            status="complete",
            decision=("smoke_only" if smoke else ("pass" if gates["passes"] else "reject")),
            best_checkpoint=chosen_path.name,
            best_checkpoint_sha256=sha256(chosen_path),
            best_checkpoint_protected=best_protected is not None,
            final_working_point=final_view,
            final_gates=gates,
            conclusion=(
                "Smoke verifies execution only; no accuracy claim."
                if smoke else
                ("Candidate passed every seed-level step-2 gate." if gates["passes"] else "Candidate failed one or more seed-level step-2 gates.")
            ),
        )
        json_write(output_dir / "summary.json", report, overwrite=True)
        print(
            f"[{model_id} seed{seed}] complete decision={report['decision']} "
            f"best_checkpoint={report['best_checkpoint']} gates={gates['passes']}",
            flush=True,
        )
        return report
    except Exception as error:
        report.update(status="failed", decision="stop", error=f"{type(error).__name__}: {error}")
        json_write(output_dir / "summary.json", report, overwrite=True)
        print(f"[{model_id} seed{seed}] failed: {type(error).__name__}: {error}", flush=True)
        raise


def repair_candidate_export(
    output_dir: Path,
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """Finish a candidate that trained successfully but failed during export.

    No optimizer step is performed.  The best saved evaluated checkpoint is
    selected with the original ranking, evaluated again, and exported using
    the now-safe ineligible-working-point path.
    """
    summary_path = output_dir / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if report.get("status") == "complete":
        return report
    expected = "float() argument must be a string or a real number, not 'NoneType'"
    if report.get("status") != "failed" or expected not in str(report.get("error")):
        raise RuntimeError(f"candidate is not eligible for export-only repair: {output_dir}")
    evaluated = [row for row in report.get("history", []) if row.get("checkpoint")]
    if not evaluated:
        raise RuntimeError(f"candidate has no evaluated checkpoint to repair: {output_dir}")

    def row_rank(row: Mapping[str, Any]) -> tuple[float, ...]:
        view = row.get("working_point", {})
        return checkpoint_rank(view, int(row["epoch"])) if view.get("eligible") else (-math.inf,) * 5

    protected = [row for row in evaluated if row.get("checkpoint_protected")]
    chosen_row = max(protected or evaluated, key=row_rank)
    chosen_path = output_dir / str(chosen_row["checkpoint"])
    if not chosen_path.is_file():
        raise FileNotFoundError(chosen_path)

    seed = int(report["seed"])
    selected_group = report["configuration"].get("feature_group_mask")
    masked_columns = FEATURE_GROUPS[selected_group] if selected_group else ()
    model = make_float_refiner(seed).to(device)
    _load_checkpoint(chosen_path, model)
    baseline = evaluate_records(dev_records, None, device, batch_size)
    h2_f_view, h2_r_view, _ = _baseline_views(baseline)
    final = evaluate_records(
        dev_records, model, device, batch_size, masked_columns, keep_pr_points=True
    )
    anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
    final_selection = select_fair_working_point(
        final["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]
    )
    final_view = working_point_view(final, final_selection)
    gates = candidate_gates(final_view, h2_f_view, h2_r_view)

    metrics_dir = output_dir / "metrics"
    if metrics_dir.exists():
        suffix = 1
        archived = output_dir / "metrics_failed_export"
        while archived.exists():
            suffix += 1
            archived = output_dir / f"metrics_failed_export_{suffix}"
        metrics_dir.replace(archived)
    export_metrics(metrics_dir, final, final_view)
    previous_error = report.pop("error", None)
    report.update(
        status="complete",
        decision=("pass" if gates["passes"] else "reject"),
        best_checkpoint=chosen_path.name,
        best_checkpoint_sha256=sha256(chosen_path),
        best_checkpoint_protected=bool(protected),
        final_working_point=final_view,
        final_gates=gates,
        repaired_export=True,
        previous_error=previous_error,
        conclusion=(
            "Candidate passed every seed-level step-2 gate."
            if gates["passes"] else
            "Candidate failed one or more seed-level step-2 gates; metrics were exported diagnostically."
        ),
    )
    json_write(summary_path, report, overwrite=True)
    return report

