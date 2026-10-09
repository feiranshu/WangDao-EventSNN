"""D3-0 sampling, calibration, feature-information, and capacity diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from step3_common import (
    CandidateCorpus,
    H2_F_THRESHOLD,
    RUNTIME_ROOT,
    SEQUENCE_DENSITY_EPS,
    aggregate_scores,
    evaluate_records,
    json_write,
    remove_private,
    select_h2_r,
    sha256,
    working_point_view,
)
from step3_training import candidate_bce, set_seed


LEGACY_QUOTAS = {
    "high_hard_positive": 256,
    "high_positive": 256,
    "ordinary_positive": 512,
    "high_hard_negative": 768,
    "high_negative": 768,
    "ordinary_hard_negative": 768,
    "ordinary_negative": 768,
}
PROBE_VARIANTS = ("base32", "plus_density", "plus_history", "drop_sequence_time", "drop_coordinates")


class LegacyPool:
    """Exact A2 sequence/window-uniform cyclic pool, retained only for audit."""

    def __init__(self, corpus: CandidateCorpus, mask: np.ndarray, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.buckets: Dict[tuple[int, int], np.ndarray] = {}
        self.positions: Dict[tuple[int, int], int] = {}
        selected = np.flatnonzero(mask)
        for sequence in np.unique(corpus.sequence[selected]):
            seq_values = selected[corpus.sequence[selected] == sequence]
            for window in np.unique(corpus.windows[seq_values]):
                values = seq_values[corpus.windows[seq_values] == window].copy()
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

    def _take(self, key: tuple[int, int], count: int) -> np.ndarray:
        pieces = []
        while count:
            values = self.buckets[key]
            position = self.positions[key]
            take = min(count, len(values) - position)
            pieces.append(values[position:position + take])
            position += take
            count -= take
            if position == len(values):
                values = values.copy()
                self.rng.shuffle(values)
                self.buckets[key] = values
                position = 0
            self.positions[key] = position
        return np.concatenate(pieces)

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray:
        sequence_choices = rng.choice(np.asarray(self.sequences), size=count, replace=True)
        output = np.empty(count, dtype=np.int64)
        for sequence in np.unique(sequence_choices):
            positions = np.flatnonzero(sequence_choices == sequence)
            window_choices = rng.choice(np.asarray(self.windows[int(sequence)]), size=len(positions), replace=True)
            for window in np.unique(window_choices):
                local = positions[window_choices == window]
                output[local] = self._take((int(sequence), int(window)), len(local))
        return output


def _legacy_masks(corpus: CandidateCorpus) -> Dict[str, np.ndarray]:
    positive = corpus.labels == 1
    high = corpus.legacy_window_high_density.astype(bool)
    return {
        "high_hard_positive": high & positive & corpus.hard_positive.astype(bool),
        "high_positive": high & positive,
        "ordinary_positive": ~high & positive,
        "high_hard_negative": high & ~positive & corpus.hard_negative.astype(bool),
        "high_negative": high & ~positive,
        "ordinary_hard_negative": ~high & ~positive & corpus.hard_negative.astype(bool),
        "ordinary_negative": ~high & ~positive,
    }


def legacy_sampling_audit(corpus: CandidateCorpus, smoke: bool = False) -> Dict[str, Any]:
    masks = _legacy_masks(corpus)
    pools = {
        name: LegacyPool(corpus, masks[name], 42 + 7919 * index)
        for index, name in enumerate(LEGACY_QUOTAS)
    }
    empty = [name for name, pool in pools.items() if not len(pool)]
    if empty:
        raise ValueError("legacy audit unexpectedly has empty pool: " + ", ".join(empty))
    updates = 1 if smoke else int(math.ceil(len(corpus) / 4096))
    selected_count = np.zeros(len(corpus), dtype=np.uint16)
    sequence_count = np.zeros(len(corpus.files), dtype=np.int64)
    observed = {name: 0 for name in LEGACY_QUOTAS}
    for update in range(updates):
        rng = np.random.default_rng(42 * 1_000_003 + update)
        pieces = []
        for name, quota in LEGACY_QUOTAS.items():
            values = pools[name].draw(quota, rng)
            pieces.append(values)
            observed[name] += len(values)
        selected = np.concatenate(pieces)
        np.add.at(selected_count, selected, 1)
        np.add.at(sequence_count, corpus.sequence[selected], 1)
    draws = int(selected_count.sum())
    unique = int(np.count_nonzero(selected_count))
    return {
        "updates": updates,
        "draws": draws,
        "unique_candidates": unique,
        "coverage": unique / len(corpus),
        "repeat_fraction": 1.0 - unique / draws,
        "stratum_draws": observed,
        "sequence_draws": {name: int(value) for name, value in zip(corpus.files, sequence_count)},
        "min_sequence_draws": int(sequence_count.min()),
        "max_sequence_draws": int(sequence_count.max()),
    }


def _stratum_rows(corpus: CandidateCorpus, high: np.ndarray) -> list[Dict[str, Any]]:
    positive = corpus.labels == 1
    result = []
    for density_name, density_mask in (("high", high), ("ordinary", ~high)):
        group_total = int(density_mask.sum())
        group_positive = int((density_mask & positive).sum())
        for label_name, label_mask in (("positive", positive), ("negative", ~positive)):
            mask = density_mask & label_mask
            result.append({
                "stratum": f"{density_name}_{label_name}",
                "candidates": int(mask.sum()),
                "natural_probability": float(mask.mean()),
                "density_positive_rate": float(group_positive / group_total),
            })
    return result


def sampling_diagnostic(corpus: CandidateCorpus, smoke: bool = False) -> Dict[str, Any]:
    sequence_high = corpus.high_density.astype(bool)
    legacy_high = corpus.legacy_window_high_density.astype(bool)
    sequence_rates = np.asarray([
        float(corpus.sequence_rate_eps[first]) for first, _stop in corpus.ranges
    ])
    return {
        "sequence_density_boundary_eps": SEQUENCE_DENSITY_EPS,
        "sequence_distribution": {
            "sequence_count": len(corpus.files),
            "ordinary_count": int((sequence_rates < SEQUENCE_DENSITY_EPS).sum()),
            "ordinary_max_eps": float(sequence_rates[sequence_rates < SEQUENCE_DENSITY_EPS].max()),
            "high_count": int((sequence_rates >= SEQUENCE_DENSITY_EPS).sum()),
            "high_min_eps": float(sequence_rates[sequence_rates >= SEQUENCE_DENSITY_EPS].min()),
            "high_max_eps": float(sequence_rates.max()),
        },
        "new_sequence_strata": _stratum_rows(corpus, sequence_high),
        "legacy_window_strata": _stratum_rows(corpus, legacy_high),
        "legacy_hard_pool_overlap": {
            name: int(mask.sum()) for name, mask in _legacy_masks(corpus).items()
        },
        "legacy_sampler_observed": legacy_sampling_audit(corpus, smoke=smoke),
    }


def dual_threshold_diagnostic(dev_records: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int) -> Dict[str, Any]:
    baseline = evaluate_records(dev_records, None, device, batch_size)
    h2_f = baseline["curve"][str(float(H2_F_THRESHOLD))]
    single_selection = select_h2_r(baseline)
    single_view = working_point_view(baseline, single_selection)
    keys = sorted(baseline["curve"], key=float)
    best = None
    for high_key in keys:
        for ordinary_key in keys:
            scores = []
            for row in baseline["per_sequence"]:
                scores.append(row["scores"][high_key if row["high_density"] else ordinary_key])
            combined = aggregate_scores(scores)
            eligible = (
                combined["localization"]["pd"] >= h2_f["localization"]["pd"]
                and combined["localization"]["fa"] <= h2_f["localization"]["fa"]
            )
            if not eligible:
                continue
            rank = (combined["event"]["iou"], combined["event"]["recall"], -combined["localization"]["fa"])
            if best is None or rank > best[0]:
                best = (rank, high_key, ordinary_key, combined)
    if best is None:
        return {"single_global": single_view, "dual_eligible": False}
    return {
        "single_global": single_view,
        "dual_eligible": True,
        "high_threshold": float(best[1]),
        "ordinary_threshold": float(best[2]),
        "dual_metrics": best[3],
        "iou_gain_over_single": float(best[3]["event"]["iou"] - single_view["overall"]["event"]["iou"]),
        "deployment_allowed": False,
    }


class ProbeMLP(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value.to(torch.float32) / 255.0)


def _density_bounds(records: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    joined = np.log1p(np.concatenate([np.asarray(record["window_rate_eps"], dtype=np.float64) for record in records]))
    return float(joined.min()), float(joined.max())


def probe_features(record: Mapping[str, Any], variant: str, density_bounds: tuple[float, float]) -> np.ndarray:
    base = np.asarray(record["features"], dtype=np.uint8)
    if variant == "base32":
        return base
    if variant == "plus_history":
        return np.ascontiguousarray(np.concatenate((base, np.asarray(record["history"], dtype=np.uint8)), axis=1))
    if variant == "plus_density":
        low, high = density_bounds
        value = np.log1p(np.asarray(record["window_rate_eps"], dtype=np.float64))
        scaled = np.rint(255.0 * np.clip((value - low) / max(high - low, 1.0e-12), 0.0, 1.0)).astype(np.uint8)
        return np.ascontiguousarray(np.column_stack((base, scaled)))
    if variant == "drop_sequence_time":
        return np.ascontiguousarray(np.delete(base, 2, axis=1))
    if variant == "drop_coordinates":
        return np.ascontiguousarray(np.delete(base, (0, 1), axis=1))
    raise KeyError(variant)


def _probe_predictions(model: ProbeMLP, blocks: Sequence[np.ndarray], device: torch.device, batch_size: int) -> list[np.ndarray]:
    model.eval()
    result = []
    with torch.no_grad():
        for values in blocks:
            output = []
            for first in range(0, len(values), batch_size):
                raw = model(torch.from_numpy(values[first:first + batch_size]).to(device))
                output.append(torch.softmax(raw.reshape(-1, 2, 2).amax(dim=2), dim=1)[:, 1].cpu().numpy())
            result.append(np.concatenate(output).astype(np.float32))
    return result


def train_probe(
    variant: str,
    seed: int,
    output_dir: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    epochs: int,
    smoke: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    bounds = _density_bounds(train_records)
    train_blocks = [probe_features(record, variant, bounds) for record in train_records]
    dev_blocks = [probe_features(record, variant, bounds) for record in dev_records]
    joined_x = np.concatenate(train_blocks)
    joined_y = np.concatenate([np.asarray(record["labels"], dtype=np.float32) for record in train_records])
    model = ProbeMLP(joined_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    baseline = evaluate_records(dev_records, None, device, batch_size)
    best_ap = -math.inf
    best_state = None
    best_eval = None
    stale = 0
    history = []
    max_epochs = 1 if smoke else epochs
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        rng = np.random.default_rng(seed * 1_000_003 + epoch)
        indices = rng.permutation(len(joined_y))
        losses = []
        model.train()
        maximum = min(len(indices), batch_size * 2) if smoke else len(indices)
        for first in range(0, maximum, batch_size):
            chosen = indices[first:first + batch_size]
            x = torch.from_numpy(joined_x[chosen]).to(device)
            y = torch.from_numpy(joined_y[chosen]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = candidate_bce(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability = _probe_predictions(model, dev_blocks, device, batch_size)
        evaluation = evaluate_records(dev_records, None, device, batch_size, candidate_probabilities=probability)
        ap = float(evaluation["average_precision"])
        improved = ap > best_ap + 1.0e-8
        if improved:
            best_ap = ap
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_eval = evaluation
            stale = 0
        else:
            stale += 1
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_ap": ap, "best_ap": best_ap, "wall_s": time.perf_counter() - started}
        history.append(row)
        print(f"[D3-0 {variant} seed{seed}] epoch {epoch:02d}/{max_epochs:02d} loss={row['loss']:.6f} dev_ap={ap:.6f} best={best_ap:.6f}", flush=True)
        if stale >= 5:
            break
    if best_state is None or best_eval is None:
        raise RuntimeError("probe did not produce a finite checkpoint")
    checkpoint = output_dir / "best.pt"
    with checkpoint.open("xb") as stream:
        torch.save({"variant": variant, "seed": seed, "state_dict": best_state, "inputs": joined_x.shape[1]}, stream)
    h2_selection = select_h2_r(baseline)
    anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
    selection = __import__("wangdao.h2_research", fromlist=["select_fair_working_point"]).select_fair_working_point(
        best_eval["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]
    )
    summary = {
        "variant": variant,
        "seed": seed,
        "inputs": int(joined_x.shape[1]),
        "best_ap": best_ap,
        "h2_ap": baseline["average_precision"],
        "ap_gain": best_ap - float(baseline["average_precision"]),
        "working_point": working_point_view(best_eval, selection),
        "h2_working_point": working_point_view(baseline, h2_selection),
        "history": history,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
    }
    json_write(output_dir / "summary.json", summary)
    return summary


def run_d3_diagnostics(
    output_dir: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int = 4096,
    probe_epochs: int = 30,
    smoke: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    corpus = CandidateCorpus(train_records)
    sampling = sampling_diagnostic(corpus, smoke=smoke)
    calibration = dual_threshold_diagnostic(dev_records, device, batch_size)
    json_write(output_dir / "sampling_prior_audit.json", sampling)
    json_write(output_dir / "dual_threshold_diagnostic.json", remove_private(calibration))
    probes: Dict[str, Any] = {}
    for variant in PROBE_VARIANTS:
        probes[variant] = {}
        for seed in (42, 43):
            summary = train_probe(
                variant, seed, output_dir / "probes" / f"{variant}-seed{seed}",
                train_records, dev_records, device, batch_size, probe_epochs, smoke,
            )
            probes[variant][str(seed)] = summary
    means = {
        variant: float(np.mean([probes[variant][str(seed)]["best_ap"] for seed in (42, 43)]))
        for variant in PROBE_VARIANTS
    }
    result = {
        "schema_version": 1,
        "status": "smoke_complete" if smoke else "complete",
        "smoke": bool(smoke),
        "validation_used": False,
        "test_used": False,
        "sampling_audit": "sampling_prior_audit.json",
        "dual_threshold": "dual_threshold_diagnostic.json",
        "probe_mean_ap": means,
        "probe_ap_gain_over_base32": {name: value - means["base32"] for name, value in means.items()},
        "interpretation_rule": "A failed finite probe is not an information-theoretic upper bound.",
        "probes": {
            variant: {seed: str(Path("probes") / f"{variant}-seed{seed}" / "summary.json") for seed in probes[variant]}
            for variant in probes
        },
    }
    json_write(output_dir / "summary.json", result)
    return result
