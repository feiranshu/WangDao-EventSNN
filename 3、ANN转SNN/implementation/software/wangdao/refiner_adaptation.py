"""S1-only offline diagnostics and floating-point Refiner adaptation.

This module deliberately retains the legacy 100 ms, whole-sequence features.
It is an offline B0/B1/B2 experiment, not the causal C0 deployment path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import validate_events
from .fixed import (
    FixedH2, FixedH2Config, FixedProposalSNN, load_fixed_proposal_snn,
    saturate_signed, signed_round_divide, _shift_channels,
)
from .h2.constants import OUTPUT_CELL, PATCHES_PER_WINDOW, WINDOW_MS, WINDOWS_PER_SEQUENCE
from .h2.features import extract_event_features
from .h2.numpy_model import H2Export, _refiner_probability, _sigmoid, load_export
from .h2.representation import (
    _component_sizes, build_patch_features, build_temporal_patch_features,
    candidate_indices, dilate_cells, patch_scores_to_heatmaps, window_indices,
)
from .h2.torch_model import QuantizedMLP, prototype_logits


PROPOSAL_THRESHOLD = 0.51
REFERENCE_THRESHOLDS = (0.28, 0.9)
CACHE_PROTOCOL = "s1-offline-candidates-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class S1LogitsOnly:
    """Exact S1 logits with immutable int64 weights and no output-spike probe."""

    def __init__(self, source: FixedProposalSNN) -> None:
        self.source = source
        self.weights = tuple(np.ascontiguousarray(w.T, dtype=np.int64) for w in source.weights)

    def _current(self, value: np.ndarray, layer: int, fraction: int) -> np.ndarray:
        raw, _ = saturate_signed(
            value.astype(np.int64) @ self.weights[layer],
            self.source.configuration.accumulator_bits,
        )
        shifts = (self.source.configuration.membrane_fraction_bits - fraction
                  - self.source.weight_fraction_bits[layer].astype(np.int64))
        return _shift_channels(raw, shifts)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        source = self.source
        value = np.asarray(features, dtype=np.uint8)
        if value.ndim != 3 or value.shape[0] != source.time_steps or value.shape[2] != 128:
            raise ValueError("S1 input must be [time, batch, 128]")
        state = [np.zeros((value.shape[1], w.shape[0]), dtype=np.int64) for w in source.weights]
        accumulated = np.zeros_like(state[2])
        for step in range(source.time_steps):
            current = self._current(value[step], 0, source.configuration.input_fraction_bits)
            spikes, state[0], _ = source._update(current, state[0], True, 0)
            current = self._current(spikes, 1, 0)
            spikes, state[1], _ = source._update(current, state[1], True, 1)
            current = self._current(spikes, 2, 0)
            _, state[2], _ = source._update(current, state[2], False, 2)
            accumulated += state[2]
        logits_q = signed_round_divide(accumulated, source.time_steps)
        return (logits_q.astype(np.float64)
                / float(1 << source.configuration.membrane_fraction_bits)).astype(np.float32)


class CachedFixedH2:
    """The same integer MLP rule as FixedH2, converting weights once."""

    def __init__(self, source: FixedH2) -> None:
        self.source = source
        self.weights = {
            "proposal": tuple(w.astype(np.int64) for w, _ in source.source.proposal_layers),
            "refiner": tuple(w.astype(np.int64) for w, _ in source.source.refiner_layers),
        }

    def logits(self, features: np.ndarray, stage: str = "refiner") -> np.ndarray:
        if stage not in self.weights:
            raise ValueError("unknown fixed H2 stage")
        cfg = self.source.configuration
        multipliers = getattr(self.source, stage + "_multipliers")
        value = np.asarray(features, dtype=np.uint8)
        weights = self.weights[stage]
        logits_q = np.empty((len(value), weights[-1].shape[1]), dtype=np.int64)
        for layer, (weight, multiplier) in enumerate(zip(weights, multipliers)):
            raw, _ = saturate_signed(value.astype(np.int64) @ weight, cfg.accumulator_bits)
            scaled = raw * multiplier[None, :]
            if layer + 1 < len(weights):
                value = np.clip(signed_round_divide(scaled, 1 << cfg.scale_fraction_bits), 0, 255).astype(np.uint8)
            else:
                logits_q = signed_round_divide(
                    scaled * (1 << cfg.logit_fraction_bits), (1 << cfg.scale_fraction_bits) * 255,
                )
        return (logits_q.astype(np.float64) / float(1 << cfg.logit_fraction_bits)).astype(np.float32)


def refiner_features_shared(
    events: np.ndarray, indices: np.ndarray, windows: Sequence[int],
    heatmaps: np.ndarray, threshold: float, base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exactly reproduce legacy Refiner features while reusing label-free base features."""
    if len(indices) == 0:
        return np.empty((0, 32), dtype=np.uint8)
    base = extract_event_features(events)[0] if base is None else base
    if base.shape != (len(events), 28):
        raise ValueError("base features must have shape [events, 28]")
    lookup = np.full(WINDOWS_PER_SEQUENCE, -1, dtype=np.int64)
    lookup[np.asarray(windows, dtype=np.int64)] = np.arange(len(windows))
    local = lookup[window_indices(events)[indices]]
    if np.any(local < 0):
        raise ValueError("selected events must belong to selected windows")
    x = events["x"][indices].astype(np.int64) // OUTPUT_CELL
    y = events["y"][indices].astype(np.int64) // OUTPUT_CELL
    mask = dilate_cells(heatmaps >= threshold)
    padded = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge")
    neighbor = np.zeros_like(heatmaps)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor += padded[:, 1 + dy:1 + dy + heatmaps.shape[1],
                               1 + dx:1 + dx + heatmaps.shape[2]] / 9.0
    sizes = np.stack([_component_sizes(item) for item in mask])
    within = np.mod(events["t"][indices].astype(np.float64) - float(events["t"][0]), WINDOW_MS) / WINDOW_MS
    context = np.column_stack((
        np.rint(255.0 * heatmaps[local, y, x]),
        np.rint(255.0 * neighbor[local, y, x]),
        np.rint(255.0 * np.minimum(sizes[local, y, x].astype(np.float64) / 64.0, 1.0)),
        np.rint(255.0 * np.clip(within, 0.0, 1.0)),
    )).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate((base[indices], context), axis=1))


@dataclass
class FrozenCandidates:
    indices: np.ndarray
    features: np.ndarray
    heatmaps: np.ndarray
    base_features: np.ndarray
    windows: Tuple[int, ...]
    timings: Dict[str, float]


class FrozenS1Pipeline:
    """Frozen S1 front end and H2 fixed Refiner; labels are never consumed."""

    def __init__(self, h2_path: Path, s1_path: Path, batch_size: int = 4096) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.h2_path, self.s1_path = h2_path, s1_path
        self.h2 = load_export(h2_path)
        self.s1, self.configuration, self.metadata = load_fixed_proposal_snn(s1_path)
        if not np.isclose(float(self.metadata["proposal_threshold"]), PROPOSAL_THRESHOLD, atol=1e-7):
            raise ValueError("B experiments require frozen Proposal threshold 0.51")
        config = FixedH2Config(**self.metadata["h2_fixed_configuration"])
        self.fixed = FixedH2(self.h2, config)
        self.fast_fixed = CachedFixedH2(self.fixed)
        self.fast_s1 = S1LogitsOnly(self.s1)
        self.batch_size = batch_size

    def prepare(self, events: np.ndarray, windows: Optional[Sequence[int]] = None) -> FrozenCandidates:
        validate_events(events)
        chosen = tuple(range(WINDOWS_PER_SEQUENCE)) if windows is None else tuple(windows)
        if not chosen or len(set(chosen)) != len(chosen) or min(chosen) < 0 or max(chosen) >= WINDOWS_PER_SEQUENCE:
            raise ValueError("windows must be unique valid legacy window indices")
        timings: Dict[str, float] = {}
        started = time.perf_counter()
        temporal = build_temporal_patch_features(events, chosen, self.s1.time_steps)
        flat = temporal.transpose(1, 0, 2, 3).reshape(self.s1.time_steps, -1, 128)
        base = extract_event_features(events)[0]
        timings["representation_s"] = time.perf_counter() - started
        started = time.perf_counter()
        logits = np.concatenate([
            self.fast_s1(flat[:, i:i + self.batch_size]) for i in range(0, flat.shape[1], self.batch_size)
        ])
        heatmaps = patch_scores_to_heatmaps(_sigmoid(logits).reshape(len(chosen), PATCHES_PER_WINDOW, 16))
        timings["proposal_s"] = time.perf_counter() - started
        started = time.perf_counter()
        indices = candidate_indices(events, chosen, heatmaps, PROPOSAL_THRESHOLD)
        timings["routing_s"] = time.perf_counter() - started
        started = time.perf_counter()
        features = refiner_features_shared(events, indices, chosen, heatmaps, PROPOSAL_THRESHOLD, base)
        timings["refiner_features_s"] = time.perf_counter() - started
        return FrozenCandidates(indices, features, heatmaps, base, chosen, timings)

    def fixed_probability(self, features: np.ndarray) -> np.ndarray:
        if not len(features):
            return np.empty((0,), dtype=np.float32)
        raw = np.concatenate([
            self.fast_fixed.logits(features[i:i + self.batch_size]) for i in range(0, len(features), self.batch_size)
        ])
        return _refiner_probability(raw, self.h2.prototypes)

    def infer(self, events: np.ndarray, windows: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        prepared = self.prepare(events, windows)
        refiner_started = time.perf_counter()
        candidate_probability = self.fixed_probability(prepared.features)
        prepared.timings["refiner_s"] = time.perf_counter() - refiner_started
        output_started = time.perf_counter()
        probability = np.zeros(len(events), dtype=np.float32)
        probability[prepared.indices] = candidate_probability
        prepared.timings["output_s"] = time.perf_counter() - output_started
        prepared.timings["pipeline_s"] = time.perf_counter() - started
        return {"probabilities": probability, "prepared": prepared, "timings": prepared.timings}

    def cross_context(self, events: np.ndarray, prepared: FrozenCandidates) -> Dict[str, Any]:
        """Separate diagnostic pass; its cost is excluded from S1 inference timings."""
        features = build_patch_features(events, prepared.windows).reshape(-1, 128)
        raw = np.concatenate([self.fast_fixed.logits(features[i:i + self.batch_size], "proposal")
                              for i in range(0, len(features), self.batch_size)])
        maps = patch_scores_to_heatmaps(_sigmoid(raw).reshape(len(prepared.windows), PATCHES_PER_WINDOW, 16))
        h2_indices = candidate_indices(events, prepared.windows, maps, self.h2.proposal_threshold)
        common = np.intersect1d(prepared.indices, h2_indices, assume_unique=True)
        h2_context = refiner_features_shared(events, common, prepared.windows, maps,
                                             self.h2.proposal_threshold, prepared.base_features)
        s1_context = refiner_features_shared(events, common, prepared.windows, prepared.heatmaps,
                                             PROPOSAL_THRESHOLD, prepared.base_features)
        return {"common_indices": common, "h2_indices": h2_indices,
                "h2_context_probability": self.fixed_probability(h2_context),
                "s1_context_probability": self.fixed_probability(s1_context)}


def build_refiner(variant: str, exported: H2Export, seed: int = 42) -> QuantizedMLP:
    if variant not in ("B1", "B2"):
        raise ValueError("variant must be B1 or B2")
    torch.manual_seed(seed)
    model = QuantizedMLP((32, 64, 32, 4) if variant == "B1" else (32, 128, 64, 4), quantized=False)
    if variant == "B1":
        with torch.no_grad():
            for layer, (weight, scale) in zip(model.layers, exported.refiner_layers):
                layer.weight.copy_(torch.from_numpy((weight.astype(np.float32) * scale[None, :]).T.copy()))
    return model


def predict_refiner(model: QuantizedMLP, features: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model.eval()
    probability = np.empty(len(features), dtype=np.float32)
    with torch.no_grad():
        for first in range(0, len(features), batch_size):
            tensor = torch.from_numpy(features[first:first + batch_size].copy()).to(device)
            probability[first:first + batch_size] = torch.softmax(prototype_logits(model(tensor)), 1)[:, 1].cpu().numpy()
    return probability


def full_probability(candidate_indices_: np.ndarray, candidate_probability: np.ndarray, event_count: int) -> np.ndarray:
    indices = np.asarray(candidate_indices_, dtype=np.int64)
    if len(indices) != len(candidate_probability) or len(np.unique(indices)) != len(indices):
        raise ValueError("candidate probabilities must map one-to-one to event indices")
    if np.any(indices < 0) or np.any(indices >= event_count):
        raise ValueError("candidate index is out of range")
    result = np.zeros(event_count, dtype=np.float32)
    result[indices] = candidate_probability
    return result


def error_decomposition(labels: np.ndarray, indices: np.ndarray, probability: np.ndarray,
                        threshold: float = 0.28, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    target = np.asarray(labels, dtype=bool)
    member = np.ones(len(target), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if probability.shape != target.shape or member.shape != target.shape:
        raise ValueError("labels, probabilities and mask must have matching shape")
    retained = np.zeros(len(target), dtype=bool)
    retained[indices] = True
    predicted = probability >= threshold
    targets = int(np.sum(target & member))
    kept_targets = int(np.sum(target & retained & member))
    tp = int(np.sum(target & predicted & member))
    fp = int(np.sum(~target & predicted & member))
    gate_fn = int(np.sum(target & ~retained & member))
    refiner_fn = int(np.sum(target & retained & ~predicted & member))
    n = int(member.sum())
    return {
        "events": n, "target_events": targets, "candidate_events": int(np.sum(retained & member)),
        "candidate_fraction": float(np.mean(retained[member])) if n else None,
        "proposal_recall": kept_targets / targets if targets else None,
        "refiner_conditional_recall": tp / kept_targets if kept_targets else None,
        "proposal_false_negatives": gate_fn, "refiner_false_negatives": refiner_fn,
        "tp": tp, "fp": fp, "fn": gate_fn + refiner_fn,
        "tn": int(np.sum(~target & ~predicted & member)),
    }


def diagnostic_strata(events: np.ndarray, labels: np.ndarray, identities: np.ndarray) -> Dict[str, np.ndarray]:
    """Fixed diagnostic definitions; labels/IDs are used here, never for inference."""
    windows = window_indices(events).astype(np.int64)
    targets = labels.astype(bool)
    counts = np.bincount(windows, weights=(~targets).astype(np.int64), minlength=WINDOWS_PER_SEQUENCE)
    observed = counts[np.unique(windows)]
    dense_threshold = float(np.quantile(observed, 0.75)) if len(observed) else 0.0
    sparse = np.zeros(len(events), dtype=bool)
    fast = np.zeros(len(events), dtype=bool)
    for identity in np.unique(identities[targets]):
        previous: Optional[Tuple[int, float, float]] = None
        indices = np.flatnonzero((identities == identity) & targets)
        for window in np.unique(windows[indices]):
            selected = indices[windows[indices] == window]
            sparse[selected] = len(selected) <= 10
            cx, cy = float(events["x"][selected].mean()), float(events["y"][selected].mean())
            if previous is not None and int(window) == previous[0] + 1:
                fast[selected] = np.hypot(cx - previous[1], cy - previous[2]) >= 8.0
            previous = (int(window), cx, cy)
    within = np.mod(events["t"].astype(np.float64) - float(events["t"][0]), WINDOW_MS)
    return {"sparse_target_le10_events_per_id_window": sparse,
            "dense_background_ge_sequence_window_q75": counts[windows] >= dense_threshold,
            "fast_target_ge8px_per_adjacent_100ms": fast,
            "window_boundary_within5ms": (within < 5.0) | (within >= WINDOW_MS - 5.0)}


class CandidateDiskCache:
    """Immutable one-sequence-at-a-time disk cache, independent of training RNG."""

    def __init__(self, directory: Path, protocol: Dict[str, Any]) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.protocol = dict(protocol, cache_protocol=CACHE_PROTOCOL)
        self.statistics: Dict[str, Any] = {"hits": 0, "misses": 0, "wait_s": 0.0,
                                           "build_s": 0.0, "peak_sequence_bytes": 0}

    def get(self, path: Path, builder: Any) -> Dict[str, np.ndarray]:
        started = time.perf_counter()
        key = hashlib.sha256(json.dumps(dict(self.protocol, data_sha256=file_sha256(path)),
                                       sort_keys=True).encode("utf-8")).hexdigest()
        target = self.directory / (path.stem + "_" + key + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as archive:
                if str(archive["cache_key"].item()) != key:
                    raise ValueError("cache key mismatch")
                arrays = {name: archive[name] for name in archive.files if name != "cache_key"}
            self.statistics["hits"] += 1
        else:
            build_started = time.perf_counter()
            arrays = builder(path)
            self.statistics["build_s"] += time.perf_counter() - build_started
            # Exclusive creation keeps concurrent runs from overwriting evidence.
            try:
                with target.open("xb") as handle:
                    np.savez(handle, cache_key=np.asarray(key), **arrays)
            except FileExistsError:
                raise RuntimeError("cache is being built by another process; retry after it finishes")
            self.statistics["misses"] += 1
        self.statistics["wait_s"] += time.perf_counter() - started
        self.statistics["peak_sequence_bytes"] = max(
            self.statistics["peak_sequence_bytes"], sum(array.nbytes for array in arrays.values()),
        )
        return arrays


def torch_synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
