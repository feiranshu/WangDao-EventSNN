"""ANN-only Refiner features, diagnostics, inference helpers, and candidate cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .h2.constants import OUTPUT_CELL, WINDOW_MS, WINDOWS_PER_SEQUENCE
from .h2.features import extract_event_features
from .h2.representation import _component_sizes, dilate_cells, window_indices
from .h2.torch_model import QuantizedMLP, prototype_logits


CACHE_PROTOCOL = "h2-ann-candidates-v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refiner_features_shared(
    events: np.ndarray,
    indices: np.ndarray,
    windows: Sequence[int],
    heatmaps: np.ndarray,
    threshold: float,
    base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reproduce the 32-dimensional ANN Refiner features with shared base features."""
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
            neighbor += padded[
                :, 1 + dy : 1 + dy + heatmaps.shape[1], 1 + dx : 1 + dx + heatmaps.shape[2]
            ] / 9.0
    sizes = np.stack([_component_sizes(item) for item in mask])
    within = (
        np.mod(events["t"][indices].astype(np.float64) - float(events["t"][0]), WINDOW_MS)
        / WINDOW_MS
    )
    context = np.column_stack(
        (
            np.rint(255.0 * heatmaps[local, y, x]),
            np.rint(255.0 * neighbor[local, y, x]),
            np.rint(255.0 * np.minimum(sizes[local, y, x].astype(np.float64) / 64.0, 1.0)),
            np.rint(255.0 * np.clip(within, 0.0, 1.0)),
        )
    ).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate((base[indices], context), axis=1))


def predict_refiner(
    model: QuantizedMLP, features: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model.eval()
    probability = np.empty(len(features), dtype=np.float32)
    with torch.no_grad():
        for first in range(0, len(features), batch_size):
            tensor = torch.from_numpy(features[first : first + batch_size].copy()).to(device)
            probability[first : first + batch_size] = (
                torch.softmax(prototype_logits(model(tensor)), 1)[:, 1].cpu().numpy()
            )
    return probability


def full_probability(
    candidate_indices_: np.ndarray, candidate_probability: np.ndarray, event_count: int
) -> np.ndarray:
    indices = np.asarray(candidate_indices_, dtype=np.int64)
    if len(indices) != len(candidate_probability) or len(np.unique(indices)) != len(indices):
        raise ValueError("candidate probabilities must map one-to-one to event indices")
    if np.any(indices < 0) or np.any(indices >= event_count):
        raise ValueError("candidate index is out of range")
    result = np.zeros(event_count, dtype=np.float32)
    result[indices] = candidate_probability
    return result


def error_decomposition(
    labels: np.ndarray,
    indices: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.28,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
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
    proposal_fn = int(np.sum(target & ~retained & member))
    refiner_fn = int(np.sum(target & retained & ~predicted & member))
    n = int(member.sum())
    return {
        "events": n,
        "target_events": targets,
        "candidate_events": int(np.sum(retained & member)),
        "candidate_fraction": float(np.mean(retained[member])) if n else None,
        "proposal_recall": kept_targets / targets if targets else None,
        "refiner_conditional_recall": tp / kept_targets if kept_targets else None,
        "proposal_false_negatives": proposal_fn,
        "refiner_false_negatives": refiner_fn,
        "tp": tp,
        "fp": fp,
        "fn": proposal_fn + refiner_fn,
        "tn": int(np.sum(~target & ~predicted & member)),
    }


def diagnostic_strata(
    events: np.ndarray, labels: np.ndarray, identities: np.ndarray
) -> Dict[str, np.ndarray]:
    """Fixed error strata; labels and identities are never inference inputs."""
    windows = window_indices(events).astype(np.int64)
    targets = labels.astype(bool)
    counts = np.bincount(
        windows, weights=(~targets).astype(np.int64), minlength=WINDOWS_PER_SEQUENCE
    )
    observed = counts[np.unique(windows)]
    dense_threshold = float(np.quantile(observed, 0.75)) if len(observed) else 0.0
    sparse = np.zeros(len(events), dtype=bool)
    fast = np.zeros(len(events), dtype=bool)
    for identity in np.unique(identities[targets]):
        previous: Optional[Tuple[int, float, float]] = None
        identity_indices = np.flatnonzero((identities == identity) & targets)
        for window in np.unique(windows[identity_indices]):
            selected = identity_indices[windows[identity_indices] == window]
            sparse[selected] = len(selected) <= 10
            cx = float(events["x"][selected].mean())
            cy = float(events["y"][selected].mean())
            if previous is not None and int(window) == previous[0] + 1:
                fast[selected] = np.hypot(cx - previous[1], cy - previous[2]) >= 8.0
            previous = (int(window), cx, cy)
    within = np.mod(events["t"].astype(np.float64) - float(events["t"][0]), WINDOW_MS)
    return {
        "sparse_target_le10_events_per_id_window": sparse,
        "dense_background_ge_sequence_window_q75": counts[windows] >= dense_threshold,
        "fast_target_ge8px_per_adjacent_100ms": fast,
        "window_boundary_within5ms": (within < 5.0) | (within >= WINDOW_MS - 5.0),
    }


class CandidateDiskCache:
    """Immutable one-sequence-at-a-time ANN candidate cache, independent of training RNG."""

    def __init__(self, directory: Path, protocol: Dict[str, Any]) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.protocol = dict(protocol, cache_protocol=CACHE_PROTOCOL)
        self.statistics: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "wait_s": 0.0,
            "build_s": 0.0,
            "peak_sequence_bytes": 0,
        }

    def get(self, path: Path, builder: Any) -> Dict[str, np.ndarray]:
        started = time.perf_counter()
        key = hashlib.sha256(
            json.dumps(
                dict(self.protocol, data_sha256=file_sha256(path)), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        target = self.directory / (path.stem + "_" + key + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as archive:
                if str(archive["cache_key"].item()) != key:
                    raise ValueError("cache key mismatch")
                arrays = {
                    name: archive[name] for name in archive.files if name != "cache_key"
                }
            self.statistics["hits"] += 1
        else:
            build_started = time.perf_counter()
            arrays = builder(path)
            self.statistics["build_s"] += time.perf_counter() - build_started
            try:
                with target.open("xb") as handle:
                    np.savez(handle, cache_key=np.asarray(key), **arrays)
            except FileExistsError as error:
                raise RuntimeError(
                    "cache is being built by another process; retry after it finishes"
                ) from error
            self.statistics["misses"] += 1
        self.statistics["wait_s"] += time.perf_counter() - started
        self.statistics["peak_sequence_bytes"] = max(
            self.statistics["peak_sequence_bytes"],
            sum(array.nbytes for array in arrays.values()),
        )
        return arrays


def torch_synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
