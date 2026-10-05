"""Exact-int8 deployment runtime for the final A4-2 ANN detector."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import numpy as np

from .data import validate_events
from .h2.numpy_model import Layer, _refiner_probability, quantized_mlp
from .h2_research import FrozenH2Pipeline, augment_refiner_features


A4_2_DIMENSIONS = (36, 64, 32, 4)
A4_2_MODEL_ID = "A4-2"
A4_2_SEED = 42
A4_2_EPOCH = 8
A4_2_THRESHOLD = 0.40
A4_2_PROTOTYPES = 2


@dataclass(frozen=True)
class A42Refiner:
    layers: Tuple[Layer, ...]
    threshold: float
    metadata: Dict[str, Any]


def _dimensions(layers: Tuple[Layer, ...]) -> tuple[int, ...]:
    if not layers:
        return ()
    result = [int(layers[0][0].shape[0])]
    for index, (weight, _scale) in enumerate(layers):
        if index and int(weight.shape[0]) != result[-1]:
            raise ValueError("A4-2 refiner layers do not connect")
        result.append(int(weight.shape[1]))
    return tuple(result)


def load_a4_2_refiner(path: Path) -> A42Refiner:
    """Load and strictly validate the frozen A4-2 exact-int8 refiner."""
    with np.load(str(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"format_version", "dimensions", "old_width", "new_width", "metadata_json"}
    for index in range(3):
        required.update((f"layer{index}_weight", f"layer{index}_scale"))
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError("A4-2 archive is missing: " + ", ".join(missing))
    if int(arrays["format_version"].item()) != 1:
        raise ValueError("unsupported A4-2 export format")
    if tuple(int(value) for value in arrays["dimensions"]) != A4_2_DIMENSIONS:
        raise ValueError("A4-2 dimensions do not match 36->64->32->4")
    if int(arrays["old_width"].item()) != 32 or int(arrays["new_width"].item()) != 4:
        raise ValueError("A4-2 input partition must be 32 base plus 4 history features")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    expected = {
        "model_id": A4_2_MODEL_ID,
        "seed": A4_2_SEED,
        "epoch": A4_2_EPOCH,
        "threshold": A4_2_THRESHOLD,
        "history_feature_version": 1,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("A4-2 metadata differs from the final deployment decision")
    layers = []
    for index in range(3):
        weight = arrays[f"layer{index}_weight"]
        scale = arrays[f"layer{index}_scale"]
        if weight.dtype != np.int8 or weight.ndim != 2:
            raise ValueError(f"A4-2 layer {index} weight must be 2D int8")
        if scale.dtype != np.float32 or scale.shape != (weight.shape[1],):
            raise ValueError(f"A4-2 layer {index} scale shape mismatch")
        layers.append((np.ascontiguousarray(weight), np.ascontiguousarray(scale)))
    connected = tuple(layers)
    if _dimensions(connected) != A4_2_DIMENSIONS:
        raise ValueError("A4-2 exported layer dimensions are invalid")
    return A42Refiner(connected, A4_2_THRESHOLD, metadata)


class A42Pipeline:
    """Frozen H2 Proposal followed by the final history-enhanced A4-2 Refiner."""

    def __init__(self, h2_path: Path, refiner_path: Path, batch_size: int = 4096) -> None:
        self.proposal = FrozenH2Pipeline(h2_path, batch_size=batch_size)
        self.refiner = load_a4_2_refiner(refiner_path)

    def infer(self, events: np.ndarray) -> Dict[str, Any]:
        validate_events(events)
        prepared = self.proposal.prepare(events)
        features = augment_refiner_features(prepared.features, prepared.history)
        started = time.perf_counter()
        logits = quantized_mlp(features, self.refiner.layers)
        candidate_probability = _refiner_probability(logits, A4_2_PROTOTYPES)
        refiner_s = time.perf_counter() - started
        probability = np.zeros(len(events), dtype=np.float32)
        probability[prepared.indices] = candidate_probability
        return {
            "probabilities": probability,
            "predictions": probability >= self.refiner.threshold,
            "candidate_indices": prepared.indices,
            "candidate_probabilities": candidate_probability,
            "threshold": self.refiner.threshold,
            "model_id": A4_2_MODEL_ID,
            "timings": dict(prepared.timings, refiner_s=refiner_s),
        }

