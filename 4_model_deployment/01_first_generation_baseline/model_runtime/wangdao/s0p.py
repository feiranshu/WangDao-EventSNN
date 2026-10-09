"""S0-P 数据入口与训练目标。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from .data import load_events
from .h2.constants import SEQUENCE_DURATION_MS
from .h2.losses import proposal_loss
from .h2.representation import (
    build_patch_features,
    build_patch_targets,
    build_temporal_patch_features,
    build_temporal_patch_targets,
    selected_windows,
)


TRAINING_ROLES = ("train_main", "train_dev")


def sample_patch_indices(
    targets: torch.Tensor,
    positive_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one epoch-sized patch sample with a controlled positive fraction."""
    if targets.ndim != 2:
        raise ValueError("proposal targets must have shape [patch, output]")
    if not 0.0 <= positive_fraction < 1.0:
        raise ValueError("positive_fraction must be in [0, 1)")
    count = len(targets)
    if count == 0:
        return torch.empty((0,), dtype=torch.int64)
    if positive_fraction == 0.0:
        return torch.randperm(count, generator=generator)
    positive_mask = torch.any(targets > 0.5, dim=1)
    positive = torch.nonzero(positive_mask, as_tuple=False).flatten()
    negative = torch.nonzero(~positive_mask, as_tuple=False).flatten()
    if len(positive) == 0 or len(negative) == 0:
        return torch.randperm(count, generator=generator)
    positive_count = int(round(count * positive_fraction))
    negative_count = count - positive_count
    positive_choice = positive[
        torch.randint(len(positive), (positive_count,), generator=generator)
    ]
    if negative_count <= len(negative):
        negative_choice = negative[
            torch.randperm(len(negative), generator=generator)[:negative_count]
        ]
    else:
        negative_choice = negative[
            torch.randint(len(negative), (negative_count,), generator=generator)
        ]
    combined = torch.cat((positive_choice, negative_choice))
    return combined[torch.randperm(len(combined), generator=generator)]


def role_file_paths(index_path: Path, role: str) -> List[Path]:
    """Resolve one training role without ever exposing Validation or Test."""
    if role not in TRAINING_ROLES:
        raise ValueError("S0-P training role must be train_main or train_dev")
    with index_path.open("r", encoding="utf-8") as handle:
        index: Dict[str, Any] = json.load(handle)
    roles = index.get("roles", {})
    names = roles.get(role)
    records = index.get("files")
    location = index.get("location")
    if not isinstance(names, list) or not isinstance(records, list):
        raise ValueError("dataset index is missing role or file records")
    if not isinstance(location, str):
        raise ValueError("dataset index is missing its local location")
    root = index_path.resolve().parent.parent
    location_path = Path(location)
    if location_path.parts and location_path.parts[0] == "data":
        location_path = Path("dataset", *location_path.parts[1:])
    dataset_root = root / location_path
    by_name = {
        Path(str(record["path"])).name: dataset_root / Path(str(record["path"]))
        for record in records
        if isinstance(record, dict) and "path" in record
    }
    missing = [str(name) for name in names if str(name) not in by_name]
    if missing:
        raise ValueError("dataset role references unknown files: " + ", ".join(missing))
    paths = [by_name[str(name)] for name in names]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError("dataset files are absent: " + ", ".join(absent))
    return paths


@dataclass(frozen=True)
class S0PSequence:
    path: Path
    temporal_features: np.ndarray
    ann_features: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class S0PSTCSequence:
    """S0-P arrays with training-only micro-step targets."""

    path: Path
    temporal_features: np.ndarray
    ann_features: np.ndarray
    targets: np.ndarray
    temporal_targets: np.ndarray


def build_s0p_sequence(
    path: Path, time_steps: int, windows: Optional[Sequence[int]] = None
) -> S0PSequence:
    """Load one sequence and build aligned student, teacher and target arrays."""
    events = load_events(path, require_labels=True)
    chosen = (
        selected_windows(0.0, SEQUENCE_DURATION_MS).tolist()
        if windows is None
        else [int(window) for window in windows]
    )
    temporal = build_temporal_patch_features(events, chosen, time_steps)
    ann = build_patch_features(events, chosen)
    targets = build_patch_targets(events, chosen)
    if temporal.shape[0] != ann.shape[0] or ann.shape[:2] != targets.shape[:2]:
        raise AssertionError("S0-P feature and target arrays are not aligned")
    return S0PSequence(
        path=path,
        temporal_features=np.ascontiguousarray(temporal),
        ann_features=np.ascontiguousarray(ann),
        targets=np.ascontiguousarray(targets),
    )


def build_s0p_stc_sequence(path: Path, time_steps: int) -> S0PSTCSequence:
    """Build all arrays needed by the bounded STC training path."""
    events = load_events(path, require_labels=True)
    windows = selected_windows(0.0, SEQUENCE_DURATION_MS).tolist()
    temporal = build_temporal_patch_features(events, windows, time_steps)
    ann = build_patch_features(events, windows)
    targets = build_patch_targets(events, windows)
    temporal_targets = build_temporal_patch_targets(events, windows, time_steps)
    expected = (len(windows), len(targets[0]), len(targets[0, 0]))
    if temporal_targets.shape != (expected[0], time_steps, expected[1], expected[2]):
        raise AssertionError("S0-P STC targets are not aligned")
    return S0PSTCSequence(
        path=path,
        temporal_features=np.ascontiguousarray(temporal),
        ann_features=np.ascontiguousarray(ann),
        targets=np.ascontiguousarray(targets),
        temporal_targets=np.ascontiguousarray(temporal_targets),
    )


class S0PSequenceDataset(Dataset[S0PSequence]):
    """Sequence-level dataset; each item contains all 80 aligned windows."""

    def __init__(self, index_path: Path, role: str, time_steps: int) -> None:
        if time_steps not in (4, 8):
            raise ValueError("time_steps must be 4 or 8")
        self.paths = role_file_paths(index_path, role)
        self.role = role
        self.time_steps = int(time_steps)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> S0PSequence:
        return build_s0p_sequence(self.paths[index], self.time_steps)


def s0p_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor,
    spike_rates: torch.Tensor,
    distillation_weight: float = 0.25,
    spike_rate_weight: float = 0.01,
    minimum_rate: float = 0.01,
    maximum_rate: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Combine GT supervision, H2 distillation and non-degenerate sparsity."""
    if logits.shape != targets.shape or logits.shape != teacher_logits.shape:
        raise ValueError("student, target and teacher tensors must have identical shapes")
    if spike_rates.ndim != 1:
        raise ValueError("spike rates must be a one-dimensional layer summary")
    if distillation_weight < 0.0 or spike_rate_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 <= minimum_rate <= maximum_rate <= 1.0:
        raise ValueError("spike-rate bounds must satisfy 0 <= minimum <= maximum <= 1")

    supervised = proposal_loss(logits, targets)
    teacher_probability = torch.sigmoid(teacher_logits.detach())
    distillation = functional.binary_cross_entropy_with_logits(
        logits, teacher_probability
    )
    too_low = torch.relu(float(minimum_rate) - spike_rates)
    too_high = torch.relu(spike_rates - float(maximum_rate))
    rate_penalty = torch.mean(torch.square(too_low) + torch.square(too_high))
    total = (
        supervised["total"]
        + float(distillation_weight) * distillation
        + float(spike_rate_weight) * rate_penalty
    )
    return {
        "total": total,
        "supervised": supervised["total"],
        "focal": supervised["focal"],
        "soft_dice": supervised["soft_dice"],
        "distillation": distillation,
        "rate_penalty": rate_penalty,
    }
