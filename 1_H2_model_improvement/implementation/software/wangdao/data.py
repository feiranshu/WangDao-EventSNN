"""EV-UAV event-array validation and NPZ loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

SENSOR_WIDTH = 346
SENSOR_HEIGHT = 260


INFERENCE_FIELDS = ("x", "y", "t", "p")
EVALUATION_FIELDS = INFERENCE_FIELDS + ("label", "name")


def validate_events(events: np.ndarray, require_labels: bool = False) -> None:
    """Validate fields and ranges used by H2 without imposing unused NPZ keys."""
    required: Iterable[str] = EVALUATION_FIELDS if require_labels else INFERENCE_FIELDS
    if events.ndim != 1 or events.dtype.names is None:
        raise ValueError("events must be a one-dimensional structured array")
    missing = [name for name in required if name not in events.dtype.names]
    if missing:
        raise ValueError("events are missing fields: " + ", ".join(missing))
    if len(events) == 0:
        raise ValueError("event sequence is empty")
    if np.any(np.diff(events["t"].astype(np.float64, copy=False)) < 0.0):
        raise ValueError("event timestamps must be monotonic")
    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    p = events["p"].astype(np.int64, copy=False)
    if x.min() < 0 or x.max() >= SENSOR_WIDTH:
        raise ValueError("x coordinate is outside the 346-pixel sensor width")
    if y.min() < 0 or y.max() >= SENSOR_HEIGHT:
        raise ValueError("y coordinate is outside the 260-pixel sensor height")
    if not set(np.unique(p).tolist()).issubset({0, 1}):
        raise ValueError("event polarity must be binary")
    if require_labels:
        labels = events["label"].astype(np.int64, copy=False)
        identities = events["name"].astype(np.int64, copy=False)
        if not set(np.unique(labels).tolist()).issubset({0, 1}):
            raise ValueError("event labels must be binary")
        if np.any((labels == 0) & (identities != 0)):
            raise ValueError("background events must use identity 0")
        if np.any((labels == 1) & (identities <= 0)):
            raise ValueError("target events must use a positive identity")


def load_events(path: Path, require_labels: bool = False) -> np.ndarray:
    """Load the ``ev`` array from an EV-UAV NPZ file."""
    with np.load(str(path), allow_pickle=False) as archive:
        if "ev" not in archive.files:
            raise ValueError("event package is missing the 'ev' array")
        events = archive["ev"].copy()
    validate_events(events, require_labels=require_labels)
    return events
