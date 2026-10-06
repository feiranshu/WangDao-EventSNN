from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SliceConfig:
    mode: str = "bounded_count"
    input_width: int = 304
    input_height: int = 240
    padded_width: int = 320
    padded_height: int = 256
    target_events: int = 10_000
    min_duration_us: int = 1_000
    max_duration_us: int = 20_000
    fixed_duration_us: int = 10_000
    timesteps: int = 4
    bbox_margin: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"bounded_count", "fixed_time"}:
            raise ValueError("mode must be 'bounded_count' or 'fixed_time'")
        positive = {
            "input_width": self.input_width,
            "input_height": self.input_height,
            "padded_width": self.padded_width,
            "padded_height": self.padded_height,
            "target_events": self.target_events,
            "max_duration_us": self.max_duration_us,
            "fixed_duration_us": self.fixed_duration_us,
            "timesteps": self.timesteps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_duration_us < 0 or self.min_duration_us > self.max_duration_us:
            raise ValueError("min_duration_us must be in [0, max_duration_us]")
        if self.padded_width < self.input_width or self.padded_height < self.input_height:
            raise ValueError("padded shape cannot be smaller than the sensor shape")
        if self.padded_width > 320 or self.padded_height > 256:
            raise ValueError("the frozen AER format supports at most 320x256")
        if self.timesteps > 16:
            raise ValueError("the frozen AER format supports at most 16 time bins")
        if self.bbox_margin < 0:
            raise ValueError("bbox_margin cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SliceConfig":
        return cls(**{field: values[field] for field in cls.__dataclass_fields__ if field in values})


@dataclass(frozen=True)
class EventTable:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    label: np.ndarray | None = None
    name: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = [np.asarray(self.x), np.asarray(self.y), np.asarray(self.t), np.asarray(self.p)]
        length = len(arrays[0])
        if any(array.ndim != 1 or len(array) != length for array in arrays):
            raise ValueError("x, y, t and p must be one-dimensional arrays with equal length")
        for optional_name in ("label", "name"):
            value = getattr(self, optional_name)
            if value is not None and (np.asarray(value).ndim != 1 or len(value) != length):
                raise ValueError(f"{optional_name} must have the same length as x")
        if length and np.any(np.diff(np.asarray(self.t, dtype=np.int64)) < 0):
            raise ValueError("events must be sorted by non-decreasing timestamp")

    def __len__(self) -> int:
        return len(self.x)

    def take(self, start: int, stop: int) -> "EventTable":
        return EventTable(
            x=np.asarray(self.x[start:stop]),
            y=np.asarray(self.y[start:stop]),
            t=np.asarray(self.t[start:stop]),
            p=np.asarray(self.p[start:stop]),
            label=None if self.label is None else np.asarray(self.label[start:stop]),
            name=None if self.name is None else np.asarray(self.name[start:stop]),
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "EventTable":
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[1] < 4:
            raise ValueError("event matrix must have columns x,y,t,p[,label,name]")
        order = np.argsort(matrix[:, 2], kind="stable")
        matrix = matrix[order]
        return cls(
            x=matrix[:, 0].astype(np.int64),
            y=matrix[:, 1].astype(np.int64),
            t=matrix[:, 2].astype(np.int64),
            p=matrix[:, 3].astype(np.int8),
            label=matrix[:, 4].astype(np.int8) if matrix.shape[1] >= 5 else None,
            name=matrix[:, 5].astype(np.int64) if matrix.shape[1] >= 6 else None,
        )


def _timestamps_to_us(values: np.ndarray, unit: str) -> np.ndarray:
    factors = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    if unit not in factors:
        raise ValueError("timestamp_unit must be one of ns, us, ms or s")
    converted = np.rint(np.asarray(values, dtype=np.float64) * factors[unit])
    if np.any(~np.isfinite(converted)):
        raise ValueError("timestamps contain non-finite values")
    return converted.astype(np.int64)


def load_ev_uav_npz(path: str | Path, key: str = "ev", timestamp_unit: str = "us") -> EventTable:
    """Load an EV-UAV style .npz without using labels to alter event ordering.

    The official archive has changed representation across revisions, so this
    accepts either a numeric matrix or a structured array with named fields.
    """

    with np.load(Path(path), allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f"{key!r} is not present; available keys: {archive.files}")
        raw = archive[key]
    if raw.dtype.names:
        names = set(raw.dtype.names)
        required = {"x", "y", "t", "p"}
        if not required.issubset(names):
            raise ValueError(f"structured events require fields {sorted(required)}")
        timestamps = _timestamps_to_us(raw["t"], timestamp_unit)
        order = np.argsort(timestamps, kind="stable")
        return EventTable(
            x=np.asarray(raw["x"][order], dtype=np.int64),
            y=np.asarray(raw["y"][order], dtype=np.int64),
            t=timestamps[order],
            p=np.asarray(raw["p"][order], dtype=np.int8),
            label=np.asarray(raw["label"][order], dtype=np.int8) if "label" in names else None,
            name=np.asarray(raw["name"][order], dtype=np.int64) if "name" in names else None,
        )
    matrix = np.array(raw, copy=True)
    matrix[:, 2] = _timestamps_to_us(matrix[:, 2], timestamp_unit)
    return EventTable.from_matrix(matrix)
