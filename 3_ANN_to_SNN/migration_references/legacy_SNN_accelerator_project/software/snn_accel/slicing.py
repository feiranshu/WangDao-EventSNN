from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import EventTable, SliceConfig


@dataclass(frozen=True)
class TargetBox:
    target_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class EventSlice:
    slice_id: int
    start: int
    stop: int
    start_time: int
    end_time: int
    time_bins: np.ndarray
    targets: tuple[TargetBox, ...]


def _slice_stop(events: EventTable, start: int, config: SliceConfig) -> int:
    timestamps = np.asarray(events.t, dtype=np.int64)
    n_events = len(events)
    start_time = int(timestamps[start])
    max_stop = int(np.searchsorted(timestamps, start_time + config.max_duration_us, side="right"))
    max_stop = max(start + 1, min(max_stop, n_events))

    if config.mode == "fixed_time":
        stop = int(np.searchsorted(timestamps, start_time + config.fixed_duration_us, side="right"))
        return max(start + 1, min(stop, max_stop, n_events))

    stop = min(start + config.target_events, n_events, max_stop)
    minimum_end = start_time + config.min_duration_us
    if int(timestamps[stop - 1]) < minimum_end and stop < max_stop:
        stop = int(np.searchsorted(timestamps, minimum_end, side="left")) + 1
        stop = min(max(start + 1, stop), max_stop, n_events)
    return stop


def _time_bins(timestamps: np.ndarray, timesteps: int) -> np.ndarray:
    if len(timestamps) == 0:
        return np.empty(0, dtype=np.uint8)
    start = int(timestamps[0])
    span = max(1, int(timestamps[-1]) - start + 1)
    bins = ((np.asarray(timestamps, dtype=np.int64) - start) * timesteps) // span
    return np.minimum(bins, timesteps - 1).astype(np.uint8)


def _targets(events: EventTable, config: SliceConfig) -> tuple[TargetBox, ...]:
    if events.label is None:
        return ()
    positive = np.asarray(events.label) == 1
    if not np.any(positive):
        return ()
    ids = np.zeros(len(events), dtype=np.int64) if events.name is None else np.asarray(events.name, dtype=np.int64)
    boxes: list[TargetBox] = []
    for target_id in np.unique(ids[positive]):
        mask = positive & (ids == target_id)
        x = np.asarray(events.x, dtype=np.int64)[mask]
        y = np.asarray(events.y, dtype=np.int64)[mask]
        margin = config.bbox_margin
        boxes.append(
            TargetBox(
                target_id=int(target_id),
                x_min=max(0, int(x.min()) - margin),
                y_min=max(0, int(y.min()) - margin),
                x_max=min(config.input_width - 1, int(x.max()) + margin),
                y_max=min(config.input_height - 1, int(y.max()) + margin),
            )
        )
    return tuple(boxes)


def slice_events(events: EventTable, config: SliceConfig) -> tuple[EventSlice, ...]:
    x = np.asarray(events.x, dtype=np.int64)
    y = np.asarray(events.y, dtype=np.int64)
    p = np.asarray(events.p, dtype=np.int64)
    if np.any((x < 0) | (x >= config.input_width)):
        raise ValueError("event x coordinate is outside the sensor width")
    if np.any((y < 0) | (y >= config.input_height)):
        raise ValueError("event y coordinate is outside the sensor height")
    if np.any((p != 0) & (p != 1)):
        raise ValueError("event polarity must be 0 or 1")

    result: list[EventSlice] = []
    start = 0
    while start < len(events):
        stop = _slice_stop(events, start, config)
        view = events.take(start, stop)
        result.append(
            EventSlice(
                slice_id=len(result) & 0xFFFF,
                start=start,
                stop=stop,
                start_time=int(view.t[0]),
                end_time=int(view.t[-1]),
                time_bins=_time_bins(np.asarray(view.t), config.timesteps),
                targets=_targets(view, config),
            )
        )
        start = stop
    return tuple(result)

