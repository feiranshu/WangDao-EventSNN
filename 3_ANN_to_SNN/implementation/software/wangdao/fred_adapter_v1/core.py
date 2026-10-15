"""Geometry and explicitly weak box-derived event labels, in source time."""
from __future__ import annotations

import csv
from decimal import Decimal
import io
import re

import numpy as np

EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<f8"),
                       ("p", "u1"), ("label", "u1"), ("name", "<i4")])
PROTOCOL = {
    "version": "fred_real_boxweak_v1", "source_time_unit": "us", "output_time_unit": "ms",
    "output_size": [346, 260], "shard_duration_us": 10000000,
    "geometry": "uniform letterbox; floor of transformed source pixel centers",
    "label_kind": "weak_interpolated_source_bounding_boxes_not_pixel_ground_truth",
    "label_coordinates": "source pixel centers before coordinate remapping; half-open boxes",
    "interpolation": "linear only between adjacent source frames with stable track-ID sets",
    "missing_frame_annotations": "assume no target when an existing source frame has no box record",
    "uncertain": "mask appearance/disappearance intervals, frame gaps, uncovered times and overlapping identities",
    "event_policy": "preserve every decoded event, order, timestamp and polarity; no thinning or denoising",
    "sequence_policy": "all shards of one original recording share one state; never split them across roles",
    "role": "external_adapter_pilot_not_official_test", "training_started": False,
}


def parse_annotations(text, frame_times_us):
    frames = np.asarray(frame_times_us, dtype=np.int64)
    if len(frames) < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError("Need at least two strictly ordered original event-frame timestamps")
    frame_index = {int(t): i for i, t in enumerate(frames)}
    boxes = [{} for _ in frames]
    classes = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        stamp, body = line.split(":", 1)
        stamp_us = int((Decimal(stamp.strip()) * 1000000).to_integral_value())
        if stamp_us not in frame_index:
            raise ValueError(f"Annotation line {line_number} does not match an original frame timestamp")
        if not body.strip():
            continue
        # Support one or several six-field objects, optionally separated by semicolons.
        for group in body.split(";"):
            tokens = next(csv.reader(io.StringIO(group.strip()), skipinitialspace=True))
            if len(tokens) % 6:
                raise ValueError(f"Malformed annotation line {line_number}")
            for start in range(0, len(tokens), 6):
                coordinates = np.array([float(v) for v in tokens[start:start + 4]], dtype=np.float64)
                identity = int(tokens[start + 4])
                category = tokens[start + 5].strip()
                if (not np.isfinite(coordinates).all() or coordinates[2] <= coordinates[0]
                        or coordinates[3] <= coordinates[1]):
                    raise ValueError("Invalid annotation box")
                row = boxes[frame_index[stamp_us]]
                if identity in row:
                    raise ValueError("Repeated track ID in one frame")
                if identity in classes and classes[identity] != category:
                    raise ValueError("Track class changed within a recording")
                row[identity] = coordinates
                classes[identity] = category
    id_map = {identity: i + 1 for i, identity in enumerate(sorted(classes))}
    return {"times": frames, "boxes": boxes, "classes": classes, "id_map": id_map,
            "frame_step_us": float(np.median(np.diff(frames)))}


def letterbox(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid source geometry")
    scale = min(346.0 / width, 260.0 / height)
    return {"source_width": int(width), "source_height": int(height), "scale": scale,
            "offset_x": (346 - width * scale) / 2, "offset_y": (260 - height * scale) / 2}


def validate_raw(raw, width, height, previous_timestamp=None):
    if not len(raw):
        return
    if (np.any(raw["x"] >= width) or np.any(raw["y"] >= height)
            or np.any(raw["x"] < 0) or np.any(raw["y"] < 0)
            or not np.isin(raw["p"], [0, 1]).all() or np.any(raw["t"] < 0)
            or np.any(np.diff(raw["t"]) < 0)
            or (previous_timestamp is not None and raw["t"][0] < previous_timestamp)):
        raise ValueError("Invalid coordinate, polarity, or timestamp in original events")


def label_events(raw, annotations):
    """All events remain inputs; eval_mask distinguishes unsupported ground truth."""
    times = annotations["times"]
    frames = annotations["boxes"]
    t = raw["t"]
    names = np.zeros(len(raw), dtype=np.int32)
    valid = np.zeros(len(raw), dtype=bool)
    memberships = np.zeros(len(raw), dtype=np.uint8)
    left = np.searchsorted(times, t, side="right") - 1
    for index in np.unique(left):
        selected = np.flatnonzero(left == index)
        if index < 0 or index >= len(times):
            continue
        exact = t[selected] == times[index]
        if index + 1 < len(times):
            gap = int(times[index + 1] - times[index])
            common = set(frames[index]).intersection(frames[index + 1])
            stable = (set(frames[index]) == set(frames[index + 1]) and
                      gap <= annotations["frame_step_us"] * 1.5)
            valid[selected] = stable | exact
        else:
            gap, common = 1, set()
            valid[selected] = exact
        x = raw["x"][selected].astype(np.float64) + 0.5
        y = raw["y"][selected].astype(np.float64) + 0.5
        alpha = (t[selected] - times[index]) / gap
        for identity in sorted(frames[index]):
            first = frames[index][identity]
            if identity in common:
                second = frames[index + 1][identity]
                box = first[None, :] + alpha[:, None] * (second - first)[None, :]
                eligible = np.ones(len(selected), dtype=bool)
            else:
                box = np.broadcast_to(first, (len(selected), 4))
                eligible = exact
            hit = (eligible & (x >= box[:, 0]) & (x < box[:, 2]) &
                   (y >= box[:, 1]) & (y < box[:, 3]))
            found = selected[hit]
            memberships[found] += 1
            # Lowest source identity wins only for input-format compatibility; overlaps are masked.
            empty = found[names[found] == 0]
            names[empty] = annotations["id_map"][identity]
    valid[memberships > 1] = False
    return (names > 0).astype(np.uint8), names, valid


def convert_block(raw, annotations, geometry):
    result = np.empty(len(raw), dtype=EVENT_DTYPE)
    for axis in ("x", "y"):
        result[axis] = np.floor((raw[axis].astype(np.float64) + 0.5) * geometry["scale"]
                                + geometry[f"offset_{axis}"]).astype(np.uint16)
    result["t"] = raw["t"].astype(np.float64) / 1000.0
    result["p"] = raw["p"]
    result["label"], result["name"], valid = label_events(raw, annotations)
    return result, valid


def frame_members(names):
    pattern = re.compile(r"/Event/Frames/[^/]*_frame_(\d+)\.png$")
    result = {}
    for name in names:
        match = pattern.search(name)
        if match:
            timestamp = int(match[1])
            if timestamp in result:
                raise ValueError("Duplicate event frame timestamp")
            result[timestamp] = name
    if len(result) < 2:
        raise ValueError("Original timestamped event frames are needed to establish annotation coverage")
    return dict(sorted(result.items()))
