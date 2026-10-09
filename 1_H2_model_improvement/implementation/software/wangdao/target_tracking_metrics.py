"""Observable event-track extraction and HOTA evaluation for EV-UAV.

This module deliberately keeps model inference out of the metric.  A caller
provides one probability for every input event; ground-truth labels are used
only after predicted components and track identities have been constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


WINDOW_MS = 50.0
HOTA_ALPHAS = np.arange(0.05, 0.96, 0.05, dtype=np.float64)
PROTOCOL_ID = "evuav-observable-event-track-hota-v1"


@dataclass(frozen=True)
class Detection:
    """One observable target support box in one 50 ms window."""

    frame: int
    identity: int
    bbox: Tuple[float, float, float, float]
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class _TrackState:
    identity: int
    last_frame: int
    center: np.ndarray
    velocity: np.ndarray


def frame_indices(events: np.ndarray, window_ms: float = WINDOW_MS) -> np.ndarray:
    """Assign every event exactly once using left-closed, right-open windows."""

    if len(events) == 0:
        return np.empty((0,), dtype=np.int64)
    relative = events["t"].astype(np.float64, copy=False) - float(events["t"][0])
    # A tiny positive tolerance avoids a floating representation just below an
    # exact millisecond boundary without changing non-boundary assignments.
    return np.floor((relative + 1e-9) / float(window_ms)).astype(np.int64)


def _bbox(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    return (float(x.min()), float(y.min()), float(x.max()) + 1.0, float(y.max()) + 1.0)


def ground_truth_detections(events: np.ndarray) -> List[Detection]:
    """Build one event-support box for each positive identity and window."""

    frames = frame_indices(events)
    positive = (events["label"].astype(np.int64, copy=False) == 1)
    identities = events["name"].astype(np.int64, copy=False)
    result: List[Detection] = []
    for frame in np.unique(frames[positive]):
        frame_mask = positive & (frames == frame)
        for identity in np.unique(identities[frame_mask]):
            member = frame_mask & (identities == identity)
            result.append(
                Detection(
                    int(frame),
                    int(identity),
                    _bbox(events["x"][member], events["y"][member]),
                    1.0,
                )
            )
    return result


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _spatial_components(x: np.ndarray, y: np.ndarray, radius: int) -> List[np.ndarray]:
    """Cluster occupied pixels by Chebyshev distance, retaining duplicates."""

    coordinates = np.column_stack((x.astype(np.int64), y.astype(np.int64)))
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    union = _UnionFind(len(unique))
    lookup = {(int(px), int(py)): index for index, (px, py) in enumerate(unique)}
    offsets = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if dx or dy
    ]
    for index, (px, py) in enumerate(unique):
        for dx, dy in offsets:
            other = lookup.get((int(px) + dx, int(py) + dy))
            if other is not None and other > index:
                union.union(index, other)
    grouped: Dict[int, List[int]] = {}
    for event_index, pixel_index in enumerate(inverse):
        grouped.setdefault(union.find(int(pixel_index)), []).append(event_index)
    return [np.asarray(indices, dtype=np.int64) for _, indices in sorted(grouped.items())]


def predicted_components(
    events: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cluster_radius: int,
) -> List[Detection]:
    """Build prediction boxes without consulting event labels or identities."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(events),):
        raise ValueError("probabilities must contain exactly one value per event")
    if cluster_radius < 1:
        raise ValueError("cluster_radius must be positive")
    frames = frame_indices(events)
    positive_indices = np.flatnonzero(probabilities >= float(threshold))
    result: List[Detection] = []
    temporary_identity = 1
    for frame in np.unique(frames[positive_indices]):
        indices = positive_indices[frames[positive_indices] == frame]
        components = _spatial_components(events["x"][indices], events["y"][indices], cluster_radius)
        frame_detections = []
        for component in components:
            members = indices[component]
            box = _bbox(events["x"][members], events["y"][members])
            frame_detections.append(
                Detection(
                    int(frame),
                    temporary_identity,
                    box,
                    float(probabilities[members].max()),
                )
            )
            temporary_identity += 1
        result.extend(sorted(frame_detections, key=lambda item: (item.center, item.bbox)))
    return result


def linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact rectangular Hungarian assignment without SciPy.

    Non-finite edges are treated as forbidden.  The large finite replacement
    makes the solution maximise the number of allowed pairs before minimising
    their cost; forbidden assignments are removed from the returned result.
    """

    original = np.asarray(cost, dtype=np.float64)
    if original.ndim != 2:
        raise ValueError("cost matrix must be two-dimensional")
    rows, cols = original.shape
    if rows == 0 or cols == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    transposed = rows > cols
    matrix = original.T.copy() if transposed else original.copy()
    finite = np.isfinite(matrix)
    if np.any(finite):
        scale = max(1.0, float(np.max(np.abs(matrix[finite]))))
        forbidden = scale * (matrix.shape[0] + matrix.shape[1] + 1) * 1e6
    else:
        forbidden = 1e12
    matrix[~finite] = forbidden
    n, m = matrix.shape
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for row in range(1, n + 1):
        p[0] = row
        min_value = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, m + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(m + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assigned_rows = []
    assigned_cols = []
    for column in range(1, m + 1):
        if p[column]:
            assigned_rows.append(int(p[column]) - 1)
            assigned_cols.append(column - 1)
    row_array = np.asarray(assigned_rows, dtype=np.int64)
    col_array = np.asarray(assigned_cols, dtype=np.int64)
    if transposed:
        row_array, col_array = col_array, row_array
    allowed = np.isfinite(original[row_array, col_array])
    return row_array[allowed], col_array[allowed]


def assign_track_identities(
    detections: Sequence[Detection],
    distance_per_window: float,
    max_missed_windows: int,
) -> List[Detection]:
    """Causally associate components with a constant-velocity track state."""

    if distance_per_window <= 0 or max_missed_windows < 0:
        raise ValueError("invalid tracking parameters")
    by_frame = _by_frame(detections)
    active: Dict[int, _TrackState] = {}
    next_identity = 1
    result: List[Detection] = []
    for frame in sorted(by_frame):
        active = {
            identity: state
            for identity, state in active.items()
            if frame - state.last_frame - 1 <= max_missed_windows
        }
        current = sorted(by_frame[frame], key=lambda item: (item.center, item.bbox))
        track_ids = sorted(active)
        cost = np.full((len(track_ids), len(current)), np.inf, dtype=np.float64)
        for row, identity in enumerate(track_ids):
            state = active[identity]
            elapsed = frame - state.last_frame
            predicted = state.center + state.velocity * elapsed
            gate = distance_per_window * elapsed
            for column, detection in enumerate(current):
                distance = float(np.linalg.norm(np.asarray(detection.center) - predicted))
                if distance <= gate:
                    cost[row, column] = distance
        matched_rows, matched_cols = linear_sum_assignment(cost)
        matched_by_col = {int(column): int(row) for row, column in zip(matched_rows, matched_cols)}
        for column, detection in enumerate(current):
            center = np.asarray(detection.center, dtype=np.float64)
            if column in matched_by_col:
                identity = track_ids[matched_by_col[column]]
                previous = active[identity]
                elapsed = frame - previous.last_frame
                velocity = (center - previous.center) / elapsed
            else:
                identity = next_identity
                next_identity += 1
                velocity = np.zeros(2, dtype=np.float64)
            active[identity] = _TrackState(identity, frame, center, velocity)
            result.append(Detection(frame, identity, detection.bbox, detection.confidence))
    return result


def bbox_iou(left: Detection, right: Detection) -> float:
    ax1, ay1, ax2, ay2 = left.bbox
    bx1, by1, bx2, by2 = right.bbox
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    left_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    right_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _by_frame(detections: Sequence[Detection]) -> Dict[int, List[Detection]]:
    result: Dict[int, List[Detection]] = {}
    for detection in detections:
        result.setdefault(int(detection.frame), []).append(detection)
    return result


def _id_index(detections: Sequence[Detection]) -> Tuple[List[int], Dict[int, int]]:
    values = sorted({int(item.identity) for item in detections})
    return values, {value: index for index, value in enumerate(values)}


def _similarity(gt: Sequence[Detection], pred: Sequence[Detection]) -> np.ndarray:
    return np.asarray([[bbox_iou(left, right) for right in pred] for left in gt], dtype=np.float64)


def hota_raw(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    alphas: np.ndarray = HOTA_ALPHAS,
) -> Dict[str, Any]:
    """Return additive HOTA statistics for one independent sequence."""

    gt_values, gt_lookup = _id_index(ground_truth)
    pred_values, pred_lookup = _id_index(predictions)
    gt_count = np.zeros((len(gt_values), 1), dtype=np.float64)
    pred_count = np.zeros((1, len(pred_values)), dtype=np.float64)
    for item in ground_truth:
        gt_count[gt_lookup[item.identity], 0] += 1
    for item in predictions:
        pred_count[0, pred_lookup[item.identity]] += 1
    potential = np.zeros((len(gt_values), len(pred_values)), dtype=np.float64)
    gt_by_frame = _by_frame(ground_truth)
    pred_by_frame = _by_frame(predictions)
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    cached: Dict[int, Tuple[List[Detection], List[Detection], np.ndarray]] = {}
    for frame in frames:
        gt = sorted(gt_by_frame.get(frame, []), key=lambda item: item.identity)
        pred = sorted(pred_by_frame.get(frame, []), key=lambda item: item.identity)
        similarity = _similarity(gt, pred) if gt and pred else np.zeros((len(gt), len(pred)))
        cached[frame] = (gt, pred, similarity)
        if not gt or not pred:
            continue
        denom = similarity.sum(axis=0)[None, :] + similarity.sum(axis=1)[:, None] - similarity
        local_alignment = np.divide(similarity, denom, out=np.zeros_like(similarity), where=denom > 0)
        gt_indices = np.asarray([gt_lookup[item.identity] for item in gt], dtype=np.int64)
        pred_indices = np.asarray([pred_lookup[item.identity] for item in pred], dtype=np.int64)
        potential[np.ix_(gt_indices, pred_indices)] += local_alignment
    alignment_denom = gt_count + pred_count - potential
    global_alignment = np.divide(
        potential, alignment_denom, out=np.zeros_like(potential), where=alignment_denom > 0
    )
    alpha_count = len(alphas)
    true_positive = np.zeros(alpha_count, dtype=np.int64)
    false_negative = np.zeros(alpha_count, dtype=np.int64)
    false_positive = np.zeros(alpha_count, dtype=np.int64)
    localisation_sum = np.zeros(alpha_count, dtype=np.float64)
    match_counts = [np.zeros_like(potential) for _ in alphas]
    for frame in frames:
        gt, pred, similarity = cached[frame]
        if not gt or not pred:
            false_negative += len(gt)
            false_positive += len(pred)
            continue
        gt_indices = np.asarray([gt_lookup[item.identity] for item in gt], dtype=np.int64)
        pred_indices = np.asarray([pred_lookup[item.identity] for item in pred], dtype=np.int64)
        score = global_alignment[np.ix_(gt_indices, pred_indices)] * similarity
        rows, cols = linear_sum_assignment(-score)
        assigned_similarity = similarity[rows, cols]
        for alpha_index, alpha in enumerate(alphas):
            keep = assigned_similarity >= float(alpha) - np.finfo(np.float64).eps
            kept_rows = rows[keep]
            kept_cols = cols[keep]
            matches = len(kept_rows)
            true_positive[alpha_index] += matches
            false_negative[alpha_index] += len(gt) - matches
            false_positive[alpha_index] += len(pred) - matches
            if matches:
                localisation_sum[alpha_index] += float(similarity[kept_rows, kept_cols].sum())
                for row, column in zip(kept_rows, kept_cols):
                    match_counts[alpha_index][gt_indices[row], pred_indices[column]] += 1
    association_accuracy_sum = np.zeros(alpha_count, dtype=np.float64)
    association_recall_sum = np.zeros(alpha_count, dtype=np.float64)
    association_precision_sum = np.zeros(alpha_count, dtype=np.float64)
    for alpha_index, matches in enumerate(match_counts):
        association_denom = gt_count + pred_count - matches
        ass_a = np.divide(matches, association_denom, out=np.zeros_like(matches), where=association_denom > 0)
        ass_r = np.divide(matches, gt_count, out=np.zeros_like(matches), where=gt_count > 0)
        ass_p = np.divide(matches, pred_count, out=np.zeros_like(matches), where=pred_count > 0)
        association_accuracy_sum[alpha_index] = float(np.sum(matches * ass_a))
        association_recall_sum[alpha_index] = float(np.sum(matches * ass_r))
        association_precision_sum[alpha_index] = float(np.sum(matches * ass_p))
    return {
        "alphas": np.asarray(alphas, dtype=np.float64),
        "tp": true_positive,
        "fn": false_negative,
        "fp": false_positive,
        "loc_sum": localisation_sum,
        "assa_sum": association_accuracy_sum,
        "assr_sum": association_recall_sum,
        "assp_sum": association_precision_sum,
        "gt_detections": len(ground_truth),
        "pred_detections": len(predictions),
        "gt_tracks": len(gt_values),
        "pred_tracks": len(pred_values),
    }


def combine_hota_raw(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("at least one sequence is required")
    vector_keys = ("tp", "fn", "fp", "loc_sum", "assa_sum", "assr_sum", "assp_sum")
    result: Dict[str, Any] = {"alphas": np.asarray(items[0]["alphas"], dtype=np.float64).copy()}
    for key in vector_keys:
        result[key] = sum((np.asarray(item[key]) for item in items), np.zeros_like(np.asarray(items[0][key])))
    for key in ("gt_detections", "pred_detections", "gt_tracks", "pred_tracks"):
        result[key] = int(sum(int(item[key]) for item in items))
    return result


def hota_metrics(raw: Mapping[str, Any]) -> Dict[str, Any]:
    tp = np.asarray(raw["tp"], dtype=np.float64)
    fn = np.asarray(raw["fn"], dtype=np.float64)
    fp = np.asarray(raw["fp"], dtype=np.float64)
    det_a = np.divide(tp, tp + fn + fp, out=np.zeros_like(tp), where=(tp + fn + fp) > 0)
    det_r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    det_p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ass_a = np.divide(raw["assa_sum"], tp, out=np.zeros_like(tp), where=tp > 0)
    ass_r = np.divide(raw["assr_sum"], tp, out=np.zeros_like(tp), where=tp > 0)
    ass_p = np.divide(raw["assp_sum"], tp, out=np.zeros_like(tp), where=tp > 0)
    loc_a = np.divide(raw["loc_sum"], tp, out=np.zeros_like(tp), where=tp > 0)
    hota = np.sqrt(det_a * ass_a)
    arrays = {
        "HOTA": hota,
        "DetA": det_a,
        "DetRe": det_r,
        "DetPr": det_p,
        "AssA": ass_a,
        "AssRe": ass_r,
        "AssPr": ass_p,
        "LocA": loc_a,
    }
    return {
        "averaged": {key: float(values.mean()) for key, values in arrays.items()},
        "by_alpha": {
            f"{float(alpha):.2f}": {key: float(values[index]) for key, values in arrays.items()}
            for index, alpha in enumerate(raw["alphas"])
        },
        "counts": {
            "gt_detections": int(raw["gt_detections"]),
            "pred_detections": int(raw["pred_detections"]),
            "gt_tracks": int(raw["gt_tracks"]),
            "pred_tracks": int(raw["pred_tracks"]),
        },
    }


def frame_matches(
    ground_truth: Sequence[Detection], predictions: Sequence[Detection], threshold: float = 0.5
) -> Dict[Tuple[int, str, int], Tuple[int, float]]:
    """Return symmetric per-frame matches for diagnostics and CSV output."""

    result: Dict[Tuple[int, str, int], Tuple[int, float]] = {}
    gt_by_frame = _by_frame(ground_truth)
    pred_by_frame = _by_frame(predictions)
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt = sorted(gt_by_frame.get(frame, []), key=lambda item: item.identity)
        pred = sorted(pred_by_frame.get(frame, []), key=lambda item: item.identity)
        if not gt or not pred:
            continue
        similarity = _similarity(gt, pred)
        rows, cols = linear_sum_assignment(-similarity)
        for row, column in zip(rows, cols):
            value = float(similarity[row, column])
            if value >= threshold:
                result[(frame, "gt", gt[row].identity)] = (pred[column].identity, value)
                result[(frame, "pred", pred[column].identity)] = (gt[row].identity, value)
    return result


def longest_miss_per_track_ms(
    ground_truth: Sequence[Detection], predictions: Sequence[Detection], threshold: float = 0.5
) -> List[float]:
    """Longest miss within consecutive observable GT windows for each GT track."""

    matches = frame_matches(ground_truth, predictions, threshold)
    by_identity: Dict[int, List[int]] = {}
    for item in ground_truth:
        by_identity.setdefault(item.identity, []).append(item.frame)
    result = []
    for identity, raw_frames in sorted(by_identity.items()):
        frames = sorted(set(raw_frames))
        longest = 0
        current = 0
        previous: Optional[int] = None
        for frame in frames:
            if previous is None or frame != previous + 1:
                current = 0
            if (frame, "gt", identity) in matches:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
            previous = frame
        result.append(longest * WINDOW_MS)
    return result


def evaluate_sequence(
    events: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cluster_radius: int,
    association_distance: float,
    max_missed_windows: int,
) -> Dict[str, Any]:
    gt = ground_truth_detections(events)
    components = predicted_components(events, probabilities, threshold, cluster_radius)
    pred = assign_track_identities(components, association_distance, max_missed_windows)
    raw = hota_raw(gt, pred)
    return {
        "ground_truth": gt,
        "predictions": pred,
        "raw": raw,
        "metrics": hota_metrics(raw),
        "longest_miss_per_track_ms": longest_miss_per_track_ms(gt, pred),
        "matches_iou_0_5": frame_matches(gt, pred, 0.5),
    }


def percentile95(values: Iterable[float]) -> Optional[float]:
    materialized = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(materialized, 95)) if len(materialized) else None
