"""Shared data, evaluation, and artifact helpers for H2 step 2.

The module deliberately exposes only Train-main and Train-dev.  Validation and
Test are not accepted by any public entry point in this directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
for candidate in (HERE / "software", HERE.parents[1] / "implementation" / "software"):
    if (candidate / "wangdao").is_dir():
        sys.path.insert(0, str(candidate))
        break
else:  # pragma: no cover - exercised by server preflight
    raise RuntimeError("Cannot locate the bundled or project wangdao runtime")

from wangdao.h2.constants import WINDOW_MS, WINDOWS_PER_SEQUENCE  # noqa: E402
from wangdao.h2.numpy_model import (  # noqa: E402
    _refiner_probability,
    quantized_mlp,
    quantized_mlp_with_activity,
)
from wangdao.h2.torch_model import prototype_logits  # noqa: E402
from wangdao.h2_research import FrozenH2Pipeline, build_h2_refiner, select_fair_working_point  # noqa: E402
from wangdao.phase1_common import (  # noqa: E402
    PROTOCOL_ID,
    ROOT as RUNTIME_ROOT,
    THRESHOLD_GRID,
    aggregate_scores,
    load_sequence,
    role_paths,
    score_predictions,
    sha256,
)


H2_F_THRESHOLD = 0.2800000011920929
H2_R_THRESHOLD = 0.30
TRAIN_WINDOW_DENSITY_EPS = 6820.0
REPORT_SEQUENCE_DENSITY_EPS = 6889.9906

FEATURE_NAMES = (
    "x", "y", "sequence_time", "polarity",
    "1px_2ms_current_same", "1px_2ms_current_other",
    "1px_2ms_lag1_same", "1px_2ms_lag1_other",
    "1px_2ms_lag2_same", "1px_2ms_lag2_other",
    "2px_5ms_current_same", "2px_5ms_current_other",
    "2px_5ms_lag1_same", "2px_5ms_lag1_other",
    "2px_5ms_lag2_same", "2px_5ms_lag2_other",
    "4px_20ms_current_same", "4px_20ms_current_other",
    "4px_20ms_lag1_same", "4px_20ms_lag1_other",
    "4px_20ms_lag2_same", "4px_20ms_lag2_other",
    "8px_50ms_current_same", "8px_50ms_current_other",
    "8px_50ms_lag1_same", "8px_50ms_lag1_other",
    "8px_50ms_lag2_same", "8px_50ms_lag2_other",
    "proposal_cell_score", "proposal_3x3_mean",
    "proposal_component_size", "within_100ms",
)

FEATURE_GROUPS: Mapping[str, tuple[int, ...]] = {
    "coordinates": (0, 1),
    "sequence_time": (2,),
    "polarity": (3,),
    "counts_1px_2ms": tuple(range(4, 10)),
    "counts_2px_5ms": tuple(range(10, 16)),
    "counts_4px_20ms": tuple(range(16, 22)),
    "counts_8px_50ms": tuple(range(22, 28)),
    "proposal_scores": (28, 29),
    "proposal_component_size": (30,),
    "within_100ms": (31,),
}


def json_write(path: Path, value: Any, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if not overwrite:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        return
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    temporary.replace(path)


def csv_write(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assigned_windows(events: np.ndarray) -> np.ndarray:
    relative = events["t"].astype(np.float64) - float(events["t"][0])
    return np.clip(
        np.floor(relative / WINDOW_MS), 0, WINDOWS_PER_SEQUENCE - 1
    ).astype(np.int16)


def sequence_rate(events: np.ndarray) -> float:
    duration = max((float(events["t"][-1]) - float(events["t"][0])) / 1000.0, 1.0e-9)
    return float(len(events) / duration)


def _cache_identity(path: Path, h2_path: Path) -> str:
    value = {
        "format": 3,
        "data": sha256(path),
        "h2": sha256(h2_path),
        "window_density_eps": TRAIN_WINDOW_DENSITY_EPS,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_sequence_record(
    path: Path,
    pipeline: FrozenH2Pipeline,
    cache_dir: Path,
    include_evaluation_arrays: bool,
) -> Dict[str, Any]:
    """Build or load immutable candidate data for one indexed sequence."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_identity(path, pipeline.h2_path)
    target = cache_dir / f"{path.stem}_{key}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as archive:
            if str(archive["cache_key"].item()) != key:
                raise ValueError("candidate cache key mismatch")
            result = {name: archive[name] for name in archive.files if name != "cache_key"}
        result["file"] = path.name
        return result

    events, loc, labels, identities = load_sequence(path)
    inputs = events[["x", "y", "t", "p"]].copy()
    prepared = pipeline.prepare(inputs)
    inference = pipeline.infer_prepared(prepared, len(events))
    indices = prepared.indices.astype(np.int64, copy=False)
    candidate_labels = labels[indices].astype(np.uint8, copy=False)
    windows_all = assigned_windows(events)
    windows = windows_all[indices]
    counts = np.bincount(windows_all, minlength=WINDOWS_PER_SEQUENCE)
    rates = counts.astype(np.float64) / (WINDOW_MS / 1000.0)
    high_density = rates[windows] >= TRAIN_WINDOW_DENSITY_EPS
    probability = inference["candidate_probabilities"].astype(np.float32, copy=False)
    hard_negative = np.zeros(len(indices), dtype=np.uint8)
    for window in np.unique(windows):
        negative = np.flatnonzero((windows == window) & (candidate_labels == 0))
        if not len(negative):
            continue
        take = max(1, int(math.ceil(0.25 * len(negative))))
        ranked = negative[np.argsort(-probability[negative], kind="stable")]
        hard_negative[ranked[:take]] = 1
    relative_ms = events["t"][indices].astype(np.float64) - float(events["t"][0])
    arrays: Dict[str, np.ndarray] = {
        "features": prepared.features,
        "indices": indices.astype(np.int32),
        "labels": candidate_labels,
        "baseline_probability": probability,
        "windows": windows.astype(np.uint8),
        "high_density": high_density.astype(np.uint8),
        "hard_negative": hard_negative,
        "hard_positive": ((candidate_labels == 1) & high_density & (probability < H2_F_THRESHOLD)).astype(np.uint8),
        "x": events["x"][indices].astype(np.uint16),
        "y": events["y"][indices].astype(np.uint16),
        "t_ms": np.floor(relative_ms).astype(np.int32),
        "sequence_rate_eps": np.asarray(sequence_rate(events), dtype=np.float64),
        "event_count": np.asarray(len(events), dtype=np.int64),
    }
    if include_evaluation_arrays:
        arrays.update(
            all_labels=labels.astype(np.uint8),
            all_identities=identities.astype(np.int32),
            all_loc=loc,
        )
    try:
        with target.open("xb") as stream:
            np.savez_compressed(stream, cache_key=np.asarray(key), **arrays)
    except FileExistsError as error:  # pragma: no cover - concurrent server misuse
        raise RuntimeError("candidate cache was created concurrently; retry") from error
    arrays["file"] = path.name
    return arrays


def load_records(
    role: str,
    cache_dir: Path,
    limit: Optional[int] = None,
    include_evaluation_arrays: Optional[bool] = None,
) -> tuple[FrozenH2Pipeline, List[Dict[str, Any]]]:
    if role not in ("train_main", "train_dev"):
        raise ValueError("step 2 accepts only train_main and train_dev")
    h2_path = RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"
    pipeline = FrozenH2Pipeline(h2_path)
    keep_eval = role == "train_dev" if include_evaluation_arrays is None else include_evaluation_arrays
    records = [
        build_sequence_record(path, pipeline, cache_dir / role, keep_eval)
        for path in role_paths(role, limit=limit)
    ]
    return pipeline, records


class CandidateCorpus:
    """In-memory Train-main candidate arrays and sequence ranges."""

    ARRAY_NAMES = (
        "features", "labels", "baseline_probability", "windows", "high_density",
        "hard_negative", "hard_positive", "x", "y", "t_ms",
    )

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            raise ValueError("candidate corpus cannot be empty")
        self.files: List[str] = []
        self.ranges: List[tuple[int, int]] = []
        start = 0
        for record in records:
            stop = start + len(record["labels"])
            self.files.append(str(record["file"]))
            self.ranges.append((start, stop))
            start = stop
        for name in self.ARRAY_NAMES:
            setattr(self, name, np.concatenate([np.asarray(record[name]) for record in records]))
        self.sequence = np.concatenate(
            [np.full(len(record["labels"]), index, dtype=np.int16) for index, record in enumerate(records)]
        )
        if self.features.shape != (start, 32):
            raise ValueError("candidate feature shape mismatch")
        if not np.all(np.isin(self.labels, (0, 1))):
            raise ValueError("candidate labels must be binary")

    def __len__(self) -> int:
        return len(self.labels)

    def mask_for_group(self, group: str) -> np.ndarray:
        positive = self.labels == 1
        high = self.high_density.astype(bool)
        mapping = {
            "high_positive": high & positive,
            "high_hard_positive": high & positive & self.hard_positive.astype(bool),
            "high_negative": high & ~positive,
            "high_hard_negative": high & ~positive & self.hard_negative.astype(bool),
            "ordinary_positive": ~high & positive,
            "ordinary_negative": ~high & ~positive,
            "ordinary_hard_negative": ~high & ~positive & self.hard_negative.astype(bool),
        }
        if group not in mapping:
            raise KeyError(group)
        return mapping[group]


def make_float_refiner(seed: int = 42) -> torch.nn.Module:
    h2_path = RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"
    pipeline = FrozenH2Pipeline(h2_path)
    return build_h2_refiner("A1", pipeline.h2, seed)


def predict_model(
    model: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
    masked_columns: Sequence[int] = (),
) -> np.ndarray:
    model.eval()
    result = np.empty(len(features), dtype=np.float32)
    columns = tuple(int(value) for value in masked_columns)
    with torch.no_grad():
        for first in range(0, len(features), batch_size):
            block = features[first:first + batch_size].copy()
            if columns:
                block[:, columns] = 0
            tensor = torch.from_numpy(block).to(device)
            probability = torch.softmax(prototype_logits(model(tensor)), dim=1)[:, 1]
            result[first:first + len(block)] = probability.detach().cpu().numpy()
    return result


def _full_probability(record: Mapping[str, Any], candidate_probability: np.ndarray) -> np.ndarray:
    result = np.zeros(int(record["event_count"]), dtype=np.float32)
    result[np.asarray(record["indices"], dtype=np.int64)] = candidate_probability
    return result


def exact_pr(labels: np.ndarray, probability: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    probability = np.asarray(probability, dtype=np.float32)
    if labels.shape != probability.shape:
        raise ValueError("PR labels and probabilities must match")
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    target = labels[order] == 1
    if not len(scores):
        return {"average_precision": None, "target_events": 0, "points": ()}
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(target, dtype=np.int64)[ends]
    positives = int(target.sum())
    precision = tp / (ends + 1)
    recall = tp / positives if positives else np.zeros(len(ends))
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision)) if positives else None
    return {
        "average_precision": ap,
        "target_events": positives,
        "points": (scores[ends], precision.astype(np.float64), recall.astype(np.float64)),
    }


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    model: Optional[torch.nn.Module],
    device: torch.device,
    batch_size: int,
    masked_columns: Sequence[int] = (),
    thresholds: Sequence[float] = THRESHOLD_GRID,
    keep_pr_points: bool = False,
) -> Dict[str, Any]:
    rows = {str(float(value)): [] for value in thresholds}
    group_rows = {
        "high_density": {str(float(value)): [] for value in thresholds},
        "ordinary_density": {str(float(value)): [] for value in thresholds},
    }
    per_sequence = []
    all_labels: List[np.ndarray] = []
    all_probability: List[np.ndarray] = []
    for record in records:
        candidate_probability = (
            np.asarray(record["baseline_probability"], dtype=np.float32)
            if model is None
            else predict_model(model, record["features"], batch_size, device, masked_columns)
        )
        probability = _full_probability(record, candidate_probability)
        labels = np.asarray(record["all_labels"], dtype=np.uint8)
        high = float(record["sequence_rate_eps"]) >= REPORT_SEQUENCE_DENSITY_EPS
        sequence_scores = {}
        for threshold in thresholds:
            key = str(float(threshold))
            score = score_predictions(
                record["all_loc"], labels, record["all_identities"], probability, float(threshold)
            )
            rows[key].append(score)
            group_rows["high_density" if high else "ordinary_density"][key].append(score)
            sequence_scores[key] = score
        per_sequence.append(
            {
                "file": str(record["file"]),
                "high_density": high,
                "events": int(record["event_count"]),
                "candidate_events": int(len(candidate_probability)),
                "scores": sequence_scores,
            }
        )
        all_labels.append(labels)
        all_probability.append(probability)
    labels_joined = np.concatenate(all_labels)
    probability_joined = np.concatenate(all_probability)
    pr = exact_pr(labels_joined, probability_joined)
    result: Dict[str, Any] = {
        "curve": {key: aggregate_scores(value) for key, value in rows.items()},
        "density_curves": {
            group: {
                key: (aggregate_scores(value) if value else None)
                for key, value in curves.items()
            }
            for group, curves in group_rows.items()
        },
        "per_sequence": per_sequence,
        "average_precision": pr["average_precision"],
        "target_events": pr["target_events"],
        "sequence_count": len(records),
        "metric_protocol": PROTOCOL_ID,
    }
    if keep_pr_points:
        result["_pr_points"] = pr["points"]
    return result


def remove_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: remove_private(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [remove_private(item) for item in value]
    return value


def select_h2_r(baseline: Mapping[str, Any]) -> Dict[str, Any]:
    anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
    return select_fair_working_point(
        baseline["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]
    )


def working_point_view(evaluation: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    if not selection.get("eligible"):
        return {"eligible": False, "threshold": None, "reason": selection.get("reason")}
    key = str(float(selection["threshold"]))
    return {
        "eligible": True,
        "threshold": float(selection["threshold"]),
        "overall": evaluation["curve"][key],
        "average_precision": evaluation["average_precision"],
        "high_density": evaluation["density_curves"]["high_density"][key],
        "ordinary_density": evaluation["density_curves"]["ordinary_density"][key],
        "per_sequence": {
            row["file"]: row["scores"][key] for row in evaluation["per_sequence"]
        },
    }


def _at_least(candidate: Optional[float], baseline: Optional[float], delta: float) -> bool:
    return candidate is not None and baseline is not None and candidate >= baseline + delta


def _at_most_ratio(candidate: Optional[float], baseline: Optional[float], ratio: float) -> bool:
    if candidate is None or baseline is None:
        return False
    return candidate <= (0.0 if baseline == 0.0 else baseline * ratio)


def candidate_gates(
    candidate_view: Mapping[str, Any],
    h2_f_view: Mapping[str, Any],
    h2_r_view: Mapping[str, Any],
) -> Dict[str, Any]:
    if not candidate_view.get("eligible"):
        return {"passes": False, "checks": {"working_point_exists": False}}
    c, f, r = candidate_view, h2_f_view, h2_r_view
    checks = {
        "working_point_exists": True,
        "overall_iou_gain_0_5pp": _at_least(c["overall"]["event"]["iou"], r["overall"]["event"]["iou"], 0.005),
        "overall_ap_not_lower": _at_least(c["average_precision"], r["average_precision"], 0.0),
        "overall_pd_loss_within_0_5pp": _at_least(c["overall"]["localization"]["pd"], f["overall"]["localization"]["pd"], -0.005),
        "overall_fa_rise_within_10pct": _at_most_ratio(c["overall"]["localization"]["fa"], f["overall"]["localization"]["fa"], 1.10),
        "high_recall_gain_2pp": _at_least(c["high_density"]["event"]["recall"], r["high_density"]["event"]["recall"], 0.02),
        "high_iou_gain_2pp": _at_least(c["high_density"]["event"]["iou"], r["high_density"]["event"]["iou"], 0.02),
        "high_fa_rise_within_10pct": _at_most_ratio(c["high_density"]["localization"]["fa"], r["high_density"]["localization"]["fa"], 1.10),
        "ordinary_iou_loss_within_0_3pp": _at_least(c["ordinary_density"]["event"]["iou"], r["ordinary_density"]["event"]["iou"], -0.003),
        "ordinary_recall_loss_within_0_3pp": _at_least(c["ordinary_density"]["event"]["recall"], r["ordinary_density"]["event"]["recall"], -0.003),
        "ordinary_pd_loss_within_0_3pp": _at_least(c["ordinary_density"]["localization"]["pd"], r["ordinary_density"]["localization"]["pd"], -0.003),
        "ordinary_fa_rise_within_5pct": _at_most_ratio(c["ordinary_density"]["localization"]["fa"], r["ordinary_density"]["localization"]["fa"], 1.05),
    }
    for file_name in ("train_089.npz", "train_092.npz"):
        checks[f"{file_name}_iou_not_lower"] = _at_least(
            c["per_sequence"][file_name]["event"]["iou"],
            r["per_sequence"][file_name]["event"]["iou"],
            0.0,
        )
    return {"passes": all(checks.values()), "checks": checks}


def checkpoint_protection(
    candidate_view: Mapping[str, Any], h2_f_view: Mapping[str, Any], h2_r_view: Mapping[str, Any]
) -> bool:
    if not candidate_view.get("eligible"):
        return False
    c, f, r = candidate_view, h2_f_view, h2_r_view
    return all(
        (
            _at_least(c["ordinary_density"]["event"]["iou"], r["ordinary_density"]["event"]["iou"], -0.003),
            _at_least(c["ordinary_density"]["event"]["recall"], r["ordinary_density"]["event"]["recall"], -0.003),
            _at_least(c["ordinary_density"]["localization"]["pd"], r["ordinary_density"]["localization"]["pd"], -0.003),
            _at_most_ratio(c["ordinary_density"]["localization"]["fa"], r["ordinary_density"]["localization"]["fa"], 1.05),
            _at_most_ratio(c["overall"]["localization"]["fa"], f["overall"]["localization"]["fa"], 1.10),
            _at_most_ratio(c["high_density"]["localization"]["fa"], r["high_density"]["localization"]["fa"], 1.10),
        )
    )


def checkpoint_rank(view: Mapping[str, Any], epoch: int) -> tuple[float, float, float, float, int]:
    return (
        float(view["overall"]["event"]["iou"]),
        float(view["average_precision"]),
        float(view["high_density"]["event"]["recall"]),
        -float(view["overall"]["localization"]["fa"]),
        -int(epoch),
    )


def quantized_activity(records: Sequence[Mapping[str, Any]], layers: Any) -> List[Dict[str, Any]]:
    zero = None
    full = None
    total = None
    for record in records:
        value = np.asarray(record["features"], dtype=np.uint8)
        layer_zero = [int(np.sum(value == 0))]
        layer_full = [int(np.sum(value == 255))]
        layer_total = [int(value.size)]
        for index, (weight, scale) in enumerate(layers):
            raw = np.matmul(value.astype(np.int32), weight.astype(np.int32))
            floating = raw.astype(np.float64) * scale[None, :].astype(np.float64) / 255.0
            if index + 1 < len(layers):
                value = np.clip(np.rint(np.clip(floating, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
                layer_zero.append(int(np.sum(value == 0)))
                layer_full.append(int(np.sum(value == 255)))
                layer_total.append(int(value.size))
        arrays = (np.asarray(layer_zero), np.asarray(layer_full), np.asarray(layer_total))
        if zero is None:
            zero, full, total = arrays
        else:
            zero += arrays[0]
            full += arrays[1]
            total += arrays[2]
    return [
        {
            "layer": "input" if index == 0 else f"hidden_{index}",
            "zero_fraction": float(zero[index] / total[index]),
            "one_fraction": float(full[index] / total[index]),
            "values": int(total[index]),
        }
        for index in range(len(total))
    ]


def decimate(values: Sequence[np.ndarray], limit: int = 2500) -> tuple[np.ndarray, ...]:
    if not values or len(values[0]) <= limit:
        return tuple(np.asarray(value) for value in values)
    indices = np.unique(np.linspace(0, len(values[0]) - 1, limit, dtype=np.int64))
    return tuple(np.asarray(value)[indices] for value in values)


def svg_curve(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, np.ndarray, np.ndarray, str]],
) -> None:
    width, height = 900, 620
    left, top, plot_w, plot_h = 90, 55, 760, 470
    all_x = np.concatenate([item[1] for item in series if len(item[1])])
    all_y = np.concatenate([item[2] for item in series if len(item[2])])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>',
        f'<text x="{left+plot_w/2}" y="{height-28}" text-anchor="middle">{x_label}</text>',
        f'<text transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle">{y_label}</text>',
    ]
    for tick in range(6):
        x_value = xmin + (xmax - xmin) * tick / 5
        y_value = ymin + (ymax - ymin) * tick / 5
        x_pos = left + plot_w * tick / 5
        y_pos = top + plot_h * (1 - tick / 5)
        lines.extend(
            (
                f'<text x="{x_pos}" y="{top+plot_h+22}" text-anchor="middle" font-size="12">{x_value:.3g}</text>',
                f'<text x="{left-10}" y="{y_pos+4}" text-anchor="end" font-size="12">{y_value:.3g}</text>',
                f'<line x1="{x_pos}" y1="{top}" x2="{x_pos}" y2="{top+plot_h}" stroke="#eee"/>',
                f'<line x1="{left}" y1="{y_pos}" x2="{left+plot_w}" y2="{y_pos}" stroke="#eee"/>',
            )
        )
    for index, (name, x_values, y_values, color) in enumerate(series):
        x_values, y_values = decimate((x_values, y_values), 2500)
        points = " ".join(
            f"{left + plot_w*(float(x)-xmin)/(xmax-xmin):.2f},{top + plot_h*(1-(float(y)-ymin)/(ymax-ymin)):.2f}"
            for x, y in zip(x_values, y_values)
        )
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        lines.append(f'<text x="{left+plot_w-150}" y="{top+20+index*20}" fill="{color}">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_metrics(directory: Path, evaluation: Dict[str, Any], view: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    pr_threshold, pr_precision, pr_recall = evaluation.pop("_pr_points")
    if view.get("eligible") and view.get("threshold") is not None:
        key = str(float(view["threshold"]))
        reporting_kind = "eligible_working_point"
    else:
        # A candidate can legitimately have no threshold satisfying the frozen
        # H2 Pd/Fa fairness constraints.  Preserve that rejection in `view`,
        # but still export diagnostic per-sequence metrics at the unconstrained
        # maximum-IoU threshold instead of crashing on float(None).
        def diagnostic_rank(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float]:
            threshold, metrics = item
            event = metrics["event"]
            return (
                -math.inf if event["iou"] is None else float(event["iou"]),
                -math.inf if event["precision"] is None else float(event["precision"]),
                -abs(float(threshold) - H2_R_THRESHOLD),
            )

        key = max(evaluation["curve"].items(), key=diagnostic_rank)[0]
        reporting_kind = "diagnostic_max_iou_not_an_eligible_working_point"
    export_context = {
        "working_point": view,
        "reporting_threshold": float(key),
        "reporting_threshold_kind": reporting_kind,
        "evaluation": remove_private(evaluation),
    }
    json_write(directory / "summary.json", export_context)
    threshold_rows = []
    for threshold, metrics in evaluation["curve"].items():
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "iou": metrics["event"]["iou"],
                "precision": metrics["event"]["precision"],
                "recall": metrics["event"]["recall"],
                "pd": metrics["localization"]["pd"],
                "fa": metrics["localization"]["fa"],
            }
        )
    csv_write(directory / "threshold_curve.csv", threshold_rows, ("threshold", "iou", "precision", "recall", "pd", "fa"))
    sampled = decimate((pr_threshold, pr_precision, pr_recall), 5000)
    csv_write(
        directory / "pr_curve.csv",
        (
            {"threshold": float(t), "precision": float(p), "recall": float(r)}
            for t, p, r in zip(*sampled)
        ),
        ("threshold", "precision", "recall"),
    )
    per_sequence_rows = []
    for row in evaluation["per_sequence"]:
        metrics = row["scores"][key]
        per_sequence_rows.append(
            {
                "file": row["file"],
                "high_density": row["high_density"],
                "events": row["events"],
                "candidate_events": row["candidate_events"],
                "iou": metrics["event"]["iou"],
                "precision": metrics["event"]["precision"],
                "recall": metrics["event"]["recall"],
                "pd": metrics["localization"]["pd"],
                "fa": metrics["localization"]["fa"],
            }
        )
    csv_write(
        directory / "per_sequence.csv",
        per_sequence_rows,
        ("file", "high_density", "events", "candidate_events", "iou", "precision", "recall", "pd", "fa"),
    )
    svg_curve(
        directory / "pr_curve.svg", "Event PR curve", "Recall", "Precision",
        (("candidate", pr_recall, pr_precision, "#1565c0"),),
    )
    thresholds = np.asarray([row["threshold"] for row in threshold_rows])
    svg_curve(
        directory / "threshold_iou_recall.svg", "Threshold metrics", "Threshold", "Metric",
        (
            ("IoU", thresholds, np.asarray([row["iou"] for row in threshold_rows]), "#2e7d32"),
            ("Recall", thresholds, np.asarray([row["recall"] for row in threshold_rows]), "#c62828"),
        ),
    )

