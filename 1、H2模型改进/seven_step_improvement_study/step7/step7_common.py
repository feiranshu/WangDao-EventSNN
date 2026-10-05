"""Frozen independent-acceptance helpers for H2 ANN Step 7."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step3"):
    try:
        if (candidate / "step3_common.py").is_file():
            sys.path.insert(0, str(candidate))
            break
    except (OSError, PermissionError):
        continue
else:  # pragma: no cover - reported by preflight in a malformed server package
    raise RuntimeError("Cannot locate bundled step3_common.py")
for candidate in (HERE, HERE.parent / "step4"):
    try:
        if (candidate / "step4_common.py").is_file():
            sys.path.insert(0, str(candidate))
            break
    except (OSError, PermissionError):
        continue
else:  # pragma: no cover
    raise RuntimeError("Cannot locate bundled step4_common.py")
for candidate in (HERE, HERE.parent / "step6"):
    try:
        if (candidate / "step6_common.py").is_file():
            sys.path.insert(0, str(candidate))
            break
    except (OSError, PermissionError):
        continue
else:  # pragma: no cover
    raise RuntimeError("Cannot locate bundled step6_common.py")

from step3_common import (  # noqa: E402
    RUNTIME_ROOT, SEQUENCE_DENSITY_EPS, build_sequence_record, exact_pr,
    json_write, sha256,
)
from step6_common import _ap_replicates  # noqa: E402
from wangdao.h2_research import FrozenH2Pipeline  # noqa: E402


PROTOCOL = "h2-ann-step7-supplemental-independent-acceptance-v1"
ROLES = ("validation", "test")
FIXED_MODEL = "A4-2"
FIXED_SEED = 42
FIXED_EPOCH = 8
FIXED_THRESHOLD = 0.40
H2_F_THRESHOLD = 0.2800000011920929
H2_R_THRESHOLD = 0.30
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261005
FRAME_PIXELS = 346 * 260


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_write(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def acceptance_role_paths(role: str, limit: int | None = None) -> list[Path]:
    """Return only the frozen Validation/Test files after index/hash verification."""
    if role not in ROLES:
        raise ValueError("Step7 accepts only validation and test")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    index_path = RUNTIME_ROOT / "data" / "index.json"
    index = read_json(index_path)
    names = list(index["roles"][role])
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid indexed role: {role}")
    indexed = {row["path"]: row for row in index["files"]}
    prefix = "val" if role == "validation" else "test"
    result: list[Path] = []
    for name in names[:limit]:
        expected_prefix = "val_" if role == "validation" else "test_"
        if Path(name).name != name or not name.startswith(expected_prefix) or not name.endswith(".npz"):
            raise ValueError(f"unexpected {role} filename: {name}")
        key = f"{prefix}/{name}"
        if key not in indexed:
            raise ValueError(f"indexed file record missing: {key}")
        path = RUNTIME_ROOT / "data" / "EV-UAV-dataset" / key
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing indexed data: {path}. Place or link the full dataset at "
                f"{RUNTIME_ROOT / 'data' / 'EV-UAV-dataset'}; do not edit data/index.json."
            )
        if sha256(path) != indexed[key]["sha256"]:
            raise ValueError(f"dataset hash mismatch: {key}")
        result.append(path)
    return result


def load_acceptance_records(
    role: str, cache_dir: Path, limit: int | None = None,
) -> tuple[FrozenH2Pipeline, list[Dict[str, Any]]]:
    h2_path = RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"
    pipeline = FrozenH2Pipeline(h2_path)
    records = [
        build_sequence_record(path, pipeline, cache_dir / role, include_evaluation_arrays=True)
        for path in acceptance_role_paths(role, limit=limit)
    ]
    return pipeline, records


def _value(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def compact_view(view: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "threshold": _value(view.get("threshold")),
        "average_precision": _value(view.get("average_precision")),
    }
    for prefix, group in (("overall", view.get("overall")), ("high", view.get("high_density")), ("ordinary", view.get("ordinary_density"))):
        if group is None:
            result[prefix] = None
            continue
        result[prefix] = {
            "event": {name: group["event"].get(name) for name in ("tp", "fp", "fn", "tn", "iou", "precision", "recall", "f1")},
            "localization": {name: group["localization"].get(name) for name in ("targets", "correct_targets", "false_components", "pd", "fa", "fa_times_1e4")},
        }
    return result


def delta_report(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, Any]:
    paths = {
        "average_precision": ("average_precision",),
        "overall_iou": ("overall", "event", "iou"),
        "overall_precision": ("overall", "event", "precision"),
        "overall_recall": ("overall", "event", "recall"),
        "overall_f1": ("overall", "event", "f1"),
        "overall_pd": ("overall", "localization", "pd"),
        "overall_fa": ("overall", "localization", "fa"),
        "high_iou": ("high_density", "event", "iou"),
        "high_recall": ("high_density", "event", "recall"),
        "high_pd": ("high_density", "localization", "pd"),
        "high_fa": ("high_density", "localization", "fa"),
        "ordinary_iou": ("ordinary_density", "event", "iou"),
        "ordinary_recall": ("ordinary_density", "event", "recall"),
        "ordinary_pd": ("ordinary_density", "localization", "pd"),
        "ordinary_fa": ("ordinary_density", "localization", "fa"),
    }

    def get(row: Mapping[str, Any], path: Sequence[str]) -> float | None:
        value: Any = row
        for key in path:
            if value is None or not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return _value(value)

    result: Dict[str, Any] = {}
    for name, path in paths.items():
        first, second = get(candidate, path), get(reference, path)
        result[name] = None if first is None or second is None else first - second
        if name.endswith("_fa"):
            result[name + "_relative"] = None if first is None or second in (None, 0.0) else first / second - 1.0
    return result


def acceptance_gates(candidate: Mapping[str, Any], h2_f: Mapping[str, Any], h2_r: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the unchanged safety guards; missing density groups are reported, never silently passed."""
    d = delta_report(candidate, h2_r)
    overall_fa = _value(candidate.get("overall", {}).get("localization", {}).get("fa"))
    h2f_fa = _value(h2_f.get("overall", {}).get("localization", {}).get("fa"))
    groups_present = candidate.get("high_density") is not None and candidate.get("ordinary_density") is not None
    checks = {
        "both_density_groups_present": groups_present,
        "overall_iou_strictly_higher": d["overall_iou"] is not None and d["overall_iou"] > 0.0,
        "average_precision_not_lower": d["average_precision"] is not None and d["average_precision"] >= 0.0,
        "overall_pd_loss_within_0_5pp": d["overall_pd"] is not None and d["overall_pd"] >= -0.005,
        "overall_fa_rise_within_10pct_vs_H2_F": overall_fa is not None and h2f_fa is not None and overall_fa <= h2f_fa * 1.10,
        "high_iou_loss_within_0_3pp": d["high_iou"] is not None and d["high_iou"] >= -0.003,
        "high_recall_loss_within_0_3pp": d["high_recall"] is not None and d["high_recall"] >= -0.003,
        "high_fa_rise_within_10pct": d["high_fa_relative"] is not None and d["high_fa_relative"] <= 0.10,
        "ordinary_iou_loss_within_0_3pp": d["ordinary_iou"] is not None and d["ordinary_iou"] >= -0.003,
        "ordinary_recall_loss_within_0_3pp": d["ordinary_recall"] is not None and d["ordinary_recall"] >= -0.003,
        "ordinary_pd_loss_within_0_3pp": d["ordinary_pd"] is not None and d["ordinary_pd"] >= -0.003,
        "ordinary_fa_rise_within_5pct": d["ordinary_fa_relative"] is not None and d["ordinary_fa_relative"] <= 0.05,
    }
    research_targets = {
        "overall_iou_gain_at_least_0_5pp": d["overall_iou"] is not None and d["overall_iou"] >= 0.005,
        "high_recall_gain_at_least_2pp": d["high_recall"] is not None and d["high_recall"] >= 0.02,
        "high_iou_gain_at_least_2pp": d["high_iou"] is not None and d["high_iou"] >= 0.02,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "research_targets_diagnostic_only": research_targets,
        "delta_vs_H2_R": d,
    }


def stratified_weights(high_mask: Sequence[bool], iterations: int, seed: int) -> tuple[np.ndarray, Dict[str, Any]]:
    high_mask = np.asarray(high_mask, dtype=bool)
    if high_mask.ndim != 1 or not len(high_mask):
        raise ValueError("high-density mask must be one-dimensional and nonempty")
    groups = [np.flatnonzero(high_mask), np.flatnonzero(~high_mask)]
    method = "density-stratified sequence paired percentile bootstrap"
    if any(len(group) == 0 for group in groups):
        groups = [np.arange(len(high_mask), dtype=np.int64)]
        method = "unstratified sequence paired percentile bootstrap; one density group absent"
    rng = np.random.default_rng(seed)
    weights = np.zeros((iterations, len(high_mask)), dtype=np.int16)
    row_index = np.arange(iterations)[:, None]
    for group in groups:
        draws = group[rng.integers(0, len(group), size=(iterations, len(group)))]
        np.add.at(weights, (np.broadcast_to(row_index, draws.shape), draws), 1)
    return weights, {
        "method": method,
        "high_density_sequences": int(high_mask.sum()),
        "ordinary_density_sequences": int((~high_mask).sum()),
    }


def _summary(values: np.ndarray) -> Dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": None, "lower_95": None, "upper_95": None}
    return {
        "mean": float(np.mean(finite)),
        "lower_95": float(np.percentile(finite, 2.5)),
        "upper_95": float(np.percentile(finite, 97.5)),
    }


def _ratio(weights: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    top = weights @ np.asarray(numerator, dtype=np.float64)
    bottom = weights @ np.asarray(denominator, dtype=np.float64)
    return np.divide(top, bottom, out=np.full(len(weights), np.nan), where=bottom > 0)


def _counts(view: Mapping[str, Any], files: Sequence[str]) -> Dict[str, np.ndarray]:
    rows = [view["per_sequence"][Path(name).name] for name in files]
    return {
        "tp": np.asarray([row["event"]["tp"] for row in rows]),
        "fp": np.asarray([row["event"]["fp"] for row in rows]),
        "fn": np.asarray([row["event"]["fn"] for row in rows]),
        "correct": np.asarray([row["localization"]["correct_targets"] for row in rows]),
        "targets": np.asarray([row["localization"]["targets"] for row in rows]),
        "false": np.asarray([row["localization"]["false_components"] for row in rows]),
        "fa_denominator": np.asarray([row["localization"]["frame_denominator"] * FRAME_PIXELS for row in rows]),
    }


def _load_scores(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def paired_bootstrap(
    candidate_scores: Path,
    h2_scores: Path,
    candidate_view: Mapping[str, Any],
    h2_view: Mapping[str, Any],
    high_mask: Sequence[bool],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    candidate, baseline = _load_scores(candidate_scores), _load_scores(h2_scores)
    files = [str(value) for value in candidate["sequence_files"]]
    if [Path(str(value)).name for value in baseline["sequence_files"]] != [Path(value).name for value in files]:
        raise ValueError("candidate and H2 sequence order differs")
    if not np.array_equal(candidate["labels"], baseline["labels"]) or not np.array_equal(candidate["sequence"], baseline["sequence"]):
        raise ValueError("candidate and H2 event identities differ")
    weights, metadata = stratified_weights(high_mask, iterations, seed)
    candidate_ap = _ap_replicates(candidate["labels"], candidate["probability"], candidate["sequence"], weights)
    h2_ap = _ap_replicates(baseline["labels"], baseline["probability"], baseline["sequence"], weights)
    cc, hc = _counts(candidate_view, files), _counts(h2_view, files)

    def metric_delta(key: str, numerator: str, denominator: np.ndarray, hdenominator: np.ndarray) -> tuple[str, Dict[str, Any]]:
        return key, _summary(_ratio(weights, cc[numerator], denominator) - _ratio(weights, hc[numerator], hdenominator))

    ciou_denom = cc["tp"] + cc["fp"] + cc["fn"]
    hiou_denom = hc["tp"] + hc["fp"] + hc["fn"]
    cprecision_denom = cc["tp"] + cc["fp"]
    hprecision_denom = hc["tp"] + hc["fp"]
    crecall_denom = cc["tp"] + cc["fn"]
    hrecall_denom = hc["tp"] + hc["fn"]
    result = dict((metric_delta("delta_event_iou", "tp", ciou_denom, hiou_denom),))
    result.update(dict((metric_delta("delta_event_precision", "tp", cprecision_denom, hprecision_denom),)))
    result.update(dict((metric_delta("delta_event_recall", "tp", crecall_denom, hrecall_denom),)))
    result["delta_average_precision"] = _summary(candidate_ap - h2_ap)
    result["delta_pd"] = _summary(_ratio(weights, cc["correct"], cc["targets"]) - _ratio(weights, hc["correct"], hc["targets"]))
    result["delta_fa"] = _summary(_ratio(weights, cc["false"], cc["fa_denominator"]) - _ratio(weights, hc["false"], hc["fa_denominator"]))
    return {
        **metadata,
        "iterations": int(iterations),
        "seed": int(seed),
        "ap_compute_backend": "torch_cuda_float64" if _torch_cuda_available() and iterations > 256 else "numpy_float64",
        **result,
    }


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def export_score_archive(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    candidate_probabilities: Sequence[np.ndarray],
) -> None:
    labels, probability, sequence = [], [], []
    for index, (record, candidate) in enumerate(zip(records, candidate_probabilities)):
        full = np.zeros(int(record["event_count"]), dtype=np.float32)
        full[np.asarray(record["indices"], dtype=np.int64)] = np.asarray(candidate, dtype=np.float32)
        labels.append(np.asarray(record["all_labels"], dtype=np.uint8))
        probability.append(full)
        sequence.append(np.full(len(full), index, dtype=np.int16))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            labels=np.concatenate(labels),
            probability=np.concatenate(probability),
            sequence=np.concatenate(sequence),
            sequence_files=np.asarray([str(record["file"]) for record in records]),
        )


def per_sequence_rows(
    candidate: Mapping[str, Any], h2_f: Mapping[str, Any], h2_r: Mapping[str, Any],
    rates: Mapping[str, float],
) -> list[Dict[str, Any]]:
    rows = []
    for name, c in candidate["per_sequence"].items():
        f, r = h2_f["per_sequence"][name], h2_r["per_sequence"][name]
        rows.append({
            "file": name,
            "density_group": "high" if rates[name] >= SEQUENCE_DENSITY_EPS else "ordinary",
            "events_per_second": rates[name],
            "A4_2_iou": c["event"]["iou"],
            "H2_R_iou": r["event"]["iou"],
            "delta_iou": c["event"]["iou"] - r["event"]["iou"],
            "A4_2_recall": c["event"]["recall"],
            "H2_R_recall": r["event"]["recall"],
            "delta_recall": c["event"]["recall"] - r["event"]["recall"],
            "A4_2_pd": c["localization"]["pd"],
            "H2_F_pd": f["localization"]["pd"],
            "H2_R_pd": r["localization"]["pd"],
            "A4_2_fa": c["localization"]["fa"],
            "H2_F_fa": f["localization"]["fa"],
            "H2_R_fa": r["localization"]["fa"],
        })
    return rows


def comparison_svg(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    metrics = ("average_precision", "overall_iou", "overall_recall", "overall_pd")
    labels = {"average_precision": "AP", "overall_iou": "IoU", "overall_recall": "Recall", "overall_pd": "Pd"}
    colors = {"H2-F": "#64748b", "H2-R": "#2563eb", "A4-2-int8": "#dc2626"}
    width, height = 920, 560
    left, top, plot_w, plot_h = 85, 60, 790, 390
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="21">{title}</text>',
    ]
    for tick in range(6):
        value = 0.2 * tick
        y = top + plot_h * (1.0 - value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.1f}</text>')
    models = [str(row["model"]) for row in rows]
    group_w = plot_w / len(metrics)
    bar_w = min(45.0, group_w / (len(rows) + 1))
    for metric_index, metric in enumerate(metrics):
        center = left + group_w * (metric_index + 0.5)
        lines.append(f'<text x="{center:.1f}" y="{top+plot_h+28}" text-anchor="middle">{labels[metric]}</text>')
        for model_index, row in enumerate(rows):
            value = max(0.0, min(1.0, float(row[metric])))
            x = center + (model_index - (len(rows)-1)/2) * bar_w - bar_w * 0.42
            y = top + plot_h * (1.0 - value)
            h = plot_h * value
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.84:.1f}" height="{h:.1f}" fill="{colors.get(models[model_index], "#111827")}"/>')
    for index, model in enumerate(models):
        x = left + 140 + index * 210
        lines.append(f'<rect x="{x}" y="{height-55}" width="18" height="12" fill="{colors.get(model, "#111827")}"/>')
        lines.append(f'<text x="{x+25}" y="{height-44}" font-size="13">{model}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

