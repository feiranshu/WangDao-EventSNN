"""Feature audit and STC applicability diagnosis for H2 step 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from step2_common import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    H2_R_THRESHOLD,
    REPORT_SEQUENCE_DENSITY_EPS,
    RUNTIME_ROOT,
    csv_write,
    evaluate_records,
    json_write,
    quantized_activity,
    remove_private,
    working_point_view,
)
from step2_training import stc_weights_sparse

from wangdao.h2.numpy_model import _refiner_probability, quantized_mlp
from wangdao.h2_research import FrozenH2Pipeline


def _quantiles(values: np.ndarray) -> Dict[str, Any]:
    if not len(values):
        return {key: None for key in ("min", "p01", "p25", "median", "p75", "p99", "max", "mean")}
    array = np.asarray(values, dtype=np.float64)
    levels = np.quantile(array, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    return {
        "min": float(levels[0]), "p01": float(levels[1]), "p25": float(levels[2]),
        "median": float(levels[3]), "p75": float(levels[4]), "p99": float(levels[5]),
        "max": float(levels[6]), "mean": float(array.mean()),
    }


def _feature_stat_rows(records: Sequence[Mapping[str, Any]], role: str) -> list[Dict[str, Any]]:
    rows = []
    for density_name, density_value in (("high", 1), ("ordinary", 0)):
        pieces = [
            np.asarray(record["features"])[np.asarray(record["high_density"]) == density_value]
            for record in records
        ]
        selected = np.concatenate([piece for piece in pieces if len(piece)])
        for index, name in enumerate(FEATURE_NAMES):
            values = selected[:, index]
            rows.append(
                {
                    "role": role,
                    "density": density_name,
                    "feature_index": index,
                    "feature": name,
                    "samples": len(values),
                    **_quantiles(values),
                    "zero_fraction": float(np.mean(values == 0)),
                    "saturated_fraction": float(np.mean(values == 255)),
                }
            )
    return rows


def _error_feature_rows(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for record in records:
        target = np.asarray(record["labels"]) == 1
        predicted = np.asarray(record["baseline_probability"]) >= H2_R_THRESHOLD
        groups = {
            "TP": target & predicted,
            "FN_in_gate": target & ~predicted,
            "FP": ~target & predicted,
            "TN_in_gate": ~target & ~predicted,
        }
        for group_name, mask in groups.items():
            values = np.asarray(record["features"])[mask]
            for index, name in enumerate(FEATURE_NAMES):
                column = values[:, index] if len(values) else np.empty(0)
                rows.append(
                    {
                        "file": record["file"],
                        "group": group_name,
                        "feature_index": index,
                        "feature": name,
                        "samples": len(column),
                        **_quantiles(column),
                        "zero_fraction": float(np.mean(column == 0)) if len(column) else None,
                        "saturated_fraction": float(np.mean(column == 255)) if len(column) else None,
                    }
                )
    return rows


def _group_correlations(records: Sequence[Mapping[str, Any]], limit: int = 200_000) -> tuple[list[Dict[str, Any]], np.ndarray]:
    # Sample per sequence before concatenation so the audit never materializes
    # the complete training candidate matrix merely to keep 200k rows.
    rng = np.random.default_rng(42)
    per_record = max(1, int(np.ceil(limit / max(1, len(records)))))
    pieces: list[np.ndarray] = []
    for record in records:
        values = np.asarray(record["features"], dtype=np.float32)
        if len(values) > per_record:
            values = values[rng.choice(len(values), size=per_record, replace=False)]
        pieces.append(values)
    all_features = np.concatenate(pieces).astype(np.float64, copy=False)
    if len(all_features) > limit:
        all_features = all_features[rng.choice(len(all_features), size=limit, replace=False)]
    names = tuple(FEATURE_GROUPS)
    group_values = np.column_stack(
        [all_features[:, FEATURE_GROUPS[name]].mean(axis=1) for name in names]
    )
    correlation = np.nan_to_num(np.corrcoef(group_values, rowvar=False), nan=0.0)
    rows = [
        {"group_a": names[i], "group_b": names[j], "pearson": float(correlation[i, j])}
        for i in range(len(names)) for j in range(len(names))
    ]
    return rows, correlation


def _modified_records(
    records: Sequence[Mapping[str, Any]],
    layers: Any,
    group: str,
    mode: str,
    seed: int,
) -> list[Dict[str, Any]]:
    columns = FEATURE_GROUPS[group]
    output = []
    for sequence_index, record in enumerate(records):
        features = np.asarray(record["features"]).copy()
        if mode == "zero":
            features[:, columns] = 0
        elif mode == "permute":
            rng = np.random.default_rng(seed * 1009 + sequence_index)
            windows = np.asarray(record["windows"])
            for window in np.unique(windows):
                selected = np.flatnonzero(windows == window)
                if len(selected) < 2:
                    continue
                shuffled = selected[rng.permutation(len(selected))]
                features[np.ix_(selected, columns)] = features[np.ix_(shuffled, columns)]
        else:
            raise ValueError("feature perturbation mode must be zero or permute")
        raw = quantized_mlp(features, layers)
        probability = _refiner_probability(raw, 2)
        copied = dict(record)
        copied["baseline_probability"] = probability
        output.append(copied)
    return output


def _fixed_view(evaluation: Mapping[str, Any], threshold: float) -> Dict[str, Any]:
    return working_point_view(
        evaluation,
        {"eligible": True, "threshold": float(threshold), "metrics": evaluation["curve"][str(float(threshold))]},
    )


def _ordinary_protection(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    c = candidate["ordinary_density"]
    b = baseline["ordinary_density"]
    fa_limit = 0.0 if b["localization"]["fa"] == 0 else b["localization"]["fa"] * 1.05
    return all(
        (
            c["event"]["iou"] >= b["event"]["iou"] - 0.003,
            c["event"]["recall"] >= b["event"]["recall"] - 0.003,
            c["localization"]["pd"] >= b["localization"]["pd"] - 0.003,
            c["localization"]["fa"] <= fa_limit,
        )
    )


def run_feature_audit(
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    h2_path = RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"
    pipeline = FrozenH2Pipeline(h2_path)
    train_stats = _feature_stat_rows(train_records, "train_main")
    error_stats = _error_feature_rows(dev_records)
    correlations, _ = _group_correlations(train_records)
    csv_write(
        output_dir / "train_feature_statistics.csv", train_stats,
        ("role", "density", "feature_index", "feature", "samples", "min", "p01", "p25", "median", "p75", "p99", "max", "mean", "zero_fraction", "saturated_fraction"),
    )
    csv_write(
        output_dir / "dev_error_feature_statistics.csv", error_stats,
        ("file", "group", "feature_index", "feature", "samples", "min", "p01", "p25", "median", "p75", "p99", "max", "mean", "zero_fraction", "saturated_fraction"),
    )
    csv_write(output_dir / "feature_group_correlations.csv", correlations, ("group_a", "group_b", "pearson"))

    thresholds = tuple(sorted(set((H2_R_THRESHOLD, 0.28))))
    baseline = evaluate_records(dev_records, None, device, batch_size, thresholds=thresholds)
    baseline_view = _fixed_view(baseline, H2_R_THRESHOLD)
    perturbations: Dict[str, Any] = {}
    eligible = []
    for order, group in enumerate(FEATURE_GROUPS):
        zero_records = _modified_records(dev_records, pipeline.h2.refiner_layers, group, "zero", 42)
        zero_eval = evaluate_records(zero_records, None, device, batch_size, thresholds=thresholds)
        zero_view = _fixed_view(zero_eval, H2_R_THRESHOLD)
        permutation_ap = []
        for seed in (42, 43, 44):
            permuted = _modified_records(dev_records, pipeline.h2.refiner_layers, group, "permute", seed)
            perm_eval = evaluate_records(permuted, None, device, batch_size, thresholds=thresholds)
            permutation_ap.append(float(perm_eval["average_precision"]))
        checks = {
            "zero_ap_gain_at_least_0_001": zero_eval["average_precision"] >= baseline["average_precision"] + 0.001,
            "zero_overall_iou_not_lower": zero_view["overall"]["event"]["iou"] >= baseline_view["overall"]["event"]["iou"],
            "zero_high_recall_not_lower": zero_view["high_density"]["event"]["recall"] >= baseline_view["high_density"]["event"]["recall"],
            "zero_high_iou_not_lower": zero_view["high_density"]["event"]["iou"] >= baseline_view["high_density"]["event"]["iou"],
            "ordinary_protection": _ordinary_protection(zero_view, baseline_view),
            "mean_permutation_ap_not_lower": float(np.mean(permutation_ap)) >= baseline["average_precision"],
        }
        passes = all(checks.values())
        perturbations[group] = {
            "columns": FEATURE_GROUPS[group],
            "zero": {"average_precision": zero_eval["average_precision"], "working_point": zero_view},
            "permutation_average_precision": permutation_ap,
            "permutation_mean_ap": float(np.mean(permutation_ap)),
            "checks": checks,
            "passes": passes,
        }
        if passes:
            eligible.append(
                (
                    float(zero_eval["average_precision"] - baseline["average_precision"]),
                    -len(FEATURE_GROUPS[group]),
                    -order,
                    group,
                )
            )
    selected = max(eligible)[3] if eligible else None
    report = {
        "schema_version": 1,
        "status": "complete",
        "data_roles": {"descriptive": "train_main", "perturbation": "train_dev"},
        "validation_used": False,
        "test_used": False,
        "feature_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
        "activation_saturation": quantized_activity(train_records, pipeline.h2.refiner_layers),
        "baseline": {"average_precision": baseline["average_precision"], "working_point": baseline_view},
        "perturbations": perturbations,
        "selected_feature_group": selected,
        "a2_3_eligible": selected is not None,
        "decision": "run_A2-3" if selected is not None else "skip_A2-3",
    }
    json_write(output_dir / "summary.json", report)
    return report


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    value = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        stop = start + 1
        while stop < len(value) and sorted_value[stop] == sorted_value[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_sum = float(ranks[truth].sum())
    return (positive_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def run_stc_diagnostic(
    dev_records: Sequence[Mapping[str, Any]], output_path: Path
) -> Dict[str, Any]:
    weights_all = []
    target_all = []
    predicted_all = []
    high_all = []
    paper_saturated = []
    per_sequence = []
    for record in dev_records:
        result = stc_weights_sparse(
            record["baseline_probability"], record["x"], record["y"], record["t_ms"]
        )
        weight = result["released"]
        target = np.asarray(record["labels"]) == 1
        predicted = np.asarray(record["baseline_probability"]) >= H2_R_THRESHOLD
        high = np.full(len(target), float(record["sequence_rate_eps"]) >= REPORT_SEQUENCE_DENSITY_EPS)
        weights_all.append(weight)
        target_all.append(target)
        predicted_all.append(predicted)
        high_all.append(high)
        paper_saturated.append(np.mean((result["paper"] < 0.01) | (result["paper"] > 0.99)))
        per_sequence.append(
            {
                "file": record["file"],
                "released_mean": float(weight.mean()),
                "released_saturated_fraction": float(np.mean((weight < 0.01) | (weight > 0.99))),
                "paper_saturated_fraction": float(paper_saturated[-1]),
            }
        )
    weight = np.concatenate(weights_all)
    target = np.concatenate(target_all)
    predicted = np.concatenate(predicted_all)
    high = np.concatenate(high_all)
    union = target | predicted
    auc = _rank_auc(weight[union], target[union])
    target_median = _median(weight[target])
    false_median = _median(weight[predicted & ~target])
    high_fn_median = _median(weight[high & target & ~predicted])
    ordinary_fp_median = _median(weight[~high & ~target & predicted])
    ordinary_fp_penalty = _median(1.0 - weight[~high & ~target & predicted])
    released_saturation = float(np.mean((weight < 0.01) | (weight > 0.99)))
    checks = {
        "auc_at_least_0_60": auc is not None and auc >= 0.60,
        "target_median_above_false_positive": target_median is not None and false_median is not None and target_median > false_median,
        "high_density_fn_weight_above_ordinary_fp": high_fn_median is not None and ordinary_fp_median is not None and high_fn_median > ordinary_fp_median,
        "ordinary_fp_negative_penalty_above_0_5": ordinary_fp_penalty is not None and ordinary_fp_penalty > 0.5,
        "released_weights_not_saturated": released_saturation < 0.95,
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "data_role": "train_dev",
        "validation_used": False,
        "test_used": False,
        "variant": "released_centered_detached",
        "kernel": {"spatial_pixels": 3, "temporal_ms": 5},
        "paper_gamma": 2,
        "released_code_uses_gamma": False,
        "metrics": {
            "auroc_on_target_prediction_union": auc,
            "target_weight_median": target_median,
            "false_positive_weight_median": false_median,
            "high_density_fn_weight_median": high_fn_median,
            "ordinary_fp_weight_median": ordinary_fp_median,
            "ordinary_fp_negative_penalty_median": ordinary_fp_penalty,
            "released_saturated_fraction": released_saturation,
            "paper_mean_saturated_fraction": float(np.mean(paper_saturated)),
        },
        "checks": checks,
        "passes": all(checks.values()),
        "decision": "permit_A2-4" if all(checks.values()) else "skip_A2-4",
        "per_sequence": per_sequence,
    }
    json_write(output_path, report)
    return report

