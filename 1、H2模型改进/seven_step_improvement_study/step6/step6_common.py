"""Frozen decision, sensitivity and bootstrap helpers for ANN Step6."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step4"):
    if (candidate / "step4_common.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:  # pragma: no cover
    raise RuntimeError("Cannot locate Step4 modules")
for candidate in (HERE, HERE.parent / "step5"):
    if (candidate / "step5_common.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:  # pragma: no cover
    raise RuntimeError("Cannot locate Step5 modules")

from step4_common import exact_pr  # noqa: E402
from step5_common import corrected_noninferiority, revised_candidate_gates  # noqa: E402


PROTOCOL = "h2-ann-step6-three-seed-stability-v1"
SEEDS = (42, 43, 44)
TRAIN_SEEDS = (43, 44)
FIXED_EPOCH = 8
FIXED_THRESHOLD = 0.40
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261004
HIGH_FILES = ("train_089.npz", "train_092.npz")


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def compact(view: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "average_precision": float(view["average_precision"]),
        "overall_iou": float(view["overall"]["event"]["iou"]),
        "overall_pd": float(view["overall"]["localization"]["pd"]),
        "overall_fa": float(view["overall"]["localization"]["fa"]),
        "high_recall": float(view["high_density"]["event"]["recall"]),
        "high_iou": float(view["high_density"]["event"]["iou"]),
        "high_fa": float(view["high_density"]["localization"]["fa"]),
        "ordinary_iou": float(view["ordinary_density"]["event"]["iou"]),
        "ordinary_recall": float(view["ordinary_density"]["event"]["recall"]),
        "ordinary_pd": float(view["ordinary_density"]["localization"]["pd"]),
        "ordinary_fa": float(view["ordinary_density"]["localization"]["fa"]),
        "train_089_iou": float(view["per_sequence"]["train_089.npz"]["event"]["iou"]),
        "train_092_iou": float(view["per_sequence"]["train_092.npz"]["event"]["iou"]),
    }


def deltas(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, float]:
    first, second = compact(candidate), compact(reference)
    return {name: first[name] - second[name] for name in first}


def matched_attribution(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> Dict[str, Any]:
    ap_delta = float(candidate["average_precision"]) - float(control["average_precision"])
    iou_delta = float(candidate["overall"]["event"]["iou"]) - float(control["overall"]["event"]["iou"])
    checks = {
        "ap_not_lower": ap_delta >= 0.0,
        "iou_not_lower": iou_delta >= 0.0,
        "one_strict": ap_delta > 0.0 or iou_delta > 0.0,
    }
    return {
        "passes": all(checks.values()), "checks": checks,
        "delta_ap": ap_delta, "delta_iou": iou_delta,
    }


def seed_report(
    seed: int, candidate: Mapping[str, Any], control: Mapping[str, Any],
    h2_f: Mapping[str, Any], h2_r: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "seed": int(seed), "fixed_epoch": FIXED_EPOCH,
        "fixed_threshold": FIXED_THRESHOLD,
        "candidate": compact(candidate), "matched_null": compact(control),
        "delta_vs_h2": deltas(candidate, h2_r),
        "delta_vs_matched_null": deltas(candidate, control),
        "revised_candidate_gates": revised_candidate_gates(candidate, h2_f, h2_r),
        "noninferiority": corrected_noninferiority(candidate, h2_f, h2_r),
        "matched_attribution": matched_attribution(candidate, control),
    }


def aggregate_decision(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in rows}
    if tuple(sorted(by_seed)) != SEEDS:
        raise ValueError("Step6 requires exactly seeds 42, 43 and 44")
    deltas_iou = [float(by_seed[seed]["delta_vs_h2"]["overall_iou"]) for seed in SEEDS]
    deltas_ap = [float(by_seed[seed]["delta_vs_h2"]["average_precision"]) for seed in SEEDS]
    candidate_wins = [deltas_iou[i] > 0.0 and deltas_ap[i] >= 0.0 for i in range(3)]
    attribution = [bool(by_seed[seed]["matched_attribution"]["passes"]) for seed in SEEDS]
    checks = {
        "seed42_passes_revised_candidate_gates": bool(by_seed[42]["revised_candidate_gates"]["passes"]),
        "seed43_passes_noninferiority": bool(by_seed[43]["noninferiority"]["passes"]),
        "seed44_passes_noninferiority": bool(by_seed[44]["noninferiority"]["passes"]),
        "at_least_two_seeds_improve_iou_without_ap_loss": sum(candidate_wins) >= 2,
        "mean_delta_iou_positive": float(np.mean(deltas_iou)) > 0.0,
        "mean_delta_ap_nonnegative": float(np.mean(deltas_ap)) >= 0.0,
        "at_least_two_matched_attribution_passes": sum(attribution) >= 2,
    }
    return {
        "passes": all(checks.values()), "checks": checks,
        "seed_candidate_wins": dict(zip(map(str, SEEDS), candidate_wins)),
        "seed_attribution_passes": dict(zip(map(str, SEEDS), attribution)),
        "mean_delta_iou": float(np.mean(deltas_iou)),
        "mean_delta_ap": float(np.mean(deltas_ap)),
    }


def stratified_weights(
    sequence_files: Sequence[str], iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    names = [Path(str(value)).name for value in sequence_files]
    high = np.asarray([i for i, name in enumerate(names) if name in HIGH_FILES], dtype=np.int64)
    ordinary = np.asarray([i for i, name in enumerate(names) if name not in HIGH_FILES], dtype=np.int64)
    if len(high) != 2 or not len(ordinary):
        raise ValueError("frozen Train-dev must contain exactly two high-density sequences")
    rng = np.random.default_rng(seed)
    weights = np.zeros((iterations, len(names)), dtype=np.int16)
    rows = np.arange(iterations)[:, None]
    for indices in (high, ordinary):
        draws = indices[rng.integers(0, len(indices), size=(iterations, len(indices)))]
        np.add.at(weights, (np.broadcast_to(rows, draws.shape), draws), 1)
    if not np.all(weights[:, high].sum(axis=1) == len(high)):
        raise AssertionError("high-density bootstrap stratum size changed")
    if not np.all(weights[:, ordinary].sum(axis=1) == len(ordinary)):
        raise AssertionError("ordinary bootstrap stratum size changed")
    return weights


def _ap_replicates(labels: np.ndarray, probability: np.ndarray, sequence: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(probability, dtype=np.float32), kind="stable")
    scores = np.asarray(probability, dtype=np.float32)[order]
    target = np.asarray(labels, dtype=np.uint8)[order] == 1
    seq = np.asarray(sequence, dtype=np.int64)[order]
    starts = np.r_[True, scores[1:] != scores[:-1]]
    group = np.cumsum(starts, dtype=np.int64) - 1
    groups = int(group[-1]) + 1
    nseq = weights.shape[1]
    totals = np.zeros((groups, nseq), dtype=np.float64)
    positives = np.zeros((groups, nseq), dtype=np.float64)
    np.add.at(totals, (group, seq), 1.0)
    np.add.at(positives, (group[target], seq[target]), 1.0)
    unique_weights, inverse = np.unique(weights, axis=0, return_inverse=True)
    unique_result = np.empty(len(unique_weights), dtype=np.float64)
    if torch.cuda.is_available() and len(unique_weights) > 256:
        device = torch.device("cuda")
        totals_tensor = torch.from_numpy(totals).to(device=device, dtype=torch.float64)
        positives_tensor = torch.from_numpy(positives).to(device=device, dtype=torch.float64)
        for start in range(0, len(unique_weights), 64):
            stop = min(start + 64, len(unique_weights))
            w = torch.from_numpy(unique_weights[start:stop].astype(np.float64).T).to(device)
            group_total = totals_tensor @ w
            group_positive = positives_tensor @ w
            total_positive = group_positive.sum(dim=0)
            cumulative_positive = torch.cumsum(group_positive, dim=0)
            cumulative_total = torch.cumsum(group_total, dim=0)
            precision = torch.where(cumulative_total > 0, cumulative_positive / cumulative_total, 0.0)
            values = torch.where(
                total_positive > 0,
                torch.sum(group_positive * precision, dim=0) / total_positive,
                torch.nan,
            )
            unique_result[start:stop] = values.cpu().numpy()
    else:
        for start in range(0, len(unique_weights), 64):
            stop = min(start + 64, len(unique_weights))
            w = unique_weights[start:stop].astype(np.float64).T
            group_total = totals @ w
            group_positive = positives @ w
            total_positive = group_positive.sum(axis=0)
            cumulative_positive = np.cumsum(group_positive, axis=0)
            cumulative_total = np.cumsum(group_total, axis=0)
            precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
            unique_result[start:stop] = np.divide(
                np.sum(group_positive * precision, axis=0), total_positive,
                out=np.full(stop - start, np.nan), where=total_positive > 0,
            )
    return unique_result[inverse]


def _ratio_from_counts(weights: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    top = weights @ numerator.astype(np.float64)
    bottom = weights @ denominator.astype(np.float64)
    return np.divide(top, bottom, out=np.full(len(weights), np.nan), where=bottom > 0)


def _sequence_counts(view: Mapping[str, Any], sequence_files: Sequence[str]) -> Dict[str, np.ndarray]:
    rows = [view["per_sequence"][Path(str(name)).name] for name in sequence_files]
    return {
        "tp": np.asarray([row["event"]["tp"] for row in rows]),
        "fp": np.asarray([row["event"]["fp"] for row in rows]),
        "fn": np.asarray([row["event"]["fn"] for row in rows]),
        "correct": np.asarray([row["localization"]["correct_targets"] for row in rows]),
        "targets": np.asarray([row["localization"]["targets"] for row in rows]),
        "false_components": np.asarray([row["localization"]["false_components"] for row in rows]),
        "fa_denominator": np.asarray([row["localization"]["frame_denominator"] * 346 * 260 for row in rows]),
    }


def _summary(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(np.mean(finite)),
        "lower_95": float(np.percentile(finite, 2.5)),
        "upper_95": float(np.percentile(finite, 97.5)),
    }


def paired_bootstrap(
    candidate_scores: Path, h2_scores: Path,
    candidate_view: Mapping[str, Any], h2_view: Mapping[str, Any],
    iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    candidate, baseline = read_npz(candidate_scores), read_npz(h2_scores)
    files = [str(value) for value in candidate["sequence_files"]]
    if [Path(str(value)).name for value in baseline["sequence_files"]] != [Path(value).name for value in files]:
        raise ValueError("candidate and H2 bootstrap sequence order differs")
    if not np.array_equal(candidate["labels"], baseline["labels"]) or not np.array_equal(candidate["sequence"], baseline["sequence"]):
        raise ValueError("candidate and H2 bootstrap event identities differ")
    weights = stratified_weights(files, iterations, seed)
    cap = _ap_replicates(candidate["labels"], candidate["probability"], candidate["sequence"], weights)
    hap = _ap_replicates(baseline["labels"], baseline["probability"], baseline["sequence"], weights)
    cc, hc = _sequence_counts(candidate_view, files), _sequence_counts(h2_view, files)
    ciou = _ratio_from_counts(weights, cc["tp"], cc["tp"] + cc["fp"] + cc["fn"])
    hiou = _ratio_from_counts(weights, hc["tp"], hc["tp"] + hc["fp"] + hc["fn"])
    cpd = _ratio_from_counts(weights, cc["correct"], cc["targets"])
    hpd = _ratio_from_counts(weights, hc["correct"], hc["targets"])
    cfa = _ratio_from_counts(weights, cc["false_components"], cc["fa_denominator"])
    hfa = _ratio_from_counts(weights, hc["false_components"], hc["fa_denominator"])
    return {
        "method": "density-stratified sequence paired percentile bootstrap",
        "iterations": int(iterations), "seed": int(seed),
        "ap_compute_backend": "torch_cuda_float64" if torch.cuda.is_available() and iterations > 256 else "numpy_float64",
        "high_density_sequences": list(HIGH_FILES),
        "delta_average_precision": _summary(cap - hap),
        "delta_event_iou": _summary(ciou - hiou),
        "delta_pd": _summary(cpd - hpd),
        "delta_fa": _summary(cfa - hfa),
    }


def _aggregate_subset(view: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    rows = [view["per_sequence"][name] for name in names]
    tp = sum(row["event"]["tp"] for row in rows)
    fp = sum(row["event"]["fp"] for row in rows)
    fn = sum(row["event"]["fn"] for row in rows)
    correct = sum(row["localization"]["correct_targets"] for row in rows)
    targets = sum(row["localization"]["targets"] for row in rows)
    false = sum(row["localization"]["false_components"] for row in rows)
    frames = sum(row["localization"]["frame_denominator"] for row in rows)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "pd": correct / targets if targets else None,
        "fa": false / (frames * 346 * 260) if frames else None,
    }


def sensitivity_report(
    candidate_scores: Path, h2_scores: Path,
    candidate_view: Mapping[str, Any], h2_view: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate, baseline = read_npz(candidate_scores), read_npz(h2_scores)
    files = [Path(str(value)).name for value in candidate["sequence_files"]]
    modes = {
        "all": files,
        "without_089": [name for name in files if name != "train_089.npz"],
        "without_092": [name for name in files if name != "train_092.npz"],
        "without_089_092": [name for name in files if name not in HIGH_FILES],
        "only_089": ["train_089.npz"],
        "only_092": ["train_092.npz"],
    }
    result: Dict[str, Any] = {}
    for mode, names in modes.items():
        indices = [files.index(name) for name in names]
        mask = np.isin(candidate["sequence"], indices)
        candidate_ap = exact_pr(candidate["labels"][mask], candidate["probability"][mask])["average_precision"]
        h2_ap = exact_pr(baseline["labels"][mask], baseline["probability"][mask])["average_precision"]
        cv, hv = _aggregate_subset(candidate_view, names), _aggregate_subset(h2_view, names)
        result[mode] = {
            "sequences": names, "candidate": {**cv, "average_precision": candidate_ap},
            "h2": {**hv, "average_precision": h2_ap},
            "delta_iou": cv["iou"] - hv["iou"],
            "delta_average_precision": candidate_ap - h2_ap,
        }
    return {"scope": "Train-dev sensitivity; no threshold search", "groups": result}
