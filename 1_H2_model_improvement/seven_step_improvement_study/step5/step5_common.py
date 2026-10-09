"""Selection and freeze helpers for H2 ANN Step5.

Step5 reads only the completed Step4 Train-dev artifacts.  It never exposes
Validation or Test and it never trains a model.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step4"):
    try:
        available = (candidate / "step4_common.py").is_file()
    except (OSError, PermissionError):
        available = False
    if available:
        sys.path.insert(0, str(candidate))
        break
else:  # pragma: no cover - reported by preflight in a broken package
    raise RuntimeError("Cannot locate bundled or project Step4 modules")

from step4_common import candidate_gates, working_point_view  # noqa: E402


PROTOCOL = "h2-ann-step5-history-freeze-v1"
REGISTERED_EPOCHS = (0, 1, 2, 5, 8, 10)
DIFFICULT_FILES = ("train_089.npz", "train_092.npz")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _at_least(value: Any, reference: Any, delta: float) -> bool:
    return _finite(value) and _finite(reference) and float(value) >= float(reference) + delta


def _at_most_ratio(value: Any, reference: Any, ratio: float) -> bool:
    if not _finite(value) or not _finite(reference):
        return False
    return float(value) <= (0.0 if float(reference) == 0.0 else float(reference) * ratio)


def _metric(view: Mapping[str, Any], *path: str) -> Any:
    value: Any = view
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def required_metrics_are_finite(view: Mapping[str, Any]) -> bool:
    paths = (
        ("average_precision",),
        ("overall", "event", "iou"),
        ("overall", "localization", "pd"),
        ("overall", "localization", "fa"),
        ("high_density", "event", "recall"),
        ("high_density", "event", "iou"),
        ("high_density", "localization", "fa"),
        ("ordinary_density", "event", "iou"),
        ("ordinary_density", "event", "recall"),
        ("ordinary_density", "localization", "pd"),
        ("ordinary_density", "localization", "fa"),
    )
    if not view.get("eligible") or not all(_finite(_metric(view, *path)) for path in paths):
        return False
    return all(
        _finite(_metric(view, "per_sequence", name, "event", "iou"))
        for name in DIFFICULT_FILES
    )


def corrected_noninferiority(
    view: Mapping[str, Any], h2_f: Mapping[str, Any], h2_r: Mapping[str, Any]
) -> Dict[str, Any]:
    checks = {
        "working_point_and_required_metrics_exist": required_metrics_are_finite(view),
        "ap_loss_within_0_1pp": _at_least(_metric(view, "average_precision"), _metric(h2_r, "average_precision"), -0.001),
        "overall_iou_loss_within_0_3pp": _at_least(_metric(view, "overall", "event", "iou"), _metric(h2_r, "overall", "event", "iou"), -0.003),
        "overall_pd_loss_within_0_5pp": _at_least(_metric(view, "overall", "localization", "pd"), _metric(h2_f, "overall", "localization", "pd"), -0.005),
        "overall_fa_rise_within_10pct": _at_most_ratio(_metric(view, "overall", "localization", "fa"), _metric(h2_f, "overall", "localization", "fa"), 1.10),
        "high_iou_loss_within_0_3pp": _at_least(_metric(view, "high_density", "event", "iou"), _metric(h2_r, "high_density", "event", "iou"), -0.003),
        "high_recall_loss_within_0_3pp": _at_least(_metric(view, "high_density", "event", "recall"), _metric(h2_r, "high_density", "event", "recall"), -0.003),
        "high_fa_rise_within_10pct": _at_most_ratio(_metric(view, "high_density", "localization", "fa"), _metric(h2_r, "high_density", "localization", "fa"), 1.10),
        "ordinary_iou_loss_within_0_3pp": _at_least(_metric(view, "ordinary_density", "event", "iou"), _metric(h2_r, "ordinary_density", "event", "iou"), -0.003),
        "ordinary_recall_loss_within_0_3pp": _at_least(_metric(view, "ordinary_density", "event", "recall"), _metric(h2_r, "ordinary_density", "event", "recall"), -0.003),
        "ordinary_pd_loss_within_0_3pp": _at_least(_metric(view, "ordinary_density", "localization", "pd"), _metric(h2_r, "ordinary_density", "localization", "pd"), -0.003),
        "ordinary_fa_rise_within_5pct": _at_most_ratio(_metric(view, "ordinary_density", "localization", "fa"), _metric(h2_r, "ordinary_density", "localization", "fa"), 1.05),
    }
    return {"passes": all(checks.values()), "checks": checks}


def revised_candidate_gates(
    view: Mapping[str, Any], h2_f: Mapping[str, Any], h2_r: Mapping[str, Any]
) -> Dict[str, Any]:
    checks = {
        "working_point_and_required_metrics_exist": required_metrics_are_finite(view),
        "overall_iou_strictly_higher": _at_least(_metric(view, "overall", "event", "iou"), _metric(h2_r, "overall", "event", "iou"), 0.0) and float(_metric(view, "overall", "event", "iou") or -math.inf) > float(_metric(h2_r, "overall", "event", "iou") or math.inf),
        "overall_ap_not_lower": _at_least(_metric(view, "average_precision"), _metric(h2_r, "average_precision"), 0.0),
        "overall_pd_loss_within_0_5pp": _at_least(_metric(view, "overall", "localization", "pd"), _metric(h2_f, "overall", "localization", "pd"), -0.005),
        "overall_fa_rise_within_10pct": _at_most_ratio(_metric(view, "overall", "localization", "fa"), _metric(h2_f, "overall", "localization", "fa"), 1.10),
        "high_recall_loss_within_0_3pp": _at_least(_metric(view, "high_density", "event", "recall"), _metric(h2_r, "high_density", "event", "recall"), -0.003),
        "high_iou_loss_within_0_3pp": _at_least(_metric(view, "high_density", "event", "iou"), _metric(h2_r, "high_density", "event", "iou"), -0.003),
        "high_fa_rise_within_10pct": _at_most_ratio(_metric(view, "high_density", "localization", "fa"), _metric(h2_r, "high_density", "localization", "fa"), 1.10),
        "ordinary_iou_loss_within_0_3pp": _at_least(_metric(view, "ordinary_density", "event", "iou"), _metric(h2_r, "ordinary_density", "event", "iou"), -0.003),
        "ordinary_recall_loss_within_0_3pp": _at_least(_metric(view, "ordinary_density", "event", "recall"), _metric(h2_r, "ordinary_density", "event", "recall"), -0.003),
        "ordinary_pd_loss_within_0_3pp": _at_least(_metric(view, "ordinary_density", "localization", "pd"), _metric(h2_r, "ordinary_density", "localization", "pd"), -0.003),
        "ordinary_fa_rise_within_5pct": _at_most_ratio(_metric(view, "ordinary_density", "localization", "fa"), _metric(h2_r, "ordinary_density", "localization", "fa"), 1.05),
    }
    for name in DIFFICULT_FILES:
        checks[f"{name}_iou_not_lower"] = _at_least(
            _metric(view, "per_sequence", name, "event", "iou"),
            _metric(h2_r, "per_sequence", name, "event", "iou"), 0.0,
        )
    return {"passes": all(checks.values()), "checks": checks}


def threshold_grid(summary: Mapping[str, Any]) -> tuple[float, ...]:
    grids = []
    registered = set(REGISTERED_EPOCHS)
    for row in summary.get("evaluations", []):
        if int(row.get("epoch", -1)) in registered:
            grids.append(tuple(sorted(float(key) for key in row["int8"]["curve"])))
    if len(grids) != len(REGISTERED_EPOCHS) or not grids or any(grid != grids[0] for grid in grids[1:]):
        raise ValueError("registered Step4 evaluations do not share one complete threshold grid")
    return grids[0]


def enumerate_combinations(summary: Mapping[str, Any]) -> list[Dict[str, Any]]:
    if tuple(summary.get("configuration", {}).get("registered_epochs", ())) != REGISTERED_EPOCHS:
        raise ValueError("Step4 registered epochs do not match the frozen Step5 protocol")
    grid = threshold_grid(summary)
    h2_f, h2_r = summary["h2_f"], summary["h2_r"]
    by_epoch = {int(row["epoch"]): row for row in summary["evaluations"]}
    combinations = []
    for epoch in REGISTERED_EPOCHS:
        evaluation = by_epoch[epoch]
        for threshold in grid:
            view = working_point_view(evaluation["int8"], {"eligible": True, "threshold": threshold})
            original_full = candidate_gates(view, h2_f, h2_r)
            original_noninferior = corrected_noninferiority(view, h2_f, h2_r)
            combinations.append({
                "epoch": epoch, "threshold": threshold, "view": view,
                "original_full": original_full,
                "original_noninferior": original_noninferior,
                "revised": revised_candidate_gates(view, h2_f, h2_r),
            })
    return combinations


def _rank(row: Mapping[str, Any], ordinary_fa: bool) -> tuple[float, ...]:
    view = row["view"]
    fa_group = "ordinary_density" if ordinary_fa else "overall"
    return (
        float(view["overall"]["event"]["iou"]),
        float(view["average_precision"]),
        float(view["high_density"]["event"]["recall"]),
        -float(view[fa_group]["localization"]["fa"]),
        -float(row["epoch"]), -float(row["threshold"]),
    )


def select_original(combinations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def level(row: Mapping[str, Any]) -> int:
        return 2 if row["original_full"]["passes"] else (1 if row["original_noninferior"]["passes"] else 0)
    selected = max(combinations, key=lambda row: (level(row), *_rank(row, ordinary_fa=False)))
    return {"eligibility_level": level(selected), **dict(selected)}


def select_revised(combinations: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    qualified = [row for row in combinations if row["revised"]["passes"]]
    return None if not qualified else dict(max(qualified, key=lambda row: _rank(row, ordinary_fa=True)))


def compact(row: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if row is None:
        return None
    view = row["view"]
    return {
        "epoch": int(row["epoch"]), "threshold": float(row["threshold"]),
        "average_precision": view["average_precision"],
        "overall_iou": view["overall"]["event"]["iou"],
        "overall_pd": view["overall"]["localization"]["pd"],
        "overall_fa": view["overall"]["localization"]["fa"],
        "high_recall": view["high_density"]["event"]["recall"],
        "high_iou": view["high_density"]["event"]["iou"],
        "high_fa": view["high_density"]["localization"]["fa"],
        "ordinary_iou": view["ordinary_density"]["event"]["iou"],
        "ordinary_recall": view["ordinary_density"]["event"]["recall"],
        "ordinary_pd": view["ordinary_density"]["localization"]["pd"],
        "ordinary_fa": view["ordinary_density"]["localization"]["fa"],
        "train_089_iou": view["per_sequence"]["train_089.npz"]["event"]["iou"],
        "train_092_iou": view["per_sequence"]["train_092.npz"]["event"]["iou"],
    }


def count_passes(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    return sum(bool(row[key]["passes"]) for row in rows)
