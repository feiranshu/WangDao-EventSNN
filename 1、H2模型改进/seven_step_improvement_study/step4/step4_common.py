"""Shared feature, model, evaluation and decision helpers for ANN Step 4.

Only Train-main and Train-dev are reachable.  Validation and Test deliberately
have no entry point in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step3"):
    try:
        available = (candidate / "step3_common.py").is_file()
    except (OSError, PermissionError):
        available = False
    if available:
        sys.path.insert(0, str(candidate))
        break
else:  # pragma: no cover - server-package preflight reports this clearly
    raise RuntimeError("Cannot locate bundled step3_common.py")

from step3_common import (  # noqa: E402,F401
    CandidateCorpus,
    H2_F_THRESHOLD,
    INT8_REFERENCE_AP,
    RUNTIME_ROOT,
    candidate_gates,
    evaluate_records,
    exact_pr,
    export_metrics,
    json_write,
    load_records,
    make_float_refiner,
    quantized_layers_from_model,
    remove_private,
    role_paths,
    select_h2_r,
    sha256,
    working_point_view,
)
from wangdao.h2.export import quantize_weight  # noqa: E402
from wangdao.h2.numpy_model import _refiner_probability, quantized_mlp  # noqa: E402
from wangdao.h2.torch_model import prototype_logits  # noqa: E402
from wangdao.h2_research import select_fair_working_point  # noqa: E402


SCHEMA_VERSION = 1
PROTOCOL = "h2-ann-step4-feature-information-v1"
REGISTERED_INITIAL = (0, 1, 2, 5, 8, 10)
REGISTERED_CONTINUATION = (11, 12, 15, 18, 20)
SEQUENCE_TIME_COLUMN = 2

MODEL_DESCRIPTIONS = {
    "A4-1": "32D H2 plus one causal current-window density feature",
    "A4-1-null": "33D matched null control for A4-1",
    "A4-2": "32D H2 plus four causal proposal-history features",
    "A4-2-null": "36D matched null control for A4-2",
    "A4-3": "31D H2 retrained after removing non-causal sequence_time",
    "A4-3-control": "matched 32D H2 retraining control for A4-3",
    "A4-4": "conditional 37D density plus history combination",
    "A4-4-null": "37D matched null control for A4-4",
    "A4-5": "conditional effective Step4 information with sequence_time removed",
    "A4-5-control": "matched effective-information control retaining sequence_time",
}


@dataclass(frozen=True)
class DensityTransform:
    log1p_min: float
    log1p_max: float
    train_candidates: int
    train_sequences: int

    def apply(self, rates: np.ndarray) -> np.ndarray:
        value = np.log1p(np.maximum(np.asarray(rates, dtype=np.float64), 0.0))
        width = self.log1p_max - self.log1p_min
        if not math.isfinite(width) or width <= 0:
            raise ValueError("invalid density normalization range")
        scaled = np.clip((value - self.log1p_min) / width, 0.0, 1.0)
        return np.rint(255.0 * scaled).astype(np.uint8)[:, None]

    def as_dict(self) -> Dict[str, Any]:
        value = {
            "kind": "log1p_minmax_uint8",
            "fit_role": "Train-main only",
            "log1p_min": self.log1p_min,
            "log1p_max": self.log1p_max,
            "train_candidates": self.train_candidates,
            "train_sequences": self.train_sequences,
        }
        import hashlib
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        value["parameter_sha256"] = hashlib.sha256(encoded).hexdigest()
        return value


def fit_density(records: Sequence[Mapping[str, Any]]) -> DensityTransform:
    if not records:
        raise ValueError("density fitting requires Train-main records")
    joined = np.concatenate([np.asarray(row["window_rate_eps"], dtype=np.float64) for row in records])
    transformed = np.log1p(joined)
    return DensityTransform(
        float(transformed.min()), float(transformed.max()), int(len(joined)), int(len(records))
    )


def feature_kind(model_id: str, selected_information: Optional[str] = None) -> str:
    if model_id.startswith("A4-1"):
        return "density"
    if model_id.startswith("A4-2"):
        return "history"
    if model_id in ("A4-3", "A4-3-control"):
        return "drop_time" if model_id == "A4-3" else "base32"
    if model_id.startswith("A4-4"):
        return "both"
    if model_id.startswith("A4-5"):
        if selected_information not in ("density", "history", "both"):
            raise ValueError("A4-5 requires selected_information=density, history, or both")
        return selected_information + ("_drop_time" if model_id == "A4-5" else "")
    raise ValueError(f"unknown Step4 model: {model_id}")


def is_null(model_id: str) -> bool:
    return model_id.endswith("-null")


def feature_parts(
    record: Mapping[str, Any],
    model_id: str,
    density: DensityTransform,
    selected_information: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return old-input and new-input uint8 blocks for one candidate record."""
    kind = feature_kind(model_id, selected_information)
    base = np.asarray(record["features"], dtype=np.uint8)
    drop_time = kind.endswith("_drop_time") or kind == "drop_time"
    old = np.delete(base, SEQUENCE_TIME_COLUMN, axis=1) if drop_time else base
    core = kind.removesuffix("_drop_time")
    extras = []
    if core in ("density", "both"):
        extras.append(density.apply(np.asarray(record["window_rate_eps"])))
    if core in ("history", "both"):
        extras.append(np.asarray(record["history"], dtype=np.uint8))
    new = np.concatenate(extras, axis=1) if extras else np.empty((len(base), 0), dtype=np.uint8)
    if is_null(model_id):
        new = np.zeros_like(new)
    return np.ascontiguousarray(old), np.ascontiguousarray(new)


def joined_features(
    record: Mapping[str, Any], model_id: str, density: DensityTransform,
    selected_information: Optional[str] = None,
) -> np.ndarray:
    old, new = feature_parts(record, model_id, density, selected_information)
    return np.ascontiguousarray(np.concatenate((old, new), axis=1), dtype=np.uint8)


class Step4Refiner(nn.Module):
    """H2 Refiner with an independently optimizable zero-initialized input block."""

    def __init__(self, old_width: int, new_width: int, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.old_width = int(old_width)
        self.new_width = int(new_width)
        self.dimensions = (self.old_width + self.new_width, 64, 32, 4)
        self.old_input = nn.Linear(self.old_width, 64, bias=False)
        self.new_input = nn.Linear(self.new_width, 64, bias=False) if self.new_width else None
        self.hidden = nn.Linear(64, 32, bias=False)
        self.output = nn.Linear(32, 4, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.dimensions[0]:
            raise ValueError(f"expected [N,{self.dimensions[0]}] Step4 features")
        value = values.to(torch.float32) / 255.0
        first = self.old_input(value[:, : self.old_width])
        if self.new_input is not None:
            first = first + self.new_input(value[:, self.old_width :])
        value = F.hardtanh(first, 0.0, 1.0)
        value = F.hardtanh(self.hidden(value), 0.0, 1.0)
        return self.output(value)

    def combined_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self.old_input.weight
        if self.new_input is not None:
            first = torch.cat((first, self.new_input.weight), dim=1)
        return first, self.hidden.weight, self.output.weight

    def old_parameters(self) -> list[nn.Parameter]:
        return [*self.old_input.parameters(), *self.hidden.parameters(), *self.output.parameters()]

    def new_parameters(self) -> list[nn.Parameter]:
        return [] if self.new_input is None else list(self.new_input.parameters())


def build_model(model_id: str, seed: int = 42, selected_information: Optional[str] = None) -> Step4Refiner:
    kind = feature_kind(model_id, selected_information)
    drop_time = kind.endswith("_drop_time") or kind == "drop_time"
    old_width = 31 if drop_time else 32
    core = kind.removesuffix("_drop_time")
    new_width = {"density": 1, "history": 4, "both": 5}.get(core, 0)
    reference = make_float_refiner(seed)
    model = Step4Refiner(old_width, new_width, seed)
    with torch.no_grad():
        source = reference.layers[0].weight
        model.old_input.weight.copy_(
            torch.cat((source[:, :SEQUENCE_TIME_COLUMN], source[:, SEQUENCE_TIME_COLUMN + 1 :]), dim=1)
            if drop_time else source
        )
        if model.new_input is not None:
            model.new_input.weight.zero_()
        model.hidden.weight.copy_(reference.layers[1].weight)
        model.output.weight.copy_(reference.layers[2].weight)
    return model


def quantized_layers(model: Step4Refiner) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    result = []
    for weight in model.combined_weights():
        quantized, scale = quantize_weight(weight)
        result.append((np.ascontiguousarray(quantized), np.ascontiguousarray(scale)))
    return tuple(result)


def predict_float(model: Step4Refiner, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = np.empty(len(features), dtype=np.float32)
    with torch.no_grad():
        for first in range(0, len(features), batch_size):
            block = torch.from_numpy(features[first:first + batch_size]).to(device)
            probability = torch.softmax(prototype_logits(model(block)), dim=1)[:, 1]
            result[first:first + len(block)] = probability.detach().cpu().numpy()
    return result


def predict_int8(model: Step4Refiner, features: np.ndarray) -> np.ndarray:
    raw = quantized_mlp(np.ascontiguousarray(features, dtype=np.uint8), quantized_layers(model))
    return _refiner_probability(raw, 2)


def evaluate_model(
    records: Sequence[Mapping[str, Any]], model: Step4Refiner, model_id: str,
    density: DensityTransform, device: torch.device, batch_size: int,
    selected_information: Optional[str] = None, keep_pr_points: bool = False,
) -> Dict[str, Any]:
    blocks = [joined_features(row, model_id, density, selected_information) for row in records]
    started = time.perf_counter()
    float_probability = [predict_float(model, block, device, batch_size) for block in blocks]
    float_result = evaluate_records(records, None, device, batch_size, candidate_probabilities=float_probability, keep_pr_points=keep_pr_points)
    float_s = time.perf_counter() - started
    started = time.perf_counter()
    int8_probability = [predict_int8(model, block) for block in blocks]
    int8_result = evaluate_records(records, None, device, batch_size, candidate_probabilities=int8_probability, keep_pr_points=keep_pr_points)
    int8_s = time.perf_counter() - started
    return {
        "float": float_result, "int8": int8_result,
        "_timings": {"float_evaluation_s": float_s, "int8_evaluation_s": int8_s},
    }


def baseline_views(records: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    baseline = evaluate_records(records, None, device, batch_size)
    anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
    h2_f = working_point_view(baseline, {"eligible": True, "threshold": H2_F_THRESHOLD, "metrics": anchor})
    h2_r = working_point_view(baseline, select_h2_r(baseline))
    return baseline, h2_f, h2_r


def evaluation_row(epoch: int, pair: Mapping[str, Any], baseline: Mapping[str, Any], h2_f: Mapping[str, Any], h2_r: Mapping[str, Any]) -> Dict[str, Any]:
    anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
    selected = select_fair_working_point(pair["int8"]["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"])
    float_selected = select_fair_working_point(pair["float"]["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"])
    int8_view = working_point_view(pair["int8"], selected)
    float_view = working_point_view(pair["float"], float_selected)
    return {
        "epoch": int(epoch),
        "float": remove_private(pair["float"]),
        "int8": remove_private(pair["int8"]),
        "float_working_point": float_view,
        "int8_working_point": int8_view,
        "int8_selection": selected,
        "replacement_gates": candidate_gates(int8_view, h2_f, h2_r),
        "noninferiority": noninferiority(int8_view, h2_r),
    }


def _at_least(candidate: Any, reference: Any, delta: float) -> bool:
    return candidate is not None and reference is not None and float(candidate) >= float(reference) + delta


def _at_most_ratio(candidate: Any, reference: Any, ratio: float) -> bool:
    if candidate is None or reference is None:
        return False
    return float(candidate) <= (0.0 if float(reference) == 0.0 else float(reference) * ratio)


def noninferiority(view: Mapping[str, Any], h2_r: Mapping[str, Any]) -> Dict[str, Any]:
    if not view.get("eligible"):
        return {"passes": False, "checks": {"working_point_exists": False}}
    checks = {
        "working_point_exists": True,
        "ap_loss_within_0_1pp": _at_least(view["average_precision"], h2_r["average_precision"], -0.001),
        "overall_iou_loss_within_0_3pp": _at_least(view["overall"]["event"]["iou"], h2_r["overall"]["event"]["iou"], -0.003),
        "high_iou_loss_within_0_3pp": _at_least(view["high_density"]["event"]["iou"], h2_r["high_density"]["event"]["iou"], -0.003),
        "high_recall_loss_within_0_3pp": _at_least(view["high_density"]["event"]["recall"], h2_r["high_density"]["event"]["recall"], -0.003),
        "ordinary_iou_loss_within_0_3pp": _at_least(view["ordinary_density"]["event"]["iou"], h2_r["ordinary_density"]["event"]["iou"], -0.003),
        "ordinary_recall_loss_within_0_3pp": _at_least(view["ordinary_density"]["event"]["recall"], h2_r["ordinary_density"]["event"]["recall"], -0.003),
        "ordinary_pd_loss_within_0_3pp": _at_least(view["ordinary_density"]["localization"]["pd"], h2_r["ordinary_density"]["localization"]["pd"], -0.003),
        "ordinary_fa_rise_within_5pct": _at_most_ratio(view["ordinary_density"]["localization"]["fa"], h2_r["ordinary_density"]["localization"]["fa"], 1.05),
    }
    return {"passes": all(checks.values()), "checks": checks}


def eligibility(evaluation: Mapping[str, Any]) -> int:
    if evaluation["replacement_gates"]["passes"]:
        return 2
    if evaluation["noninferiority"]["passes"]:
        return 1
    return 0


def primary_rank(evaluation: Mapping[str, Any]) -> tuple[float, ...]:
    view = evaluation["int8_working_point"]
    if not view.get("eligible"):
        return (float(eligibility(evaluation)),) + (-math.inf,) * 6
    return (
        float(eligibility(evaluation)), float(view["overall"]["event"]["iou"]),
        float(view["average_precision"]), float(view["high_density"]["event"]["recall"]),
        -float(view["overall"]["localization"]["fa"]), -float(evaluation["epoch"]),
        -float(view["threshold"]),
    )


def high_density_rank(evaluation: Mapping[str, Any]) -> tuple[float, ...]:
    view = evaluation["int8_working_point"]
    if not view.get("eligible"):
        return (float(eligibility(evaluation)),) + (-math.inf,) * 6
    return (
        float(eligibility(evaluation)), float(view["high_density"]["event"]["recall"]),
        float(view["high_density"]["event"]["iou"]), float(view["overall"]["event"]["iou"]),
        float(view["average_precision"]), -float(evaluation["epoch"]), -float(view["threshold"]),
    )


def pair_entry(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> Dict[str, Any]:
    c, n = candidate["final"]["primary"]["int8_working_point"], control["final"]["primary"]["int8_working_point"]
    if not c.get("eligible") or not n.get("eligible"):
        return {"passes": False, "checks": {"working_points_exist": False}}
    checks = {
        "working_points_exist": True,
        "candidate_noninferior": candidate["final"]["primary"]["noninferiority"]["passes"],
        "control_noninferior": control["final"]["primary"]["noninferiority"]["passes"],
        "ap_not_below_null": _at_least(c["average_precision"], n["average_precision"], 0.0),
        "iou_not_below_null": _at_least(c["overall"]["event"]["iou"], n["overall"]["event"]["iou"], 0.0),
        "one_overall_metric_strict": c["average_precision"] > n["average_precision"] or c["overall"]["event"]["iou"] > n["overall"]["event"]["iou"],
        "high_recall_or_iou_strict_other_not_lower": (
            (c["high_density"]["event"]["recall"] > n["high_density"]["event"]["recall"] and c["high_density"]["event"]["iou"] >= n["high_density"]["event"]["iou"])
            or (c["high_density"]["event"]["iou"] > n["high_density"]["event"]["iou"] and c["high_density"]["event"]["recall"] >= n["high_density"]["event"]["recall"])
        ),
    }
    h2 = candidate["h2_r"]
    for file_name in ("train_089.npz", "train_092.npz"):
        checks[f"{file_name}_within_0_3pp"] = _at_least(
            c["per_sequence"][file_name]["event"]["iou"], h2["per_sequence"][file_name]["event"]["iou"], -0.003
        )
    return {"passes": all(checks.values()), "checks": checks}


def continue10_allowed(summary: Mapping[str, Any]) -> Dict[str, Any]:
    rows = {int(row["epoch"]): row for row in summary["evaluations"]}
    if not all(epoch in rows for epoch in (8, 9, 10)):
        return {"passes": False, "reason": "epochs 8, 9 and 10 are required"}
    views = [rows[epoch]["int8_working_point"] for epoch in (8, 9, 10)]
    if not all(view.get("eligible") for view in views):
        return {"passes": False, "reason": "all final three working points must exist"}
    iou = [view["overall"]["event"]["iou"] for view in views]
    ap = [view["average_precision"] for view in views]
    h2 = summary["h2_r"]
    checks = {
        "epochs_8_9_10_noninferior": all(rows[e]["noninferiority"]["passes"] for e in (8, 9, 10)),
        "iou_monotonic": iou[0] <= iou[1] <= iou[2],
        "iou_gain_8_to_10_at_least_0_1pp": iou[2] - iou[0] >= 0.001,
        "ap_monotonic": ap[0] <= ap[1] <= ap[2],
        "ap10_within_0_1pp": ap[2] >= h2["average_precision"] - 0.001,
    }
    for file_name in ("train_089.npz", "train_092.npz"):
        checks[f"{file_name}_epochs_8_9_10_within_0_3pp"] = all(
            rows[e]["int8_working_point"]["per_sequence"][file_name]["event"]["iou"]
            >= h2["per_sequence"][file_name]["event"]["iou"] - 0.003 for e in (8, 9, 10)
        )
    return {"passes": all(checks.values()), "checks": checks}


def weight_norms(model: Step4Refiner) -> Dict[str, float]:
    old = float(torch.linalg.vector_norm(model.old_input.weight.detach()).cpu())
    new = 0.0 if model.new_input is None else float(torch.linalg.vector_norm(model.new_input.weight.detach()).cpu())
    return {"old_input": old, "new_input": new, "new_to_old": 0.0 if old == 0 else new / old}


def activation_saturation(model: Step4Refiner, features: np.ndarray, device: torch.device) -> list[Dict[str, float]]:
    value = torch.from_numpy(features).to(device).to(torch.float32) / 255.0
    with torch.no_grad():
        first = model.old_input(value[:, :model.old_width])
        if model.new_input is not None:
            first = first + model.new_input(value[:, model.old_width:])
        values = [torch.clamp(first, 0.0, 1.0)]
        values.append(torch.clamp(model.hidden(values[-1]), 0.0, 1.0))
    return [{
        "hidden_layer": index,
        "zero_fraction": float((item <= 0).float().mean().cpu()),
        "one_fraction": float((item >= 1).float().mean().cpu()),
        "linear_fraction": float(((item > 0) & (item < 1)).float().mean().cpu()),
    } for index, item in enumerate(values, 1)]


def density_from_json(path: Path) -> DensityTransform:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return DensityTransform(float(value["log1p_min"]), float(value["log1p_max"]), int(value["train_candidates"]), int(value["train_sequences"]))
