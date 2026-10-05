"""Run one Step6 seed/model and export the frozen epoch-8/threshold-0.40 view."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "step4"):
    if (candidate / "step4_common.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from step4_common import (  # noqa: E402
    DensityTransform, baseline_views, build_model, evaluate_model, export_metrics,
    json_write, working_point_view,
)
from step4_training import (  # noqa: E402
    _export_bootstrap_scores, _history_availability_report, export_int8,
    load_checkpoint, train_candidate,
)
from step6_common import (  # noqa: E402
    FIXED_EPOCH, FIXED_THRESHOLD, PROTOCOL, corrected_noninferiority,
    revised_candidate_gates,
)


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_fixed_seed(
    model_id: str, output: Path,
    train_records: Sequence[Mapping[str, Any]], dev_records: Sequence[Mapping[str, Any]],
    density: DensityTransform, device: torch.device, seed: int,
    batch_size: int = 4096, smoke: bool = False,
) -> Dict[str, Any]:
    if model_id not in ("A4-2", "A4-2-null"):
        raise ValueError("Step6 trains only A4-2 and A4-2-null")
    if seed not in (43, 44):
        raise ValueError("Step6 trains only additional seeds 43 and 44")
    train_candidate(
        model_id, output, train_records, dev_records, density, device,
        seed=seed, epochs=1 if smoke else 10, batch_size=batch_size, smoke=smoke,
    )
    summary_path = output / "summary.json"
    summary = _read(summary_path)
    evaluation_epoch = 1 if smoke else FIXED_EPOCH
    checkpoint = output / f"epoch_{evaluation_epoch:03d}.pt"
    package = load_checkpoint(checkpoint)
    if int(package["metadata"]["epoch"]) != evaluation_epoch:
        raise RuntimeError("fixed evaluation checkpoint epoch mismatch")
    model = build_model(model_id, seed).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    pair = evaluate_model(
        dev_records, model, model_id, density, device, batch_size,
        keep_pr_points=True,
    )
    _, h2_f, h2_r = baseline_views(dev_records, device, batch_size)
    int8_view = working_point_view(pair["int8"], {"eligible": True, "threshold": FIXED_THRESHOLD})
    float_view = working_point_view(pair["float"], {"eligible": True, "threshold": FIXED_THRESHOLD})
    fixed_dir = output / ("smoke_fixed" if smoke else "fixed_epoch_008_threshold_040")
    metrics = fixed_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=False)
    export_metrics(metrics / "int8", pair["int8"], int8_view)
    export_metrics(metrics / "float", pair["float"], float_view)
    bootstrap_path = metrics / "int8" / "bootstrap_event_scores.npz"
    _export_bootstrap_scores(bootstrap_path, dev_records, model, model_id, density, None)
    export_path = fixed_dir / "refiner_int8.npz"
    export_int8(export_path, model, {
        "protocol": PROTOCOL, "model_id": model_id, "seed": seed,
        "epoch": evaluation_epoch, "threshold": FIXED_THRESHOLD,
    })
    history = _history_availability_report(
        dev_records, model, model_id, density, device, batch_size, FIXED_THRESHOLD, None,
    )
    fixed = {
        "protocol": PROTOCOL, "model_id": model_id, "seed": seed,
        "formal_fixed_epoch": FIXED_EPOCH,
        "evaluated_epoch": evaluation_epoch,
        "fixed_threshold": FIXED_THRESHOLD,
        "smoke": bool(smoke), "checkpoint": checkpoint.name,
        "int8_export": str(export_path.relative_to(output)),
        "int8_working_point": int8_view,
        "float_working_point": float_view,
        "noninferiority": corrected_noninferiority(int8_view, h2_f, h2_r),
        "revised_candidate_gates": revised_candidate_gates(int8_view, h2_f, h2_r),
        "history_availability": history,
        "timings": pair["_timings"],
        "validation_used": False, "test_used": False,
    }
    json_write(fixed_dir / "fixed_evaluation.json", fixed)
    summary["step6_protocol"] = PROTOCOL
    summary["step6_fixed"] = fixed
    summary["step6_status"] = "smoke_complete" if smoke else "complete"
    json_write(summary_path, summary, overwrite=True)
    print(
        f"[{model_id} seed{seed}] Step6 fixed epoch={evaluation_epoch} "
        f"threshold={FIXED_THRESHOLD:.2f} ap={int8_view['average_precision']:.6f} "
        f"iou={int8_view['overall']['event']['iou']:.6f}",
        flush=True,
    )
    return summary
