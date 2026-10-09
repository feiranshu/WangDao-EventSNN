"""在冻结配置下比较 S1 与 S3；只开放一次 Validation。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
sys.path.insert(0, str(ROOT / "software" / "scripts"))

from evaluate_approx import (
    choose_device,
    event_output,
    run_proposal,
    update_totals,
)
from evaluate_s1 import (
    add_counts,
    atomic_json,
    empty_totals,
    finalize,
    sha256_file,
    validation_paths,
)
from wangdao.data import load_events
from wangdao.fixed import (
    FixedH2,
    FixedH2Config,
    TorchApproximateProposalSNN,
    load_approximate_proposal_snn,
    load_fixed_proposal_snn,
)
from wangdao.h2.constants import PROPOSAL_DIMENSIONS
from wangdao.h2.numpy_model import load_export
from wangdao.s0p import build_s0p_sequence


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
S1_WEIGHT_PATH = (
    ROOT / "software" / "weights" / "S1P_INT8_ACC24_MEM24_Q12.npz"
)
S3_WEIGHT_PATH = (
    ROOT
    / "software"
    / "weights"
    / "S3P_LOA_SYN13_MEMALL_K2_ACC24_MEM24_Q12.npz"
)
CALIBRATION_PATH = ROOT / "results" / "approx_combination.json"
OUTPUT_PATH = ROOT / "results" / "s3_validation.json"
ROLE = "validation"


def model_state() -> Dict[str, Any]:
    return {
        "totals": empty_totals(),
        "accumulator_saturations": [0, 0, 0],
        "membrane_saturations": [0, 0, 0, 0],
        "refiner_saturations": [0, 0, 0],
        "spikes": [0, 0, 0],
        "approximate_additions": [0, 0, 0, 0, 0, 0],
    }


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    h2_path = arguments.h2_weight.resolve()
    s1_path = arguments.s1_weight.resolve()
    s3_path = arguments.s3_weight.resolve()
    calibration_path = arguments.calibration_result.resolve()
    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if (
        not isinstance(calibration, dict)
        or calibration.get("decision") != "freeze_for_validation"
        or not calibration.get("scope", {}).get("complete_scan")
    ):
        raise ValueError("S3 Validation 要求完整且已冻结的组合校准结果")
    selection = calibration.get("selection", {})
    expected_export = selection.get("frozen_export")
    if expected_export != s3_path.relative_to(ROOT).as_posix():
        raise ValueError("组合校准记录指向的 S3 权重与当前输入不一致")
    if selection.get("frozen_export_sha256") != sha256_file(s3_path):
        raise ValueError("S3 冻结权重哈希与组合校准记录不一致")

    h2 = load_export(h2_path)
    s1, s1_configuration, s1_metadata = load_fixed_proposal_snn(s1_path)
    s3, s3_configuration, s3_metadata = load_approximate_proposal_snn(s3_path)
    if s3_metadata.get("source_s1_weight_sha256") != sha256_file(s1_path):
        raise ValueError("S3 与当前 S1 权重不一致")
    if s3_metadata.get("h2_weight_sha256") != sha256_file(h2_path):
        raise ValueError("S3 与当前 H2 权重不一致")
    if s3_metadata.get("dataset_index_sha256") != sha256_file(INDEX_PATH):
        raise ValueError("S3 与当前数据索引不一致")
    h2_configuration = s3_metadata.get("h2_fixed_configuration")
    if not isinstance(h2_configuration, dict):
        raise ValueError("S3 缺少 H2 定点配置")
    fixed_h2 = FixedH2(
        h2,
        FixedH2Config(
            accumulator_bits=int(h2_configuration["accumulator_bits"]),
            scale_fraction_bits=int(h2_configuration["scale_fraction_bits"]),
            logit_fraction_bits=int(h2_configuration["logit_fraction_bits"]),
        ),
    )
    proposal_threshold = float(s3_metadata["proposal_threshold"])
    event_threshold = float(s3_metadata["event_threshold"])
    device = choose_device(arguments.device)
    s3_runtime: Any = (
        TorchApproximateProposalSNN(s3, s3.placement, device)
        if device.type == "cuda"
        else s3
    )

    states = {"s1": model_state(), "s3": model_state()}
    labels_parts: List[np.ndarray] = []
    total_events = 0
    total_targets = 0
    total_patches = 0
    logit_absolute_error_sum = 0
    logit_difference_count = 0
    logit_maximum_absolute_error = 0
    per_sequence: List[Dict[str, Any]] = []
    paths = validation_paths()
    started = time.perf_counter()

    for number, path in enumerate(paths, start=1):
        sequence_started = time.perf_counter()
        events = load_events(path, require_labels=True)
        labels = events["label"].astype(np.uint8, copy=False)
        sequence = build_s0p_sequence(path, s1.time_steps)
        patches = sequence.ann_features.shape[0] * sequence.ann_features.shape[1]
        temporal = sequence.temporal_features.transpose(1, 0, 2, 3).reshape(
            s1.time_steps, patches, PROPOSAL_DIMENSIONS[0]
        )
        outputs = {
            "s1": run_proposal(s1, temporal, arguments.batch_size),
            "s3": run_proposal(s3_runtime, temporal, arguments.batch_size),
        }
        difference = np.abs(outputs["s3"][0] - outputs["s1"][0])
        logit_absolute_error_sum += int(difference.sum())
        logit_difference_count += int(np.count_nonzero(difference))
        logit_maximum_absolute_error = max(
            logit_maximum_absolute_error, int(difference.max())
        )
        record: Dict[str, Any] = {
            "file": path.name,
            "events": len(events),
            "target_events": int(labels.sum()),
        }
        for name in ("s1", "s3"):
            output = outputs[name]
            probability, candidates, refiner_saturations = event_output(
                events,
                output[1],
                fixed_h2,
                h2,
                proposal_threshold,
                arguments.batch_size,
            )
            state = states[name]
            record[name] = update_totals(
                state["totals"],
                events,
                labels,
                probability,
                candidates,
                event_threshold,
            )
            add_counts(state["accumulator_saturations"], tuple(output[2]))
            add_counts(state["membrane_saturations"], tuple(output[3]))
            add_counts(state["spikes"], tuple(output[4]))
            add_counts(state["approximate_additions"], tuple(output[5]))
            add_counts(state["refiner_saturations"], refiner_saturations)
        record["elapsed_s"] = time.perf_counter() - sequence_started
        per_sequence.append(record)
        labels_parts.append(labels)
        total_events += len(events)
        total_targets += int(labels.sum())
        total_patches += patches
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"S1 IoU={record['s1']['event']['iou']:.4f}，"
            f"S3 IoU={record['s3']['event']['iou']:.4f}，"
            f"耗时={record['elapsed_s']:.2f}s",
            flush=True,
        )

    labels = np.concatenate(labels_parts)
    evaluations: Dict[str, Dict[str, Any]] = {}
    spike_denominators = [
        s1.time_steps * total_patches * dimension
        for dimension in PROPOSAL_DIMENSIONS[1:]
    ]
    for name in ("s1", "s3"):
        state = states[name]
        result = finalize(
            state["totals"], labels, event_threshold, total_events, total_targets
        )
        result["proposal_accumulator_saturations"] = state[
            "accumulator_saturations"
        ]
        result["proposal_membrane_saturations"] = state["membrane_saturations"]
        result["refiner_accumulator_saturations"] = state["refiner_saturations"]
        result["spike_counts"] = state["spikes"]
        result["spike_rates"] = [
            count / denominator
            for count, denominator in zip(state["spikes"], spike_denominators)
        ]
        result["approximate_additions"] = state["approximate_additions"]
        result["approximate_addition_total"] = sum(state["approximate_additions"])
        result["saturation_total"] = sum(state["accumulator_saturations"]) + sum(
            state["membrane_saturations"]
        ) + sum(state["refiner_saturations"])
        evaluations[name] = result

    s1_probability = np.concatenate(states["s1"]["totals"]["probabilities"])
    s3_probability = np.concatenate(states["s3"]["totals"]["probabilities"])
    deltas = {
        "event_iou": evaluations["s3"]["event"]["iou"]
        - evaluations["s1"]["event"]["iou"],
        "event_f1": evaluations["s3"]["event"]["f1"]
        - evaluations["s1"]["event"]["f1"],
        "pd": evaluations["s3"]["pd"] - evaluations["s1"]["pd"],
        "proposal_recall": evaluations["s3"]["proposal_recall"]
        - evaluations["s1"]["proposal_recall"],
        "candidate_fraction": evaluations["s3"]["candidate_fraction"]
        - evaluations["s1"]["candidate_fraction"],
        "probability_mae": float(
            np.mean(np.abs(s3_probability.astype(np.float64) - s1_probability))
        ),
        "prediction_disagreement": float(
            np.mean(
                (s3_probability >= event_threshold)
                != (s1_probability >= event_threshold)
            )
        ),
        "logit_different_fraction": logit_difference_count
        / (total_patches * 16),
        "logit_mean_absolute_q": logit_absolute_error_sum
        / (total_patches * 16),
        "logit_maximum_absolute_q": logit_maximum_absolute_error,
    }
    maximum_iou_loss = float(s3_metadata["selection_rule"]["maximum_iou_loss"])
    gates = {
        "event_iou_loss_within_1_5_points": deltas["event_iou"]
        >= -maximum_iou_loss,
        "no_new_saturation": evaluations["s3"]["saturation_total"]
        <= evaluations["s1"]["saturation_total"],
    }
    passed = all(gates.values())
    decision = "accept_s3" if passed else "reject_s3"
    conclusion = (
        f"S3 Validation Event IoU 为 {evaluations['s3']['event']['iou']:.4%}，"
        f"相对 S1 变化 {deltas['event_iou'] * 100:+.4f} 个百分点，"
        f"饱和 {evaluations['s3']['saturation_total']} 次；"
        + ("通过冻结门槛，进入 RTL 对齐。" if passed else "未通过门槛，保留 S1 exact。")
    )
    return {
        "schema_version": 1,
        "status": "ok" if passed else "gate_failed",
        "experiment_id": arguments.experiment_id,
        "stage": "S3",
        "data_role": ROLE,
        "primary_metrics": {
            "s1_event_iou": evaluations["s1"]["event"]["iou"],
            "s3_event_iou": evaluations["s3"]["event"]["iou"],
            "iou_delta_to_s1": deltas["event_iou"],
            "s1_proposal_recall": evaluations["s1"]["proposal_recall"],
            "s3_proposal_recall": evaluations["s3"]["proposal_recall"],
            "s3_approximate_additions": evaluations["s3"][
                "approximate_addition_total"
            ],
            "s3_saturation_total": evaluations["s3"]["saturation_total"],
        },
        "decision": decision,
        "conclusion": conclusion,
        "scope": {
            "role": ROLE,
            "test_used": False,
            "threshold_scan": False,
            "training_performed": False,
            "sequences": len(paths),
            "events": total_events,
            "target_events": total_targets,
            "fixed_models_run_on": "torch_cuda_integer"
            if device.type == "cuda"
            else "cpu_numpy_integer",
            "elapsed_s": time.perf_counter() - started,
        },
        "inputs": {
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "h2_weight": h2_path.name,
            "h2_weight_sha256": sha256_file(h2_path),
            "s1_weight": s1_path.name,
            "s1_weight_sha256": sha256_file(s1_path),
            "s1_configuration": s1_configuration,
            "s1_metadata": s1_metadata,
            "s3_weight": s3_path.name,
            "s3_weight_sha256": sha256_file(s3_path),
            "s3_configuration": s3_configuration,
            "s3_metadata": s3_metadata,
            "calibration_result": calibration_path.relative_to(ROOT).as_posix(),
            "calibration_result_sha256": sha256_file(calibration_path),
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
        },
        "evaluation": evaluations,
        "deltas_to_s1": deltas,
        "gates": gates,
        "per_sequence": per_sequence,
        "reproducibility": {
            "command": [str(value) for value in sys.argv],
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "fixed_source_sha256": sha256_file(
                ROOT / "software" / "wangdao" / "fixed.py"
            ),
            "shared_calibrator_sha256": sha256_file(
                ROOT / "software" / "scripts" / "evaluate_approx.py"
            ),
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT_PATH)
    parser.add_argument("--s1-weight", type=Path, default=S1_WEIGHT_PATH)
    parser.add_argument("--s3-weight", type=Path, default=S3_WEIGHT_PATH)
    parser.add_argument(
        "--calibration-result", type=Path, default=CALIBRATION_PATH
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--experiment-id", default="S3-LOA-validation-v1")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("batch-size 必须为正数")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = evaluate(arguments)
    atomic_json(result, arguments.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "conclusion": result["conclusion"],
                "primary_metrics": result["primary_metrics"],
                "deltas_to_s1": result["deltas_to_s1"],
                "gates": result["gates"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
