"""在 Train-dev 上端到端选择 STC 候选，并导出 HW-calibration 输入。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
sys.path.insert(0, str(ROOT / "software" / "scripts"))

from evaluate_s0p import atomic_json, evaluate, load_student, sha256_file
from wangdao.snn import export_proposal_snn


BASELINE_WEIGHT = ROOT / "software" / "weights" / "S0P_T4_POS15_KD50_S42.npz"
CALIBRATION_RECORD = ROOT / "results" / "stc_calibration.json"
CANDIDATE_NAMES = ("STC00", "STC05", "STC10", "STC20")
PROPOSAL_THRESHOLD = 0.51


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE_WEIGHT)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--finalize-calibration",
        action="store_true",
        help="将正式选择结论合并写回 STC 校准记录",
    )
    parser.add_argument(
        "--calibration-record", type=Path, default=CALIBRATION_RECORD
    )
    arguments = parser.parse_args()
    if arguments.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    return arguments


def evaluate_checkpoint(
    checkpoint: Path,
    experiment_id: str,
    output: Path,
    arguments: argparse.Namespace,
) -> Dict[str, Any]:
    evaluation_arguments = argparse.Namespace(
        checkpoint=checkpoint,
        role="train_dev",
        proposal_threshold=PROPOSAL_THRESHOLD,
        chunk_size=arguments.chunk_size,
        device=arguments.device,
        experiment_id=experiment_id,
        output=output,
    )
    result = evaluate(evaluation_arguments)
    atomic_json(result, output)
    return result


def metric_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = result["evaluation"]
    event = evaluation["event"]
    return {
        "event_iou": float(event["iou"]),
        "precision": float(event["precision"]),
        "recall": float(event["recall"]),
        "pd": float(evaluation["pd"]),
        "fa": float(evaluation["fa"]),
        "proposal_recall": float(evaluation["proposal_recall"]),
        "candidate_fraction": float(evaluation["candidate_fraction_global"]),
        "mean_spike_rates": [
            float(value) for value in evaluation["mean_spike_rates"]
        ],
        "event_driven_synaptic_accumulations": int(
            evaluation["event_driven_synaptic_accumulations"]
        ),
        "hybrid_synaptic_proxy": int(evaluation["hybrid_synaptic_proxy"]),
    }


def deltas(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    return {
        "event_iou": candidate["event_iou"] - baseline["event_iou"],
        "pd": candidate["pd"] - baseline["pd"],
        "proposal_recall": candidate["proposal_recall"]
        - baseline["proposal_recall"],
        "candidate_fraction": candidate["candidate_fraction"]
        - baseline["candidate_fraction"],
        "event_driven_synaptic_ratio": candidate[
            "event_driven_synaptic_accumulations"
        ]
        / baseline["event_driven_synaptic_accumulations"],
        "hybrid_synaptic_ratio": candidate["hybrid_synaptic_proxy"]
        / baseline["hybrid_synaptic_proxy"],
    }


def finite_metrics(metrics: Dict[str, Any]) -> bool:
    values: List[float] = [
        float(metrics[name])
        for name in (
            "event_iou",
            "precision",
            "recall",
            "pd",
            "fa",
            "proposal_recall",
            "candidate_fraction",
        )
    ]
    values.extend(float(value) for value in metrics["mean_spike_rates"])
    return all(math.isfinite(value) for value in values)


def candidate_gates(
    metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    control_passes: bool,
) -> tuple[Dict[str, bool], Dict[str, float]]:
    delta = deltas(metrics, baseline)
    benefit = (
        delta["event_iou"] >= 0.005 and delta["pd"] >= -0.005
    ) or (
        delta["pd"] >= 0.010 and delta["event_iou"] >= -0.002
    )
    rates = metrics["mean_spike_rates"]
    gates = {
        "lambda_zero_control_passes": control_passes,
        "benefit_threshold": benefit,
        "proposal_recall_not_lower": delta["proposal_recall"] >= 0.0,
        "candidate_growth_within_5_points": delta["candidate_fraction"] <= 0.05,
        "hybrid_synaptic_proxy_growth_within_5_percent": delta[
            "hybrid_synaptic_ratio"
        ]
        <= 1.05,
        "finite_metrics": finite_metrics(metrics),
        "spike_rates_not_abnormal": all(0.0 <= float(rate) < 0.95 for rate in rates),
    }
    return gates, delta


def run(arguments: argparse.Namespace) -> Dict[str, Any]:
    run_directory = arguments.run.resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"STC 训练目录不存在：{run_directory}")
    config_path = run_directory / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        training_config = json.load(handle)
    if not isinstance(training_config, dict) or training_config.get("mode") != "stc_sweep":
        raise ValueError("运行目录缺少有效的 STC 训练配置")
    complete_protocol = (
        int(training_config.get("epochs", 0)) == 20
        and int(training_config.get("max_sequences", -1)) == 0
        and int(training_config.get("max_dev_sequences", -1)) == 0
        and int(training_config.get("time_steps", 0)) == 4
        and list(training_config.get("stc_weights", [])) == [0.0, 0.05, 0.1, 0.2]
        and float(training_config.get("positive_fraction", -1.0)) == 0.15
        and float(training_config.get("distillation_weight", -1.0)) == 0.5
        and int(training_config.get("seed", -1)) == 42
    )
    checkpoints = {
        name: run_directory / f"{name}-best.pt" for name in CANDIDATE_NAMES
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少 STC 候选 checkpoint：" + ", ".join(missing))

    baseline_path = arguments.baseline.resolve()
    baseline_result = evaluate_checkpoint(
        baseline_path,
        "S0P-STC-BASELINE-train-dev-v1",
        run_directory / "baseline-train-dev.json",
        arguments,
    )
    baseline = metric_summary(baseline_result)
    evaluated = []
    by_name: Dict[str, Dict[str, Any]] = {}
    for name, path in checkpoints.items():
        result = evaluate_checkpoint(
            path,
            f"S0P-{name}-train-dev-v1",
            run_directory / f"{name}-train-dev.json",
            arguments,
        )
        metrics = metric_summary(result)
        checkpoint = {
            "name": name,
            "stc_weight": int(name[-2:]) / 100.0,
            "checkpoint": path.name,
            "checkpoint_sha256": sha256_file(path),
            "metrics": metrics,
        }
        evaluated.append(checkpoint)
        by_name[name] = checkpoint

    control_delta = deltas(by_name["STC00"]["metrics"], baseline)
    control_gates = {
        "event_iou_degradation_within_0_5_points": control_delta["event_iou"]
        >= -0.005,
        "pd_degradation_within_1_point": control_delta["pd"] >= -0.010,
        "proposal_recall_degradation_within_0_5_points": control_delta[
            "proposal_recall"
        ]
        >= -0.005,
        "finite_metrics": finite_metrics(by_name["STC00"]["metrics"]),
    }
    control_passes = all(control_gates.values())

    acceptable = []
    for candidate in evaluated:
        if candidate["stc_weight"] == 0.0:
            candidate["deltas_to_baseline"] = control_delta
            candidate["gates"] = control_gates
            candidate["passes"] = control_passes
            continue
        gates, delta = candidate_gates(
            candidate["metrics"], baseline, control_passes
        )
        candidate["deltas_to_baseline"] = delta
        candidate["gates"] = gates
        candidate["passes"] = all(gates.values())
        if candidate["passes"]:
            acceptable.append(candidate)

    acceptable.sort(
        key=lambda candidate: (
            candidate["metrics"]["event_iou"],
            candidate["metrics"]["pd"],
            -candidate["metrics"]["hybrid_synaptic_proxy"],
        ),
        reverse=True,
    )
    selected = acceptable[0] if acceptable else None
    export_record = None
    if selected is not None:
        checkpoint_path = checkpoints[selected["name"]]
        student, configuration, epoch = load_student(
            checkpoint_path, device=torch.device("cpu")
        )
        export_path = run_directory / "selected-s0p-stc.npz"
        if export_path.exists():
            raise FileExistsError(f"拒绝覆盖候选权重：{export_path}")
        metadata = {
            "model_id": f"S0P_{selected['name']}_TRAIN_DEV_SELECTED",
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "proposal_threshold": PROPOSAL_THRESHOLD,
            "selection_role": "train_dev",
            "selection_result": "selection.json",
        }
        export_proposal_snn(export_path, student, configuration, metadata)
        export_record = {
            "path": export_path.name,
            "sha256": sha256_file(export_path),
            "metadata": metadata,
        }

    decision = "proceed_to_hw_calibration" if selected is not None else "reject_stc"
    conclusion = (
        f"{selected['name']} 通过 Train-dev 收益、候选规模和计算代理门槛，进入 HW-calibration。"
        if selected is not None
        else "没有正 STC 权重候选通过全部门槛，保留现有 S0-P/S1 并恢复 S3。"
    )
    return {
        "schema_version": 1,
        "status": "ok" if selected is not None else "gate_failed",
        "experiment_id": "S0P-STC-SELECTION-train-dev-v1",
        "stage": "S0-P-STC-selection",
        "data_role": "train_dev",
        "primary_metrics": {
            "selected_candidate": selected["name"] if selected is not None else None,
            "selected_event_iou": selected["metrics"]["event_iou"]
            if selected is not None
            else None,
            "selected_pd": selected["metrics"]["pd"]
            if selected is not None
            else None,
        },
        "decision": decision,
        "conclusion": conclusion,
        "scope": {
            "test_used": False,
            "validation_used": False,
            "threshold_scan": False,
            "proposal_threshold": PROPOSAL_THRESHOLD,
        },
        "training_protocol": {
            "complete_protocol": complete_protocol,
            "config": config_path.name,
            "config_sha256": sha256_file(config_path),
            "epochs": training_config.get("epochs"),
            "max_sequences": training_config.get("max_sequences"),
            "max_dev_sequences": training_config.get("max_dev_sequences"),
            "time_steps": training_config.get("time_steps"),
            "stc_weights": training_config.get("stc_weights"),
            "positive_fraction": training_config.get("positive_fraction"),
            "distillation_weight": training_config.get("distillation_weight"),
            "seed": training_config.get("seed"),
        },
        "baseline": {
            "checkpoint": str(baseline_path),
            "checkpoint_sha256": sha256_file(baseline_path),
            "metrics": baseline,
        },
        "lambda_zero_control": {
            "deltas_to_baseline": control_delta,
            "gates": control_gates,
            "passes": control_passes,
        },
        "candidates": evaluated,
        "selected": selected,
        "export": export_record,
        "selection_rule": {
            "event_iou_gain_minimum": 0.005,
            "pd_loss_maximum_when_iou_improves": 0.005,
            "pd_gain_alternative_minimum": 0.010,
            "event_iou_loss_maximum_when_pd_improves": 0.002,
            "proposal_recall_may_decrease": False,
            "candidate_growth_maximum_points": 0.05,
            "hybrid_synaptic_proxy_growth_maximum_ratio": 1.05,
            "ranking": "Train-dev Event IoU, Pd, then lower hybrid synaptic proxy",
        },
        "reproducibility": {
            "command": [str(value) for value in sys.argv],
            "selection_source_sha256": sha256_file(Path(__file__).resolve()),
            "evaluation_source_sha256": baseline_result[
                "evaluation_source_sha256"
            ],
        },
    }


def finalize_calibration(
    selection: Dict[str, Any], run_directory: Path, path: Path
) -> None:
    calibration_path = path.resolve()
    with calibration_path.open("r", encoding="utf-8") as handle:
        diagnostic = json.load(handle)
    if not isinstance(diagnostic, dict):
        raise ValueError("STC 校准记录根节点必须是对象")
    if diagnostic.get("experiment_id") == "S0P-STC-CALIBRATION-v1":
        diagnostic = diagnostic.get("diagnostic")
        if not isinstance(diagnostic, dict):
            raise ValueError("正式 STC 校准记录缺少原始诊断")
    if diagnostic.get("experiment_id") != "STC-DIAG-train-dev-v1":
        raise ValueError("只允许归并当前 STC Train-dev 诊断记录")
    if not selection.get("training_protocol", {}).get("complete_protocol"):
        raise ValueError("只有完整 20 轮、全量数据的固定协议才能写入正式校准记录")
    finalized = {
        "schema_version": 1,
        "status": selection["status"],
        "experiment_id": "S0P-STC-CALIBRATION-v1",
        "stage": "S0-P-STC",
        "data_role": "train_dev",
        "primary_metrics": {
            "diagnostic_best_auroc": diagnostic["primary_metrics"]["best_auroc"],
            **selection["primary_metrics"],
        },
        "decision": selection["decision"],
        "conclusion": selection["conclusion"],
        "scope": {
            "test_used": False,
            "validation_used": False,
            "diagnostic_sequences": diagnostic["scope"]["sequence_count"],
            "selection_role": "train_dev",
        },
        "diagnostic": diagnostic,
        "training_selection": {
            "run_directory": str(run_directory),
            "selection": selection,
        },
    }
    atomic_json(finalized, calibration_path)


def main() -> None:
    arguments = parse_arguments()
    result = run(arguments)
    output = arguments.run.resolve() / "selection.json"
    atomic_json(result, output)
    if arguments.finalize_calibration:
        finalize_calibration(
            result, arguments.run.resolve(), arguments.calibration_record
        )
    print(
        f"STC 选择结果：{result['decision']}；"
        f"输出：{output}"
        + (
            f"；正式记录：{arguments.calibration_record.resolve()}"
            if arguments.finalize_calibration
            else ""
        )
    )


if __name__ == "__main__":
    main()
