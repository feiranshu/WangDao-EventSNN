"""在 HW-calibration 上比较全 PoT 与混合 PoT，并冻结通过门槛的 S2。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.fixed import (
    FixedH2,
    FixedH2Config,
    FixedMLPOutput,
    PoTProposalSNN,
    export_pot_proposal_snn,
    load_fixed_proposal_snn,
)
from wangdao.h2.constants import PATCHES_PER_WINDOW, PROPOSAL_DIMENSIONS, WINDOWS_PER_SEQUENCE
from wangdao.h2.numpy_model import _refiner_probability, load_export
from wangdao.h2.representation import (
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
)
from wangdao.metrics import event_metrics
from wangdao.s0p import build_s0p_sequence


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
S1_WEIGHT_PATH = ROOT / "software" / "weights" / "S1P_INT8_ACC24_MEM24_Q12.npz"
OUTPUT_PATH = ROOT / "results" / "pot_calibration.json"
ROLE = "hw_calibration"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象：{path}")
    return value


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary_name, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def calibration_paths() -> List[Path]:
    index = load_json(INDEX_PATH)
    if index.get("status") != "ok":
        raise ValueError("数据索引状态无效")
    location = index.get("location")
    records = index.get("files")
    names = index.get("roles", {}).get(ROLE)
    if not isinstance(location, str) or not isinstance(records, list):
        raise ValueError("数据索引缺少位置或文件记录")
    if not isinstance(names, list) or not names:
        raise ValueError("HW-calibration 数据角色为空")
    dataset_root = ROOT / location
    by_name = {
        Path(str(record["path"])).name: dataset_root / str(record["path"])
        for record in records
        if isinstance(record, dict) and "path" in record
    }
    if any(str(name) not in by_name for name in names):
        raise ValueError("HW-calibration 引用了索引中不存在的文件")
    paths = [by_name[str(name)] for name in names]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError("数据文件缺失：" + ", ".join(absent))
    return paths


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def proposal_heatmaps(logits: np.ndarray) -> np.ndarray:
    return patch_scores_to_heatmaps(
        sigmoid(logits).reshape(
            WINDOWS_PER_SEQUENCE, PATCHES_PER_WINDOW, PROPOSAL_DIMENSIONS[-1]
        )
    )


def fixed_mlp_batched(
    function: Any, features: np.ndarray, batch_size: int
) -> FixedMLPOutput:
    logits_q = []
    logits = []
    saturations = [0, 0, 0]
    for start in range(0, len(features), batch_size):
        output = function(features[start : start + batch_size])
        if not isinstance(output, FixedMLPOutput):
            raise TypeError("定点 MLP 必须返回 FixedMLPOutput")
        logits_q.append(output.logits_q)
        logits.append(output.logits)
        for index, count in enumerate(output.accumulator_saturations):
            saturations[index] += count
    if not logits:
        raise ValueError("定点 MLP 不接受空输入")
    return FixedMLPOutput(
        logits_q=np.concatenate(logits_q),
        logits=np.concatenate(logits),
        accumulator_saturations=tuple(saturations),
    )


def add_counts(destination: List[int], source: Tuple[int, ...]) -> None:
    for index, count in enumerate(source):
        destination[index] += int(count)


def run_proposal(
    model: Any, temporal: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, List[int], List[int], List[int]]:
    logits = []
    accumulator_saturations = [0, 0, 0]
    membrane_saturations = [0, 0, 0, 0]
    spikes = [0, 0, 0]
    for start in range(0, temporal.shape[1], batch_size):
        output = model.forward(temporal[:, start : start + batch_size])
        logits.append(output.logits)
        add_counts(accumulator_saturations, output.accumulator_saturations)
        add_counts(membrane_saturations, output.membrane_saturations)
        add_counts(spikes, output.spike_counts)
    return np.concatenate(logits), accumulator_saturations, membrane_saturations, spikes


def variant_identifier(architecture: str, minimum_exponent: int) -> str:
    return f"{architecture}_emin{abs(minimum_exponent)}"


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    h2_path = arguments.h2_weight.resolve()
    s1_path = arguments.s1_weight.resolve()
    h2 = load_export(h2_path)
    s1, s1_configuration, s1_metadata = load_fixed_proposal_snn(s1_path)
    h2_configuration = s1_metadata.get("h2_fixed_configuration")
    if not isinstance(h2_configuration, dict):
        raise ValueError("S1 缺少 H2 定点配置")
    fixed_h2 = FixedH2(
        h2,
        FixedH2Config(
            accumulator_bits=int(h2_configuration["accumulator_bits"]),
            scale_fraction_bits=int(h2_configuration["scale_fraction_bits"]),
            logit_fraction_bits=int(h2_configuration["logit_fraction_bits"]),
        ),
    )
    proposal_threshold = float(s1_metadata["proposal_threshold"])
    event_threshold = float(s1_metadata["event_threshold"])
    model_definitions = []
    models: Dict[str, PoTProposalSNN] = {}
    for architecture, modes in (
        ("full", ("pot", "pot", "pot")),
        ("mixed", ("int8", "pot", "int8")),
    ):
        for minimum_exponent in arguments.minimum_exponents:
            identifier = variant_identifier(architecture, minimum_exponent)
            model = PoTProposalSNN(
                s1,
                layer_modes=modes,
                minimum_exponent=minimum_exponent,
                maximum_exponent=arguments.maximum_exponent,
            )
            models[identifier] = model
            exponent_values = arguments.maximum_exponent - minimum_exponent + 1
            code_bits = int(math.ceil(math.log2(2 * exponent_values + 1)))
            pot_weights = sum(
                int(layer["weights"])
                for layer in model.weight_statistics()
                if layer["mode"] == "pot"
            )
            total_weights = sum(
                int(layer["weights"]) for layer in model.weight_statistics()
            )
            model_definitions.append(
                {
                    "id": identifier,
                    "architecture": architecture,
                    "layer_modes": list(modes),
                    "minimum_exponent": minimum_exponent,
                    "maximum_exponent": arguments.maximum_exponent,
                    "pot_code_bits": code_bits,
                    "pot_weight_fraction": pot_weights / total_weights,
                    "estimated_weight_bits": pot_weights * code_bits
                    + (total_weights - pot_weights) * 8,
                    "weight_statistics": model.weight_statistics(),
                }
            )

    paths = calibration_paths()
    if arguments.max_sequences:
        paths = paths[: arguments.max_sequences]
    labels_parts: List[np.ndarray] = []
    reference_parts: List[np.ndarray] = []
    variant_state: Dict[str, Dict[str, Any]] = {
        identifier: {
            "probabilities": [],
            "candidates": 0,
            "retained": 0,
            "accumulator_saturations": [0, 0, 0],
            "membrane_saturations": [0, 0, 0, 0],
            "refiner_saturations": [0, 0, 0],
            "spikes": [0, 0, 0],
        }
        for identifier in models
    }
    reference_candidates = 0
    reference_retained = 0
    total_events = 0
    total_targets = 0
    total_patches = 0
    per_sequence: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for number, path in enumerate(paths, start=1):
        sequence_started = time.perf_counter()
        events = load_events(path, require_labels=True)
        labels = events["label"].astype(np.uint8, copy=False)
        target_count = int(labels.sum())
        sequence = build_s0p_sequence(path, s1.time_steps)
        patches = sequence.ann_features.shape[0] * sequence.ann_features.shape[1]
        temporal = sequence.temporal_features.transpose(1, 0, 2, 3).reshape(
            s1.time_steps, patches, PROPOSAL_DIMENSIONS[0]
        )
        windows = list(range(WINDOWS_PER_SEQUENCE))

        reference_logits, _acc, _mem, _spikes = run_proposal(
            s1, temporal, arguments.batch_size
        )
        reference_heatmaps = proposal_heatmaps(reference_logits)
        candidates = candidate_indices(
            events, windows, reference_heatmaps, proposal_threshold
        )
        reference_probability = np.zeros(len(events), dtype=np.float32)
        if len(candidates):
            features = build_refiner_features(
                events, candidates, windows, reference_heatmaps, proposal_threshold
            )
            refiner = fixed_mlp_batched(
                fixed_h2.refiner, features, arguments.batch_size
            )
            reference_probability[candidates] = _refiner_probability(
                refiner.logits, h2.prototypes
            )
        reference_candidates += len(candidates)
        reference_retained += int(labels[candidates].sum())

        sequence_record: Dict[str, Any] = {
            "file": path.name,
            "events": len(events),
            "target_events": target_count,
            "reference_candidates": len(candidates),
            "variants": {},
        }
        for identifier, model in models.items():
            logits, accumulator, membrane, spikes = run_proposal(
                model, temporal, arguments.batch_size
            )
            heatmaps = proposal_heatmaps(logits)
            candidates = candidate_indices(
                events, windows, heatmaps, proposal_threshold
            )
            probability = np.zeros(len(events), dtype=np.float32)
            refiner_saturations = (0, 0, 0)
            if len(candidates):
                features = build_refiner_features(
                    events, candidates, windows, heatmaps, proposal_threshold
                )
                refiner = fixed_mlp_batched(
                    fixed_h2.refiner, features, arguments.batch_size
                )
                refiner_saturations = refiner.accumulator_saturations
                probability[candidates] = _refiner_probability(
                    refiner.logits, h2.prototypes
                )
            state = variant_state[identifier]
            state["probabilities"].append(probability)
            state["candidates"] += len(candidates)
            state["retained"] += int(labels[candidates].sum())
            add_counts(state["accumulator_saturations"], tuple(accumulator))
            add_counts(state["membrane_saturations"], tuple(membrane))
            add_counts(state["refiner_saturations"], refiner_saturations)
            add_counts(state["spikes"], tuple(spikes))
            sequence_record["variants"][identifier] = {
                "candidates": len(candidates),
                "proposal_recall": float(labels[candidates].sum() / target_count)
                if target_count
                else 1.0,
                "saturations": sum(accumulator)
                + sum(membrane)
                + sum(refiner_saturations),
            }
        labels_parts.append(labels)
        reference_parts.append(reference_probability)
        total_events += len(events)
        total_targets += target_count
        total_patches += patches
        sequence_record["elapsed_s"] = time.perf_counter() - sequence_started
        per_sequence.append(sequence_record)
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"参考候选={sequence_record['reference_candidates']}, "
            f"变体={len(models)}, 耗时={sequence_record['elapsed_s']:.2f}s",
            flush=True,
        )

    labels = np.concatenate(labels_parts)
    reference_probability = np.concatenate(reference_parts)
    reference_event = event_metrics(labels, reference_probability, event_threshold)
    results: Dict[str, Any] = {}
    definition_by_id = {value["id"]: value for value in model_definitions}
    for identifier, state in variant_state.items():
        probability = np.concatenate(state.pop("probabilities"))
        event = event_metrics(labels, probability, event_threshold)
        event["iou_delta_to_s1"] = event["iou"] - reference_event["iou"]
        event["probability_mae_to_s1"] = float(
            np.mean(
                np.abs(
                    probability.astype(np.float64)
                    - reference_probability.astype(np.float64)
                )
            )
        )
        event["prediction_disagreement_to_s1"] = float(
            np.mean(
                (probability >= event_threshold)
                != (reference_probability >= event_threshold)
            )
        )
        saturation_total = (
            sum(state["accumulator_saturations"])
            + sum(state["membrane_saturations"])
            + sum(state["refiner_saturations"])
        )
        spike_denominators = [
            models[identifier].time_steps * total_patches * dimension
            for dimension in PROPOSAL_DIMENSIONS[1:]
        ]
        results[identifier] = {
            "definition": definition_by_id[identifier],
            "event": event,
            "candidate_events": state["candidates"],
            "candidate_fraction": state["candidates"] / total_events,
            "proposal_recall": state["retained"] / total_targets,
            "proposal_accumulator_saturations": state[
                "accumulator_saturations"
            ],
            "proposal_membrane_saturations": state["membrane_saturations"],
            "refiner_accumulator_saturations": state["refiner_saturations"],
            "saturation_total": saturation_total,
            "spike_counts": state["spikes"],
            "spike_rates": [
                count / denominator
                for count, denominator in zip(state["spikes"], spike_denominators)
            ],
        }

    acceptable = [
        identifier
        for identifier, result in results.items()
        if result["saturation_total"] == 0
        and result["event"]["iou_delta_to_s1"] >= -arguments.maximum_iou_loss
    ]
    full_acceptable = [
        identifier
        for identifier in acceptable
        if results[identifier]["definition"]["architecture"] == "full"
    ]
    selection_pool = full_acceptable if full_acceptable else acceptable
    selected = (
        max(
            selection_pool,
            key=lambda identifier: (
                results[identifier]["definition"]["minimum_exponent"],
                -results[identifier]["definition"]["estimated_weight_bits"],
                results[identifier]["event"]["iou"],
            ),
        )
        if selection_pool
        else None
    )
    export_path: Path | None = None
    export_sha256: str | None = None
    if selected is not None and arguments.max_sequences == 0:
        definition = results[selected]["definition"]
        architecture_name = "FULL" if definition["architecture"] == "full" else "MIXED"
        candidate_export_path = arguments.export_dir / (
            f"S2P_POT_{architecture_name}_EMIN{abs(definition['minimum_exponent'])}_"
            f"EMAX{definition['maximum_exponent']}_ACC{s1.configuration.accumulator_bits}_"
            f"MEM{s1.configuration.membrane_bits}_Q"
            f"{s1.configuration.membrane_fraction_bits}.npz"
        )
        metadata = {
            "model_id": candidate_export_path.stem,
            "source_s1_weight": s1_path.name,
            "source_s1_weight_sha256": sha256_file(s1_path),
            "h2_weight_sha256": sha256_file(h2_path),
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "calibration_role": ROLE,
            "calibration_result": arguments.output.name,
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
            "h2_fixed_configuration": h2_configuration,
            "selection_rule": {
                "maximum_iou_loss": arguments.maximum_iou_loss,
                "requires_zero_saturation": True,
                "prefer_full_pot": True,
                "then_prefer_larger_minimum_exponent": True,
            },
        }
        export_pot_proposal_snn(
            candidate_export_path, models[selected], metadata
        )
        export_path = candidate_export_path
        export_sha256 = sha256_file(candidate_export_path)

    complete_role = arguments.max_sequences == 0
    selected_result = results[selected] if selected is not None else None
    primary_metrics: Dict[str, Any] = {
        "reference_s1_event_iou": reference_event["iou"],
        "maximum_iou_loss": arguments.maximum_iou_loss,
        "acceptable_variant_count": len(acceptable),
        "selected_variant": selected,
        "selected_event_iou": None,
        "selected_iou_delta_to_s1": None,
        "selected_proposal_recall": None,
        "selected_candidate_fraction": None,
        "selected_saturation_total": None,
    }
    if selected_result is not None:
        primary_metrics.update(
            {
                "selected_event_iou": selected_result["event"]["iou"],
                "selected_iou_delta_to_s1": selected_result["event"][
                    "iou_delta_to_s1"
                ],
                "selected_proposal_recall": selected_result["proposal_recall"],
                "selected_candidate_fraction": selected_result[
                    "candidate_fraction"
                ],
                "selected_saturation_total": selected_result[
                    "saturation_total"
                ],
            }
        )
    if not complete_role:
        decision = "smoke_only"
        conclusion = (
            f"仅使用 {len(paths)} 条 HW-calibration 序列完成冒烟检查，"
            "结果不用于冻结 S2。"
        )
    elif selected_result is None:
        decision = "reject"
        conclusion = (
            "没有 PoT 方案同时满足 Event IoU 损失门槛和零饱和要求，"
            "S2 不进入 Validation。"
        )
    else:
        decision = "freeze_for_validation"
        conclusion = (
            f"HW-calibration 选中 {selected}：Event IoU "
            f"{selected_result['event']['iou']:.4%}，相对 S1 变化 "
            f"{selected_result['event']['iou_delta_to_s1'] * 100:+.4f} 个百分点，"
            f"饱和 {selected_result['saturation_total']} 次；"
            "配置已冻结并进入 Validation。"
        )

    return {
        "schema_version": 1,
        "status": "ok" if selected is not None else "gate_failed",
        "experiment_id": arguments.experiment_id,
        "stage": "S2",
        "data_role": ROLE,
        "primary_metrics": primary_metrics,
        "decision": decision,
        "conclusion": conclusion,
        "scope": {
            "role": ROLE,
            "test_used": False,
            "validation_used": False,
            "training_performed": False,
            "complete_role": complete_role,
            "sequences": len(paths),
            "events": total_events,
            "target_events": total_targets,
            "fixed_models_run_on": "cpu_numpy_integer",
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
        },
        "arithmetic_contract": {
            "pot_weights": "零或带符号二次幂，运行时以移位和加减替代一般乘法",
            "quantization": "对 S1 实际权重取最近二次幂，低于最小指数的权重置零",
            "ties": "对数距离相等时选择绝对值更大的二次幂",
            "accumulation": "先按指数对齐到 Q12，再在 24 位累加器中饱和",
            "lif": "完全继承 S1 的 24 位 Q12 膜电位、7/8 泄漏和软复位",
        },
        "reference_s1": {
            "event": reference_event,
            "candidate_events": reference_candidates,
            "candidate_fraction": reference_candidates / total_events,
            "proposal_recall": reference_retained / total_targets,
        },
        "variants": results,
        "selection": {
            "maximum_iou_loss": arguments.maximum_iou_loss,
            "requires_zero_saturation": True,
            "acceptable_variants": acceptable,
            "full_pot_acceptable_variants": full_acceptable,
            "selected": selected,
            "export": export_path.relative_to(ROOT).as_posix()
            if export_path is not None
            else None,
            "export_sha256": export_sha256,
        },
        "per_sequence": per_sequence,
        "reproducibility": {
            "command": [str(value) for value in sys.argv],
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "fixed_source_sha256": sha256_file(
                ROOT / "software" / "wangdao" / "fixed.py"
            ),
        },
    }


def parse_integer_list(value: str) -> List[int]:
    result = []
    for item in value.split(","):
        number = int(item.strip())
        if number >= 0 or number < -14:
            raise ValueError("最小指数必须在 -14 到 -1 之间")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError("至少需要一个最小指数")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT_PATH)
    parser.add_argument("--s1-weight", type=Path, default=S1_WEIGHT_PATH)
    parser.add_argument(
        "--minimum-exponents", type=parse_integer_list, default=parse_integer_list("-7,-6,-5,-4")
    )
    parser.add_argument("--maximum-exponent", type=int, default=3)
    parser.add_argument("--maximum-iou-loss", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--experiment-id", default="S2-POT-HW-CAL-v1")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "software" / "weights")
    arguments = parser.parse_args()
    if arguments.batch_size <= 0 or arguments.max_sequences < 0:
        raise ValueError("batch-size 必须为正数，max-sequences 不能为负数")
    if arguments.maximum_exponent <= max(arguments.minimum_exponents):
        raise ValueError("最大指数必须大于所有最小指数")
    if arguments.maximum_iou_loss < 0.0:
        raise ValueError("maximum-iou-loss 不能为负数")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = evaluate(arguments)
    atomic_json(result, arguments.output)
    concise = {
        identifier: {
            "iou": value["event"]["iou"],
            "iou_delta_to_s1": value["event"]["iou_delta_to_s1"],
            "proposal_recall": value["proposal_recall"],
            "saturation_total": value["saturation_total"],
        }
        for identifier, value in result["variants"].items()
    }
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "conclusion": result["conclusion"],
                "reference_s1": result["reference_s1"],
                "variants": concise,
                "selection": result["selection"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
