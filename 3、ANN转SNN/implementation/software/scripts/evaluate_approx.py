"""在 HW-calibration 上比较 S1 与 S3 单因素 LOA 候选。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
sys.path.insert(0, str(ROOT / "software" / "scripts"))

from evaluate_s1 import (
    add_counts,
    atomic_json,
    empty_totals,
    finalize,
    fixed_mlp_batched,
    proposal_heatmaps,
    sha256_file,
)
from wangdao.data import load_events
from wangdao.fixed import (
    ApproximateProposalSNN,
    FixedH2,
    FixedH2Config,
    FixedProposalSNN,
    LOAPlacement,
    TorchApproximateProposalSNN,
    export_approximate_proposal_snn,
    load_fixed_proposal_snn,
)
from wangdao.h2.constants import PROPOSAL_DIMENSIONS, WINDOWS_PER_SEQUENCE
from wangdao.h2.numpy_model import _refiner_probability, load_export
from wangdao.h2.representation import build_refiner_features, candidate_indices
from wangdao.metrics import event_metrics, localization_metrics
from wangdao.s0p import build_s0p_sequence


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
S1_WEIGHT_PATH = (
    ROOT / "software" / "weights" / "S1P_INT8_ACC24_MEM24_Q12.npz"
)
OUTPUT_PATH = ROOT / "results" / "approx_calibration.json"
ROLE = "hw_calibration"
LOW_BITS = (2, 4, 6, 8)
GROUPS: Dict[str, Tuple[Tuple[bool, bool, bool], Tuple[bool, bool, bool]]] = {
    "syn_l1": ((True, False, False), (False, False, False)),
    "syn_l2": ((False, True, False), (False, False, False)),
    "syn_l3": ((False, False, True), (False, False, False)),
    "syn_all": ((True, True, True), (False, False, False)),
    "mem_l1": ((False, False, False), (True, False, False)),
    "mem_l2": ((False, False, False), (False, True, False)),
    "mem_readout": ((False, False, False), (False, False, True)),
    "mem_all": ((False, False, False), (True, True, True)),
}
COMBINATIONS: Dict[
    str, Tuple[int, Tuple[bool, bool, bool], Tuple[bool, bool, bool]]
] = {
    "syn1_memall_k2": (2, (True, False, False), (True, True, True)),
    "syn1_memall_k4": (4, (True, False, False), (True, True, True)),
    "syn1_memall_k6": (6, (True, False, False), (True, True, True)),
    "syn1_memall_k8": (8, (True, False, False), (True, True, True)),
    "syn13_k2": (2, (True, False, True), (False, False, False)),
    "syn13_k4": (4, (True, False, True), (False, False, False)),
    "syn13_memall_k2": (2, (True, False, True), (True, True, True)),
    "syn13_memall_k4": (4, (True, False, True), (True, True, True)),
}


def calibration_paths() -> List[Path]:
    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
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


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("S3 扫描只支持 cpu 或 cuda")
    return device


def run_proposal(
    model: Any,
    temporal: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[int], List[int]]:
    logits_q = []
    logits = []
    accumulator_saturations = [0, 0, 0]
    membrane_saturations = [0, 0, 0, 0]
    spikes = [0, 0, 0]
    approximate_additions = [0, 0, 0, 0, 0, 0]
    for start in range(0, temporal.shape[1], batch_size):
        output = model.forward(temporal[:, start : start + batch_size])
        logits_q.append(output.logits_q)
        logits.append(output.logits)
        add_counts(accumulator_saturations, output.accumulator_saturations)
        add_counts(membrane_saturations, output.membrane_saturations)
        add_counts(spikes, output.spike_counts)
        additions = getattr(output, "approximate_additions", (0, 0, 0, 0, 0, 0))
        add_counts(approximate_additions, additions)
    return (
        np.concatenate(logits_q),
        np.concatenate(logits),
        accumulator_saturations,
        membrane_saturations,
        spikes,
        approximate_additions,
    )


def event_output(
    events: np.ndarray,
    logits: np.ndarray,
    fixed_h2: FixedH2,
    h2: Any,
    proposal_threshold: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    windows = list(range(WINDOWS_PER_SEQUENCE))
    heatmaps = proposal_heatmaps(logits)
    candidates = candidate_indices(events, windows, heatmaps, proposal_threshold)
    probability = np.zeros(len(events), dtype=np.float32)
    refiner_saturations: Tuple[int, ...] = (0, 0, 0)
    if len(candidates):
        features = build_refiner_features(
            events, candidates, windows, heatmaps, proposal_threshold
        )
        refiner = fixed_mlp_batched(fixed_h2.refiner, features, batch_size)
        refiner_saturations = refiner.accumulator_saturations
        probability[candidates] = _refiner_probability(
            refiner.logits, h2.prototypes
        )
    return probability, candidates, refiner_saturations


def update_totals(
    totals: Dict[str, Any],
    events: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    candidates: np.ndarray,
    event_threshold: float,
) -> Dict[str, Any]:
    localization = localization_metrics(
        events["t"],
        events["x"],
        events["y"],
        labels,
        events["name"],
        probability,
        threshold=event_threshold,
    )
    target_count = int(labels.sum())
    retained = int(labels[candidates].sum())
    totals["probabilities"].append(probability)
    totals["candidate_events"] += len(candidates)
    totals["retained_targets"] += retained
    totals["correct_targets"] += localization["correct_targets"]
    totals["localization_targets"] += localization["targets"]
    totals["false_components"] += localization["false_components"]
    totals["frame_denominator"] += localization["frame_denominator"]
    return {
        "event": event_metrics(labels, probability, event_threshold),
        "pd": localization["correct_targets"] / localization["targets"]
        if localization["targets"]
        else 1.0,
        "proposal_recall": retained / target_count if target_count else 1.0,
        "candidate_fraction": len(candidates) / len(events),
    }


def empty_model_state() -> Dict[str, Any]:
    return {
        "totals": empty_totals(),
        "accumulator_saturations": [0, 0, 0],
        "membrane_saturations": [0, 0, 0, 0],
        "refiner_saturations": [0, 0, 0],
        "spikes": [0, 0, 0],
        "approximate_additions": [0, 0, 0, 0, 0, 0],
        "logit_absolute_error_sum": 0,
        "logit_difference_count": 0,
        "logit_maximum_absolute_error": 0,
    }


def parse_low_bits(value: str) -> List[int]:
    result = []
    for item in value.split(","):
        number = int(item.strip())
        if number not in LOW_BITS:
            raise ValueError("S3 首轮低位数只能从 2、4、6、8 中选择")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError("至少需要一个近似低位数")
    return result


def parse_groups(value: str) -> List[str]:
    result = []
    for item in value.split(","):
        name = item.strip()
        if name not in GROUPS:
            raise ValueError(f"未知 S3 扫描组：{name}")
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("至少需要一个 S3 扫描组")
    return result


def pareto_front(results: Dict[str, Dict[str, Any]], acceptable: Sequence[str]) -> List[str]:
    front = []
    for candidate in acceptable:
        candidate_result = results[candidate]
        candidate_count = candidate_result["approximate_addition_total"]
        candidate_iou = candidate_result["event"]["iou"]
        dominated = False
        for other in acceptable:
            if other == candidate:
                continue
            other_result = results[other]
            other_count = other_result["approximate_addition_total"]
            other_iou = other_result["event"]["iou"]
            if (
                other_count >= candidate_count
                and other_iou >= candidate_iou
                and (other_count > candidate_count or other_iou > candidate_iou)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(
        front,
        key=lambda name: (
            -results[name]["approximate_addition_total"],
            -results[name]["event"]["iou"],
            name,
        ),
    )


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    h2_path = arguments.h2_weight.resolve()
    s1_path = arguments.s1_weight.resolve()
    h2 = load_export(h2_path)
    s1, s1_configuration, s1_metadata = load_fixed_proposal_snn(s1_path)
    h2_configuration = s1_metadata.get("h2_fixed_configuration")
    if not isinstance(h2_configuration, dict):
        raise ValueError("S1 缺少 H2 定点配置")
    if s1_metadata.get("dataset_index_sha256") != sha256_file(INDEX_PATH):
        raise ValueError("S1 使用的数据索引与当前仓库不一致")
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

    device = choose_device(arguments.device)

    def approximate_model(placement: LOAPlacement) -> Any:
        if device.type == "cuda":
            return TorchApproximateProposalSNN(s1, placement, device)
        return ApproximateProposalSNN(s1, placement)

    exact_tree = approximate_model(
        LOAPlacement(
            low_bits=0,
            synaptic_layers=(True, True, True),
            membrane_updates=(True, True, True),
        ),
    )
    models: Dict[str, Any] = {}
    definitions: Dict[str, Dict[str, Any]] = {}
    single_factor_source: Dict[str, Any] | None = None
    single_factor_path = arguments.single_factor_result.resolve()
    if arguments.exact_only:
        candidates = []
    elif arguments.phase == "single":
        candidates = [
            (
                f"{group}_k{low_bits}",
                low_bits,
                GROUPS[group][0],
                GROUPS[group][1],
                group,
            )
            for group in arguments.groups
            for low_bits in arguments.low_bits
        ]
    else:
        with single_factor_path.open("r", encoding="utf-8") as handle:
            source_result = json.load(handle)
        if (
            not isinstance(source_result, dict)
            or source_result.get("decision") != "proceed_to_combination"
            or not source_result.get("scope", {}).get("complete_scan")
        ):
            raise ValueError("组合扫描要求完整且已授权的单因素结果")
        source_inputs = source_result.get("inputs", {})
        if source_inputs.get("s1_weight_sha256") != sha256_file(s1_path):
            raise ValueError("单因素结果与当前 S1 权重不一致")
        single_factor_source = {
            "path": single_factor_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(single_factor_path),
            "experiment_id": source_result.get("experiment_id"),
            "decision": source_result.get("decision"),
            "selected_single_factor": source_result.get("selection", {}).get(
                "selected_single_factor"
            ),
            "acceptable_variants": source_result.get("selection", {}).get(
                "acceptable_variants"
            ),
        }
        candidates = [
            (identifier, low_bits, synaptic, membrane, "combination")
            for identifier, (low_bits, synaptic, membrane) in COMBINATIONS.items()
        ]
    for identifier, low_bits, synaptic, membrane, group in candidates:
        placement = LOAPlacement(
            low_bits=low_bits,
            synaptic_layers=synaptic,
            membrane_updates=membrane,
        )
        models[identifier] = approximate_model(placement)
        definitions[identifier] = {
            "group": group,
            "low_bits": low_bits,
            "synaptic_layers": list(synaptic),
            "membrane_updates": list(membrane),
        }

    paths = calibration_paths()
    if arguments.max_sequences:
        paths = paths[: arguments.max_sequences]
    reference_state = empty_model_state()
    variant_states = {name: empty_model_state() for name in models}
    labels_parts: List[np.ndarray] = []
    total_events = 0
    total_targets = 0
    total_patches = 0
    exact_batches = 0
    exact_logits = 0
    per_sequence: List[Dict[str, Any]] = []
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

        reference = run_proposal(s1, temporal, arguments.batch_size)
        exact_patches = (
            patches
            if arguments.exact_check_patches == 0
            else min(patches, arguments.exact_check_patches)
        )
        exact_reference = run_proposal(
            s1, temporal[:, :exact_patches], arguments.batch_size
        )
        exact = run_proposal(
            exact_tree, temporal[:, :exact_patches], arguments.batch_size
        )
        exact_batches += (
            exact_patches + arguments.batch_size - 1
        ) // arguments.batch_size
        exact_logits += int(exact[0].size)
        if not np.array_equal(exact_reference[0], exact[0]):
            difference = np.abs(exact_reference[0] - exact[0])
            raise RuntimeError(
                "k=0 固定加法树未逐位复现 S1："
                f"{path.name} 最大差 {int(difference.max())}，"
                f"不一致 {int(np.count_nonzero(difference))} 项"
            )
        if exact_reference[2:5] != exact[2:5]:
            raise RuntimeError(f"k=0 固定加法树统计未复现 S1：{path.name}")

        reference_probability, reference_candidates, reference_refiner = event_output(
            events,
            reference[1],
            fixed_h2,
            h2,
            proposal_threshold,
            arguments.batch_size,
        )
        reference_record = update_totals(
            reference_state["totals"],
            events,
            labels,
            reference_probability,
            reference_candidates,
            event_threshold,
        )
        add_counts(reference_state["accumulator_saturations"], tuple(reference[2]))
        add_counts(reference_state["membrane_saturations"], tuple(reference[3]))
        add_counts(reference_state["spikes"], tuple(reference[4]))
        add_counts(reference_state["refiner_saturations"], reference_refiner)

        sequence_record: Dict[str, Any] = {
            "file": path.name,
            "events": len(events),
            "target_events": int(labels.sum()),
            "reference": reference_record,
            "variants": {},
        }
        for identifier, model in models.items():
            output = run_proposal(model, temporal, arguments.batch_size)
            difference = np.abs(output[0] - reference[0])
            probability, candidates, refiner_saturations = event_output(
                events,
                output[1],
                fixed_h2,
                h2,
                proposal_threshold,
                arguments.batch_size,
            )
            state = variant_states[identifier]
            sequence_record["variants"][identifier] = update_totals(
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
            state["logit_absolute_error_sum"] += int(difference.sum())
            state["logit_difference_count"] += int(np.count_nonzero(difference))
            state["logit_maximum_absolute_error"] = max(
                state["logit_maximum_absolute_error"], int(difference.max())
            )

        labels_parts.append(labels)
        total_events += len(events)
        total_targets += int(labels.sum())
        total_patches += patches
        sequence_record["elapsed_s"] = time.perf_counter() - sequence_started
        per_sequence.append(sequence_record)
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"S1 IoU={reference_record['event']['iou']:.4f}，"
            f"候选={len(models)}，耗时={sequence_record['elapsed_s']:.2f}s",
            flush=True,
        )

    labels = np.concatenate(labels_parts)
    reference = finalize(
        reference_state["totals"], labels, event_threshold, total_events, total_targets
    )
    reference["proposal_accumulator_saturations"] = reference_state[
        "accumulator_saturations"
    ]
    reference["proposal_membrane_saturations"] = reference_state[
        "membrane_saturations"
    ]
    reference["refiner_accumulator_saturations"] = reference_state[
        "refiner_saturations"
    ]
    reference["spike_counts"] = reference_state["spikes"]
    spike_denominators = [
        s1.time_steps * total_patches * dimension
        for dimension in PROPOSAL_DIMENSIONS[1:]
    ]
    reference["spike_rates"] = [
        count / denominator
        for count, denominator in zip(reference_state["spikes"], spike_denominators)
    ]
    reference_saturation_total = sum(reference_state["accumulator_saturations"]) + sum(
        reference_state["membrane_saturations"]
    ) + sum(reference_state["refiner_saturations"])
    reference_probability = np.concatenate(reference_state["totals"]["probabilities"])

    results: Dict[str, Dict[str, Any]] = {}
    for identifier, state in variant_states.items():
        result = finalize(
            state["totals"], labels, event_threshold, total_events, total_targets
        )
        probability = np.concatenate(state["totals"]["probabilities"])
        result["definition"] = definitions[identifier]
        result["event"]["iou_delta_to_s1"] = (
            result["event"]["iou"] - reference["event"]["iou"]
        )
        result["pd_delta_to_s1"] = result["pd"] - reference["pd"]
        result["proposal_recall_delta_to_s1"] = (
            result["proposal_recall"] - reference["proposal_recall"]
        )
        result["candidate_fraction_delta_to_s1"] = (
            result["candidate_fraction"] - reference["candidate_fraction"]
        )
        result["proposal_accumulator_saturations"] = state[
            "accumulator_saturations"
        ]
        result["proposal_membrane_saturations"] = state["membrane_saturations"]
        result["refiner_accumulator_saturations"] = state["refiner_saturations"]
        result["saturation_total"] = sum(state["accumulator_saturations"]) + sum(
            state["membrane_saturations"]
        ) + sum(state["refiner_saturations"])
        result["spike_counts"] = state["spikes"]
        result["spike_rates"] = [
            count / denominator
            for count, denominator in zip(state["spikes"], spike_denominators)
        ]
        result["approximate_additions"] = state["approximate_additions"]
        result["approximate_addition_total"] = sum(state["approximate_additions"])
        result["logit_difference_to_s1"] = {
            "different": state["logit_difference_count"],
            "fraction": state["logit_difference_count"] / (total_patches * 16),
            "mean_absolute_q": state["logit_absolute_error_sum"]
            / (total_patches * 16),
            "maximum_absolute_q": state["logit_maximum_absolute_error"],
        }
        result["event"]["probability_mae_to_s1"] = float(
            np.mean(np.abs(probability.astype(np.float64) - reference_probability))
        )
        result["event"]["prediction_disagreement_to_s1"] = float(
            np.mean(
                (probability >= event_threshold)
                != (reference_probability >= event_threshold)
            )
        )
        result["gates"] = {
            "event_iou_loss_within_1_5_points": result["event"][
                "iou_delta_to_s1"
            ]
            >= -arguments.maximum_iou_loss,
            "proposal_recall_loss_within_1_point": result[
                "proposal_recall_delta_to_s1"
            ]
            >= -arguments.maximum_proposal_recall_loss,
            "candidate_growth_within_10_points": result[
                "candidate_fraction_delta_to_s1"
            ]
            <= arguments.maximum_candidate_growth,
            "no_new_saturation": result["saturation_total"]
            <= reference_saturation_total,
            "changes_rtl_additions": result["approximate_addition_total"] > 0,
        }
        result["passes"] = all(result["gates"].values())
        results[identifier] = result

    acceptable = [name for name, result in results.items() if result["passes"]]
    front = pareto_front(results, acceptable)
    selected = (
        max(
            acceptable,
            key=lambda name: (
                results[name]["approximate_addition_total"],
                results[name]["event"]["iou"],
            ),
        )
        if acceptable
        else None
    )
    complete_scan = arguments.max_sequences == 0 and not arguments.exact_only and (
        (
            arguments.phase == "single"
            and arguments.low_bits == list(LOW_BITS)
            and arguments.groups == list(GROUPS)
        )
        or (
            arguments.phase == "combination"
            and list(models) == list(COMBINATIONS)
        )
    )
    export_path: Path | None = None
    export_sha256: str | None = None
    if complete_scan and selected is not None and arguments.phase == "combination":
        placement = models[selected].placement
        candidate_export_path = arguments.export_dir / (
            f"S3P_LOA_{selected.upper()}_ACC24_MEM24_Q12.npz"
        )
        metadata = {
            "model_id": candidate_export_path.stem,
            "source_s1_weight": s1_path.name,
            "source_s1_weight_sha256": sha256_file(s1_path),
            "h2_weight_sha256": sha256_file(h2_path),
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "calibration_role": ROLE,
            "calibration_result": arguments.output.name,
            "single_factor_result": single_factor_path.name,
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
            "h2_fixed_configuration": h2_configuration,
            "selection_rule": {
                "maximum_iou_loss": arguments.maximum_iou_loss,
                "maximum_proposal_recall_loss": arguments.maximum_proposal_recall_loss,
                "maximum_candidate_growth": arguments.maximum_candidate_growth,
                "requires_no_new_saturation": True,
                "ranking": "先最大化真实近似加法次数，再最大化 Event IoU",
            },
        }
        export_approximate_proposal_snn(
            candidate_export_path,
            ApproximateProposalSNN(s1, placement),
            metadata,
        )
        export_path = candidate_export_path
        export_sha256 = sha256_file(candidate_export_path)
    exact_alignment_complete = (
        arguments.exact_only
        and arguments.max_sequences == 0
        and arguments.exact_check_patches == 0
    )
    if arguments.exact_only:
        decision = "exact_alignment_passed"
        conclusion = (
            f"k=0 固定平衡树在 {len(paths)} 条 HW-calibration 序列、"
            f"{exact_logits} 个 logits 上逐位复现 S1，统计完全一致。"
        )
    elif not complete_scan:
        decision = "smoke_only"
        conclusion = (
            f"使用 {len(paths)} 条 HW-calibration 序列和 {len(models)} 个候选完成冒烟，"
            "结果不用于冻结 S3。"
        )
    elif selected is None and arguments.phase == "single":
        decision = "reject_single_factor"
        conclusion = "没有单因素 LOA 候选通过全部门槛，S3 不进入组合或 Validation。"
    elif selected is None:
        decision = "reject_combination"
        conclusion = "没有组合 LOA 候选通过全部门槛，S3 停止且不运行 Validation。"
    elif arguments.phase == "single":
        decision = "proceed_to_combination"
        conclusion = (
            f"单因素扫描有 {len(acceptable)} 个候选通过，优先候选 {selected}；"
            "允许在 HW-calibration 上进行一次组合选择，不运行 Validation。"
        )
    else:
        decision = "freeze_for_validation"
        conclusion = (
            f"组合扫描选中 {selected}：Event IoU "
            f"{results[selected]['event']['iou']:.4%}，相对 S1 变化 "
            f"{results[selected]['event']['iou_delta_to_s1'] * 100:+.4f} 个百分点，"
            f"近似 {results[selected]['approximate_addition_total']} 次加法；"
            "配置已冻结，允许执行唯一一次 Validation。"
        )

    selected_result = results[selected] if selected is not None else None
    return {
        "schema_version": 1,
        "status": "ok"
        if selected is not None or not complete_scan or exact_alignment_complete
        else "gate_failed",
        "experiment_id": arguments.experiment_id,
        "stage": "S3-exact-alignment"
        if arguments.exact_only
        else (
            "S3-single-factor"
            if arguments.phase == "single"
            else "S3-combination"
        ),
        "data_role": ROLE,
        "primary_metrics": {
            "reference_s1_event_iou": reference["event"]["iou"],
            "reference_s1_proposal_recall": reference["proposal_recall"],
            "acceptable_variant_count": len(acceptable),
            "selected_variant": selected,
            "selected_event_iou": selected_result["event"]["iou"]
            if selected_result is not None
            else None,
            "selected_approximate_additions": selected_result[
                "approximate_addition_total"
            ]
            if selected_result is not None
            else None,
        },
        "decision": decision,
        "conclusion": conclusion,
        "scope": {
            "role": ROLE,
            "test_used": False,
            "validation_used": False,
            "training_performed": False,
            "complete_scan": complete_scan,
            "complete_exact_alignment": exact_alignment_complete,
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
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
            "single_factor_source": single_factor_source,
        },
        "arithmetic_contract": {
            "width": 24,
            "low_bits": arguments.low_bits,
            "low_output": "对应输入低位逐位或",
            "carry_to_exact": "两个输入第 k-1 位逐位与",
            "high_output": "从第 k 位开始精确相加并接收近似进位",
            "tree": "输入通道按索引升序进行固定平衡二叉树归约，奇数尾项直接传递",
            "overflow": "每个树节点和膜更新输出均执行 24 位有符号饱和",
            "lif_unchanged": "泄漏舍入、阈值、软复位、读出平均和统计膜保持 S1 exact",
        },
        "exact_tree_alignment": {
            "matches_s1": True,
            "checked_batches": exact_batches,
            "checked_logits": exact_logits,
            "checked_sequences": len(paths),
            "maximum_patches_per_sequence": arguments.exact_check_patches
            if arguments.exact_check_patches
            else None,
            "statistics_match": True,
            "placements": "syn_all+mem_all",
            "low_bits": 0,
            "backend": "torch_cuda_integer"
            if device.type == "cuda"
            else "cpu_numpy_integer",
        },
        "reference_s1": reference,
        "variants": results,
        "selection": {
            "maximum_iou_loss": arguments.maximum_iou_loss,
            "maximum_proposal_recall_loss": arguments.maximum_proposal_recall_loss,
            "maximum_candidate_growth": arguments.maximum_candidate_growth,
            "requires_no_new_saturation": True,
            "acceptable_variants": acceptable,
            "pareto_front": front,
            "selected_single_factor": selected,
            "ranking": "先最大化真实近似加法次数，再最大化 Event IoU",
            "frozen_export": export_path.relative_to(ROOT).as_posix()
            if export_path is not None
            else None,
            "frozen_export_sha256": export_sha256,
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT_PATH)
    parser.add_argument("--s1-weight", type=Path, default=S1_WEIGHT_PATH)
    parser.add_argument("--phase", choices=("single", "combination"), default="single")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument(
        "--single-factor-result",
        type=Path,
        default=OUTPUT_PATH,
    )
    parser.add_argument("--low-bits", type=parse_low_bits, default=parse_low_bits("2,4,6,8"))
    parser.add_argument("--groups", type=parse_groups, default=parse_groups(",".join(GROUPS)))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exact-check-patches", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--maximum-iou-loss", type=float, default=0.015)
    parser.add_argument("--maximum-proposal-recall-loss", type=float, default=0.01)
    parser.add_argument("--maximum-candidate-growth", type=float, default=0.10)
    parser.add_argument("--experiment-id", default="S3-LOA-HW-CAL-v1")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--export-dir", type=Path, default=ROOT / "software" / "weights"
    )
    arguments = parser.parse_args()
    if (
        arguments.batch_size <= 0
        or arguments.exact_check_patches < 0
        or arguments.max_sequences < 0
    ):
        raise ValueError(
            "batch-size 必须为正数，exact-check-patches 和 max-sequences 不能为负数"
        )
    if arguments.exact_only and arguments.phase != "single":
        raise ValueError("exact-only 不能与组合扫描同时使用")
    if min(
        arguments.maximum_iou_loss,
        arguments.maximum_proposal_recall_loss,
        arguments.maximum_candidate_growth,
    ) < 0.0:
        raise ValueError("S3 选择门槛不能为负数")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = evaluate(arguments)
    atomic_json(result, arguments.output)
    concise = {
        name: {
            "iou": value["event"]["iou"],
            "iou_delta_to_s1": value["event"]["iou_delta_to_s1"],
            "proposal_recall": value["proposal_recall"],
            "candidate_fraction": value["candidate_fraction"],
            "approximate_additions": value["approximate_addition_total"],
            "passes": value["passes"],
        }
        for name, value in result["variants"].items()
    }
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "conclusion": result["conclusion"],
                "exact_tree_alignment": result["exact_tree_alignment"],
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
