"""在 HW-calibration 上统一统计 H2 ANN 与 S0-P SNN 的计算活动。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.h2.constants import (
    PATCHES_PER_WINDOW,
    PROPOSAL_DIMENSIONS,
    REFINER_DIMENSIONS,
    WINDOWS_PER_SEQUENCE,
)
from wangdao.h2.numpy_model import load_export, quantized_mlp_with_activity
from wangdao.h2.representation import candidate_indices, patch_scores_to_heatmaps
from wangdao.s0p import build_s0p_sequence
from wangdao.snn import load_proposal_snn_export, profile_proposal_snn


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
S0P_WEIGHT_PATH = ROOT / "software" / "weights" / "S0P_T4_POS15_KD50_S42.npz"
OUTPUT_PATH = ROOT / "results" / "compute_profile.json"
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


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已要求 CUDA，但当前环境不可用")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calibration_paths() -> List[Path]:
    """只解析 HW-calibration，避免画像脚本接触 Validation 或 Test。"""
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
    missing_names = [str(name) for name in names if str(name) not in by_name]
    if missing_names:
        raise ValueError("HW-calibration 引用了未知文件：" + ", ".join(missing_names))
    paths = [by_name[str(name)] for name in names]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError("数据文件缺失：" + ", ".join(absent))
    return paths


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def dense_macs(dimensions: Sequence[int]) -> int:
    return sum(int(left) * int(right) for left, right in zip(dimensions, dimensions[1:]))


def descriptive(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def profile(arguments: argparse.Namespace) -> Dict[str, Any]:
    device = choose_device(arguments.device)
    h2_path = arguments.h2_weight.resolve()
    s0p_path = arguments.s0p_weight.resolve()
    h2 = load_export(h2_path)
    s0p, s0p_config, s0p_metadata = load_proposal_snn_export(s0p_path)
    s0p = s0p.to(device).eval()
    time_steps = int(s0p_config["time_steps"])
    s0p_threshold = float(s0p_metadata["proposal_threshold"])
    paths = calibration_paths()
    if arguments.max_sequences:
        paths = paths[: arguments.max_sequences]

    proposal_macs_per_patch = dense_macs(PROPOSAL_DIMENSIONS)
    refiner_macs_per_event = dense_macs(REFINER_DIMENSIONS)
    per_sequence: List[Dict[str, Any]] = []
    totals = {
        "events": 0,
        "targets": 0,
        "patches": 0,
        "ann_activation_nonzero": [0, 0, 0],
        "ann_activation_values": [0, 0, 0],
        "ann_candidates": 0,
        "ann_retained_targets": 0,
        "snn_input_nonzero": 0,
        "snn_input_values": 0,
        "snn_spikes": [0, 0, 0],
        "snn_candidates": 0,
        "snn_retained_targets": 0,
    }
    started = time.perf_counter()

    for number, path in enumerate(paths, start=1):
        sequence_started = time.perf_counter()
        events = load_events(path, require_labels=True)
        sequence = build_s0p_sequence(path, time_steps)
        patches = int(sequence.ann_features.shape[0] * sequence.ann_features.shape[1])
        flat_ann = sequence.ann_features.reshape(patches, PROPOSAL_DIMENSIONS[0])
        ann_activity = quantized_mlp_with_activity(flat_ann, h2.proposal_layers)
        ann_heatmaps = patch_scores_to_heatmaps(
            sigmoid(ann_activity.logits).reshape(
                WINDOWS_PER_SEQUENCE, PATCHES_PER_WINDOW, PROPOSAL_DIMENSIONS[-1]
            )
        )
        windows = list(range(WINDOWS_PER_SEQUENCE))
        ann_candidates = candidate_indices(
            events, windows, ann_heatmaps, h2.proposal_threshold
        )

        temporal = torch.from_numpy(sequence.temporal_features).permute(1, 0, 2, 3)
        temporal = temporal.reshape(time_steps, patches, PROPOSAL_DIMENSIONS[0])
        snn_logits: List[np.ndarray] = []
        snn_input_nonzero = 0
        snn_spikes = [0, 0, 0]
        with torch.no_grad():
            for start in range(0, patches, arguments.batch_size):
                batch = temporal[:, start : start + arguments.batch_size].to(
                    device, non_blocking=True
                )
                activity = profile_proposal_snn(s0p, batch)
                snn_logits.append(activity.output.logits.cpu().numpy())
                snn_input_nonzero += activity.input_nonzero
                for index, count in enumerate(activity.spike_counts):
                    snn_spikes[index] += count
        flat_snn_logits = np.concatenate(snn_logits, axis=0)
        snn_heatmaps = patch_scores_to_heatmaps(
            sigmoid(flat_snn_logits).reshape(
                WINDOWS_PER_SEQUENCE, PATCHES_PER_WINDOW, PROPOSAL_DIMENSIONS[-1]
            )
        )
        snn_candidates = candidate_indices(
            events, windows, snn_heatmaps, s0p_threshold
        )

        labels = events["label"].astype(bool, copy=False)
        target_count = int(labels.sum())
        ann_retained = int(labels[ann_candidates].sum())
        snn_retained = int(labels[snn_candidates].sum())
        ann_scheduled = patches * proposal_macs_per_patch
        ann_zero_skip_layers = [
            int(nonzero) * int(output_dim)
            for nonzero, output_dim in zip(
                ann_activity.activation_nonzero, PROPOSAL_DIMENSIONS[1:]
            )
        ]
        snn_event_layers = [
            snn_input_nonzero * PROPOSAL_DIMENSIONS[1],
            snn_spikes[0] * PROPOSAL_DIMENSIONS[2],
            snn_spikes[1] * PROPOSAL_DIMENSIONS[3],
        ]
        snn_scheduled = time_steps * ann_scheduled
        snn_membrane_updates = time_steps * patches * (
            PROPOSAL_DIMENSIONS[1]
            + PROPOSAL_DIMENSIONS[2]
            + 2 * PROPOSAL_DIMENSIONS[3]
        )
        ann_hybrid_synaptic = ann_scheduled + len(ann_candidates) * refiner_macs_per_event
        snn_hybrid_synaptic = sum(snn_event_layers) + len(snn_candidates) * refiner_macs_per_event
        elapsed = time.perf_counter() - sequence_started
        record = {
            "file": path.name,
            "events": len(events),
            "target_events": target_count,
            "patches": patches,
            "ann": {
                "activation_nonzero": list(ann_activity.activation_nonzero),
                "activation_values": list(ann_activity.activation_values),
                "scheduled_macs": ann_scheduled,
                "zero_skip_product_proxy_by_layer": ann_zero_skip_layers,
                "zero_skip_product_proxy": sum(ann_zero_skip_layers),
                "candidate_events": len(ann_candidates),
                "candidate_fraction": safe_ratio(len(ann_candidates), len(events)),
                "proposal_recall": safe_ratio(ann_retained, target_count),
            },
            "snn": {
                "time_steps": time_steps,
                "input_nonzero": snn_input_nonzero,
                "input_values": int(temporal.numel()),
                "spike_counts": snn_spikes,
                "spike_rates": [
                    safe_ratio(
                        snn_spikes[index],
                        time_steps * patches * PROPOSAL_DIMENSIONS[index + 1],
                    )
                    for index in range(3)
                ],
                "scheduled_dense_synaptic_products": snn_scheduled,
                "event_driven_synaptic_accumulations_by_layer": snn_event_layers,
                "event_driven_synaptic_accumulations": sum(snn_event_layers),
                "membrane_updates": snn_membrane_updates,
                "candidate_events": len(snn_candidates),
                "candidate_fraction": safe_ratio(len(snn_candidates), len(events)),
                "proposal_recall": safe_ratio(snn_retained, target_count),
            },
            "comparison": {
                "proposal_event_driven_to_ann_dense_ratio": safe_ratio(
                    sum(snn_event_layers), ann_scheduled
                ),
                "hybrid_synaptic_proxy_ann": ann_hybrid_synaptic,
                "hybrid_synaptic_proxy_snn": snn_hybrid_synaptic,
                "hybrid_snn_to_ann_ratio": safe_ratio(
                    snn_hybrid_synaptic, ann_hybrid_synaptic
                ),
            },
            "elapsed_s": elapsed,
        }
        per_sequence.append(record)

        totals["events"] += len(events)
        totals["targets"] += target_count
        totals["patches"] += patches
        totals["ann_candidates"] += len(ann_candidates)
        totals["ann_retained_targets"] += ann_retained
        totals["snn_input_nonzero"] += snn_input_nonzero
        totals["snn_input_values"] += int(temporal.numel())
        totals["snn_candidates"] += len(snn_candidates)
        totals["snn_retained_targets"] += snn_retained
        for index in range(3):
            totals["ann_activation_nonzero"][index] += ann_activity.activation_nonzero[index]
            totals["ann_activation_values"][index] += ann_activity.activation_values[index]
            totals["snn_spikes"][index] += snn_spikes[index]
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"ANN候选={len(ann_candidates)}, SNN候选={len(snn_candidates)}, "
            f"SNN事件驱动比={record['comparison']['proposal_event_driven_to_ann_dense_ratio']:.4f}, "
            f"耗时={elapsed:.2f}s",
            flush=True,
        )

    patches = int(totals["patches"])
    ann_scheduled = patches * proposal_macs_per_patch
    ann_zero_skip_layers = [
        int(totals["ann_activation_nonzero"][index]) * PROPOSAL_DIMENSIONS[index + 1]
        for index in range(3)
    ]
    snn_event_layers = [
        int(totals["snn_input_nonzero"]) * PROPOSAL_DIMENSIONS[1],
        int(totals["snn_spikes"][0]) * PROPOSAL_DIMENSIONS[2],
        int(totals["snn_spikes"][1]) * PROPOSAL_DIMENSIONS[3],
    ]
    snn_event_total = sum(snn_event_layers)
    snn_scheduled = time_steps * ann_scheduled
    snn_membrane_updates = time_steps * patches * (
        PROPOSAL_DIMENSIONS[1]
        + PROPOSAL_DIMENSIONS[2]
        + 2 * PROPOSAL_DIMENSIONS[3]
    )
    ann_refiner = int(totals["ann_candidates"]) * refiner_macs_per_event
    snn_refiner = int(totals["snn_candidates"]) * refiner_macs_per_event
    ann_hybrid = ann_scheduled + ann_refiner
    snn_hybrid = snn_event_total + snn_refiner
    spike_denominators = [
        time_steps * patches * PROPOSAL_DIMENSIONS[index + 1]
        for index in range(3)
    ]
    s0p_weight_nonzero = []
    for layer in s0p.layers:
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError("S0-P Proposal 层必须是线性层")
        s0p_weight_nonzero.append(int(torch.count_nonzero(layer.weight).item()))
    h2_weight_nonzero = [
        int(np.count_nonzero(layer[0])) for layer in h2.proposal_layers
    ]

    result = {
        "status": "ok",
        "experiment_id": arguments.experiment_id,
        "scope": {
            "role": ROLE,
            "test_used": False,
            "validation_used": False,
            "training_performed": False,
            "complete_role": arguments.max_sequences == 0,
            "sequences": len(paths),
            "device": str(device),
            "elapsed_s": time.perf_counter() - started,
        },
        "inputs": {
            "dataset_index": INDEX_PATH.relative_to(ROOT).as_posix(),
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "h2_weight": H2_WEIGHT_PATH.relative_to(ROOT).as_posix()
            if h2_path == H2_WEIGHT_PATH.resolve()
            else str(h2_path),
            "h2_weight_sha256": sha256_file(h2_path),
            "s0p_weight": S0P_WEIGHT_PATH.relative_to(ROOT).as_posix()
            if s0p_path == S0P_WEIGHT_PATH.resolve()
            else str(s0p_path),
            "s0p_weight_sha256": sha256_file(s0p_path),
            "s0p_configuration": s0p_config,
            "s0p_metadata": s0p_metadata,
        },
        "definitions": {
            "ann_scheduled_mac": "稠密 ANN 每个输入通道与输出通道的一次整数乘加。",
            "ann_zero_skip_product_proxy": "假设硬件能跳过零激活时，需要执行的输入非零值×扇出数；不是当前 NumPy 延迟。",
            "snn_event_driven_synaptic_accumulation": "非零时间片输入或上游脉冲触发的一次权重累加；第一层仍含非二值 uint8×权重。",
            "membrane_update": "一个神经元在一个微时间步的一次膜状态更新；输出同时维护读出膜和发放膜，因此计两份。",
            "hybrid_synaptic_proxy": "Proposal 的对应突触运算代理加候选事件上的 H2 Refiner 稠密 MAC；不含预处理、路由、膜更新和后处理。",
            "logical_memory_access": "未假设片上缓存复用时的逻辑访问次数，用于后续体系结构设计，不能直接换算功耗。",
        },
        "architecture": {
            "proposal_dimensions": list(PROPOSAL_DIMENSIONS),
            "refiner_dimensions": list(REFINER_DIMENSIONS),
            "proposal_weights": proposal_macs_per_patch,
            "refiner_weights": refiner_macs_per_event,
            "h2_proposal_weight_nonzero": h2_weight_nonzero,
            "s0p_proposal_weight_nonzero": s0p_weight_nonzero,
            "current_weight_storage": {
                "h2_proposal_int8_weight_bytes": proposal_macs_per_patch,
                "s0p_proposal_fp32_weight_bytes": 4 * proposal_macs_per_patch,
                "note": "S0-P 尚未定点；此处是当前文件格式，不是最终 FPGA 位宽。",
            },
        },
        "aggregate": {
            "events": totals["events"],
            "target_events": totals["targets"],
            "patches": patches,
            "ann_proposal": {
                "activation_nonzero": totals["ann_activation_nonzero"],
                "activation_values": totals["ann_activation_values"],
                "activation_nonzero_rates": [
                    safe_ratio(nonzero, size)
                    for nonzero, size in zip(
                        totals["ann_activation_nonzero"], totals["ann_activation_values"]
                    )
                ],
                "scheduled_macs": ann_scheduled,
                "zero_skip_product_proxy_by_layer": ann_zero_skip_layers,
                "zero_skip_product_proxy": sum(ann_zero_skip_layers),
                "candidate_events": totals["ann_candidates"],
                "candidate_fraction": safe_ratio(totals["ann_candidates"], totals["events"]),
                "proposal_recall": safe_ratio(
                    totals["ann_retained_targets"], totals["targets"]
                ),
                "logical_memory_access": {
                    "weight_reads_dense": ann_scheduled,
                    "activation_reads_dense": ann_scheduled,
                    "accumulator_updates_dense": ann_scheduled,
                    "activation_writes": patches * sum(PROPOSAL_DIMENSIONS[1:]),
                },
            },
            "snn_proposal": {
                "time_steps": time_steps,
                "input_nonzero": totals["snn_input_nonzero"],
                "input_values": totals["snn_input_values"],
                "input_nonzero_rate": safe_ratio(
                    totals["snn_input_nonzero"], totals["snn_input_values"]
                ),
                "spike_counts": totals["snn_spikes"],
                "spike_rates": [
                    safe_ratio(totals["snn_spikes"][index], spike_denominators[index])
                    for index in range(3)
                ],
                "scheduled_dense_synaptic_products": snn_scheduled,
                "event_driven_synaptic_accumulations_by_layer": snn_event_layers,
                "event_driven_synaptic_accumulations": snn_event_total,
                "membrane_updates": snn_membrane_updates,
                "candidate_events": totals["snn_candidates"],
                "candidate_fraction": safe_ratio(totals["snn_candidates"], totals["events"]),
                "proposal_recall": safe_ratio(
                    totals["snn_retained_targets"], totals["targets"]
                ),
                "logical_memory_access": {
                    "weight_reads_event_driven": snn_event_total,
                    "synaptic_accumulator_updates": snn_event_total,
                    "membrane_state_reads": snn_membrane_updates,
                    "membrane_state_writes": snn_membrane_updates,
                    "spike_writes": sum(totals["snn_spikes"]),
                },
            },
            "hybrid_comparison": {
                "ann_refiner_macs": ann_refiner,
                "snn_route_refiner_macs": snn_refiner,
                "ann_total_synaptic_proxy": ann_hybrid,
                "snn_total_synaptic_proxy": snn_hybrid,
                "proposal_snn_event_driven_to_ann_dense_ratio": safe_ratio(
                    snn_event_total, ann_scheduled
                ),
                "proposal_snn_event_driven_reduction": 1.0
                - safe_ratio(snn_event_total, ann_scheduled),
                "hybrid_snn_to_ann_synaptic_ratio": safe_ratio(snn_hybrid, ann_hybrid),
                "hybrid_synaptic_reduction": 1.0 - safe_ratio(snn_hybrid, ann_hybrid),
                "snn_membrane_updates_reported_separately": snn_membrane_updates,
            },
        },
        "distribution": {
            "ann_candidate_fraction": descriptive(
                [item["ann"]["candidate_fraction"] for item in per_sequence]
            ),
            "snn_candidate_fraction": descriptive(
                [item["snn"]["candidate_fraction"] for item in per_sequence]
            ),
            "snn_input_nonzero_rate": descriptive(
                [
                    safe_ratio(item["snn"]["input_nonzero"], item["snn"]["input_values"])
                    for item in per_sequence
                ]
            ),
            "snn_spike_rates": [
                descriptive([item["snn"]["spike_rates"][index] for item in per_sequence])
                for index in range(3)
            ],
            "proposal_snn_to_ann_ratio": descriptive(
                [item["comparison"]["proposal_event_driven_to_ann_dense_ratio"] for item in per_sequence]
            ),
            "hybrid_snn_to_ann_ratio": descriptive(
                [item["comparison"]["hybrid_snn_to_ann_ratio"] for item in per_sequence]
            ),
        },
        "interpretation_limits": [
            "这些是模型活动与逻辑访问计数，不是 FPGA 延迟、功耗或能量实测。",
            "第一层的非零时间片输入是 uint8 计数，仍需要乘法或后续 PoT/移位实现，不能按二值脉冲累加处理。",
            "SNN 的突触运算减少必须与额外膜更新、T 个时间步、控制逻辑和状态存储一起评估。",
            "下一阶段需用位精确 ANN/SNN 黄金模型和同器件 RTL 结果验证真实收益。",
        ],
        "per_sequence": per_sequence,
        "reproducibility": {
            "command": [str(value) for value in sys.argv],
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "snn_source_sha256": sha256_file(ROOT / "software" / "wangdao" / "snn.py"),
            "h2_numpy_source_sha256": sha256_file(
                ROOT / "software" / "wangdao" / "h2" / "numpy_model.py"
            ),
        },
    }
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT_PATH)
    parser.add_argument("--s0p-weight", type=Path, default=S0P_WEIGHT_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--experiment-id", default="HW-CAL-COMPUTE-PROFILE-v1")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("batch-size 必须为正数")
    if arguments.max_sequences < 0:
        raise ValueError("max-sequences 不能为负数")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = profile(arguments)
    atomic_json(result, arguments.output)
    aggregate = result["aggregate"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "role": result["scope"]["role"],
                "sequences": result["scope"]["sequences"],
                "proposal_snn_to_ann_ratio": aggregate["hybrid_comparison"][
                    "proposal_snn_event_driven_to_ann_dense_ratio"
                ],
                "hybrid_snn_to_ann_ratio": aggregate["hybrid_comparison"][
                    "hybrid_snn_to_ann_synaptic_ratio"
                ],
                "snn_input_nonzero_rate": aggregate["snn_proposal"][
                    "input_nonzero_rate"
                ],
                "snn_spike_rates": aggregate["snn_proposal"]["spike_rates"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
