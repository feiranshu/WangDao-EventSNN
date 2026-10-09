"""在 HW-calibration 上扫描 A1-H2 与 S1-P 的定点位宽并冻结最佳 S1。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.fixed import (
    FixedH2,
    FixedH2Config,
    FixedMLPOutput,
    FixedProposalSNN,
    FixedSNNConfig,
    export_fixed_proposal_snn,
)
from wangdao.h2.constants import PATCHES_PER_WINDOW, PROPOSAL_DIMENSIONS, WINDOWS_PER_SEQUENCE
from wangdao.h2.numpy_model import (
    _refiner_probability,
    load_export,
    quantized_mlp,
    run_pipeline,
)
from wangdao.h2.representation import (
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
)
from wangdao.metrics import event_metrics
from wangdao.s0p import build_s0p_sequence
from wangdao.snn import load_proposal_snn_export


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
S0P_WEIGHT_PATH = ROOT / "software" / "weights" / "S0P_T4_POS15_KD50_S42.npz"
OUTPUT_PATH = ROOT / "results" / "fixed_calibration.json"
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


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已要求 CUDA，但当前环境不可用")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def fixed_mlp_batched(
    function: Any, features: np.ndarray, batch_size: int
) -> FixedMLPOutput:
    logits_q = []
    logits = []
    saturation_counts: List[int] = []
    for start in range(0, len(features), batch_size):
        output = function(features[start : start + batch_size])
        if not isinstance(output, FixedMLPOutput):
            raise TypeError("定点 MLP 必须返回 FixedMLPOutput")
        logits_q.append(output.logits_q)
        logits.append(output.logits)
        if not saturation_counts:
            saturation_counts = [0] * len(output.accumulator_saturations)
        for index, count in enumerate(output.accumulator_saturations):
            saturation_counts[index] += count
    output_dimension = 0 if not len(features) else logits[0].shape[1]
    return FixedMLPOutput(
        logits_q=np.concatenate(logits_q, axis=0)
        if logits_q
        else np.empty((0, output_dimension), dtype=np.int64),
        logits=np.concatenate(logits, axis=0)
        if logits
        else np.empty((0, output_dimension), dtype=np.float32),
        accumulator_saturations=tuple(saturation_counts),
    )


def proposal_heatmaps(logits: np.ndarray) -> np.ndarray:
    return patch_scores_to_heatmaps(
        sigmoid(logits).reshape(
            WINDOWS_PER_SEQUENCE, PATCHES_PER_WINDOW, PROPOSAL_DIMENSIONS[-1]
        )
    )


def float_s0p_logits(
    model: torch.nn.Module,
    temporal: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    pieces = []
    with torch.no_grad():
        for start in range(0, temporal.shape[1], batch_size):
            batch = temporal[:, start : start + batch_size].to(
                device, non_blocking=True
            )
            output = model(batch)
            pieces.append(output.logits.cpu().numpy())
    return np.concatenate(pieces, axis=0)


def float_refiner_probabilities(
    features: np.ndarray, h2: Any
) -> np.ndarray:
    raw = quantized_mlp(features, h2.refiner_layers)
    return _refiner_probability(raw, h2.prototypes)


def fixed_refiner_probabilities(
    features: np.ndarray, model: FixedH2, batch_size: int
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    output = fixed_mlp_batched(model.refiner, features, batch_size)
    return (
        _refiner_probability(output.logits, model.source.prototypes),
        output.accumulator_saturations,
    )


def add_counts(destination: List[int], source: Tuple[int, ...]) -> None:
    if not destination:
        destination.extend([0] * len(source))
    for index, count in enumerate(source):
        destination[index] += int(count)


def probability_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    reference: np.ndarray,
) -> Dict[str, Any]:
    result = event_metrics(labels, probabilities, threshold)
    result["probability_mae_to_reference"] = float(
        np.mean(np.abs(probabilities.astype(np.float64) - reference.astype(np.float64)))
    )
    result["prediction_disagreement_to_reference"] = float(
        np.mean((probabilities >= threshold) != (reference >= threshold))
    )
    return result


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    device = choose_device(arguments.device)
    h2_path = arguments.h2_weight.resolve()
    s0p_path = arguments.s0p_weight.resolve()
    h2 = load_export(h2_path)
    s0p, s0p_config, s0p_metadata = load_proposal_snn_export(s0p_path)
    s0p = s0p.to(device).eval()
    time_steps = int(s0p_config["time_steps"])
    proposal_threshold = float(s0p_metadata["proposal_threshold"])
    paths = calibration_paths()
    if arguments.max_sequences:
        paths = paths[: arguments.max_sequences]
    widths = arguments.widths
    fixed_h2_models = {
        width: FixedH2(
            h2,
            FixedH2Config(
                accumulator_bits=width,
                scale_fraction_bits=arguments.scale_fraction_bits,
                logit_fraction_bits=arguments.logit_fraction_bits,
            ),
        )
        for width in widths
    }
    fixed_snn_models = {
        width: FixedProposalSNN.from_float(
            s0p,
            time_steps,
            FixedSNNConfig(
                accumulator_bits=width,
                membrane_bits=width,
                membrane_fraction_bits=arguments.membrane_fraction_bits,
            ),
        )
        for width in widths
    }
    labels_parts: List[np.ndarray] = []
    h2_reference_parts: List[np.ndarray] = []
    s0p_reference_parts: List[np.ndarray] = []
    variants: Dict[int, Dict[str, Any]] = {
        width: {
            "a1_probability": [],
            "s1_probability": [],
            "a1_proposal_saturations": [],
            "a1_refiner_saturations": [],
            "s1_accumulator_saturations": [0, 0, 0],
            "s1_membrane_saturations": [0, 0, 0, 0],
            "s1_refiner_saturations": [],
            "a1_candidates": 0,
            "a1_retained": 0,
            "s1_candidates": 0,
            "s1_retained": 0,
        }
        for width in widths
    }
    total_events = 0
    total_targets = 0
    h2_reference_candidates = 0
    h2_reference_retained = 0
    s0p_reference_candidates = 0
    s0p_reference_retained = 0
    per_sequence: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for number, path in enumerate(paths, start=1):
        sequence_started = time.perf_counter()
        events = load_events(path, require_labels=True)
        labels = events["label"].astype(np.uint8, copy=False)
        target_count = int(labels.sum())
        sequence = build_s0p_sequence(path, time_steps)
        patches = sequence.ann_features.shape[0] * sequence.ann_features.shape[1]
        ann_features = sequence.ann_features.reshape(patches, PROPOSAL_DIMENSIONS[0])
        temporal = torch.from_numpy(sequence.temporal_features).permute(1, 0, 2, 3)
        temporal = temporal.reshape(time_steps, patches, PROPOSAL_DIMENSIONS[0])
        temporal_numpy = temporal.numpy()
        windows = list(range(WINDOWS_PER_SEQUENCE))

        h2_reference = run_pipeline(events, h2)
        h2_probability = h2_reference["probabilities"]
        h2_candidates = h2_reference["candidate_indices"]
        s0p_float_heatmaps = proposal_heatmaps(
            float_s0p_logits(s0p, temporal, device, arguments.batch_size)
        )
        s0p_candidates = candidate_indices(
            events, windows, s0p_float_heatmaps, proposal_threshold
        )
        s0p_probability = np.zeros(len(events), dtype=np.float32)
        if len(s0p_candidates):
            features = build_refiner_features(
                events,
                s0p_candidates,
                windows,
                s0p_float_heatmaps,
                proposal_threshold,
            )
            s0p_probability[s0p_candidates] = float_refiner_probabilities(features, h2)

        sequence_record: Dict[str, Any] = {
            "file": path.name,
            "events": len(events),
            "target_events": target_count,
            "h2_reference_candidates": len(h2_candidates),
            "s0p_reference_candidates": len(s0p_candidates),
            "variants": {},
        }
        for width in widths:
            variant = variants[width]
            fixed_h2 = fixed_h2_models[width]
            fixed_snn = fixed_snn_models[width]

            a1_proposal = fixed_mlp_batched(
                fixed_h2.proposal, ann_features, arguments.batch_size
            )
            a1_heatmaps = proposal_heatmaps(a1_proposal.logits)
            a1_candidates = candidate_indices(
                events, windows, a1_heatmaps, h2.proposal_threshold
            )
            a1_probability = np.zeros(len(events), dtype=np.float32)
            a1_refiner_saturations: Tuple[int, ...] = (0, 0, 0)
            if len(a1_candidates):
                features = build_refiner_features(
                    events,
                    a1_candidates,
                    windows,
                    a1_heatmaps,
                    h2.proposal_threshold,
                )
                probability, a1_refiner_saturations = fixed_refiner_probabilities(
                    features, fixed_h2, arguments.batch_size
                )
                a1_probability[a1_candidates] = probability

            fixed_logits = []
            snn_accumulator_saturations = [0, 0, 0]
            snn_membrane_saturations = [0, 0, 0, 0]
            for start in range(0, patches, arguments.batch_size):
                output = fixed_snn.forward(
                    temporal_numpy[:, start : start + arguments.batch_size]
                )
                fixed_logits.append(output.logits)
                add_counts(snn_accumulator_saturations, output.accumulator_saturations)
                add_counts(snn_membrane_saturations, output.membrane_saturations)
            s1_heatmaps = proposal_heatmaps(np.concatenate(fixed_logits, axis=0))
            s1_candidates = candidate_indices(
                events, windows, s1_heatmaps, proposal_threshold
            )
            s1_probability = np.zeros(len(events), dtype=np.float32)
            s1_refiner_saturations: Tuple[int, ...] = (0, 0, 0)
            if len(s1_candidates):
                features = build_refiner_features(
                    events,
                    s1_candidates,
                    windows,
                    s1_heatmaps,
                    proposal_threshold,
                )
                probability, s1_refiner_saturations = fixed_refiner_probabilities(
                    features, fixed_h2, arguments.batch_size
                )
                s1_probability[s1_candidates] = probability

            variant["a1_probability"].append(a1_probability)
            variant["s1_probability"].append(s1_probability)
            add_counts(variant["a1_proposal_saturations"], a1_proposal.accumulator_saturations)
            add_counts(variant["a1_refiner_saturations"], a1_refiner_saturations)
            add_counts(variant["s1_accumulator_saturations"], tuple(snn_accumulator_saturations))
            add_counts(variant["s1_membrane_saturations"], tuple(snn_membrane_saturations))
            add_counts(variant["s1_refiner_saturations"], s1_refiner_saturations)
            variant["a1_candidates"] += len(a1_candidates)
            variant["a1_retained"] += int(labels[a1_candidates].sum())
            variant["s1_candidates"] += len(s1_candidates)
            variant["s1_retained"] += int(labels[s1_candidates].sum())
            sequence_record["variants"][str(width)] = {
                "a1_candidates": len(a1_candidates),
                "s1_candidates": len(s1_candidates),
                "a1_proposal_saturations": list(a1_proposal.accumulator_saturations),
                "s1_accumulator_saturations": snn_accumulator_saturations,
                "s1_membrane_saturations": snn_membrane_saturations,
            }

        labels_parts.append(labels)
        h2_reference_parts.append(h2_probability)
        s0p_reference_parts.append(s0p_probability)
        total_events += len(events)
        total_targets += target_count
        h2_reference_candidates += len(h2_candidates)
        h2_reference_retained += int(labels[h2_candidates].sum())
        s0p_reference_candidates += len(s0p_candidates)
        s0p_reference_retained += int(labels[s0p_candidates].sum())
        sequence_record["elapsed_s"] = time.perf_counter() - sequence_started
        per_sequence.append(sequence_record)
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"H2候选={len(h2_candidates)}, S0-P候选={len(s0p_candidates)}, "
            f"耗时={sequence_record['elapsed_s']:.2f}s",
            flush=True,
        )

    labels = np.concatenate(labels_parts)
    h2_reference_probability = np.concatenate(h2_reference_parts)
    s0p_reference_probability = np.concatenate(s0p_reference_parts)
    h2_reference_event = event_metrics(
        labels, h2_reference_probability, h2.event_threshold
    )
    s0p_reference_event = event_metrics(
        labels, s0p_reference_probability, h2.event_threshold
    )
    width_results: Dict[str, Any] = {}
    for width in widths:
        variant = variants[width]
        a1_probability = np.concatenate(variant.pop("a1_probability"))
        s1_probability = np.concatenate(variant.pop("s1_probability"))
        a1_event = probability_summary(
            labels,
            a1_probability,
            h2.event_threshold,
            h2_reference_probability,
        )
        s1_event = probability_summary(
            labels,
            s1_probability,
            h2.event_threshold,
            s0p_reference_probability,
        )
        a1_event["iou_delta_to_reference"] = a1_event["iou"] - h2_reference_event["iou"]
        s1_event["iou_delta_to_reference"] = s1_event["iou"] - s0p_reference_event["iou"]
        a1_saturation_total = sum(variant["a1_proposal_saturations"]) + sum(
            variant["a1_refiner_saturations"]
        )
        s1_saturation_total = (
            sum(variant["s1_accumulator_saturations"])
            + sum(variant["s1_membrane_saturations"])
            + sum(variant["s1_refiner_saturations"])
        )
        width_results[str(width)] = {
            "configuration": {
                "accumulator_bits": width,
                "membrane_bits": width,
                "membrane_fraction_bits": arguments.membrane_fraction_bits,
                "h2_scale_fraction_bits": arguments.scale_fraction_bits,
                "h2_logit_fraction_bits": arguments.logit_fraction_bits,
            },
            "a1_h2": {
                "event": a1_event,
                "candidate_events": variant["a1_candidates"],
                "candidate_fraction": variant["a1_candidates"] / total_events,
                "proposal_recall": variant["a1_retained"] / total_targets,
                "proposal_accumulator_saturations": variant[
                    "a1_proposal_saturations"
                ],
                "refiner_accumulator_saturations": variant[
                    "a1_refiner_saturations"
                ],
                "saturation_total": a1_saturation_total,
            },
            "s1_mixed": {
                "event": s1_event,
                "candidate_events": variant["s1_candidates"],
                "candidate_fraction": variant["s1_candidates"] / total_events,
                "proposal_recall": variant["s1_retained"] / total_targets,
                "proposal_accumulator_saturations": variant[
                    "s1_accumulator_saturations"
                ],
                "proposal_membrane_saturations": variant[
                    "s1_membrane_saturations"
                ],
                "refiner_accumulator_saturations": variant[
                    "s1_refiner_saturations"
                ],
                "saturation_total": s1_saturation_total,
            },
        }

    acceptable = [
        width
        for width in sorted(widths)
        if width_results[str(width)]["a1_h2"]["saturation_total"] == 0
        and width_results[str(width)]["s1_mixed"]["saturation_total"] == 0
        and width_results[str(width)]["a1_h2"]["event"]["iou_delta_to_reference"]
        >= -arguments.maximum_iou_loss
        and width_results[str(width)]["s1_mixed"]["event"]["iou_delta_to_reference"]
        >= -arguments.maximum_iou_loss
    ]
    selected_width = acceptable[0] if acceptable else None
    export_path: Path | None = None
    export_sha256: str | None = None
    if selected_width is not None and arguments.max_sequences == 0:
        candidate_export_path = arguments.export_dir / (
            f"S1P_INT8_ACC{selected_width}_MEM{selected_width}_Q"
            f"{arguments.membrane_fraction_bits}.npz"
        )
        metadata = {
            "model_id": candidate_export_path.stem,
            "source_s0p_weight": s0p_path.name,
            "source_s0p_weight_sha256": sha256_file(s0p_path),
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "calibration_role": ROLE,
            "calibration_result": arguments.output.name,
            "proposal_threshold": proposal_threshold,
            "event_threshold": h2.event_threshold,
            "h2_fixed_configuration": {
                "accumulator_bits": selected_width,
                "scale_fraction_bits": arguments.scale_fraction_bits,
                "logit_fraction_bits": arguments.logit_fraction_bits,
            },
            "selection_rule": {
                "maximum_iou_loss": arguments.maximum_iou_loss,
                "requires_zero_saturation": True,
                "select_smallest_width": True,
            },
        }
        export_fixed_proposal_snn(
            candidate_export_path, fixed_snn_models[selected_width], metadata
        )
        export_path = candidate_export_path
        export_sha256 = sha256_file(candidate_export_path)

    return {
        "status": "ok" if selected_width is not None else "gate_failed",
        "experiment_id": arguments.experiment_id,
        "scope": {
            "role": ROLE,
            "test_used": False,
            "validation_used": False,
            "training_performed": False,
            "complete_role": arguments.max_sequences == 0,
            "sequences": len(paths),
            "events": total_events,
            "target_events": total_targets,
            "device_for_float_reference": str(device),
            "fixed_models_run_on": "cpu_numpy_integer",
            "elapsed_s": time.perf_counter() - started,
        },
        "arithmetic_contract": {
            "weights": "int8；S1 逐输出通道选择二次幂缩放",
            "input": "uint8，S1 按 Q0.8 解释，因此 255 表示 255/256",
            "rounding": "最近舍入，正中间时远离零",
            "overflow": "点积结束后饱和；每次膜更新后饱和",
            "leak": "beta=7/8，整数乘 7 后右移 3 位",
            "reset": "达到阈值后减去阈值的软复位",
            "readout": "T=4 个读出膜电位的整数均值",
            "h2_scaling": "原 int8 权重不变，浮点通道缩放量化为定点整数乘数",
        },
        "inputs": {
            "dataset_index": INDEX_PATH.relative_to(ROOT).as_posix(),
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "h2_weight": h2_path.name,
            "h2_weight_sha256": sha256_file(h2_path),
            "s0p_weight": s0p_path.name,
            "s0p_weight_sha256": sha256_file(s0p_path),
            "s0p_configuration": s0p_config,
            "s0p_metadata": s0p_metadata,
        },
        "references": {
            "h2_float_scale": {
                "event": h2_reference_event,
                "candidate_events": h2_reference_candidates,
                "candidate_fraction": h2_reference_candidates / total_events,
                "proposal_recall": h2_reference_retained / total_targets,
            },
            "s0p_float_plus_h2": {
                "event": s0p_reference_event,
                "candidate_events": s0p_reference_candidates,
                "candidate_fraction": s0p_reference_candidates / total_events,
                "proposal_recall": s0p_reference_retained / total_targets,
            },
        },
        "width_results": width_results,
        "selection": {
            "maximum_iou_loss": arguments.maximum_iou_loss,
            "requires_zero_saturation": True,
            "acceptable_widths": acceptable,
            "selected_width": selected_width,
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


def parse_widths(value: str) -> List[int]:
    widths = []
    for item in value.split(","):
        width = int(item.strip())
        if width not in (16, 20, 24, 32):
            raise ValueError("位宽只允许 16、20、24、32")
        if width not in widths:
            widths.append(width)
    if not widths:
        raise ValueError("至少需要一个位宽")
    return widths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT_PATH)
    parser.add_argument("--s0p-weight", type=Path, default=S0P_WEIGHT_PATH)
    parser.add_argument("--widths", type=parse_widths, default=parse_widths("32,24,20,16"))
    parser.add_argument("--membrane-fraction-bits", type=int, default=12)
    parser.add_argument("--scale-fraction-bits", type=int, default=20)
    parser.add_argument("--logit-fraction-bits", type=int, default=16)
    parser.add_argument("--maximum-iou-loss", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--experiment-id", default="S1-FIXED-HW-CAL-v1")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "software" / "weights")
    arguments = parser.parse_args()
    if arguments.batch_size <= 0 or arguments.max_sequences < 0:
        raise ValueError("batch-size 必须为正数，max-sequences 不能为负数")
    if arguments.maximum_iou_loss < 0.0:
        raise ValueError("maximum-iou-loss 不能为负数")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = evaluate(arguments)
    atomic_json(result, arguments.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "references": result["references"],
                "width_results": result["width_results"],
                "selection": result["selection"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
