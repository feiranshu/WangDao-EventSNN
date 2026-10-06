"""固定配置评测 A1-H2 与 S1-P + A1-Refiner；只开放 Validation。"""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.fixed import FixedH2, FixedH2Config, FixedMLPOutput, load_fixed_proposal_snn
from wangdao.h2.constants import (
    PATCHES_PER_WINDOW,
    PROPOSAL_DIMENSIONS,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    WINDOWS_PER_SEQUENCE,
)
from wangdao.h2.numpy_model import _refiner_probability, load_export
from wangdao.h2.representation import (
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
)
from wangdao.metrics import event_metrics, localization_metrics
from wangdao.s0p import build_s0p_sequence


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
S1_WEIGHT_PATH = ROOT / "software" / "weights" / "S1P_INT8_ACC24_MEM24_Q12.npz"
OUTPUT_PATH = ROOT / "results" / "s1_validation.json"
ROLE = "validation"


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


def validation_paths() -> List[Path]:
    """只解析 Validation；该入口不接受可调角色参数，也不暴露 Test。"""
    index = load_json(INDEX_PATH)
    if index.get("status") != "ok":
        raise ValueError("数据索引状态无效")
    location = index.get("location")
    records = index.get("files")
    names = index.get("roles", {}).get(ROLE)
    if not isinstance(location, str) or not isinstance(records, list):
        raise ValueError("数据索引缺少位置或文件记录")
    if not isinstance(names, list) or not names:
        raise ValueError("Validation 数据角色为空")
    dataset_root = ROOT / location
    by_name = {
        Path(str(record["path"])).name: dataset_root / str(record["path"])
        for record in records
        if isinstance(record, dict) and "path" in record
    }
    if any(str(name) not in by_name for name in names):
        raise ValueError("Validation 引用了索引中不存在的文件")
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
    if not logits:
        raise ValueError("定点 MLP 不接受空输入")
    return FixedMLPOutput(
        logits_q=np.concatenate(logits_q, axis=0),
        logits=np.concatenate(logits, axis=0),
        accumulator_saturations=tuple(saturation_counts),
    )


def add_counts(destination: List[int], source: Tuple[int, ...]) -> None:
    for index, count in enumerate(source):
        destination[index] += int(count)


def empty_totals() -> Dict[str, Any]:
    return {
        "probabilities": [],
        "candidate_events": 0,
        "retained_targets": 0,
        "correct_targets": 0.0,
        "localization_targets": 0.0,
        "false_components": 0.0,
        "frame_denominator": 0.0,
    }


def finalize(
    totals: Dict[str, Any], labels: np.ndarray, threshold: float, total_events: int,
    total_targets: int,
) -> Dict[str, Any]:
    probabilities = np.concatenate(totals["probabilities"])
    denominator_pixels = totals["frame_denominator"] * SENSOR_WIDTH * SENSOR_HEIGHT
    return {
        "event": event_metrics(labels, probabilities, threshold),
        "pd": totals["correct_targets"] / totals["localization_targets"]
        if totals["localization_targets"]
        else 1.0,
        "fa": totals["false_components"] / denominator_pixels
        if denominator_pixels
        else 0.0,
        "candidate_events": totals["candidate_events"],
        "candidate_fraction": totals["candidate_events"] / total_events,
        "proposal_recall": totals["retained_targets"] / total_targets,
    }


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    h2_path = arguments.h2_weight.resolve()
    s1_path = arguments.s1_weight.resolve()
    h2 = load_export(h2_path)
    s1, s1_configuration, s1_metadata = load_fixed_proposal_snn(s1_path)
    required_metadata = (
        "source_s0p_weight_sha256",
        "dataset_index_sha256",
        "proposal_threshold",
        "event_threshold",
        "h2_fixed_configuration",
    )
    missing = [name for name in required_metadata if name not in s1_metadata]
    if missing:
        raise ValueError("S1 冻结权重缺少元数据：" + ", ".join(missing))
    if s1_metadata["dataset_index_sha256"] != sha256_file(INDEX_PATH):
        raise ValueError("S1 使用的数据索引与当前仓库不一致")
    h2_configuration = s1_metadata["h2_fixed_configuration"]
    if not isinstance(h2_configuration, dict):
        raise ValueError("S1 中的 H2 定点配置无效")
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
    paths = validation_paths()
    labels_parts: List[np.ndarray] = []
    a1_totals = empty_totals()
    s1_totals = empty_totals()
    a1_saturations = {"proposal": [0, 0, 0], "refiner": [0, 0, 0]}
    s1_saturations = {
        "proposal_accumulator": [0, 0, 0],
        "proposal_membrane": [0, 0, 0, 0],
        "refiner": [0, 0, 0],
    }
    s1_spikes = [0, 0, 0]
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
        ann_features = sequence.ann_features.reshape(patches, PROPOSAL_DIMENSIONS[0])
        temporal = sequence.temporal_features.transpose(1, 0, 2, 3).reshape(
            s1.time_steps, patches, PROPOSAL_DIMENSIONS[0]
        )
        windows = list(range(WINDOWS_PER_SEQUENCE))

        a1_proposal = fixed_mlp_batched(
            fixed_h2.proposal, ann_features, arguments.batch_size
        )
        add_counts(a1_saturations["proposal"], a1_proposal.accumulator_saturations)
        a1_heatmaps = proposal_heatmaps(a1_proposal.logits)
        a1_candidates = candidate_indices(
            events, windows, a1_heatmaps, h2.proposal_threshold
        )
        a1_probability = np.zeros(len(events), dtype=np.float32)
        if len(a1_candidates):
            features = build_refiner_features(
                events, a1_candidates, windows, a1_heatmaps, h2.proposal_threshold
            )
            refiner = fixed_mlp_batched(
                fixed_h2.refiner, features, arguments.batch_size
            )
            add_counts(a1_saturations["refiner"], refiner.accumulator_saturations)
            a1_probability[a1_candidates] = _refiner_probability(
                refiner.logits, h2.prototypes
            )

        s1_logits = []
        for start in range(0, patches, arguments.batch_size):
            output = s1.forward(temporal[:, start : start + arguments.batch_size])
            s1_logits.append(output.logits)
            add_counts(
                s1_saturations["proposal_accumulator"],
                output.accumulator_saturations,
            )
            add_counts(
                s1_saturations["proposal_membrane"], output.membrane_saturations
            )
            add_counts(s1_spikes, output.spike_counts)
        s1_heatmaps = proposal_heatmaps(np.concatenate(s1_logits, axis=0))
        s1_candidates = candidate_indices(
            events, windows, s1_heatmaps, proposal_threshold
        )
        s1_probability = np.zeros(len(events), dtype=np.float32)
        if len(s1_candidates):
            features = build_refiner_features(
                events, s1_candidates, windows, s1_heatmaps, proposal_threshold
            )
            refiner = fixed_mlp_batched(
                fixed_h2.refiner, features, arguments.batch_size
            )
            add_counts(s1_saturations["refiner"], refiner.accumulator_saturations)
            s1_probability[s1_candidates] = _refiner_probability(
                refiner.logits, h2.prototypes
            )

        sequence_record: Dict[str, Any] = {
            "file": path.name,
            "events": len(events),
            "target_events": target_count,
        }
        for name, probability, candidates, totals in (
            ("a1_h2", a1_probability, a1_candidates, a1_totals),
            ("s1_mixed", s1_probability, s1_candidates, s1_totals),
        ):
            event = event_metrics(labels, probability, event_threshold)
            localization = localization_metrics(
                events["t"],
                events["x"],
                events["y"],
                labels,
                events["name"],
                probability,
                threshold=event_threshold,
            )
            retained = int(labels[candidates].sum())
            totals["probabilities"].append(probability)
            totals["candidate_events"] += len(candidates)
            totals["retained_targets"] += retained
            totals["correct_targets"] += localization["correct_targets"]
            totals["localization_targets"] += localization["targets"]
            totals["false_components"] += localization["false_components"]
            totals["frame_denominator"] += localization["frame_denominator"]
            sequence_record[name] = {
                "event": event,
                "proposal_recall": retained / target_count if target_count else 1.0,
                "candidate_fraction": len(candidates) / len(events),
            }
        sequence_record["elapsed_s"] = time.perf_counter() - sequence_started
        per_sequence.append(sequence_record)
        labels_parts.append(labels)
        total_events += len(events)
        total_targets += target_count
        total_patches += patches
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"A1 IoU={sequence_record['a1_h2']['event']['iou']:.4f}, "
            f"S1 IoU={sequence_record['s1_mixed']['event']['iou']:.4f}, "
            f"耗时={sequence_record['elapsed_s']:.2f}s",
            flush=True,
        )

    labels = np.concatenate(labels_parts)
    a1_result = finalize(
        a1_totals, labels, event_threshold, total_events, total_targets
    )
    s1_result = finalize(
        s1_totals, labels, event_threshold, total_events, total_targets
    )
    spike_denominators = [
        s1.time_steps * total_patches * dimension
        for dimension in PROPOSAL_DIMENSIONS[1:]
    ]
    a1_result["accumulator_saturations"] = a1_saturations
    s1_result["accumulator_saturations"] = {
        "proposal": s1_saturations["proposal_accumulator"],
        "refiner": s1_saturations["refiner"],
    }
    s1_result["membrane_saturations"] = s1_saturations["proposal_membrane"]
    s1_result["spike_counts"] = s1_spikes
    s1_result["spike_rates"] = [
        count / denominator for count, denominator in zip(s1_spikes, spike_denominators)
    ]
    return {
        "status": "ok",
        "experiment_id": arguments.experiment_id,
        "scope": {
            "role": ROLE,
            "test_used": False,
            "threshold_scan": False,
            "training_performed": False,
            "sequences": len(paths),
            "events": total_events,
            "target_events": total_targets,
            "fixed_models_run_on": "cpu_numpy_integer",
            "elapsed_s": time.perf_counter() - started,
        },
        "model": {
            "h2_weight": h2_path.name,
            "h2_weight_sha256": sha256_file(h2_path),
            "s1_weight": s1_path.name,
            "s1_weight_sha256": sha256_file(s1_path),
            "s1_configuration": s1_configuration,
            "s1_metadata": s1_metadata,
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
        },
        "evaluation": {
            "a1_h2": a1_result,
            "s1_mixed": s1_result,
            "s1_minus_a1": {
                "iou": s1_result["event"]["iou"] - a1_result["event"]["iou"],
                "f1": s1_result["event"]["f1"] - a1_result["event"]["f1"],
                "precision": s1_result["event"]["precision"]
                - a1_result["event"]["precision"],
                "recall": s1_result["event"]["recall"]
                - a1_result["event"]["recall"],
            },
        },
        "per_sequence": per_sequence,
        "reproducibility": {
            "command": [str(value) for value in sys.argv],
            "dataset_index_sha256": sha256_file(INDEX_PATH),
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
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--experiment-id", default="S1-FIXED-validation-v1")
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
                "evaluation": result["evaluation"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
