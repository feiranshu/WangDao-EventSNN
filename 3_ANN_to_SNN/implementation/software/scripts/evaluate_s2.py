"""固定配置比较 S1 与 S2-PoT；只开放 Validation。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np


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
    validation_paths,
)
from wangdao.data import load_events
from wangdao.fixed import (
    FixedH2,
    FixedH2Config,
    FixedProposalSNN,
    PoTProposalSNN,
    load_fixed_proposal_snn,
    load_pot_proposal_snn,
)
from wangdao.h2.constants import PROPOSAL_DIMENSIONS, WINDOWS_PER_SEQUENCE
from wangdao.h2.numpy_model import _refiner_probability, load_export
from wangdao.h2.representation import build_refiner_features, candidate_indices
from wangdao.metrics import event_metrics, localization_metrics
from wangdao.s0p import build_s0p_sequence


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
S1_WEIGHT_PATH = ROOT / "software" / "weights" / "S1P_INT8_ACC24_MEM24_Q12.npz"
S2_WEIGHT_PATH = (
    ROOT
    / "software"
    / "weights"
    / "S2P_POT_MIXED_EMIN4_EMAX3_ACC24_MEM24_Q12.npz"
)
OUTPUT_PATH = ROOT / "results" / "s2_validation.json"
ROLE = "validation"


def run_proposal(
    model: FixedProposalSNN,
    temporal: np.ndarray,
    batch_size: int,
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
        "proposal_recall": retained / int(labels.sum()) if labels.sum() else 1.0,
        "candidate_fraction": len(candidates) / len(events),
    }


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    h2_path = arguments.h2_weight.resolve()
    s1_path = arguments.s1_weight.resolve()
    s2_path = arguments.s2_weight.resolve()
    h2 = load_export(h2_path)
    s1, s1_configuration, s1_metadata = load_fixed_proposal_snn(s1_path)
    s2, s2_configuration, s2_metadata = load_pot_proposal_snn(s2_path)
    required = (
        "source_s1_weight_sha256",
        "h2_weight_sha256",
        "dataset_index_sha256",
        "proposal_threshold",
        "event_threshold",
        "h2_fixed_configuration",
        "selection_rule",
    )
    missing = [name for name in required if name not in s2_metadata]
    if missing:
        raise ValueError("S2 冻结权重缺少元数据：" + ", ".join(missing))
    if s2_metadata["source_s1_weight_sha256"] != sha256_file(s1_path):
        raise ValueError("S2 与当前 S1 权重不一致")
    if s2_metadata["h2_weight_sha256"] != sha256_file(h2_path):
        raise ValueError("S2 与当前 H2 权重不一致")
    if s2_metadata["dataset_index_sha256"] != sha256_file(INDEX_PATH):
        raise ValueError("S2 与当前数据索引不一致")
    h2_configuration = s2_metadata["h2_fixed_configuration"]
    if not isinstance(h2_configuration, dict):
        raise ValueError("S2 中的 H2 定点配置无效")
    fixed_h2 = FixedH2(
        h2,
        FixedH2Config(
            accumulator_bits=int(h2_configuration["accumulator_bits"]),
            scale_fraction_bits=int(h2_configuration["scale_fraction_bits"]),
            logit_fraction_bits=int(h2_configuration["logit_fraction_bits"]),
        ),
    )
    proposal_threshold = float(s2_metadata["proposal_threshold"])
    event_threshold = float(s2_metadata["event_threshold"])
    paths = validation_paths()
    labels_parts: List[np.ndarray] = []
    totals = {"s1": empty_totals(), "s2": empty_totals()}
    saturation = {
        "s1": {"proposal": [0, 0, 0], "membrane": [0, 0, 0, 0], "refiner": [0, 0, 0]},
        "s2": {"proposal": [0, 0, 0], "membrane": [0, 0, 0, 0], "refiner": [0, 0, 0]},
    }
    spikes = {"s1": [0, 0, 0], "s2": [0, 0, 0]}
    total_events = 0
    total_targets = 0
    total_patches = 0
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
        windows = list(range(WINDOWS_PER_SEQUENCE))
        record: Dict[str, Any] = {
            "file": path.name,
            "events": len(events),
            "target_events": int(labels.sum()),
        }
        for name, model in (("s1", s1), ("s2", s2)):
            logits, accumulator, membrane, layer_spikes = run_proposal(
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
            record[name] = update_totals(
                totals[name],
                events,
                labels,
                probability,
                candidates,
                event_threshold,
            )
            add_counts(saturation[name]["proposal"], tuple(accumulator))
            add_counts(saturation[name]["membrane"], tuple(membrane))
            add_counts(saturation[name]["refiner"], refiner_saturations)
            add_counts(spikes[name], tuple(layer_spikes))
        record["elapsed_s"] = time.perf_counter() - sequence_started
        per_sequence.append(record)
        labels_parts.append(labels)
        total_events += len(events)
        total_targets += int(labels.sum())
        total_patches += patches
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"S1 IoU={record['s1']['event']['iou']:.4f}, "
            f"S2 IoU={record['s2']['event']['iou']:.4f}, "
            f"耗时={record['elapsed_s']:.2f}s",
            flush=True,
        )

    labels = np.concatenate(labels_parts)
    evaluations: Dict[str, Any] = {}
    for name, model in (("s1", s1), ("s2", s2)):
        result = finalize(
            totals[name], labels, event_threshold, total_events, total_targets
        )
        denominators = [
            model.time_steps * total_patches * dimension
            for dimension in PROPOSAL_DIMENSIONS[1:]
        ]
        result["proposal_accumulator_saturations"] = saturation[name]["proposal"]
        result["proposal_membrane_saturations"] = saturation[name]["membrane"]
        result["refiner_accumulator_saturations"] = saturation[name]["refiner"]
        result["spike_counts"] = spikes[name]
        result["spike_rates"] = [
            count / denominator
            for count, denominator in zip(spikes[name], denominators)
        ]
        evaluations[name] = result
    evaluations["s2_minus_s1"] = {
        metric: evaluations["s2"]["event"][metric]
        - evaluations["s1"]["event"][metric]
        for metric in ("iou", "f1", "precision", "recall")
    }
    selection_rule = s2_metadata["selection_rule"]
    if not isinstance(selection_rule, dict):
        raise ValueError("S2 选择规则无效")
    maximum_iou_loss = float(selection_rule["maximum_iou_loss"])
    s2_saturation_total = sum(
        int(value)
        for field in (
            "proposal_accumulator_saturations",
            "proposal_membrane_saturations",
            "refiner_accumulator_saturations",
        )
        for value in evaluations["s2"][field]
    )
    iou_delta = evaluations["s2_minus_s1"]["iou"]
    passed = iou_delta >= -maximum_iou_loss and s2_saturation_total == 0
    decision = "retain_as_ablation" if passed else "reject"
    if passed:
        conclusion = (
            f"S2 Validation Event IoU 为 "
            f"{evaluations['s2']['event']['iou']:.4%}，相对 S1 变化 "
            f"{iou_delta * 100:+.4f} 个百分点，饱和 {s2_saturation_total} 次；"
            "通过精度门槛并保留为消融，S1 继续作为 FPGA 主线。"
        )
    else:
        conclusion = (
            f"S2 Validation Event IoU 为 "
            f"{evaluations['s2']['event']['iou']:.4%}，相对 S1 变化 "
            f"{iou_delta * 100:+.4f} 个百分点，饱和 {s2_saturation_total} 次；"
            "未通过冻结门槛，停止 S2。"
        )
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment_id": arguments.experiment_id,
        "stage": "S2",
        "data_role": ROLE,
        "primary_metrics": {
            "s1_event_iou": evaluations["s1"]["event"]["iou"],
            "s2_event_iou": evaluations["s2"]["event"]["iou"],
            "iou_delta_to_s1": iou_delta,
            "s1_event_f1": evaluations["s1"]["event"]["f1"],
            "s2_event_f1": evaluations["s2"]["event"]["f1"],
            "s1_proposal_recall": evaluations["s1"]["proposal_recall"],
            "s2_proposal_recall": evaluations["s2"]["proposal_recall"],
            "s1_candidate_fraction": evaluations["s1"]["candidate_fraction"],
            "s2_candidate_fraction": evaluations["s2"]["candidate_fraction"],
            "maximum_iou_loss": maximum_iou_loss,
            "s2_saturation_total": s2_saturation_total,
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
            "s2_weight": s2_path.name,
            "s2_weight_sha256": sha256_file(s2_path),
            "s2_configuration": s2_configuration,
            "s2_metadata": s2_metadata,
            "s2_weight_statistics": s2.weight_statistics(),
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
        },
        "evaluation": evaluations,
        "per_sequence": per_sequence,
        "reproducibility": {
            "command": [str(value) for value in sys.argv],
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "fixed_source_sha256": sha256_file(
                ROOT / "software" / "wangdao" / "fixed.py"
            ),
            "shared_evaluator_sha256": sha256_file(
                ROOT / "software" / "scripts" / "evaluate_s1.py"
            ),
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT_PATH)
    parser.add_argument("--s1-weight", type=Path, default=S1_WEIGHT_PATH)
    parser.add_argument("--s2-weight", type=Path, default=S2_WEIGHT_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--experiment-id", default="S2-POT-validation-v1")
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
                "evaluation": result["evaluation"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
