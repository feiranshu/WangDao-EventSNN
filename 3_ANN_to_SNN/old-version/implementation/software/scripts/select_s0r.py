"""在 Train-dev 上联合选择 S0-R checkpoint 与事件阈值。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.h2.numpy_model import load_export
from wangdao.h2.torch_model import prototype_logits
from wangdao.metrics import event_metrics
from wangdao.s0p import role_file_paths
from wangdao.s0r import build_s0r_windows
from wangdao.snn import RefinerSNN


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_dev() -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    h2 = load_export(H2_WEIGHT_PATH)
    feature_parts = []
    candidate_parts = []
    label_parts = []
    event_offset = 0
    retained_targets = 0
    started = time.perf_counter()
    paths = role_file_paths(INDEX_PATH, "train_dev")
    for number, path in enumerate(paths, start=1):
        events = load_events(path, require_labels=True)
        labels = events["label"].astype(np.uint8, copy=False)
        windows = build_s0r_windows(path, h2)
        for window in windows:
            feature_parts.append(window.features)
            candidate_parts.append(event_offset + window.event_indices)
            retained_targets += int(window.labels.sum())
        label_parts.append(labels)
        event_offset += len(events)
        print(
            f"[数据准备 {number:02d}/{len(paths):02d}] {path.name}，"
            f"累计事件 {event_offset}，耗时 {time.perf_counter() - started:.1f} 秒",
            flush=True,
        )
    features = np.ascontiguousarray(np.concatenate(feature_parts))
    candidates = np.ascontiguousarray(np.concatenate(candidate_parts), dtype=np.int64)
    labels = np.ascontiguousarray(np.concatenate(label_parts))
    if len(features) != len(candidates):
        raise AssertionError("S0-R features and candidate indices are not aligned")
    return features, candidates, labels, {
        "sequences": len(paths),
        "events": len(labels),
        "target_events": int(labels.sum()),
        "candidate_events": len(candidates),
        "retained_target_events": retained_targets,
    }


def load_student(
    path: Path, device: torch.device
) -> tuple[RefinerSNN, Dict[str, Any], int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"checkpoint 缺少模型参数：{path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint 缺少配置：{path}")
    arguments = config.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"checkpoint 缺少训练参数：{path}")
    model = RefinerSNN(
        beta=float(arguments["beta"]),
        threshold=float(arguments["threshold"]),
        surrogate_slope=float(arguments["surrogate_slope"]),
        readout=str(arguments["readout"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, int(checkpoint["epoch"])


def infer_candidates(
    model: RefinerSNN,
    features: np.ndarray,
    time_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, List[float]]:
    probability_parts = []
    spike_sum = np.zeros(3, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(
                device, non_blocking=True
            )
            temporal = batch.unsqueeze(0).expand(time_steps, -1, -1)
            output = model(temporal)
            probability = torch.softmax(
                prototype_logits(output.raw_logits), dim=1
            )[:, 1]
            probability_parts.append(probability.cpu().numpy())
            spike_sum += output.spike_rates.cpu().numpy() * len(batch)
    return (
        np.ascontiguousarray(np.concatenate(probability_parts)),
        (spike_sum / len(features)).tolist(),
    )


def threshold_scan(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum: float,
    maximum: float,
    step: float,
) -> Dict[str, Any]:
    thresholds = np.arange(minimum, maximum + 0.5 * step, step)
    records = [event_metrics(labels, probabilities, float(value)) for value in thresholds]
    best = max(records, key=lambda item: (item["iou"], item["f1"]))
    return {
        "best_threshold": best["threshold"],
        "best_event": best,
        "grid_min": minimum,
        "grid_max": maximum,
        "grid_step": step,
        "evaluated_thresholds": len(records),
    }


def run(arguments: argparse.Namespace) -> Dict[str, Any]:
    device = choose_device(arguments.device)
    run_directory = arguments.run_directory.resolve()
    checkpoints = sorted(run_directory.glob("epoch-*.pt"))
    if not checkpoints:
        raise FileNotFoundError("实验目录中没有 epoch checkpoint")
    features, candidates, labels, data_summary = prepare_dev()
    h2 = load_export(H2_WEIGHT_PATH)
    records = []
    started = time.perf_counter()
    reference_config: Dict[str, Any] | None = None
    for number, path in enumerate(checkpoints, start=1):
        model, config, epoch = load_student(path, device)
        if reference_config is None:
            reference_config = config
        elif config.get("arguments") != reference_config.get("arguments"):
            raise ValueError("checkpoint 训练参数不一致")
        config_arguments = config["arguments"]
        if not isinstance(config_arguments, dict):
            raise ValueError("checkpoint 缺少训练参数")
        candidate_probability, spike_rates = infer_candidates(
            model,
            features,
            int(config_arguments["time_steps"]),
            arguments.batch_size,
            device,
        )
        probabilities = np.zeros(len(labels), dtype=np.float32)
        probabilities[candidates] = candidate_probability
        scan = threshold_scan(
            labels,
            probabilities,
            arguments.threshold_min,
            arguments.threshold_max,
            arguments.threshold_step,
        )
        record = {
            "epoch": epoch,
            "checkpoint": path.name,
            "checkpoint_sha256": sha256_file(path),
            "fixed_event": event_metrics(labels, probabilities, h2.event_threshold),
            "threshold_scan": scan,
            "mean_spike_rates": spike_rates,
        }
        records.append(record)
        print(
            f"[{number:02d}/{len(checkpoints):02d}] epoch={epoch:02d}，"
            f"固定 IoU={record['fixed_event']['iou']:.6f}，"
            f"最优 IoU={scan['best_event']['iou']:.6f}，"
            f"阈值={scan['best_threshold']:.2f}",
            flush=True,
        )
        del model
    ranking = sorted(
        records,
        key=lambda item: (
            item["threshold_scan"]["best_event"]["iou"],
            item["threshold_scan"]["best_event"]["f1"],
        ),
        reverse=True,
    )
    selected = ranking[0]
    return {
        "status": "ok",
        "selection_metric": "Train-dev Event IoU with threshold scan",
        "test_used": False,
        "data": {
            **data_summary,
            "index": INDEX_PATH.relative_to(ROOT).as_posix(),
            "index_sha256": sha256_file(INDEX_PATH),
            "proposal_recall": data_summary["retained_target_events"]
            / data_summary["target_events"],
            "candidate_fraction": data_summary["candidate_events"]
            / data_summary["events"],
        },
        "run_directory": str(run_directory),
        "training_config": reference_config,
        "selected": selected,
        "ranking": ranking,
        "elapsed_s": time.perf_counter() - started,
        "selection_source_sha256": sha256_file(Path(__file__).resolve()),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not (
        0.0 <= arguments.threshold_min <= arguments.threshold_max <= 1.0
        and arguments.threshold_step > 0.0
    ):
        raise ValueError("invalid threshold scan range")
    return arguments


def main() -> None:
    if "--study" in sys.argv[1:]:
        from wangdao.refiner_study import select_main
        select_main([value for value in sys.argv[1:] if value != "--study"])
        return
    arguments = parse_arguments()
    result = run(arguments)
    atomic_json(result, arguments.output)
    print(
        json.dumps(
            {
                "selected": result["selected"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
