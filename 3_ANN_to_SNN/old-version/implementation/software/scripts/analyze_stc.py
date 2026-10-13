"""在 Train-dev 上诊断 Proposal 时空相关性，不更新模型权重。"""

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

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.h2.constants import PATCHES_PER_WINDOW, WINDOWS_PER_SEQUENCE
from wangdao.h2.representation import (
    build_temporal_patch_features,
    build_temporal_patch_targets,
    selected_windows,
)
from wangdao.s0p import role_file_paths
from wangdao.snn import load_proposal_snn_export
from wangdao.stc import (
    STC_KERNELS,
    choose_stc_diagnostic,
    diagnose_stc_heatmaps,
    patch_probabilities_to_heatmaps,
)


INDEX_PATH = ROOT / "data" / "index.json"
CHECKPOINT_PATH = ROOT / "software" / "weights" / "S0P_T4_POS15_KD50_S42.npz"
OUTPUT_PATH = ROOT / "results" / "stc_calibration.json"


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    if arguments.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if arguments.max_sequences < 0:
        raise ValueError("max-sequences cannot be negative")
    return arguments


def run(arguments: argparse.Namespace) -> Dict[str, Any]:
    device = choose_device(arguments.device)
    checkpoint_path = arguments.checkpoint.resolve()
    model, configuration, metadata = load_proposal_snn_export(checkpoint_path)
    time_steps = int(configuration["time_steps"])
    if time_steps != 4:
        raise ValueError("the bounded STC diagnostic is fixed to T=4")
    model.eval().to(device)
    paths = role_file_paths(INDEX_PATH, "train_dev")
    if arguments.max_sequences:
        paths = paths[: arguments.max_sequences]
    windows = selected_windows(0.0, 8000.0).tolist()
    probability_sequences: List[torch.Tensor] = []
    target_sequences: List[torch.Tensor] = []
    started = time.perf_counter()
    with torch.no_grad():
        for number, path in enumerate(paths, start=1):
            events = load_events(path, require_labels=True)
            temporal_features = build_temporal_patch_features(
                events, windows, time_steps
            )
            temporal_targets = build_temporal_patch_targets(
                events, windows, time_steps
            )
            features = torch.from_numpy(temporal_features).permute(1, 0, 2, 3)
            features = features.reshape(time_steps, -1, features.shape[-1])
            currents = []
            for start in range(0, features.shape[1], arguments.chunk_size):
                batch = features[:, start : start + arguments.chunk_size].to(
                    device, non_blocking=True
                )
                currents.append(model.forward_temporal(batch).step_currents.cpu())
            step_currents = torch.cat(currents, dim=1)
            patch_probability = torch.sigmoid(step_currents).reshape(
                time_steps,
                WINDOWS_PER_SEQUENCE,
                PATCHES_PER_WINDOW,
                -1,
            ).permute(1, 0, 2, 3)
            probability_sequences.append(
                patch_probabilities_to_heatmaps(patch_probability)
            )
            target_sequences.append(
                patch_probabilities_to_heatmaps(
                    torch.from_numpy(temporal_targets).to(torch.float32)
                )
            )
            print(
                f"[STC 诊断] {number}/{len(paths)} {path.name}，"
                f"累计 {time.perf_counter() - started:.1f} 秒",
                flush=True,
            )
    probabilities = torch.stack(probability_sequences, dim=0)
    targets = torch.stack(target_sequences, dim=0)
    candidates = [
        diagnose_stc_heatmaps(
            probabilities,
            targets,
            spatial_kernel=spatial,
            temporal_kernel=temporal,
        )
        for spatial, temporal in STC_KERNELS
    ]
    selected = choose_stc_diagnostic(candidates)
    ranked = sorted(
        candidates,
        key=lambda candidate: float(
            candidate["released"]["auroc_on_prediction_target_union"] or -1.0
        ),
        reverse=True,
    )
    best = ranked[0]
    proceed = selected is not None
    decision = "proceed_to_training" if proceed else "stop_stc"
    conclusion = (
        "Train-dev 时空相关性通过预设门槛，允许执行受限 STC 训练扫描。"
        if proceed
        else "Train-dev 时空相关性未通过全部预设门槛，STC 不进入训练，继续现有 S1→S3 主线。"
    )
    selected_kernel = selected["kernel"] if selected is not None else None
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment_id": "STC-DIAG-train-dev-v1",
        "stage": "S0-P-STC-diagnostic",
        "data_role": "train_dev",
        "primary_metrics": {
            "best_auroc": best["released"]["auroc_on_prediction_target_union"],
            "best_target_weight_median": best["released"]["target_weight_median"],
            "best_false_weight_median": best["released"]["false_weight_median"],
            "selected_kernel": selected_kernel,
        },
        "decision": decision,
        "conclusion": conclusion,
        "scope": {
            "test_used": False,
            "training_performed": False,
            "sequence_count": len(paths),
            "windows_per_sequence": WINDOWS_PER_SEQUENCE,
            "micro_time_steps": time_steps,
            "micro_step_ms": 25.0,
            "coarse_cell_pixels": 8,
        },
        "model": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metadata": metadata,
        },
        "protocol": {
            "paper_formula_gamma": 2.0,
            "training_variant": "released_centered_detached",
            "prediction_threshold": 0.5,
            "candidate_kernels": [
                {
                    "spatial": spatial,
                    "temporal_micro_steps": temporal,
                }
                for spatial, temporal in STC_KERNELS
            ],
            "gates": {
                "auroc_minimum": 0.60,
                "target_median_above_false": True,
                "maximum_saturated_fraction": 0.95,
                "probe_recall_must_improve": True,
                "maximum_probe_candidate_growth_points": 0.05,
            },
        },
        "candidates": candidates,
        "selected": selected,
        "reproducibility": {
            "command": sys.argv,
            "device": str(device),
            "torch": torch.__version__,
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "stc_source_sha256": sha256_file(
                ROOT / "software" / "wangdao" / "stc.py"
            ),
            "elapsed_s": time.perf_counter() - started,
        },
    }


def main() -> None:
    arguments = parse_arguments()
    result = run(arguments)
    atomic_json(result, arguments.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "primary_metrics": result["primary_metrics"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
