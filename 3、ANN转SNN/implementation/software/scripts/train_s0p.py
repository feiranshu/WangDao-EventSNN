"""S0-P 训练入口；默认只执行训练前检查，不更新权重。"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.h2.torch_model import load_exported_models
from wangdao.h2.constants import PATCHES_PER_WINDOW
from wangdao.s0p import (
    S0PSTCSequence,
    S0PSequence,
    S0PSequenceDataset,
    build_s0p_stc_sequence,
    build_s0p_sequence,
    sample_patch_indices,
    s0p_loss,
)
from wangdao.snn import ProposalSNN
from wangdao.stc import proposal_stc_loss, stc_block_ranges


INDEX_PATH = ROOT / "data" / "index.json"
WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
RUN_ROOT = ROOT / "results" / "runs"
STC_DIAGNOSTIC_PATH = ROOT / "results" / "stc_calibration.json"
STC_WEIGHTS = (0.0, 0.05, 0.10, 0.20)
STC_POSITIVE_FRACTION = 0.15
STC_DISTILLATION_WEIGHT = 0.50


SWEEP_GRIDS = {
    "coarse": (
        ("POS10-KD25", 0.10, 0.25),
        ("POS15-KD25", 0.15, 0.25),
        ("POS15-KD50", 0.15, 0.50),
    ),
    "refine": (
        ("POS20-KD35", 0.20, 0.35),
        ("POS20-KD50", 0.20, 0.50),
        ("POS25-KD35", 0.25, 0.35),
        ("POS25-KD50", 0.25, 0.50),
    ),
}


@dataclass
class SweepCandidate:
    name: str
    positive_fraction: float
    distillation_weight: float
    student: ProposalSNN
    optimizer: torch.optim.Optimizer
    generator: torch.Generator
    best_dev_loss: float = float("inf")
    best_epoch: int = 0


@dataclass
class STCSweepCandidate:
    name: str
    stc_weight: float
    student: ProposalSNN
    optimizer: torch.optim.Optimizer
    best_dev_base_total: float = float("inf")
    best_epoch: int = 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 H2 派生的 S0-P；未指定 --train 时只做安全检查"
    )
    parser.add_argument("--train", action="store_true", help="明确开始更新模型权重")
    parser.add_argument(
        "--targeted-sweep",
        action="store_true",
        help="共享数据表征，同时训练多个候选采样/蒸馏方案",
    )
    parser.add_argument(
        "--stc-sweep",
        action="store_true",
        help="按 Train-dev 诊断结果训练 λ=0/0.05/0.10/0.20 四个 STC 候选",
    )
    parser.add_argument(
        "--stc-diagnostic",
        type=Path,
        default=STC_DIAGNOSTIC_PATH,
        help="完整 Train-dev STC 诊断 JSON",
    )
    parser.add_argument(
        "--stc-core-windows",
        type=int,
        default=4,
        help="每个连续训练核心块包含的 100 ms 窗口数",
    )
    parser.add_argument(
        "--sweep-stage",
        choices=tuple(SWEEP_GRIDS),
        default="refine",
        help="coarse 为上一轮粗扫，refine 为正样本比例与蒸馏权重联合细化",
    )
    parser.add_argument("--role", choices=("train_main", "train_dev"), default="train_main")
    parser.add_argument("--time-steps", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--readout", choices=ProposalSNN.READOUTS, default="mean_membrane"
    )
    parser.add_argument("--beta", type=float, default=0.875)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--distillation-weight", type=float, default=0.25)
    parser.add_argument("--spike-rate-weight", type=float, default=0.01)
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-dev-sequences", type=int, default=0)
    parser.add_argument(
        "--no-sequence-prefetch",
        action="store_true",
        help="关闭下一序列后台预取，仅用于等价性排查",
    )
    parser.add_argument(
        "--no-dev-sequence-cache",
        action="store_true",
        help="关闭跨 epoch 的 Train-dev 内存缓存",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5,
        help="每处理多少条序列显示一次进度；0 表示关闭",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.epochs <= 0:
        raise ValueError("epochs must be positive")
    if arguments.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if arguments.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= arguments.positive_fraction < 1.0:
        raise ValueError("positive-fraction must be in [0, 1)")
    if arguments.max_sequences < 0 or arguments.max_dev_sequences < 0:
        raise ValueError("sequence limits cannot be negative")
    if arguments.progress_interval < 0:
        raise ValueError("progress-interval cannot be negative")
    if arguments.targeted_sweep and arguments.stc_sweep:
        raise ValueError("targeted-sweep and stc-sweep cannot be enabled together")
    if arguments.stc_core_windows <= 0:
        raise ValueError("stc-core-windows must be positive")
    if arguments.stc_sweep and arguments.time_steps != 4:
        raise ValueError("the bounded STC sweep is fixed to T=4")
    if arguments.stc_sweep and not np.isclose(
        arguments.positive_fraction, STC_POSITIVE_FRACTION
    ):
        raise ValueError("the bounded STC sweep is fixed to positive-fraction=0.15")
    if arguments.stc_sweep and not np.isclose(
        arguments.distillation_weight, STC_DISTILLATION_WEIGHT
    ):
        raise ValueError("the bounded STC sweep is fixed to distillation-weight=0.50")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> Dict[str, str]:
    """Hash runnable sources, including files not yet committed to Git."""
    paths = sorted((ROOT / "software").rglob("*.py"))
    paths.extend([ROOT / "pyproject.toml", INDEX_PATH])
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(arguments: argparse.Namespace, device: torch.device) -> tuple[ProposalSNN, torch.nn.Module]:
    teacher, _ = load_exported_models(WEIGHT_PATH)
    student = ProposalSNN(
        beta=arguments.beta,
        threshold=arguments.threshold,
        surrogate_slope=arguments.surrogate_slope,
        readout=arguments.readout,
    )
    student.initialize_from_h2(teacher)
    teacher.set_quantized(True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return student.to(device), teacher.to(device)


def flatten_sequence(
    sequence: S0PSequence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporal = torch.from_numpy(sequence.temporal_features)
    temporal = temporal.permute(1, 0, 2, 3).reshape(
        temporal.shape[1], -1, temporal.shape[-1]
    )
    ann = torch.from_numpy(sequence.ann_features.reshape(-1, sequence.ann_features.shape[-1]))
    targets = torch.from_numpy(sequence.targets.reshape(-1, sequence.targets.shape[-1]))
    return temporal, ann, targets.to(torch.float32)


def compute_chunk(
    student: ProposalSNN,
    teacher: torch.nn.Module,
    temporal: torch.Tensor,
    ann: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    arguments: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    student_input = temporal[:, indices].to(device, non_blocking=True)
    teacher_input = ann[indices].to(device, non_blocking=True)
    target = targets[indices].to(device, non_blocking=True)
    output = student(student_input)
    with torch.no_grad():
        teacher_logits = teacher(teacher_input)
    return s0p_loss(
        output.logits,
        target,
        teacher_logits,
        output.spike_rates,
        distillation_weight=arguments.distillation_weight,
        spike_rate_weight=arguments.spike_rate_weight,
    ) | {"spike_rates": output.spike_rates}


def teacher_sequence_logits(
    teacher: torch.nn.Module,
    ann: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """一次计算整条序列的教师输出，供扫描候选共同使用。"""
    parts = []
    with torch.no_grad():
        for start in range(0, len(ann), chunk_size):
            parts.append(
                teacher(ann[start:start + chunk_size].to(device, non_blocking=True))
                .detach()
                .cpu()
            )
    return torch.cat(parts, dim=0)


def compute_sweep_chunk(
    candidate: SweepCandidate,
    temporal: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    arguments: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    output = candidate.student(
        temporal[:, indices].to(device, non_blocking=True)
    )
    losses = s0p_loss(
        output.logits,
        targets[indices].to(device, non_blocking=True),
        teacher_logits[indices].to(device, non_blocking=True),
        output.spike_rates,
        distillation_weight=candidate.distillation_weight,
        spike_rate_weight=arguments.spike_rate_weight,
    )
    return losses | {"spike_rates": output.spike_rates}


def add_metrics(
    totals: Dict[str, float], losses: Dict[str, torch.Tensor]
) -> None:
    for name, value in losses.items():
        if name == "spike_rates":
            for layer, rate in enumerate(value.detach().cpu().tolist()):
                key = f"spike_rate_{layer}"
                totals[key] = totals.get(key, 0.0) + float(rate)
        else:
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())


def preflight(
    dataset: S0PSequenceDataset,
    student: ProposalSNN,
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
) -> None:
    sequence = build_s0p_sequence(dataset.paths[0], dataset.time_steps, windows=[0])
    temporal, ann, targets = flatten_sequence(sequence)
    count = min(arguments.chunk_size, temporal.shape[1])
    indices = torch.arange(count)
    student.eval()
    with torch.no_grad():
        losses = compute_chunk(
            student, teacher, temporal, ann, targets, indices, device, arguments
        )
    rates = losses["spike_rates"].detach().cpu().tolist()
    positive_patches = torch.any(targets > 0.5, dim=1)
    print("训练前检查通过")
    print(f"设备：{device}")
    print(f"数据角色：{dataset.role}，序列数：{len(dataset)}")
    print(f"时间步：{dataset.time_steps}，检查窗口：1，patch：{count}")
    print(f"初始损失：{float(losses['total'].cpu()):.6f}")
    sampling_description = (
        "由下列定向扫描候选分别指定"
        if arguments.targeted_sweep
        else f"{arguments.positive_fraction:.1%}"
    )
    print(
        f"检查窗口自然正 patch：{int(positive_patches.sum())}/{len(targets)}；"
        f"训练采样目标：{sampling_description}"
    )
    print("各层发放率：" + ", ".join(f"{float(value):.6f}" for value in rates))
    print("未更新任何模型权重；添加 --train 才会开始训练")


def train_epoch(
    dataset: S0PSequenceDataset,
    student: ProposalSNN,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    arguments: argparse.Namespace,
    generator: torch.Generator,
) -> Dict[str, float]:
    student.train()
    sequence_count = len(dataset)
    if arguments.max_sequences:
        sequence_count = min(sequence_count, arguments.max_sequences)
    order = torch.randperm(len(dataset), generator=generator)[:sequence_count].tolist()
    totals: Dict[str, float] = {}
    chunks = 0
    started = time.perf_counter()
    for position, sequence_index in enumerate(order, start=1):
        temporal, ann, targets = flatten_sequence(dataset[sequence_index])
        permutation = sample_patch_indices(
            targets, arguments.positive_fraction, generator
        )
        for start in range(0, len(permutation), arguments.chunk_size):
            indices = permutation[start:start + arguments.chunk_size]
            optimizer.zero_grad(set_to_none=True)
            losses = compute_chunk(
                student, teacher, temporal, ann, targets, indices, device, arguments
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            for name, value in losses.items():
                if name == "spike_rates":
                    for layer, rate in enumerate(value.detach().cpu().tolist()):
                        totals[f"spike_rate_{layer}"] = totals.get(
                            f"spike_rate_{layer}", 0.0
                        ) + float(rate)
                else:
                    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            chunks += 1
        if arguments.progress_interval and (
            position % arguments.progress_interval == 0 or position == sequence_count
        ):
            print(
                f"[训练] 序列 {position}/{sequence_count}，"
                f"累计 {time.perf_counter() - started:.1f} 秒",
                flush=True,
            )
    return {name: value / max(chunks, 1) for name, value in totals.items()}


def evaluate_epoch(
    dataset: S0PSequenceDataset,
    student: ProposalSNN,
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
) -> Dict[str, float]:
    student.eval()
    sequence_count = len(dataset)
    if arguments.max_dev_sequences:
        sequence_count = min(sequence_count, arguments.max_dev_sequences)
    totals: Dict[str, float] = {}
    chunks = 0
    started = time.perf_counter()
    with torch.no_grad():
        for sequence_index in range(sequence_count):
            temporal, ann, targets = flatten_sequence(dataset[sequence_index])
            for start in range(0, temporal.shape[1], arguments.chunk_size):
                indices = torch.arange(start, min(start + arguments.chunk_size, temporal.shape[1]))
                losses = compute_chunk(
                    student, teacher, temporal, ann, targets, indices, device, arguments
                )
                for name, value in losses.items():
                    if name == "spike_rates":
                        for layer, rate in enumerate(value.detach().cpu().tolist()):
                            totals[f"spike_rate_{layer}"] = totals.get(
                                f"spike_rate_{layer}", 0.0
                            ) + float(rate)
                    else:
                        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
                chunks += 1
            position = sequence_index + 1
            if arguments.progress_interval and (
                position % arguments.progress_interval == 0
                or position == sequence_count
            ):
                print(
                    f"[开发] 序列 {position}/{sequence_count}，"
                    f"累计 {time.perf_counter() - started:.1f} 秒",
                    flush=True,
                )
    return {name: value / max(chunks, 1) for name, value in totals.items()}


def build_sweep_candidates(
    initial_student: ProposalSNN,
    device: torch.device,
    arguments: argparse.Namespace,
) -> List[SweepCandidate]:
    candidates = []
    for index, (name, positive_fraction, distillation_weight) in enumerate(
        SWEEP_GRIDS[arguments.sweep_stage]
    ):
        student = initial_student if index == 0 else copy.deepcopy(initial_student)
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=arguments.learning_rate
        )
        candidates.append(
            SweepCandidate(
                name=name,
                positive_fraction=positive_fraction,
                distillation_weight=distillation_weight,
                student=student.to(device),
                optimizer=optimizer,
                generator=torch.Generator().manual_seed(
                    arguments.seed + 1000 * (index + 1)
                ),
            )
        )
    return candidates


def train_sweep_epoch(
    dataset: S0PSequenceDataset,
    candidates: Sequence[SweepCandidate],
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
    order_generator: torch.Generator,
) -> Dict[str, Dict[str, float]]:
    for candidate in candidates:
        candidate.student.train()
    sequence_count = len(dataset)
    if arguments.max_sequences:
        sequence_count = min(sequence_count, arguments.max_sequences)
    order = torch.randperm(len(dataset), generator=order_generator)[
        :sequence_count
    ].tolist()
    totals: Dict[str, Dict[str, float]] = {
        candidate.name: {} for candidate in candidates
    }
    chunks = {candidate.name: 0 for candidate in candidates}
    started = time.perf_counter()
    for position, sequence_index in enumerate(order, start=1):
        temporal, ann, targets = flatten_sequence(dataset[sequence_index])
        teacher_logits = teacher_sequence_logits(
            teacher, ann, arguments.chunk_size, device
        )
        for candidate in candidates:
            permutation = sample_patch_indices(
                targets, candidate.positive_fraction, candidate.generator
            )
            for start in range(0, len(permutation), arguments.chunk_size):
                indices = permutation[start:start + arguments.chunk_size]
                candidate.optimizer.zero_grad(set_to_none=True)
                losses = compute_sweep_chunk(
                    candidate,
                    temporal,
                    targets,
                    teacher_logits,
                    indices,
                    device,
                    arguments,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    candidate.student.parameters(), max_norm=5.0
                )
                candidate.optimizer.step()
                add_metrics(totals[candidate.name], losses)
                chunks[candidate.name] += 1
        if arguments.progress_interval and (
            position % arguments.progress_interval == 0
            or position == sequence_count
        ):
            print(
                f"[扫描训练] 序列 {position}/{sequence_count}，"
                f"累计 {time.perf_counter() - started:.1f} 秒",
                flush=True,
            )
    return {
        candidate.name: {
            name: value / max(chunks[candidate.name], 1)
            for name, value in totals[candidate.name].items()
        }
        for candidate in candidates
    }


def evaluate_sweep_epoch(
    dataset: S0PSequenceDataset,
    candidates: Sequence[SweepCandidate],
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    for candidate in candidates:
        candidate.student.eval()
    sequence_count = len(dataset)
    if arguments.max_dev_sequences:
        sequence_count = min(sequence_count, arguments.max_dev_sequences)
    totals: Dict[str, Dict[str, float]] = {
        candidate.name: {} for candidate in candidates
    }
    chunks = {candidate.name: 0 for candidate in candidates}
    started = time.perf_counter()
    with torch.no_grad():
        for sequence_index in range(sequence_count):
            temporal, ann, targets = flatten_sequence(dataset[sequence_index])
            teacher_logits = teacher_sequence_logits(
                teacher, ann, arguments.chunk_size, device
            )
            for start in range(0, temporal.shape[1], arguments.chunk_size):
                indices = torch.arange(
                    start, min(start + arguments.chunk_size, temporal.shape[1])
                )
                for candidate in candidates:
                    losses = compute_sweep_chunk(
                        candidate,
                        temporal,
                        targets,
                        teacher_logits,
                        indices,
                        device,
                        arguments,
                    )
                    add_metrics(totals[candidate.name], losses)
                    chunks[candidate.name] += 1
            position = sequence_index + 1
            if arguments.progress_interval and (
                position % arguments.progress_interval == 0
                or position == sequence_count
            ):
                print(
                    f"[扫描开发] 序列 {position}/{sequence_count}，"
                    f"累计 {time.perf_counter() - started:.1f} 秒",
                    flush=True,
                )
    return {
        candidate.name: {
            name: value / max(chunks[candidate.name], 1)
            for name, value in totals[candidate.name].items()
        }
        for candidate in candidates
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def train(
    dataset: S0PSequenceDataset,
    dev_dataset: S0PSequenceDataset,
    student: ProposalSNN,
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
) -> None:
    output = arguments.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = RUN_ROOT / f"S0P-T{arguments.time_steps}-{arguments.readout}-S{arguments.seed}-{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "role": arguments.role,
        "time_steps": arguments.time_steps,
        "readout": arguments.readout,
        "beta": arguments.beta,
        "threshold": arguments.threshold,
        "surrogate_slope": arguments.surrogate_slope,
        "epochs": arguments.epochs,
        "chunk_size": arguments.chunk_size,
        "learning_rate": arguments.learning_rate,
        "distillation_weight": arguments.distillation_weight,
        "spike_rate_weight": arguments.spike_rate_weight,
        "positive_fraction": arguments.positive_fraction,
        "seed": arguments.seed,
        "max_sequences": arguments.max_sequences,
        "max_dev_sequences": arguments.max_dev_sequences,
        "progress_interval": arguments.progress_interval,
        "device": str(device),
        "h2_weight_sha256": sha256_file(WEIGHT_PATH),
        "dataset_index_sha256": sha256_file(INDEX_PATH),
        "command": sys.argv,
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "source_manifest": source_manifest(),
    }
    write_json(output / "config.json", configuration)
    optimizer = torch.optim.AdamW(student.parameters(), lr=arguments.learning_rate)
    generator = torch.Generator().manual_seed(arguments.seed)
    history: List[Dict[str, Any]] = []
    best_dev_loss = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        train_metrics = train_epoch(
            dataset, student, teacher, optimizer, device, arguments, generator
        )
        dev_metrics = evaluate_epoch(
            dev_dataset, student, teacher, device, arguments
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"dev_{name}": value for name, value in dev_metrics.items()},
        }
        history.append(record)
        write_json(output / "history.json", history)
        torch.save(
            {
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": configuration,
            },
            output / "last.pt",
        )
        if dev_metrics.get("total", float("inf")) < best_dev_loss:
            best_dev_loss = dev_metrics["total"]
            torch.save(
                {
                    "model": student.state_dict(),
                    "epoch": epoch,
                    "dev_loss": best_dev_loss,
                    "config": configuration,
                },
                output / "best.pt",
            )
        print(json.dumps(record, ensure_ascii=False))


def train_targeted_sweep(
    dataset: S0PSequenceDataset,
    dev_dataset: S0PSequenceDataset,
    initial_student: ProposalSNN,
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
) -> None:
    output = arguments.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stage = str(arguments.sweep_stage).upper()
        output = RUN_ROOT / f"S0P-{stage}-T{arguments.time_steps}-S{arguments.seed}-{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates = build_sweep_candidates(initial_student, device, arguments)
    configuration = {
        "mode": "targeted_sweep",
        "sweep_stage": arguments.sweep_stage,
        "role": arguments.role,
        "time_steps": arguments.time_steps,
        "readout": arguments.readout,
        "beta": arguments.beta,
        "threshold": arguments.threshold,
        "surrogate_slope": arguments.surrogate_slope,
        "epochs": arguments.epochs,
        "chunk_size": arguments.chunk_size,
        "learning_rate": arguments.learning_rate,
        "spike_rate_weight": arguments.spike_rate_weight,
        "seed": arguments.seed,
        "max_sequences": arguments.max_sequences,
        "max_dev_sequences": arguments.max_dev_sequences,
        "progress_interval": arguments.progress_interval,
        "device": str(device),
        "candidates": [
            {
                "name": candidate.name,
                "positive_fraction": candidate.positive_fraction,
                "distillation_weight": candidate.distillation_weight,
                "sampling_seed": arguments.seed + 1000 * (index + 1),
            }
            for index, candidate in enumerate(candidates)
        ],
        "h2_weight_sha256": sha256_file(WEIGHT_PATH),
        "dataset_index_sha256": sha256_file(INDEX_PATH),
        "command": sys.argv,
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "source_manifest": source_manifest(),
    }
    write_json(output / "config.json", configuration)
    history: List[Dict[str, Any]] = []
    order_generator = torch.Generator().manual_seed(arguments.seed)
    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        train_metrics = train_sweep_epoch(
            dataset,
            candidates,
            teacher,
            device,
            arguments,
            order_generator,
        )
        dev_metrics = evaluate_sweep_epoch(
            dev_dataset, candidates, teacher, device, arguments
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "candidates": {},
        }
        for candidate in candidates:
            candidate_configuration = {
                **configuration,
                "candidate": {
                    "name": candidate.name,
                    "positive_fraction": candidate.positive_fraction,
                    "distillation_weight": candidate.distillation_weight,
                },
            }
            metrics = {
                **{
                    f"train_{name}": value
                    for name, value in train_metrics[candidate.name].items()
                },
                **{
                    f"dev_{name}": value
                    for name, value in dev_metrics[candidate.name].items()
                },
            }
            record["candidates"][candidate.name] = metrics
            torch.save(
                {
                    "model": candidate.student.state_dict(),
                    "optimizer": candidate.optimizer.state_dict(),
                    "epoch": epoch,
                    "config": candidate_configuration,
                },
                output / f"{candidate.name}-last.pt",
            )
            current_dev_loss = dev_metrics[candidate.name].get(
                "total", float("inf")
            )
            if current_dev_loss < candidate.best_dev_loss:
                candidate.best_dev_loss = current_dev_loss
                candidate.best_epoch = epoch
                torch.save(
                    {
                        "model": candidate.student.state_dict(),
                        "epoch": epoch,
                        "dev_loss": current_dev_loss,
                        "config": candidate_configuration,
                    },
                    output / f"{candidate.name}-best.pt",
                )
        history.append(record)
        write_json(output / "history.json", history)
        ranking = sorted(candidates, key=lambda item: item.best_dev_loss)
        selection = {
            "selection_metric": "dev_total",
            "selected": ranking[0].name,
            "candidates": [
                {
                    "name": candidate.name,
                    "best_epoch": candidate.best_epoch,
                    "best_dev_total": candidate.best_dev_loss,
                    "checkpoint": f"{candidate.name}-best.pt",
                }
                for candidate in ranking
            ],
            "note": "训练损失仅作初选，最终候选需按 Train-dev 端到端 Event IoU 复核。",
        }
        write_json(output / "selection.json", selection)
        print(json.dumps(record, ensure_ascii=False))
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def load_stc_protocol(path: Path) -> Dict[str, Any]:
    diagnostic_path = path.resolve()
    with diagnostic_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("STC diagnostic must be a JSON object")
    if value.get("experiment_id") == "S0P-STC-CALIBRATION-v1":
        value = value.get("diagnostic")
        if not isinstance(value, dict):
            raise ValueError("finalized STC calibration is missing its diagnostic")
    if value.get("data_role") != "train_dev":
        raise ValueError("STC diagnostic must use Train-dev")
    scope = value.get("scope")
    if not isinstance(scope, dict) or int(scope.get("sequence_count", 0)) != 10:
        raise ValueError("STC training requires the complete 10-sequence diagnostic")
    if value.get("decision") != "proceed_to_training":
        raise ValueError("STC diagnostic did not authorize training")
    selected = value.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("kernel"), dict):
        raise ValueError("STC diagnostic is missing its selected kernel")
    kernel = selected["kernel"]
    spatial = int(kernel.get("spatial", 0))
    temporal = int(kernel.get("temporal_micro_steps", 0))
    if (spatial, temporal) not in ((1, 3), (3, 3), (3, 5)):
        raise ValueError("STC diagnostic selected an unsupported kernel")
    return {
        "path": str(diagnostic_path),
        "sha256": sha256_file(diagnostic_path),
        "experiment_id": value.get("experiment_id"),
        "spatial_kernel": spatial,
        "temporal_kernel": temporal,
    }


def build_stc_candidates(
    initial_student: ProposalSNN,
    device: torch.device,
    arguments: argparse.Namespace,
) -> List[STCSweepCandidate]:
    candidates = []
    for index, weight in enumerate(STC_WEIGHTS):
        student = initial_student if index == 0 else copy.deepcopy(initial_student)
        student = student.to(device)
        candidates.append(
            STCSweepCandidate(
                name=f"STC{int(round(weight * 100)):02d}",
                stc_weight=weight,
                student=student,
                optimizer=torch.optim.AdamW(
                    student.parameters(), lr=arguments.learning_rate
                ),
            )
        )
    return candidates


def _add_stc_metrics(
    totals: Dict[str, float],
    base: Dict[str, torch.Tensor],
    stc: Dict[str, torch.Tensor],
    total: torch.Tensor,
    spike_rates: torch.Tensor,
) -> None:
    totals["total"] = totals.get("total", 0.0) + float(total.detach().cpu())
    for name, value in base.items():
        totals[f"base_{name}"] = totals.get(f"base_{name}", 0.0) + float(
            value.detach().cpu()
        )
    for name, value in stc.items():
        totals[f"stc_{name}"] = totals.get(f"stc_{name}", 0.0) + float(
            value.detach().cpu()
        )
    for layer, rate in enumerate(spike_rates.detach().cpu().tolist()):
        key = f"spike_rate_{layer}"
        totals[key] = totals.get(key, 0.0) + float(rate)


def run_stc_epoch(
    dataset: S0PSequenceDataset,
    candidates: Sequence[STCSweepCandidate],
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
    protocol: Dict[str, Any],
    training: bool,
    generator: torch.Generator,
    sequence_cache: Dict[int, S0PSTCSequence] | None = None,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    for candidate in candidates:
        candidate.student.train(training)
    sequence_count = len(dataset)
    limit = arguments.max_sequences if training else arguments.max_dev_sequences
    if limit:
        sequence_count = min(sequence_count, limit)
    if training:
        order = torch.randperm(len(dataset), generator=generator)[
            :sequence_count
        ].tolist()
    else:
        order = list(range(sequence_count))
    totals: Dict[str, Dict[str, float]] = {
        candidate.name: {} for candidate in candidates
    }
    block_counts = {candidate.name: 0 for candidate in candidates}
    started = time.perf_counter()
    data_wait_seconds = 0.0
    cache_hits = 0
    cache_misses = 0

    def load_sequence(index: int) -> tuple[int, S0PSTCSequence, bool]:
        cached = sequence_cache is not None and index in sequence_cache
        if cached:
            if sequence_cache is None:
                raise AssertionError("sequence cache state is inconsistent")
            return index, sequence_cache[index], True
        sequence = build_s0p_stc_sequence(
            dataset.paths[index], arguments.time_steps
        )
        if sequence_cache is not None:
            sequence_cache[index] = sequence
        return index, sequence, False

    executor = (
        None
        if arguments.no_sequence_prefetch
        else ThreadPoolExecutor(max_workers=1, thread_name_prefix="s0p-stc-prefetch")
    )
    pending: Future[tuple[int, S0PSTCSequence, bool]] | None = None
    if executor is not None and order:
        pending = executor.submit(load_sequence, order[0])
    try:
        for position, sequence_index in enumerate(order, start=1):
            wait_started = time.perf_counter()
            if pending is None:
                loaded_index, sequence, cache_hit = load_sequence(sequence_index)
            else:
                loaded_index, sequence, cache_hit = pending.result()
            data_wait_seconds += time.perf_counter() - wait_started
            if loaded_index != sequence_index:
                raise AssertionError("prefetched sequence order changed")
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            if executor is not None and position < sequence_count:
                pending = executor.submit(load_sequence, order[position])
            else:
                pending = None

            for load_start, load_stop, core_start, core_stop in stc_block_ranges(
                len(sequence.targets), arguments.stc_core_windows
            ):
                temporal = torch.from_numpy(
                    sequence.temporal_features[load_start:load_stop]
                ).permute(1, 0, 2, 3)
                temporal = temporal.reshape(
                    temporal.shape[0], -1, temporal.shape[-1]
                ).to(device, non_blocking=True)
                ann = torch.from_numpy(
                    sequence.ann_features[load_start:load_stop].reshape(
                        -1, sequence.ann_features.shape[-1]
                    )
                ).to(device, non_blocking=True)
                targets = torch.from_numpy(
                    sequence.targets[load_start:load_stop].reshape(
                        -1, sequence.targets.shape[-1]
                    )
                ).to(device, dtype=torch.float32, non_blocking=True)
                temporal_targets = torch.from_numpy(
                    sequence.temporal_targets[load_start:load_stop]
                ).to(device, dtype=torch.float32, non_blocking=True)
                core_indices = torch.arange(
                    core_start * PATCHES_PER_WINDOW,
                    core_stop * PATCHES_PER_WINDOW,
                    dtype=torch.int64,
                )
                sampled = sample_patch_indices(
                    targets[core_indices.to(device)].detach().cpu(),
                    STC_POSITIVE_FRACTION,
                    generator,
                )
                selected = core_indices[sampled].to(device)
                with torch.no_grad():
                    teacher_logits = teacher(ann)
                for candidate in candidates:
                    if training:
                        candidate.optimizer.zero_grad(set_to_none=True)
                    context = torch.enable_grad() if training else torch.no_grad()
                    with context:
                        temporal_output = candidate.student.forward_temporal(temporal)
                        base = s0p_loss(
                            temporal_output.output.logits[selected],
                            targets[selected],
                            teacher_logits[selected],
                            temporal_output.output.spike_rates,
                            distillation_weight=STC_DISTILLATION_WEIGHT,
                            spike_rate_weight=arguments.spike_rate_weight,
                        )
                        stc = proposal_stc_loss(
                            temporal_output.step_currents,
                            temporal_targets,
                            spatial_kernel=int(protocol["spatial_kernel"]),
                            temporal_kernel=int(protocol["temporal_kernel"]),
                            core_start=core_start,
                            core_stop=core_stop,
                        )
                        total = base["total"] + candidate.stc_weight * stc["loss"]
                        if training:
                            total.backward()
                            torch.nn.utils.clip_grad_norm_(
                                candidate.student.parameters(), max_norm=5.0
                            )
                            candidate.optimizer.step()
                    _add_stc_metrics(
                        totals[candidate.name],
                        base,
                        stc,
                        total,
                        temporal_output.output.spike_rates,
                    )
                    block_counts[candidate.name] += 1
            if arguments.progress_interval and (
                position % arguments.progress_interval == 0
                or position == sequence_count
            ):
                phase = "STC 训练" if training else "STC 开发"
                print(
                    f"[{phase}] 序列 {position}/{sequence_count}，"
                    f"累计 {time.perf_counter() - started:.1f} 秒",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed = time.perf_counter() - started
    cached_bytes = 0
    if sequence_cache is not None:
        cached_bytes = sum(
            sequence.temporal_features.nbytes
            + sequence.ann_features.nbytes
            + sequence.targets.nbytes
            + sequence.temporal_targets.nbytes
            for sequence in sequence_cache.values()
        )
    metrics = {
        candidate.name: {
            name: value / max(block_counts[candidate.name], 1)
            for name, value in totals[candidate.name].items()
        }
        for candidate in candidates
    }
    performance = {
        "elapsed_s": elapsed,
        "data_wait_s": data_wait_seconds,
        "data_wait_fraction": data_wait_seconds / elapsed if elapsed else 0.0,
        "sequence_prefetch": executor is not None,
        "sequence_cache": sequence_cache is not None,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cached_sequences": len(sequence_cache) if sequence_cache is not None else 0,
        "cached_mib": cached_bytes / (1024.0 * 1024.0),
    }
    return metrics, performance


def train_stc_sweep(
    dataset: S0PSequenceDataset,
    dev_dataset: S0PSequenceDataset,
    initial_student: ProposalSNN,
    teacher: torch.nn.Module,
    device: torch.device,
    arguments: argparse.Namespace,
    protocol: Dict[str, Any],
) -> None:
    output = arguments.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = RUN_ROOT / f"S0P-STC-T4-S{arguments.seed}-{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates = build_stc_candidates(initial_student, device, arguments)
    configuration = {
        "mode": "stc_sweep",
        "role": arguments.role,
        "time_steps": arguments.time_steps,
        "readout": arguments.readout,
        "beta": arguments.beta,
        "threshold": arguments.threshold,
        "surrogate_slope": arguments.surrogate_slope,
        "epochs": arguments.epochs,
        "core_windows": arguments.stc_core_windows,
        "halo_windows": 1,
        "learning_rate": arguments.learning_rate,
        "distillation_weight": STC_DISTILLATION_WEIGHT,
        "spike_rate_weight": arguments.spike_rate_weight,
        "positive_fraction": STC_POSITIVE_FRACTION,
        "stc_weights": list(STC_WEIGHTS),
        "stc_variant": "released_centered_detached",
        "stc_spatial_kernel": protocol["spatial_kernel"],
        "stc_temporal_kernel": protocol["temporal_kernel"],
        "stc_diagnostic": protocol,
        "seed": arguments.seed,
        "max_sequences": arguments.max_sequences,
        "max_dev_sequences": arguments.max_dev_sequences,
        "progress_interval": arguments.progress_interval,
        "sequence_prefetch": not arguments.no_sequence_prefetch,
        "dev_sequence_cache": not arguments.no_dev_sequence_cache,
        "device": str(device),
        "h2_weight_sha256": sha256_file(WEIGHT_PATH),
        "dataset_index_sha256": sha256_file(INDEX_PATH),
        "command": sys.argv,
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "source_manifest": source_manifest(),
    }
    write_json(output / "config.json", configuration)
    history: List[Dict[str, Any]] = []
    training_generator = torch.Generator().manual_seed(arguments.seed)
    dev_sequence_cache: Dict[int, S0PSTCSequence] | None = (
        None if arguments.no_dev_sequence_cache else {}
    )
    selection: Dict[str, Any] = {}
    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        train_metrics, train_performance = run_stc_epoch(
            dataset,
            candidates,
            teacher,
            device,
            arguments,
            protocol,
            training=True,
            generator=training_generator,
        )
        dev_metrics, dev_performance = run_stc_epoch(
            dev_dataset,
            candidates,
            teacher,
            device,
            arguments,
            protocol,
            training=False,
            generator=torch.Generator().manual_seed(arguments.seed + 9000),
            sequence_cache=dev_sequence_cache,
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "performance": {
                "train": train_performance,
                "train_dev": dev_performance,
            },
            "candidates": {},
        }
        for candidate in candidates:
            metrics = {
                **{
                    f"train_{name}": value
                    for name, value in train_metrics[candidate.name].items()
                },
                **{
                    f"dev_{name}": value
                    for name, value in dev_metrics[candidate.name].items()
                },
            }
            record["candidates"][candidate.name] = metrics
            candidate_configuration = {
                **configuration,
                "stc_weight": candidate.stc_weight,
                "candidate": candidate.name,
            }
            torch.save(
                {
                    "model": candidate.student.state_dict(),
                    "optimizer": candidate.optimizer.state_dict(),
                    "epoch": epoch,
                    "config": candidate_configuration,
                },
                output / f"{candidate.name}-last.pt",
            )
            current_dev_base = dev_metrics[candidate.name].get(
                "base_total", float("inf")
            )
            if current_dev_base < candidate.best_dev_base_total:
                candidate.best_dev_base_total = current_dev_base
                candidate.best_epoch = epoch
                torch.save(
                    {
                        "model": candidate.student.state_dict(),
                        "epoch": epoch,
                        "dev_base_total": current_dev_base,
                        "config": candidate_configuration,
                    },
                    output / f"{candidate.name}-best.pt",
                )
        history.append(record)
        write_json(output / "history.json", history)
        ranking = sorted(candidates, key=lambda item: item.best_dev_base_total)
        selection = {
            "selection_metric": "dev_base_total",
            "preliminary_selected": ranking[0].name,
            "candidates": [
                {
                    "name": candidate.name,
                    "stc_weight": candidate.stc_weight,
                    "best_epoch": candidate.best_epoch,
                    "best_dev_base_total": candidate.best_dev_base_total,
                    "checkpoint": f"{candidate.name}-best.pt",
                }
                for candidate in ranking
            ],
            "note": "训练损失仅用于各候选早停；最终选择必须比较 Train-dev 端到端 Event IoU、Pd、Proposal Recall、候选比例和计算代理。",
        }
        write_json(output / "selection.json", selection)
        print(json.dumps(record, ensure_ascii=False))
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    set_seed(arguments.seed)
    device = choose_device(arguments.device)
    dataset = S0PSequenceDataset(INDEX_PATH, arguments.role, arguments.time_steps)
    student, teacher = build_models(arguments, device)
    if not arguments.train:
        preflight(dataset, student, teacher, device, arguments)
        if arguments.stc_sweep:
            protocol = load_stc_protocol(arguments.stc_diagnostic)
            print(
                "STC 候选："
                + ", ".join(f"λ={weight:.2f}" for weight in STC_WEIGHTS)
            )
            print(
                f"诊断核：{protocol['spatial_kernel']}×"
                f"{protocol['spatial_kernel']}×{protocol['temporal_kernel']}"
            )
        if arguments.targeted_sweep:
            print("定向扫描候选：")
            for name, positive_fraction, distillation_weight in SWEEP_GRIDS[
                arguments.sweep_stage
            ]:
                print(
                    f"- {name}：正 patch {positive_fraction:.0%}，"
                    f"蒸馏权重 {distillation_weight:.2f}"
                )
        return
    if arguments.role != "train_main":
        raise ValueError("training may update weights only from train_main")
    dev_dataset = S0PSequenceDataset(INDEX_PATH, "train_dev", arguments.time_steps)
    if arguments.stc_sweep:
        protocol = load_stc_protocol(arguments.stc_diagnostic)
        train_stc_sweep(
            dataset,
            dev_dataset,
            student,
            teacher,
            device,
            arguments,
            protocol,
        )
    elif arguments.targeted_sweep:
        train_targeted_sweep(
            dataset, dev_dataset, student, teacher, device, arguments
        )
    else:
        train(dataset, dev_dataset, student, teacher, device, arguments)


if __name__ == "__main__":
    main()
