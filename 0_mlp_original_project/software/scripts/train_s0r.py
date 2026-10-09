"""S0-R 训练入口；默认只执行训练前检查，不更新权重。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.h2.numpy_model import H2Export, load_export
from wangdao.h2.torch_model import load_exported_models
from wangdao.s0p import role_file_paths
from wangdao.s0r import build_s0r_windows, flatten_s0r_windows, s0r_loss
from wangdao.snn import RefinerSNN


INDEX_PATH = ROOT / "data" / "index.json"
WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
RUN_ROOT = ROOT / "results" / "runs"


@dataclass(frozen=True)
class S0RData:
    role: str
    sequences: int
    windows: int
    features: np.ndarray
    labels: np.ndarray
    teacher_probability: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练事件独立、内部具有 LIF 微时间状态的 S0-R；未指定 --train 时只检查"
    )
    parser.add_argument("--train", action="store_true", help="明确开始更新模型权重")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-steps", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--readout", choices=RefinerSNN.READOUTS, default="mean_membrane"
    )
    parser.add_argument("--beta", type=float, default=0.875)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--distillation-weight", type=float, default=0.20)
    parser.add_argument("--spike-rate-weight", type=float, default=0.01)
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-dev-sequences", type=int, default=0)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="每处理多少条序列或多少批显示一次进度；0 表示关闭",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if arguments.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if arguments.distillation_weight < 0.0 or arguments.spike_rate_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 <= arguments.positive_fraction < 1.0:
        raise ValueError("positive-fraction must be in [0, 1)")
    if arguments.max_sequences < 0 or arguments.max_dev_sequences < 0:
        raise ValueError("sequence limits cannot be negative")
    if arguments.progress_interval < 0:
        raise ValueError("progress-interval cannot be negative")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> Dict[str, str]:
    paths = sorted((ROOT / "software").rglob("*.py"))
    paths.extend([ROOT / "pyproject.toml", INDEX_PATH])
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_model(arguments: argparse.Namespace, device: torch.device) -> RefinerSNN:
    _, teacher = load_exported_models(WEIGHT_PATH)
    model = RefinerSNN(
        beta=arguments.beta,
        threshold=arguments.threshold,
        surrogate_slope=arguments.surrogate_slope,
        readout=arguments.readout,
    )
    model.initialize_from_h2(teacher)
    return model.to(device)


def prepare_data(
    role: str,
    h2: H2Export,
    sequence_limit: int,
    progress_interval: int,
) -> S0RData:
    """一次生成并缓存候选特征，避免每轮重复执行 H2 Proposal。"""
    paths = role_file_paths(INDEX_PATH, role)
    if sequence_limit:
        paths = paths[:sequence_limit]
    windows = []
    started = time.perf_counter()
    for position, path in enumerate(paths, start=1):
        windows.extend(build_s0r_windows(path, h2))
        if progress_interval and (
            position % progress_interval == 0 or position == len(paths)
        ):
            events = sum(len(window.features) for window in windows)
            print(
                f"[{role} 数据准备] 序列 {position}/{len(paths)}，"
                f"窗口 {len(windows)}，候选事件 {events}，"
                f"累计 {time.perf_counter() - started:.1f} 秒",
                flush=True,
            )
    features, labels, teacher = flatten_s0r_windows(windows)
    return S0RData(
        role=role,
        sequences=len(paths),
        windows=len(windows),
        features=features,
        labels=labels,
        teacher_probability=teacher,
    )


def sample_indices(
    labels: np.ndarray,
    positive_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    count = len(labels)
    if positive_fraction == 0.0:
        return torch.randperm(count, generator=generator)
    target = torch.from_numpy(labels) > 0.5
    positive = torch.nonzero(target, as_tuple=False).flatten()
    negative = torch.nonzero(~target, as_tuple=False).flatten()
    if len(positive) == 0 or len(negative) == 0:
        return torch.randperm(count, generator=generator)
    positive_count = int(round(count * positive_fraction))
    negative_count = count - positive_count
    positive_choice = positive[
        torch.randint(len(positive), (positive_count,), generator=generator)
    ]
    if negative_count <= len(negative):
        negative_choice = negative[
            torch.randperm(len(negative), generator=generator)[:negative_count]
        ]
    else:
        negative_choice = negative[
            torch.randint(len(negative), (negative_count,), generator=generator)
        ]
    combined = torch.cat((positive_choice, negative_choice))
    return combined[torch.randperm(len(combined), generator=generator)]


def add_metrics(
    totals: Dict[str, float], losses: Dict[str, torch.Tensor], count: int
) -> None:
    for name, value in losses.items():
        if name == "spike_rates":
            for layer, rate in enumerate(value.detach().cpu().tolist()):
                key = f"spike_rate_{layer}"
                totals[key] = totals.get(key, 0.0) + float(rate) * count
        else:
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * count


def run_epoch(
    data: S0RData,
    model: RefinerSNN,
    device: torch.device,
    arguments: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator | None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    indices = (
        sample_indices(data.labels, arguments.positive_fraction, generator)
        if generator is not None
        else torch.arange(len(data.labels))
    )
    totals: Dict[str, float] = {}
    total = 0
    batch_count = (len(indices) + arguments.batch_size - 1) // arguments.batch_size
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_number, start in enumerate(
            range(0, len(indices), arguments.batch_size), start=1
        ):
            selected = indices[start : start + arguments.batch_size].numpy()
            features = torch.from_numpy(data.features[selected]).to(
                device, non_blocking=True
            )
            temporal = features.unsqueeze(0).expand(arguments.time_steps, -1, -1)
            labels = torch.from_numpy(data.labels[selected]).to(
                device, non_blocking=True
            )
            teacher = torch.from_numpy(data.teacher_probability[selected]).to(
                device, non_blocking=True
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(temporal)
            losses = s0r_loss(
                output.raw_logits,
                labels,
                teacher,
                output.spike_rates,
                distillation_weight=arguments.distillation_weight,
                spike_rate_weight=arguments.spike_rate_weight,
            ) | {"spike_rates": output.spike_rates}
            if optimizer is not None:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            count = len(selected)
            add_metrics(totals, losses, count)
            total += count
            if arguments.progress_interval and (
                batch_number % arguments.progress_interval == 0
                or batch_number == batch_count
            ):
                stage = "训练" if training else "开发"
                print(
                    f"[{stage}] 批次 {batch_number}/{batch_count}，"
                    f"事件 {total}/{len(indices)}，"
                    f"累计 {time.perf_counter() - started:.1f} 秒",
                    flush=True,
                )
    return {name: value / max(total, 1) for name, value in totals.items()}


def preflight(
    h2: H2Export,
    model: RefinerSNN,
    device: torch.device,
    arguments: argparse.Namespace,
) -> None:
    data = prepare_data("train_main", h2, 1, 0)
    count = min(len(data.features), arguments.batch_size)
    features = torch.from_numpy(data.features[:count]).to(device)
    temporal = features.unsqueeze(0).expand(arguments.time_steps, -1, -1)
    labels = torch.from_numpy(data.labels[:count]).to(device)
    teacher = torch.from_numpy(data.teacher_probability[:count]).to(device)
    model.eval()
    with torch.no_grad():
        output = model(temporal)
        losses = s0r_loss(
            output.raw_logits,
            labels,
            teacher,
            output.spike_rates,
            distillation_weight=arguments.distillation_weight,
            spike_rate_weight=arguments.spike_rate_weight,
        )
    print("S0-R 训练前检查通过")
    print(f"设备：{device}")
    print(
        f"事件独立分类，微时间步：{arguments.time_steps}，"
        f"读出：{arguments.readout}，检查事件：{count}"
    )
    print(f"初始损失：{float(losses['total'].cpu()):.6f}")
    print(
        "各层发放率："
        + ", ".join(
            f"{float(value):.6f}"
            for value in output.spike_rates.detach().cpu().tolist()
        )
    )
    print("未更新任何模型权重；添加 --train 才会开始训练")


def configuration(
    arguments: argparse.Namespace, train_data: S0RData, dev_data: S0RData
) -> Dict[str, Any]:
    excluded = {"train", "output"}
    return {
        "stage": "S0-R-independent-event-v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": {
            name: value
            for name, value in vars(arguments).items()
            if name not in excluded
        },
        "train_sequences": train_data.sequences,
        "train_windows": train_data.windows,
        "train_candidate_events": len(train_data.features),
        "dev_sequences": dev_data.sequences,
        "dev_windows": dev_data.windows,
        "dev_candidate_events": len(dev_data.features),
        "h2_weight": WEIGHT_PATH.relative_to(ROOT).as_posix(),
        "h2_weight_sha256": sha256_file(WEIGHT_PATH),
        "data_index": INDEX_PATH.relative_to(ROOT).as_posix(),
        "data_index_sha256": sha256_file(INDEX_PATH),
        "source_manifest": source_manifest(),
    }


def checkpoint(
    model: RefinerSNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
    }


def train(
    h2: H2Export,
    model: RefinerSNN,
    device: torch.device,
    arguments: argparse.Namespace,
) -> None:
    output = arguments.output or (
        RUN_ROOT / f"S0R-DIRECT-S{arguments.seed}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=False)
    train_data = prepare_data(
        "train_main", h2, arguments.max_sequences, arguments.progress_interval
    )
    dev_data = prepare_data(
        "train_dev", h2, arguments.max_dev_sequences, arguments.progress_interval
    )
    config = configuration(arguments, train_data, dev_data)
    write_json(output / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    generator = torch.Generator().manual_seed(arguments.seed)
    history: List[Dict[str, Any]] = []
    best_dev_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            train_data, model, device, arguments, optimizer, generator
        )
        dev_metrics = run_epoch(dev_data, model, device, arguments, None, None)
        metrics = {
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"dev_{name}": value for name, value in dev_metrics.items()},
        }
        record: Dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        history.append(record)
        state = checkpoint(model, optimizer, epoch, config, metrics)
        torch.save(state, output / f"epoch-{epoch:02d}.pt")
        torch.save(state, output / "last.pt")
        current_dev_loss = dev_metrics.get("total", float("inf"))
        if current_dev_loss < best_dev_loss:
            best_dev_loss = current_dev_loss
            best_epoch = epoch
            torch.save(state, output / "best.pt")
        write_json(output / "history.json", history)
        write_json(
            output / "selection.json",
            {
                "selection_metric": "dev_total",
                "best_epoch": best_epoch,
                "best_dev_total": best_dev_loss,
                "checkpoint": "best.pt",
                "note": "损失只作初选，最终 checkpoint 须按 Train-dev 端到端 Event IoU 复核。",
            },
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "best_epoch": best_epoch,
                "best_dev_total": best_dev_loss,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    set_seed(arguments.seed)
    device = choose_device(arguments.device)
    h2 = load_export(WEIGHT_PATH)
    model = build_model(arguments, device)
    if not arguments.train:
        preflight(h2, model, device, arguments)
        return
    train(h2, model, device, arguments)


if __name__ == "__main__":
    main()
