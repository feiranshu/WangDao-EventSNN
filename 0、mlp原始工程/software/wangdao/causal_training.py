"""Training and replay evaluation for the phase-one causal ANN experiments.

Four refiner outputs are two prototypes per binary class, matching H2. Labels
are never passed to the feature stream or the inference model. Timing describes
synchronized software replay, not a board or live-camera measurement.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import json
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .causal import CausalConfig, CausalModel, FeatureStream
from .h2.losses import proposal_loss, refiner_loss
from .phase1_common import (
    ROOT, aggregate_scores, load_sequence, role_paths, score_predictions,
    select_working_point, sha256, write_json, register_experiment, pr_curve, PROTOCOL_ID,
)


CHECKPOINT_VERSION = 1
REFERENCE_THRESHOLDS = (0.28, 0.9)
FINAL_THRESHOLDS = tuple(sorted(set([i / 100.0 for i in range(5, 100, 5)] + [0.28])))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 42
    tbptt_blocks: int = 8
    cache_mib: int = 256
    smoke: bool = False
    smoke_blocks: int = 8

    def validate(self) -> None:
        if not 1 <= self.epochs <= 20:
            raise ValueError("epochs must be between 1 and the planned maximum of 20")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive; weight decay nonnegative")
        if self.tbptt_blocks != 8:
            raise ValueError("the frozen training protocol uses eight 25 ms blocks")
        if self.cache_mib < 0 or self.smoke_blocks <= 0:
            raise ValueError("cache size must be nonnegative and smoke blocks positive")


def choose_device(requested: str) -> torch.device:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inference_events(events: np.ndarray) -> np.ndarray:
    """Copy only the four inference fields; labels cannot leak into features."""
    if events.dtype.names is None:
        array = np.asarray(events, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("inference events must have exactly x/y/t/p columns")
        return array
    return np.column_stack([events[name] for name in ("x", "y", "t", "p")]).astype(np.float64)


class SequenceCache:
    """Bounded cache of immutable NPZ arrays, consumed by one prefetch worker."""

    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.max_bytes = int(max_bytes)
        self.items: OrderedDict = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.load_seconds = 0.0

    def get(self, path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = str(path.resolve())
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key]
        self.misses += 1
        started = time.perf_counter()
        values = tuple(load_sequence(path))
        self.load_seconds += time.perf_counter() - started
        size = sum(value.nbytes for value in values)
        for value in values:
            value.flags.writeable = False
        if size <= self.max_bytes:
            while self.items and self.bytes + size > self.max_bytes:
                _, old = self.items.popitem(last=False)
                self.bytes -= sum(value.nbytes for value in old)
            self.items[key] = values
            self.bytes += size
        return values  # type: ignore[return-value]

    def statistics(self) -> Dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "resident_bytes": self.bytes,
                "limit_bytes": self.max_bytes, "load_seconds": self.load_seconds}


def prefetch_sequences(paths: Sequence[Path], cache: SequenceCache) -> Iterator[Tuple[Path, Any, float]]:
    """Prefetch exactly one sequence without changing sequence or event order."""
    if not paths:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache.get, paths[0])
        for position, path in enumerate(paths):
            started = time.perf_counter()
            data = future.result()
            wait_seconds = time.perf_counter() - started
            if position + 1 < len(paths):
                future = executor.submit(cache.get, paths[position + 1])
            yield path, data, wait_seconds


def iter_blocks(events: np.ndarray, config: CausalConfig,
                max_blocks: Optional[int] = None) -> Iterator[Tuple[Any, float]]:
    """Publish half-open blocks, including empty gaps and the final partial block."""
    raw = inference_events(events)
    if len(raw) == 0:
        return
    step = float(config.block_ms)
    start = float(np.floor(raw[0, 2] / step) * step)
    stream = FeatureStream(config, start_ms=start)
    stop = float((np.floor(raw[-1, 2] / step) + 1) * step)
    position = 0
    number = 0
    while start < stop and (max_blocks is None or number < max_blocks):
        end = start + step
        right = int(np.searchsorted(raw[:, 2], end, side="left"))
        started = time.perf_counter()
        blocks = stream.push(raw[position:right], cutoff_ms=end)
        preprocessing_ms = (time.perf_counter() - started) * 1000.0
        if len(blocks) != 1:
            raise RuntimeError("one complete publication interval must produce exactly one block")
        yield blocks[0], preprocessing_ms
        position, start, number = right, end, number + 1


def block_losses(output: Dict[str, Any], block: Any, labels: np.ndarray,
                 variant: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Supervise proposals and binary event prototypes, including both C4 heads."""
    target = torch.as_tensor(labels[block.event_indices].copy(), dtype=torch.float32, device=device)
    proposal_target = torch.zeros_like(output["proposal_logits"])
    positives = np.flatnonzero(labels[block.event_indices] > 0)
    if len(positives):
        xy = block.events[positives, :2].astype(np.int64)
        proposal_target[torch.as_tensor(xy[:, 1] // 8, device=device),
                        torch.as_tensor(xy[:, 0] // 8, device=device)] = 1.0
    p_loss = proposal_loss(output["proposal_logits"], proposal_target)
    zero = output["proposal_logits"].sum() * 0.0
    event = zero
    light = zero
    if len(target):
        mask = torch.zeros_like(target)
        event = refiner_loss(output["logits"], target, mask, mask,
                             distillation_weight=0.0)["total"]
        if variant == "C4":
            light = refiner_loss(output["light_logits"], target, mask, mask,
                                 distillation_weight=0.0)["total"]
    return {"total": p_loss["total"] + event + light, "proposal": p_loss["total"],
            "event": event, "light": light}


def train_sequence(model: CausalModel, events: np.ndarray, labels: np.ndarray,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   config: CausalConfig, tbptt_blocks: int = 8,
                   max_blocks: Optional[int] = None) -> Dict[str, Any]:
    """State is reset per sequence and detached only at optimizer boundaries."""
    model.train()
    state = None
    pending: List[torch.Tensor] = []
    total: Dict[str, float] = {name: 0.0 for name in ("total", "proposal", "event", "light")}
    block_count = 0
    event_count = 0
    updates = 0
    preprocessing_ms = 0.0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    def update() -> None:
        nonlocal state, updates
        if not pending:
            return
        loss = torch.stack(pending).mean()
        if not bool(torch.isfinite(loss).detach()):
            raise FloatingPointError("non-finite causal training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = state.detach() if state is not None else None
        pending.clear()
        updates += 1

    for block, build_ms in iter_blocks(events, config, max_blocks):
        output = model.forward_block(block, state, force_deep=True)
        state = output["state"]
        losses = block_losses(output, block, labels, config.variant, device)
        pending.append(losses["total"])
        for name, value in losses.items():
            total[name] += float(value.detach().cpu())
        block_count += 1
        event_count += len(block.event_indices)
        preprocessing_ms += build_ms
        if len(pending) == tbptt_blocks:
            update()
    update()
    synchronize(device)
    return {"losses": {key: value / max(block_count, 1) for key, value in total.items()},
            "blocks": block_count, "events": event_count, "optimizer_updates": updates,
            "preprocessing_ms": preprocessing_ms,
            "elapsed_seconds": time.perf_counter() - started}


def percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(values) == 0:
        return {"p50": None, "p95": None, "p99": None}
    return {"p%d" % p: float(np.percentile(values, p)) for p in (50, 95, 99)}


def predict_sequence(model: CausalModel, events: np.ndarray, device: torch.device,
                     config: CausalConfig, max_blocks: Optional[int] = None,
                     output_transfer_ms: float = 0.0) -> Dict[str, Any]:
    """Evaluate all emitted events, accounting for rejected targets as negatives.

    Host representation, synchronized forward, and device-to-host scores are
    timed. Output transmission is an explicit assumed cost (default 0/unmeasured).
    Waiting and queue delay are deterministic replay simulation, not wall clock.
    """
    if output_transfer_ms < 0 or not np.isfinite(output_transfer_ms):
        raise ValueError("output transfer time must be finite and nonnegative")
    model.eval()
    raw = inference_events(events)
    probability = np.zeros(len(raw), dtype=np.float32)
    candidate = np.zeros(len(raw), dtype=bool)
    deep = np.zeros(len(raw), dtype=bool)
    emitted = np.zeros(len(raw), dtype=bool)
    output_times = np.zeros(len(raw), dtype=np.float64)
    state = None
    records = []
    available_ms = float(raw[0, 2]) if len(raw) else 0.0
    max_state_bytes = 0
    pending_completions = []
    max_backlog_blocks = 0
    with torch.no_grad():
        for block, build_ms in iter_blocks(raw, config, max_blocks):
            synchronize(device)
            started = time.perf_counter()
            output = model.forward_block(block, state)
            synchronize(device)
            forward_ms = (time.perf_counter() - started) * 1000.0
            state = output["state"]
            started = time.perf_counter()
            scores = output["probabilities"].detach().cpu().numpy()
            candidates = output["candidate_mask"].detach().cpu().numpy().astype(bool)
            deep_mask = output["deep_mask"].detach().cpu().numpy().astype(bool)
            synchronize(device)
            readout_ms = (time.perf_counter() - started) * 1000.0
            indices = block.event_indices
            if len(indices) != len(scores) or np.any(emitted[indices]):
                raise RuntimeError("each event must be emitted exactly once")
            if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
                raise FloatingPointError("invalid event probabilities")
            processing_ms = build_ms + forward_ms + readout_ms
            pending_completions = [t for t in pending_completions if t > block.end_ms]
            max_backlog_blocks = max(max_backlog_blocks, len(pending_completions))
            queue_ms = max(0.0, available_ms - float(block.end_ms))
            finish_ms = max(available_ms, float(block.end_ms)) + processing_ms + output_transfer_ms
            available_ms = finish_ms
            pending_completions.append(finish_ms)
            probability[indices] = scores
            candidate[indices] = candidates
            deep[indices] = deep_mask
            emitted[indices] = True
            output_times[indices] = finish_ms
            state_bytes = state.tensor_bytes() if callable(state.tensor_bytes) else state.tensor_bytes
            max_state_bytes = max(max_state_bytes, int(state_bytes))
            records.append({"start_ms": float(block.start_ms), "end_ms": float(block.end_ms),
                            "events": len(indices), "representation_ms": build_ms,
                            "forward_ms": forward_ms, "host_readout_ms": readout_ms,
                            "processing_ms": processing_ms, "queue_ms": queue_ms,
                            "simulated_finish_ms": finish_ms})
    indices = np.flatnonzero(emitted)
    if max_blocks is None and len(indices) != len(raw):
        raise RuntimeError("full replay omitted events")
    if not np.array_equal(indices, np.arange(len(indices))):
        raise RuntimeError("partial replay must emit a contiguous event prefix")
    latencies = output_times[indices] - raw[indices, 2]
    processing = [record["processing_ms"] for record in records]
    total_ms = sum(processing)
    span_ms = records[-1]["end_ms"] - records[0]["start_ms"] if records else 0.0
    return {"event_indices": indices, "probabilities": probability[indices],
            "candidate_mask": candidate[indices], "deep_mask": deep[indices],
            "simulated_output_times_ms": output_times[indices], "timing_blocks": records,
            "timing": {"kind": "synchronized_software_replay_simulation",
                       "device": str(device), "processing_ms": percentiles(processing),
                       "event_end_to_end_ms": percentiles(latencies),
                       "queue_ms": percentiles([record["queue_ms"] for record in records]),
                       "total_processing_ms": total_ms,
                       "seconds_per_million_events": total_ms * 1000.0 / len(indices) if len(indices) else None,
                       "input_events_per_second": len(indices) * 1000.0 / span_ms if span_ms else None,
                       "max_backlog_blocks": max_backlog_blocks,
                       "backlog_definition": "unfinished prior blocks at each publication deadline",
                       "peak_neural_state_bytes": max_state_bytes,
                       "representation_state_bytes": FeatureStream(config).state_bytes,
                       "event_wait_ms": percentiles(np.mod(-raw[indices, 2], config.block_ms) +
                                                   (np.mod(raw[indices, 2], config.block_ms) == 0) * config.block_ms),
                       "assumed_output_transfer_ms": output_transfer_ms,
                       "power": {"status": "not_measured", "energy_joules": None},
                       "includes_cold_first_block": True, "excludes": ["file_io", "metric_scoring"]}}


def evaluate_paths(model: CausalModel, paths: Sequence[Path], device: torch.device,
                   config: CausalConfig, thresholds: Sequence[float] = REFERENCE_THRESHOLDS,
                   cache: Optional[SequenceCache] = None, max_blocks: Optional[int] = None,
                   prediction_dir: Optional[Path] = None, output_transfer_ms: float = 0.0) -> Dict[str, Any]:
    cache = cache if cache is not None else SequenceCache()
    rows: Dict[str, List[Dict[str, Any]]] = {format(float(t), ".8g"): [] for t in thresholds}
    per_sequence = []
    pr_labels, pr_probabilities = [], []
    data_wait = 0.0
    scoring_seconds = 0.0
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=False)
    for path, (events, loc, labels, identities), wait in prefetch_sequences(paths, cache):
        data_wait += wait
        result = predict_sequence(model, events, device, config, max_blocks, output_transfer_ms)
        indices = result["event_indices"]
        if len(indices) == 0:
            raise RuntimeError("sequence replay produced no events; increase smoke block count")
        scores = {}
        started = time.perf_counter()
        for threshold in thresholds:
            key = format(float(threshold), ".8g")
            score = score_predictions(loc[indices], labels[indices], identities[indices],
                                      result["probabilities"], threshold)
            rows[key].append(score)
            scores[key] = score
        scoring_seconds += time.perf_counter() - started
        if prediction_dir is not None:
            pr_labels.append(labels[indices])
            pr_probabilities.append(result["probabilities"])
            target = prediction_dir / (path.stem + ".npz")
            with target.open("xb") as handle:
                np.savez_compressed(handle, event_indices=indices, probabilities=result["probabilities"],
                                    candidate_mask=result["candidate_mask"], deep_mask=result["deep_mask"],
                                    simulated_output_times_ms=result["simulated_output_times_ms"])
        count = len(indices)
        per_sequence.append({"sequence": path.name, "events": count,
                             "full_sequence_events": len(events), "partial": count != len(events),
                             "candidate_events": int(result["candidate_mask"].sum()),
                             "deep_events": int(result["deep_mask"].sum()),
                             "target_events": int(np.count_nonzero(labels[indices])),
                             "candidate_target_events": int(np.count_nonzero(result["candidate_mask"] & (labels[indices] > 0))),
                             "scores": scores, "timing": result["timing"],
                             "timing_blocks": result["timing_blocks"]})
        print("[evaluate] %s events=%d IoU@0.28=%s" %
              (path.name, count, scores.get("0.28", {}).get("event", {}).get("iou")), flush=True)
    total_events = sum(row["events"] for row in per_sequence)
    targets = sum(row["target_events"] for row in per_sequence)
    return {"sequence_count": len(per_sequence), "event_count": total_events,
            "event_pr_curve": (pr_curve(np.concatenate(pr_labels), np.concatenate(pr_probabilities))
                               if pr_labels else None),
            "partial": max_blocks is not None,
            "curve": {key: aggregate_scores(value) for key, value in rows.items()},
            "candidate_ratio": sum(row["candidate_events"] for row in per_sequence) / total_events if total_events else 0.0,
            "deep_ratio": sum(row["deep_events"] for row in per_sequence) / total_events if total_events else 0.0,
            "proposal_recall": sum(row["candidate_target_events"] for row in per_sequence) / targets if targets else None,
            "per_sequence": per_sequence,
            "performance": {"data_wait_seconds": data_wait, "metric_seconds": scoring_seconds,
                            "cache": cache.statistics()}}


def save_checkpoint(path: Path, model: CausalModel, config: CausalConfig,
                    metadata: Dict[str, Any]) -> None:
    with path.open("xb") as handle:
        torch.save({"format_version": CHECKPOINT_VERSION, "model_config": asdict(config),
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "metadata": metadata, "recurrent_state_exported": False}, handle)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[CausalModel, CausalConfig, Dict[str, Any]]:
    try:
        package = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        package = torch.load(str(path), map_location="cpu")
    if package.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported causal checkpoint format")
    config = CausalConfig(**package["model_config"])
    model = CausalModel(config).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    model.eval()
    return model, config, package.get("metadata", {})


def environment(device: torch.device) -> Dict[str, Any]:
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "platform": platform.platform()}


def train_experiment(run_dir: Path, model_config: CausalConfig, config: TrainConfig,
                     device: torch.device, initialize_from: Optional[Path] = None) -> Dict[str, Any]:
    config.validate()
    if model_config.variant == "C4" and not config.smoke and initialize_from is None:
        raise ValueError("Formal C4 must start from its trained C3 checkpoint (--initialize-from)")
    set_seed(config.seed)
    paths = role_paths("train_main", limit=1 if config.smoke else None)
    dev_paths = role_paths("train_dev", limit=1 if config.smoke else None)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    model = CausalModel(model_config).to(device)
    initialization = None
    if initialize_from is not None:
        previous, previous_config, previous_metadata = load_checkpoint(initialize_from, device)
        if previous_metadata.get("smoke", True) and not config.smoke:
            raise ValueError("Formal training cannot initialize from a smoke checkpoint")
        if previous_metadata.get("data_index_sha256") != sha256(ROOT / "data/index.json"):
            raise ValueError("Initialization checkpoint uses a different dataset index")
        if model_config.variant == "C4" and previous_config.variant != "C3":
            raise ValueError("C4 initialization must be the measured C3 strong-refiner model")
        source_state = previous.state_dict()
        target_state = model.state_dict()
        shared = {key: value for key, value in source_state.items()
                  if key in target_state and target_state[key].shape == value.shape}
        if not shared:
            raise ValueError("No compatible parameters in initialization checkpoint")
        target_state.update(shared)
        model.load_state_dict(target_state, strict=True)
        initialization = {"checkpoint": str(initialize_from), "sha256": sha256(initialize_from),
                          "source_variant": previous_config.variant, "transferred_tensors": len(shared)}
    recipe_id = register_experiment({"model": asdict(model_config),
                                    **{k: v for k, v in asdict(config).items()
                                       if k not in ("smoke", "smoke_blocks")}}, config.smoke)
    run_dir.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epochs = 1 if config.smoke else config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    cache = SequenceCache(config.cache_mib * 1024 * 1024)
    max_blocks = config.smoke_blocks if config.smoke else None
    history = []
    summary: Dict[str, Any] = {
        "schema_version": 1, "status": "running", "experiment_id": run_dir.name,
        "stage": "phase1_causal_ann", "data_role": "train_main+train_dev",
        "primary_metrics": {}, "decision": "training", "conclusion": "No accuracy or efficiency claim before measurement.",
        "smoke": config.smoke, "formal_configuration_budget_used": not config.smoke,
        "recipe_id": recipe_id, "metric_protocol": PROTOCOL_ID,
        "initialization": initialization,
        "test_used": False, "validation_used": False, "model_config": asdict(model_config),
        "training_config": asdict(config), "environment": environment(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selection_protocol": {"checkpoint": "full Train-dev micro IoU at 0.28", "reference_thresholds": list(REFERENCE_THRESHOLDS),
                               "deployment": "requires separate Pd/Fa-constrained Train-dev selection",
                               "labels_in_inference": False, "tbptt_blocks": 8,
                               "event_loss": "all events, two binary prototypes/class; C4 both heads; inference retains proposal gate"},
        "provenance": {"data_index_sha256": sha256(ROOT / "data" / "index.json"),
                       "train_sequences": [path.name for path in paths], "dev_sequences": [path.name for path in dev_paths]},
        "history": history,
        "conditional_branches": {"teacher": "not_triggered_requires_Train_dev_evidence",
                                 "integer_export": "not_implemented_no_bit_exact_validation"}}
    write_json(run_dir / "summary.json", summary)
    best_iou = -float("inf")
    best_path = None
    try:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            order = list(paths)
            random.Random(config.seed + epoch - 1).shuffle(order)
            train_rows = []
            data_wait = 0.0
            for path, (events, _loc, labels, _ids), wait in prefetch_sequences(order, cache):
                data_wait += wait
                row = train_sequence(model, events, labels, optimizer, device, model_config,
                                     config.tbptt_blocks, max_blocks)
                row["sequence"] = path.name
                train_rows.append(row)
                print("[train epoch %d/%d] %s loss=%.6f" %
                      (epoch, epochs, path.name, row["losses"]["total"]), flush=True)
            dev = evaluate_paths(model, dev_paths, device, model_config, cache=cache, max_blocks=max_blocks)
            iou_value = dev["curve"]["0.28"]["event"]["iou"]
            iou = float(iou_value) if iou_value is not None else -float("inf")
            checkpoint = run_dir / ("epoch_%03d.pt" % epoch)
            save_checkpoint(checkpoint, model, model_config,
                            {"epoch": epoch, "seed": config.seed, "smoke": config.smoke,
                             "train_dev_iou_at_0p28": iou_value, "data_index_sha256": summary["provenance"]["data_index_sha256"]})
            # A short smoke prefix may contain no positive event and no prediction.
            # Keep its IoU null; save a smoke checkpoint solely to test round-trips.
            selected = iou > best_iou or (config.smoke and best_path is None)
            if selected:
                best_iou, best_path = iou, checkpoint
                summary["primary_metrics"] = dev["curve"]["0.28"]
                summary["best_checkpoint"] = checkpoint.name
                summary["best_epoch"] = epoch
            history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                            "train": train_rows, "dev": dev, "selected": selected,
                            "performance": {"data_wait_seconds": data_wait,
                                            "epoch_seconds": time.perf_counter() - started, "cache": cache.statistics()}})
            scheduler.step()
            write_json(run_dir / "summary.json", summary, overwrite=True)
        if best_path is None:
            raise RuntimeError("no defined Train-dev IoU; no checkpoint can be selected")
        summary["status"] = "complete"
        summary["decision"] = "smoke_only" if config.smoke else "checkpoint_selected_not_deployment_frozen"
        summary["best_checkpoint_sha256"] = sha256(best_path)
        summary["conclusion"] = "Short synthetic/partial protocol verification only." if config.smoke else "Training finished; constrained Train-dev selection and independent stage validation remain required."
        write_json(run_dir / "summary.json", summary, overwrite=True)
        return summary
    except Exception as error:
        summary.update(status="failed", decision="stop", error=type(error).__name__ + ": " + str(error))
        write_json(run_dir / "summary.json", summary, overwrite=True)
        raise


def select_evaluation(result: Dict[str, Any], baseline: Optional[Dict[str, Any]],
                      variant: str) -> Dict[str, Any]:
    if result.get("partial"):
        return {"eligible": False, "threshold": None, "metrics": None, "reason": "partial/smoke results cannot freeze a deployment"}
    if baseline is None:
        if variant != "C0":
            return {"eligible": False, "threshold": None, "metrics": None,
                    "reason": "supply the same-condition C0 Train-dev evaluation as baseline"}
        metrics = result["curve"]["0.28"]
        if metrics["event"]["iou"] is None or metrics["localization"]["pd"] is None:
            return {"eligible": False, "threshold": None, "metrics": None,
                    "reason": "Undefined baseline metrics cannot establish a working point"}
        return {"eligible": True, "threshold": 0.28, "metrics": metrics,
                "reason": "predeclared C0 baseline working point; no threshold optimization"}
    metrics = baseline.get("selection", {}).get("metrics")
    if (not metrics or baseline.get("data_role") != "train_dev" or baseline.get("smoke", True)
            or baseline.get("status") != "complete" or not baseline.get("selection", {}).get("eligible")
            or baseline.get("model_config", {}).get("variant") != "C0"
            or baseline.get("metric_protocol") != PROTOCOL_ID
            or baseline.get("data_index_sha256") != sha256(ROOT / "data/index.json")
            or baseline.get("evaluation", {}).get("partial", True)):
        raise ValueError("baseline must be a complete selected Train-dev evaluation")
    if baseline.get("evaluation", {}).get("sequence_count") != result["sequence_count"]:
        raise ValueError("baseline and candidate sequence counts differ")
    index = json.loads((ROOT / "data/index.json").read_text(encoding="utf-8"))
    records = {row["path"].split("/")[-1]: row["events"] for row in index["files"]}
    expected = set(index["roles"]["train_dev"])
    for evaluation in (result, baseline["evaluation"]):
        rows = evaluation.get("per_sequence", [])
        if len(rows) != len(expected) or {r["sequence"] for r in rows} != expected:
            raise ValueError("Candidate/baseline must contain exactly the indexed Train-dev sequences")
        if any(r.get("partial", True) or r["events"] != records[r["sequence"]] for r in rows):
            raise ValueError("Candidate/baseline event coverage mismatch")
    localization = metrics["localization"]
    return select_working_point(result["curve"], localization["pd"], localization["fa"])
