"""Bounded, replayable diagnostic snapshots; never substitute for checkpoints."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import torch

from ..h2pro_snn.training import checkpoint_save, cpu_weights, memory_stats, rng_state
from .refiner import LAYERS, STAT_NAMES


def json_safe(value) -> Any:
    """Keep NaN/Inf evidence in valid JSON, including failure paths."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result: dict = {"count": len(values), "finite_count": len(finite), "nan_count": int(np.isnan(values).sum()),
              "positive_inf_count": int(np.isposinf(values).sum()), "negative_inf_count": int(np.isneginf(values).sum())}
    if len(finite):
        result.update(min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()),
                      rms=float(np.sqrt(np.mean(finite*finite))), absmax=float(np.abs(finite).max()),
                      quantiles=dict(zip(("p01", "p05", "p50", "p95", "p99"),
                                         np.quantile(finite, [.01, .05, .5, .95, .99]).tolist())))
    return result


def batch_summary(batch, logits, warning_abs, sequence_metadata=None):
    labels = np.asarray(batch["labels"], bool)
    candidate = np.asarray(batch["candidate"], bool)
    refs, offset = [], 0
    for ref in batch.get("refs", []):
        ids = np.asarray(ref["ids"])
        stop = offset + len(ids)
        refs.append({"sequence": ref["sequence"], "slot": ref["slot"], "window": ref["window"],
                     "events": len(ids), "candidate_events": int(candidate[offset:stop].sum()),
                     "targets": int(labels[offset:stop].sum()),
                     "event_id_first": int(ids[0]) if len(ids) else None,
                     "event_id_last": int(ids[-1]) if len(ids) else None,
                     "high_density": (sequence_metadata or {}).get(ref["sequence"], {}).get("high_density")})
        offset = stop
    result = {"events": len(labels), "candidate_events": int(candidate.sum()),
              "targets": int(labels.sum()), "candidate_targets": int(labels[candidate].sum()),
              "active_slots": np.flatnonzero(batch["active"]).tolist(),
              "reset_slots": np.flatnonzero(batch["reset"]).tolist(), "sequences": refs}
    if logits is None:
        result["logits_available"] = False
        return result
    values = logits.detach().cpu().numpy()
    result.update(logits_available=True, candidate_logits=distribution(values[candidate]),
                  target_logits=distribution(values[candidate & labels]),
                  background_logits=distribution(values[candidate & ~labels]))
    finite = np.isfinite(values) & candidate
    positive = values >= 0
    large = finite & (np.abs(values) > warning_abs)
    result["diagnostic_at_logit_zero"] = {
        "note": "Candidate-only diagnostic; not the selected development operating point",
        "tp": int((finite & positive & labels).sum()), "fp": int((finite & positive & ~labels).sum()),
        "tn": int((finite & ~positive & ~labels).sum()), "fn": int((finite & ~positive & labels).sum()),
        "above_warning_count": int(large.sum()), "above_warning_positive": int((large & positive).sum()),
        "above_warning_negative": int((large & ~positive).sum()),
        "above_warning_correct": int((large & (positive == labels)).sum()),
        "above_warning_wrong": int((large & (positive != labels)).sum())}
    ids = np.flatnonzero(candidate)
    priority = np.nan_to_num(np.abs(values[ids]).astype(np.float64), nan=np.inf, posinf=np.inf)
    extremes = []
    for index in ids[np.argsort(-priority, kind="stable")[:8]]:
        item = {"batch_event_index": int(index), "logit": float(values[index]), "label": int(labels[index]),
                "teacher_probability": float(batch["teacher"][index]), "query": batch["maps"][:, index].tolist()}
        start = 0
        for ref in batch.get("refs", []):
            if start <= index < start + len(ref["ids"]):
                item.update(sequence=ref["sequence"], slot=int(ref["slot"]), window=int(ref["window"]),
                            event_id=int(ref["ids"][index-start]))
                break
            start += len(ref["ids"])
        extremes.append(item)
    result["extreme_candidates"] = extremes
    return result


class RunDiagnostics:
    """One full warning snapshot per phase per attempt, plus a fatal snapshot.

    JSON progress is appended every 50 batches and on phase completion; warning
    summaries are rate-limited. Labels are used only for diagnostics and losses.
    """
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = identity
        self.attempt = f"{time.time_ns()}_{os.getpid()}"
        self.folder = self.directory / "diagnostics" / self.attempt
        self.folder.mkdir(parents=True)
        self.context = None
        self.warning_snapshots = set()
        self.warning_counts = {}
        self.metadata = {}

    def begin(self, model, batch, state, monitor, device, *, phase, epoch, index, metadata=None):
        self.context = {"model": model, "batch": batch, "state_before": state, "state_after": None,
                        "logits": None, "monitor": monitor, "device": device, "phase": phase,
                        "epoch": epoch, "batch_index": index, "stage": "forward", "loss": None,
                        "loss_components": {}, "gradient_norm_pre_clip": None}
        self.metadata = metadata or {}
        model.last_diagnostics = None  # A failing forward must not expose the previous batch's statistics.
        model.last_voltage_tail = None

    def stage(self, name, **values):
        if self.context is not None:
            self.context.update(stage=name, **values)

    def observation(self, logits, state):
        self.stage("observe", logits=logits.detach(), state_after=tuple(s.detach() for s in state))

    def _record(self, kind, **extra):
        context = self.context
        record = {"schema_version": 1, "attempt": self.attempt, "kind": kind,
                  "activation_units": "continuous charge / theta; not spikes",
                  "utc": datetime.now(timezone.utc).isoformat(), **extra}
        if context is None:
            return record
        model, monitor = context["model"], context["monitor"]
        layers = None
        if model.last_diagnostics is not None:
            stats = model.last_diagnostics[:, context["batch"]["active"]].detach().cpu().numpy().reshape(-1, 3, len(STAT_NAMES))
            if len(stats):
                layers = {name: {"max": dict(zip(STAT_NAMES, stats[:, i].max(0).tolist())),
                                 "mean": dict(zip(STAT_NAMES, stats[:, i].mean(0, dtype=np.float64).tolist()))}
                          for i, name in enumerate(LAYERS)}
        record.update({k: context[k] for k in ("phase", "epoch", "batch_index", "stage", "loss", "loss_components", "gradient_norm_pre_clip")})
        record.update(batch=batch_summary(context["batch"], context["logits"], monitor.warning_limits["max_abs_logit"], self.metadata),
                      batch_layers=layers, stability=monitor.summary(), **memory_stats(context["device"]))
        record["parameters"] = {n: distribution(p.detach().cpu().numpy()) for n, p in model.named_parameters()}
        record["gradients_at_capture"] = {n: distribution(p.grad.detach().cpu().numpy()) if p.grad is not None else None
                                          for n, p in model.named_parameters()}
        tail = model.last_voltage_tail
        if tail is not None:
            active = tail[:, context["batch"]["active"]].detach().cpu().numpy()
            record["head_voltage_regularizer"] = {
                "soft_limit": model.config.head_voltage_soft_limit,
                "positive": distribution(active[..., 0]), "negative": distribution(active[..., 1]),
                "meaning": "Squared per-stream-step peak excess; signs separate; never a clamp"}
        return record

    def append(self, record):
        record = json_safe(record)
        with (self.directory / "diagnostics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def snapshot(self, stem):
        if self.context is None:
            return None
        context = self.context
        model = context["model"]
        path = self.folder / (stem + ".pt")
        checkpoint_save(path, {"schema_version": 1, "purpose": "diagnostic_only_not_resumable",
                              "identity": self.identity, "model_config": asdict(model.config),
                              "model": cpu_weights(model), "gradients": {n: cpu_tree(p.grad) for n, p in model.named_parameters()},
                              "batch": cpu_tree(context["batch"]), "state_before": cpu_tree(context["state_before"]),
                              "state_after": cpu_tree(context["state_after"]), "logits": cpu_tree(context["logits"]),
                              "layer_diagnostics": cpu_tree(model.last_diagnostics),
                              "head_voltage_tail": cpu_tree(model.last_voltage_tail), "rng_at_capture": rng_state(),
                              "position": {k: context[k] for k in ("phase", "epoch", "batch_index", "stage")},
                              "forward_replay_weights_match": context["stage"] not in ("optimizer_step", "after_optimizer_step"),
                              "note": "No optimizer/scheduler/cursor recovery; never load as last.pt or best.pt"})
        return path.relative_to(self.directory).as_posix()

    def warnings(self, alerts):
        if not alerts or self.context is None:
            return
        phase = self.context["phase"]
        count = self.warning_counts.get(phase, 0) + 1
        self.warning_counts[phase] = count
        first = phase not in self.warning_snapshots
        if first or count % 50 == 0:
            snapshot = self.snapshot(f"first_warning_{phase}") if first else None
            if first:
                self.warning_snapshots.add(phase)
            record = self._record("warning", warnings=alerts, warning_batch_count=count, snapshot=snapshot)
            self.append(record)
            write_json(self.folder / f"latest_warning_{phase}.json", record)
            print(json.dumps({"stage": "stability_warning", "phase": phase, "epoch": self.context["epoch"],
                              "batch": self.context["batch_index"], "warnings": alerts,
                              "snapshot": snapshot, "action": "recorded_continuing_no_numeric_change"}), flush=True)

    def progress(self, **values):
        record = self._record("progress", **values)
        self.append(record)
        write_json(self.directory / "progress.json", record)

    def release(self):
        self.context = None

    def failure(self, error):
        """Best effort: disk/CUDA failures while diagnosing must not mask the cause."""
        record = {"exception": type(error).__name__, "message": str(error), "capture_errors": [],
                  "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}
        if self.context is not None:
            record.update({k: self.context[k] for k in ("phase", "epoch", "batch_index", "stage")})
        try:
            record.update(self._record("failure"))
        except Exception as secondary:
            record["capture_errors"].append("summary: " + repr(secondary))
        try:
            record["snapshot"] = self.snapshot("failure")
        except Exception as secondary:
            record["capture_errors"].append("snapshot: " + repr(secondary))
        try:
            write_json(self.folder / "failure.json", record)
            self.append(record)
            write_json(self.directory / "progress.json", {**record, "status": "stopped"})
        except Exception as secondary:
            record["capture_errors"].append("write: " + repr(secondary))
        record["diagnostic_file"] = (self.folder / "failure.json").relative_to(self.directory).as_posix()
        self.release()
        return json_safe(record)
