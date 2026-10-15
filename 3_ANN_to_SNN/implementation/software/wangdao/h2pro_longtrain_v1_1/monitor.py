"""Finite excursions are recorded, never clamped, skipped or used to stop a run."""
import copy
import json
import math
from pathlib import Path
from typing import Any, cast

import torch

from ..h2pro_ann_state_v1.monitor import HealthMonitor as StateMonitor
from ..h2pro_ann_readout_v1.monitor import HealthMonitor as LinearMonitor
from .models import POLICY


class AdvisoryPolicy:
    limits: dict[str, float]

    def __init__(self, limits=None, warnings=None):
        # Cooperative mixin: the next class is the matching immutable ANN monitor.
        cast(Any, super()).__init__(limits, warnings)
        self.advisory_counts = {k: 0 for k in self.limits}
        self.advisory_maxima = {k: None for k in self.limits}
        self.pending_advisories = []

    def _limit(self, name, value):
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite numerical value: {name}={value}")
        level = self.limits[name]
        if value <= level:
            return
        self.advisory_counts[name] += 1
        previous = self.advisory_maxima[name]
        self.advisory_maxima[name] = value if previous is None else max(previous, value)
        count = self.advisory_counts[name]
        # Keep aggregate counts/maxima for every observation; rate-limit text I/O.
        if count == 1 or count % 1000 == 0 or (previous is not None and value > previous*2):
            self.pending_advisories.append({"quantity": name, "value": value, "reference_level": level,
                                           "occurrences": count, "action": "record_only_continue_training"})

    def drain(self):
        values, self.pending_advisories = self.pending_advisories, []
        return values

    def summary(self):
        result = cast(Any, super()).summary()
        result.update(monitoring_policy=POLICY, failure_limits={}, advisory_limits=dict(self.limits),
                      finite_advisories={k: {"occurrences": n, "maximum": self.advisory_maxima[k]}
                                         for k, n in self.advisory_counts.items()},
                      numerical_stop_policy="NaN/Inf only; data, state and checkpoint integrity checks remain",
                      finite_logit_policy="Advisory only; no clipping or automatic hyperparameter change")
        result["warnings"] += ["finite_advisory_"+k for k, n in self.advisory_counts.items() if n]
        return result


class StateHealthMonitor(AdvisoryPolicy, StateMonitor):
    pass


class LinearHealthMonitor(AdvisoryPolicy, LinearMonitor):
    pass


def monitor_for(arm, config, restored=None):
    if arm not in ("ann_state", "ann_linear", "snn_linear"):
        raise ValueError("Unknown control")
    cls = StateHealthMonitor if arm == "ann_state" else LinearHealthMonitor
    monitor = cls(config["advisory_limits"], config["warning_limits"])
    if restored is not None:
        if restored["limits"] != monitor.limits or restored["warning_limits"] != monitor.warning_limits:
            raise ValueError("Monitor reference levels changed")
        monitor.__dict__.update({k: v.numpy().copy() if torch.is_tensor(v) else copy.deepcopy(v)
                                 for k, v in restored.items()})
    return monitor


def emit_advisories(monitor, directory, *, phase, epoch=None, batch=None):
    for event in monitor.drain():
        row = {"stage": "numerical_advisory", "phase": phase, "epoch": epoch, "batch": batch, **event}
        if directory is not None:
            with (Path(directory)/"advisories.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps(row, allow_nan=False), flush=True)
