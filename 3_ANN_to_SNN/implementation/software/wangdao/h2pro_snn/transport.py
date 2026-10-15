"""Event-driven numerical reference for threshold-before/after-transport controls.

Routes are externally supplied hypotheses, NOT a learned motion estimator.
Currents are already encoded. Neither the encoder nor routing inference is costed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

import numpy as np

from .costs import OperationLedger

MODES = ("fixed", "spike_route", "membrane_route")


@dataclass(frozen=True)
class TransportConfig:
    rows: int = 68
    columns: int = 88
    channels: int = 32
    local_channels: int = 16
    tau_short_ms: float = 20.0
    tau_long_ms: float = 100.0
    threshold: float = 1.0

    def __post_init__(self):
        for name in ("rows", "columns", "channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if type(self.local_channels) is not int or not 0 <= self.local_channels <= self.channels:
            raise ValueError("local_channels must lie between zero and channels")
        for name in ("tau_short_ms", "tau_long_ms", "threshold"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")

    @property
    def cells(self) -> int:
        return self.rows * self.columns

    @property
    def tau(self) -> np.ndarray:
        result = np.full(self.channels, self.tau_long_ms, dtype=np.float64)
        result[:self.local_channels] = self.tau_short_ms
        return result

    def validate_cell(self, cell: int) -> int:
        if isinstance(cell, (bool, np.bool_)) or not isinstance(cell, (int, np.integer)) or not 0 <= cell < self.cells:
            raise ValueError("Cell index outside the grid")
        return int(cell)

    def validate_routes(self, routes: Mapping[int, int]) -> dict:
        result = {}
        for source, destination in routes.items():
            source, destination = self.validate_cell(source), self.validate_cell(destination)
            sy, sx = divmod(source, self.columns)
            dy, dx = divmod(destination, self.columns)
            if max(abs(sy - dy), abs(sx - dx)) > 1:
                raise ValueError("Transport must remain within the 3x3 neighborhood")
            result[source] = destination
        return result


class SparseTransportLIF:
    """Lazy timestamp decay and simultaneous, single-owner local transport.

    A burst count n denotes n binary spikes at one timestamp. In spike_route mode
    only the emitted spikes move; residual voltage remains at the source. No
    spontaneous spikes occur between inputs because all post-reset positive
    residuals are below threshold and there is no autonomous bias current.
    """
    def __init__(self, config: TransportConfig, mode: str = "fixed"):
        if mode not in MODES:
            raise ValueError("Unknown transport mode")
        self.config, self.mode = config, mode
        self.reset()

    def reset(self, t0_ms: float = 0.0) -> None:
        if not np.isfinite(t0_ms):
            raise ValueError("Time origin must be finite")
        self.voltage = np.zeros((self.config.cells, self.config.channels), dtype=np.float64)
        self.updated_ms = np.full(self.config.cells, float(t0_ms), dtype=np.float64)
        self.watermark_ms = float(t0_ms)
        self.ledger = OperationLedger()

    def snapshot(self, time_ms: Optional[float] = None) -> np.ndarray:
        """Read-only diagnostic materialization, excluded from inference counters."""
        time_ms = self.watermark_ms if time_ms is None else float(time_ms)
        if not np.isfinite(time_ms) or time_ms < self.watermark_ms:
            raise ValueError("Snapshot cannot precede the processed prefix")
        return self.voltage * np.exp(-(time_ms - self.updated_ms[:, None]) / self.config.tau)

    def step(self, time_ms: float, currents: Mapping[int, np.ndarray],
             routes: Optional[Mapping[int, int]] = None) -> dict:
        if not np.isfinite(time_ms) or time_ms < self.watermark_ms:
            raise ValueError("Event times must be finite and nondecreasing")
        routes = self.config.validate_routes({} if routes is None else routes)
        checked = {}
        for cell, current in currents.items():
            cell = self.config.validate_cell(cell)
            current = np.asarray(current, dtype=np.float64)
            if current.shape != (self.config.channels,) or not np.all(np.isfinite(current)):
                raise ValueError("Current must be a finite vector of channel width")
            checked[cell] = current.copy()
        touched = sorted(set(checked) | set(routes) | set(routes.values()))
        if not touched:
            self.watermark_ms = float(time_ms)
            return {}
        c, local = self.config.channels, self.config.local_channels
        positions = {cell: row for row, cell in enumerate(touched)}
        elapsed = float(time_ms) - self.updated_ms[touched]
        value = self.voltage[touched].copy()
        decayed = elapsed > 0
        value[decayed] *= np.exp(-elapsed[decayed, None] / self.config.tau)
        moving = [(positions[s], positions[d]) for s, d in routes.items() if s != d]
        if self.mode == "membrane_route" and moving and local < c:
            original = value[:, local:].copy()
            for source, _ in moving:
                value[source, local:] = 0.0
            for source, destination in moving:
                value[destination, local:] += original[source]
        for cell, current in checked.items():
            value[positions[cell]] += current
        count = np.floor(np.maximum(value / self.config.threshold, 0.0))
        if not np.all(np.isfinite(value)) or np.sum(count, dtype=np.float64) >= 2**63:
            raise ValueError("Nonfinite state or unrepresentable spike burst")
        spikes = count.astype(np.int64)
        residual = value - self.config.threshold * count
        emitted = spikes.copy()
        if self.mode == "spike_route" and moving and local < c:
            original = spikes[:, local:].copy()
            for source, _ in moving:
                emitted[source, local:] = 0
            for source, destination in moving:
                emitted[destination, local:] += original[source]
        # Commit only after validating the entire update. Snapshot routing handles
        # cycles and collisions without letting newly received state move twice.
        self.voltage[touched] = residual
        self.updated_ms[touched] = time_ms
        self.watermark_ms = float(time_ms)
        n, d = len(touched), int(decayed.sum())
        moved = len(moving) * (c - local) if self.mode != "fixed" else 0
        self.ledger.add(ac=int(len(checked)*c + n*c + moved), scalar_mul=int(d*c+n*c),
                        scalar_div=int(d*c+n*c), exp=int(d*c), floor=int(n*c),
                        comparisons=int(n*c), state_scalar_updates=int(n*c),
                        state_read_bytes=int(n*(c+1)*8), state_write_bytes=int(n*(c+1)*8),
                        routed_scalars=int(moved), route_checks=int(len(routes)),
                        unit_spikes=int(spikes.sum()))
        return {cell: emitted[row].copy() for row, cell in enumerate(touched) if np.any(emitted[row])}

    def cost_report(self) -> dict:
        report = self.ledger.report("sparse LIF state bank and supplied-route execution",
                                    ["event encoder", "learned route selection", "Proposal", "Refiner readout"])
        report["persistent_state_bytes"] = int(self.voltage.nbytes + self.updated_ms.nbytes)
        report["state_dtype"] = "float64 numerical reference"
        return report


def mechanism_demo() -> dict:
    """A supplied-motion counterexample, never an EV-UAV performance result."""
    config = TransportConfig(rows=1, columns=3, channels=1, local_channels=0,
                             tau_long_ms=float(-10 / np.log(0.9)))
    results = {}
    for mode in MODES:
        model = SparseTransportLIF(config, mode)
        outputs = []
        for k in range(3):
            pulses = model.step(k * 10.0, {k: np.array([0.4])}, {} if k == 0 else {k-1: k})
            outputs.append({str(cell): count.tolist() for cell, count in pulses.items()})
        results[mode] = {"outputs": outputs, "residual": model.snapshot().tolist(),
                         "cost": model.cost_report()}
    return {"schema_version": 1, "experiment_id": "threshold_order_counterexample",
            "status": "synthetic_mechanism_only", "stage": "mechanism_reference",
            "data_role": "synthetic", "primary_metrics": {m: r["cost"]["counts"]["unit_spikes"] for m, r in results.items()},
            "decision": "not_an_accuracy_or_efficiency_acceptance", "config": asdict(config),
            "training_started": False, "supplied_routes_use_ground_truth": True,
            "results": results,
            "conclusion": "Shows noncommuting threshold/transport under correct supplied motion only."}
