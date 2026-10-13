"""Matched research ANN and event-time, grid-addressed LIF Refiner for protocol A."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTH, HEIGHT = 87, 65
CELLS = WIDTH * HEIGHT
CHANNELS = (48, 32, 4)
STATE_CHANNELS = sum(CHANNELS) + 4
THRESHOLD = 0.5


def class_logits(prototypes: torch.Tensor) -> torch.Tensor:
    return prototypes.reshape(-1, 2, 2).amax(dim=2)


class ResearchANN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(36, 48, bias=False)
        self.fc2 = nn.Linear(48, 32, bias=False)
        self.fc3 = nn.Linear(32, 4, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.hardtanh(self.fc1(features), 0.0, 1.0)
        x = F.hardtanh(self.fc2(x), 0.0, 1.0)
        return self.fc3(x)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        return gradient * torch.clamp(1.0 - value.abs(), min=0.0)


class ResearchSNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(36, 48, bias=False)
        self.fc2 = nn.Linear(48, 32, bias=False)
        self.fc3 = nn.Linear(32, 4, bias=False)
        self.trace_gain = nn.Parameter(torch.full((4,), 0.25))
        self.mem_gain = nn.Parameter(torch.full((4,), 0.25))
        self.readout_bias = nn.Parameter(torch.zeros(4))
        self.register_buffer("tau1", torch.tensor([5.0] * 16 + [25.0] * 16 + [100.0] * 16))
        self.register_buffer("tau2", torch.tensor([25.0] * 32))
        self.register_buffer("tau3", torch.tensor([100.0] * 4))
        self.register_buffer("tau_trace", torch.tensor([100.0] * 4))
        for layer, scale in zip((self.fc1, self.fc2, self.fc3), (1.0, 0.65, 0.35)):
            nn.init.xavier_uniform_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(scale)

    def step(self, features: torch.Tensor, dt: torch.Tensor,
             state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        u1, u2, u3, trace = state
        dt = dt[:, None]
        p1 = u1 * torch.exp(-dt / self.tau1) + self.fc1(features)
        s1 = SurrogateSpike.apply(p1 - THRESHOLD)
        v1 = p1 - THRESHOLD * s1
        p2 = u2 * torch.exp(-dt / self.tau2) + self.fc2(s1)
        s2 = SurrogateSpike.apply(p2 - THRESHOLD)
        v2 = p2 - THRESHOLD * s2
        p3 = u3 * torch.exp(-dt / self.tau3) + self.fc3(s2)
        s3 = SurrogateSpike.apply(p3 - THRESHOLD)
        v3 = p3 - THRESHOLD * s3
        tr = trace * torch.exp(-dt / self.tau_trace) + s3
        logits = self.trace_gain * tr + self.mem_gain * (v3 / THRESHOLD) + self.readout_bias
        return logits, (v1, v2, v3, tr), (s1, s2, s3)


@dataclass
class WindowPack:
    cells: np.ndarray
    event_slots: np.ndarray
    delta_ms: np.ndarray
    count: int


class SNNRunner:
    """Independent sequence state; exact per-cell event order, lazy real-time decay."""
    def __init__(self, model: ResearchSNN, t0: float = 0.0):
        self.model = model
        self.reset(t0)

    def reset(self, t0: float) -> None:
        if not math.isfinite(t0):
            raise ValueError("t0 must be finite")
        self.t0 = float(t0)
        self.state = [np.zeros((CELLS, n), dtype=np.float32) for n in (*CHANNELS, 4)]
        self.last_time = np.full(CELLS, self.t0, dtype=np.float64)
        self.last_event_time = -np.inf
        self.counts = {"events": 0, "input_nonzero": 0, "spikes_l1": 0,
                       "spikes_l2": 0, "spikes_out": 0, "positive_dt": 0,
                       "state_channel_reads": 0, "state_channel_writes": 0}

    def pack(self, features: np.ndarray, x: np.ndarray, y: np.ndarray,
             t: np.ndarray) -> WindowPack:
        n = len(features)
        if features.shape != (n, 36) or features.dtype != np.uint8:
            raise ValueError("features must be uint8[N,36]")
        if not (len(x) == len(y) == len(t) == n):
            raise ValueError("event columns differ in length")
        if n == 0:
            return WindowPack(np.empty(0, np.int64), np.empty((0, 0), np.int64),
                              np.empty((0, 0), np.float32), 0)
        if (np.any(x < 0) or np.any(x >= 346) or np.any(y < 0) or np.any(y >= 260)
                or not np.all(np.isfinite(t)) or t[0] < self.t0
                or t[0] < self.last_event_time or np.any(np.diff(t) < 0)):
            raise ValueError("invalid or unsorted event stream")
        self.last_event_time = float(t[-1])
        address = (y.astype(np.int64) // 4) * WIDTH + x.astype(np.int64) // 4
        order = np.lexsort((np.arange(n), address))
        sorted_address = address[order]
        cells, start, lengths = np.unique(sorted_address, return_index=True, return_counts=True)
        column = np.repeat(np.arange(len(cells)), lengths)
        row = np.arange(n) - np.repeat(start, lengths)
        slots = np.full((int(lengths.max()), len(cells)), -1, dtype=np.int64)
        slots[row, column] = order
        delta = np.zeros(slots.shape, dtype=np.float32)
        previous = self.last_time[cells].copy()
        for k in range(len(slots)):
            valid = slots[k] >= 0
            now = t[slots[k, valid]]
            difference = now - previous[valid]
            if np.any(difference < -1e-9):
                raise ValueError("cell time moved backward")
            delta[k, valid] = difference.astype(np.float32)
            previous[valid] = now
        self.last_time[cells] = previous
        return WindowPack(cells, slots, delta, n)

    def run_window(self, features: np.ndarray, x: np.ndarray, y: np.ndarray,
                   t: np.ndarray, labels: np.ndarray | None = None,
                   class_weight: torch.Tensor | None = None) -> tuple[torch.Tensor | None, np.ndarray | None]:
        pack = self.pack(features, x, y, t)
        if pack.count == 0:
            return None, np.empty(0, dtype=np.float32) if labels is None else None
        device = self.model.fc1.weight.device
        state = tuple(torch.from_numpy(item[pack.cells].copy()).to(device) for item in self.state)
        result = np.empty(pack.count, dtype=np.float32) if labels is None else None
        loss = None
        for k in range(len(pack.event_slots)):
            valid = pack.event_slots[k] >= 0
            safe = np.maximum(pack.event_slots[k], 0)
            inp = torch.from_numpy(features[safe].astype(np.float32) / 255.0).to(device)
            dt = torch.from_numpy(pack.delta_ms[k]).to(device)
            logits, updated, spikes = self.model.step(inp, dt, state)
            mask = torch.from_numpy(valid).to(device)[:, None]
            state = tuple(torch.where(mask, new, old) for new, old in zip(updated, state))
            if labels is not None:
                target = torch.from_numpy(labels[pack.event_slots[k, valid]].astype(np.int64)).to(device)
                current = F.cross_entropy(class_logits(logits[mask[:, 0]]), target,
                                          weight=class_weight, reduction="sum")
                loss = current if loss is None else loss + current
            else:
                with torch.no_grad():
                    prob = torch.softmax(class_logits(logits[mask[:, 0]]), dim=1)[:, 1]
                    result[pack.event_slots[k, valid]] = prob.cpu().numpy()
            self.counts["spikes_l1"] += int(spikes[0][mask[:, 0]].detach().sum().item())
            self.counts["spikes_l2"] += int(spikes[1][mask[:, 0]].detach().sum().item())
            self.counts["spikes_out"] += int(spikes[2][mask[:, 0]].detach().sum().item())
        for destination, values in zip(self.state, state):
            destination[pack.cells] = values.detach().cpu().numpy()
        self.counts["events"] += pack.count
        self.counts["input_nonzero"] += int(np.count_nonzero(features))
        self.counts["positive_dt"] += int(np.count_nonzero(pack.delta_ms))
        self.counts["state_channel_reads"] += pack.count * STATE_CHANNELS
        self.counts["state_channel_writes"] += pack.count * STATE_CHANNELS
        return loss, result

    def cost(self) -> dict:
        n = self.counts["events"]
        return {**self.counts,
                "refiner_dense_mac": n * (36 * 48 + 48 * 32 + 32 * 4),
                "refiner_effective_input_mac": self.counts["input_nonzero"] * 48,
                "refiner_binary_ac": self.counts["spikes_l1"] * 32 + self.counts["spikes_l2"] * 4,
                "voltage_and_trace_bytes": CELLS * STATE_CHANNELS * 4,
                "timestamp_bytes": CELLS * 8,
                "state_decay_multiplications": self.counts["positive_dt"] * STATE_CHANNELS,
                "threshold_comparisons": n * sum(CHANNELS),
                "readout_multiplications": n * 8}
