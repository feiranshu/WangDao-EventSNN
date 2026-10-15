"""Phase-preserving, stream-state SNN Refiner; dense CUDA training reference.

One step consumes a CLOSED 10 ms bin. The frozen Proposal is available only at
100 ms closure. Neither the binning nor GPU batching implies event-time latency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .torch_cell import burst
from .transport import MODES

OFFSETS = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1))


@dataclass(frozen=True)
class RefinerConfig:
    version: str = "phase_grid_refiner_v1"
    rows: int = 68
    columns: int = 88
    micro_ms: int = 10
    stem_channels: int = 16
    hidden_channels: int = 32
    local_channels: int = 16
    threshold: float = 0.25
    tau_short_ms: float = 20.0
    tau_long_ms: float = 100.0
    readout_tau_ms: float = 20.0
    mode: str = "fixed"

    def __post_init__(self):
        if self.version != "phase_grid_refiner_v1" or self.mode not in MODES:
            raise ValueError("Unsupported Refiner version/mode")
        if self.micro_ms != 10 or min(self.rows, self.columns, self.stem_channels) < 1:
            raise ValueError("Invalid spatial shape or bin duration")
        if not 0 < self.local_channels < self.hidden_channels:
            raise ValueError("Require local and transportable hidden channels")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.threshold, self.tau_short_ms, self.tau_long_ms, self.readout_tau_ms)):
            raise ValueError("Threshold and time constants must be finite and positive")


def shift(value, dy, dx):
    """Move a source image by an integer offset, with zero outside the grid."""
    height, width = value.shape[-2:]
    return F.pad(value, (1, 1, 1, 1))[..., 1-dy:1-dy+height, 1-dx:1-dx+width]


@torch.no_grad()
def activity_routes(previous, current):
    """Label-free *reference* router, not a learned optical-flow estimator.

    An occupied source follows the strongest current 3x3 activity if it exceeds
    the stationary score by 25%. Ties and uncertain/empty sources stay. Every
    source has one owner. This heuristic is an ablation, not validated motion.
    """
    scores = torch.cat([shift(current, -dy, -dx) for dy, dx in OFFSETS], dim=1)
    best, route = scores.max(dim=1, keepdim=True)
    stationary = scores[:, 4:5]
    unique = (scores == best).sum(dim=1, keepdim=True) == 1
    move = (previous > 0) & (best > 0) & (best > stationary * 1.25) & unique
    return torch.where(move, route, torch.full_like(route, 4))


def transport_grid(value, routes):
    """Deterministic fixed-order gather/sum; no nondeterministic CUDA scatter-add."""
    result = torch.zeros_like(value)
    for index, (dy, dx) in enumerate(OFFSETS):
        result = result + shift(value * (routes == index).to(value.dtype), dy, dx)
    return result


class StreamRefiner(nn.Module):
    """64 phase/polarity input channels -> 16 LIF -> 32 LIF -> 4 prototypes/query.

    All 32 (polarity, 4x4 pixel phase) queries have their own four output LIF
    states: 128 readout channels per grid cell. This avoids forcing all pixels
    in a 4x4 cell to share one probability. Synaptic convs are dense emulation;
    MAC/AC deployment proxies must be reported separately from actual execution.
    """
    def __init__(self, config=RefinerConfig(), foreground_prior=0.1):
        super().__init__()
        if not 0 < foreground_prior < 1:
            raise ValueError("Foreground prior must lie in (0,1)")
        self.config = config
        self.stem = nn.Conv2d(64, config.stem_channels, 1, bias=False)
        self.hidden = nn.Conv2d(config.stem_channels, config.hidden_channels, 3, padding=1, bias=False)
        self.head = nn.Conv2d(config.hidden_channels, 128, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(32, 4))
        with torch.no_grad():
            self.bias[:, 2:] = math.log(foreground_prior / (1-foreground_prior))
        for layer in (self.stem, self.hidden, self.head):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        tau = [config.tau_short_ms]*config.local_channels + [config.tau_long_ms]*(
            config.hidden_channels-config.local_channels)
        self.register_buffer("hidden_decay", torch.exp(-config.micro_ms / torch.tensor(tau))[None, :, None, None])
        self.stem_decay = math.exp(-config.micro_ms/config.tau_short_ms)
        self.head_decay = math.exp(-config.micro_ms/config.readout_tau_ms)

    def initial_state(self, batch, *, device=None, dtype=None):
        weight = self.stem.weight
        kwargs = {"device": device or weight.device, "dtype": dtype or weight.dtype}
        shape = (batch, self.config.rows, self.config.columns)
        return tuple(torch.zeros(shape[0], channels, *shape[1:], **kwargs)
                     for channels in (self.config.stem_channels, self.config.hidden_channels, 128, 1))

    @staticmethod
    def detach_state(state):
        return tuple(value.detach() for value in state)

    @staticmethod
    def reset_slots(state, reset):
        keep = (~reset).to(state[0]).view(-1, 1, 1, 1)
        return tuple(value * keep for value in state)

    def _lif(self, voltage):
        spikes = burst(voltage, self.config.threshold)
        return spikes, voltage - spikes*self.config.threshold

    def forward(self, inputs, state=None):
        c = self.config
        if (inputs.ndim != 5 or inputs.shape[2:] != (64, c.rows, c.columns)
                or inputs.shape[0] < 1 or inputs.shape[1] < 1 or inputs.dtype != self.stem.weight.dtype):
            raise ValueError("Expected FP32 [steps,batch,64,rows,columns]")
        state = self.initial_state(inputs.shape[1]) if state is None else state
        expected = (c.stem_channels, c.hidden_channels, 128, 1)
        if len(state) != 4 or any(s.shape != (inputs.shape[1], n, c.rows, c.columns)
                                 for s, n in zip(state, expected)):
            raise ValueError("Stream state shape mismatch")
        stem_v, hidden_v, head_v, previous = state
        logits, rates, costs = [], [], []
        for frame in inputs:
            activity = frame[:, :32].sum(1, keepdim=True)
            # No bias: zero input has zero current and zero weight gradient.
            # Project only occupied cells; this is an implemented MAC saving.
            flat = frame.permute(0, 2, 3, 1).reshape(-1, 64)
            indices = (flat != 0).any(1).nonzero(as_tuple=True)[0]
            projected = F.linear(flat[indices], self.stem.weight[:, :, 0, 0])
            full = frame.new_zeros((len(flat), c.stem_channels)).index_copy(0, indices, projected)
            current = full.view(inputs.shape[1], c.rows, c.columns, -1).permute(0, 3, 1, 2)
            stem_s, stem_v = self._lif(stem_v*self.stem_decay + current)
            decayed = hidden_v*self.hidden_decay
            routes = None if c.mode == "fixed" else activity_routes(previous, activity)
            if c.mode == "membrane_route":
                decayed = torch.cat((decayed[:, :c.local_channels],
                                     transport_grid(decayed[:, c.local_channels:], routes)), 1)
            hidden_s, hidden_v = self._lif(decayed + self.hidden(stem_s))
            if c.mode == "spike_route":
                hidden_s = torch.cat((hidden_s[:, :c.local_channels],
                                      transport_grid(hidden_s[:, c.local_channels:], routes)), 1)
            head_pre = head_v*self.head_decay + self.head(hidden_s)
            _, head_v = self._lif(head_pre)
            # The four LIF voltages *before* reset form two max-pooled prototypes
            # per class. Bias is applied once, never integrated into the state.
            raw = head_pre.reshape(inputs.shape[1], 32, 4, c.rows, c.columns)
            raw = raw + self.bias[None, :, :, None, None]
            grouped = raw.reshape(inputs.shape[1], 32, 2, 2, c.rows, c.columns).amax(3)
            logits.append(grouped[:, :, 1]-grouped[:, :, 0])
            support = F.max_pool2d((activity > 0).to(frame.dtype), 3, 1, 1)
            denominator = support.sum((1, 2, 3)).clamp_min(1)
            rates.append(torch.stack(((stem_s*support).sum((1, 2, 3))/(denominator*c.stem_channels),
                                      (hidden_s*support).sum((1, 2, 3))/(denominator*c.hidden_channels)), 1))
            # Per-stream counts retained on device until the caller summarizes.
            costs.append(torch.stack(((activity > 0).sum((1, 2, 3)),
                                      (frame != 0).sum((1, 2, 3)),
                                      stem_s.detach().sum((1, 2, 3)),
                                      hidden_s.detach().sum((1, 2, 3)),
                                      (routes != 4).sum((1, 2, 3)) if routes is not None
                                      else frame.new_zeros(inputs.shape[1])), 1))
            previous = activity.detach()
        return (torch.stack(logits), (stem_v, hidden_v, head_v, previous),
                torch.stack(rates), torch.stack(costs))

    def architecture(self):
        return {"config": asdict(self.config), "parameters": sum(p.numel() for p in self.parameters()),
                "state_scalars_per_stream": self.config.rows*self.config.columns*(
                    self.config.stem_channels+self.config.hidden_channels+128+1),
                "readout": "4 LIF prototypes per pixel-phase/polarity query; pre-reset voltage + bias",
                "router": "causal activity heuristic; no learned/oracle motion",
                "backend": "active-cell stem, dense convolutional spike emulation"}
