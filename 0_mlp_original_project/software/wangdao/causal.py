"""Versioned, label-free 25 ms streaming representation and causal ANN ablations.

An output at a block end may use every event in that half-open block, never an
event from the next block. These models are newly initialized research models;
they do not replace, or inherit accuracy claims from, the frozen H2/S1 exports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .h2.constants import PADDED_HEIGHT, PADDED_WIDTH, SENSOR_HEIGHT, SENSOR_WIDTH
from .h2.torch_model import prototype_logits


CAUSAL_FORMAT_VERSION = 1
FEATURE_VERSION = "causal_block_v1"
FINE_SHAPE = (PADDED_HEIGHT // 4, PADDED_WIDTH // 4)
COARSE_SHAPE = (PADDED_HEIGHT // 8, PADDED_WIDTH // 8)
EMPTY_EVENTS = np.empty((0, 4), dtype=np.float64)


@dataclass(frozen=True)
class CausalConfig:
    variant: str = "C0"
    block_ms: float = 25.0
    proposal_threshold: float = 0.51
    event_threshold: float = 0.28
    routing_margin: float = 0.2
    temperature: float = 1.0
    no_memory: bool = False
    mlp_hidden: Tuple[int, ...] = (64, 32)
    sparse_count: int = 8

    def __post_init__(self) -> None:
        if self.variant not in ("C0", "C1", "C2", "C3", "C4"):
            raise ValueError("variant must be C0, C1, C2, C3 or C4")
        if self.block_ms != 25.0:
            raise ValueError("causal_block_v1 fixes blocks to 25 ms and horizons to 25/50/100 ms")
        for name in ("proposal_threshold", "event_threshold"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(name + " must be in [0, 1]")
        if not 0 <= self.routing_margin <= 0.5:
            raise ValueError("routing_margin must be in [0, 0.5]")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not self.mlp_hidden or any(int(n) <= 0 for n in self.mlp_hidden):
            raise ValueError("mlp_hidden must contain positive dimensions")
        if self.sparse_count < 0:
            raise ValueError("sparse_count must be nonnegative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalBlock:
    event_indices: np.ndarray
    events: np.ndarray
    event_features: np.ndarray
    spatial: np.ndarray
    event_density: np.ndarray
    start_ms: float
    end_ms: float

    @property
    def cutoff_ms(self) -> float:
        return self.end_ms

    def to(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        return {
            "event_indices": torch.as_tensor(self.event_indices, dtype=torch.long, device=device),
            "events": torch.as_tensor(self.events, dtype=torch.float64, device=device),
            "event_features": torch.as_tensor(self.event_features, dtype=torch.float32, device=device),
            "spatial": torch.as_tensor(self.spatial, dtype=torch.float32, device=device),
            "event_density": torch.as_tensor(self.event_density, dtype=torch.long, device=device),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "cutoff_ms": self.end_ms,
        }


def _count_features(counts: np.ndarray) -> np.ndarray:
    return (np.clip(np.rint(32.0 * np.log1p(counts)), 0, 255) / 255.0).astype(np.float32)


def _validate_events(events: np.ndarray) -> np.ndarray:
    raw = np.asarray(events)
    if raw.dtype.names is not None or raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("inference accepts only a numeric Nx4 x/y/t_ms/p array; strip labels explicitly")
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("events must be finite")
    if len(values):
        xy_p = values[:, (0, 1, 3)]
        if not np.equal(xy_p, np.floor(xy_p)).all():
            raise ValueError("coordinates and polarity must be integral")
        if ((values[:, 0] < 0) | (values[:, 0] >= SENSOR_WIDTH)).any():
            raise ValueError("x is outside the sensor")
        if ((values[:, 1] < 0) | (values[:, 1] >= SENSOR_HEIGHT)).any():
            raise ValueError("y is outside the sensor")
        if not np.isin(values[:, 3], (0, 1)).all():
            raise ValueError("polarity must be 0 or 1")
        if (np.diff(values[:, 2]) < 0).any():
            raise ValueError("events must be sorted by timestamp")
    return values


class FeatureStream:
    """Fixed-size count history plus an unfinished input block.

    ``cutoff_ms`` is a watermark: later calls must not introduce an event before
    it. Events exactly on the watermark are allowed because blocks are [a, b).
    A call cannot contain events later than its watermark. ``final=True`` means
    end of sequence; the last partial block is released at its usual 25 ms end.
    Input indices refer to arrival order and remain stable across fragmentation.
    """

    def __init__(self, config: Optional[CausalConfig] = None, start_ms: float = 0.0) -> None:
        self.config = config or CausalConfig()
        self.reset(start_ms)

    def reset(self, start_ms: float = 0.0) -> None:
        if not np.isfinite(start_ms):
            raise ValueError("start_ms must be finite")
        self.start_ms = float(start_ms)
        self._next_start = float(start_ms)
        self._watermark = float(start_ms)
        self._last_timestamp = float(start_ms)
        self._next_index = 0
        self._block_count = 0
        self._finished = False
        self._pending = EMPTY_EVENTS.copy()
        self._pending_indices = np.empty(0, dtype=np.int64)
        self._rings = {
            cell: np.zeros((4, 2, (PADDED_HEIGHT + cell - 1) // cell,
                            (PADDED_WIDTH + cell - 1) // cell), dtype=np.int64)
            for cell in (1, 2, 4, 8)
        }
        self._last_activity = np.full((2, *FINE_SHAPE), -np.inf, dtype=np.float64)

    @property
    def state_bytes(self) -> int:
        return int(sum(v.nbytes for v in self._rings.values()) + self._last_activity.nbytes
                   + self._pending.nbytes + self._pending_indices.nbytes)

    @property
    def pending_events(self) -> int:
        return len(self._pending)

    def push(self, events: np.ndarray, cutoff_ms: float, final: bool = False) -> List[CausalBlock]:
        values = _validate_events(events)
        cutoff = float(cutoff_ms)
        if self._finished:
            raise ValueError("stream is finalized; reset before a new sequence")
        if not np.isfinite(cutoff) or cutoff < self._watermark:
            raise ValueError("cutoff must be finite and nondecreasing")
        if len(values):
            if values[0, 2] < max(self._watermark, self._last_timestamp):
                raise ValueError("late or out-of-order events would revise a published block")
            if values[-1, 2] > cutoff:
                raise ValueError("an event exceeds the declared cutoff")
            new_indices = np.arange(self._next_index, self._next_index + len(values), dtype=np.int64)
            self._pending = np.concatenate((self._pending, values), axis=0)
            self._pending_indices = np.concatenate((self._pending_indices, new_indices))
            self._next_index += len(values)
            self._last_timestamp = float(values[-1, 2])
        self._watermark = cutoff
        output: List[CausalBlock] = []
        final_end = cutoff
        if final:
            elapsed = max(cutoff - self.start_ms, 0.0)
            final_end = self.start_ms + np.ceil(elapsed / self.config.block_ms) * self.config.block_ms
            if len(self._pending):
                event_end = (self.start_ms +
                             (np.floor((self._pending[-1, 2] - self.start_ms) / self.config.block_ms) + 1)
                             * self.config.block_ms)
                final_end = max(final_end, float(event_end))
        while self._next_start + self.config.block_ms <= final_end:
            end = self._next_start + self.config.block_ms
            split = int(np.searchsorted(self._pending[:, 2], end, side="left"))
            block_events = self._pending[:split].copy()
            block_indices = self._pending_indices[:split].copy()
            output.append(self._make_block(block_events, block_indices, self._next_start, end))
            self._pending = self._pending[split:]
            self._pending_indices = self._pending_indices[split:]
            self._next_start = end
        # A slice can otherwise keep a whole uploaded sequence alive behind a tiny tail.
        self._pending = self._pending.copy()
        self._pending_indices = self._pending_indices.copy()
        self._finished = bool(final)
        return output

    def _make_block(self, events: np.ndarray, indices: np.ndarray,
                    start: float, end: float) -> CausalBlock:
        slot = self._block_count % 4
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        p = events[:, 3].astype(np.int64)
        features = np.zeros((len(events), 32), dtype=np.float32)
        features[:, 0] = x / float(SENSOR_WIDTH - 1)
        features[:, 1] = y / float(SENSOR_HEIGHT - 1)
        features[:, 2] = (events[:, 2] - start) / self.config.block_ms
        features[:, 3] = p
        spatial_counts: List[np.ndarray] = []
        density = np.zeros(len(events), dtype=np.int64)
        column = 4
        for cell, ring in self._rings.items():
            ring[slot].fill(0)
            xc, yc = x // cell, y // cell
            np.add.at(ring[slot], (p, yc, xc), 1)
            for horizon in (1, 2, 4):
                slots = [(slot - lag) % 4 for lag in range(horizon)]
                counts = ring[slots].sum(axis=0, dtype=np.int64)
                features[:, column] = _count_features(counts[p, yc, xc])
                features[:, column + 1] = _count_features(counts[1 - p, yc, xc])
                column += 2
                if cell == 4:
                    spatial_counts.append(_count_features(counts))
                    if horizon == 1:
                        density = counts[:, yc, xc].sum(axis=0)
        np.maximum.at(self._last_activity, (p, y // 4, x // 4), events[:, 2])
        age = end - self._last_activity
        recency = np.maximum(1.0 - age / 100.0, 0.0).astype(np.float32)
        spatial = np.concatenate((*spatial_counts, recency), axis=0)
        features[:, 31] = features[:, 2]
        self._block_count += 1
        return CausalBlock(indices, events, features, spatial, density, start, end)


@dataclass
class CausalState:
    short: Optional[torch.Tensor] = None
    long: Optional[torch.Tensor] = None
    previous_probability: Optional[torch.Tensor] = None
    previous_seen: Optional[torch.Tensor] = None
    end_ms: Optional[float] = None

    def detach(self) -> "CausalState":
        return CausalState(*(
            None if value is None else value.detach()
            for value in (self.short, self.long, self.previous_probability, self.previous_seen)
        ), end_ms=self.end_ms)

    def reset(self) -> "CausalState":
        self.short = self.long = self.previous_probability = self.previous_seen = None
        self.end_ms = None
        return self

    @property
    def tensor_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in
                   (self.short, self.long, self.previous_probability, self.previous_seen)
                   if v is not None)


class _MLP(nn.Module):
    def __init__(self, dimensions: Tuple[int, ...]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.Hardtanh(0.0, 1.0))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _Residual(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mixed = self.pointwise(F.hardtanh(self.depthwise(value), 0.0, 1.0))
        return F.hardtanh(value + mixed, 0.0, 1.0)


class CausalModel(nn.Module):
    """C0-C4 ablations with explicit state and H2's two prototypes per class.

    Training may use ``force_deep=True`` to train classifiers on every event;
    reported probabilities still obey the hard, label-free Proposal gate.
    Production/evaluation uses the default ``force_deep=False``. Returned raw
    logits on unevaluated events are zero placeholders, not model predictions.
    """

    def __init__(self, config: Optional[CausalConfig] = None) -> None:
        super().__init__()
        self.config = config or CausalConfig()
        variant = self.config.variant
        if variant == "C0":
            self.proposal_mlp = _MLP((32, 48, 24, 1))
            self.light_head = _MLP((32, *tuple(self.config.mlp_hidden), 4))
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(8, 24, 3, padding=1), nn.Hardtanh(0.0, 1.0),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.Hardtanh(0.0, 1.0),
                _Residual(), _Residual(),
            )
            self.proposal_head = nn.Conv2d(32, 1, 1)
            if variant in ("C2", "C3", "C4") and not self.config.no_memory:
                self.memory_fusion = nn.Conv2d(96, 32, 1)
            if variant in ("C1", "C2", "C4"):
                self.light_head = _MLP((64, 32, 4))
            if variant in ("C3", "C4"):
                self.deep_head = _MLP((64, 128, 64, 4))

    def new_state(self) -> CausalState:
        return CausalState()

    def _target_probability(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(prototype_logits(logits) / self.config.temperature, dim=1)[:, 1]

    def forward(self, block: Union[CausalBlock, Dict[str, Any]],
                state: Optional[CausalState] = None, force_deep: bool = False) -> Dict[str, Any]:
        return self.forward_block(block, state, force_deep)

    def forward_block(self, block: Union[CausalBlock, Dict[str, Any]],
                      state: Optional[CausalState] = None,
                      force_deep: bool = False) -> Dict[str, Any]:
        if isinstance(block, CausalBlock):
            block = block.to(next(self.parameters()).device)
        required = {"events", "event_indices", "event_features", "spatial", "start_ms", "end_ms"}
        if not required.issubset(block):
            raise ValueError("block is missing required causal features")
        if "labels" in block or "label" in block or "target_ids" in block:
            raise ValueError("labels must remain outside the inference interface")
        state = CausalState() if state is None else state
        start, end = float(block["start_ms"]), float(block["end_ms"])
        if not np.isfinite(start + end) or not np.isclose(end - start, self.config.block_ms):
            raise ValueError("invalid block duration")
        if state.end_ms is not None and not np.isclose(start, state.end_ms):
            raise ValueError("state must advance through every consecutive block, including empty blocks")
        spatial = block["spatial"]
        events = block["events"]
        features = block["event_features"]
        if tuple(spatial.shape) != (8, *FINE_SHAPE) or tuple(features.shape) != (len(events), 32):
            raise ValueError("causal feature shape mismatch")
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("events must contain only x/y/t_ms/p")
        if len(events) and bool(((events[:, 2] < start) | (events[:, 2] >= end)).any()):
            raise ValueError("block contains an event outside [start, end)")
        if tuple(block["event_indices"].shape) != (len(events),):
            raise ValueError("one original event index is required per event")
        x = events[:, 0].long() // 8
        y = events[:, 1].long() // 8
        next_state = CausalState(end_ms=end)
        encoded: Optional[torch.Tensor] = None
        if self.config.variant == "C0":
            cells = F.unfold(spatial[None], kernel_size=2, stride=2).squeeze(0).t()
            proposal_logits = self.proposal_mlp(cells).reshape(COARSE_SHAPE)
        else:
            encoded = self.encoder(spatial[None])
            if hasattr(self, "memory_fusion"):
                short = encoded * 0.5 if state.short is None else state.short * 0.5 + encoded * 0.5
                long = encoded * 0.125 if state.long is None else state.long * 0.875 + encoded * 0.125
                next_state.short, next_state.long = short, long
                encoded = F.hardtanh(self.memory_fusion(torch.cat((encoded, short, long), dim=1)), 0.0, 1.0)
            proposal_logits = self.proposal_head(encoded)[0, 0]
        heatmap = torch.sigmoid(proposal_logits)
        neighborhood = F.avg_pool2d(F.pad(heatmap[None, None], (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0, 0]
        gate = (heatmap >= self.config.proposal_threshold).float()
        dilated = F.max_pool2d(gate[None, None], 3, stride=1, padding=1)[0, 0]
        coverage = F.avg_pool2d(dilated[None, None], 3, stride=1, padding=1)[0, 0]
        candidate = dilated[y, x].bool()
        context = torch.stack((heatmap[y, x], neighborhood[y, x], coverage[y, x], features[:, 2]), dim=1)
        event_features = torch.cat((features[:, :28], context), dim=1)
        if encoded is not None:
            event_features = torch.cat((event_features, encoded[0, :, y, x].t()), dim=1)
        active = torch.ones_like(candidate) if force_deep else candidate
        logits = event_features.new_zeros((len(events), 4))
        light_logits = torch.zeros_like(logits)
        deep_mask = torch.zeros_like(candidate)
        evaluated_light = torch.zeros_like(candidate)
        if self.config.variant == "C3":
            deep_mask = active
            logits = logits.index_copy(0, torch.where(active)[0], self.deep_head(event_features[active]))
        else:
            evaluated_light = active
            light_logits = light_logits.index_copy(0, torch.where(active)[0], self.light_head(event_features[active]))
            logits = light_logits
            if self.config.variant == "C4":
                light_probability = self._target_probability(light_logits)
                uncertain = (light_probability - 0.5).abs() <= self.config.routing_margin
                density = block.get("event_density")
                sparse = torch.zeros_like(candidate) if density is None else density <= self.config.sparse_count
                conflict = torch.zeros_like(candidate)
                if state.previous_probability is not None and state.previous_seen is not None:
                    conflict = state.previous_seen[y, x] & (
                        (state.previous_probability[y, x] >= self.config.event_threshold)
                        != (light_probability >= self.config.event_threshold))
                deep_mask = active if force_deep else candidate & (uncertain | sparse | conflict)
                logits = logits.index_copy(0, torch.where(deep_mask)[0], self.deep_head(event_features[deep_mask]))
        raw_probability = self._target_probability(logits)
        probability = torch.where(candidate, raw_probability, torch.zeros_like(raw_probability))
        if self.config.variant == "C4":
            flat_index = y * COARSE_SHAPE[1] + x
            total = probability.new_zeros(COARSE_SHAPE[0] * COARSE_SHAPE[1])
            count = torch.zeros_like(total)
            total = total.scatter_add(0, flat_index, probability.detach())
            count = count.scatter_add(0, flat_index, torch.ones_like(probability))
            next_state.previous_probability = (total / count.clamp_min(1)).reshape(COARSE_SHAPE)
            next_state.previous_seen = (count > 0).reshape(COARSE_SHAPE)
        return {
            "probabilities": probability, "logits": logits, "raw_probabilities": raw_probability,
            "light_logits": light_logits, "proposal_logits": proposal_logits,
            "proposal_probabilities": heatmap, "heatmap": heatmap,
            "candidate_mask": candidate, "deep_mask": deep_mask, "light_mask": evaluated_light,
            "event_indices": block["event_indices"], "event_features": event_features,
            "state": next_state, "output_time_ms": end,
        }

    def resource_summary(self, state: Optional[CausalState] = None) -> Dict[str, Any]:
        return {
            "variant": self.config.variant,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in self.parameters()),
            "model_state_bytes": 0 if state is None else state.tensor_bytes,
            "representation_state_bytes": FeatureStream(self.config).state_bytes,
            "feature_version": FEATURE_VERSION,
            "fine_grid": list(FINE_SHAPE), "coarse_grid": list(COARSE_SHAPE),
            "measurement_kind": "software_storage_accounting_not_power_or_latency",
        }
