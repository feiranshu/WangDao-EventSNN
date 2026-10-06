"""Causal 25 ms event representation and independent recurrent ANN/SNN models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import validate_events
from .snn import LIFCell

ARMS = ("Local-ANN", "Local-SNN", "Dilated-ANN", "Dilated-SNN")
HEIGHT, WIDTH, PAD_WIDTH = 260, 346, 348
CELLS = HEIGHT * PAD_WIDTH


@dataclass
class SparseSequence:
    events: np.ndarray
    bins: np.ndarray
    keys: np.ndarray
    counts: np.ndarray
    last: np.ndarray

    @property
    def blocks(self) -> np.ndarray:
        return np.unique(self.bins // 4)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.events, self.bins, self.keys, self.counts, self.last))

    def freeze(self) -> None:
        for array in (self.events, self.bins, self.keys, self.counts, self.last):
            array.flags.writeable = False

    def block(self, block: int, *, out: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if block < 0:
            raise ValueError("negative state block")
        if out is None:
            out = np.empty((4, 64, 65, 87), dtype=np.float32)
        if out.shape != (4, 64, 65, 87) or out.dtype != np.float32 or not out.flags.writeable:
            raise ValueError("block destination must be writable float32 [4,64,65,87]")
        out.fill(0)
        low, high = np.searchsorted(self.keys, [block * 8 * CELLS, (block + 1) * 8 * CELLS])
        keys = self.keys[low:high]
        micro, remainder = np.divmod(keys, 2 * CELLS)
        polarity, pixel = np.divmod(remainder, CELLS)
        yy, xx = np.divmod(pixel, PAD_WIDTH)
        step = micro % 4
        # Pixel-unshuffle maps (channel,y,x) to (channel*16+(y%4)*4+x%4,y//4,x//4).
        # Scatter straight into the final batch view, avoiding dense-image and stacking copies.
        channel = polarity * 16 + (yy % 4) * 4 + xx % 4
        out[step, channel, yy // 4, xx // 4] = self.counts[low:high].astype(np.float32) / 255.0
        out[step, channel + 32, yy // 4, xx // 4] = self.last[low:high]
        begin, end = np.searchsorted(self.bins, [block * 4, (block + 1) * 4])
        return out, np.arange(begin, end, dtype=np.int64)


def pack_blocks(blocks: list[tuple[SparseSequence, int]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Allocate one owned batch; each block writes into its strided batch slice."""
    packed = np.empty((4, len(blocks), 64, 65, 87), dtype=np.float32)
    indices = [record.block(block, out=packed[:, batch])[1] for batch, (record, block) in enumerate(blocks)]
    return packed, indices


def represent(events: np.ndarray) -> SparseSequence:
    validate_events(events)
    if not np.isfinite(events["t"]).all():
        raise ValueError("nonfinite timestamp")
    relative = events["t"].astype(np.float64) - float(events["t"][0])
    bins = np.floor(relative / 25.0).astype(np.int64)
    keys = ((bins * 2 + events["p"].astype(np.int64)) * CELLS
            + events["y"].astype(np.int64) * PAD_WIDTH + events["x"].astype(np.int64))
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    last = np.zeros(len(unique), dtype=np.float64)
    np.maximum.at(last, inverse, (relative - bins * 25.0) / 25.0)
    compressed = np.clip(np.rint(32.0 * np.log1p(counts)), 0, 255).astype(np.uint8)
    result = SparseSequence(events.copy(), bins, unique, compressed, last.astype(np.float32))
    result.freeze()
    return result


def collision_counts(record: SparseSequence) -> tuple[np.ndarray, np.ndarray]:
    e = record.events
    keys = (record.bins * 2 + e["p"].astype(np.int64)) * CELLS + e["y"].astype(np.int64) * PAD_WIDTH + e["x"]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    positive = np.asarray(np.bincount(inverse, weights=e["label"], minlength=len(counts))).astype(np.int64)
    return positive, counts - positive


def collision_summary(positive: np.ndarray, negative: np.ndarray) -> dict:
    if positive.shape != negative.shape or np.any(positive < 0) or np.any(negative < 0):
        raise ValueError("invalid group counts")
    total = int(positive.sum())
    # The optimal IoU subset is a prefix ordered by group purity, not majority vote.
    order = np.argsort(-positive / np.maximum(positive + negative, 1), kind="stable")
    tp, fp = np.cumsum(positive[order]), np.cumsum(negative[order])
    upper = float(np.max(tp / np.maximum(total + fp, 1))) if total else 1.0
    conflict = (positive > 0) & (negative > 0)
    return {"groups": len(positive), "mixed_groups": int(conflict.sum()),
            "events_in_mixed_groups": int((positive[conflict] + negative[conflict]).sum()),
            "oracle_event_iou": upper, "oracle_uses_labels": True}


class TemporalOutput(NamedTuple):
    logits: torch.Tensor
    state: tuple[torch.Tensor, ...]
    rates: torch.Tensor


class TemporalSegmenter(nn.Module):
    def __init__(self, arm: str, seed: int = 42, priors: tuple[float, float] = (0.03, 0.03),
                 *, head_std: float = 0.01, readout: str = "instant") -> None:
        super().__init__()
        if arm not in ARMS or any(not 0 < p < 1 for p in priors):
            raise ValueError("invalid architecture or foreground priors")
        self.arm = arm
        self.spiking = arm.endswith("SNN")
        if head_std <= 0 or not np.isfinite(head_std) or readout not in ("instant", "membrane"):
            raise ValueError("invalid readout configuration")
        if readout == "membrane" and not self.spiking:
            raise ValueError("membrane readout study is restricted to SNN")
        self.readout = readout
        self.head_std = head_std
        dilation = 2 if arm.startswith("Dilated") else 1
        # Local RNG scope makes pair initialization independent of arm execution order.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.ModuleList([nn.Conv2d(64, 8, 1, bias=False),
                                         nn.Conv2d(8, 8, 3, padding=1, bias=False),
                                         nn.Conv2d(8, 8, 3, padding=dilation, dilation=dilation, bias=False)])
            self.head = nn.Conv2d(8, 32, 1)
            for layer in self.hidden:
                assert isinstance(layer, nn.Conv2d)
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.normal_(self.head.weight, std=head_std)
            with torch.no_grad():
                assert self.head.bias is not None
                self.head.bias.copy_(torch.logit(torch.tensor(priors)).repeat_interleave(16))
        self.cell = LIFCell(beta=0.875, threshold=0.25, surrogate_slope=10.0)

    def step(self, value: torch.Tensor, state: tuple[torch.Tensor, ...] | None = None) -> TemporalOutput:
        if value.ndim != 4 or tuple(value.shape[1:]) != (64, 65, 87):
            raise ValueError("step input must be [batch,64,65,87]")
        expected_states = 4 if self.readout == "membrane" else 3
        if state is not None and len(state) != expected_states:
            raise ValueError(f"{expected_states} explicit states required")
        states, rates = [], []
        for index, layer in enumerate(self.hidden):
            current = layer(value)
            previous = torch.zeros_like(current) if state is None else state[index]
            if previous.shape != current.shape:
                raise ValueError("state batch/shape mismatch")
            if self.spiking:
                value, membrane = self.cell(current, previous)
            else:
                value = F.relu(0.875 * previous + current)
                membrane = value
            states.append(membrane)
            rates.append(value.mean() if self.spiking else (value != 0).float().mean())
        if self.readout == "membrane":
            # Integrate only weighted spikes. A repeated prior bias would create temporal drift.
            current = F.conv2d(value, self.head.weight)
            previous = torch.zeros_like(current) if state is None else state[3]
            if previous.shape != current.shape:
                raise ValueError("readout state shape mismatch")
            output_membrane = .875 * previous + current
            states.append(output_membrane)
            assert self.head.bias is not None
            raw = output_membrane + self.head.bias[None, :, None, None]
        else:
            raw = self.head(value)
        logits = F.pixel_shuffle(raw, 4)[..., :WIDTH]
        return TemporalOutput(logits, tuple(states), torch.stack(rates))

    def forward(self, block: torch.Tensor) -> TemporalOutput:
        if block.ndim != 5 or block.shape[0] != 4:
            raise ValueError("block must be [4,batch,64,65,87]; state resets on every call")
        state = None
        outputs, rates = [], []
        for value in block:
            result = self.step(value, state)
            state = result.state
            outputs.append(result.logits)
            rates.append(result.rates)
        return TemporalOutput(torch.stack(outputs), result.state, torch.stack(rates).mean(0))


def event_loss(logits: torch.Tensor, labels: torch.Tensor, rates: torch.Tensor, spiking: bool) -> torch.Tensor:
    if logits.ndim != 1 or labels.shape != logits.shape or not len(labels):
        raise ValueError("loss requires aligned nonempty individual events")
    probability = logits.sigmoid()
    pt = torch.where(labels > 0.5, probability, 1.0 - probability)
    alpha = torch.where(labels > 0.5, 0.75, 0.25)
    focal = (alpha * (1.0 - pt).square() * F.binary_cross_entropy_with_logits(logits, labels, reduction="none")).mean()
    intersection = (probability * labels).sum()
    union = (probability + labels - probability * labels).sum()
    loss = focal + 0.5 * (1.0 - (intersection + 1e-6) / (union + 1e-6))
    if spiking:
        loss = loss + 0.01 * (F.relu(0.01 - rates).square() + F.relu(rates - 0.25).square()).mean()
    return loss


def structural_cost(spiking: bool) -> dict:
    pixels = 65 * 87
    return {"parameters": 1952, "weight_values": 1920, "bias_values": 32,
            "dense_products_per_100ms": pixels * 1920 * 4,
            "nonbinary_synaptic_products_per_100ms": pixels * (512 if spiking else 1920) * 4,
            "scheduled_binary_weight_accumulations_per_100ms": pixels * 1408 * 4 if spiking else 0,
            "hidden_updates_per_100ms": pixels * 8 * 3 * 4,
            "hidden_state_values": pixels * 8 * 3, "hidden_state_float32_bytes": pixels * 8 * 3 * 4,
            "state_24bit_bytes_hypothetical": pixels * 8 * 3 * 3,
            "scope": "dense schedule includes padded convolution taps; no FPGA/energy claim"}
