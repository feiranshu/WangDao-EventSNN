"""Window batched S3 execution with state resident on the selected torch device.

S2's exact cell ordering and LIF equations are reused. A sequence keeps state
between windows in persistent mode, while window_reset discards all state at
each new 100 ms candidate window. Every window detaches the numerical state.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

S2 = Path(__file__).absolute().parent.parent / "s2"
sys.path.insert(0, str(S2))
from core import CELLS, CHANNELS, ResearchSNN, SNNRunner, class_logits  # noqa: E402


class DeviceSNNRunner:
    def __init__(self, model: ResearchSNN, t0: float, mode: str = "persistent"):
        if mode not in ("persistent", "window_reset"):
            raise ValueError(mode)
        self.model = model
        self.mode = mode
        self.device = model.fc1.weight.device
        self.packer = SNNRunner(model, t0)
        self.state = [torch.zeros((CELLS, channels), dtype=torch.float32, device=self.device)
                      for channels in (*CHANNELS, 4)]
        self.events = 0
        self.nonzero_inputs = 0
        self.spike_totals = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.nonempty_windows = 0
        self.total_slots = 0

    def run_window(self, features: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                   labels: np.ndarray | None = None, weight: torch.Tensor | None = None):
        if self.mode == "window_reset":
            for value in self.state:
                value.zero_()
        pack = self.packer.pack(features, x, y, t)
        if pack.count == 0:
            return None if labels is not None else np.empty(0, dtype=np.float32)
        cells = torch.as_tensor(pack.cells, dtype=torch.long, device=self.device)
        safe_slots = np.maximum(pack.event_slots, 0)
        input_tensor = torch.as_tensor(features[safe_slots].astype(np.float32) / 255.0,
                                       device=self.device)
        dt_tensor = torch.as_tensor(pack.delta_ms, device=self.device)
        valid_tensor = torch.as_tensor(pack.event_slots >= 0, device=self.device)
        state = tuple(value.index_select(0, cells) for value in self.state)
        probabilities = torch.empty(pack.count, dtype=torch.float32, device=self.device) if labels is None else None
        weighted = unweighted = weight_sum = None
        spike_sums = [torch.zeros((), dtype=torch.float64, device=self.device) for _ in range(3)]
        for k in range(len(pack.event_slots)):
            logits, update, spikes = self.model.step(input_tensor[k], dt_tensor[k], state)
            mask = valid_tensor[k]
            state = tuple(torch.where(mask[:, None], new, old) for new, old in zip(update, state))
            for layer in range(3):
                spike_sums[layer] = spike_sums[layer] + spikes[layer][mask].sum(dtype=torch.float64).detach()
            slots = pack.event_slots[k, pack.event_slots[k] >= 0]
            if labels is None:
                probabilities[torch.as_tensor(slots, dtype=torch.long, device=self.device)] = (
                    torch.softmax(class_logits(logits[mask]), dim=1)[:, 1])
            else:
                target = torch.as_tensor(labels[slots].astype(np.int64), device=self.device)
                ce = F.cross_entropy(class_logits(logits[mask]), target, reduction="none")
                weighted_part = (ce * weight[target]).sum()
                plain_part = ce.sum()
                mass = weight[target].sum()
                weighted = weighted_part if weighted is None else weighted + weighted_part
                unweighted = plain_part if unweighted is None else unweighted + plain_part
                weight_sum = mass if weight_sum is None else weight_sum + mass
        with torch.no_grad():
            for destination, value in zip(self.state, state):
                destination.index_copy_(0, cells, value.detach())
            for layer in range(3):
                self.spike_totals[layer] += spike_sums[layer]
        self.events += pack.count
        self.nonzero_inputs += int(np.count_nonzero(features))
        self.nonempty_windows += 1
        self.total_slots += int(pack.event_slots.size)
        if labels is None:
            return probabilities.detach().cpu().numpy()
        return weighted, unweighted, weight_sum, pack.count

    def counters(self):
        return {"candidate_events": self.events,
                "nonempty_windows": self.nonempty_windows,
                "packed_slots": self.total_slots,
                "valid_slot_fraction": self.events / self.total_slots if self.total_slots else None,
                "input_nonzero": self.nonzero_inputs,
                "spikes_l1": int(self.spike_totals[0].item()),
                "spikes_l2": int(self.spike_totals[1].item()),
                "spikes_out": int(self.spike_totals[2].item())}
