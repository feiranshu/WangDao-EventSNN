"""100 ms watermark-driven H2-pro ANN. Input fields beyond x/y/t/p are discarded."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

from wangdao.h2.constants import PATCHES_PER_WINDOW
from wangdao.h2.numpy_model import _refiner_probability, _sigmoid, load_export, quantized_mlp
from wangdao.h2.representation import (_component_sizes, _fine_counts_to_patch_features,
                                        dilate_cells, patch_scores_to_heatmaps)
from .causal_feature_builder import base_features, neighbor_mean
from .motion_coherence import lmec, proposal_motion

EVENT_DTYPE = np.dtype([('x', '<i2'), ('y', '<i2'), ('t', '<f8'), ('p', 'u1')])
VARIANTS = ('C0-zero', 'C1-proposal-motion', 'C2-LMEC')


def load_initial_refiner(path: Path):
    """Keep the A4-2 exact-int8 numerical path while zeroing its noncausal input row."""
    with np.load(path, allow_pickle=False) as archive:
        dimensions = tuple(int(v) for v in archive['dimensions'])
        if dimensions != (36, 64, 32, 4):
            raise ValueError('expected frozen A4-2 36-64-32-4 checkpoint')
        layers = []
        for i in range(3):
            weight = archive[f'layer{i}_weight'].copy()
            scale = archive[f'layer{i}_scale'].copy()
            if i == 0:
                weight[2, :] = 0
            layers.append((weight, scale))
    return tuple(layers)


@dataclass(frozen=True)
class WindowResult:
    window: int
    right_boundary_ms: float
    event_indices: np.ndarray
    candidate_mask: np.ndarray
    features: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    heatmap: np.ndarray
    computation_s: float


class CausalPipeline:
    """One independent sequence per reset; push accepts sorted events below its watermark."""
    def __init__(self, proposal_path: Path, refiner_path: Path, variant: str = 'C0-zero',
                 threshold: float = 0.40, initial_refiner: bool = True):
        if variant not in VARIANTS:
            raise ValueError(f'unknown variant: {variant}')
        self.proposal = load_export(Path(proposal_path))
        self.refiner_layers = (load_initial_refiner(Path(refiner_path)) if initial_refiner
                               else self._load_trained(Path(refiner_path), variant))
        self.variant = variant
        self.threshold = float(threshold)
        self.reset(0.0)

    @staticmethod
    def _load_trained(path: Path, variant: str):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive['variant'].item()) != variant or tuple(archive['dimensions']) != (36,64,32,4):
                raise ValueError('trained model variant or dimensions mismatch')
            expected = hashlib.sha256((Path(__file__).absolute().parent.parent/'protocol.json').read_bytes()).hexdigest()
            if str(archive['protocol_sha256'].item()) != expected:
                raise ValueError('trained model feature protocol mismatch')
            layers = tuple((archive[f'layer{i}_weight'].copy(), archive[f'layer{i}_scale'].copy())
                           for i in range(3))
            dimensions = (36,64,32,4)
            for i,(weight,scale) in enumerate(layers):
                if (weight.dtype != np.int8 or weight.shape != (dimensions[i],dimensions[i+1]) or
                        scale.dtype != np.float32 or scale.shape != (dimensions[i+1],)):
                    raise ValueError('trained exact-int8 layer contract mismatch')
            return layers

    def reset(self, t0: float) -> None:
        if not np.isfinite(t0):
            raise ValueError('t0 must be finite')
        self.t0 = float(t0)
        self.finalized = False
        self.watermark = self.t0
        self.next_window = 0
        self.next_event_index = 0
        self.last_timestamp = -np.inf
        self.pending = np.empty(0, dtype=EVENT_DTYPE)
        self.previous_events = np.empty(0, dtype=EVENT_DTYPE)
        self.previous_heatmaps: list[np.ndarray] = []
        self.ready: list[WindowResult] = []

    def push(self, events: np.ndarray, watermark: float) -> None:
        if self.finalized:
            raise ValueError('reset is required after finalize')
        if not np.isfinite(watermark) or watermark < self.watermark:
            raise ValueError('watermark cannot move backward')
        if events.ndim != 1 or events.dtype.names is None or any(
            field not in events.dtype.names for field in ('x','y','t','p')):
            raise ValueError('events require structured x/y/t/p fields')
        packet = np.empty(len(events), dtype=EVENT_DTYPE)
        for field in ('x','y','t','p'):
            packet[field] = events[field]
        if len(packet):
            t = packet['t']
            if (not np.all(np.isfinite(t)) or t[0] < self.watermark or
                    t[0] < self.last_timestamp or np.any(np.diff(t) < 0) or
                    t[-1] >= watermark):
                raise ValueError('late, unsorted or not yet watermarked events')
            if (np.any(packet['x'] < 0) or np.any(packet['x'] >= 346) or
                    np.any(packet['y'] < 0) or np.any(packet['y'] >= 260) or
                    np.any(packet['p'] > 1)):
                raise ValueError('event coordinate or polarity outside sensor')
            self.pending = np.concatenate((self.pending, packet))
            self.last_timestamp = float(t[-1])
        self.watermark = float(watermark)
        # The watermark closes every boundary at or below it, including empty windows.
        while self.t0 + (self.next_window+1)*100.0 <= self.watermark:
            right = self.t0 + (self.next_window+1)*100.0
            count = int(np.searchsorted(self.pending['t'], right, side='left'))
            current, self.pending = self.pending[:count].copy(), self.pending[count:].copy()
            self.ready.append(self._close(current, right))
            self.next_event_index += count
            self.next_window += 1

    def flush_completed_windows(self) -> list[WindowResult]:
        result, self.ready = self.ready, []
        return result

    def finalize(self, watermark: float | None = None) -> tuple[list[WindowResult], int]:
        if watermark is not None:
            self.push(np.empty(0, dtype=EVENT_DTYPE), watermark)
        result = self.flush_completed_windows()
        # A trailing incomplete window is deliberately not published.
        leftover = len(self.pending)
        self.finalized = True
        return result, leftover

    def _close(self, current: np.ndarray, right: float) -> WindowResult:
        started = time.perf_counter()
        n = len(current)
        indices = np.arange(self.next_event_index, self.next_event_index+n, dtype=np.int64)
        if n:
            fine = np.zeros((1, 2, 68, 88), dtype=np.int32)
            np.add.at(fine[0], (current['p'], current['y']//4, current['x']//4), 1)
            patch = _fine_counts_to_patch_features(fine)
            raw = quantized_mlp(patch.reshape(-1, 128), self.proposal.proposal_layers)
            scores = _sigmoid(raw).reshape(1, PATCHES_PER_WINDOW, 16)
            heatmap = patch_scores_to_heatmaps(scores)[0]
        else:
            # Zero input, no bias, sigmoid(0)=0.5, below the frozen 0.51 gate.
            heatmap = np.full((34, 44), 0.5, dtype=np.float32)
        mask = dilate_cells((heatmap >= self.proposal.proposal_threshold)[None])[0]
        cx = current['x'].astype(np.int64)//8
        cy = current['y'].astype(np.int64)//8
        candidate = mask[cy, cx] if n else np.empty(0, dtype=bool)
        selected = np.flatnonzero(candidate)
        features = np.empty((len(selected), 36), dtype=np.uint8)
        history_events = np.concatenate((self.previous_events, current))
        if len(selected):
            base = base_features(history_events, len(self.previous_events), self.t0)
            features[:, :28] = base[selected]
            x, y = current['x'][selected].astype(np.int64), current['y'][selected].astype(np.int64)
            if self.variant == 'C1-proposal-motion':
                features[:, 2] = proposal_motion(x, y, self.previous_heatmaps)
            elif self.variant == 'C2-LMEC':
                features[:, 2] = lmec(history_events, selected+len(self.previous_events), self.t0)
            else:
                features[:, 2] = 0
            mean = neighbor_mean(heatmap)
            sizes = _component_sizes(mask)
            within = np.mod(current['t'][selected]-self.t0, 100.0)/100.0
            context = np.column_stack((
                np.rint(255*heatmap[y//8, x//8]),
                np.rint(255*mean[y//8, x//8]),
                np.rint(255*np.minimum(sizes[y//8, x//8].astype(np.float64)/64.0, 1)),
                np.rint(255*np.clip(within, 0, 1)),
            )).astype(np.uint8)
            features[:, 28:32] = context
            history = np.zeros((len(selected), 4), dtype=np.float32)
            if self.previous_heatmaps:
                h1 = self.previous_heatmaps[-1]
                m1 = neighbor_mean(h1)
                history[:, 0] = h1[y//8, x//8]
                history[:, 1] = m1[y//8, x//8]
                history[:, 2] = history[:, 0]
                history[:, 3] = history[:, 1]
            if len(self.previous_heatmaps) > 1:
                h2 = self.previous_heatmaps[-2]
                m2 = neighbor_mean(h2)
                history[:, 2] = np.maximum(history[:, 2], h2[y//8, x//8])
                history[:, 3] = np.maximum(history[:, 3], m2[y//8, x//8])
            features[:, 32:36] = np.rint(255*np.clip(history, 0, 1)).astype(np.uint8)
            cp = _refiner_probability(quantized_mlp(features, self.refiner_layers), 2)
        else:
            cp = np.empty(0, dtype=np.float32)
        probabilities = np.zeros(n, dtype=np.float32)
        probabilities[selected] = cp
        self.previous_events = current
        self.previous_heatmaps.append(heatmap)
        self.previous_heatmaps = self.previous_heatmaps[-2:]
        return WindowResult(self.next_window, right, indices, candidate, features,
                            probabilities, probabilities >= self.threshold, heatmap,
                            time.perf_counter()-started)
