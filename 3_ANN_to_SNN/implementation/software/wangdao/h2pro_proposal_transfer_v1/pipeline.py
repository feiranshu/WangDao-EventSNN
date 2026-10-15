"""Small adapter around the UNMODIFIED frozen causal pipeline (isolated worker)."""
from __future__ import annotations

from pathlib import Path
from importlib import import_module

import numpy as np

from wangdao.h2 import representation
from ..h2pro_snn.foundation import ANN_ROOT
from .core import ProposalExecutor

stream_state = import_module('wangdao.h2_pro.stream_state')


def temporal_features(current, t0, window, steps=4):
    """The old temporal representation with an explicit sequence time origin.

    Do not rebase each window to its first arriving event: that would change bins.
    Labels are neither accessed nor passed to the frozen inference pipeline.
    """
    if steps != 4:
        raise ValueError('This frozen transfer uses T4')
    if current.dtype.names is None or any(k not in current.dtype.names for k in ('x', 'y', 't', 'p')):
        raise ValueError('Require structured x/y/t/p')
    left, right = t0 + window * 100., t0 + (window + 1) * 100.
    if (not np.isfinite(current['t']).all() or np.any(current['t'] < left)
            or np.any(current['t'] >= right) or np.any(current['x'] < 0) or np.any(current['x'] >= 346)
            or np.any(current['y'] < 0) or np.any(current['y'] >= 260)
            or np.any(current['p'] < 0) or np.any(current['p'] > 1)):
        raise ValueError('Invalid window coordinates/timestamps')
    micro = np.floor(np.mod(current['t'] - t0, 100.) / 25.).astype(np.int64)
    fine = np.zeros((4, 2, 68, 88), dtype=np.int32)
    np.add.at(fine, (micro, current['p'], current['y'] // 4, current['x'] // 4), 1)
    return representation._fine_counts_to_patch_features(fine)


class TransferPipeline(stream_state.CausalPipeline):
    def __init__(self, mode, executor=None):
        expected = ANN_ROOT / 'software/wangdao'
        if any(Path(str(m.__file__)).absolute().parent.parent != expected
               for m in (stream_state, representation)):
            raise RuntimeError('Use the isolated ANN namespace worker for Proposal transfer')
        self.executor = executor if executor is not None else ProposalExecutor(mode)
        super().__init__(ANN_ROOT / 'software/weights/H2_D1_seed42.npz',
                         ANN_ROOT / 'software/weights/H2-pro-Causal-ANN-v1_int8.npz',
                         variant='C1-proposal-motion', threshold=.35, initial_refiner=False)

    def _close(self, current, right):
        original = stream_state.quantized_mlp
        temporal = (temporal_features(current, self.t0, self.next_window)
                    if len(current) and not self.executor.mode.startswith('ann_') else None)

        def dispatch(features, layers):
            if features.shape[1] == 128:
                return self.executor(features, layers, temporal)
            if features.shape[1] != 36:
                raise ValueError('Unexpected neural interface dimension')
            # All variants call the original frozen Refiner with their OWN context.
            self.executor.cost.refiner_mac += len(features) * sum(w.size for w, _ in layers)
            self.executor.cost.refiner_layer_output_scalars += len(features) * 100
            return original(features, layers)

        # Only this single-threaded isolated worker swaps the function reference.
        # Restoring in finally also protects subsequent windows after exceptions.
        setattr(stream_state, 'quantized_mlp', dispatch)
        try:
            return super()._close(current, right)
        finally:
            setattr(stream_state, 'quantized_mlp', original)


def infer(arrays, metadata, mode):
    packet = np.empty(len(arrays['t']), dtype=stream_state.EVENT_DTYPE)
    for key in ('x', 'y', 't', 'p'):
        packet[key] = arrays[key]
    pipeline = TransferPipeline(mode)
    pipeline.reset(metadata['t0_ms'])
    pipeline.push(packet, metadata['t0_ms'] + 100. * metadata['closed_windows'])
    windows, trailing = pipeline.finalize()
    ids = np.concatenate([w.event_indices for w in windows]) if windows else np.empty(0, np.int64)
    if (trailing or len(windows) != metadata['closed_windows']
            or not np.array_equal(ids, np.arange(len(packet)))):
        raise AssertionError('Transfer changed the closed-prefix event coverage')
    output = {
        'candidate': np.concatenate([w.candidate_mask for w in windows]) if windows else np.empty(0, bool),
        'probabilities': np.concatenate([w.probabilities for w in windows]) if windows else np.empty(0, np.float32),
        'features': np.concatenate([w.features for w in windows]) if windows else np.empty((0, 36), np.uint8),
        'heatmaps': np.stack([w.heatmap for w in windows]) if windows else np.empty((0, 34, 44), np.float32),
        'window_events': np.asarray([len(w.event_indices) for w in windows], np.int64),
        'window_candidates': np.asarray([int(w.candidate_mask.sum()) for w in windows], np.int64),
    }
    return output, pipeline.executor
