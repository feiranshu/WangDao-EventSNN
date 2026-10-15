"""No-update lineage, integer backend, temporal bins and causal adapter checks."""
from __future__ import annotations

from importlib import util

import numpy as np
import torch

from ..fixed import FixedProposalSNN, load_fixed_proposal_snn
from ..snn import load_proposal_snn_export
from ..h2pro_snn.foundation import IMPLEMENTATION
from .common import lineage, source_hashes
from .core import Cost, EventFixedProposal, sparse_ann
from .pipeline import TransferPipeline, temporal_features, stream_state


def checks():
    torch.set_num_threads(1)
    origin = lineage()
    weights = IMPLEMENTATION / 'software/weights'
    s0, _, _ = load_proposal_snn_export(weights / 'S0P_T4_POS15_KD50_S42.npz')
    s1, _, _ = load_fixed_proposal_snn(weights / 'S1P_INT8_ACC24_MEM24_Q12.npz')
    derived = FixedProposalSNN.from_float(s0, 4, s1.configuration)
    for a, b in zip(derived.weights + derived.weight_fraction_bits, s1.weights + s1.weight_fraction_bits):
        np.testing.assert_array_equal(a, b)
    event = EventFixedProposal(s1)
    rng = np.random.default_rng(714)
    inputs = [np.zeros((4, 17, 128), np.uint8), np.full((4, 17, 128), 255, np.uint8),
              rng.integers(0, 256, (4, 23, 128), dtype=np.uint8)]
    sparse = inputs[-1].copy()
    sparse[rng.random(sparse.shape) < .95] = 0
    inputs.append(sparse)
    for x in inputs:
        a, b = event.forward(x), s1.forward(x)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        assert event.mac == np.count_nonzero(x) * 48
        assert event.ac == 24 * a.spike_counts[0] + 16 * a.spike_counts[1]
    pipe = TransferPipeline('ann_dense')
    x = rng.integers(0, 256, (23, 128), dtype=np.uint8)
    sparse_raw, _ = sparse_ann(x, pipe.proposal.proposal_layers)
    np.testing.assert_array_equal(sparse_raw, stream_state.quantized_mlp(x, pipe.proposal.proposal_layers))

    dtype = np.dtype(stream_state.EVENT_DTYPE.descr + [('label', 'u1')])
    events = np.zeros(15, dtype=dtype)
    t0 = 13.25
    events['t'] = t0 + np.array([0., 24.999, 25., 49.999, 50., 75., 99.999,
                                100., 125., 150., 199.999, 200., 225., 250., 299.999])
    events['x'] = rng.integers(0, 346, len(events))
    events['y'] = rng.integers(0, 260, len(events))
    events['p'] = np.arange(len(events)) % 2
    # Load the legacy temporal feature builder as an isolated reference module.
    spec = util.spec_from_file_location('wangdao.h2._transfer_temporal_reference',
                                      IMPLEMENTATION / 'software/wangdao/h2/representation.py')
    assert spec is not None and spec.loader is not None
    legacy = util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    reference = legacy.build_temporal_patch_features(events, [0, 1, 2], 4)
    for window in range(3):
        chosen = events[(events['t'] >= t0 + 100 * window) & (events['t'] < t0 + 100 * (window + 1))]
        np.testing.assert_array_equal(temporal_features(chosen, t0, window), reference[window])

    def run(mode, records, partitioned):
        p = TransferPipeline(mode)
        p.reset(t0)
        if partitioned:
            for window in range(3):
                lo, hi = np.searchsorted(records['t'], [t0 + 100 * window, t0 + 100 * (window + 1)])
                p.push(records[lo:hi], t0 + 100 * (window + 1))
        else:
            p.push(records, t0 + 300.)
        windows, trailing = p.finalize()
        assert trailing == 0
        return p, windows

    for mode in ('ann_dense', 'ann_sparse_proposal', 's0p_dense', 's1_event'):
        p, together = run(mode, events, False)
        _, divided = run(mode, events, True)
        poisoned = events.copy()
        poisoned['label'] = 1
        _, relabelled = run(mode, poisoned, False)
        for a, b, c in zip(together, divided, relabelled):
            for key in ('candidate_mask', 'features', 'heatmap', 'probabilities'):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
                np.testing.assert_array_equal(getattr(a, key), getattr(c, key))
        # Reset must discard Proposal heatmap/event history, while arithmetic is unchanged.
        p.reset(t0)
        p.push(events, t0 + 300.)
        again, _ = p.finalize()
        for a, b in zip(together, again):
            np.testing.assert_array_equal(a.features, b.features)
            np.testing.assert_array_equal(a.probabilities, b.probabilities)
        # No future window can change a previously published output.
        changed = events.copy()
        changed['x'][changed['t'] >= t0 + 100.] = 345 - changed['x'][changed['t'] >= t0 + 100.]
        _, future = run(mode, changed, False)
        np.testing.assert_array_equal(together[0].features, future[0].features)
        np.testing.assert_array_equal(together[0].probabilities, future[0].probabilities)

    # Force different Proposal heatmaps with the same event stream. This guards
    # against accidentally feeding old ANN candidate/context caches to Refiner.
    class ConstantProposal:
        mode = 'ann_dense'
        def __init__(self, logit):
            self.logit = logit
            self.cost = Cost()
        def __call__(self, patch, _layers, _temporal):
            return np.full((len(patch), 16), self.logit, np.float32)

    low = TransferPipeline('ann_dense', executor=ConstantProposal(-8.))
    high = TransferPipeline('ann_dense', executor=ConstantProposal(8.))
    for p in (low, high):
        p.reset(t0)
        p.push(events, t0 + 300.)
    low_windows, _ = low.finalize()
    high_windows, _ = high.finalize()
    assert all(not w.candidate_mask.any() for w in low_windows)
    assert all(w.candidate_mask.all() for w in high_windows)
    expected_current = np.rint(255 * high_windows[0].heatmap[
        events['y'][:7] // 8, events['x'][:7] // 8]).astype(np.uint8)
    np.testing.assert_array_equal(high_windows[0].features[:, 28], expected_current)
    selected = events[(events['t'] >= t0 + 100.) & (events['t'] < t0 + 200.)]
    expected_history = np.rint(255 * high_windows[0].heatmap[
        selected['y'] // 8, selected['x'] // 8]).astype(np.uint8)
    np.testing.assert_array_equal(high_windows[1].features[:, 32], expected_history)

    original = stream_state.quantized_mlp
    broken = TransferPipeline('ann_dense')
    broken.reset(t0)
    def fail(*_args):
        raise RuntimeError('injected test exception')
    broken.executor = fail
    # A function can carry the metadata used by the adapter.
    setattr(fail, 'mode', 'ann_dense')
    try:
        broken.push(events[:7], t0 + 100.)
    except RuntimeError as exc:
        assert str(exc) == 'injected test exception'
    else:
        raise AssertionError('Exception injection did not run')
    assert stream_state.quantized_mlp is original
    empty = TransferPipeline('s1_event')
    empty.reset(t0)
    empty.push(np.empty(0, dtype=dtype), t0 + 200.)
    windows, trailing = empty.finalize()
    assert len(windows) == 2 and trailing == 0 and empty.executor.cost.nonempty_windows == 0
    assert all(np.all(w.heatmap == .5) and len(w.features) == 0 for w in windows)
    return {'status': 'passed', 'lineage': origin, 's1_rederived_from_s0p_exact': True,
            'integer_event_backend_exact_cases': len(inputs), 'ann_sparse_proposal_exact': True,
            'legacy_microbin_encoding_exact_windows': 3,
            'adapter_checks': ['packet_partition', 'labels_ignored', 'reset', 'future_window_independence',
                               'empty_windows', 'own_candidates_current_and_history_context',
                               'exception_restores_frozen_dispatch'],
            'training_started': False, 'weights_updated': False, 'reserved_sequences_read': 0,
            'sources_utf8_lf': source_hashes()}
