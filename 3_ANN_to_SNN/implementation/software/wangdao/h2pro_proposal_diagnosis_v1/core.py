"""All-event features for interventions, and exact threshold frontiers."""
from __future__ import annotations

import numpy as np

from ..h2pro_proposal_transfer_v1.pipeline import stream_state as ss

GROUPS = {'motion': [2], 'current': list(range(28, 32)),
          'history': list(range(32, 36)), 'component': [30]}


def event_packet(arrays):
    packet = np.empty(len(arrays['t']), dtype=ss.EVENT_DTYPE)
    for key in ('x', 'y', 't', 'p'):
        packet[key] = arrays[key]
    return packet


def max_neighbors(heatmaps):
    if heatmaps.ndim != 3 or not np.isfinite(heatmaps).all():
        raise ValueError('Expected finite window/height/width heatmaps')
    padded = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    out = np.full_like(heatmaps, -np.inf)
    for dy in range(3):
        for dx in range(3):
            np.maximum(out, padded[:, dy:dy+heatmaps.shape[1], dx:dx+heatmaps.shape[2]], out=out)
    return out


def event_scores(packet, windows, heatmaps):
    return max_neighbors(heatmaps)[windows, packet['y']//8, packet['x']//8]


def context_features(packet, windows, base, heatmaps, gate, t0):
    """Build frozen C1 features for EVERY event, without reading labels or routing.

    Component size uses its own explicit gate. Unselected events have well-defined
    counterfactual features; final prediction must still be masked by routing.
    """
    if base.shape != (len(packet), 28) or not 0 <= gate <= 1:
        raise ValueError('Bad base features or gate')
    out = np.empty((len(packet), 36), np.uint8)
    out[:, :28] = base
    for w in range(len(heatmaps)):
        lo, hi = np.searchsorted(windows, [w, w+1])
        if lo == hi:
            continue
        rows = slice(lo, hi)
        x, y = packet['x'][rows].astype(np.int64), packet['y'][rows].astype(np.int64)
        hx, hy = x//8, y//8
        heatmap = heatmaps[w]
        previous = list(heatmaps[max(0, w-2):w])
        out[rows, 2] = ss.proposal_motion(x, y, previous)
        mask = ss.dilate_cells((heatmap >= gate)[None])[0]
        size = ss._component_sizes(mask)
        within = np.mod(packet['t'][rows]-t0, 100.)/100.
        out[rows, 28:32] = np.column_stack((
            np.rint(255*heatmap[hy, hx]), np.rint(255*ss.neighbor_mean(heatmap)[hy, hx]),
            np.rint(255*np.minimum(size[hy, hx].astype(np.float64)/64., 1.)),
            np.rint(255*np.clip(within, 0, 1)))).astype(np.uint8)
        history = np.zeros((hi-lo, 4), np.float32)
        if previous:
            history[:, 0] = previous[-1][hy, hx]
            history[:, 1] = ss.neighbor_mean(previous[-1])[hy, hx]
            history[:, 2:] = history[:, :2]
        if len(previous) > 1:
            history[:, 2] = np.maximum(history[:, 2], previous[-2][hy, hx])
            history[:, 3] = np.maximum(history[:, 3], ss.neighbor_mean(previous[-2])[hy, hx])
        out[rows, 32:36] = np.rint(255*np.clip(history, 0, 1)).astype(np.uint8)
    return out


def component_column(packet, windows, heatmaps, gate):
    out = np.empty(len(packet), np.uint8)
    for w in range(len(heatmaps)):
        lo, hi = np.searchsorted(windows, [w, w+1])
        if lo == hi:
            continue
        mask = ss.dilate_cells((heatmaps[w] >= gate)[None])[0]
        size = ss._component_sizes(mask)
        value = size[packet['y'][lo:hi]//8, packet['x'][lo:hi]//8]
        out[lo:hi] = np.rint(255*np.minimum(value.astype(np.float64)/64., 1.)).astype(np.uint8)
    return out


def refiner(features, layers):
    return ss._refiner_probability(ss.quantized_mlp(features, layers), 2)


def swapped(ann, snn, group):
    out = ann.copy()
    out[:, GROUPS[group]] = snn[:, GROUPS[group]]
    return out


def threshold_curve(scores, labels, probabilities, high_density):
    """Every distinct >= threshold, ties kept together; no grid approximation."""
    scores, labels, probabilities, hd = map(np.asarray, (scores, labels, probabilities, high_density))
    if (scores.ndim != 1 or any(a.shape != scores.shape for a in (labels, probabilities, hd))
            or not np.isfinite(scores).all() or not np.isfinite(probabilities).all()
            or np.any((scores < 0) | (scores > 1)) or not np.isin(labels, [0, 1]).all()):
        raise ValueError('Invalid curve inputs')
    order = np.argsort(-scores, kind='stable')
    s, y, p, h = scores[order], labels[order] == 1, probabilities[order] >= .35, hd[order].astype(bool)
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1] if len(s) else np.array([], np.int64)
    first = 1. if not len(s) or s[0] < 1 else np.nextafter(1., np.inf)
    out = {'threshold': np.r_[first, s[ends].astype(np.float64)],
           'candidate_events': np.r_[0, ends+1], 'target_events': np.full(len(ends)+1, y.sum(), np.int64),
           'hd_target_events': np.full(len(ends)+1, (y & h).sum(), np.int64)}
    for key, value in {'candidate_targets': y, 'tp': y & p, 'fp': ~y & p,
                       'hd_candidate_targets': y & h, 'hd_tp': y & p & h, 'hd_fp': ~y & p & h}.items():
        out[key] = np.r_[0, np.cumsum(value, dtype=np.int64)[ends]]
    out['iou'] = np.divide(out['tp'], out['target_events']+out['fp'],
                           out=np.zeros(len(ends)+1), where=out['target_events']+out['fp'] > 0)
    return out


def recover_gate(curve, overall_targets, hd_targets):
    valid = (curve['candidate_targets'] >= overall_targets) & (curve['hd_candidate_targets'] >= hd_targets)
    valid &= curve['threshold'] <= 1
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError('No gate can attain requested counts')
    return float(curve['threshold'][indices[0]])


def logit(heatmaps):
    p = np.clip(heatmaps.astype(np.float64), 2.**-24, 1-2.**-24)
    return np.log(p)-np.log1p(-p)


def map_heatmaps(heatmaps, calibration):
    # Stable sigmoid, fitted scalars only; no clipping of neural states/weights.
    z = calibration['slope']*logit(heatmaps)+calibration['intercept']
    out = np.empty(z.shape, np.float64)
    positive = z >= 0
    out[positive] = 1/(1+np.exp(-z[positive]))
    exp = np.exp(z[~positive])
    out[~positive] = exp/(1+exp)
    return out.astype(np.float32)
