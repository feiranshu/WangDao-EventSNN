"""Frozen v1 motion-support definitions; all reads point to completed past bins."""
from __future__ import annotations

import numpy as np


def _lookup(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.zeros(len(x), dtype=np.float32)
    valid = (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
    result[valid] = image[y[valid], x[valid]]
    return result


def proposal_motion(x: np.ndarray, y: np.ndarray, previous: list[np.ndarray]) -> np.ndarray:
    if len(previous) < 2:
        return np.zeros(len(x), dtype=np.uint8)
    cx, cy = x // 8, y // 8
    score = np.zeros(len(x), dtype=np.float32)
    for vx in (-1, 0, 1):
        for vy in (-1, 0, 1):
            score = np.maximum(score, np.minimum(
                _lookup(previous[-1], cx-vx, cy-vy),
                _lookup(previous[-2], cx-2*vx, cy-2*vy)))
    return np.rint(255*np.clip(score, 0, 1)).astype(np.uint8)


def _quality(counts: np.ndarray) -> np.ndarray:
    padded = np.pad(counts.astype(np.float32), ((1, 1), (1, 1)))
    background = np.zeros(counts.shape, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                background += padded[1+dy:1+dy+counts.shape[0], 1+dx:1+dx+counts.shape[1]] / 8
    value = counts.astype(np.float32)
    return np.minimum(1, np.log1p(value)/np.log1p(8)) * value/(value+2) * np.maximum(
        0, 1-background/(value+background+1))


def lmec(events: np.ndarray, selected: np.ndarray, t0: float) -> np.ndarray:
    """Compute C2 from two preceding completed 25 ms bins. The event's own bin is not read."""
    if not len(selected):
        return np.empty(0, dtype=np.uint8)
    bins = np.floor((events['t'].astype(np.float64)-t0)/25.0).astype(np.int64)
    x = events['x'].astype(np.int64)//4
    y = events['y'].astype(np.int64)//4
    needed = set(int(bins[i]-lag) for i in selected for lag in (1, 2))
    quality = {}
    for b in needed:
        image = np.zeros((65, 87), dtype=np.int32)
        member = bins == b
        np.add.at(image, (y[member], x[member]), 1)
        quality[b] = _quality(image)
    score = np.zeros(len(selected), dtype=np.float32)
    for b in np.unique(bins[selected]):
        local = np.flatnonzero(bins[selected] == b)
        cx, cy = x[selected[local]], y[selected[local]]
        for vx in (-1, 0, 1):
            for vy in (-1, 0, 1):
                score[local] = np.maximum(score[local], np.minimum(
                    _lookup(quality[int(b)-1], cx-vx, cy-vy),
                    _lookup(quality[int(b)-2], cx-2*vx, cy-2*vy)))
    return np.rint(255*np.clip(score, 0, 1)).astype(np.uint8)
