"""Keep every query; use a deterministic subset only for neighborhood state.

Only the original 24 count columns (4:28) are replaced. The current query
contributes itself to its same-polarity current cell even when it does not
update persistent statistics. Selected queries already occur in that state,
so their own contribution is never counted twice. No inverse-probability
rescaling is used: this is intentionally a change of input density statistics.
"""
from __future__ import annotations

import numpy as np

from wangdao.h2.constants import EVENT_FEATURE_SPECS, SENSOR_HEIGHT, SENSOR_WIDTH
from wangdao.h2.features import NEIGHBOR_OFFSETS, _lookup_counts, compress_counts


UNCHANGED_COLUMNS = np.r_[0:4, 28:36]


def update_mask(source_ids, denominator=16, sequence_number=0):
    if type(denominator) is not int or denominator < 1 or denominator & (denominator-1):
        raise ValueError('The update denominator must be a positive power of two')
    ids = np.asarray(source_ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('Nonnegative integer source-event IDs required')
    x = ids.astype(np.uint64) + np.uint64(42 + sequence_number*1000000007)
    with np.errstate(over='ignore'):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return (x & np.uint64(denominator-1)) == 0


def count_columns(context, queries, query_updates_state, t0):
    """Query past/current selected events using the unchanged cell/bin definitions.

    context contains selected events from the previous and current closed
    100 ms windows. The caller owns the original watermark boundary; no event
    from a later window is supplied. Time bin aggregation within a closed
    window is inherited from the original ANN, not new per-event causality.
    """
    updates = np.asarray(query_updates_state)
    if updates.dtype != np.bool_ or updates.shape != (len(queries),):
        raise ValueError('One Boolean update flag per query required')
    for events in (context, queries):
        if events.ndim != 1 or events.dtype.names is None or not {'x','y','t','p'} <= set(events.dtype.names):
            raise ValueError('Structured x/y/t/p events required')
    out = np.empty((len(queries),24),dtype=np.uint8)
    if not len(queries):
        return out
    col = 0
    for cell, ms in EVENT_FEATURE_SPECS:
        nx = (SENSOR_WIDTH + cell-1)//cell
        ny = (SENSOR_HEIGHT + cell-1)//cell
        cx = context['x'].astype(np.int64)//cell
        cy = context['y'].astype(np.int64)//cell
        cp = context['p'].astype(np.int64)
        ct = np.floor((context['t']-t0)/ms).astype(np.int64)
        ck = (((ct*ny+cy)*nx+cx)*2+cp).astype(np.int64)
        unique, counts = np.unique(ck,return_counts=True)
        qx = queries['x'].astype(np.int64)//cell
        qy = queries['y'].astype(np.int64)//cell
        qp = queries['p'].astype(np.int64)
        qt = np.floor((queries['t']-t0)/ms).astype(np.int64)
        qk = (((qt*ny+qy)*nx+qx)*2+qp).astype(np.int64)
        # Repeated events sharing a query key reuse the exact lookup result.
        _, first, inverse = np.unique(qk,return_index=True,return_inverse=True)
        x,y,p,t = qx[first],qy[first],qp[first],qt[first]
        same = _lookup_counts(unique,counts,t,x,y,p,nx,ny)
        out[:,col] = compress_counts(same[inverse] + (~updates).astype(np.int64))
        out[:,col+1] = compress_counts(_lookup_counts(unique,counts,t,x,y,1-p,nx,ny))[inverse]
        col += 2
        for lag in (1,2):
            for polarity in (p,1-p):
                value = np.zeros(len(first),dtype=np.int64)
                for dx,dy in NEIGHBOR_OFFSETS:
                    value += _lookup_counts(unique,counts,t-lag,x+dx,y+dy,polarity,nx,ny)
                out[:,col] = compress_counts(value)[inverse]
                col += 1
    assert col == 24
    return out


def replace_counts(original_features, new_counts):
    if (original_features.dtype != np.uint8 or original_features.ndim != 2 or
            original_features.shape[1] != 36 or new_counts.dtype != np.uint8 or
            new_counts.shape != (len(original_features),24)):
        raise ValueError('Original uint8 N x 36 and adapted uint8 N x 24 required')
    result = original_features.copy()
    result[:,4:28] = new_counts
    return result
