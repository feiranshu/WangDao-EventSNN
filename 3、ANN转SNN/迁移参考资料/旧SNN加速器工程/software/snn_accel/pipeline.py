from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aer import PacketType, pack_packet
from .events import EventTable, SliceConfig
from .slicing import EventSlice, slice_events


@dataclass(frozen=True)
class PreprocessResult:
    packets: np.ndarray
    tlast: np.ndarray
    slices: tuple[EventSlice, ...]


def preprocess(events: EventTable, config: SliceConfig) -> PreprocessResult:
    """Create AER packets and targets from events.

    Slice boundaries are computed before labels are inspected. This invariant is
    tested explicitly to prevent annotation leakage into the data representation.
    """

    slices = slice_events(events, config)
    packets: list[np.uint64] = []
    tlast: list[bool] = []
    for item in slices:
        packets.append(pack_packet(PacketType.SLICE_START, slice_id=item.slice_id))
        tlast.append(False)
        for offset, time_bin in enumerate(item.time_bins):
            index = item.start + offset
            packets.append(
                pack_packet(
                    PacketType.EVENT,
                    x=int(events.x[index]),
                    y=int(events.y[index]),
                    polarity=int(events.p[index]),
                    time_bin=int(time_bin),
                    slice_id=item.slice_id,
                )
            )
            tlast.append(False)
        packets.append(pack_packet(PacketType.SLICE_END, slice_id=item.slice_id))
        tlast.append(True)
    return PreprocessResult(
        packets=np.asarray(packets, dtype=np.uint64),
        tlast=np.asarray(tlast, dtype=np.bool_),
        slices=slices,
    )

