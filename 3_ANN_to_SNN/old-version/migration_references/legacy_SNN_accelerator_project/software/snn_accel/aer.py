from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numpy as np


class PacketType(IntEnum):
    EVENT = 0
    SLICE_START = 1
    SLICE_END = 2
    RESET = 3


class DecodedPacket(NamedTuple):
    packet_type: int
    x: int
    y: int
    polarity: int
    time_bin: int
    slice_id: int
    reserved: int


def pack_packet(
    packet_type: int | PacketType,
    *,
    x: int = 0,
    y: int = 0,
    polarity: int = 0,
    time_bin: int = 0,
    slice_id: int = 0,
) -> np.uint64:
    fields = {
        "packet_type": (int(packet_type), 0, 3),
        "x": (x, 0, 319),
        "y": (y, 0, 255),
        "polarity": (polarity, 0, 1),
        "time_bin": (time_bin, 0, 15),
        "slice_id": (slice_id, 0, 65535),
    }
    for name, (value, low, high) in fields.items():
        if not low <= int(value) <= high:
            raise ValueError(f"{name}={value} is outside [{low}, {high}]")
    word = (
        int(packet_type)
        | (int(x) << 2)
        | (int(y) << 11)
        | (int(polarity) << 19)
        | (int(time_bin) << 20)
        | (int(slice_id) << 24)
    )
    return np.uint64(word)


def unpack_packet(word: int | np.uint64, *, strict: bool = True) -> DecodedPacket:
    value = int(word)
    decoded = DecodedPacket(
        packet_type=value & 0x3,
        x=(value >> 2) & 0x1FF,
        y=(value >> 11) & 0xFF,
        polarity=(value >> 19) & 0x1,
        time_bin=(value >> 20) & 0xF,
        slice_id=(value >> 24) & 0xFFFF,
        reserved=(value >> 40) & 0xFFFFFF,
    )
    if strict and decoded.reserved:
        raise ValueError("reserved AER bits must be zero")
    return decoded

