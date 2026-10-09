"""Shared helpers for the frozen KV260 Proposal-SNN validation tools."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parents[2]
ORIGINAL_SOFTWARE = PROJECT_ROOT / "0_mlp_original_project" / "software"
DEFAULT_MODEL = (
    ORIGINAL_SOFTWARE / "weights" / "S1P_INT8_ACC24_MEM24_Q12.npz"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "0_mlp_original_project" / "data" / "EV-UAV-dataset"


def add_software_path() -> None:
    path = str(ORIGINAL_SOFTWARE)
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path = DEFAULT_MODEL):
    add_software_path()
    from wangdao.fixed import load_fixed_proposal_snn

    return load_fixed_proposal_snn(path)


def parse_time_to_lane(value: str) -> tuple[int, int, int, int]:
    compact = "".join(character for character in value if character.isdigit())
    if len(compact) != 4:
        raise ValueError("time-to-lane must contain exactly four digits, e.g. 0123")
    result = tuple(int(character) for character in compact)
    if tuple(sorted(result)) != (0, 1, 2, 3):
        raise ValueError("time-to-lane must be a permutation of 0,1,2,3")
    return result  # type: ignore[return-value]


def order_name(time_to_lane: Sequence[int]) -> str:
    return "".join(str(int(value)) for value in time_to_lane)


def all_time_to_lane_orders() -> Iterable[tuple[int, int, int, int]]:
    return itertools.permutations((0, 1, 2, 3))


def pack_temporal_features(
    temporal: np.ndarray, time_to_lane: Sequence[int]
) -> np.ndarray:
    """Pack [time=4, feature=128] uint8 into 128 uint32 words.

    ``time_to_lane[t]`` gives the byte lane used by software time step ``t``.
    Lane 0 is bits [7:0], lane 3 is bits [31:24].
    """

    value = np.asarray(temporal)
    if value.shape != (4, 128):
        raise ValueError(f"temporal features must have shape (4, 128), got {value.shape}")
    if value.dtype != np.uint8:
        if np.any((value < 0) | (value > 255)):
            raise ValueError("temporal feature values must be in [0, 255]")
        value = value.astype(np.uint8)
    order = parse_time_to_lane(order_name(time_to_lane))
    words = np.zeros(128, dtype=np.uint32)
    for time_step, lane in enumerate(order):
        words |= value[time_step].astype(np.uint32) << np.uint32(8 * lane)
    return words


def temporal_from_lane_values(
    lane_values: np.ndarray, time_to_lane: Sequence[int]
) -> np.ndarray:
    """Map physical byte-lane values to software time order."""

    lanes = np.asarray(lane_values, dtype=np.uint8)
    if lanes.shape != (4, 128):
        raise ValueError(f"lane values must have shape (4, 128), got {lanes.shape}")
    order = parse_time_to_lane(order_name(time_to_lane))
    return np.ascontiguousarray(lanes[np.asarray(order, dtype=np.int64)])


def pack_physical_lanes(lane_values: np.ndarray) -> np.ndarray:
    """Pack raw lane0..lane3 values without assuming their time semantics."""

    lanes = np.asarray(lane_values, dtype=np.uint8)
    if lanes.shape != (4, 128):
        raise ValueError(f"lane values must have shape (4, 128), got {lanes.shape}")
    words = np.zeros(128, dtype=np.uint32)
    for lane in range(4):
        words |= lanes[lane].astype(np.uint32) << np.uint32(8 * lane)
    return words


def decode_time_major_lanes(
    lane_values: np.ndarray,
    block_for_time: Sequence[int],
    lane_for_feature_offset: Sequence[int],
) -> np.ndarray:
    """Decode 128 physical words as four 32-word time blocks.

    Each word stores four adjacent features.  ``block_for_time[t]`` selects
    the physical 32-word block for software time ``t``.  Likewise,
    ``lane_for_feature_offset[r]`` selects the physical byte lane for feature
    offset ``r`` inside each group of four features.
    """

    lanes = np.asarray(lane_values, dtype=np.uint8)
    if lanes.shape != (4, 128):
        raise ValueError(f"lane values must have shape (4, 128), got {lanes.shape}")
    blocks = parse_time_to_lane(order_name(block_for_time))
    offsets = parse_time_to_lane(order_name(lane_for_feature_offset))
    temporal = np.empty((4, 128), dtype=np.uint8)
    for time_step, block in enumerate(blocks):
        for group in range(32):
            word_index = block * 32 + group
            for feature_offset, lane in enumerate(offsets):
                temporal[time_step, 4 * group + feature_offset] = lanes[
                    lane, word_index
                ]
    return temporal


def pack_time_major_features(
    temporal: np.ndarray,
    block_for_time: Sequence[int],
    lane_for_feature_offset: Sequence[int],
) -> np.ndarray:
    """Inverse of :func:`decode_time_major_lanes` for a canonical sample."""

    value = np.asarray(temporal)
    if value.shape != (4, 128):
        raise ValueError(f"temporal features must have shape (4, 128), got {value.shape}")
    if value.dtype != np.uint8:
        if np.any((value < 0) | (value > 255)):
            raise ValueError("temporal feature values must be in [0, 255]")
        value = value.astype(np.uint8)
    blocks = parse_time_to_lane(order_name(block_for_time))
    offsets = parse_time_to_lane(order_name(lane_for_feature_offset))
    lanes = np.zeros((4, 128), dtype=np.uint8)
    for time_step, block in enumerate(blocks):
        for group in range(32):
            word_index = block * 32 + group
            for feature_offset, lane in enumerate(offsets):
                lanes[lane, word_index] = value[
                    time_step, 4 * group + feature_offset
                ]
    return pack_physical_lanes(lanes)


def signed_to_u32(values: np.ndarray) -> np.ndarray:
    signed = np.asarray(values, dtype=np.int64)
    if np.any((signed < -(1 << 31)) | (signed > (1 << 31) - 1)):
        raise ValueError("value does not fit signed int32")
    return (signed & np.int64(0xFFFFFFFF)).astype(np.uint32)


def write_hex_words(path: Path, values: np.ndarray) -> None:
    words = np.asarray(values, dtype=np.uint32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"0x{int(value):08X}\n" for value in words)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(text)


def read_hex_words(path: Path, expected_count: int | None = None) -> np.ndarray:
    values: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            parsed = int(value, 0)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid integer {value!r}") from error
        if not 0 <= parsed <= 0xFFFFFFFF:
            raise ValueError(f"{path}:{line_number}: value is outside uint32")
        values.append(parsed)
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} words, found {len(values)}"
        )
    return np.asarray(values, dtype=np.uint32)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def assert_known_patch_baseline(model) -> None:
    zero = np.zeros((4, 1, 128), dtype=np.uint8)
    full = np.full((4, 1, 128), 255, dtype=np.uint8)
    expected_full = np.asarray(
        [
            0xFFFFB0D8,
            0xFFFFCC02,
            0xFFFFECC2,
            0xFFFFF973,
            0xFFFF984B,
            0xFFFF8E5D,
            0xFFFFBA6C,
            0xFFFFE996,
            0xFFFFAC04,
            0xFFFF9330,
            0xFFFF9D64,
            0xFFFFCBEF,
            0xFFFFD86C,
            0xFFFFAE80,
            0xFFFF9BAC,
            0xFFFFBA5C,
        ],
        dtype=np.uint32,
    )
    actual_zero = signed_to_u32(model.forward(zero).logits_q[0])
    actual_full = signed_to_u32(model.forward(full).logits_q[0])
    if not np.array_equal(actual_zero, np.zeros(16, dtype=np.uint32)):
        raise RuntimeError("software model no longer reproduces patch0")
    if not np.array_equal(actual_full, expected_full):
        raise RuntimeError("software model no longer reproduces patch1")
