"""Explicit operation ledgers; operation proxies are not device energy measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class OperationLedger:
    dense_neural_mac: int = 0
    zero_skip_neural_mac_proxy: int = 0
    ac: int = 0
    scalar_mul: int = 0
    scalar_div: int = 0
    exp: int = 0
    floor: int = 0
    round_to_nearest: int = 0
    comparisons: int = 0
    state_scalar_updates: int = 0
    state_read_bytes: int = 0
    state_write_bytes: int = 0
    routed_scalars: int = 0
    route_checks: int = 0
    unit_spikes: int = 0

    def add(self, **values) -> None:
        for key, value in values.items():
            if key not in self.__dataclass_fields__ or type(value) is not int or value < 0:
                raise ValueError("Invalid operation count: " + key)
        for key, value in values.items():
            setattr(self, key, getattr(self, key) + value)

    def merge(self, other: "OperationLedger") -> None:
        self.add(**asdict(other))

    def report(self, scope: str, missing: list) -> dict:
        return {"scope": scope, "counts": asdict(self), "unaccounted_components": list(missing),
                "end_to_end_complete": not missing,
                "memory_definition": "logical persistent-state traffic; not cache or DRAM profiling",
                "zero_skip_is_implemented": False,
                "energy_measured": False}


def profile_exact_mlp(features: np.ndarray, layers: tuple):
    """Independent diagnostic replay of frozen quantization, including zero weights.

    Returns logits for parity checking. Its extra profiling work must never be
    included in a claimed frozen-model latency or omitted from actual wall time.
    """
    value = np.ascontiguousarray(features, dtype=np.uint8)
    ledger = OperationLedger()
    for index, (weight, scale) in enumerate(layers):
        if weight.dtype != np.int8 or value.ndim != 2 or value.shape[1] != weight.shape[0]:
            raise ValueError("Unexpected frozen MLP dimensions or numeric type")
        n, width = len(value), weight.shape[1]
        # Each nonzero input connects only to nonzero weights in its own row.
        fanout = np.count_nonzero(weight, axis=1).astype(np.int64)
        effective = int(np.sum((value != 0).astype(np.int64) @ fanout))
        ledger.add(dense_neural_mac=int(n * weight.size), zero_skip_neural_mac_proxy=effective,
                   scalar_mul=int(n * width), scalar_div=int(n * width))
        raw = value.astype(np.int32) @ weight.astype(np.int32)
        floating = raw.astype(np.float64) * scale[None, :].astype(np.float64) / 255.0
        if index + 1 < len(layers):
            value = np.clip(np.rint(np.clip(floating, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)
            ledger.add(scalar_mul=int(n * width), comparisons=int(4 * n * width),
                       round_to_nearest=int(n * width))
        else:
            return floating.astype(np.float32), ledger
    raise ValueError("At least one MLP layer is required")
