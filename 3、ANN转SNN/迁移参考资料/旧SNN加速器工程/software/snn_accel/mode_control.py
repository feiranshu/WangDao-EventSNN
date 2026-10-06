from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class PrecisionMode(IntEnum):
    EXACT = 0
    APPROXIMATE = 1


@dataclass
class HysteresisController:
    density_low: int
    density_high: int
    mode: PrecisionMode = PrecisionMode.EXACT

    def __post_init__(self) -> None:
        if self.density_low < 0 or self.density_high <= self.density_low:
            raise ValueError("density thresholds require 0 <= low < high")

    def update(self, event_count: int) -> PrecisionMode:
        if event_count < 0:
            raise ValueError("event_count cannot be negative")
        if event_count >= self.density_high:
            self.mode = PrecisionMode.EXACT
        elif event_count <= self.density_low:
            self.mode = PrecisionMode.APPROXIMATE
        return self.mode

