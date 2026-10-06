from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheStats:
    requests: int
    hits: int
    external_reads: int

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.requests == 0 else self.hits / self.requests


class StateTileCacheModel:
    """Deterministic LRU model for evaluating state-tile prefetch policies."""

    def __init__(self, *, width: int, height: int, tile_size: int = 8, capacity_tiles: int = 32):
        if min(width, height, tile_size, capacity_tiles) <= 0:
            raise ValueError("cache dimensions and capacity must be positive")
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.capacity_tiles = capacity_tiles
        self._tiles: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._requests = 0
        self._hits = 0
        self._external_reads = 0

    def _load(self, tile: tuple[int, int]) -> None:
        if tile in self._tiles:
            self._tiles.move_to_end(tile)
            return
        self._external_reads += 1
        self._tiles[tile] = None
        if len(self._tiles) > self.capacity_tiles:
            self._tiles.popitem(last=False)

    def access(self, x: int, y: int, *, prefetch_3x3: bool = True) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("state coordinate outside cache geometry")
        tile = (x // self.tile_size, y // self.tile_size)
        self._requests += 1
        hit = tile in self._tiles
        self._hits += int(hit)
        self._load(tile)
        if prefetch_3x3:
            max_tx = (self.width - 1) // self.tile_size
            max_ty = (self.height - 1) // self.tile_size
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (tile[0] + dx, tile[1] + dy)
                    if 0 <= neighbor[0] <= max_tx and 0 <= neighbor[1] <= max_ty:
                        self._load(neighbor)
        return hit

    @property
    def stats(self) -> CacheStats:
        return CacheStats(self._requests, self._hits, self._external_reads)

