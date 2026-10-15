"""Bounded teacher cache, chronological stream batches and label-free encoding."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Iterator, Tuple, TypeVar

import numpy as np

from .foundation import IMPLEMENTATION, read_json, role_records, sha256, verify_baseline

CACHE_VERSION = "h2pro_closed_prefix_v1"
ENCODER_VERSION = "phase32_count_last_10ms_v1"
ROWS, COLS, STEPS = 68, 88, 10
_Batch = TypeVar("_Batch")


def run_path(path):
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(IMPLEMENTATION / "results/runs"))
    if path == root or root not in path.parents:
        raise ValueError("Output/cache directories must be inside implementation/results/runs")
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


@dataclass
class CachedSequence:
    arrays: dict
    metadata: dict

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self.arrays.values())


def validate_sequence(arrays, meta):
    required = {"x", "y", "p", "t", "label", "candidate", "teacher"}
    if set(arrays) != required or len({len(a) for a in arrays.values()}) != 1:
        raise ValueError("Invalid teacher cache arrays")
    n = len(arrays["t"])
    if any(a.ndim != 1 for a in arrays.values()) or n != meta["published_events"]:
        raise ValueError("Teacher cache event coverage mismatch")
    if (meta["closed_windows"] < 0 or n + meta["trailing_events"] != meta["input_events"]
            or meta["original_events"] != meta["input_events"]+meta["beyond_prefix_events"]):
        raise ValueError("Teacher cache prefix accounting mismatch")
    a = arrays
    if (not np.isfinite(a["t"]).all() or np.any(np.diff(a["t"]) < 0)
            or np.any(a["t"] < meta["t0_ms"])
            or np.any(a["t"] >= meta["t0_ms"] + 100*meta["closed_windows"])
            or np.any((a["x"] < 0) | (a["x"] >= 346))
            or np.any((a["y"] < 0) | (a["y"] >= 260))
            or not np.isin(a["p"], [0, 1]).all() or not np.isin(a["label"], [0, 1]).all()
            or a["candidate"].dtype != np.bool_
            or not np.isfinite(a["teacher"]).all() or np.any((a["teacher"] < 0) | (a["teacher"] > 1))
            or np.any(a["teacher"][~a["candidate"]] != 0)):
        raise ValueError("Invalid teacher cache values")


class SequenceStore:
    def __init__(self, directory, cache_mib=1024):
        if not 0 <= cache_mib <= 1024:
            raise ValueError("RAM cache must be within [0,1024] MiB")
        self.directory = run_path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        m = self.manifest
        baseline = verify_baseline()
        if (m["version"] != CACHE_VERSION or m["baseline_lock_sha256"] != baseline["baseline_lock_sha256"]
                or m["status"] != "complete"):
            raise ValueError("Teacher cache identity/version mismatch")
        self.entries = {}
        for role in ("train_main", "train_dev"):
            allowed = {r["path"]: r for r in role_records(role)}
            rows = m["roles"][role]
            if not rows:
                raise ValueError("Both training and development caches are required")
            for row in rows:
                key = row["sequence"]
                if (key not in allowed or key in self.entries or row["source_sha256"] != allowed[key]["sha256"]
                        or Path(row["cache_file"]).name != row["cache_file"]):
                    raise ValueError("Cache contains wrong or duplicate data role")
                if sha256(self.directory / row["cache_file"]) != row["cache_sha256"]:
                    raise ValueError("Teacher cache file hash mismatch: " + key)
                self.entries[key] = (role, row)
        actual_complete = all(len(m["roles"][r]) == len(role_records(r)) for r in ("train_main", "train_dev"))
        actual_complete = actual_complete and m["max_windows"] is None
        if bool(m["complete_development_roles"]) != actual_complete:
            raise ValueError("False complete-cache declaration")
        self.budget = cache_mib*1024*1024
        self.resident = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "resident_bytes": 0, "peak_bytes": 0}

    def names(self, role):
        if role not in ("train_main", "train_dev"):
            raise ValueError("Only development roles allowed")
        return [r["sequence"] for r in self.manifest["roles"][role]]

    def get(self, role, name):
        if name not in self.entries or self.entries[name][0] != role:
            raise ValueError("Cache access crossed a data-role boundary")
        if name in self.resident:
            self.stats["hits"] += 1
            self.resident.move_to_end(name)
            return self.resident[name]
        self.stats["misses"] += 1
        row = self.entries[name][1]
        with np.load(self.directory / row["cache_file"], allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"].item()))
            arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        if (meta["sequence"] != name or meta["role"] != role or meta["source_sha256"] != row["source_sha256"]
                or meta["version"] != CACHE_VERSION
                or meta["baseline_lock_sha256"] != self.manifest["baseline_lock_sha256"]
                or meta["max_windows"] != self.manifest["max_windows"]
                or meta["published_events"] != row["published_events"]
                or meta["closed_windows"] != row["closed_windows"]
                or int(arrays["candidate"].sum()) != row["candidate_events"]
                or int(arrays["label"][arrays["candidate"]].sum()) != row["candidate_targets"]):
            raise ValueError("Cache payload identity mismatch")
        validate_sequence(arrays, meta)
        for value in arrays.values():
            value.setflags(write=False)
        record = CachedSequence(arrays, meta)
        if record.nbytes <= self.budget:
            while self.resident and self.stats["resident_bytes"] + record.nbytes > self.budget:
                _, old = self.resident.popitem(last=False)
                self.stats["resident_bytes"] -= old.nbytes
                self.stats["evictions"] += 1
            self.resident[name] = record
            self.stats["resident_bytes"] += record.nbytes
            self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.stats["resident_bytes"])
        return record


def encode_window(record, window, out=None):
    """Never use labels, candidates or teacher scores to construct model input."""
    a, meta = record.arrays, record.metadata
    if not 0 <= window < meta["closed_windows"]:
        raise ValueError("Window outside closed prefix")
    left = meta["t0_ms"] + window*100.0
    lo, hi = np.searchsorted(a["t"], [left, left+100.0], side="left")
    ids = np.arange(lo, hi, dtype=np.int64)
    relative = a["t"][ids] - left
    steps = np.floor(relative / 10.0).astype(np.int64)
    phase = (a["p"][ids].astype(np.int64)*16 + (a["y"][ids] % 4)*4 + a["x"][ids] % 4)
    row, col = a["y"][ids].astype(np.int64)//4, a["x"][ids].astype(np.int64)//4
    coordinates = (steps, phase, row, col)
    if out is None:
        out = np.empty((STEPS, 64, ROWS, COLS), np.float32)
    if out.shape != (STEPS, 64, ROWS, COLS) or out.dtype != np.float32:
        raise ValueError("Encoder output must have the fixed FP32 shape")
    counts, last = out[:, :32], out[:, 32:]
    counts.fill(0)
    last.fill(0)
    np.add.at(counts, coordinates, 1)
    np.maximum.at(last, coordinates, ((relative-steps*10.0)/10.0).astype(np.float32))
    # count=1 -> 0.632; bounded input without using corpus-wide statistics.
    np.negative(counts, out=counts)
    np.expm1(counts, out=counts)
    np.negative(counts, out=counts)
    return out, (steps, phase, row, col), ids


def stream_batches(store, role, batch_size, *, seed=42, epoch=0):
    """Shuffle sequences only. Slots traverse every closed window in order.

    Empty windows update state; no epoch/sequence crosses a gradient boundary.
    Replacement slots carry reset=True. Each event has exactly one query.
    """
    if batch_size < 1:
        raise ValueError("Positive stream batch size required")
    names = store.names(role)
    if role == "train_main":
        names = [names[i] for i in np.random.default_rng(seed+epoch).permutation(len(names))]
    pending = iter(names)
    slots = [None]*batch_size
    while True:
        reset = np.zeros(batch_size, bool)
        for slot in range(batch_size):
            if slots[slot] is None:
                for name in pending:
                    record = store.get(role, name)
                    if record.metadata["closed_windows"]:
                        slots[slot] = [name, record, 0]
                        reset[slot] = True
                        break
        if all(slot is None for slot in slots):
            break
        inputs = np.zeros((STEPS, batch_size, 64, ROWS, COLS), np.float32)
        maps, labels, candidates, teacher, refs = [], [], [], [], []
        active = np.zeros(batch_size, bool)
        for slot, item in enumerate(slots):
            if item is None:
                continue
            active[slot] = True
            name, record, window = item
            _, coords, ids = encode_window(record, window, out=inputs[:, slot])
            maps.append(np.stack((coords[0], np.full(len(ids), slot), *coords[1:])))
            for target, key in ((labels, "label"), (candidates, "candidate"), (teacher, "teacher")):
                target.append(record.arrays[key][ids])
            refs.append({"sequence": name, "slot": slot, "window": window, "ids": ids})
            if window+1 == record.metadata["closed_windows"]:
                slots[slot] = None
            else:
                item[2] += 1
        yield {"inputs": inputs, "maps": np.concatenate(maps, axis=1),
               "labels": np.concatenate(labels), "candidate": np.concatenate(candidates),
               "teacher": np.concatenate(teacher), "reset": reset, "active": active, "refs": refs}


def prefetch(iterator: Iterator[_Batch]) -> Iterator[Tuple[_Batch, float]]:
    """At most one pending batch; only the producer accesses its SequenceStore."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(next, iterator, None)
        while True:
            started = time.perf_counter()
            batch = future.result()
            wait = time.perf_counter()-started
            if batch is None:
                break
            future = executor.submit(next, iterator, None)
            yield batch, wait
