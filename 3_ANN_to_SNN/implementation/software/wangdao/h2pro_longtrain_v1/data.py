"""Exact pre-encoding, resident sparse inputs and random-access stream schedules.

No labels, Proposal decisions or teacher scores enter the input encoder. Sparse
values retain the original float32 encoding; GPU reconstruction uses unique
indices, with no unordered scatter reductions.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..h2pro_snn.stream_data import (SequenceStore, encode_window, atomic_json, run_path,
                                   ROWS, COLS, STEPS, ENCODER_VERSION)
from ..h2pro_snn.foundation import sha256
from .storage import process_lock

D = 64 * ROWS * COLS
CACHE_VERSION = "exact_sparse_phase_v1"


def prepare(teacher_directory, directory):
    directory = run_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with process_lock(directory / ".prepare.lock"):
        store = SequenceStore(teacher_directory)
        if not store.manifest["complete_development_roles"]:
            raise ValueError("Full 79/10 development cache required")
        identity = {"version": CACHE_VERSION, "encoder": ENCODER_VERSION,
                    "teacher_manifest_sha256": sha256(store.directory / "manifest.json")}
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"identity": identity, "rows": []}
        if manifest["identity"] != identity:
            raise ValueError("Encoded cache identity changed; choose a new directory")
        rows = {r["sequence"]: r for r in manifest["rows"]}
        started = time.perf_counter()
        for role in ("train_main", "train_dev"):
            for name in store.names(role):
                if name in rows and sha256(directory / rows[name]["file"]) == rows[name]["sha256"]:
                    continue
                record = store.get(role, name)
                indices, values, maps, cuts, pointers = [], [], [], [0], [0]
                for w in range(record.metadata["closed_windows"]):
                    dense, coords, ids = encode_window(record, w)
                    flat = dense.reshape(-1)
                    nonzero = np.flatnonzero(flat).astype(np.int32)
                    encoded = flat[nonzero].copy()
                    reconstructed = np.zeros_like(flat)
                    reconstructed[nonzero] = encoded
                    if not np.array_equal(reconstructed.view(np.uint32), flat.view(np.uint32)):
                        raise AssertionError("Sparse encoder changed FP32 bits")
                    indices.append(nonzero)
                    values.append(encoded)
                    maps.append(np.stack(coords).astype(np.int32))
                    cuts.append(cuts[-1] + len(ids))
                    pointers.append(pointers[-1] + len(nonzero))
                if cuts[-1] != len(record.arrays["t"]):
                    raise AssertionError("Closed prefix coverage mismatch")
                payload = {"indices": np.concatenate(indices), "values": np.concatenate(values),
                           "maps": np.concatenate(maps, axis=1), "cuts": np.asarray(cuts, np.int64),
                           "pointers": np.asarray(pointers, np.int64)}
                filename = hashlib.sha256(name.encode()).hexdigest()[:20] + ".npz"
                target = directory / filename
                with target.with_suffix(".tmp").open("wb") as f:
                    np.savez(f, **payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(target.with_suffix(".tmp"), target)
                rows[name] = {"sequence": name, "role": role, "file": filename, "sha256": sha256(target),
                              "windows": record.metadata["closed_windows"], "events": cuts[-1],
                              "bytes": target.stat().st_size, "bitwise_checked_windows": len(indices)}
                manifest.update(rows=list(rows.values()), status="preparing")
                atomic_json(manifest_path, manifest)
                print(json.dumps({"stage": "prepare", "sequences": len(rows), "total": 89,
                                  "seconds": time.perf_counter()-started}), flush=True)
        manifest.update(status="complete", total_bytes=sum(r["bytes"] for r in rows.values()),
                        bitwise_checked_windows=sum(r["bitwise_checked_windows"] for r in rows.values()))
        atomic_json(manifest_path, manifest)
        return manifest


def schedule(store, role, batch_size, seed=42, epoch=0):
    names = store.names(role)
    if role == "train_main":
        names = [names[i] for i in np.random.default_rng(seed+epoch).permutation(len(names))]
    windows = {r["sequence"]: r["closed_windows"] for r in store.manifest["roles"][role]}
    pending = iter(names)
    slots = [None] * batch_size
    batches = []
    while True:
        reset = [False] * batch_size
        for i in range(batch_size):
            if slots[i] is None:
                for n in pending:
                    if windows[n]:
                        slots[i] = (n, 0)
                        reset[i] = True
                        break
        if all(s is None for s in slots):
            return batches
        batches.append((list(slots), reset))
        slots = [None if s is None or s[1]+1 == windows[s[0]] else (s[0], s[1]+1) for s in slots]


class ResidentData:
    def __init__(self, teacher_directory, directory, device, gpu_budget_mib=1536):
        self.store = SequenceStore(teacher_directory)
        self.directory = run_path(directory)
        self.device = device
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        m = self.manifest
        if (m["status"] != "complete" or m["identity"] != {"version": CACHE_VERSION,
                "encoder": ENCODER_VERSION, "teacher_manifest_sha256": sha256(self.store.directory/"manifest.json")}):
            raise ValueError("Invalid resident cache identity")
        expected = {(r, n) for r in ("train_main", "train_dev") for n in self.store.names(r)}
        if {(r["role"], r["sequence"]) for r in m["rows"]} != expected or len(m["rows"]) != len(expected):
            raise ValueError("Encoded cache role coverage differs")
        self.records, self.encoded, self.tensors = {}, {}, {}
        byte_count = 0
        for row in m["rows"]:
            n = row["sequence"]
            path = self.directory / row["file"]
            if Path(row["file"]).name != row["file"] or sha256(path) != row["sha256"]:
                raise ValueError("Encoded cache file changed")
            self.records[n] = self.store.get(row["role"], n)
            with np.load(path, allow_pickle=False) as archive:
                a = {k: archive[k].copy() for k in archive.files}
            self.encoded[n] = a
            raw = self.records[n].arrays
            arrays = {k: a[k] for k in ("indices", "values", "maps")}
            arrays.update({k: raw[k] for k in ("label", "candidate", "teacher")})
            converted = {}
            for k, v in arrays.items():
                t = torch.from_numpy(v.copy())
                if k in ("indices", "maps"):
                    t = t.long()
                byte_count += t.numel()*t.element_size()
                if device.type == "cuda" and byte_count > gpu_budget_mib*1024**2:
                    raise MemoryError("Resident cache exceeds explicit GPU budget; no partial silent fallback")
                converted[k] = t.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else t
            self.tensors[n] = converted
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.stats = {"resident_tensor_bytes": byte_count, "host_encoded_bytes": sum(
            a.nbytes for r in self.encoded.values() for a in r.values()),
            "host_raw_bytes": sum(r.nbytes for r in self.records.values()),
            "placement": str(device), "per_batch_disk_reads": 0,
            "per_batch_input_host_to_device_bytes": 0, "sequence_count": len(expected)}

    def batches(self, role, batch_size=2, seed=42, epoch=0, start=0):
        plan = schedule(self.store, role, batch_size, seed, epoch)
        if not 0 <= start <= len(plan):
            raise ValueError("Invalid checkpoint batch cursor")
        for number in range(start, len(plan)):
            yield number, self.batch(plan[number]), len(plan)

    def batch(self, specification):
        slots, reset = specification
        b = len(slots)
        inputs = torch.zeros((STEPS, b, 64, ROWS, COLS), device=self.device)
        cpu_parts = {k: [] for k in ("labels", "candidate", "teacher", "maps")}
        device_parts = {k: [] for k in ("labels", "candidate", "teacher", "maps")}
        refs = []
        active = np.asarray([s is not None for s in slots], bool)
        for slot, s in enumerate(slots):
            if s is None:
                continue
            name, w = s
            a, t = self.encoded[name], self.tensors[name]
            lo, hi = a["cuts"][w:w+2]
            left, right = a["pointers"][w:w+2]
            i = t["indices"][left:right]
            global_i = (i // D * b + slot)*D + i % D
            inputs.view(-1).index_copy_(0, global_i, t["values"][left:right])
            mc, mg = a["maps"][:, lo:hi], t["maps"][:, lo:hi]
            cpu_parts["maps"].append(np.vstack((mc[:1], np.full((1, hi-lo), slot, np.int64), mc[1:])))
            device_parts["maps"].append(torch.cat((mg[:1], mg.new_full((1, hi-lo), slot), mg[1:])))
            for dst, src in (("labels", "label"), ("candidate", "candidate"), ("teacher", "teacher")):
                cpu_parts[dst].append(self.records[name].arrays[src][lo:hi])
                device_parts[dst].append(t[src][lo:hi])
            refs.append({"sequence": name, "slot": slot, "window": w, "ids": np.arange(lo, hi)})
        result: dict[str, Any] = {k: np.concatenate(v, axis=1 if k == "maps" else 0) for k, v in cpu_parts.items()}
        result["device"] = {k: torch.cat(v, dim=1 if k == "maps" else 0) for k, v in device_parts.items()}
        result.update(inputs=inputs, reset=np.asarray(reset, bool), active=active, refs=refs)
        return result
