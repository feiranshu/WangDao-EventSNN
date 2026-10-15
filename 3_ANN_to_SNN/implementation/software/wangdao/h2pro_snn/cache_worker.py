"""Run only in the frozen ANN subprocess; prepare immutable development caches."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import time

import numpy as np

from .foundation import ANN_ROOT, role_records, sha256, verified_sequence, verify_baseline
from .stream_data import CACHE_VERSION, SequenceStore, atomic_json, run_path, validate_sequence


def prepare(dataset_root, output, limit=None, max_windows=None):
    stream_state = import_module("wangdao.h2_pro.stream_state")
    from wangdao.h2 import representation
    from wangdao.data import load_events
    expected = ANN_ROOT / "software/wangdao"
    if any(module.__file__ is None or Path(module.__file__).absolute().parent.parent != expected
           for module in (stream_state, representation)):
        raise RuntimeError("Must use the frozen ANN implementation")
    if any(v is not None and (type(v) is not int or v < 1) for v in (limit, max_windows)):
        raise ValueError("Limits must be positive integers")
    baseline = verify_baseline()
    output = run_path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        existing = SequenceStore(output, 0).manifest
        if existing["sequence_limit"] != limit or existing["max_windows"] != max_windows:
            raise ValueError("Existing complete cache has different limits; choose a new directory")
        print(json.dumps({"stage": "prepare", "status": "reused_verified_cache", "cache": str(output)}), flush=True)
        return existing
    lock = output / ".prepare.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    manifest = {"version": CACHE_VERSION, "status": "preparing", "training_started": False,
                "baseline_lock_sha256": baseline["baseline_lock_sha256"], "baseline": baseline,
                "sequence_limit": limit, "max_windows": max_windows,
                "complete_development_roles": limit is None and max_windows is None,
                "roles": {"train_main": [], "train_dev": []}, "reserved_sequences_read": 0}
    # Limits >= the full role count are also a complete cache.
    manifest["complete_development_roles"] = (max_windows is None and
        all(len(role_records(r, limit)) == len(role_records(r)) for r in manifest["roles"]))
    try:
        for role in manifest["roles"]:
            for record in role_records(role, limit):
                path = verified_sequence(dataset_root, record)
                identity = {"version": CACHE_VERSION, "role": role, "sequence": record["path"],
                            "source_sha256": record["sha256"], "baseline_lock_sha256": manifest["baseline_lock_sha256"],
                            "max_windows": max_windows}
                target = output / (role + "_" + path.name)
                if target.exists():
                    with np.load(target, allow_pickle=False) as archive:
                        meta = json.loads(str(archive["metadata"].item()))
                        arrays = {k: archive[k].copy() for k in archive.files if k != "metadata"}
                    if any(meta.get(k) != v for k, v in identity.items()):
                        raise ValueError("Existing cache has different identity; choose a new directory")
                    validate_sequence(arrays, meta)
                else:
                    ev = load_events(path, require_labels=True)
                    t0 = float(ev["t"][0])
                    final = float(ev["t"][-1])+1e-6
                    watermark = min(final, t0+max_windows*100.0) if max_windows else final
                    count = int(np.searchsorted(ev["t"], watermark, side="left"))
                    packet = np.empty(count, dtype=stream_state.EVENT_DTYPE)
                    for field in ("x", "y", "t", "p"):
                        packet[field] = ev[field][:count]
                    pipeline = stream_state.CausalPipeline(
                        ANN_ROOT / "software/weights/H2_D1_seed42.npz",
                        ANN_ROOT / "software/weights/H2-pro-Causal-ANN-v1_int8.npz",
                        variant="C1-proposal-motion", threshold=0.35, initial_refiner=False)
                    pipeline.reset(t0)
                    pipeline.push(packet, watermark)
                    windows, trailing = pipeline.finalize()
                    ids = np.concatenate([w.event_indices for w in windows]) if windows else np.empty(0, np.int64)
                    if not np.array_equal(ids, np.arange(count-trailing)):
                        raise AssertionError("Frozen Proposal did not publish a unique closed prefix")
                    arrays = {key: ev[key][ids].copy() for key in ("x", "y", "p", "t", "label")}
                    arrays["candidate"] = np.concatenate([w.candidate_mask for w in windows]) if windows else np.empty(0, bool)
                    arrays["teacher"] = np.concatenate([w.probabilities for w in windows]) if windows else np.empty(0, np.float32)
                    elapsed = float(ev["t"][-1])-t0
                    rate = len(ev)*1000/elapsed if elapsed > 0 else 0.0
                    meta = {**identity, "t0_ms": t0, "closed_windows": len(windows),
                            "published_events": len(ids), "trailing_events": trailing, "input_events": count,
                            "original_events": len(ev), "beyond_prefix_events": len(ev)-count,
                            "sequence_events_per_second": rate, "high_density": rate >= 20000}
                    validate_sequence(arrays, meta)
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, x=arrays["x"], y=arrays["y"], t=arrays["t"],
                            p=arrays["p"], label=arrays["label"], candidate=arrays["candidate"],
                            teacher=arrays["teacher"], metadata=np.array(json.dumps(meta)))
                    os.replace(temporary, target)
                row = {"sequence": record["path"], "source_sha256": record["sha256"],
                       "cache_file": target.name, "cache_sha256": sha256(target),
                       "published_events": meta["published_events"], "closed_windows": meta["closed_windows"],
                       "candidate_events": int(arrays["candidate"].sum()),
                       "candidate_targets": int(arrays["label"][arrays["candidate"]].sum()),
                       "high_density": meta["high_density"]}
                manifest["roles"][role].append(row)
                print(json.dumps({"stage": "prepare", "role": role, **row}), flush=True)
        manifest.update(status="complete", elapsed_seconds=time.perf_counter()-started)
        atomic_json(output / "manifest.json", manifest)
        return manifest
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()
    prepare(args.dataset_root, args.cache, args.limit, args.max_windows)


if __name__ == "__main__":
    main()
