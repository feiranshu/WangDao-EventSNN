"""Validated v1 -> v1.1 migration; all learned state is copied without updates."""
import copy
from pathlib import Path

import numpy as np
import torch

from ..h2pro_snn.foundation import sha256
from ..h2pro_snn.stream_data import run_path, atomic_json
from ..h2pro_snn.training import environment
from ..h2pro_longtrain_v1.storage import load, save_atomic, process_lock, cpu_tree, progress
from ..h2pro_longtrain_v1.data import schedule
from ..h2pro_longtrain_v1.models import sources as old_sources, profile_for as old_profile
from ..h2pro_longtrain_v1.runtime import legacy_origin
from .models import ARMS, POLICY, sources, profile_for
from .monitor import monitor_for


STATE_KEYS = ("model", "optimizer", "scheduler", "rng", "best_model", "cursor", "stream_state", "totals")


def tree_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def finite_tensors(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True


def make_identity(data, arm, profile, provenance):
    return {"runtime": "h2pro_longtrain_v1_1", "arm": arm, "profile": profile, "sources": sources(),
            "environment": environment(data.device), "monitoring_policy": POLICY,
            "encoded_manifest_sha256": sha256(data.directory/"manifest.json"),
            "teacher_manifest_sha256": sha256(data.store.directory/"manifest.json"),
            "migration": provenance, "input_pipeline": "resident sparse GPU; unchanged v1 encoding",
            "checkpoint_contract": "completed batch cursor + detached stream state + complete optimizer/RNG/monitor"}


def validate_source(saved, data, arm):
    required = {*STATE_KEYS, "identity", "report", "monitor"}
    if not required.issubset(saved):
        raise ValueError("Source is not a complete resumable checkpoint (diagnostic snapshots are unsupported)")
    profile = old_profile(arm)
    _, origin = legacy_origin(arm, data, data.device, profile)
    expected = {"runtime": "h2pro_longtrain_v1", "arm": arm, "profile": profile, "sources": old_sources(),
                "environment": environment(data.device), "encoded_manifest_sha256": sha256(data.directory/"manifest.json"),
                "teacher_manifest_sha256": sha256(data.store.directory/"manifest.json"), "origin": origin,
                "input_pipeline": "resident sparse GPU inputs; original numerical profile preserved",
                "checkpoint_contract": "completed batch cursor + detached stream state + complete optimizer/RNG/monitor"}
    if saved["identity"] != expected or saved["report"]["identity"] != expected:
        raise ValueError("Source code/profile/cache/environment/arm identity mismatch")
    r, c = saved["report"], saved["cursor"]
    if r.get("disposable_check") or r["arm"] != arm:
        raise ValueError("Cannot migrate disposable checks as formal training")
    if (r["completed_epochs"] != len(r["history"]) or c["epoch"] != r["completed_epochs"]+1
            or saved["scheduler"]["last_epoch"] != r["completed_epochs"]):
        raise ValueError("Checkpoint epoch/history/scheduler mismatch")
    if c["phase"] not in ("train", "evaluate", "audit", "complete"):
        raise ValueError("Unsupported cursor phase")
    if c["phase"] in ("train", "evaluate"):
        plan = schedule(data.store, "train_main", profile["training"]["batch_size"],
                        profile["training"]["seed"], c["epoch"])
        if not 0 <= c["batch"] <= len(plan):
            raise ValueError("Checkpoint batch cursor is outside this epoch")
        if c["batch"]:
            if saved["totals"] is None or saved["monitor"] is None:
                raise ValueError("Missing mid-epoch training accumulators")
            windows = events = 0
            for slots, _ in plan[:c["batch"]]:
                for item in slots:
                    if item is not None:
                        name, w = item
                        lo, hi = data.encoded[name]["cuts"][w:w+2]
                        events += int(hi-lo)
                        windows += 1
            if (saved["totals"]["batches"] != c["batch"] or saved["totals"]["windows"] != windows
                    or saved["totals"]["events"] != events):
                raise ValueError("Batch cursor and accumulated event coverage differ")
            if c["phase"] == "train":
                channels = (profile["model"]["stem_channels"], profile["model"]["hidden_channels"], 128, 1)
                expected_shapes = [(profile["training"]["batch_size"], n, profile["model"]["rows"],
                                    profile["model"]["columns"]) for n in channels]
                if saved["stream_state"] is None or [tuple(v.shape) for v in saved["stream_state"]] != expected_shapes:
                    raise ValueError("Missing or incompatible cross-window state")
        if c["phase"] == "evaluate" and c["batch"] != len(plan):
            raise ValueError("Evaluation phase before training coverage completed")
    if not all(finite_tensors(saved[k]) for k in ("model", "optimizer", "best_model", "stream_state")):
        raise FloatingPointError("Source checkpoint contains nonfinite learned state")
    return expected


def migrate_payload(saved, identity):
    result = copy.deepcopy(saved)
    result["identity"] = identity
    result["report"].update(identity=identity, status="ready", monitoring_policy=POLICY,
                           migration=identity["migration"], launcher="prepared_for_user")
    if saved["monitor"] is not None:
        monitor = monitor_for(identity["arm"], identity["profile"]["training"], saved["monitor"])
        result["monitor"] = cpu_tree(monitor.__dict__)
    for k in STATE_KEYS:
        if not tree_equal(saved[k], result[k]):
            raise AssertionError("Migration changed learned state: " + k)
    if not tree_equal(saved["report"]["history"], result["report"]["history"]):
        raise AssertionError("Migration changed historical evaluations")
    return result


def migrate_suite(data, source, target):
    source, target = run_path(source), run_path(target)
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Migration requires independent source and target run directories")
    # OS locks also reject a source that is still training. No source checkpoint is written.
    with process_lock(source/".suite.lock"), process_lock(target/".suite.lock"):
        rows = {}
        for arm in ARMS:
            with process_lock(source/arm/".run.lock"), process_lock(target/arm/".run.lock"):
                path = source/arm/"last.pt"
                saved = load(path)
                validate_source(saved, data, arm)
                provenance = {"source_runtime": "h2pro_longtrain_v1", "source_checkpoint": str(path),
                              "source_checkpoint_sha256": sha256(path), "source_cursor": saved["cursor"],
                              "source_completed_epochs": saved["report"]["completed_epochs"],
                              "transition": "Only finite-value stop policy changes; no optimizer steps or resets"}
                identity = make_identity(data, arm, profile_for(arm), provenance)
                dest = target/arm
                last = dest/"last.pt"
                if last.exists():
                    if load(last)["identity"] != identity:
                        raise ValueError("Existing destination belongs to a different migration")
                    # Idempotent preparation must never overwrite subsequent training.
                    rows[arm] = {"status": "already_exists", "source_cursor": saved["cursor"]}
                    continue
                if any(p.name not in (".run.lock", "last.pt.tmp") for p in dest.iterdir()):
                    raise ValueError("Unknown files in destination; preserve it and select another directory")
                migrated = migrate_payload(saved, identity)
                save_atomic(last, migrated)
                replay = load(last)
                for key in STATE_KEYS:
                    if not tree_equal(saved[key], replay[key]):
                        raise AssertionError("Serialized migration mismatch: " + key)
                atomic_json(dest/"training.json", replay["report"])
                progress(dest, status="ready", arm=arm, cursor=replay["cursor"],
                         completed_epochs=replay["report"]["completed_epochs"], requested_epochs=30,
                         checkpoint=str(last), monitoring_policy=POLICY)
                rows[arm] = {"status": "migrated_without_updates", "cursor": replay["cursor"],
                             "completed_epochs": replay["report"]["completed_epochs"],
                             "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
                             "model_optimizer_rng_stream_state_exact": True}
        result = {"status": "ready", "formal_training_started": False, "source": str(source),
                  "target": str(target), "monitoring_policy": POLICY, "controls": rows}
        # Do not reset root progress when rerunning prepare after actual training began.
        if any(r["status"] == "migrated_without_updates" for r in rows.values()):
            atomic_json(target/"migration.json", result)
            progress(target, status="ready", monitoring_policy=POLICY, controls=rows)
        return result
