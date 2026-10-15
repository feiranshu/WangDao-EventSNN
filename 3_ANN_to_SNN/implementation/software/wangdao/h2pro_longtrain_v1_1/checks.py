"""Replay real failed batches and recovery on disposable copies, never formal runs."""
import copy
import gc
import json
from pathlib import Path
from typing import Any

import torch

from ..h2pro_snn.foundation import sha256
from ..h2pro_snn.stream_data import atomic_json
from ..h2pro_snn.training import setup_device, cpu_weights, memory_stats
from ..h2pro_longtrain_v1.data import ResidentData, schedule
from ..h2pro_longtrain_v1.storage import load, save_atomic, set_rng, PauseControl, cpu_tree
from ..h2pro_longtrain_v1.runtime import update_step
from .models import ARMS, POLICY, profile_for, make_model, sources
from .migration import migrate_suite, tree_equal, finite_tensors
from .monitor import monitor_for, emit_advisories
from .runtime import run_arm, evaluate, final_health


def replay_failure(data, arm, saved, diagnostic, directory):
    """Reconstruct the old failing pre-update state, then actually update past it."""
    # AdamW can share the CPU step tensor passed to load_state_dict.
    # Protect the source payload also in memory, before preparing further copies.
    saved = copy.deepcopy(saved)
    profile = profile_for(arm)
    config = profile["training"]
    model = make_model(arm, profile, saved["report"]["foreground_prior"]).to(data.device)
    model.load_state_dict(saved["model"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    optimizer.load_state_dict(saved["optimizer"])
    state = tuple(v.to(data.device) for v in saved["stream_state"])
    monitor = monitor_for(arm, config, saved["monitor"])
    totals = copy.deepcopy(saved["totals"])
    set_rng(saved["rng"])
    cursor, failure_cursor = saved["cursor"], diagnostic["failure"]["cursor"]
    if cursor["phase"] != "train" or cursor["epoch"] != failure_cursor["epoch"]:
        raise ValueError("Failure replay needs the matching mid-epoch checkpoint")
    plan = schedule(data.store, "train_main", config["batch_size"], config["seed"], cursor["epoch"])
    failed_index = failure_cursor["batch"]
    if not cursor["batch"] <= failed_index < len(plan):
        raise ValueError("Failure cursor does not follow the resumable checkpoint")
    end = min(failed_index+21, len(plan))
    result = None
    for i in range(cursor["batch"], end):
        if i == failed_index:
            if not tree_equal(cpu_weights(model), diagnostic["model"]):
                raise AssertionError("Replayed weights differ before the original failed batch")
            if not tree_equal(cpu_tree(state), diagnostic["state_before"]):
                raise AssertionError("Replayed stream state differs before the original failed batch")
        batch = data.batch(plan[i])
        state = update_step(model, batch, state, optimizer, arm, config, monitor, totals)
        if i == failed_index:
            if not tree_equal(cpu_tree(model.last_diagnostics), diagnostic["layer_diagnostics"]):
                raise AssertionError("Original failing forward diagnostics did not replay exactly")
            result = {"original_error": diagnostic["failure"]["error"],
                      "epoch": cursor["epoch"], "batch_one_based": i+1,
                      "pre_failure_model_state_and_diagnostics_bit_exact": True,
                      "loss": monitor.last_loss, "gradient_norm_before_clip": monitor.last_gradient_norm,
                      "advisories": copy.deepcopy(monitor.advisory_maxima),
                      "former_failed_batch_optimizer_update_completed": True}
        emit_advisories(monitor, directory, phase="disposable_failure_replay",
                       epoch=cursor["epoch"], batch=i+1)
    if result is None or not all(finite_tensors(v) for v in (model.state_dict(), optimizer.state_dict(), state)):
        raise AssertionError("Post-failure replay did not finish with finite state")
    result.update(disposable_updates=end-cursor["batch"], later_batches_completed=end-failed_index-1,
                  end_cursor_batch=end, all_final_state_finite=True, stability=monitor.summary())
    return result


def runtime_recovery(data, arm, saved, failed_index, directory):
    """Native CUDA runtime: continuous vs pause just after the former failure."""
    continuous, split = directory/"continuous", directory/"split"
    save_atomic(continuous/"last.pt", saved)
    save_atomic(split/"last.pt", saved)
    end, pause_at = failed_index+11, failed_index+1
    with PauseControl(directory) as pause:
        for dest, stop in ((continuous, end), (split, pause_at), (split, end)):
            report = run_arm(data, dest, arm, 30, pause, stop_after_batches=stop)
            if report["status"] != "paused":
                raise AssertionError("Disposable native runtime failed instead of reaching its test boundary")
    a, b = load(continuous/"last.pt"), load(split/"last.pt")
    for key in ("model", "optimizer", "scheduler", "rng", "cursor", "stream_state", "best_model", "monitor"):
        if not tree_equal(a[key], b[key]):
            raise AssertionError("Native interrupted recovery mismatch: "+key)
    excluded = {"train_s", "input_s", "checkpoint_s"}
    for key in a["totals"].keys()-excluded:
        if not tree_equal(a["totals"][key], b["totals"][key]):
            raise AssertionError("Native recovered accumulators differ: "+key)
    if a["cursor"]["batch"] != end or a["report"]["history"] != saved["report"]["history"]:
        raise AssertionError("Native replay advanced unexpected epoch state")
    return {"bit_exact": True, "pause_after_batch": pause_at, "end_batch": end,
            "scope": "weights, optimizer, scheduler, RNG, cursor, stream state, best model, monitor, non-time accumulators",
            "formal_training_started": False}


def check(teacher, encoded, source_suite, output):
    output, source_suite = Path(output), Path(source_suite)
    work = output.parent/(output.stem+"_work")
    if work.exists():
        raise ValueError("Choose a fresh check path; disposable outputs are preserved")
    work.mkdir(parents=True)
    before = {str(p): sha256(p) for p in source_suite.rglob("*")
              if p.is_file() and not p.name.startswith(".")}
    device = setup_device("cuda", profile_for("ann_linear")["training"])
    data = ResidentData(teacher, encoded, device)
    result = {"status": "checking", "sources": sources(), "monitoring_policy": POLICY,
              "formal_training_started": False, "cache": data.stats, "controls": {}}
    atomic_json(output, result)
    result["migration"] = migrate_suite(data, source_suite, work/"migrated")
    for arm in ARMS:
        print(json.dumps({"stage": "real_failure_replay_and_recovery", "arm": arm}), flush=True)
        dest = work/arm
        dest.mkdir()
        saved = load(work/"migrated"/arm/"last.pt")
        diagnostic = load(source_suite/arm/"failure_context.pt")
        row: dict[str, Any] = {"failure_replay": replay_failure(data, arm, saved, diagnostic, dest),
               "native_runtime_recovery": runtime_recovery(data, arm, saved,
                    diagnostic["failure"]["cursor"]["batch"], dest)}
        result["controls"][arm] = row
        atomic_json(output, result)
        print(json.dumps({"stage": "full_dev_best_replay", "arm": arm}), flush=True)
        profile = profile_for(arm)
        model = make_model(arm, profile, saved["report"]["foreground_prior"]).to(device)
        model.load_state_dict(saved["best_model"])
        row["dev_best_replay"] = evaluate(model, data, arm, profile["training"],
                fixed_threshold=saved["report"]["best"]["threshold"], log_directory=dest)
        if row["dev_best_replay"]["selected"] != saved["report"]["best"]["evaluation"]["selected"]:
            raise AssertionError("Unchanged historical best weights did not reproduce their Dev metrics")
        row["dev_best_replay_exact"] = True
        atomic_json(output, result)
        print(json.dumps({"stage": "full_train_fixed_latest_weights", "arm": arm}), flush=True)
        model.load_state_dict(saved["model"])
        row["fixed_latest_weights_health"] = final_health(model, data, arm, profile["training"], log_directory=dest)
        if not tree_equal(cpu_weights(model), saved["model"]):
            raise AssertionError("Read-only full-stream check changed parameters")
        row["memory"] = memory_stats(device)
        atomic_json(output, result)
        del model, saved, diagnostic
        gc.collect()
        torch.cuda.empty_cache()
    if any(sha256(Path(path)) != digest for path, digest in before.items()):
        raise AssertionError("Preserved source training files changed")
    if sources() != result["sources"]:
        raise AssertionError("Source code changed during the checks")
    result.update(status="passed", preserved_source_result_files=len(before), memory=memory_stats(device))
    atomic_json(output, result)
    return {"status": "passed", "output": str(output), "formal_training_started": False}
