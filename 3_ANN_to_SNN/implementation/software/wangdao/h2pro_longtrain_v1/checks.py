"""Real CUDA equivalence, recovery, hard-prefix, coverage and pipeline checks.

Optimizer updates in recovery tests affect disposable model copies only. The
benchmark uses lr=0. No formal run or preserved checkpoint is modified.
"""
import copy
import gc
import itertools
import json
from pathlib import Path
import time
from typing import Any, cast

import numpy as np
import torch

from ..h2pro_snn.training import setup_device, cpu_weights, memory_stats, sync
from ..h2pro_snn.stream_data import encode_window, stream_batches, prefetch, atomic_json
from ..h2pro_ann_readout_v1.audit import HARD_PREFIXES
from .data import ResidentData, schedule
from .models import ARMS, profile_for, make_model, training_module, sources
from .runtime import (legacy_origin, monitor_for, forward, update_step, accumulator,
                      evaluate, final_health, gradient_norms)
from .storage import all_rng, set_rng, cpu_tree, save_atomic, load


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def old_device_batch(batch, device):
    result = dict(batch)
    result["inputs"] = torch.from_numpy(batch["inputs"]).to(device)
    result["device"] = {k: torch.as_tensor(batch[k], device=device) for k in ("maps", "labels", "candidate", "teacher")}
    return result


def new_model(arm, data):
    p = profile_for(arm)
    origin, _ = legacy_origin(arm, data, data.device, p)
    rows = data.store.manifest["roles"]["train_main"]
    prior = sum(r["candidate_targets"] for r in rows)/sum(r["candidate_events"] for r in rows)
    torch.manual_seed(42)
    model = make_model(arm, p, prior).to(data.device)
    if origin:
        model.load_state_dict(origin["model"])
    return model, origin


def optimizer_for(model, config, origin=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if origin:
        optimizer.load_state_dict(origin["optimizer"])
    return optimizer


def recovery_and_update_parity(data, arm, output):
    config = profile_for(arm)["training"]
    models = []
    for _ in range(3):
        model, origin = new_model(arm, data)
        models.append([model, optimizer_for(model, config, origin), None, monitor_for(arm, config), accumulator(model)])
    plan = schedule(data.store, "train_main", 2, epoch=2)
    legacy = stream_batches(data.store, "train_main", 2, seed=42, epoch=2)
    for i in range(8):
        fast = data.batch(plan[i])
        slow = old_device_batch(next(legacy), data.device)
        for key in ("inputs", "maps", "labels", "candidate", "teacher", "reset", "active"):
            if not equal(fast[key], slow[key]):
                raise AssertionError("Input changed: " + key)
        for j, batch in enumerate((slow, fast, fast)):
            m, opt, state, monitor, total = models[j]
            models[j][2] = update_step(m, batch, state, opt, arm, config, monitor, total)
        if i == 2:
            m, opt, state, monitor, total = models[2]
            save_atomic(output/f"{arm}_recovery_test.pt", {"model": m.state_dict(), "optimizer": opt.state_dict(),
                        "state": state, "monitor": monitor.__dict__, "total": total, "rng": all_rng(), "cursor": i+1})
            saved = load(output/f"{arm}_recovery_test.pt")
            replacement, _ = new_model(arm, data)
            replacement.load_state_dict(saved["model"])
            replacement_opt = optimizer_for(replacement, config)
            replacement_opt.load_state_dict(saved["optimizer"])
            models[2] = [replacement, replacement_opt, tuple(s.to(data.device) for s in saved["state"]),
                         monitor_for(arm, config, saved["monitor"]), saved["total"]]
            set_rng(saved["rng"])
    baseline = models[0]
    for other in models[1:]:
        for a, b in ((baseline[0].state_dict(), other[0].state_dict()), (baseline[1].state_dict(), other[1].state_dict()),
                     (baseline[2], other[2]), (baseline[3].__dict__, other[3].__dict__), (baseline[4], other[4])):
            if not equal(a, b):
                raise AssertionError("Legacy/cached/recovered training differs")
    return {"batches": 8, "interrupted_after": 3, "bit_exact": True,
            "scope": "inputs, logits-dependent metrics, model, optimizer, stream state, monitor, loss accumulators",
            "disposable_optimizer_steps_per_copy": 8, "formal_training_started": False}


def hard_checks(data, arm):
    model, _ = new_model(arm, data)
    config = profile_for(arm)["training"]
    weights = cpu_weights(model)
    results = []
    for number, last in HARD_PREFIXES.items():
        name = "train/train_"+number+".npz"
        state = None
        monitor = monitor_for(arm, config)
        for window in range(last+1):
            batch = data.batch(([(name, window)], [window == 0]))
            model.zero_grad(set_to_none=True)
            logits, state, rates, _ = forward(model, batch, state)
            monitor.observe(model, logits, batch["candidate"], batch["active"])
            # Backpropagate only at the reported difficult window, with the full causal prefix.
            if window == last:
                mask = batch["device"]["candidate"]
                loss, _ = training_module(arm).task_loss(logits[mask], batch["device"]["labels"][mask],
                    batch["device"]["teacher"][mask], rates, config, model.voltage_tail())
                monitor.observe_loss(loss)
                loss.backward()
                g = gradient_norms(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                monitor.observe_gradients(norm, g, config["gradient_clip"])
            state = model.detach_state(state)
        results.append({"sequence": name, "windows": last+1, "stability": monitor.summary()})
    if not equal(weights, cpu_weights(model)):
        raise AssertionError("Read-only checks changed weights")
    return results


def benchmark(data, count=96):
    arm = "ann_linear"
    cfg = profile_for(arm)["training"]
    results = {"legacy_prefetch": [], "resident_gpu": []}
    # AB/BA ordering limits warmup/temperature bias; lr=0 preserves parameters.
    for source in ("legacy_prefetch", "resident_gpu", "resident_gpu", "legacy_prefetch"):
        model, origin = new_model(arm, data)
        optimizer = optimizer_for(model, cfg, origin)
        for group in optimizer.param_groups:
            group["lr"] = 0.
        state, totals, monitor = None, accumulator(model), monitor_for(arm, cfg)
        before = cpu_weights(model)
        plan = schedule(data.store, "train_main", 2, epoch=2)
        iterator = prefetch(iter(itertools.islice(stream_batches(data.store, "train_main", 2, epoch=2), count+8)))
        sync(data.device)
        measured_start = 0.
        input_sum = 0.
        for i in range(count+8):
            if i == 8:
                sync(data.device)
                measured_start = time.perf_counter()
                input_sum = 0.
            t = time.perf_counter()
            batch = old_device_batch(next(iterator)[0], data.device) if source == "legacy_prefetch" else data.batch(plan[i])
            input_time = time.perf_counter()-t
            if i >= 8:
                input_sum += input_time
            state = update_step(model, batch, state, optimizer, arm, cfg, monitor, totals)
        sync(data.device)
        elapsed = time.perf_counter()-measured_start
        cast(Any, iterator).close()
        if not equal(before, cpu_weights(model)):
            raise AssertionError("Benchmark updated weights")
        results[source].append({"batches": count, "seconds": elapsed, "batches_per_s": count/elapsed,
                                "input_dispatch_cpu_s": input_sum,
                                "input_note": "CPU dispatch/wait only; GPU scatter time included in total wall time"})
    old = float(np.median([r["seconds"] for r in results["legacy_prefetch"]]))
    new = float(np.median([r["seconds"] for r in results["resident_gpu"]]))
    return {"runs": results, "median_speedup": old/new, "resident_batch_ms": new/count*1000,
            "formal_parameter_updates": 0, "memory": memory_stats(data.device)}


def check(teacher, encoded, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = setup_device("cuda", profile_for("ann_linear")["training"])
    data = ResidentData(teacher, encoded, device)
    result = {"status": "checking", "sources": sources(), "cache": data.stats, "checks": {}, "training_started": False}
    atomic_json(output, result)
    for arm in ARMS:
        print(json.dumps({"stage": "cuda_parity_and_recovery", "arm": arm}), flush=True)
        result["checks"][arm] = {"recovery": recovery_and_update_parity(data, arm, output.parent),
                                 "hard_prefixes": hard_checks(data, arm)}
        model, origin = new_model(arm, data)
        config = profile_for(arm)["training"]
        replay = evaluate(model, data, arm, config, fixed_threshold=origin["report"]["best"]["threshold"] if origin else None)
        if origin and replay["selected"] != origin["report"]["best"]["evaluation"]["selected"]:
            raise AssertionError("Original epoch-1 Dev metrics changed")
        result["checks"][arm]["dev"] = replay
        print(json.dumps({"stage": "full_fixed_weight_train_health", "arm": arm}), flush=True)
        result["checks"][arm]["full_train_health"] = final_health(model, data, arm, config)
        if result["checks"][arm]["full_train_health"]["hard_guard_batches"]:
            raise AssertionError("Initial checkpoint has full-stream hard outliers")
        atomic_json(output, result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"stage": "throughput_benchmark"}), flush=True)
    result["benchmark"] = benchmark(data)
    result.update(status="passed", memory=memory_stats(device))
    atomic_json(output, result)
    return {"status": "passed", "cache": result["cache"], "benchmark": result["benchmark"], "output": str(output)}
