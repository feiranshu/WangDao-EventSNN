"""User-launched single-GPU training, evaluation and epoch-boundary recovery."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Union

import numpy as np
import torch
from torch.nn import functional as F

from .diagnostics import aggregate_counts, error_counts, summarize_counts
from .foundation import PACKAGE, read_json, research_sources, sha256, verify_baseline
from .refiner import RefinerConfig, StreamRefiner
from .stream_data import ENCODER_VERSION, SequenceStore, atomic_json, prefetch, run_path, stream_batches


def load_profile(path=None, mode=None, batch_size=None):
    profile = read_json(path or PACKAGE / "local_4060.json")
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported profile schema")
    if mode is not None:
        profile["model"]["mode"] = mode
    if batch_size is not None:
        profile["training"]["batch_size"] = batch_size
    profile["model"] = asdict(RefinerConfig(**profile["model"]))
    if (profile["model"]["rows"], profile["model"]["columns"]) != (68, 88):
        raise ValueError("The EV-UAV encoder requires the frozen 68x88 padded grid")
    t = profile["training"]
    for key in ("seed", "batch_size", "schedule_epochs", "cache_mib", "cpu_threads"):
        if type(t[key]) is not int or t[key] < (0 if key in ("seed", "cache_mib") else 1):
            raise ValueError("Invalid integer training option: " + key)
    if t["batch_size"] > 8 or t["cache_mib"] > 1024 or t["schedule_epochs"] > 30:
        raise ValueError("Local budget permits batch <=8, RAM cache <=1024 MiB, epochs <=30")
    for key in ("learning_rate", "minimum_learning_rate", "weight_decay", "gradient_clip",
                "kd_weight", "soft_iou_weight", "focal_gamma", "spike_weight", "spike_ceiling", "vram_limit_gib"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError("Invalid numeric training option: " + key)
    if (not 0 < t["minimum_learning_rate"] <= t["learning_rate"] or t["gradient_clip"] <= 0
            or not 0 < t["focal_alpha"] < 1 or not 0 < t["vram_limit_gib"] <= 6.5):
        raise ValueError("Invalid optimizer/loss/VRAM limits")
    thresholds = t["thresholds"]
    if not thresholds or thresholds != sorted(set(thresholds)) or any(not 0 < v < 1 for v in thresholds):
        raise ValueError("Thresholds must be sorted, unique and inside (0,1)")
    expected = read_json(PACKAGE / "local_4060.json")["execution"]
    if profile["execution"] != expected:
        raise ValueError("Execution contract is fixed; AMP/time/reset changes require a new protocol")
    return profile


def setup_device(requested, config):
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use C:\\Anaconda\\envs\\gazelle-train\\python.exe; no CPU fallback")
    if requested not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda explicitly")
    device = torch.device("cuda:0" if requested == "cuda" else "cpu")
    if requested == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, config["vram_limit_gib"]*1024**3/total), device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def environment(device):
    return {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def memory_stats(device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def task_loss(logits, labels, teacher, rates, config):
    """Supervision and KD use only admitted events. Refused events stay in metrics."""
    if logits.numel() == 0:
        raise ValueError("Loss requires at least one candidate")
    labels = labels.to(logits)
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = torch.where(labels > 0, probability, 1-probability)
    alpha = torch.where(labels > 0, config["focal_alpha"], 1-config["focal_alpha"])
    focal = (alpha*(1-pt).pow(config["focal_gamma"])*bce).mean()
    intersection = (probability*labels).sum()
    union = probability.sum()+labels.sum()-intersection
    soft_iou = 1-(intersection+1)/(union+1)
    kd = F.binary_cross_entropy_with_logits(logits, teacher.to(logits))
    spike = F.relu(rates-config["spike_ceiling"]).square().mean()
    loss = focal+config["soft_iou_weight"]*soft_iou+config["kd_weight"]*kd+config["spike_weight"]*spike
    return loss, {"focal": focal, "soft_iou": soft_iou, "kd": kd, "spike": spike}


def forward_batch(model, batch, state, device):
    inputs = torch.from_numpy(batch["inputs"]).to(device)
    if state is None:
        state = model.initial_state(inputs.shape[1])
    state = model.reset_slots(state, torch.from_numpy(batch["reset"]).to(device))
    output, state, rates, counts = model(inputs, state)
    indices = tuple(torch.as_tensor(a, dtype=torch.long, device=device) for a in batch["maps"])
    event_logits = output[indices]
    if not torch.isfinite(event_logits).all() or any(not torch.isfinite(s).all() for s in state):
        raise FloatingPointError("Nonfinite SNN output/state; run stopped")
    return event_logits, state, rates, counts


def cost_report(model, counts, stream_steps, executed_steps=None):
    c = model.config
    occupied, input_nonzeros, stem_spikes, hidden_spikes, moved = counts.tolist()
    cells = c.rows*c.columns
    executed_steps = stream_steps if executed_steps is None else executed_steps
    stem_mac = int(occupied)*64*c.stem_channels
    hidden_dense = executed_steps*cells*9*c.stem_channels*c.hidden_channels
    head_dense = executed_steps*cells*c.hidden_channels*128
    return {"scope": "SNN Refiner reference only; closed-prefix event encoding and frozen Proposal excluded",
            "dense_backend_synaptic_mac_estimate": stem_mac+hidden_dense+head_dense,
            "active_cell_stem_mac_estimate": stem_mac,
            "ideal_nonzero_input_stem_mac_upper_bound": int(input_nonzeros)*c.stem_channels,
            "ideal_unit_spike_ac_upper_bound": int(stem_spikes)*9*c.hidden_channels+int(hidden_spikes)*128,
            "reference_leak_scalar_multiplications": executed_steps*cells*(c.stem_channels+c.hidden_channels+128),
            "logical_stream_steps": stream_steps, "executed_stream_steps_including_batch_padding": executed_steps,
            "state_scalars_per_stream": model.architecture()["state_scalars_per_stream"],
            "moved_sources": int(moved), "spike_convolution_zero_skip_implemented": False,
            "ac_note": "Burst n counts as n unit-spike additions; 3x3 fanout includes padding (upper bound)",
            "count_basis": "Executed tensor shapes and observed spike counts; dense-convolution equivalent, not GPU instruction counters",
            "end_to_end_complete": False, "energy_measured": False,
            "unaccounted_components": ["frozen Proposal", "event encoding", "activity router",
                                       "threshold/reset/bias/max/probability operations", "memory traffic"]}


@torch.no_grad()
def evaluate(model, store, device, batch_size, thresholds, *, fixed_threshold=None):
    model.eval()
    records = {name: store.get("train_dev", name) for name in store.names("train_dev")}
    probabilities = {name: np.zeros(len(r.arrays["t"]), np.float32) for name, r in records.items()}
    seen = {name: np.zeros(len(r.arrays["t"]), bool) for name, r in records.items()}
    state, stream_steps, executed_steps = None, 0, 0
    counts = np.zeros(5, np.float64)
    started = time.perf_counter()
    # One fixed chronological pass. Threshold selection never reruns the network.
    for batch, _ in prefetch(iter(stream_batches(store, "train_dev", batch_size))):
        logits, state, _, activity = forward_batch(model, batch, state, device)
        values = logits.sigmoid().cpu().numpy()
        values[~batch["candidate"]] = 0
        offset = 0
        for ref in batch["refs"]:
            name, ids = ref["sequence"], ref["ids"]
            if seen[name][ids].any():
                raise AssertionError("Duplicate event prediction")
            probabilities[name][ids] = values[offset:offset+len(ids)]
            seen[name][ids] = True
            offset += len(ids)
        if offset != len(values):
            raise AssertionError("Unmapped event output")
        counts += activity[:, batch["active"]].sum((0, 1)).cpu().numpy()
        stream_steps += int(batch["active"].sum())*batch["inputs"].shape[0]
        executed_steps += batch["inputs"].shape[0]*batch["inputs"].shape[1]
    if any(not flags.all() for flags in seen.values()):
        raise AssertionError("Missing event prediction")

    def metrics(threshold, teacher=False):
        rows = []
        for name, r in records.items():
            a = r.arrays
            row = error_counts(a["label"], a["teacher"] if teacher else probabilities[name], a["candidate"], threshold)
            rows.append({"sequence": name, "high_density": r.metadata["high_density"], **summarize_counts(row)})
        return {"overall": aggregate_counts(rows),
                "high_density": aggregate_counts([r for r in rows if r["high_density"]]),
                "ordinary": aggregate_counts([r for r in rows if not r["high_density"]]), "sequences": rows}

    scans = [{"threshold": float(t), **metrics(float(t))} for t in thresholds]
    selected = max(scans, key=selection_key) if fixed_threshold is None else {"threshold": fixed_threshold, **metrics(fixed_threshold)}
    teacher = metrics(0.35, True)
    def delta(group, metric):
        a, b = selected[group][metric], teacher[group][metric]
        return a-b if a is not None and b is not None else None
    sync(device)
    return {"selected": selected, "scan": [{"threshold": s["threshold"], "overall": s["overall"],
                                            "high_density": s["high_density"]} for s in scans],
            "frozen_teacher": teacher,
            "delta_vs_teacher": {"iou": delta("overall", "iou"), "high_density_recall": delta("high_density", "recall")},
            "cost": cost_report(model, counts, stream_steps, executed_steps), "seconds": time.perf_counter()-started,
            "data_role": "train_dev", "complete_development_roles": store.manifest["complete_development_roles"],
            "event_coverage": "same frozen ANN closed prefix; rejected candidates included as background",
            "latency": "100 ms Proposal closure; 10 ms internal bins; no end-to-end timing claim"}


def selection_key(entry):
    def number(v):
        return -1.0 if v is None else v
    return (number(entry["overall"]["iou"]), number(entry["high_density"]["recall"]), -entry["threshold"])


def checkpoint_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
    os.replace(temporary, path)


def checkpoint_load(path, device: Union[str, torch.device] = "cpu") -> dict:
    # Only this tool's local checkpoints are supported; do not load unknown .pt files.
    return torch.load(path, map_location=device, weights_only=True)


def rng_state():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_epoch(model, store, optimizer, device, config, epoch):
    model.train()
    state = None
    start = time.perf_counter()
    waiting = total_loss = 0.0
    events = candidates = updates = windows = 0
    components = {k: 0.0 for k in ("focal", "soft_iou", "kd", "spike")}
    rates_sum = np.zeros(2, np.float64)
    batches = 0
    iterator = stream_batches(store, "train_main", config["batch_size"], seed=config["seed"], epoch=epoch)
    for batch, wait in prefetch(iter(iterator)):
        waiting += wait
        optimizer.zero_grad(set_to_none=True)
        logits, state, rates, _ = forward_batch(model, batch, state, device)
        events += len(batch["labels"])
        windows += int(batch["active"].sum())
        mask = torch.from_numpy(batch["candidate"]).to(device)
        n = int(batch["candidate"].sum())
        if n:
            target = torch.from_numpy(batch["labels"][batch["candidate"]].copy()).to(device)
            teacher = torch.from_numpy(batch["teacher"][batch["candidate"]].copy()).to(device)
            loss, parts = task_loss(logits[mask], target, teacher, rates[:, batch["active"]], config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            candidates += n
            total_loss += float(loss.detach())*n
            for key in components:
                components[key] += float(parts[key].detach())*n
        state = model.detach_state(state)
        rates_sum += rates[:, batch["active"]].detach().mean((0, 1)).cpu().numpy()
        batches += 1
        if batches % 50 == 0:
            print(json.dumps({"stage": "train", "epoch": epoch, "batch": batches, "events": events,
                              "input_wait_s": waiting, "elapsed_s": time.perf_counter()-start}), flush=True)
    if not updates:
        raise ValueError("No candidate training events in this cache")
    expected = sum(r["published_events"] for r in store.manifest["roles"]["train_main"])
    expected_windows = sum(r["closed_windows"] for r in store.manifest["roles"]["train_main"])
    if events != expected or windows != expected_windows:
        raise AssertionError("Training did not cover each closed-prefix event/window exactly once")
    sync(device)
    return {"epoch": epoch, "loss": total_loss/candidates,
            "loss_components": {k: v/candidates for k, v in components.items()},
            "mean_support_spikes_per_neuron_step": (rates_sum/batches).tolist(),
            "events": events, "candidate_events": candidates, "windows": windows, "updates": updates,
            "batches": batches, "input_wait_s": waiting, "train_s": time.perf_counter()-start,
            "cache": dict(store.stats), **memory_stats(device)}


def train(cache, directory, profile, device_name, epochs, *, resume=False, pilot=False):
    config = profile["training"]
    if not 1 <= epochs <= config["schedule_epochs"]:
        raise ValueError("Epoch budget must be within [1,schedule_epochs]")
    store = SequenceStore(cache, config["cache_mib"])
    if not store.manifest["complete_development_roles"] and not pilot:
        raise ValueError("Partial cache is pilot only; prepare all 79/10 sequences for normal training")
    directory = run_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    last = directory / "last.pt"
    if resume != last.exists():
        raise ValueError("Use --resume for an existing run; a new run must not use --resume")
    if not resume and any(directory.iterdir()):
        raise ValueError("New training requires an empty run directory")
    device = setup_device(device_name, config)
    identity = {"profile": profile, "teacher_cache_sha256": sha256(store.directory/"manifest.json"),
                "encoder": ENCODER_VERSION, "baseline": verify_baseline(),
                "sources": research_sources(), "environment": environment(device), "pilot": pilot}
    lock = directory / ".training.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        torch.manual_seed(config["seed"])
        main_rows = store.manifest["roles"]["train_main"]
        prior = sum(r["candidate_targets"] for r in main_rows)/max(sum(r["candidate_events"] for r in main_rows), 1)
        prior = float(np.clip(prior, 1e-4, 1-1e-4))
        model = StreamRefiner(RefinerConfig(**profile["model"]), prior).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["schedule_epochs"], eta_min=config["minimum_learning_rate"])
        report = {"schema_version": 1, "status": "running", "experiment_id": directory.name,
                  "stage": "single_refiner_control", "data_role": ["train_main", "train_dev"],
                  "training_started": True, "launcher": "explicit_manual_train_command", "pilot": pilot,
                  "identity": identity, "architecture": model.architecture(), "foreground_prior": prior,
                  "history": [], "best": None, "primary_metrics": {},
                  "decision": "research_only_no_automatic_promotion", "conclusion": "Training in progress."}
        best_weights = None
        if resume:
            saved = checkpoint_load(last, device)
            if saved["identity"] != identity:
                raise ValueError("Resume identity changed (code/cache/profile/environment); use a new run directory")
            report = saved["report"]
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            best_weights = {k: v.cpu() for k, v in saved["best_model"].items()} if saved["best_model"] else None
            restore_rng(saved["rng"])
        def save():
            checkpoint_save(last, {"identity": identity, "model": cpu_weights(model),
                                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                  "rng": rng_state(), "report": report, "best_model": best_weights})
            if best_weights is not None:
                checkpoint_save(directory/"best.pt", {"identity": identity, "model": best_weights,
                    "threshold": report["best"]["threshold"], "epoch": report["best"]["epoch"],
                    "foreground_prior": prior, "evaluation": report["best"]["evaluation"]})
            atomic_json(directory/"training.json", report)
        save()  # epoch 0 or recovery rebuild: last.pt is the atomic source of truth
        for epoch in range(len(report["history"])+1, epochs+1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            entry = train_epoch(model, store, optimizer, device, config, epoch)
            entry["learning_rate"] = optimizer.param_groups[0]["lr"]
            evaluation = evaluate(model, store, device, config["batch_size"], config["thresholds"])
            entry["evaluation"] = evaluation
            entry["epoch_memory"] = memory_stats(device)
            report["history"].append(entry)
            selected = evaluation["selected"]
            key = list(selection_key(selected))
            if report["best"] is None or key > report["best"]["key"]:
                report["best"] = {"key": key, "epoch": epoch, "threshold": selected["threshold"], "evaluation": evaluation}
                best_weights = cpu_weights(model)
            scheduler.step()
            report.update(status="running", requested_epochs=epochs, completed_epochs=epoch)
            save()
            print(json.dumps({"epoch": epoch, "loss": entry["loss"],
                              "iou": selected["overall"]["iou"], "high_density_recall": selected["high_density"]["recall"],
                              "threshold": selected["threshold"], "train_s": entry["train_s"],
                              "dev_s": evaluation["seconds"], **memory_stats(device)}), flush=True)
        if report["best"] is None or best_weights is None:
            raise ValueError("Run has no completed epoch")
        model.load_state_dict(best_weights)
        final = evaluate(model, store, device, config["batch_size"], config["thresholds"],
                         fixed_threshold=report["best"]["threshold"])
        if final["selected"] != report["best"]["evaluation"]["selected"]:
            raise AssertionError("Best checkpoint did not reproduce development metrics")
        # Do not replace the resumable last model with the selected best model.
        saved = checkpoint_load(last, device)
        model.load_state_dict(saved["model"])
        report.update(status="budget_complete", primary_metrics=final["selected"], final_evaluation=final,
                      conclusion="Manual training budget completed; review full Train-dev accuracy and cost before promotion.")
        save()
        return report
    finally:
        lock.unlink()


def evaluate_checkpoint(cache, directory, device_name, output):
    directory = run_path(directory)
    saved = checkpoint_load(directory/"best.pt")
    if saved["identity"]["sources"] != research_sources():
        raise ValueError("Evaluation code differs from the checkpoint")
    profile = saved["identity"]["profile"]
    store = SequenceStore(cache, profile["training"]["cache_mib"])
    if sha256(store.directory/"manifest.json") != saved["identity"]["teacher_cache_sha256"]:
        raise ValueError("Evaluation cache differs from the checkpoint")
    device = setup_device(device_name, profile["training"])
    model = StreamRefiner(RefinerConfig(**profile["model"]), saved["foreground_prior"]).to(device)
    model.load_state_dict(saved["model"])
    result = evaluate(model, store, device, profile["training"]["batch_size"],
                      profile["training"]["thresholds"], fixed_threshold=saved["threshold"])
    if result["selected"] != saved["evaluation"]["selected"]:
        raise AssertionError("Checkpoint metrics do not reproduce on this backend")
    from .foundation import write_report
    write_report(output, {"schema_version": 1, "status": "verified", "data_role": "train_dev",
                          "training_started": False, "checkpoint_sha256": sha256(directory/"best.pt"),
                          "environment": environment(device), "evaluation": result})
    return result


def check(profile, device_name, output, cache=None):
    """Full-size forward/backward resource check: no optimizer, no weight update."""
    config = profile["training"]
    device = setup_device(device_name, config)
    verify_baseline()
    torch.manual_seed(config["seed"])
    model = StreamRefiner(RefinerConfig(**profile["model"])).to(device)
    before = cpu_weights(model)
    started = time.perf_counter()
    if cache:
        store = SequenceStore(cache, config["cache_mib"])
        batch = next(stream_batches(store, "train_main", config["batch_size"]))
        logits, state, rates, _ = forward_batch(model, batch, None, device)
        # This checks graph connectivity, not task fitting or threshold selection.
        objective = logits.square().mean() if logits.numel() else sum(s.square().mean() for s in state[:3])
        objective = objective + rates.mean()
        provenance = "first chronological batch from verified Train-main cache"
    else:
        inputs = torch.full((10, config["batch_size"], 64, 68, 88), .5, device=device)
        inputs[:, :, :32] = 0.632
        logits, state, rates, _ = model(inputs)
        objective = logits.square().mean()+rates.mean()
        provenance = "dense synthetic full-size 10-step batch; all cell input channels occupied, not accuracy evidence"
    objective.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("Nonfinite or disconnected gradient")
    if any(not torch.equal(before[k], v) for k, v in cpu_weights(model).items()):
        raise AssertionError("Check unexpectedly modified model weights")
    sync(device)
    result = {"schema_version": 1, "status": "forward_backward_checked_no_weight_update",
              "training_started": False, "optimizer_created": False, "weight_updates": 0,
              "environment": environment(device), "profile": profile, "architecture": model.architecture(),
              "input": provenance, "elapsed_seconds": time.perf_counter()-started,
              **memory_stats(device), "note": "Full-epoch time and high-density peak must be measured by the manual pilot."}
    from .foundation import write_report
    write_report(output, result)
    return result
