"""Fixed v2.2: signed head-voltage loss, unchanged forward and hard guards."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..h2pro_snn.diagnostics import aggregate_counts, error_counts, summarize_counts
from .foundation import PACKAGE, read_json, research_sources, sha256, verify_baseline
from .refiner import RefinerConfig, StreamRefiner
from ..h2pro_snn.training import (setup_device, environment, memory_stats, sync, task_loss as base_task_loss,
    selection_key, checkpoint_save, checkpoint_load, rng_state, restore_rng, cpu_weights)
from .diagnostics import RunDiagnostics, write_json
from .monitor import HealthMonitor, gradient_norms, parameter_norms
from ..h2pro_snn.stream_data import ENCODER_VERSION, SequenceStore, atomic_json, prefetch, run_path, stream_batches


def load_profile(path=None, mode=None, batch_size=None):
    profile = read_json(path or PACKAGE / "local_4060.json")
    if profile.get("schema_version") != 2 or profile.get("diagnostic_revision") != "fixed_v2_2":
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
                "kd_weight", "soft_iou_weight", "focal_gamma", "spike_weight", "spike_ceiling", "vram_limit_gib",
                "head_voltage_weight"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError("Invalid numeric training option: " + key)
    if (not 0 < t["minimum_learning_rate"] <= t["learning_rate"] or t["gradient_clip"] <= 0
            or not 0 < t["focal_alpha"] < 1 or not 0 < t["vram_limit_gib"] <= 6.5):
        raise ValueError("Invalid optimizer/loss/VRAM limits")
    thresholds = t["logit_thresholds"]
    if (not thresholds or thresholds != sorted(set(thresholds))
            or any(not math.isfinite(v) for v in thresholds)):
        raise ValueError("Logit thresholds must be finite, sorted and unique")
    HealthMonitor(t["stability_limits"], t["warning_limits"])
    if profile["model"]["head_voltage_soft_limit"] >= profile["model"]["threshold"] * t["stability_limits"]["max_unit_burst"]:
        raise ValueError("Voltage soft limit must precede the unchanged burst hard guard")
    expected = read_json(PACKAGE / "local_4060.json")["execution"]
    if profile["execution"] != expected:
        raise ValueError("Execution contract is fixed; AMP/time/reset changes require a new protocol")
    return profile


def task_loss(logits, labels, teacher, rates, config, voltage_tail):
    """One extra loss term. Existing supervision, KD and spike loss are intact."""
    if voltage_tail.ndim != 3 or voltage_tail.shape[-1] != 2 or voltage_tail.numel() == 0:
        raise ValueError("Expected nonempty [steps,active_streams,2] signed voltage tails")
    loss, parts = base_task_loss(logits, labels, teacher, rates, config)
    positive, negative = voltage_tail.mean((0, 1)).unbind()
    voltage = positive + negative
    parts.update(head_voltage=voltage, head_voltage_positive=positive, head_voltage_negative=negative)
    return loss + config["head_voltage_weight"] * voltage, parts


def forward_batch(model, batch, state, device, *, diagnostics=None):
    """Same v2 computation; expose outputs to diagnostics before finite checks."""
    inputs = torch.from_numpy(batch["inputs"]).to(device)
    if state is None:
        state = model.initial_state(inputs.shape[1])
    state = model.reset_slots(state, torch.from_numpy(batch["reset"]).to(device))
    output, state, rates, counts = model(inputs, state)
    indices = tuple(torch.as_tensor(a, dtype=torch.long, device=device) for a in batch["maps"])
    event_logits = output[indices]
    if diagnostics is not None:
        diagnostics.observation(event_logits, state)
    if (not torch.isfinite(event_logits).all() or any(not torch.isfinite(s).all() for s in state)
            or not torch.isfinite(rates).all() or not torch.isfinite(counts).all()
            or not torch.isfinite(model.voltage_tail()).all()):
        raise FloatingPointError("Nonfinite SNN output/state/rates/counts; run stopped")
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
            "reference_charge_scale_scalar_multiplications": executed_steps*cells*(c.stem_channels+c.hidden_channels),
            "logical_stream_steps": stream_steps, "executed_stream_steps_including_batch_padding": executed_steps,
            "state_scalars_per_stream": model.architecture()["state_scalars_per_stream"],
            "moved_sources": int(moved), "spike_convolution_zero_skip_implemented": False,
            "ac_note": "Burst n counts as n unit-spike additions with theta-scaled weights; dense reference scales charge explicitly; 3x3 fanout includes padding (upper bound)",
            "count_basis": "Executed tensor shapes and observed spike counts; dense-convolution equivalent, not GPU instruction counters",
            "end_to_end_complete": False, "energy_measured": False,
            "unaccounted_components": ["frozen Proposal", "event encoding", "diagnostic reductions",
                                       "threshold/reset/bias/max/probability operations", "memory traffic"]}


def logit_counts(labels, logits, candidate, threshold):
    values = np.asarray(logits)
    if not np.isfinite(values).all() or not math.isfinite(threshold):
        raise ValueError("Nonfinite logits or threshold")
    # Decisions remain ordered even when float32 sigmoid rounds both scores to 1.
    prediction = ((values >= threshold) & candidate).astype(np.float32)
    return error_counts(labels, prediction, candidate, .5)


def operating_point(threshold):
    return {"threshold": float(threshold), "threshold_domain": "logit",
            "probability_threshold_display_only": math.exp(-float(np.logaddexp(0., -threshold)))}


@torch.no_grad()
def evaluate(model, store, device, batch_size, thresholds, *, fixed_threshold=None, stability_limits=None,
             warning_limits=None, diagnostics=None, epoch=None, phase="train_dev"):
    model.eval()
    records = {name: store.get("train_dev", name) for name in store.names("train_dev")}
    scores = {name: np.zeros(len(r.arrays["t"]), np.float32) for name, r in records.items()}
    seen = {name: np.zeros(len(r.arrays["t"]), bool) for name, r in records.items()}
    state, stream_steps, executed_steps = None, 0, 0
    counts = np.zeros(5, np.float64)
    monitor = HealthMonitor(stability_limits, warning_limits)
    started = time.perf_counter()
    # One fixed chronological pass. Threshold selection never reruns the network.
    for index, (batch, _) in enumerate(prefetch(iter(stream_batches(store, "train_dev", batch_size))), 1):
        if diagnostics is not None:
            diagnostics.begin(model, batch, state, monitor, device, phase=phase, epoch=epoch, index=index,
                              metadata={n: r.metadata for n, r in records.items()})
        logits, state, _, activity = forward_batch(model, batch, state, device, diagnostics=diagnostics)
        alerts = monitor.observe(model, logits, batch["candidate"], batch["active"])
        if diagnostics is not None:
            diagnostics.warnings(alerts)
        values = logits.cpu().numpy()
        offset = 0
        for ref in batch["refs"]:
            name, ids = ref["sequence"], ref["ids"]
            if seen[name][ids].any():
                raise AssertionError("Duplicate event prediction")
            scores[name][ids] = values[offset:offset+len(ids)]
            seen[name][ids] = True
            offset += len(ids)
        if offset != len(values):
            raise AssertionError("Unmapped event output")
        counts += activity[:, batch["active"]].sum((0, 1)).cpu().numpy()
        stream_steps += int(batch["active"].sum())*batch["inputs"].shape[0]
        executed_steps += batch["inputs"].shape[0]*batch["inputs"].shape[1]
        if diagnostics is not None and index % 50 == 0:
            diagnostics.progress(status="running", elapsed_s=time.perf_counter()-started)
    if any(not flags.all() for flags in seen.values()):
        raise AssertionError("Missing event prediction")

    def metrics(threshold, teacher=False):
        rows = []
        for name, r in records.items():
            a = r.arrays
            row = (error_counts(a["label"], a["teacher"], a["candidate"], threshold) if teacher
                   else logit_counts(a["label"], scores[name], a["candidate"], threshold))
            rows.append({"sequence": name, "high_density": r.metadata["high_density"], **summarize_counts(row)})
        return {"overall": aggregate_counts(rows),
                "high_density": aggregate_counts([r for r in rows if r["high_density"]]),
                "ordinary": aggregate_counts([r for r in rows if not r["high_density"]]), "sequences": rows}

    scans = [{**operating_point(float(t)), **metrics(float(t))} for t in thresholds]
    selected = max(scans, key=selection_key) if fixed_threshold is None else {**operating_point(fixed_threshold), **metrics(fixed_threshold)}
    teacher = metrics(0.35, True)
    def delta(group, metric):
        a, b = selected[group][metric], teacher[group][metric]
        return a-b if a is not None and b is not None else None
    sync(device)
    if diagnostics is not None:
        diagnostics.progress(status="phase_complete", elapsed_s=time.perf_counter()-started)
        diagnostics.release()
    return {"selected": selected, "scan": [{**operating_point(s["threshold"]), "overall": s["overall"],
                                            "high_density": s["high_density"]} for s in scans],
            "frozen_teacher": teacher,
            "delta_vs_teacher": {"iou": delta("overall", "iou"), "high_density_recall": delta("high_density", "recall")},
            "cost": cost_report(model, counts, stream_steps, executed_steps), "seconds": time.perf_counter()-started,
            "stability": monitor.summary(),
            "selected_at_scan_boundary": selected["threshold"] in (thresholds[0], thresholds[-1]),
            "data_role": "train_dev", "complete_development_roles": store.manifest["complete_development_roles"],
            "event_coverage": "same frozen ANN closed prefix; rejected candidates included as background",
            "latency": "100 ms Proposal closure; 10 ms internal bins; no end-to-end timing claim"}


def train_epoch(model, store, optimizer, device, config, epoch, *, diagnostics=None):
    model.train()
    state = None
    start = time.perf_counter()
    waiting = total_loss = 0.0
    events = candidates = updates = windows = 0
    components = {k: 0.0 for k in ("focal", "soft_iou", "kd", "spike", "head_voltage",
                                  "head_voltage_positive", "head_voltage_negative")}
    rates_sum = np.zeros(2, np.float64)
    monitor = HealthMonitor(config["stability_limits"], config["warning_limits"])
    initial_norms = parameter_norms(model)
    batches = 0
    iterator = stream_batches(store, "train_main", config["batch_size"], seed=config["seed"], epoch=epoch)
    for batch, wait in prefetch(iter(iterator)):
        waiting += wait
        optimizer.zero_grad(set_to_none=True)
        if diagnostics is not None:
            diagnostics.begin(model, batch, state, monitor, device, phase="train_main", epoch=epoch, index=batches+1,
                              metadata={r["sequence"]: r for r in store.manifest["roles"]["train_main"]})
        logits, state, rates, _ = forward_batch(model, batch, state, device, diagnostics=diagnostics)
        alerts = monitor.observe(model, logits, batch["candidate"], batch["active"])
        if diagnostics is not None:
            diagnostics.warnings(alerts)
        events += len(batch["labels"])
        windows += int(batch["active"].sum())
        mask = torch.from_numpy(batch["candidate"]).to(device)
        n = int(batch["candidate"].sum())
        if n:
            target = torch.from_numpy(batch["labels"][batch["candidate"]].copy()).to(device)
            teacher = torch.from_numpy(batch["teacher"][batch["candidate"]].copy()).to(device)
            if diagnostics is not None:
                diagnostics.stage("loss")
            loss, parts = task_loss(logits[mask], target, teacher, rates[:, batch["active"]], config,
                                    model.voltage_tail()[:, batch["active"]])
            if diagnostics is not None:
                diagnostics.stage("loss_guard", loss=float(loss.detach()),
                                  loss_components={k: float(v.detach()) for k, v in parts.items()})
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            monitor.observe_loss(loss)
            if diagnostics is not None:
                diagnostics.stage("backward")
            loss.backward()
            if diagnostics is not None:
                diagnostics.stage("gradient_check")
            gradients = gradient_norms(model)
            if diagnostics is not None:
                diagnostics.stage("gradient_clip")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            if diagnostics is not None:
                diagnostics.stage("gradient_guard", gradient_norm_pre_clip=float(norm))
            monitor.observe_gradients(norm, gradients, config["gradient_clip"])
            if diagnostics is not None:
                diagnostics.stage("optimizer_step")
            optimizer.step()
            if diagnostics is not None:
                diagnostics.stage("after_optimizer_step")
            updates += 1
            candidates += n
            total_loss += float(loss.detach())*n
            for key in components:
                components[key] += float(parts[key].detach())*n
        state = model.detach_state(state)
        rates_sum += rates[:, batch["active"]].detach().mean((0, 1)).cpu().numpy()
        batches += 1
        if batches % 50 == 0:
            if diagnostics is not None:
                diagnostics.progress(status="running", elapsed_s=time.perf_counter()-start,
                                     cumulative_loss=total_loss/max(candidates, 1),
                                     cumulative_loss_components={k: v/max(candidates, 1) for k, v in components.items()},
                                     completed_batches=batches, completed_updates=updates)
            print(json.dumps({"stage": "train", "epoch": epoch, "batch": batches, "events": events,
                              "logit_absmax_so_far": monitor.logit_absmax,
                              "gradient_norm_max_so_far": monitor.gradient_max,
                              "warning_batches": monitor.logit_warning_batches,
                              "input_wait_s": waiting, "elapsed_s": time.perf_counter()-start,
                              **memory_stats(device)}), flush=True)
    if not updates:
        raise ValueError("No candidate training events in this cache")
    expected = sum(r["published_events"] for r in store.manifest["roles"]["train_main"])
    expected_windows = sum(r["closed_windows"] for r in store.manifest["roles"]["train_main"])
    if events != expected or windows != expected_windows:
        raise AssertionError("Training did not cover each closed-prefix event/window exactly once")
    sync(device)
    if diagnostics is not None:
        diagnostics.progress(status="phase_complete", elapsed_s=time.perf_counter()-start,
                             cumulative_loss=total_loss/candidates,
                             cumulative_loss_components={k: v/candidates for k, v in components.items()},
                             completed_batches=batches, completed_updates=updates)
        diagnostics.release()
    return {"epoch": epoch, "loss": total_loss/candidates,
            "loss_components": {k: v/candidates for k, v in components.items()},
            "mean_support_spikes_per_neuron_step": (rates_sum/batches).tolist(),
            "events": events, "candidate_events": candidates, "windows": windows, "updates": updates,
            "batches": batches, "input_wait_s": waiting, "train_s": time.perf_counter()-start,
            "cache": dict(store.stats), "stability": monitor.summary(),
            "parameter_norms_before": initial_norms, "parameter_norms_after": parameter_norms(model),
            **memory_stats(device)}


def train(cache, directory, profile, device_name, epochs, *, resume=False, pilot=False, launched_by="user"):
    if launched_by not in ("user", "user_authorized_agent"):
        raise ValueError("Unknown training launcher attribution")
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
                "numerical_protocol_sha256": sha256(PACKAGE/"protocol.json"),
                "sources": research_sources(), "environment": environment(device), "pilot": pilot}
    lock = directory / ".training.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    diagnostics = None
    try:
        torch.manual_seed(config["seed"])
        main_rows = store.manifest["roles"]["train_main"]
        prior = sum(r["candidate_targets"] for r in main_rows)/max(sum(r["candidate_events"] for r in main_rows), 1)
        prior = float(np.clip(prior, 1e-4, 1-1e-4))
        model = StreamRefiner(RefinerConfig(**profile["model"]), prior).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["schedule_epochs"], eta_min=config["minimum_learning_rate"])
        report = {"schema_version": 2, "status": "running", "experiment_id": directory.name,
                  "stage": "fixed_v2_2_signed_head_regularization", "threshold_domain": "logit", "data_role": ["train_main", "train_dev"],
                  "training_started": True, "launcher": launched_by, "pilot": pilot,
                  "identity": identity, "architecture": model.architecture(), "foreground_prior": prior,
                  "history": [], "best": None, "primary_metrics": {}, "completed_epochs": 0, "requested_epochs": epochs,
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
        report.update(status="running", requested_epochs=epochs, launcher=launched_by, conclusion="Training in progress.")
        diagnostics = RunDiagnostics(directory, identity)
        def save():
            checkpoint_save(last, {"identity": identity, "model": cpu_weights(model),
                                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                  "rng": rng_state(), "report": report, "best_model": best_weights})
            if best_weights is not None:
                checkpoint_save(directory/"best.pt", {"identity": identity, "model": best_weights,
                    "threshold": report["best"]["threshold"], "epoch": report["best"]["epoch"],
                    "threshold_domain": "logit",
                    "foreground_prior": prior, "evaluation": report["best"]["evaluation"]})
            atomic_json(directory/"training.json", report)
        if not resume:
            save()  # Initialize epoch 0 once; failures never replace a committed checkpoint.
        else:
            atomic_json(directory/"training.json", report)
        for epoch in range(len(report["history"])+1, epochs+1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            entry = train_epoch(model, store, optimizer, device, config, epoch, diagnostics=diagnostics)
            entry["learning_rate"] = optimizer.param_groups[0]["lr"]
            evaluation = evaluate(model, store, device, config["batch_size"], config["logit_thresholds"],
                                  stability_limits=config["stability_limits"], warning_limits=config["warning_limits"],
                                  diagnostics=diagnostics, epoch=epoch)
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
                              "threshold_logit": selected["threshold"], "train_s": entry["train_s"],
                              "support_spikes": entry["mean_support_spikes_per_neuron_step"],
                              "selected_at_scan_boundary": evaluation["selected_at_scan_boundary"],
                              "stability_warnings": entry["stability"]["warnings"] + evaluation["stability"]["warnings"],
                              "dev_s": evaluation["seconds"], **memory_stats(device)}), flush=True)
        if report["best"] is None or best_weights is None:
            raise ValueError("Run has no completed epoch")
        model.load_state_dict(best_weights)
        final = evaluate(model, store, device, config["batch_size"], config["logit_thresholds"],
                         fixed_threshold=report["best"]["threshold"], stability_limits=config["stability_limits"],
                         warning_limits=config["warning_limits"], diagnostics=diagnostics,
                         epoch=report["best"]["epoch"], phase="best_replay")
        if final["selected"] != report["best"]["evaluation"]["selected"]:
            raise AssertionError("Best checkpoint did not reproduce development metrics")
        # Do not replace the resumable last model with the selected best model.
        saved = checkpoint_load(last, device)
        model.load_state_dict(saved["model"])
        report.update(status="budget_complete", primary_metrics=final["selected"], final_evaluation=final,
                      conclusion="Explicitly authorized training budget completed; review full Train-dev accuracy and cost before promotion.")
        save()
        diagnostics.progress(status="budget_complete", completed_epochs=report["completed_epochs"])
        return report
    except (Exception, KeyboardInterrupt) as error:
        if diagnostics is None:
            raise  # Identity/setup rejection must not alter an existing run's report.
        failure = diagnostics.failure(error)
        committed_report = checkpoint_load(last)["report"] if last.exists() else {}
        committed = committed_report.get("completed_epochs", 0)
        failure.update(status="stopped_diagnostic_snapshot_only", last_committed_epoch=committed,
                       training_launched_by=launched_by, resume_from="last.pt_only",
                       note="Diagnostic snapshots are not resumable. last.pt is the last complete epoch or initialization.")
        write_json(directory/"stability_failure.json", failure)
        write_json(directory/"training.json", {**committed_report, "status": "stopped", "completed_epochs": committed,
                    "requested_epochs": epochs, "failure": failure,
                    "conclusion": "Stopped before the requested budget completed; inspect diagnostics before resuming."})
        raise
    finally:
        lock.unlink()


def evaluate_checkpoint(cache, directory, device_name, output):
    directory = run_path(directory)
    saved = checkpoint_load(directory/"best.pt")
    if saved.get("threshold_domain") != "logit" or saved["identity"]["profile"].get("diagnostic_revision") != "fixed_v2_2":
        raise ValueError("Expected a v2.2 checkpoint; earlier versions keep their original entrypoints")
    if saved["identity"]["sources"] != research_sources():
        raise ValueError("Evaluation code differs from the checkpoint")
    profile = saved["identity"]["profile"]
    store = SequenceStore(cache, profile["training"]["cache_mib"])
    if sha256(store.directory/"manifest.json") != saved["identity"]["teacher_cache_sha256"]:
        raise ValueError("Evaluation cache differs from the checkpoint")
    device = setup_device(device_name, profile["training"])
    model = StreamRefiner(RefinerConfig(**profile["model"]), saved["foreground_prior"]).to(device)
    model.load_state_dict(saved["model"])
    diagnostics = RunDiagnostics(directory, saved["identity"])
    try:
        result = evaluate(model, store, device, profile["training"]["batch_size"],
                          profile["training"]["logit_thresholds"], fixed_threshold=saved["threshold"],
                          stability_limits=profile["training"]["stability_limits"],
                          warning_limits=profile["training"]["warning_limits"], diagnostics=diagnostics,
                          epoch=saved["epoch"], phase="independent_evaluation")
    except (Exception, KeyboardInterrupt) as error:
        diagnostics.failure(error)
        raise
    if result["selected"] != saved["evaluation"]["selected"]:
        raise AssertionError("Checkpoint metrics do not reproduce on this backend")
    from ..h2pro_snn.foundation import write_report
    write_report(output, {"schema_version": 2, "status": "verified", "data_role": "train_dev",
                          "training_started": False, "checkpoint_sha256": sha256(directory/"best.pt"),
                          "environment": environment(device), "evaluation": result})
    return result


def check(profile, device_name, output, cache=None):
    """Dense stress plus ordinary/high-density Train-main prefixes; no fitting."""
    config = profile["training"]
    device = setup_device(device_name, config)
    baseline = verify_baseline()
    started = time.perf_counter()
    store = SequenceStore(cache, config["cache_mib"]) if cache else None
    prior = .1
    if store:
        rows = store.manifest["roles"]["train_main"]
        prior = float(np.clip(sum(r["candidate_targets"] for r in rows) /
                              max(sum(r["candidate_events"] for r in rows), 1), 1e-4, 1-1e-4))
    scenarios: List[Tuple[str, Optional[list]]] = [("dense_synthetic", None)]
    if store:
        for density in (False, True):
            rows = [r for r in store.manifest["roles"]["train_main"] if r["high_density"] == density]
            rows.sort(key=lambda r: (-r["published_events"], r["sequence"]))
            names = [r["sequence"] for r in rows[:config["batch_size"]]]
            if names:
                scenarios.append(("train_main_high_density" if density else "train_main_ordinary", names))
    cases = []

    class PrefixStore:
        def __init__(self, source, names):
            self.source, self.selected = source, names
        def names(self, role):
            if role != "train_main":
                raise ValueError("Check uses Train-main only")
            return list(self.selected)
        def get(self, role, name):
            return self.source.get(role, name)

    for name, names in scenarios:
        torch.manual_seed(config["seed"])
        model = StreamRefiner(RefinerConfig(**profile["model"]), prior).to(device)
        before = cpu_weights(model)
        monitor = HealthMonitor(config["stability_limits"], config["warning_limits"])
        losses = []
        if names is None:
            inputs = torch.full((10, config["batch_size"], 64, 68, 88), .5, device=device)
            inputs[:, :, :32] = .632
            logits, state, rates, _ = model(inputs)
            monitor.observe(model, logits.flatten(), torch.ones(logits.numel(), dtype=torch.bool, device=device),
                            np.ones(config["batch_size"], bool))
            objective = logits.square().mean()+rates.mean()+config["head_voltage_weight"]*model.voltage_tail().sum(-1).mean()
            objective.backward()
            gradients = gradient_norms(model)
            norm = math.sqrt(sum(v*v for v in gradients.values()))
            monitor.observe_gradients(norm, gradients, config["gradient_clip"])
            losses.append(float(objective.detach()))
            del inputs, logits, state, rates, objective
        else:
            state = None
            iterator = stream_batches(PrefixStore(store, names), "train_main", config["batch_size"])
            for index, batch in enumerate(iterator):
                if index == 4:
                    break
                model.zero_grad(set_to_none=True)
                logits, state, rates, _ = forward_batch(model, batch, state, device)
                monitor.observe(model, logits, batch["candidate"], batch["active"])
                mask = torch.as_tensor(batch["candidate"], dtype=torch.bool, device=device)
                if mask.any():
                    target = torch.as_tensor(batch["labels"], device=device)[mask]
                    teacher = torch.as_tensor(batch["teacher"], device=device)[mask]
                    objective, _ = task_loss(logits[mask], target, teacher, rates[:, batch["active"]], config,
                                             model.voltage_tail()[:, batch["active"]])
                    monitor.observe_loss(objective)
                    objective.backward()
                    gradients = gradient_norms(model)
                    norm = math.sqrt(sum(v*v for v in gradients.values()))
                    monitor.observe_gradients(norm, gradients, config["gradient_clip"])
                    losses.append(float(objective.detach()))
                    del objective
                state = model.detach_state(state)
                del logits, rates
            iterator.close()
            if not losses:
                raise AssertionError("Real prefix had no candidate loss to check")
            del state
        if any(not torch.equal(before[k], v) for k, v in cpu_weights(model).items()):
            raise AssertionError("Check unexpectedly modified weights")
        cases.append({"name": name, "sequences": names, "objective_values": losses,
                      "parameters_unchanged": True, "gradient_clipping_applied": False,
                      "stability": monitor.summary()})
        print(json.dumps({"check": name, "status": "passed_no_parameter_update",
                          "logit_absmax": monitor.logit_absmax, "gradient_norm_max": monitor.gradient_max}), flush=True)
    sync(device)
    result = {"schema_version": 2, "status": "forward_backward_checked_no_weight_update",
              "training_started": False, "optimizer_created": False, "weight_updates": 0,
              "environment": environment(device), "profile": profile, "architecture": model.architecture(),
              "baseline": baseline, "sources": research_sources(), "cases": cases,
              "elapsed_seconds": time.perf_counter()-started, **memory_stats(device),
              "note": "Initialization and finite-gradient checks only; no convergence, trained accuracy or full-epoch timing claim."}
    from ..h2pro_snn.foundation import write_report
    write_report(output, result)
    return result
