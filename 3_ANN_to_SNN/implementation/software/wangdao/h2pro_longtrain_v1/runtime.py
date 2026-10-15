"""Full learning curves with exact safe-point recovery and automatic comparison."""
import copy
import gc
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from ..h2pro_snn.foundation import sha256
from ..h2pro_snn.stream_data import atomic_json, run_path
from ..h2pro_snn.training import (setup_device, environment, memory_stats, sync,
                                 cpu_weights, selection_key, restore_rng)
from ..h2pro_snn.diagnostics import aggregate_counts, error_counts, summarize_counts
from ..h2pro_ann_readout_v1.training import logit_counts, operating_point
from ..h2pro_ann_readout_v1.monitor import gradient_norms, parameter_norms
from .models import ARMS, ORIGINS, IMPLEMENTATION, make_model, profile_for, training_module, sources
from .storage import (all_rng, set_rng, save_atomic, load, cpu_tree, process_lock,
                      PauseControl, PauseRequested, progress)
from .data import ResidentData, schedule


def forward(model, batch, state):
    x = batch["inputs"]
    if state is None:
        state = model.initial_state(x.shape[1])
    state = model.reset_slots(state, torch.as_tensor(batch["reset"], device=x.device))
    outputs, state, rates, counts = model(x, state)
    logits = outputs[tuple(batch["device"]["maps"].unbind())]
    checks = [torch.isfinite(t).all() for t in (logits, *state, rates, counts, model.voltage_tail())]
    if not bool(torch.stack(checks).all()):
        raise FloatingPointError("Nonfinite output/state/activity/tail")
    return logits, state, rates, counts


def monitor_for(arm, config, restored=None):
    monitor = training_module(arm).HealthMonitor(config["stability_limits"], config["warning_limits"])
    if restored:
        monitor.__dict__.update({k: v.numpy().copy() if torch.is_tensor(v) else copy.deepcopy(v)
                                 for k, v in restored.items()})
    return monitor


def cost(model, arm, counts, steps, executed):
    result = training_module(arm).cost_report(model, counts, steps, executed)
    if arm == "snn_linear":
        result.update(scope="Hybrid SNN Refiner: spiking stem/hidden, continuous linear readout",
                      continuous_activation_ac_estimate=None,
                      ideal_unit_spike_ac_upper_bound=int(counts[2])*9*model.config.hidden_channels + int(counts[3])*128,
                      ac_note="Hypothetical unit-spike additions, padding included; dense kernels still execute MACs",
                      spike_convolution_zero_skip_implemented=False)
    return result


@torch.no_grad()
def evaluate(model, data, arm, config, pause=None, fixed_threshold=None):
    model.eval()
    records = {n: data.records[n] for n in data.store.names("train_dev")}
    scores = {n: np.zeros(len(r.arrays["t"]), np.float32) for n, r in records.items()}
    seen = {n: np.zeros(len(v), bool) for n, v in scores.items()}
    state, steps, executed = None, 0, 0
    counts = np.zeros(5, np.float64)
    monitor = monitor_for(arm, config)
    start = time.perf_counter()
    for _, batch, _ in data.batches("train_dev", config["batch_size"]):
        if pause and pause.pending():
            raise PauseRequested()
        logits, state, _, activity = forward(model, batch, state)
        monitor.observe(model, logits, batch["candidate"], batch["active"])
        values = logits.cpu().numpy()
        offset = 0
        for ref in batch["refs"]:
            name, ids = ref["sequence"], ref["ids"]
            if seen[name][ids].any():
                raise AssertionError("Duplicate Dev event")
            scores[name][ids] = values[offset:offset+len(ids)]
            seen[name][ids] = True
            offset += len(ids)
        if offset != len(values):
            raise AssertionError("Unmapped Dev event")
        counts += activity[:, batch["active"]].sum((0, 1)).cpu().numpy()
        steps += int(batch["active"].sum())*10
        executed += len(batch["active"])*10
    if any(not v.all() for v in seen.values()):
        raise AssertionError("Missing Dev event")

    def metrics(threshold, teacher=False):
        rows = []
        for n, r in records.items():
            a = r.arrays
            ct = (error_counts(a["label"], a["teacher"], a["candidate"], threshold) if teacher
                  else logit_counts(a["label"], scores[n], a["candidate"], threshold))
            rows.append({"sequence": n, "high_density": r.metadata["high_density"], **summarize_counts(ct)})
        return {"overall": aggregate_counts(rows), "high_density": aggregate_counts([r for r in rows if r["high_density"]]),
                "ordinary": aggregate_counts([r for r in rows if not r["high_density"]]), "sequences": rows}
    scans = [{**operating_point(t), **metrics(t)} for t in config["logit_thresholds"]]
    selected = (max(scans, key=selection_key) if fixed_threshold is None
                else {**operating_point(fixed_threshold), **metrics(fixed_threshold)})
    return {"selected": selected, "scan": scans, "frozen_teacher": metrics(.35, True),
            "seconds": time.perf_counter()-start, "stability": monitor.summary(),
            "cost": cost(model, arm, counts, steps, executed), "data_role": "train_dev",
            "selected_at_scan_boundary": selected["threshold"] in (config["logit_thresholds"][0], config["logit_thresholds"][-1]),
            "complete_development_roles": True, "event_coverage": sum(len(v) for v in scores.values())}


def accumulator(model):
    return {"events": 0, "candidate_events": 0, "windows": 0, "updates": 0, "batches": 0,
            "loss_sum": 0., "components": {}, "rates_sum": [0., 0.], "train_s": 0., "input_s": 0.,
            "checkpoint_s": 0., "parameter_norms_before": parameter_norms(model)}


def update_step(model, batch, state, optimizer, arm, config, monitor, totals):
    optimizer.zero_grad(set_to_none=True)
    logits, state, rates, _ = forward(model, batch, state)
    monitor.observe(model, logits, batch["candidate"], batch["active"])
    active = batch["active"]
    mask = batch["device"]["candidate"]
    n = int(batch["candidate"].sum())
    if n:
        loss, parts = training_module(arm).task_loss(logits[mask], batch["device"]["labels"][mask],
                    batch["device"]["teacher"][mask], rates[:, active], config, model.voltage_tail()[:, active])
        monitor.observe_loss(loss)
        loss.backward()
        gradients = gradient_norms(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        monitor.observe_gradients(norm, gradients, config["gradient_clip"])
        optimizer.step()
        totals["updates"] += 1
        totals["candidate_events"] += n
        totals["loss_sum"] += float(loss.detach())*n
        for k, v in parts.items():
            totals["components"][k] = totals["components"].get(k, 0.) + float(v.detach())*n
    totals["events"] += len(batch["labels"])
    totals["windows"] += int(active.sum())
    totals["batches"] += 1
    means = rates[:, active].detach().mean((0, 1)).cpu().tolist()
    totals["rates_sum"] = [a+b for a, b in zip(totals["rates_sum"], means)]
    return model.detach_state(state)


def epoch_result(totals, monitor, model, epoch, data):
    rows = data.store.manifest["roles"]["train_main"]
    if (totals["events"] != sum(r["published_events"] for r in rows)
            or totals["windows"] != sum(r["closed_windows"] for r in rows) or not totals["updates"]):
        raise AssertionError("Training coverage mismatch")
    n = totals["candidate_events"]
    return {**{k: totals[k] for k in ("events", "candidate_events", "windows", "updates", "batches", "train_s", "input_s", "checkpoint_s")},
            "epoch": epoch, "loss": totals["loss_sum"]/n,
            "loss_components": {k: v/n for k, v in totals["components"].items()},
            "mean_support_activation_in_threshold_units": [x/totals["batches"] for x in totals["rates_sum"]],
            "stability": monitor.summary(), "parameter_norms_before": totals["parameter_norms_before"],
            "parameter_norms_after": parameter_norms(model)}


def legacy_origin(arm, data, device, profile):
    if arm not in ORIGINS:
        return None, None
    path = IMPLEMENTATION / "results/runs" / ORIGINS[arm] / "last.pt"
    saved = load(path)
    old = saved["identity"]
    current_sources = training_module(arm).research_sources()
    if (old["profile"] != profile or old["sources"] != current_sources
            or old["teacher_cache_sha256"] != sha256(data.store.directory/"manifest.json")
            or old["environment"] != environment(device) or old.get("pilot")
            or saved["report"]["completed_epochs"] != 1 or len(saved["report"]["history"]) != 1):
        raise ValueError("Original epoch-1 checkpoint identity differs; cannot silently fork it")
    return saved, {"path": str(path), "sha256": sha256(path), "completed_epochs": 1}


@torch.no_grad()
def final_health(model, data, arm, config, pause=None):
    model.eval()
    monitor = monitor_for(arm, config)
    state, events, windows, failures = None, 0, 0, []
    started = time.perf_counter()
    for i, batch, _ in data.batches("train_main", config["batch_size"], epoch=0):
        if pause and pause.pending():
            raise PauseRequested()
        logits, state, _, _ = forward(model, batch, state)
        try:
            monitor.observe(model, logits, batch["candidate"], batch["active"])
        except FloatingPointError as e:
            failures.append({"batch": i+1, "reason": str(e), "refs": [
                {k: v for k, v in r.items() if k != "ids"} for r in batch["refs"]]})
        events += len(batch["labels"])
        windows += int(batch["active"].sum())
    expected = data.store.manifest["roles"]["train_main"]
    if events != sum(r["published_events"] for r in expected) or windows != sum(r["closed_windows"] for r in expected):
        raise AssertionError("Fixed-weight health scan coverage mismatch")
    return {"status": "passed" if not failures else "hard_guard_outliers", "events": events, "windows": windows,
            "seconds": time.perf_counter()-started, "hard_guard_batches": len(failures),
            "outliers": failures, "stability": monitor.summary(), "weights_fixed": True}


def run_arm(data, directory, arm, epochs, pause, checkpoint_batches=100, retry_failed=False,
            stop_after_batches=None, import_original=True):
    directory = run_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    profile = profile_for(arm)
    config, device = profile["training"], data.device
    origin, provenance = legacy_origin(arm, data, device, profile) if import_original else (None, None)
    identity = {"runtime": "h2pro_longtrain_v1", "arm": arm, "profile": profile, "sources": sources(),
                "environment": environment(device), "encoded_manifest_sha256": sha256(data.directory/"manifest.json"),
                "teacher_manifest_sha256": sha256(data.store.directory/"manifest.json"), "origin": provenance,
                "input_pipeline": "resident sparse GPU inputs; original numerical profile preserved",
                "checkpoint_contract": "completed batch cursor + detached stream state + complete optimizer/RNG/monitor"}
    prior = sum(r["candidate_targets"] for r in data.store.manifest["roles"]["train_main"])/sum(
        r["candidate_events"] for r in data.store.manifest["roles"]["train_main"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    model = make_model(arm, profile, float(np.clip(prior, 1e-4, 1-1e-4))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=config["minimum_learning_rate"])
    report = {"arm": arm, "identity": identity, "status": "running", "history": [], "best": None,
              "completed_epochs": 0, "requested_epochs": epochs, "architecture": model.architecture(),
              "cache": data.stats, "foreground_prior": prior,
              "launcher": "validation_test" if stop_after_batches is not None else "user",
              "disposable_check": stop_after_batches is not None,
              "data_roles": ["train_main", "train_dev"], "threshold_domain": "logit"}
    best_model, state, totals, monitor = None, None, None, None
    cursor = {"epoch": 1, "batch": 0, "phase": "train"}
    last = directory / "last.pt"
    if not last.exists() and any(p.name != ".run.lock" for p in directory.iterdir()):
        raise ValueError("Unrecognized nonempty training directory")
    if last.exists():
        saved = load(last)
        if saved["identity"] != identity:
            raise ValueError("Resume code/config/cache/environment/origin changed; preserve this run and use a new directory")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        report, best_model, cursor = saved["report"], saved["best_model"], saved["cursor"]
        state = tuple(s.to(device) for s in saved["stream_state"]) if saved["stream_state"] is not None else None
        totals = saved["totals"]
        monitor = monitor_for(arm, config, saved["monitor"]) if saved["monitor"] else None
        set_rng(saved["rng"])
        if report["completed_epochs"] >= epochs and report["status"] == "complete":
            return report
        if (directory/"failure.json").exists() and not retry_failed:
            return {**report, "status": "failed_requires_review", "failure": str(directory/"failure.json")}
        if retry_failed and (directory/"failure.json").exists():
            (directory/"failure.json").rename(directory/f"failure_previous_{time.time_ns()}.json")
        if cursor["phase"] == "complete":
            cursor["phase"] = "train"
    elif origin is not None:
        model.load_state_dict(origin["model"])
        optimizer.load_state_dict(origin["optimizer"])
        scheduler.load_state_dict(origin["scheduler"])
        restore_rng(origin["rng"])
        report.update(history=origin["report"]["history"], best=origin["report"]["best"], completed_epochs=1)
        best_model = origin["best_model"]
        cursor["epoch"] = 2
    if epochs < report["completed_epochs"]:
        raise ValueError("Requested total budget precedes completed epochs")

    def commit(status="running"):
        report["status"] = status
        report["requested_epochs"] = epochs
        started = time.perf_counter()
        save_atomic(last, {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "rng": all_rng(), "report": report,
                          "best_model": best_model, "cursor": cursor, "stream_state": state,
                          "totals": totals, "monitor": monitor.__dict__ if monitor else None})
        if totals:
            totals["checkpoint_s"] += time.perf_counter()-started
        atomic_json(directory/"training.json", report)
        progress(directory, status=status, arm=arm, cursor=cursor, completed_epochs=report["completed_epochs"],
                 requested_epochs=epochs, checkpoint=str(last), **memory_stats(device))

    def publish_best():
        if best_model is not None:
            save_atomic(directory/"best.pt", {"identity": identity, "model": best_model,
                        "epoch": report["best"]["epoch"], "threshold": report["best"]["threshold"],
                        "foreground_prior": prior, "evaluation": report["best"]["evaluation"]})

    commit()
    publish_best()
    try:
        while cursor["epoch"] <= epochs:
            epoch = cursor["epoch"]
            if cursor["phase"] == "train":
                model.train()
                if totals is None:
                    totals, monitor = accumulator(model), monitor_for(arm, config)
                assert monitor is not None
                for i, spec in enumerate(schedule(data.store, "train_main", config["batch_size"], config["seed"], epoch)):
                    if i < cursor["batch"]:
                        continue
                    if pause.pending():
                        raise PauseRequested()
                    start = time.perf_counter()
                    batch = data.batch(spec)
                    input_s = time.perf_counter()-start
                    state_before = state
                    state = update_step(model, batch, state, optimizer, arm, config, monitor, totals)
                    totals["input_s"] += input_s
                    totals["train_s"] += time.perf_counter()-start
                    cursor["batch"] = i+1
                    if (i+1) % checkpoint_batches == 0 or pause.pending():
                        commit()
                    if (i+1) % 100 == 0:
                        event = {"stage": "train", "arm": arm, "epoch": epoch, "batch": i+1,
                                 "loss": totals["loss_sum"]/max(totals["candidate_events"], 1),
                                 "elapsed_s": totals["train_s"], "input_s": totals["input_s"],
                                 "gradient_max": monitor.gradient_max, **memory_stats(device)}
                        print(json.dumps(event), flush=True)
                        with (directory/"progress.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps(event)+"\n")
                    if stop_after_batches is not None and cursor["batch"] >= stop_after_batches:
                        raise PauseRequested()
                cursor["phase"] = "evaluate"
                state = None
                commit()
            if pause.pending():
                raise PauseRequested()
            entry = epoch_result(totals, monitor, model, epoch, data)
            entry["learning_rate"] = optimizer.param_groups[0]["lr"]
            evaluation = evaluate(model, data, arm, config, pause)
            entry.update(evaluation=evaluation, epoch_memory=memory_stats(device))
            report["history"].append(entry)
            selected = evaluation["selected"]
            key = list(selection_key(selected))
            if report["best"] is None or key > report["best"]["key"]:
                best_model = cpu_weights(model)
                report["best"] = {"key": key, "epoch": epoch, "threshold": selected["threshold"], "evaluation": evaluation}
            scheduler.step()
            report["completed_epochs"] = epoch
            cursor = {"epoch": epoch+1, "batch": 0, "phase": "train"}
            state, totals, monitor = None, None, None
            commit()
            publish_best()
            if epoch in (5, 10, 20, 30):
                save_atomic(directory/f"epoch_{epoch:02d}.pt", load(last))
            print(json.dumps({"stage": "epoch_complete", "arm": arm, "epoch": epoch,
                              "loss": entry["loss"], "selected": selected["overall"],
                              "high_density_recall": selected["high_density"]["recall"]}), flush=True)
        # Last resumable training state remains distinct from selected-best audit weights.
        cursor["phase"] = "audit"
        commit()
        final_weights = cpu_weights(model)
        if best_model is None:
            raise ValueError("No selected checkpoint after training")
        model.load_state_dict(best_model)
        try:
            replay = evaluate(model, data, arm, config, pause, fixed_threshold=report["best"]["threshold"])
            if replay["selected"] != report["best"]["evaluation"]["selected"]:
                raise AssertionError("Best weights failed exact Dev metric replay")
            health = final_health(model, data, arm, config, pause)
            report.update(best_replay=replay, final_health=health, primary_metrics=replay["selected"])
        finally:
            model.load_state_dict(final_weights)
        cursor["phase"] = "complete"
        commit("complete")
        return report
    except PauseRequested:
        commit("paused")
        print(json.dumps({"stage": "paused", "arm": arm, "cursor": cursor, "checkpoint": str(last)}), flush=True)
        return report
    except Exception as e:
        failure = {"arm": arm, "cursor": cursor, "error": repr(e), "traceback": traceback.format_exc(),
                   "resumable_checkpoint": str(last), "diagnostic_checkpoint_is_resumable": False}
        atomic_json(directory/"failure.json", failure)
        save_atomic(directory/"failure_context.pt", {"failure": failure, "model": model.state_dict(),
                    "state_before": locals().get("state_before"), "batch": locals().get("batch"),
                    "layer_diagnostics": getattr(model, "last_diagnostics", None),
                    "monitor": monitor.__dict__ if monitor else None})
        progress(directory, status="failed", **failure)
        print(json.dumps(failure), flush=True)
        return {**report, "status": "failed", "failure": failure}


def comparison(directory, reports):
    def difference(a, b):
        return a-b if a is not None and b is not None else None
    result = {"status": "complete" if len(reports) == 3 and all(r["status"] == "complete" for r in reports) else "incomplete",
              "models": {}, "data": "Train-dev only; no Validation/Test access", "energy_measured": False}
    for r in reports:
        best = r.get("best")
        result["models"][r["arm"]] = {"status": r["status"], "completed_epochs": r["completed_epochs"],
            "best_epoch": best["epoch"] if best else None,
            "metrics": best["evaluation"]["selected"] if best else None,
            "cost": best["evaluation"]["cost"] if best else None, "final_health": r.get("final_health"),
            "learning_curve": [{"epoch": e["epoch"], "loss": e["loss"],
                                "iou": e["evaluation"]["selected"]["overall"]["iou"],
                                "recall": e["evaluation"]["selected"]["overall"]["recall"]} for e in r["history"]]}
        if best:
            selected = best["evaluation"]["selected"]
            teacher = best["evaluation"].get("frozen_teacher")
            result["models"][r["arm"]]["delta_vs_frozen_teacher"] = {
                k: difference(selected["overall"][k], teacher["overall"][k]) for k in ("iou", "precision", "recall")
            } if teacher else None
        if r["history"]:
            result["models"][r["arm"]]["last_epoch_metrics"] = r["history"][-1]["evaluation"]["selected"]
            result["models"][r["arm"]]["last_epoch_memory"] = r["history"][-1].get("epoch_memory")
    if all(a in result["models"] and result["models"][a]["metrics"] for a in ("ann_linear", "snn_linear")):
        a, b = [result["models"][k]["metrics"] for k in ("ann_linear", "snn_linear")]
        result["snn_minus_ann_linear_at_independently_selected_dev_best"] = {
            **{k: difference(b["overall"][k], a["overall"][k]) for k in ("iou", "precision", "recall")},
            "high_density_recall": difference(b["high_density"]["recall"], a["high_density"]["recall"])}
    atomic_json(Path(directory)/"comparison.json", result)
    return result


def run_suite(teacher, encoded, directory, epochs=30, checkpoint_batches=100, retry_failed=False):
    directory = run_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with process_lock(directory/".suite.lock"), PauseControl(directory) as pause:
        # An explicit train command acknowledges a previous deliberate pause.
        pause.path.unlink(missing_ok=True)
        progress(directory, status="loading_cache", requested_epochs=epochs)
        device = setup_device("cuda", profile_for("ann_linear")["training"])
        data = ResidentData(teacher, encoded, device)
        print(json.dumps({"stage": "resident_ready", **data.stats, **memory_stats(device)}), flush=True)
        reports = []
        for arm in ARMS:
            if pause.pending():
                break
            arm_dir = directory/arm
            torch.cuda.reset_peak_memory_stats(device)
            progress(directory, status="running", active_control=arm, requested_epochs=epochs)
            with process_lock(arm_dir/".run.lock"):
                report = run_arm(data, arm_dir, arm, epochs, pause, checkpoint_batches, retry_failed)
            reports.append(report)
            comparison(directory, reports)
            gc.collect()
            torch.cuda.empty_cache()
            if report["status"] == "paused":
                break
        result = comparison(directory, reports)
        progress(directory, status="paused" if pause.pending() or any(r["status"] == "paused" for r in reports) else result["status"],
                 completed_controls=[r["arm"] for r in reports if r["status"] == "complete"],
                 comparison=str(directory/"comparison.json"))
        return result
