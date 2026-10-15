"""Independent advisory monitoring runtime; unchanged training mathematics."""
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from ..h2pro_snn.stream_data import atomic_json, run_path
from ..h2pro_snn.training import setup_device, memory_stats, cpu_weights, selection_key
from ..h2pro_snn.diagnostics import aggregate_counts, error_counts, summarize_counts
from ..h2pro_ann_readout_v1.training import logit_counts, operating_point
from ..h2pro_longtrain_v1.data import ResidentData, schedule
from ..h2pro_longtrain_v1.storage import all_rng, set_rng, save_atomic, load, process_lock, PauseControl, PauseRequested, progress
from ..h2pro_longtrain_v1.runtime import forward, cost, accumulator, update_step, epoch_result, comparison
from .models import ARMS, POLICY, make_model, profile_for
from .monitor import monitor_for, emit_advisories
from .migration import make_identity

@torch.no_grad()
def evaluate(model, data, arm, config, pause=None, fixed_threshold=None, log_directory=None):
    model.eval()
    records = {n: data.records[n] for n in data.store.names("train_dev")}
    scores = {n: np.zeros(len(r.arrays["t"]), np.float32) for n, r in records.items()}
    seen = {n: np.zeros(len(v), bool) for n, v in scores.items()}
    state, steps, executed = None, 0, 0
    counts = np.zeros(5, np.float64)
    monitor = monitor_for(arm, config)
    start = time.perf_counter()
    for index, batch, _ in data.batches("train_dev", config["batch_size"]):
        if pause and pause.pending():
            raise PauseRequested()
        logits, state, _, activity = forward(model, batch, state)
        monitor.observe(model, logits, batch["candidate"], batch["active"])
        emit_advisories(monitor, log_directory, phase="train_dev", batch=index+1)
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


@torch.no_grad()
def final_health(model, data, arm, config, pause=None, log_directory=None):
    model.eval()
    monitor = monitor_for(arm, config)
    state, events, windows, failures = None, 0, 0, []
    started = time.perf_counter()
    for i, batch, _ in data.batches("train_main", config["batch_size"], epoch=0):
        if pause and pause.pending():
            raise PauseRequested()
        logits, state, _, _ = forward(model, batch, state)
        monitor.observe(model, logits, batch["candidate"], batch["active"])
        emit_advisories(monitor, log_directory, phase="fixed_weight_train_health", batch=i+1)
        events += len(batch["labels"])
        windows += int(batch["active"].sum())
    expected = data.store.manifest["roles"]["train_main"]
    if events != sum(r["published_events"] for r in expected) or windows != sum(r["closed_windows"] for r in expected):
        raise AssertionError("Fixed-weight health scan coverage mismatch")
    return {"status": "passed" if not failures else "hard_guard_outliers", "events": events, "windows": windows,
            "seconds": time.perf_counter()-started, "hard_guard_batches": len(failures),
            "outliers": failures, "stability": monitor.summary(), "weights_fixed": True}


def run_arm(data, directory, arm, epochs, pause, checkpoint_batches=100, retry_failed=False,
            stop_after_batches=None):
    directory = run_path(directory)
    last = directory/"last.pt"
    if not last.exists():
        raise ValueError("Prepare the validated v1 checkpoint migration before training")
    profile = profile_for(arm)
    config, device = profile["training"], data.device
    saved = load(last)
    identity = make_identity(data, arm, profile, saved["identity"].get("migration"))
    if saved["identity"] != identity:
        raise ValueError("Resume code/config/cache/environment changed; checkpoint preserved")
    report, cursor = saved["report"], saved["cursor"]
    if not 1 <= epochs <= config["schedule_epochs"] or epochs < report["completed_epochs"]:
        raise ValueError("Invalid total epoch budget")
    if report["completed_epochs"] >= epochs and report["status"] == "complete":
        return report
    if (directory/"failure.json").exists() and not retry_failed:
        return {**report, "status": "failed_requires_review", "failure": str(directory/"failure.json")}
    if retry_failed and (directory/"failure.json").exists():
        (directory/"failure.json").rename(directory/f"failure_previous_{time.time_ns()}.json")
    model = make_model(arm, profile, report["foreground_prior"]).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["schedule_epochs"],
                                                         eta_min=config["minimum_learning_rate"])
    scheduler.load_state_dict(saved["scheduler"])
    best_model = saved["best_model"]
    state = tuple(s.to(device) for s in saved["stream_state"]) if saved["stream_state"] is not None else None
    totals = saved["totals"]
    monitor = monitor_for(arm, config, saved["monitor"]) if saved["monitor"] else None
    set_rng(saved["rng"])
    if cursor["phase"] == "complete":
        cursor["phase"] = "train"
    report.update(launcher="validation_test" if stop_after_batches is not None else "user",
                  disposable_check=stop_after_batches is not None, monitoring_policy=POLICY)

    def commit(status="running"):
        report.update(status=status, requested_epochs=epochs)
        started = time.perf_counter()
        save_atomic(last, {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "rng": all_rng(), "report": report,
                          "best_model": best_model, "cursor": cursor, "stream_state": state,
                          "totals": totals, "monitor": monitor.__dict__ if monitor else None})
        if totals:
            totals["checkpoint_s"] += time.perf_counter()-started
        atomic_json(directory/"training.json", report)
        progress(directory, status=status, arm=arm, cursor=cursor, completed_epochs=report["completed_epochs"],
                 requested_epochs=epochs, checkpoint=str(last), monitoring_policy=POLICY, **memory_stats(device))

    def publish_best():
        if best_model is not None:
            save_atomic(directory/"best.pt", {"identity": identity, "model": best_model,
                        "epoch": report["best"]["epoch"], "threshold": report["best"]["threshold"],
                        "foreground_prior": report["foreground_prior"], "evaluation": report["best"]["evaluation"]})

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
                    emit_advisories(monitor, directory, phase="train", epoch=epoch, batch=i+1)
                    if (i+1) % checkpoint_batches == 0 or pause.pending():
                        commit()
                    if (i+1) % 100 == 0:
                        event = {"stage": "train", "arm": arm, "epoch": epoch, "batch": i+1,
                                 "loss": totals["loss_sum"]/max(totals["candidate_events"], 1),
                                 "elapsed_s": totals["train_s"], "input_s": totals["input_s"], "gradient_max": monitor.gradient_max,
                                 "finite_advisories": dict(monitor.advisory_counts), **memory_stats(device)}
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
            evaluation = evaluate(model, data, arm, config, pause, log_directory=directory)
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
        cursor["phase"] = "audit"
        commit()
        final_weights = cpu_weights(model)
        if best_model is None:
            raise ValueError("No selected checkpoint after training")
        model.load_state_dict(best_model)
        try:
            replay = evaluate(model, data, arm, config, pause, report["best"]["threshold"], directory)
            if replay["selected"] != report["best"]["evaluation"]["selected"]:
                raise AssertionError("Best weights failed exact Dev metric replay")
            health = final_health(model, data, arm, config, pause, directory)
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
        report = {**report, "status": "failed", "failure": failure}
        atomic_json(directory/"training.json", report)
        progress(directory, status="failed", **failure)
        print(json.dumps(failure), flush=True)
        return report


def run_suite(teacher, encoded, directory, epochs=30, checkpoint_batches=100, retry_failed=False):
    directory = run_path(directory)
    if any(not (directory/a/"last.pt").exists() for a in ARMS):
        raise ValueError("Run prepare first to migrate all three existing checkpoints")
    with process_lock(directory/".suite.lock"), PauseControl(directory) as pause:
        pause.path.unlink(missing_ok=True)
        progress(directory, status="loading_cache", requested_epochs=epochs, monitoring_policy=POLICY)
        device = setup_device("cuda", profile_for("ann_linear")["training"])
        data = ResidentData(teacher, encoded, device)
        print(json.dumps({"stage": "resident_ready", **data.stats, **memory_stats(device)}), flush=True)
        reports = []
        for arm in ARMS:
            if pause.pending():
                break
            arm_dir = directory/arm
            torch.cuda.reset_peak_memory_stats(device)
            progress(directory, status="running", active_control=arm, requested_epochs=epochs, monitoring_policy=POLICY)
            with process_lock(arm_dir/".run.lock"):
                report = run_arm(data, arm_dir, arm, epochs, pause, checkpoint_batches, retry_failed)
            reports.append(report)
            comparison(directory, reports)
            gc.collect()
            torch.cuda.empty_cache()
            if report["status"] == "paused":
                break
        result = comparison(directory, reports)
        status = "paused" if pause.pending() or any(r["status"] == "paused" for r in reports) else result["status"]
        progress(directory, status=status, completed_controls=[r["arm"] for r in reports if r["status"] == "complete"],
                 comparison=str(directory/"comparison.json"), monitoring_policy=POLICY)
        return {**result, "status": status}

