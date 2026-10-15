"""Select existing linear controls for a bounded trial without changing their runtime."""
import gc
import json
from pathlib import Path

import torch

from .h2pro_snn.foundation import sha256
from .h2pro_snn.stream_data import atomic_json, run_path
from .h2pro_snn.training import setup_device, memory_stats
from .h2pro_longtrain_v1.data import ResidentData
from .h2pro_longtrain_v1.storage import load, process_lock, PauseControl, progress
from .h2pro_longtrain_v1.runtime import comparison
from .h2pro_longtrain_v1_1.models import IMPLEMENTATION, profile_for, POLICY
from .h2pro_longtrain_v1_1.migration import make_identity
from .h2pro_longtrain_v1_1.runtime import run_arm

CONTROLS = ("ann_linear", "snn_linear")


def launcher_sources():
    paths = [Path(__file__), IMPLEMENTATION/"software/scripts/train_h2pro_short.py",
             IMPLEMENTATION/"train_local_4060_short.ps1"]
    return {p.relative_to(IMPLEMENTATION).as_posix(): sha256(p) for p in paths}


def validate_plan(data, directory, epochs, arms=CONTROLS):
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(a not in CONTROLS for a in arms):
        raise ValueError("Select ann_linear and/or snn_linear; ANN-state is excluded from this launcher")
    rows = {}
    for arm in arms:
        saved = load(Path(directory)/arm/"last.pt")
        profile = profile_for(arm)
        expected = make_identity(data, arm, profile, saved["identity"].get("migration"))
        if saved["identity"] != expected:
            raise ValueError("Existing checkpoint source/config/cache/environment changed: "+arm)
        if not 1 <= epochs <= profile["training"]["schedule_epochs"] or epochs < saved["report"]["completed_epochs"]:
            raise ValueError("Total epoch budget is outside the existing checkpoint schedule: "+arm)
        rows[arm] = {"completed_epochs": saved["report"]["completed_epochs"], "cursor": saved["cursor"],
                     "best_epoch": saved["report"]["best"]["epoch"] if saved["report"]["best"] else None}
    return {"selected_controls": list(arms), "excluded_controls": ["ann_state"],
            "requested_total_epochs": epochs, "automatic_extension": False,
            "scheduler_note": "Original 30-epoch learning-rate schedule retained; stop budget only changes",
            "monitoring_policy": POLICY, "resume_points": rows, "launcher_sources": launcher_sources()}


def publish(directory, reports, plan, status):
    # Include the held model as historical context, without opening or saving its checkpoint.
    by_arm = {r["arm"]: r for r in reports}
    for arm in ("ann_state", *CONTROLS):
        path = Path(directory)/arm/"training.json"
        if arm not in by_arm and path.exists():
            by_arm[arm] = json.loads(path.read_text(encoding="utf-8"))
    result = comparison(directory, list(by_arm.values()))
    result.update(status=status, selected_controls=plan["selected_controls"],
                  excluded_controls=plan["excluded_controls"], requested_total_epochs=plan["requested_total_epochs"],
                  scope="Selected short trial only; ANN-state held, not a completed three-model 30-epoch suite",
                  automatic_extension=False)
    atomic_json(Path(directory)/"comparison.json", result)
    return result


def run_queue(data, directory, epochs, pause, arms=CONTROLS, checkpoint_batches=100, retry_failed=False):
    """Caller owns suite lock. Native run_arm retains all numerical and recovery behavior."""
    directory = Path(directory)
    if checkpoint_batches < 1:
        raise ValueError("Checkpoint interval must be positive")
    plan = validate_plan(data, directory, epochs, arms)
    plan.update(status="running", launcher="user", resume_checkpoint_directory=str(directory))
    atomic_json(directory/"short_training_plan.json", plan)
    reports = []
    for arm in plan["selected_controls"]:
        if pause.pending():
            break
        if data.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(data.device)
        progress(directory, status="running", active_control=arm, requested_epochs=epochs,
                 selected_controls=plan["selected_controls"], excluded_controls=["ann_state"],
                 automatic_extension=False, monitoring_policy=POLICY)
        with process_lock(directory/arm/".run.lock"):
            report = run_arm(data, directory/arm, arm, epochs, pause, checkpoint_batches, retry_failed)
        reports.append(report)
        publish(directory, reports, plan, "running")
        gc.collect()
        if data.device.type == "cuda":
            torch.cuda.empty_cache()
        if report["status"] == "paused":
            break
    if pause.pending() or any(r["status"] == "paused" for r in reports):
        status = "paused"
    elif len(reports) == len(plan["selected_controls"]) and all(r["status"] == "complete" for r in reports):
        status = "short_budget_complete"
    else:
        status = "incomplete"
    result = publish(directory, reports, plan, status)
    plan.update(status=status, review_required=status == "short_budget_complete")
    atomic_json(directory/"short_training_plan.json", plan)
    progress(directory, status=status, selected_controls=plan["selected_controls"], excluded_controls=["ann_state"],
             requested_epochs=epochs, automatic_extension=False, review_required=status == "short_budget_complete",
             completed_controls=[r["arm"] for r in reports if r["status"] == "complete"],
             comparison=str(directory/"comparison.json"), monitoring_policy=POLICY)
    return result


def execute(teacher, encoded, directory, epochs=5, arms=CONTROLS, checkpoint_batches=100,
            retry_failed=False, check_only=False):
    directory = run_path(directory)
    with process_lock(directory/".suite.lock"), PauseControl(directory) as pause:
        device = setup_device("cuda", profile_for("ann_linear")["training"])
        data = ResidentData(teacher, encoded, device)
        if check_only:
            return {"status": "ready", "formal_training_started": False,
                    **validate_plan(data, directory, epochs, arms), "cache": data.stats,
                    "memory": memory_stats(device)}
        # Validate before acknowledging the old pause request or changing progress.
        validate_plan(data, directory, epochs, arms)
        pause.path.unlink(missing_ok=True)
        print(json.dumps({"stage": "resident_ready", **data.stats, **memory_stats(device)}), flush=True)
        return run_queue(data, directory, epochs, pause, arms, checkpoint_batches, retry_failed)
