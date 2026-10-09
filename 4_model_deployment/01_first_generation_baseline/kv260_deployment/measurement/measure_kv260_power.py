"""Measure KV260 VCC_SOM power before and during one board-side command.

This reads Linux hwmon data exposed by the board's INA260.  VCC_SOM covers the
whole SOM, so the result is deliberately labelled as SOM power, not PL-only
power.  No third-party Python package is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Tuple


def find_power_node(explicit: Path | None) -> Tuple[Path, str]:
    if explicit is not None:
        node = explicit.resolve()
        if not node.is_file():
            raise FileNotFoundError(node)
        return node, node.parent.joinpath("name").read_text(encoding="utf-8").strip()
    candidates: List[Tuple[Path, str]] = []
    for directory in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name_path = directory / "name"
        power_path = directory / "power1_input"
        if not name_path.is_file() or not power_path.is_file():
            continue
        name = name_path.read_text(encoding="utf-8").strip()
        if "ina260" in name.lower():
            candidates.append((power_path.resolve(), name))
    if not candidates:
        raise FileNotFoundError(
            "no INA260 power1_input found under /sys/class/hwmon; "
            "use --power-node or install/enable the board telemetry driver"
        )
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path, _name in candidates)
        raise ValueError(f"multiple INA260 power nodes found; select one with --power-node: {paths}")
    return candidates[0]


def read_power_w(node: Path) -> float:
    # Linux hwmon power*_input ABI is expressed in microwatts.
    value = float(node.read_text(encoding="ascii").strip()) / 1_000_000.0
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid power sample from {node}: {value}")
    return value


def percentile(values: List[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty power sample")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent / 100.0) - 1)
    return float(ordered[index])


def summary(values: List[float]) -> Dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 95.0),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def atomic_json(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent,
        mode="w", encoding="utf-8", newline="\n", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-node", type=Path,
                        help="explicit INA260 power1_input sysfs node")
    parser.add_argument("--sample-rate-hz", type=float, default=10.0)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--task-count", type=int,
                        help="number of complete sequences/tasks executed by the command")
    parser.add_argument("--patch-count", type=int,
                        help="number of Proposal-SNN patches executed by the command")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to measure; place it after --")
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    if arguments.sample_rate_hz <= 0.0 or arguments.sample_rate_hz > 1000.0:
        raise ValueError("--sample-rate-hz must be in (0, 1000]")
    if arguments.idle_seconds <= 0.0:
        raise ValueError("--idle-seconds must be positive")
    if arguments.task_count is not None and arguments.task_count <= 0:
        raise ValueError("--task-count must be positive")
    if arguments.patch_count is not None and arguments.patch_count <= 0:
        raise ValueError("--patch-count must be positive")

    node, device_name = find_power_node(arguments.power_node)
    interval = 1.0 / arguments.sample_rate_hz
    samples: List[Dict[str, Any]] = []
    phase = {"value": "idle"}
    stop = threading.Event()
    sample_error: List[BaseException] = []

    def sampler() -> None:
        deadline = time.monotonic()
        while not stop.is_set():
            try:
                samples.append({
                    "monotonic_ns": time.monotonic_ns(),
                    "wall_time_ns": time.time_ns(),
                    "phase": phase["value"],
                    "power_w": read_power_w(node),
                })
            except BaseException as error:  # retain the sampling failure for the main thread
                sample_error.append(error)
                stop.set()
                return
            deadline += interval
            stop.wait(max(0.0, deadline - time.monotonic()))

    thread = threading.Thread(target=sampler, name="kv260-power-sampler", daemon=True)
    thread.start()
    time.sleep(arguments.idle_seconds)
    phase["value"] = "active"
    command_started_ns = time.monotonic_ns()
    completed = subprocess.run(command, check=False)
    command_finished_ns = time.monotonic_ns()
    stop.set()
    thread.join()
    if sample_error:
        raise sample_error[0]
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    idle = [float(item["power_w"]) for item in samples if item["phase"] == "idle"]
    active = [float(item["power_w"]) for item in samples if item["phase"] == "active"]
    if not idle or not active:
        raise ValueError("idle and active phases both require at least one power sample")

    arguments.csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("monotonic_ns", "wall_time_ns", "phase", "power_w")
        )
        writer.writeheader()
        writer.writerows(samples)

    idle_summary = summary(idle)
    active_summary = summary(active)
    duration_s = (command_finished_ns - command_started_ns) / 1.0e9
    dynamic_power = float(active_summary["mean"]) - float(idle_summary["mean"])
    active_energy_j = float(active_summary["mean"]) * duration_s
    dynamic_energy_j = dynamic_power * duration_s
    result: Dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "measurement_scope": "VCC_SOM_total",
        "tool": "Linux hwmon INA260 power1_input",
        "device_name": device_name,
        "power_node": str(node),
        "sample_rate_hz": arguments.sample_rate_hz,
        "idle_seconds": arguments.idle_seconds,
        "command": command,
        "command_returncode": completed.returncode,
        "command_duration_s": duration_s,
        "idle_power_w": idle_summary,
        "active_power_w": active_summary,
        "dynamic_power_w": dynamic_power,
        "pl_power_w": None,
        "pl_power_is_directly_measured": False,
        "active_energy_j": active_energy_j,
        "dynamic_energy_j": dynamic_energy_j,
        "energy_method": "active/dynamic mean VCC_SOM power multiplied by monotonic command duration",
        "task_count": arguments.task_count,
        "patch_count": arguments.patch_count,
        "active_energy_per_task_j": (
            active_energy_j / arguments.task_count if arguments.task_count else None
        ),
        "dynamic_energy_per_task_j": (
            dynamic_energy_j / arguments.task_count if arguments.task_count else None
        ),
        "active_energy_per_patch_j": (
            active_energy_j / arguments.patch_count if arguments.patch_count else None
        ),
        "dynamic_energy_per_patch_j": (
            dynamic_energy_j / arguments.patch_count if arguments.patch_count else None
        ),
        "samples_csv": str(arguments.csv.resolve()),
        "notes": (
            "VCC_SOM covers the whole SOM. active-idle is a workload dynamic-power proxy, "
            "not a direct PL-only measurement."
        ),
    }
    atomic_json(result, arguments.summary.resolve())
    print(json.dumps({
        "status": "ok",
        "scope": result["measurement_scope"],
        "idle_mean_w": idle_summary["mean"],
        "active_mean_w": active_summary["mean"],
        "dynamic_power_w": dynamic_power,
        "active_energy_j": active_energy_j,
        "dynamic_energy_j": dynamic_energy_j,
        "summary": str(arguments.summary.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
