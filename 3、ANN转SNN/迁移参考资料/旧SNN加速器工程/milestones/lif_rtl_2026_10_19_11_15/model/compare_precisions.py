"""Generate deterministic functional accuracy metrics for the six LIF tops."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

from precision_models import float_lif_step, int8_lif_step


SEED = 24307140011


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def error_summary(
    errors: list[float],
    signed_errors: list[float],
    references: list[float],
    *,
    relative_floor: float,
    normalization_range: float,
) -> dict[str, float]:
    squared = [value * value for value in errors]
    relative = [
        error / max(abs(reference), 1e-9)
        for error, reference in zip(errors, references)
        if math.isfinite(error)
        and math.isfinite(reference)
        and abs(reference) >= relative_floor
    ]
    return {
        "mae": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(squared) / len(squared)),
        "p95_abs_error": percentile(errors, 0.95),
        "max_abs_error": max(errors),
        "mean_relative_error_pct": 100.0 * sum(relative) / len(relative),
        "mean_signed_error": sum(signed_errors) / len(signed_errors),
        "normalized_mae_pct": 100.0 * sum(errors) / len(errors) / normalization_range,
    }


def compare_float(fmt: str, count: int, trajectory_count: int, steps: int) -> dict:
    rng = random.Random(SEED + (16 if fmt == "fp16" else 32))
    errors: list[float] = []
    signed_errors: list[float] = []
    references: list[float] = []
    spike_mismatch = 0
    for _ in range(count):
        membrane = rng.uniform(0.0, 0.95)
        current = rng.uniform(-0.10, 0.25)
        alpha = rng.uniform(0.75, 0.999)
        threshold = 0.80
        exact = float_lif_step(
            membrane, current, alpha=alpha, threshold=threshold, fmt=fmt
        )
        approximate = float_lif_step(
            membrane,
            current,
            alpha=alpha,
            threshold=threshold,
            fmt=fmt,
            approximate=True,
        )
        errors.append(abs(approximate.candidate - exact.candidate))
        signed_errors.append(approximate.candidate - exact.candidate)
        references.append(exact.candidate)
        spike_mismatch += exact.spike != approximate.spike

    trajectory_errors: list[float] = []
    trajectory_spike_mismatch = 0
    exact_spikes = 0
    approximate_spikes = 0
    for _ in range(trajectory_count):
        exact_membrane = 0.0
        approximate_membrane = 0.0
        for _ in range(steps):
            current = rng.uniform(-0.04, 0.13)
            exact = float_lif_step(
                exact_membrane, current, alpha=0.9375, threshold=0.75, fmt=fmt
            )
            approximate = float_lif_step(
                approximate_membrane,
                current,
                alpha=0.9375,
                threshold=0.75,
                fmt=fmt,
                approximate=True,
            )
            exact_membrane = exact.membrane
            approximate_membrane = approximate.membrane
            trajectory_errors.append(abs(approximate_membrane - exact_membrane))
            trajectory_spike_mismatch += exact.spike != approximate.spike
            exact_spikes += exact.spike
            approximate_spikes += approximate.spike

    total_trajectory_steps = trajectory_count * steps
    summary = error_summary(
        errors,
        signed_errors,
        references,
        relative_floor=0.05,
        normalization_range=0.80,
    )
    summary.update(
        {
            "single_step_samples": count,
            "single_step_spike_mismatch_pct": 100.0 * spike_mismatch / count,
            "trajectory_steps": total_trajectory_steps,
            "trajectory_membrane_mae": sum(trajectory_errors) / len(trajectory_errors),
            "trajectory_spike_mismatch_pct": 100.0
            * trajectory_spike_mismatch
            / total_trajectory_steps,
            "exact_spike_count": exact_spikes,
            "approximate_spike_count": approximate_spikes,
        }
    )
    return summary


def compare_int8(count: int, trajectory_count: int, steps: int) -> dict:
    rng = random.Random(SEED + 8)
    errors: list[float] = []
    signed_errors: list[float] = []
    references: list[float] = []
    spike_mismatch = 0
    for _ in range(count):
        membrane = rng.randint(0, 95)
        current = rng.randint(-20, 35)
        exact = int8_lif_step(membrane, current)
        approximate = int8_lif_step(membrane, current, approximate=True)
        errors.append(float(abs(approximate.candidate - exact.candidate)))
        signed_errors.append(float(approximate.candidate - exact.candidate))
        references.append(float(exact.candidate))
        spike_mismatch += exact.spike != approximate.spike

    trajectory_errors: list[float] = []
    trajectory_spike_mismatch = 0
    exact_spikes = 0
    approximate_spikes = 0
    for _ in range(trajectory_count):
        exact_membrane = 0
        approximate_membrane = 0
        for _ in range(steps):
            current = rng.randint(5, 55)
            exact = int8_lif_step(exact_membrane, current)
            approximate = int8_lif_step(
                approximate_membrane, current, approximate=True
            )
            exact_membrane = exact.membrane
            approximate_membrane = approximate.membrane
            trajectory_errors.append(float(abs(approximate_membrane - exact_membrane)))
            trajectory_spike_mismatch += exact.spike != approximate.spike
            exact_spikes += exact.spike
            approximate_spikes += approximate.spike

    total_trajectory_steps = trajectory_count * steps
    summary = error_summary(
        errors,
        signed_errors,
        references,
        relative_floor=1.0,
        normalization_range=127.0,
    )
    summary.update(
        {
            "single_step_samples": count,
            "single_step_spike_mismatch_pct": 100.0 * spike_mismatch / count,
            "trajectory_steps": total_trajectory_steps,
            "trajectory_membrane_mae": sum(trajectory_errors) / len(trajectory_errors),
            "trajectory_spike_mismatch_pct": 100.0
            * trajectory_spike_mismatch
            / total_trajectory_steps,
            "exact_spike_count": exact_spikes,
            "approximate_spike_count": approximate_spikes,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--trajectories", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    result = {
        "schema_version": 1,
        "seed": SEED,
        "scope": "functional bit-oriented comparison; no synthesis data",
        "configuration": {
            "fp_equation": "candidate=round(round(alpha*membrane)+current), hard reset to zero",
            "fp_approximation": "Mitchell log-domain multiplication, FTZ subnormal policy",
            "int8_equation": "candidate=sat((membrane-(membrane>>>2))+current), hard reset to zero",
            "int8_approximation": "LOA with K=2 low bits",
            "theoretical_mitchell_max_normal_multiply_relative_error_pct": 100.0 / 9.0,
            "int8_loa_pre_saturation_error_bound_lsb": 3,
        },
        "fp32": compare_float("fp32", args.samples, args.trajectories, args.steps),
        "fp16": compare_float("fp16", args.samples, args.trajectories, args.steps),
        "int8": compare_int8(args.samples, args.trajectories, args.steps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
