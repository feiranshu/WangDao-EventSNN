"""Static cycle budgets for the existing, non-overlapped 8-lane Proposal RTL."""
from __future__ import annotations

import numpy as np

INPUTS = (128, 48, 24)
OUTPUTS = (48, 24, 16)
STEPS = 4
LANES = 8
LOAD_CYCLES = 512
COMPUTE_CYCLES = tuple(STEPS * c * (2 * (n // LANES) + 4) for n, c in zip(INPUTS, OUTPUTS))
PATCH_CYCLES = LOAD_CYCLES + sum(COMPUTE_CYCLES)
BENCH_BOUNDARY_CYCLES = 1
SCREEN_REDUCTION = .05


def group_activity(values: np.ndarray, layer: int) -> tuple[dict, np.ndarray]:
    if layer not in range(3) or values.ndim != 3 or values.shape[1:] != (STEPS, INPUTS[layer]) or not len(values):
        raise ValueError("expected nonempty [patch,4,layer_inputs]")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values > (255 if layer == 0 else 1)):
        raise ValueError("inputs must be uint8 features or binary spikes")
    grouped = values.reshape(len(values), STEPS, INPUTS[layer] // LANES, LANES)
    active = np.count_nonzero(grouped, axis=-1)
    empty = active == 0
    counts = empty.sum(axis=(1, 2)).astype(np.int64)
    return {"input_values": int(values.size), "nonzero_inputs": int(np.count_nonzero(values)),
            "input_nonzero_fraction": float(np.count_nonzero(values) / values.size),
            "unique_input_groups": int(active.size), "empty_input_groups": int(empty.sum()),
            "empty_group_fraction": float(empty.mean()),
            "group_active_lane_histogram": np.asarray(np.bincount(active.ravel(), minlength=9), dtype=np.int64).tolist(),
            "scheduled_group_visits": int(active.size * OUTPUTS[layer]),
            "skippable_group_sum_visits": int(empty.sum() * OUTPUTS[layer]),
            "scheduled_lane_products_or_selects": int(values.size * OUTPUTS[layer]),
            "nonzero_input_lane_products_or_selects": int(np.count_nonzero(values) * OUTPUTS[layer]),
            "lane_count_scope": "input-dependent opportunities; includes zero weights; not FPGA operations or energy"}, counts


def cycle_budget(empty_counts: np.ndarray, layers: tuple[int, ...], *, overhead: str = "none") -> np.ndarray:
    """Skip GROUP_SUM only; PRODUCT still spends one cycle writing the zero partial."""
    if empty_counts.ndim != 2 or empty_counts.shape[1] != 3 or not np.issubdtype(empty_counts.dtype, np.integer):
        raise ValueError("empty counts must be [patch,3] integers")
    limits = np.array(INPUTS) // LANES * STEPS
    if np.any(empty_counts < 0) or np.any(empty_counts > limits):
        raise ValueError("empty-group count out of range")
    if len(set(layers)) != len(layers) or any(i not in range(3) for i in layers):
        raise ValueError("invalid or repeated layer")
    if overhead not in ("none", "per_output_step", "per_group_visit"):
        raise ValueError("unknown overhead scenario")
    cycles = np.full(len(empty_counts), PATCH_CYCLES, dtype=np.int64)
    for layer in layers:
        cycles -= empty_counts[:, layer] * OUTPUTS[layer]
        if overhead == "per_output_step":
            cycles += STEPS * OUTPUTS[layer]
        elif overhead == "per_group_visit":
            cycles += STEPS * OUTPUTS[layer] * (INPUTS[layer] // LANES)
    return cycles


def distribution(values: np.ndarray) -> dict:
    if not len(values):
        raise ValueError("empty distribution")
    return {"minimum": int(np.min(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "maximum": int(np.max(values)),
            "mean": float(np.mean(values)), "sum": int(np.sum(values))}


def summarize_cycles(empty_counts: np.ndarray) -> dict:
    variants = {}
    for name, layers in (("layer1", (0,)), ("layer2", (1,)), ("layer3", (2,)), ("all_layers", (0, 1, 2))):
        rows = {}
        for overhead in ("none", "per_output_step", "per_group_visit"):
            cycles = cycle_budget(empty_counts, layers, overhead=overhead)
            rows[overhead] = {"patch_cycles": distribution(cycles),
                              "whole_proposal_cycle_reduction": float(1 - cycles.sum() / (len(cycles) * PATCH_CYCLES)),
                              "break_even_frequency_ratio": float(cycles.mean() / PATCH_CYCLES)}
        rows["saved_cycles_without_extra_overhead"] = int(len(empty_counts) * PATCH_CYCLES - cycle_budget(empty_counts, layers).sum())
        variants[name] = rows
    return {"patches": len(empty_counts), "fixed_patch_cycles": PATCH_CYCLES,
            "fixed_total_cycles": len(empty_counts) * PATCH_CYCLES, "variants": variants}


def recommend(windows: list[dict], complete: bool) -> dict:
    candidates = {}
    for layer in range(3):
        name = f"layer{layer + 1}"
        reductions = [w["modes"][mode]["cycle_budget"]["variants"][name]["per_output_step"]["whole_proposal_cycle_reduction"]
                      for w in windows for mode in ("exact", "approximate")]
        candidates[name] = {"mean_reduction_with_extra_output_cycle": float(np.mean(reductions)),
                            "minimum_window_reduction_with_extra_output_cycle": float(np.min(reductions)),
                            "passes_engineering_screen": bool(np.mean(reductions) >= SCREEN_REDUCTION and np.min(reductions) >= 0)}
    eligible = [k for k, v in candidates.items() if v["passes_engineering_screen"]]
    chosen = max(eligible, key=lambda k: candidates[k]["mean_reduction_with_extra_output_cycle"]) if complete and eligible else None
    return {"candidates": candidates, "recommended_first_layer": chosen,
            "screen": {"minimum_mean_whole_proposal_reduction": SCREEN_REDUCTION,
                       "no_regression_on_selected_windows": True,
                       "margin": "one extra cycle per selected-layer output channel/time step",
                       "purpose": "engineering prioritization, not hardware acceptance or an accuracy gate"},
            "next": "implement_single_layer_group_zero_bypass" if chosen else
                    ("retain_fixed_schedule_expand_replay" if complete else "complete_representative_profile"),
            "rtl_changed": False, "hardware_mainline_changed": False, "fpga_benefit_proven": False}


def cycle_contract() -> dict:
    return {"lanes": LANES, "time_steps": STEPS, "inputs": list(INPUTS), "outputs": list(OUTPUTS),
            "groups_per_output_step": [n // LANES for n in INPUTS],
            "external_input_cycles": LOAD_CYCLES, "layer_compute_and_emit_cycles": list(COMPUTE_CYCLES),
            "fixed_proposal_cycles": PATCH_CYCLES, "historical_testbench_boundary_cycles": BENCH_BOUNDARY_CYCLES,
            "state_cycles": {"PRODUCT": 1, "GROUP_SUM": 1, "FINAL_SUM": 1, "ALIGN": 1, "UPDATE": 1, "EMIT": 1},
            "load_overlap": "L2/L3 LOAD handshakes coincide with upstream EMIT; do not add 192+96 again",
            "candidate": "PRODUCT sees a zero mask, writes partial[group]=0 and advances; GROUP_SUM skipped",
            "saved_cycles_per_zero_group_visit": 1,
            "mask_storage_bits_by_layer": [STEPS * n // LANES for n in INPUTS],
            "mask_assumption": "input nonzero flags accumulated while LOAD accepts each group; no extra LOAD cycles",
            "preserved": ["all final-tree partial slots and input order", "24-bit saturation and rounding",
                          "all membrane/leak/reset updates", "four-step exact readout sum", "all EMIT cycles"],
            "excluded": ["external stalls", "cross-patch overlap", "Refiner/host latency", "actual clock/resources/power"],
            "frequency_ratio_meaning": "To reduce compute latency, new frequency / old frequency must exceed new cycles / old cycles."}
