"""Reproduce algorithmic complexity and draw the external EV-UAV comparison.

The report distinguishes executed dense INT8 MACs from effective non-zero
synaptic MACs.  The latter is a sparsity-aware *potential*, not measured CPU
speed or energy.  Validation/Test files are fixed by data/index.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tracemalloc

import numpy as np


HERE = Path(__file__).resolve().parent
METRICS = HERE.parent
PROJECT = METRICS.parent
IMPLEMENTATION = PROJECT / "1、H2模型改进" / "implementation"
sys.path.insert(0, str(IMPLEMENTATION / "software"))

if "--plot-only" not in sys.argv:
    from wangdao.a4_2_runtime import load_a4_2_refiner  # noqa: E402
    from wangdao.data import load_events  # noqa: E402
    from wangdao.h2.numpy_model import load_export, quantized_mlp  # noqa: E402
    from wangdao.h2_research import FrozenH2Pipeline, augment_refiner_features  # noqa: E402


WEIGHTS = IMPLEMENTATION / "software" / "weights"
INDEX = IMPLEMENTATION / "data" / "index.json"
DATASET = IMPLEMENTATION / "data" / "EV-UAV-dataset"
H2_WEIGHTS = WEIGHTS / "H2_D1_seed42.npz"
PLUS_WEIGHTS = WEIGHTS / "A4-2_seed42_epoch008_int8.npz"
SPLITS = {
    "Validation": ("validation", "val"),
    "Test": ("test", "test"),
}
EXPECTED_SHA256 = {
    H2_WEIGHTS.name: "21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2",
    PLUS_WEIGHTS.name: "d7d42a3876c37497bdcd5206e6f35a5f949b0230dde0675fd128f989b70d05c4",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def static_profile(layers: tuple) -> dict:
    return {
        "weight_parameters": sum(int(w.size) for w, _ in layers),
        "nonzero_weights": sum(int(np.count_nonzero(w)) for w, _ in layers),
        "weight_bytes": sum(int(w.nbytes) for w, _ in layers),
        "scale_bytes": sum(int(s.nbytes) for _, s in layers),
        "macs_per_call": sum(int(w.shape[0] * w.shape[1]) for w, _ in layers),
        "dimensions": [int(layers[0][0].shape[0])]
        + [int(w.shape[1]) for w, _ in layers],
    }


def profile_mlp(values: np.ndarray, layers: tuple) -> dict:
    """Exact quantized forward pass, with counts at each layer input.

    The number of effective MACs is sum_j active_inputs[j] * nonzero_fanout[j].
    This includes zero-skipping potential only; NumPy still runs dense matmul.
    """
    value = np.ascontiguousarray(values, dtype=np.uint8)
    dense = 0
    effective = 0
    transform_elements = 0
    activation_nonzero = []
    layer_counts = []
    for index, (weight, scale) in enumerate(layers):
        active = np.count_nonzero(value, axis=0).astype(np.int64)
        fanout = np.count_nonzero(weight, axis=1).astype(np.int64)
        layer_dense = int(value.shape[0] * weight.size)
        layer_effective = int(active @ fanout)
        layer_counts.append({
            "dense_macs": layer_dense,
            "effective_macs": layer_effective,
            "active_inputs": int(active.sum()),
            "input_elements": int(value.size),
        })
        dense += layer_dense
        effective += layer_effective
        activation_nonzero.append(int(active.sum()))
        raw = np.matmul(value.astype(np.int32), weight.astype(np.int32))
        floating = raw.astype(np.float64) * scale[None, :].astype(np.float64) / 255.0
        if index + 1 < len(layers):
            transform_elements += int(floating.size)
            value = np.clip(
                np.rint(np.clip(floating, 0.0, 1.0) * 255.0), 0.0, 255.0
            ).astype(np.uint8)
        else:
            logits = floating.astype(np.float32)
    return {
        "dense_macs": dense,
        "effective_macs": effective,
        "activation_transform_elements": transform_elements,
        "activation_nonzero": activation_nonzero,
        "layer_counts": layer_counts,
        "logits": logits,
    }


def add_counts(a: dict, b: dict) -> dict:
    keys = ("dense_macs", "effective_macs", "activation_transform_elements")
    return {key: int(a[key] + b[key]) for key in keys}


def make_static(h2, plus) -> dict:
    proposal = static_profile(h2.proposal_layers)
    h2_refiner = static_profile(h2.refiner_layers)
    plus_refiner = static_profile(plus.layers)
    result = {}
    for name, refiner in (("H2", h2_refiner), ("H2-plus", plus_refiner)):
        result[name] = {
            "proposal": proposal,
            "refiner": refiner,
            "total_weight_parameters": proposal["weight_parameters"] + refiner["weight_parameters"],
            "total_nonzero_weights": proposal["nonzero_weights"] + refiner["nonzero_weights"],
            "total_weight_bytes": proposal["weight_bytes"] + refiner["weight_bytes"],
            "total_scale_bytes": proposal["scale_bytes"] + refiner["scale_bytes"],
            "total_tensor_payload_bytes": proposal["weight_bytes"] + refiner["weight_bytes"] + proposal["scale_bytes"] + refiner["scale_bytes"],
            "persistent_neuron_state_bytes": 0,
        }
    return result


def measured_python_working_peak(events: np.ndarray, baseline: FrozenH2Pipeline, layers: tuple, model: str) -> int:
    """Measure replay's traced allocations after the raw input and weights load."""
    tracemalloc.start()
    try:
        prepared = baseline.prepare(events)
        values = prepared.features if model == "H2" else augment_refiner_features(prepared.features, prepared.history)
        logits = quantized_mlp(values, layers)
        probabilities = np.zeros(len(events), dtype=np.float32)
        probabilities[prepared.indices] = logits[:, 0]
        del probabilities
        return int(tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()


def save_output(report: dict) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "模型复杂度完整结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model in ("H2", "H2-plus"):
        subdir = HERE / ("0、原始H2模型" if model == "H2" else "1、H2-plus模型")
        subdir.mkdir(exist_ok=True)
        payload = {"model": model, "protocol": report["protocol"], "weights": report["source_weights_sha256"], "splits": {role: result["models"][model] for role, result in report["results"].items()}}
        (subdir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def draw_charts(report: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for preferred in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(preferred.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = preferred
            break
    plt.rcParams["axes.unicode_minus"] = False
    chart_dir = HERE / "Charts and pictures"
    chart_dir.mkdir(exist_ok=True)
    with (HERE / "EV-UAV" / "summary.json").open(encoding="utf-8") as stream:
        evuav = json.load(stream)
    ev_parameters = int(evuav["paper"]["reported_weight_parameters"])
    labels = ("EV-SpSegNet\n（论文）", "H2", "H2-plus")
    colors = ("#7B8794", "#2764AB", "#E67835")
    for role in SPLITS:
        models = report["results"][role]["models"]
        h, p = models["H2"], models["H2-plus"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 9.4), facecolor="white")
        fig.suptitle(f"EV-SpSegNet、H2 与 H2-plus 算法复杂度对比（{role}）", x=0.07, y=0.975, ha="left", fontsize=20, color="#18324E", fontweight="bold")
        fig.text(0.07, 0.931, "EV-SpSegNet 使用 ICCV 2025 论文报告值；H2 / H2-plus 为 exact INT8，每组 24 条完整 8 s 序列。", fontsize=11, color="#526376")

        def bars(ax, values, title, unit, digits=1, log_scale=False):
            positions = (2, 1, 0)
            numeric = [value for value in values if value is not None]
            maximum = max(numeric)
            for y, value, color in zip(positions, values, colors):
                if value is None:
                    continue
                if log_scale:
                    ax.barh(y, value - 1, left=1, color=color, height=0.45)
                    ax.text(value * 1.07, y, f"{value:,.{digits}f} {unit}", va="center", fontsize=10.5, color="#18324E")
                else:
                    ax.barh(y, value, color=color, height=0.45)
                    ax.text(value + maximum * 0.02, y, f"{value:,.{digits}f} {unit}", va="center", fontsize=10.5, color="#18324E")
            ax.set_yticks(positions, labels)
            ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
            if log_scale:
                ax.set_xscale("log")
                ax.set_xlim(5_000, maximum * 1.65)
                ax.set_xlabel("对数坐标", fontsize=9, color="#64748B")
            else:
                ax.set_xlim(0, maximum * 1.28)
            for y, value in zip(positions, values):
                if value is None:
                    ax.text(maximum * 0.03, y, "原论文未报告", va="center", fontsize=10.5, color="#7B8794", fontstyle="italic")
            ax.grid(axis="x", color="#E5EAF0", linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)

        bars(axes[0, 0], [ev_parameters, h["total_weight_parameters"], p["total_weight_parameters"]], "A01 参数量", "个", 0, log_scale=True)
        bars(axes[0, 1], [None, h["total_tensor_payload_bytes"] / 1024, p["total_tensor_payload_bytes"] / 1024], "A02 权重 + 量化尺度", "KiB", 2)
        bars(axes[1, 0], [None, h["max_core_tensor_bytes_observed"] / (1024 ** 2), p["max_core_tensor_bytes_observed"] / (1024 ** 2)], "A03 核心张量预算上限", "MiB", 2)
        bars(axes[1, 1], [None, h["dense_macs_per_sequence"] / 1e6, p["dense_macs_per_sequence"] / 1e6], "A04 平均每条序列稠密 MAC", "M", 2)
        fig.text(0.07, 0.025, "EV-SpSegNet 的 4.0M 为论文舍入参数量；A02—A04 原论文未报告，不做推算。A03 是核心张量体积预算，不是同时存活峰值或板级 SRAM。", fontsize=10, color="#64748B")
        fig.tight_layout(rect=(0.06, 0.065, 0.98, 0.90), h_pad=3.2, w_pad=2.7)
        path = chart_dir / f"EV-UAV_vs_H2_vs_H2-plus_{role}_模型复杂度.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(path, flush=True)


def summarize(rows: list[dict], static: dict, expected_events: int) -> dict:
    total = lambda key: sum(int(row[key]) for row in rows)
    events = total("events")
    if events != expected_events:
        raise AssertionError(f"Event count {events} != index count {expected_events}")
    seconds = len(rows) * 8  # fixed full 8 s EV-UAV sequences
    return {
        **static,
        "sequences": len(rows),
        "events": events,
        "candidates": total("candidates"),
        "candidate_fraction": total("candidates") / events,
        "duration_s": seconds,
        "proposal_calls": total("proposal_calls"),
        "refiner_calls": total("candidates"),
        "proposal_dense_macs": total("proposal_dense_macs"),
        "proposal_effective_macs": total("proposal_effective_macs"),
        "refiner_dense_macs": total("refiner_dense_macs"),
        "refiner_effective_macs": total("refiner_effective_macs"),
        "total_dense_macs": total("proposal_dense_macs") + total("refiner_dense_macs"),
        "total_effective_macs": total("proposal_effective_macs") + total("refiner_effective_macs"),
        "activation_transform_elements": total("activation_transform_elements"),
        "proposal_sigmoid_elements": total("proposal_sigmoid_elements"),
        "refiner_softmax_items": total("candidates"),
        "history_feature_values": total("history_feature_values"),
        "snn_neuron_updates": 0,
        "effective_acs": 0,
        "dense_macs_per_sequence": (total("proposal_dense_macs") + total("refiner_dense_macs")) / len(rows),
        "effective_macs_per_sequence": (total("proposal_effective_macs") + total("refiner_effective_macs")) / len(rows),
        "dense_macs_per_second_input": (total("proposal_dense_macs") + total("refiner_dense_macs")) / seconds,
        "effective_macs_per_second_input": (total("proposal_effective_macs") + total("refiner_effective_macs")) / seconds,
        "max_core_tensor_bytes_observed": max(row["core_tensor_bytes"] for row in rows),
        "core_tensor_definition": "sum of retained patch features, heatmap, event base features, refiner features, H2-plus history features (not used by H2), output probability; sizing budget, not simultaneous or RSS peak",
        "work_memory_peak_bytes": None,
        "work_memory_peak_status": "not measured: no hardware-independent peak from the current frozen artifacts",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("Validation", "Test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="read-only diagnostic limit; never writes official outputs")
    parser.add_argument("--memory-only", action="store_true", help="add separately measured Python replay memory peaks")
    parser.add_argument("--plot-only", action="store_true", help="make PNG comparisons from the saved JSON")
    args = parser.parse_args()
    if args.plot_only:
        with (HERE / "模型复杂度完整结果.json").open(encoding="utf-8") as stream:
            draw_charts(json.load(stream))
        return
    for path in (H2_WEIGHTS, PLUS_WEIGHTS):
        actual = sha256(path)
        if actual != EXPECTED_SHA256[path.name]:
            raise ValueError(f"Frozen weight SHA-256 mismatch: {path.name}")
    with INDEX.open(encoding="utf-8") as stream:
        index = json.load(stream)
    h2 = load_export(H2_WEIGHTS)
    plus = load_a4_2_refiner(PLUS_WEIGHTS)
    baseline = FrozenH2Pipeline(H2_WEIGHTS)
    static = make_static(h2, plus)
    if args.memory_only:
        with (HERE / "模型复杂度完整结果.json").open(encoding="utf-8") as stream:
            report = json.load(stream)
        for role, (_, dataset_dir) in SPLITS.items():
            result = report["results"][role]
            rows_h2 = result["per_sequence"]["H2"]
            result["models"]["H2"]["max_core_tensor_bytes_observed"] = max(row["core_tensor_bytes"] for row in rows_h2)
            sequence = max(result["per_sequence"]["H2-plus"], key=lambda row: row["events"])["sequence"]
            events = load_events(DATASET / dataset_dir / sequence)
            for model, layers in (("H2", h2.refiner_layers), ("H2-plus", plus.layers)):
                peak = measured_python_working_peak(events, baseline, layers, model)
                result["models"][model]["work_memory_peak_bytes"] = peak
                result["models"][model]["work_memory_peak_status"] = "measured Python/NumPy traced allocations; raw input and loaded weights excluded; software-specific, not hardware SRAM/RSS peak"
                result["models"][model]["work_memory_peak_sequence"] = sequence
                print(f"{role} {model} {sequence}: traced heap peak {peak:,} B", flush=True)
        save_output(report)
        return
    roles = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    output = {}
    for role, (index_role, dataset_dir) in roles.items():
        expected_files = index["roles"][index_role]
        if args.limit:
            expected_files = expected_files[: args.limit]
        source_rows = {}
        for model, rel in (
            ("H2", "0、原始H2模型/统一int8评测"),
            ("H2-plus", "1、H2-plus模型"),
        ):
            path = METRICS / "A、模型准确度" / rel / role.lower() / "int8" / "per_sequence.csv"
            import csv
            with path.open(encoding="utf-8-sig", newline="") as stream:
                source_rows[model] = {row["file"]: row for row in csv.DictReader(stream)}
        rows_by_model = {"H2": [], "H2-plus": []}
        for sequence_name in expected_files:
            events = load_events(DATASET / dataset_dir / sequence_name)
            prepared = baseline.prepare(events)
            patch = None
            from wangdao.h2.representation import build_patch_features
            patch = build_patch_features(events, prepared.windows)
            proposal = profile_mlp(patch.reshape(-1, 128), h2.proposal_layers)
            probe = quantized_mlp(patch.reshape(-1, 128)[:8], h2.proposal_layers)
            if not np.array_equal(proposal["logits"][:8], probe):
                raise AssertionError("Proposal activity pass diverges from exact-int8 runtime")
            proposal_calls = int(patch.shape[0] * patch.shape[1])
            candidates = len(prepared.indices)
            for model, refiner_features, layers in (
                ("H2", prepared.features, h2.refiner_layers),
                ("H2-plus", augment_refiner_features(prepared.features, prepared.history), plus.layers),
            ):
                expected = source_rows[model].get(sequence_name)
                if expected is None or int(expected["events"]) != len(events) or int(expected["candidate_events"]) != candidates:
                    raise AssertionError(f"A/B population mismatch: {model} {role} {sequence_name}")
                refiner = profile_mlp(refiner_features, layers)
                probe = quantized_mlp(refiner_features[:8], layers)
                if not np.array_equal(refiner["logits"][:8], probe):
                    raise AssertionError("Refiner activity pass diverges from exact-int8 runtime")
                core_tensor_bytes = (
                    patch.nbytes + prepared.heatmaps.nbytes + prepared.base_features.nbytes
                    + refiner_features.nbytes
                    + (prepared.history.nbytes if model == "H2-plus" else 0)
                    + len(events) * 4
                )
                rows_by_model[model].append({
                    "sequence": sequence_name,
                    "events": len(events),
                    "candidates": candidates,
                    "proposal_calls": proposal_calls,
                    "proposal_dense_macs": proposal["dense_macs"],
                    "proposal_effective_macs": proposal["effective_macs"],
                    "refiner_dense_macs": refiner["dense_macs"],
                    "refiner_effective_macs": refiner["effective_macs"],
                    "activation_transform_elements": proposal["activation_transform_elements"] + refiner["activation_transform_elements"],
                    "proposal_sigmoid_elements": int(proposal["logits"].size),
                    "history_feature_values": candidates * (4 if model == "H2-plus" else 0),
                    "core_tensor_bytes": int(core_tensor_bytes),
                })
            print(f"{role} {sequence_name}: events={len(events)} candidates={candidates}", flush=True)
        output[role] = {
            "models": {
                model: summarize(rows, static[model], sum(int(source_rows["H2"][name]["events"]) for name in expected_files) if args.limit else index["role_summaries"][index_role]["events"])
                for model, rows in rows_by_model.items()
            },
            "per_sequence": rows_by_model,
        }
    if args.limit:
        print(json.dumps({role: {model: result["total_dense_macs"] for model, result in data["models"].items()} for role, data in output.items()}, ensure_ascii=False))
        return
    report = {
        "protocol": "formal-ann-algorithm-complexity-v1",
        "numeric_path": "exact_int8_replay",
        "source_index": str(INDEX),
        "source_weights_sha256": {path.name: sha256(path) for path in (H2_WEIGHTS, PLUS_WEIGHTS)},
        "scope": "Proposal + conditional Refiner; all 80 x 100 ms windows in each 8 s sequence",
        "effective_macs_definition": "sum over layers of nonzero input activations times nonzero outgoing weights; potential on sparsity-aware hardware; dense matmul executed in this replay",
        "results": output,
    }
    save_output(report)


if __name__ == "__main__":
    main()
