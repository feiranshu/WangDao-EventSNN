"""Reproduce algorithmic complexity and write CSV comparison tables.

The report distinguishes executed dense INT8 MACs from effective non-zero
synaptic MACs.  The latter is a sparsity-aware *potential*, not measured CPU
speed or energy.  Validation/Test files are fixed by data/index.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tracemalloc

import numpy as np


IMPLEMENTATION = Path(__file__).resolve().parents[2]
PROJECT = IMPLEMENTATION.parents[1]
METRICS = PROJECT / "5_model_evaluation"
COMPLEXITY_ROOT = METRICS / "B_model_complexity"
RESULT_ROOT = COMPLEXITY_ROOT / "02_model_records" / "formal"
MODEL_IDS = {"H2": "h2__software_exact_int8", "H2-plus": "h2_plus__software_exact_int8"}
sys.path.insert(0, str(IMPLEMENTATION / "software"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_evaluation_tables import build_complexity  # noqa: E402

if "--tables-only" not in sys.argv:
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
    for model in ("H2", "H2-plus"):
        subdir = RESULT_ROOT / MODEL_IDS[model]
        first_role = next(iter(report["results"]))
        static = report["results"][first_role]["models"][model]
        common = {
            "evaluation_id": MODEL_IDS[model], "model": model,
            "protocol": report["protocol"], "numeric_path": report["numeric_path"],
            "source_index": report["source_index"],
            "source_weights_sha256": report["source_weights_sha256"],
            "summary": {key: static[key] for key in (
                "total_weight_parameters", "total_nonzero_weights", "total_weight_bytes",
                "total_scale_bytes", "total_tensor_payload_bytes", "persistent_neuron_state_bytes",
            )},
        }
        common_path = subdir / "common" / "model_profile.json"
        common_path.parent.mkdir(parents=True, exist_ok=True)
        common_path.write_text(json.dumps(common, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for role, result in report["results"].items():
            payload = {
                "evaluation_id": MODEL_IDS[model], "model": model, "data_split": role.lower(),
                "protocol": report["protocol"], "numeric_path": report["numeric_path"],
                "source_index": report["source_index"],
                "source_weights_sha256": report["source_weights_sha256"],
                "scope": report["scope"],
                "effective_macs_definition": report["effective_macs_definition"],
                "summary": result["models"][model], "per_sequence": result["per_sequence"][model],
            }
            output = subdir / role.lower() / "summary.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_complexity(COMPLEXITY_ROOT.parent)


def load_output() -> dict:
    payloads = {
        model: {
            role: json.loads((RESULT_ROOT / MODEL_IDS[model] / role.lower() / "summary.json").read_text(encoding="utf-8-sig"))
            for role in ("Validation", "Test")
        }
        for model in MODEL_IDS
    }
    h2 = payloads["H2"]["Validation"]
    roles = ("Validation", "Test")
    return {
        "protocol": h2["protocol"],
        "numeric_path": h2["numeric_path"],
        "source_index": h2["source_index"],
        "source_weights_sha256": h2["source_weights_sha256"],
        "scope": h2["scope"],
        "effective_macs_definition": h2["effective_macs_definition"],
        "results": {
            role: {
                "models": {model: payloads[model][role]["summary"] for model in payloads},
                "per_sequence": {model: payloads[model][role]["per_sequence"] for model in payloads},
            }
            for role in roles
        },
    }


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
        "effective_acs_per_sequence": 0.0,
        "effective_ops_per_sequence": (total("proposal_effective_macs") + total("refiner_effective_macs")) / len(rows),
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
    parser.add_argument("--tables-only", action="store_true", help="rebuild CSV tables from the saved JSON")
    args = parser.parse_args()
    if args.tables_only:
        build_complexity(COMPLEXITY_ROOT.parent)
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
        report = load_output()
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
        for model in ("H2", "H2-plus"):
            path = METRICS / "A_model_accuracy" / "02_model_records" / "formal" / MODEL_IDS[model] / role.lower() / "event_level" / "per_sequence.csv"
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
