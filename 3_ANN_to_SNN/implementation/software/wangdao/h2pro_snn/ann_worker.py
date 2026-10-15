"""Isolated frozen-ANN diagnostic worker. Invoked by the development CLI only."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import time

import numpy as np

from .costs import OperationLedger, profile_exact_mlp
from .diagnostics import aggregate_counts, error_counts, feature_collisions, summarize_counts, temporal_support
from .foundation import ANN_ROOT, DEVELOPMENT_ROLES, research_sources, role_records, verified_sequence, verify_baseline


def diagnose(dataset_root: Path, role: str, limit=None, max_windows=None) -> dict:
    baseline = verify_baseline()
    # The launcher places the actual ANN package first and appends only the new
    # research namespace. Refuse accidental use of the historical SNN H2 copy.
    stream_state = import_module("wangdao.h2_pro.stream_state")
    from wangdao.h2 import representation
    from wangdao.data import load_events
    expected = ANN_ROOT / "software/wangdao"
    if (stream_state.__file__ is None or Path(stream_state.__file__).absolute().parent.parent != expected or
            Path(representation.__file__).absolute().parent.parent != expected):
        raise RuntimeError("Frozen ANN must be imported from its own implementation directory")
    if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
        raise ValueError("max_windows must be a positive integer")
    records = role_records(role, limit)
    rows, raw_counts = [], []
    original = stream_state.quantized_mlp
    stage_costs = {"proposal": OperationLedger(), "refiner": OperationLedger()}
    calls = {"proposal": 0, "refiner": 0}

    def measured(features, layers):
        output = original(features, layers)
        replay, counts = profile_exact_mlp(features, layers)
        if not np.array_equal(output, replay):
            raise AssertionError("Diagnostic replay disagrees with exact frozen ANN")
        stage = "proposal" if features.shape[1] == 128 else "refiner"
        stage_costs[stage].merge(counts)
        calls[stage] += 1
        return output

    setattr(stream_state, "quantized_mlp", measured)
    started = time.perf_counter()
    try:
        for record in records:
            path = verified_sequence(dataset_root, record)
            ev = load_events(path, require_labels=True)
            t0 = float(ev["t"][0])
            final_watermark = float(ev["t"][-1]) + 1e-6
            watermark = min(final_watermark, t0 + max_windows*100.0) if max_windows else final_watermark
            input_count = int(np.searchsorted(ev["t"], watermark, side="left"))
            packet = np.empty(input_count, dtype=stream_state.EVENT_DTYPE)
            for field in ("x", "y", "t", "p"):
                packet[field] = ev[field][:input_count]
            pipeline = stream_state.CausalPipeline(
                ANN_ROOT / "software/weights/H2_D1_seed42.npz",
                ANN_ROOT / "software/weights/H2-pro-Causal-ANN-v1_int8.npz",
                variant="C1-proposal-motion", threshold=0.35, initial_refiner=False)
            pipeline.reset(t0)
            pipeline.push(packet, watermark)
            windows, trailing = pipeline.finalize()
            window_rows, features, candidate_labels = [], [], []
            for window in windows:
                labels = ev["label"][window.event_indices]
                counts = error_counts(labels, window.probabilities, window.candidate_mask)
                raw_counts.append(counts)
                window_rows.append({"window": window.window, "end_ms": window.right_boundary_ms,
                                    **summarize_counts(counts)})
                features.append(window.features)
                candidate_labels.append(labels[window.candidate_mask])
            published = sum(w["events"] for w in window_rows)
            if published + trailing != input_count:
                raise AssertionError("Incomplete event accounting")
            elapsed = float(ev["t"][-1])-t0
            seq_rate = len(ev)*1000.0/elapsed if elapsed > 0 else None
            rows.append({"sequence": record["path"], "sha256": record["sha256"],
                         "input_events": input_count, "published_events": published,
                         "trailing_unpublished_events": trailing,
                         "beyond_requested_prefix_events": len(ev)-input_count,
                         "closed_windows": len(windows),
                         "observed_sequence_events_per_second": seq_rate,
                         "high_density": seq_rate >= 20000 if seq_rate is not None else None,
                         "metrics": aggregate_counts(window_rows), "windows": window_rows,
                         "feature_collisions": feature_collisions(
                             np.concatenate(features) if features else np.empty((0, 36), dtype=np.uint8),
                             np.concatenate(candidate_labels) if candidate_labels else np.array([], dtype=int)),
                         "temporal_support": temporal_support(ev[:published], ev["label"][:published],
                                                               t0, t0+100.0*len(windows))})
    finally:
        setattr(stream_state, "quantized_mlp", original)
    cost = {stage: ledger.report("frozen ANN neural layers only", ["feature construction",
                "Proposal heatmap and dilation", "C1 motion and history", "candidate routing",
                "connected components", "probability readout", "state and workspace traffic"])
            for stage, ledger in stage_costs.items()}
    return {"schema_version": 1, "experiment_id": "h2pro_snn_development_diagnosis",
            "status": "partial_prefix_diagnostic" if max_windows else "development_diagnostic",
            "stage": "failure_diagnosis", "data_role": role, "training_started": False,
            "reserved_sequences_read": 0, "baseline": baseline, "sequence_count": len(rows),
            "research_sources_utf8_lf": research_sources(),
            "limits": {"sequence_limit": limit, "max_windows": max_windows},
            "primary_metrics": aggregate_counts(raw_counts),
            "density_groups": {name: aggregate_counts([r["metrics"] for r in rows if r["high_density"] == flag])
                               for name, flag in (("high_density", True), ("ordinary", False))},
            "density_definition": "all sequence events / (last timestamp - first timestamp)",
            "neural_cost": cost, "profiled_calls_bit_exact": calls,
            "diagnostic_wall_seconds_including_replay": time.perf_counter()-started,
            "decision": "no_SNN_selection_no_accuracy_or_MAC_gain_claim", "sequences": rows,
            "conclusion": "Frozen ANN diagnostic only; full-chain cost and learned SNN remain unmeasured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--role", choices=DEVELOPMENT_ROLES, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.dataset_root, args.role, args.limit, args.max_windows), allow_nan=False))


if __name__ == "__main__":
    main()
