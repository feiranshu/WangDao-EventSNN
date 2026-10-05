"""Step 1: H2 gate/refiner diagnosis on Train-dev without Test access."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from wangdao.h2_research import FrozenH2Pipeline, select_fair_working_point
from wangdao.phase1_common import (
    ROOT,
    PROTOCOL_ID,
    THRESHOLD_GRID,
    aggregate_scores,
    load_sequence,
    role_paths,
    score_predictions,
    sha256,
    write_json,
)
from wangdao.phase1_metrics import event_counts, event_summary
from wangdao.h2_refiner import diagnostic_strata, error_decomposition


def aggregate_errors(rows):
    keys = ("events", "target_events", "candidate_events", "proposal_false_negatives",
            "refiner_false_negatives", "tp", "fp", "fn", "tn")
    total = {key: sum(int(row[key]) for row in rows) for key in keys}
    kept = total["target_events"] - total["proposal_false_negatives"]
    total.update(
        candidate_fraction=total["candidate_events"] / total["events"] if total["events"] else None,
        proposal_recall=kept / total["target_events"] if total["target_events"] else None,
        refiner_conditional_recall=total["tp"] / kept if kept else None,
    )
    return total


def failure_windows(events, labels, indices, probability, top_k=5):
    assigned = np.floor((events["t"].astype(np.float64) - float(events["t"][0])) / 100.0)
    assigned = np.clip(assigned, 0, 79).astype(np.int64)
    retained = np.zeros(len(events), dtype=bool)
    retained[indices] = True
    target = labels.astype(bool)
    predicted = probability >= 0.28
    rows = []
    for window in np.unique(assigned):
        member = assigned == window
        rows.append({
            "window": int(window),
            "start_ms_relative": float(window * 100),
            "events": int(member.sum()),
            "targets": int(np.sum(target & member)),
            "proposal_fn": int(np.sum(target & ~retained & member)),
            "refiner_fn": int(np.sum(target & retained & ~predicted & member)),
            "fp": int(np.sum(~target & predicted & member)),
        })
    by_fn = sorted(rows, key=lambda row: (row["proposal_fn"] + row["refiner_fn"], row["fp"]), reverse=True)[:top_k]
    by_fp = sorted(rows, key=lambda row: (row["fp"], row["proposal_fn"] + row["refiner_fn"]), reverse=True)[:top_k]
    return {"top_false_negative_windows": by_fn, "top_false_positive_windows": by_fp}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    torch.set_num_threads(1)
    smoke = not args.run
    paths = role_paths("train_dev", limit=1 if smoke else None)
    weight_path = ROOT / "software/weights/H2_D1_seed42.npz"
    pipeline = FrozenH2Pipeline(weight_path, args.batch_size)
    prediction_dir = args.output.with_name(args.output.stem + "_predictions")
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=False)
    per_sequence = []
    curve_rows = {str(threshold): [] for threshold in THRESHOLD_GRID}
    ungated_rows = []
    for path in paths:
        load_started = time.perf_counter()
        events, loc, labels, identities = load_sequence(path)
        load_s = time.perf_counter() - load_started
        inputs = events[["x", "y", "t", "p"]].copy()
        prepared = pipeline.prepare(inputs, include_all=True)
        inference = pipeline.infer_prepared(prepared, len(events))
        probability = inference["probabilities"]
        ungated = pipeline.ungated_probability(prepared, len(events))
        scores = {}
        for threshold in THRESHOLD_GRID:
            score = score_predictions(loc, labels, identities, probability, threshold)
            curve_rows[str(threshold)].append(score)
            if threshold in (0.2800000011920929, 0.9):
                scores[str(threshold)] = score
        ungated_score = score_predictions(loc, labels, identities, ungated, 0.2800000011920929)
        ungated_rows.append(ungated_score)
        decomposition = error_decomposition(labels, prepared.indices, probability)
        masks = diagnostic_strata(events, labels, identities)
        masks["non_sparse_target"] = labels.astype(bool) & ~masks["sparse_target_le10_events_per_id_window"]
        strata = {name: error_decomposition(labels, prepared.indices, probability, mask=mask)
                  for name, mask in masks.items()}
        row = {
            "file": path.name,
            "events": len(events),
            "scores": scores,
            "ungated_all_event_diagnostic_at_0.28": ungated_score,
            "errors": decomposition,
            "strata": strata,
            "failure_windows": failure_windows(events, labels, prepared.indices, probability),
            "timings": inference["timings"],
            "file_load_s": load_s,
        }
        per_sequence.append(row)
        if args.save_predictions:
            with (prediction_dir / path.name).open("xb") as stream:
                np.savez_compressed(
                    stream,
                    probabilities=probability,
                    ungated_probabilities=ungated,
                    candidate_indices=prepared.indices,
                    event_index=np.arange(len(events), dtype=np.int64),
                    source_sha256=np.asarray(sha256(path)),
                )
        print(path.name, json.dumps(decomposition), flush=True)
    curve = {key: aggregate_scores(rows) for key, rows in curve_rows.items()}
    fixed = curve[str(0.2800000011920929)]
    fair = select_fair_working_point(
        curve, fixed["localization"]["pd"], fixed["localization"]["fa"]
    )
    errors = aggregate_errors([row["errors"] for row in per_sequence])
    strata_names = per_sequence[0]["strata"].keys()
    strata = {name: aggregate_errors([row["strata"][name] for row in per_sequence])
              for name in strata_names}
    total_fn = errors["fn"]
    backend_share = errors["refiner_false_negatives"] / total_fn if total_fn else None
    hypotheses = [
        {
            "id": "H1-backend-adaptation",
            "statement": "H2门控内后端漏检占总漏检至少70%，原结构Refiner适配应优先于降低门控。",
            "observed_backend_fn_share": backend_share,
            "supported_by_diagnosis": None if backend_share is None else backend_share >= 0.70,
            "falsification": "A1在相同候选和公平选点下两次受控尝试仍无稳定IoU增益。",
        },
        {
            "id": "H2-history-information",
            "statement": "稀疏目标的条件召回低于非稀疏目标，过去两窗信息可能补足弱证据。",
            "sparse_conditional_recall": strata["sparse_target_le10_events_per_id_window"]["refiner_conditional_recall"],
            "non_sparse_conditional_recall": strata["non_sparse_target"]["refiner_conditional_recall"],
            "falsification": "A3相对同维同参A3-null没有稳定增益，或高位移/边界退化抵消收益。",
        },
        {
            "id": "H3-gate-not-primary",
            "statement": "若门控漏检占总漏检不超过20%，不把扩大候选集作为首要方向。",
            "observed_gate_fn_share": None if not total_fn else errors["proposal_false_negatives"] / total_fn,
            "supported_by_diagnosis": None if not total_fn else errors["proposal_false_negatives"] / total_fn <= 0.20,
            "falsification": "真实全事件诊断显示大量候选外目标可被当前Refiner正确识别。",
        },
    ]
    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": args.output.stem,
        "stage": "step1_h2_diagnosis",
        "data_role": "train_dev",
        "primary_metrics": fixed,
        "decision": "diagnostic_only",
        "conclusion": "H2 error diagnosis only; no training, Test, Validation, GPU, or FPGA claim.",
        "smoke": smoke,
        "test_used": False,
        "validation_used": False,
        "metric_protocol": PROTOCOL_ID,
        "data_index_sha256": sha256(ROOT / "data/index.json"),
        "h2_weights_sha256": sha256(weight_path),
        "sequence_count": len(per_sequence),
        "errors": errors,
        "strata": strata,
        "threshold_curve": curve,
        "h2_fair_working_point": fair,
        "ungated_all_event_diagnostic_at_0.28": aggregate_scores(ungated_rows),
        "hypotheses": hypotheses,
        "per_sequence": per_sequence,
        "predictions_saved": args.save_predictions,
        "inference_timing_excludes": ["file_io", "labels", "metric_scoring", "ungated_diagnostic"],
        "power": {"status": "not_measured"},
    }
    write_json(args.output, report)
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
