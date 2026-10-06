"""S1-only stage timing and false-negative attribution on Train-dev."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wangdao.phase1_common import (ROOT, PROTOCOL_ID, role_paths, load_sequence, write_json,
                                  score_predictions, aggregate_scores, sha256)
from wangdao.refiner_adaptation import FrozenS1Pipeline, error_decomposition, diagnostic_strata
from wangdao.phase1_metrics import event_counts, event_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    p.add_argument("--cross-context", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        p.error("output already exists")
    torch.set_num_threads(1)
    paths = role_paths("train_dev", limit=None if a.run else 1)
    weights = ROOT / "software/weights"
    model = FrozenS1Pipeline(weights / "H2_D1_seed42.npz", weights / "S1P_INT8_ACC24_MEM24_Q12.npz")
    rows = []
    for path in paths:
        start = time.perf_counter()
        ev, loc, labels, identities = load_sequence(path)
        load_s = time.perf_counter() - start
        inputs = ev[["x", "y", "t", "p"]].copy()
        output = model.infer(inputs)
        probability, prepared = output["probabilities"], output["prepared"]
        score_start = time.perf_counter()
        scores = {str(t): score_predictions(loc, labels, identities, probability, t) for t in (.28, .9)}
        decomposition = error_decomposition(labels, prepared.indices, probability)
        strata = {name: error_decomposition(labels, prepared.indices, probability, mask=mask)
                  for name, mask in diagnostic_strata(ev, labels, identities).items()}
        row = {"file": path.name, "events": len(ev), "scores": scores, "errors": decomposition,
               "strata": strata, "timings": output["timings"], "file_load_s": load_s,
               "metric_s": time.perf_counter() - score_start}
        if a.cross_context:
            cross = model.cross_context(inputs, prepared)
            common = cross["common_indices"]
            row["cross_context"] = {
                "intersection_events": len(common), "h2_candidate_events": len(cross["h2_indices"]),
                "h2_context": event_summary(event_counts(labels[common], cross["h2_context_probability"], .28)) if len(common) else None,
                "s1_context": event_summary(event_counts(labels[common], cross["s1_context_probability"], .28)) if len(common) else None,
                "scope": "candidate intersection only; diagnostic, not full-event performance"}
        rows.append(row)
        print(path.name, json.dumps(decomposition), flush=True)
    report = {"schema_version": 1, "status": "complete", "experiment_id": a.output.stem,
              "stage": "phase1_s1_diagnosis", "data_role": "train_dev",
              "primary_metrics": {str(t): aggregate_scores([r["scores"][str(t)] for r in rows]) for t in (.28, .9)},
              "decision": "diagnostic_only", "conclusion": "Offline S1-only measurement; no causal/FPGA latency claim.",
              "smoke": not a.run, "test_used": False, "metric_protocol": PROTOCOL_ID,
              "data_index_sha256": sha256(ROOT / "data/index.json"), "per_sequence": rows,
              "sequence_count": len(rows), "power": {"status": "not_measured"},
              "inference_timing_excludes": ["file_io", "labels", "metric_scoring", "H2_cross_context"]}
    write_json(a.output, report)


if __name__ == "__main__":
    main()
