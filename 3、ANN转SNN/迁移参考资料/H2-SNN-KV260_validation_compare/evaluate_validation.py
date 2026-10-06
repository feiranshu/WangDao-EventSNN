"""Full Validation only; frozen models, no threshold search, no Test switch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from adapters import Models
from metrics import (PROTOCOL_ID, event_counts, event_summary, localization_counts,
                     localization_summary, sum_counts)
from prepare import ROOT, OFFICIAL_COMMIT, save_json, sha256
from reference_check import ReferenceEvaluator
from wangdao.data import validate_events
from wangdao.metrics import localization_metrics as legacy_localization

MODEL_NAMES = ("h2", "a1_h2", "s1_mixed")


def frozen_inputs():
    audit = json.loads((ROOT / "results/input_audit.json").read_text(encoding="utf-8"))
    for record in audit["baseline_manifest"]:
        if sha256(ROOT / "baseline" / record["path"]) != record["sha256"]:
            raise ValueError("Baseline snapshot changed: " + record["path"])
    for record in audit["official_reference_manifest"]:
        if sha256(ROOT / "official_reference" / record["path"]) != record["sha256"]:
            raise ValueError("Official reference changed: " + record["path"])
    if len(audit["per_sequence"]) != 24:
        raise ValueError("Must contain exactly 24 Validation files")
    for record in audit["per_sequence"]:
        name = record["file"]
        if Path(name).name != name or not name.startswith("val_"):
            raise ValueError("Validation only")
        path = ROOT / "data/EV-UAV-dataset/val" / name
        if sha256(path) != record["sha256"]:
            raise ValueError("Validation snapshot changed: " + name)
    return audit


def historical_results():
    h2 = json.loads((ROOT / "baseline/results/baseline_h2.json").read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / "baseline/results/s1_validation.json").read_text(encoding="utf-8"))
    return {"h2": h2["evaluation"], "a1_h2": fixed["evaluation"]["a1_h2"],
            "s1_mixed": fixed["evaluation"]["s1_mixed"]}


def run(limit, output):
    started = time.perf_counter()
    torch.set_num_threads(1)
    audit = frozen_inputs()
    models = Models()
    rows = {name: [] for name in MODEL_NAMES}
    references = {name: ReferenceEvaluator() for name in MODEL_NAMES}
    predictions_root = ROOT / "results/runs" / ("smoke_predictions" if limit else "validation_predictions")
    if predictions_root.exists():
        raise FileExistsError(f"Will not overwrite predictions: {predictions_root}")
    records = audit["per_sequence"][:limit] if limit else audit["per_sequence"]
    for number, record in enumerate(records, 1):
        path = ROOT / "data/EV-UAV-dataset/val" / record["file"]
        with np.load(path, allow_pickle=False) as z:
            ev, loc, norm = z["ev"].copy(), z["ev_loc"].copy(), z["evs_norm"].copy()
        validate_events(ev, require_labels=True)
        labels, identities = norm[:, 4], norm[:, 5]
        inference_start = time.perf_counter()
        message = []
        for name, probabilities, candidates in models.predict(ev):
            counts = event_counts(labels, probabilities, models.threshold)
            localization = localization_counts(loc, labels, identities, probabilities, models.threshold)
            legacy = legacy_localization(ev["t"], ev["x"], ev["y"], ev["label"], ev["name"],
                                         probabilities, models.threshold)
            references[name].add(record["file"], loc, labels, identities, probabilities, models.threshold)
            rows[name].append({"file": record["file"], "sha256": record["sha256"],
                               "events": len(ev), "event_counts": counts,
                               "localization_counts": localization, "legacy_localization": legacy,
                               "candidate_events": len(candidates),
                               "retained_targets": int(labels[candidates].sum()),
                               "target_events": int(labels.sum())})
            destination = predictions_root / name / record["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as f:
                np.savez_compressed(f, probabilities=probabilities,
                                    event_index=np.arange(len(ev), dtype=np.int64),
                                    source_sha256=np.array(record["sha256"]),
                                    candidate_indices=candidates,
                                    threshold=np.array(models.threshold))
            message.append(f"{name}: IoU={event_summary(counts)['iou']:.6f}")
        print(f"[{number:02d}/{len(records):02d}] {path.name} | {'; '.join(message)} | {time.perf_counter()-inference_start:.2f}s", flush=True)

    results = {}
    history = historical_results()
    for name in MODEL_NAMES:
        event = event_summary(sum_counts([r["event_counts"] for r in rows[name]]))
        local = localization_summary(sum_counts([r["localization_counts"] for r in rows[name]]))
        reference = references[name].summary(models.threshold)
        # The official implementation reports event ratios in float32.
        deltas = {"iou": event["iou"] - reference["iou"],
                  "official_acc": event["official_acc"] - reference["official_acc"],
                  "pd": local["pd"] - reference["pd"], "fa": local["fa"] - reference["fa"]}
        counts_equal = all(local[k] == reference[k] for k in
                           ("targets", "correct_targets", "false_components", "frame_denominator"))
        parity = counts_equal and max(abs(deltas[k]) for k in ("iou", "official_acc")) <= 1e-7 and abs(deltas["pd"]) <= 1e-7 and abs(deltas["fa"]) <= 1e-12
        if not parity:
            raise AssertionError(f"Official metric parity failed: {name}: {deltas}")
        legacy_rows = [r["legacy_localization"] for r in rows[name]]
        legacy_counts = {k: sum(r[k] for r in legacy_rows) for k in
                         ("targets", "correct_targets", "false_components", "frame_denominator")}
        legacy_summary = localization_summary(legacy_counts)
        regression = None
        if not limit:
            historical = history[name]
            mapped = {"tp": "true_positive", "fp": "false_positive", "fn": "false_negative", "tn": "true_negative"}
            regression = {
                "event_confusion_counts_exact": all(event[k] == historical["event"][v] for k, v in mapped.items()),
                "legacy_pd_match": abs(legacy_summary["pd"] - historical["pd"]) <= 1e-12,
                "legacy_fa_match": abs(legacy_summary["fa"] - historical["fa"]) <= 1e-12,
                "candidate_count_exact": sum(r["candidate_events"] for r in rows[name]) == historical["candidate_events"],
            }
            if not all(regression.values()):
                raise AssertionError(f"Frozen model historical regression failed: {name}: {regression}")
        results[name] = {"status": "evaluated", "event_threshold": models.threshold,
                         "event": event, "localization": local,
                         "legacy_localization": legacy_summary,
                         "official_minus_legacy": {k: local[k] - legacy_summary[k] for k in ("pd", "fa")},
                         "reference_check": {"passed": parity, "integer_counts_exact": counts_equal,
                                             "deltas": deltas, "reference": reference},
                         "historical_regression": regression,
                         "proposal_recall": sum(r["retained_targets"] for r in rows[name]) / sum(r["target_events"] for r in rows[name]),
                         "candidate_fraction": sum(r["candidate_events"] for r in rows[name]) / sum(r["events"] for r in rows[name]),
                         "per_sequence": rows[name]}
    results["official_ev_spsegnet"] = {"status": "not_run", "metrics": None,
                                      "reason": "Official checkpoint absent and CUDA/spconv/HAIS_OP inference environment unavailable; paper Test score is not a Validation measurement."}
    result = {
        "schema_version": 1, "status": "partial", "experiment_id": "validation-metric-alignment-smoke-v1" if limit else "validation-metric-alignment-v1",
        "stage": "evaluation_alignment", "data_role": "validation", "test_used": False,
        "primary_metrics": {name: {"event_iou": r["event"]["iou"], "official_acc": r["event"]["official_acc"],
                                   "pd": r["localization"]["pd"], "fa": r["localization"]["fa"]}
                            for name, r in results.items() if r["status"] == "evaluated"},
        "decision": "local_models_aligned_official_inference_pending",
        "conclusion": "Unified metrics and adapted-reference parity verified for the evaluated local models; no EV-SpSegNet Validation comparison is available.",
        "scope": {"sequences": len(records), "events": sum(r["events"] for r in records),
                  "full_validation": not bool(limit), "training": False, "threshold_search": False,
                  "network_modified": False, "label_fields_hidden_from_inference": True},
        "protocol": {"id": PROTOCOL_ID, "official_commit": OFFICIAL_COMMIT,
                     "aggregation": "sum event/target/component counts before division; no macro averaging",
                     "event_support": "all original events, including proposal-rejected events with probability 0",
                     "localization_timestamp": "ev_loc -> int64 -> float32, same as official collate/test",
                     "pd_window_ms": 50, "pd_correct_threshold": 0.0001,
                     "window_bounds": "strict > and <, without subtracting t_min",
                     "fa_mask": "uint8 increments modulo 256, 8-connectivity, once per frame",
                     "threshold_policy": "each model's pre-frozen threshold; no Validation fitting",
                     "official_default_event_threshold": 0.9,
                     "reference_adaptations": ["metric Tensor.cuda() replaced with Tensor.cpu()",
                                                "OpenCV component count uses independent union-find if cv2 is absent"],
                     "reference_not_official_network_inference": True},
        "dataset_index_sha256": audit["index_sha256"], "models": results,
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                    "platform": platform.platform(), "torch_threads": 1,
                    "total_wall_s": time.perf_counter() - started},
    }
    save_json(output, result)
    print(json.dumps(result["primary_metrics"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Only first Validation sequence; not a full evaluation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "results" / ("runs/smoke.json" if args.smoke else "validation_alignment.json")
    output = output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Output must be a new file inside this isolated comparison directory")
    run(1 if args.smoke else None, output)
