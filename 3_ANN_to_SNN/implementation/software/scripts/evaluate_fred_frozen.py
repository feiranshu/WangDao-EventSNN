"""Independent, read-only model evaluation of the two adapted FRED recordings.

Imports the existing ANN and SNN implementations without patching them. All
outputs go to a separate run directory. No fitting or training entrypoint exists.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
ANN = ROOT.parents[1] / "1_H2_model_improvement/implementation"
sys.path.insert(0, str(ANN / "software"))
import wangdao
wangdao.__path__.append(str(ROOT / "software/wangdao"))
from wangdao.h2_pro.stream_state import CausalPipeline, EVENT_DTYPE
from wangdao.h2.numpy_model import quantized_mlp_with_activity, _sigmoid
from wangdao.h2.representation import _fine_counts_to_patch_features, patch_scores_to_heatmaps
from wangdao.h2pro_direct_v1.core import Cost, forward, probabilities, load_layers, validate_layers
from wangdao.h2pro_calibration_v1.core import Encoder, forward as calibrated_forward
from wangdao.h2pro_task_calibration_v1.workflow import validate_selection
from wangdao.h2pro_snn.foundation import verify_baseline
from wangdao.fred_adapter_v1.download import atomic_json, sha256

MODELS = ("h2pro_ann", "r1_t16_task_rescore", "r2_t8_epoch088")
COUNT_KEYS = ("events", "target_events", "candidate_events", "candidate_targets",
              "proposal_fn", "refiner_fn", "tp", "fp", "fn", "tn")
DATASET = ROOT / "results/runs/fred_external_v1/dataset"
OUTPUT = ROOT / "results/runs/fred_frozen_three_v1"
CALIBRATION = ROOT / "results/runs/h2pro_task_calibration_v1_fit/calibration.json"
STUDENT = ROOT / "results/runs/h2pro_distill_r2t8_v1_s42/best_int8.npz"
PROTOCOL = {
    "version": "fred_frozen_three_v1", "threshold": 0.35,
    "models": list(MODELS), "window_ms": 100, "sequence_origin": "first source event",
    "state": "continuous across all storage shards; reset only per original recording",
    "tail": "feed every event; original pipeline does not publish an incomplete final window",
    "labels": "source-box weak labels; eval_mask applied only after inference",
    "segments": "all; raw source timestamp <5000ms; raw source timestamp >=5000ms",
    "backend": "unchanged NumPy exact-int8 ANN and dense SNN count_reference; explicit subset checks",
    "cost": "all closed-window inputs including label-masked events; neural synapses only",
    "cost_not_measured": ["feature extraction", "routing", "memory traffic", "energy", "hardware latency"],
    "training_started": False, "parameter_updates": 0, "threshold_selection": False,
    "prediction_bits": {"0": "candidate", "1": MODELS[0], "2": MODELS[1],
                        "3": MODELS[2], "4": "eval_mask", "5": "weak_label",
                        "6": "closed_window", "7": "raw_timestamp_below_5000ms"},
}


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def zero_counts():
    return dict.fromkeys(COUNT_KEYS, 0)


def add_counts(target, source):
    for key in COUNT_KEYS:
        target[key] += int(source[key])


def counts(label, predicted, candidate, valid):
    label, predicted, candidate, valid = map(np.asarray, (label, predicted, candidate, valid))
    if not (label.ndim == 1 and label.shape == predicted.shape == candidate.shape == valid.shape):
        raise ValueError("One aligned vector per event required")
    if any(a.dtype != np.bool_ for a in (label, predicted, candidate, valid)):
        raise ValueError("Boolean metric vectors required")
    if np.any(predicted & ~candidate):
        raise ValueError("A rejected event cannot be predicted positive")
    y, p, c = label[valid], predicted[valid], candidate[valid]
    return {"events": len(y), "target_events": int(y.sum()),
            "candidate_events": int(c.sum()), "candidate_targets": int((c & y).sum()),
            "proposal_fn": int((~c & y).sum()), "refiner_fn": int((c & y & ~p).sum()),
            "tp": int((y & p).sum()), "fp": int((~y & p).sum()),
            "fn": int((y & ~p).sum()), "tn": int((~y & ~p).sum())}


def summarize(c):
    def ratio(n, d):
        return n / d if d else None
    if c["fn"] != c["proposal_fn"] + c["refiner_fn"]:
        raise AssertionError("FN attribution mismatch")
    return {**c, "iou": ratio(c["tp"], c["tp"] + c["fp"] + c["fn"]),
            "precision": ratio(c["tp"], c["tp"] + c["fp"]),
            "recall": ratio(c["tp"], c["target_events"]),
            "proposal_recall": ratio(c["candidate_targets"], c["target_events"]),
            "refiner_conditional_recall": ratio(c["tp"], c["candidate_targets"]),
            "f1": ratio(2*c["tp"], 2*c["tp"]+c["fp"]+c["fn"])}


def new_totals():
    return {m: {s: zero_counts() for s in ("all", "opening_5s", "after_5s")} for m in MODELS}


def score_bits(bits, totals):
    valid = ((bits & 16) != 0) & ((bits & 64) != 0)
    label, candidate, opening = (bits & 32) != 0, (bits & 1) != 0, (bits & 128) != 0
    masks = {"all": valid, "opening_5s": valid & opening, "after_5s": valid & ~opening}
    row = {}
    for i, model in enumerate(MODELS):
        predicted = (bits & (2 << i)) != 0
        row[model] = {}
        for segment, mask in masks.items():
            c = counts(label, predicted, candidate, mask)
            add_counts(totals[model][segment], c)
            row[model][segment] = c
    return row


class FrozenModels:
    def __init__(self):
        self.baseline = verify_baseline()
        self.ann = load_layers()
        calibration = read_json(CALIBRATION)
        validate_selection(calibration)
        point, = [p for p in calibration["points"] if p["stage"] == "r1" and p["steps"] == 16]
        self.encoders = tuple(Encoder(**e) for e in point["task_rescore"]["encoders"])
        with np.load(STUDENT, allow_pickle=False) as z:
            if str(z["stage"]) != "r2" or int(z["steps"]) != 8 or float(z["threshold"]) != .35:
                raise ValueError("Expected the frozen r2/T8 export at 0.35")
            self.student = tuple((z[f"layer{i}_weight"].copy(), z[f"layer{i}_scale"].copy()) for i in range(3))
        validate_layers(self.student)
        if sha256(STUDENT) != sha256(STUDENT.parent / "epochs/epoch_088_int8.npz"):
            raise ValueError("Best student is no longer epoch 88")
        for pair in self.student:
            for array in pair:
                array.setflags(write=False)

    def snn(self, x, model, backend="count_reference"):
        if model == MODELS[1]:
            return calibrated_forward(x, self.ann, "r1", 16, self.encoders, backend)
        if model == MODELS[2]:
            return forward(x, self.student, "r2", 8, backend)
        raise ValueError(model)

    def pipeline(self):
        return CausalPipeline(ANN / "software/weights/H2_D1_seed42.npz",
                              ANN / "software/weights/H2-pro-Causal-ANN-v1_int8.npz",
                              variant="C1-proposal-motion", threshold=.35, initial_refiner=False)

    def identity(self, dataset):
        return {"protocol": PROTOCOL, "dataset_index_sha256": sha256(dataset / "index.json"),
                "baseline": self.baseline, "student_sha256": sha256(STUDENT),
                "calibration_sha256": sha256(CALIBRATION),
                "r1_encoders": [asdict(e) for e in self.encoders],
                "evaluator_sha256": sha256(Path(__file__))}


def proposal_cost(pipe, window):
    """Observe unchanged inputs separately; do not replace/patch Proposal execution."""
    current = pipe.previous_events
    if not len(current):
        return {"dense_mac": 0, "zero_skip_equivalent_mac": 0, "output_scalars": 0}
    fine = np.zeros((1, 2, 68, 88), np.int32)
    np.add.at(fine[0], (current["p"], current["y"]//4, current["x"]//4), 1)
    patch = _fine_counts_to_patch_features(fine).reshape(-1, 128)
    activity = quantized_mlp_with_activity(patch, pipe.proposal.proposal_layers)
    heatmap = patch_scores_to_heatmaps(_sigmoid(activity.logits).reshape(1, 336, 16))[0]
    np.testing.assert_array_equal(heatmap, window.heatmap)
    fanouts = [w.shape[1] for w, _ in pipe.proposal.proposal_layers]
    return {"dense_mac": len(patch)*sum(w.size for w, _ in pipe.proposal.proposal_layers),
            "zero_skip_equivalent_mac": sum(n*f for n, f in zip(activity.activation_nonzero, fanouts)),
            "output_scalars": len(patch)*sum(fanouts)}


def cost_report(refiner, proposal, model, candidate_events):
    n = candidate_events
    state_updates = 0 if model == MODELS[0] else n*(64*16 if model == MODELS[1] else 96*8)
    return {
        "neural_mac_dense_proposal": proposal["dense_mac"] + refiner["rate_network_mac"],
        "neural_mac_zero_skip_proposal": proposal["zero_skip_equivalent_mac"] + refiner["rate_network_mac"],
        "neural_ac": refiner["rate_network_ac"],
        "proposal_dense_mac": proposal["dense_mac"],
        "proposal_zero_skip_equivalent_mac": proposal["zero_skip_equivalent_mac"],
        "proposal_ac": 0, "refiner_mac": refiner["rate_network_mac"],
        "refiner_ac": refiner["rate_network_ac"],
        "executed_reference_neural_mac": proposal["dense_mac"] + refiner["executed_synaptic_mac"],
        "executed_reference_neural_ac": refiner["executed_synaptic_ac"],
        "extra_proposal_activity_audit_mac": proposal["dense_mac"],
        "equivalent_if_state_adds": state_updates, "equivalent_if_comparisons": state_updates,
        "equivalent_if_reset_subtractions": refiner["unit_spikes"],
        "refiner_detail": refiner, "proposal_decode_scale_mul": proposal["output_scalars"],
        "proposal_decode_divisions": proposal["output_scalars"],
        "inference_only_excludes_audits": True, "is_hardware_measurement": False,
    }


def preservation(output):
    original = read_json(output / "preservation_before.json")
    changed = [name for name, row in original.items()
               if not Path(name).is_file() or sha256(Path(name)) != row["sha256"]]
    report = {"checked_files": len(original), "changed_files": changed, "unchanged": not changed}
    atomic_json(output / "preservation_after.json", report)
    if changed:
        raise AssertionError("Existing protected files changed: " + repr(changed))
    return report


def run_sequence(models, dataset, output, seq, identity):
    name = seq["source_sequence"]
    directory = output / name
    directory.mkdir(exist_ok=True)
    final = directory / "result.json"
    if final.exists():
        result = read_json(final)
        if (result["identity"] != identity or
                sha256(directory / "predictions.npy") != result["predictions_sha256"] or
                sha256(directory / "windows.jsonl") != result["windows_sha256"]):
            raise ValueError("Completed sequence identity or predictions changed")
        print(json.dumps({"stage": "reuse_complete_sequence", "sequence": name}), flush=True)
        return result
    start = time.perf_counter()
    size = seq["output_events"]
    for filename in ("predictions.partial.npy", "windows.partial.jsonl"):
        prior = directory / filename
        if prior.exists():
            prior.rename(prior.with_name(prior.name + f".interrupted_{time.time_ns()}"))
    bits = np.lib.format.open_memmap(directory / "predictions.partial.npy", mode="w+", dtype=np.uint8, shape=(size,))
    bits[:] = 0
    pipe = models.pipeline()
    t0 = seq["source_first_timestamp_us"] / 1000.
    end = np.nextafter(seq["source_last_timestamp_us"] / 1000., np.inf)
    pipe.reset(t0)
    totals, refiner_totals = new_totals(), {m: {} for m in MODELS}
    proposal = dict(dense_mac=0, zero_skip_equivalent_mac=0, output_scalars=0)
    fed = covered = windows = candidate_events = 0
    checked = {m: 0 for m in MODELS}
    source_paths = []
    with (directory / "windows.partial.jsonl").open("w", encoding="utf-8") as log:
        for shard in seq["shards"]:
            path = dataset / shard["path"]
            if sha256(path) != shard["sha256"]:
                raise ValueError("Adapted source shard hash changed")
            with np.load(path, allow_pickle=False) as z:
                events, valid = z["ev"], z["eval_mask"]
            if len(events) != shard["events"] or valid.dtype != np.bool_ or valid.shape != (len(events),):
                raise ValueError("Dataset field alignment mismatch")
            if np.any(np.diff(events["t"]) < 0) or not np.isin(events["label"], [0, 1]).all():
                raise ValueError("Invalid sorted/binary source contract")
            source_paths.append(shard["path"])
            stop = min(shard["shard_end_ms"], end)
            lo = 0
            while pipe.watermark < stop:
                right = min(t0 + (pipe.next_window + 1)*100., stop)
                hi = int(np.searchsorted(events["t"], right, side="left"))
                block = events[lo:hi]
                length = len(block)
                bits[fed:fed+length] = (block["label"].astype(np.uint8)*32 +
                    valid[lo:hi].astype(np.uint8)*16 + (block["t"] < 5000.).astype(np.uint8)*128)
                fed += length
                packet = np.empty(length, dtype=EVENT_DTYPE)
                for field in ("x", "y", "t", "p"):
                    packet[field] = block[field]
                # Ground truth and validity mask are never passed to any model.
                pipe.push(packet, right)
                ready = pipe.flush_completed_windows()
                if len(ready) > 1:
                    raise AssertionError("One-window packet contract failed")
                for window in ready:
                    n, nc = len(window.event_indices), len(window.features)
                    np.testing.assert_array_equal(window.event_indices, np.arange(covered, covered+n))
                    if window.features.dtype != np.uint8 or window.features.shape != (int(window.candidate_mask.sum()), 36):
                        raise AssertionError("Original feature interface changed")
                    code = bits[covered:covered+n].copy()
                    code |= 64
                    code |= window.candidate_mask.astype(np.uint8)
                    code |= window.predictions.astype(np.uint8)*2
                    if not np.isfinite(window.probabilities).all():
                        raise FloatingPointError("Nonfinite ANN output")
                    candidate_ids = np.flatnonzero(window.candidate_mask)
                    pc = proposal_cost(pipe, window)
                    for key, value in pc.items():
                        proposal[key] += value
                    costs = {MODELS[0]: asdict(Cost(executed_synaptic_mac=nc*4480,
                        rate_network_mac=nc*4480, scalar_scale_mul=nc*100,
                        scalar_normalization_div=nc*100, quantizer_scale_mul=nc*96,
                        quantizer_round=nc*96, quantizer_comparisons=nc*96*4))}
                    for mi, model in enumerate(MODELS[1:], start=1):
                        cost_sum = {}
                        for begin in range(0, nc, 32768):
                            x = window.features[begin:begin+32768]
                            raw, cost = models.snn(x, model)
                            p = probabilities(raw)
                            if not np.isfinite(p).all():
                                raise FloatingPointError("Nonfinite SNN output")
                            code[candidate_ids[begin:begin+len(x)]] |= (p >= .35).astype(np.uint8)*(2 << mi)
                            for key, value in asdict(cost).items():
                                cost_sum[key] = cost_sum.get(key, 0) + value
                        if not cost_sum:
                            _, empty_cost = models.snn(np.empty((0,36), np.uint8), model)
                            cost_sum = asdict(empty_cost)
                        costs[model] = cost_sum
                        if nc and (window.window % 50 == 0 or checked[model] == 0):
                            ids = np.unique(np.linspace(0, nc-1, min(32, nc), dtype=np.int64))
                            probe = window.features[ids]
                            dense, dc = models.snn(probe, model)
                            explicit, ec = models.snn(probe, model, "spike_add")
                            np.testing.assert_array_equal(dense, explicit)
                            if dc.rate_network_ac != ec.executed_synaptic_ac or dc.unit_spikes != ec.unit_spikes:
                                raise AssertionError("Explicit spike accounting mismatch")
                            checked[model] += len(ids)
                    for model, cost in costs.items():
                        for key, value in cost.items():
                            refiner_totals[model][key] = refiner_totals[model].get(key, 0) + value
                    bits[covered:covered+n] = code
                    metrics = score_bits(code, totals)
                    log.write(json.dumps({"window": window.window, "right_boundary_ms": window.right_boundary_ms,
                        "first_event_index": covered, "events": n, "candidate_events": nc,
                        "counts": metrics, "proposal": pc, "refiner_cost": costs}) + "\n")
                    covered += n
                    candidate_events += nc
                    windows += 1
                    if windows % 25 == 0 or windows == 1:
                        log.flush()
                        progress = {"status": "evaluating", "sequence": name, "closed_windows": windows,
                            "closed_events": covered, "input_events": fed, "sequence_events": size,
                            "elapsed_s": time.perf_counter()-start, "training_started": False}
                        atomic_json(output / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                lo = hi
            if lo != len(events):
                raise AssertionError("Source events were not all fed")
            del events, valid
        remaining, tail = pipe.finalize(end)
        if remaining or fed != size or covered+tail != size:
            raise AssertionError("Full recording coverage mismatch")
    trailing_bits = bits[covered:].copy()
    tail_report = {"events": tail, "valid_events": int(np.count_nonzero(trailing_bits & 16)),
                   "weak_target_events": int(np.count_nonzero(trailing_bits & 32)),
                   "duration_ms": end-(t0+windows*100.)}
    bits.flush()
    del bits
    os.replace(directory / "predictions.partial.npy", directory / "predictions.npy")
    os.replace(directory / "windows.partial.jsonl", directory / "windows.jsonl")
    result = {"status": "complete", "identity": identity, "sequence": name,
        "source_shards": source_paths, "input_events": fed, "closed_events": covered,
        "label_masked_closed_events": covered-totals[MODELS[0]]["all"]["events"],
        "closed_windows": windows, "candidate_events": candidate_events, "trailing": tail_report,
        "t0_ms": t0, "last_source_ms": end, "closed_duration_s": windows/10.,
        "explicit_checked_candidates": checked, "seconds": time.perf_counter()-start,
        "models": {m: {"metrics": {s: summarize(c) for s,c in totals[m].items()},
            "cost": cost_report(refiner_totals[m], proposal, m, candidate_events)} for m in MODELS},
        "predictions_sha256": sha256(directory / "predictions.npy"),
        "windows_sha256": sha256(directory / "windows.jsonl"), "training_started": False}
    atomic_json(final, result)
    print(json.dumps({"stage": "sequence_complete", "sequence": name, "seconds": result["seconds"],
                     "iou": {m: result["models"][m]["metrics"]["all"]["iou"] for m in MODELS}}), flush=True)
    return result


def aggregate(rows, identity):
    models = {}
    for model in MODELS:
        metrics = {}
        for segment in ("all", "opening_5s", "after_5s"):
            c = zero_counts()
            for row in rows:
                add_counts(c, row["models"][model]["metrics"][segment])
            metrics[segment] = summarize(c)
        costs = [r["models"][model]["cost"] for r in rows]
        cost = {k: sum(c[k] for c in costs) for k,v in costs[0].items() if type(v) is int}
        cost["refiner_detail"] = {k: sum(c["refiner_detail"][k] for c in costs) for k in costs[0]["refiner_detail"]}
        n = sum(r["closed_events"] for r in rows)
        seconds = sum(r["closed_duration_s"] for r in rows)
        cost["dense_mac_per_closed_event"] = cost["neural_mac_dense_proposal"]/n
        cost["zero_skip_mac_per_closed_event"] = cost["neural_mac_zero_skip_proposal"]/n
        cost["ac_per_closed_event"] = cost["neural_ac"]/n
        cost["dense_mac_per_source_second"] = cost["neural_mac_dense_proposal"]/seconds
        cost["ac_per_source_second"] = cost["neural_ac"]/seconds
        models[model] = {"metrics": metrics, "cost": cost}
    return {"status": "complete", "identity": identity, "protocol": PROTOCOL,
            "input_events": sum(r["input_events"] for r in rows),
            "closed_events": sum(r["closed_events"] for r in rows),
            "trailing_events": sum(r["trailing"]["events"] for r in rows),
            "label_masked_closed_events": sum(r["label_masked_closed_events"] for r in rows),
            "closed_duration_s": sum(r["closed_duration_s"] for r in rows),
            "models": models, "sequences": rows, "training_started": False,
            "limitations": ["two official training recordings only, not the FRED test benchmark",
                            "bounding-box weak labels, not pixel-exact event truth",
                            "whole recording read; incomplete last windows unpublished by original model",
                            "MAC/AC are neural arithmetic, not measured hardware energy"]}


def independent_audit(output, report):
    """Recount saved decisions with bincount, independent of metric accumulation."""
    audits = []
    for row in report["sequences"]:
        bits = np.load(output / row["sequence"] / "predictions.npy", mmap_mode="r", allow_pickle=False)
        totals = new_totals()
        closed = candidates = 0
        for lo in range(0, len(bits), 1_000_000):
            b = bits[lo:lo+1_000_000]
            is_closed, valid = (b & 64) != 0, (b & 16) != 0
            y, c, opening = (b & 32) != 0, (b & 1) != 0, (b & 128) != 0
            closed += int(is_closed.sum())
            candidates += int(c.sum())
            for mi, model in enumerate(MODELS):
                p = (b & (2 << mi)) != 0
                if np.any(p & ~c):
                    raise AssertionError("Saved predictions violate Proposal gating")
                for segment, group in (("all", np.ones(len(b),bool)), ("opening_5s", opening), ("after_5s", ~opening)):
                    take = is_closed & valid & group
                    hist = np.bincount(2*y[take].astype(np.int8)+p[take], minlength=4)
                    tn, fp, fn, tp = map(int, hist)
                    prop_fn = int(np.count_nonzero(take & y & ~c))
                    add_counts(totals[model][segment], dict(events=int(take.sum()), target_events=tp+fn,
                        candidate_events=int(np.count_nonzero(take & c)), candidate_targets=tp+fn-prop_fn,
                        proposal_fn=prop_fn, refiner_fn=fn-prop_fn, tp=tp,fp=fp,fn=fn,tn=tn))
        if closed != row["closed_events"] or candidates != row["candidate_events"]:
            raise AssertionError("Saved coverage mismatch")
        for model in MODELS:
            for segment in totals[model]:
                if summarize(totals[model][segment]) != row["models"][model]["metrics"][segment]:
                    raise AssertionError("Independent confusion recount differs")
            c = row["models"][model]["cost"]
            dense_ref = candidates*4480
            expected_ref = candidates*({MODELS[0]:4480, MODELS[1]:2432, MODELS[2]:2304}[model])
            if c["refiner_mac"] != expected_ref or c["executed_reference_neural_mac"] != c["proposal_dense_mac"]+dense_ref:
                raise AssertionError("Independent architecture MAC count differs")
        audits.append({"sequence": row["sequence"], "events_checked": len(bits),
                       "metrics_exact": True, "mac_formula_exact": True})
    result = {"status": "passed", "method": "bincount of saved full-event decision bitfields", "sequences": audits}
    atomic_json(output / "independent_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    models = FrozenModels()
    identity = models.identity(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Different frozen evaluation identity; use a new output directory")
    atomic_json(identity_path, identity)
    if args.audit_only:
        report = read_json(args.output / "comparison.json")
        if report["identity"] != identity:
            raise ValueError("Audit identity mismatch")
    else:
        index = read_json(args.dataset / "index.json")
        if index["original_recordings"] != ["train_000", "train_001"] or not index["requires_eval_mask"]:
            raise ValueError("Expected the two declared, masked FRED pilot recordings")
        rows = [run_sequence(models, args.dataset, args.output, seq, identity) for seq in index["sequences"]]
        report = aggregate(rows, identity)
        atomic_json(args.output / "comparison.json", report)
    independent_audit(args.output, report)
    preservation(args.output)
    atomic_json(args.output / "progress.json", {"status":"complete", "models": list(MODELS),
        "input_events":report["input_events"], "training_started":False})
    print(json.dumps({"status":"complete", "report":str(args.output / "comparison.json")}), flush=True)


if __name__ == "__main__":
    main()
