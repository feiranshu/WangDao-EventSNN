"""S3-A read-only diagnosis of frozen S2 epoch-5 research checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).absolute().parent
S2 = HERE.parent / "s2"
sys.path.insert(0, str(S2))
from core import ResearchANN, ResearchSNN, SNNRunner, THRESHOLD, class_logits  # noqa: E402
from train import CACHE, average_precision, read_index, role_records, windows  # noqa: E402
sys.path.insert(0, str(HERE.parent / "s1"))
from analyze import motion_and_history  # noqa: E402

RUN = S2 / "runs" / "GEN2-S2-001-revision1"
MAIN_DIAGNOSTIC = ("train_000.npz", "train_044.npz", "train_094.npz")
FP_BUDGETS = (4137, 2463)
SUBSETS = ("first_appearance", "reappearance", "cross_cell")
INTERVENTIONS = ("window_reset_all", "zero_u1", "zero_u2", "zero_u3",
                 "zero_trace", "no_trace_readout", "no_mem_readout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(kind: str):
    checkpoint = torch.load(RUN / f"{kind}_epoch05.pt", map_location="cpu")
    provenance = checkpoint["provenance"]
    for key, name in (("core_sha256", "core.py"), ("train_sha256", "train.py"),
                      ("training_protocol_sha256", "PROTOCOL_at_revision1_start.md")):
        assert sha256(S2 / name) == provenance[key], f"S2 provenance mismatch: {name}"
    assert checkpoint["model_kind"] == kind and checkpoint["epoch"] == 5
    model = ResearchANN() if kind == "ann" else ResearchSNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def replay_ann(model, records):
    result = []
    with torch.no_grad():
        for record in records:
            features = record["features"]
            probability = np.empty(len(features), dtype=np.float32)
            for start in range(0, len(features), 4096):
                stop = min(start + 4096, len(features))
                x = torch.from_numpy(features[start:stop].astype(np.float32) / 255.0)
                probability[start:stop] = torch.softmax(class_logits(model(x)), 1)[:, 1].numpy()
            result.append(probability)
    return result


def replay_snn(model, records, mode="baseline", capture=False):
    result, captures = [], []
    with torch.no_grad():
        for record in records:
            runner = SNNRunner(model, record["t0"])
            n = len(record["features"])
            probability = np.empty(n, dtype=np.float32)
            detail = None
            if capture:
                detail = {key: np.empty((n, 4), dtype=np.float32)
                          for key in ("pre3", "post3", "spike3", "trace", "trace_term", "mem_term", "streak")}
                detail.update({key: np.empty((n, 3), dtype=np.float32)
                               for key in ("layer_spike", "layer_post_ge", "layer_sg_support")})
                detail["winner"] = np.empty(n, dtype=np.int8)
                detail["local_count"] = np.empty(n, dtype=np.int32)
                detail["input_sum"] = record["features"].astype(np.float32).sum(axis=1) / 255.0
                detail["samples"] = {str(i): {"pre": [], "post": [], "slot": []} for i in range(3)}
                streak = np.zeros((runner.state[2].shape[0], 4), dtype=np.int16)
                sample_stride = max(1, n // 10000)
            for segment in windows(record):
                if mode == "window_reset_all":
                    for state in runner.state:
                        state.fill(0)
                features = record["features"][segment]
                pack = runner.pack(features, record["x"][segment],
                                   record["y"][segment], record["t"][segment])
                state = tuple(torch.from_numpy(item[pack.cells].copy()) for item in runner.state)
                if capture:
                    address = record["y"][segment].astype(np.int64) // 4 * 87 + record["x"][segment].astype(np.int64) // 4
                    detail["local_count"][segment] = np.bincount(address, minlength=runner.state[0].shape[0])[address]
                for k in range(len(pack.event_slots)):
                    valid = pack.event_slots[k] >= 0
                    safe = np.maximum(pack.event_slots[k], 0)
                    x = torch.from_numpy(features[safe].astype(np.float32) / 255.0)
                    logits, updated, spikes = model.step(x, torch.from_numpy(pack.delta_ms[k]), state)
                    if mode == "no_trace_readout":
                        logits = logits - model.trace_gain * updated[3]
                    elif mode == "no_mem_readout":
                        logits = logits - model.mem_gain * (updated[2] / THRESHOLD)
                    mask = torch.from_numpy(valid)[:, None]
                    state = tuple(torch.where(mask, new, old) for new, old in zip(updated, state))
                    if mode.startswith("zero_"):
                        layer = {"zero_u1": 0, "zero_u2": 1, "zero_u3": 2, "zero_trace": 3}[mode]
                        state = tuple(torch.zeros_like(value) if i == layer else value
                                      for i, value in enumerate(state))
                    slot = pack.event_slots[k, valid]
                    global_slot = segment.start + slot
                    probability[global_slot] = torch.softmax(class_logits(logits[valid]), 1)[:, 1].numpy()
                    if capture:
                        detail["winner"][global_slot] = logits[valid].argmax(dim=1).numpy().astype(np.int8)
                        for layer in range(3):
                            post = updated[layer][valid].numpy()
                            spike = spikes[layer][valid].numpy()
                            pre = post + THRESHOLD * spike
                            detail["layer_spike"][global_slot, layer] = spike.mean(axis=1)
                            detail["layer_post_ge"][global_slot, layer] = (post >= THRESHOLD).mean(axis=1)
                            detail["layer_sg_support"][global_slot, layer] = (np.abs(pre - THRESHOLD) < 1).mean(axis=1)
                            sampled = global_slot % sample_stride == 0
                            if np.any(sampled):
                                sample = detail["samples"][str(layer)]
                                sample["pre"].append(pre[sampled].copy())
                                sample["post"].append(post[sampled].copy())
                                sample["slot"].append(global_slot[sampled].copy())
                        post3 = updated[2][valid].numpy()
                        spike3 = spikes[2][valid].numpy()
                        trace = updated[3][valid].numpy()
                        detail["post3"][global_slot] = post3
                        detail["pre3"][global_slot] = post3 + THRESHOLD * spike3
                        detail["spike3"][global_slot] = spike3
                        detail["trace"][global_slot] = trace
                        detail["trace_term"][global_slot] = trace * model.trace_gain.numpy()
                        detail["mem_term"][global_slot] = post3 / THRESHOLD * model.mem_gain.numpy()
                        cell = pack.cells[valid]
                        streak[cell] = np.where(spike3 > 0, streak[cell] + 1, 0)
                        detail["streak"][global_slot] = streak[cell]
                for destination, value in zip(runner.state, state):
                    destination[pack.cells] = value.numpy()
            result.append(probability)
            captures.append(detail)
    return result, captures


def full_arrays(records, probabilities):
    labels, scores = [], []
    for record, candidate_scores in zip(records, probabilities):
        full = np.zeros(len(record["labels"]), dtype=np.float32)
        full[record["indices"]] = candidate_scores
        scores.append(full)
        labels.append(record["labels"].astype(bool))
    return np.concatenate(labels), np.concatenate(scores)


def point(records, probabilities, threshold):
    overall = {key: {"tp": 0, "fp": 0, "fn": 0} for key in ("all", "ordinary", "high")}
    per_sequence = []
    for record, probability in zip(records, probabilities):
        target = record["target"].astype(bool)
        pred = probability >= threshold
        tp = int(np.count_nonzero(pred & target))
        fp = int(np.count_nonzero(pred & ~target))
        fn = int(record["labels"].sum()) - tp
        density = "high" if record["event_rate"] >= 20000 else "ordinary"
        row = {"name": record["name"], "density": density, "tp": tp, "fp": fp, "fn": fn}
        per_sequence.append(row)
        for group in ("all", density):
            for key in ("tp", "fp", "fn"):
                overall[group][key] += row[key]
    for row in overall.values():
        tp, fp, fn = (row[key] for key in ("tp", "fp", "fn"))
        row["iou"] = tp / (tp + fp + fn) if tp + fp + fn else None
        row["recall"] = tp / (tp + fn) if tp + fn else None
        row["precision"] = tp / (tp + fp) if tp + fp else None
    return {"groups": overall, "sequences": per_sequence}


def matched_fp_threshold(records, probabilities, budget):
    background = np.concatenate([p[~r["target"].astype(bool)] for r, p in zip(records, probabilities)])
    assert 0 <= budget < len(background)
    boundary = np.partition(background, len(background) - budget - 1)[len(background) - budget - 1]
    threshold = float(np.nextafter(np.float32(boundary), np.float32(np.inf)))
    actual = sum(int(np.count_nonzero((p >= threshold) & ~r["target"].astype(bool)))
                 for r, p in zip(records, probabilities))
    assert actual <= budget
    return threshold


def pr_arrays(labels, scores):
    order = np.argsort(-scores, kind="stable")
    sorted_score, sorted_label = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(scores) - 1]
    tp = np.cumsum(sorted_label, dtype=np.int64)[ends]
    fp = (ends + 1) - tp
    return {"threshold": sorted_score[ends], "tp": tp, "fp": fp,
            "precision": tp / np.maximum(tp + fp, 1),
            "recall": tp / max(int(labels.sum()), 1)}


def subgroup_masks(record):
    path = CACHE / "train_dev" / (Path(record["name"]).stem + "_C1-proposal-motion.npz")
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"].copy().astype(bool)
        ids = archive["identities"].copy()
        x, y, t = archive["x"].copy(), archive["y"].copy(), archive["t"].copy()
    history = motion_and_history(labels, ids, x, y, t, record["t0"])
    return {key: history["exposed"][key][record["indices"]] & history["eligible"][key][record["indices"]]
            for key in SUBSETS}


def subset_counts(records, probabilities, threshold):
    totals = {key: {"target_candidates": 0, "candidate_fn": 0, "tp": 0} for key in SUBSETS}
    for record, probability in zip(records, probabilities):
        target = record["target"].astype(bool)
        for key, mask in subgroup_masks(record).items():
            chosen = target & mask
            positive = probability >= threshold
            totals[key]["target_candidates"] += int(chosen.sum())
            totals[key]["candidate_fn"] += int(np.count_nonzero(chosen & ~positive))
            totals[key]["tp"] += int(np.count_nonzero(chosen & positive))
    return totals


def quantile(values):
    if not len(values):
        return None
    return np.quantile(values, [0, .01, .1, .25, .5, .75, .9, .99, 1]).tolist()


def captured_summary(records, probabilities, captures, threshold):
    groups = {}
    labels_all, local_all, input_all, post_all, fp_all = [], [], [], [], []
    for record, probability, detail in zip(records, probabilities, captures):
        target = record["target"].astype(bool)
        pred = probability >= threshold
        density = "high" if record["event_rate"] >= 20000 else "ordinary"
        masks = {"all": np.ones(len(target), dtype=bool), density: np.ones(len(target), dtype=bool),
                 record["name"]: np.ones(len(target), dtype=bool),
                 "target": target, "background": ~target,
                 "tp": target & pred, "fp": ~target & pred, "candidate_fn": target & ~pred}
        for name, mask in masks.items():
            if not np.any(mask):
                continue
            bucket = groups.setdefault(name, {"candidate_updates": 0, "spike_sum": np.zeros((3,), np.float64),
                "post_ge_sum": np.zeros((3,), np.float64), "sg_support_sum": np.zeros((3,), np.float64),
                "prototype_spikes": np.zeros((4,), np.float64), "prototype_post_ge": np.zeros((4,), np.float64),
                "prototype_pre_high": np.zeros((4,), np.float64), "prototype_pre_low": np.zeros((4,), np.float64),
                "streak_ge2": np.zeros((4,), np.float64), "winner": np.zeros((4,), np.int64),
                "trace_term_sum": np.zeros((4,), np.float64), "mem_term_sum": np.zeros((4,), np.float64),
                "post3_sample": [], "pre3_sample": [], "trace_sample": [],
                "layer_pre_sample": [[], [], []], "layer_post_sample": [[], [], []]})
            bucket["candidate_updates"] += int(mask.sum())
            bucket["spike_sum"] += detail["layer_spike"][mask].sum(axis=0)
            bucket["post_ge_sum"] += detail["layer_post_ge"][mask].sum(axis=0)
            bucket["sg_support_sum"] += detail["layer_sg_support"][mask].sum(axis=0)
            bucket["prototype_spikes"] += detail["spike3"][mask].sum(axis=0)
            bucket["prototype_post_ge"] += (detail["post3"][mask] >= THRESHOLD).sum(axis=0)
            bucket["prototype_pre_high"] += (detail["pre3"][mask] >= THRESHOLD + 1).sum(axis=0)
            bucket["prototype_pre_low"] += (detail["pre3"][mask] <= THRESHOLD - 1).sum(axis=0)
            bucket["streak_ge2"] += (detail["streak"][mask] >= 2).sum(axis=0)
            bucket["winner"] += np.bincount(detail["winner"][mask], minlength=4)
            bucket["trace_term_sum"] += detail["trace_term"][mask].sum(axis=0)
            bucket["mem_term_sum"] += detail["mem_term"][mask].sum(axis=0)
            # Deterministic sparse sample for signed distributions; counts above use every update.
            bucket["post3_sample"].append(detail["post3"][mask][::max(1, int(mask.sum()) // 2000)].reshape(-1))
            bucket["pre3_sample"].append(detail["pre3"][mask][::max(1, int(mask.sum()) // 2000)].reshape(-1))
            bucket["trace_sample"].append(detail["trace"][mask][::max(1, int(mask.sum()) // 2000)].reshape(-1))
            for layer in range(3):
                sample = detail["samples"][str(layer)]
                if sample["slot"]:
                    slot = np.concatenate(sample["slot"])
                    sampled_mask = mask[slot]
                    if np.any(sampled_mask):
                        bucket["layer_pre_sample"][layer].append(np.concatenate(sample["pre"])[sampled_mask].reshape(-1))
                        bucket["layer_post_sample"][layer].append(np.concatenate(sample["post"])[sampled_mask].reshape(-1))
        labels_all.append(target)
        local_all.append(detail["local_count"])
        input_all.append(detail["input_sum"])
        post_all.append(detail["post3"].max(axis=1))
        fp_all.append(~target & pred)
    finished = {}
    for name, bucket in groups.items():
        n = bucket.pop("candidate_updates")
        finished[name] = {"candidate_updates": n,
            "layer_spike_fraction": (bucket["spike_sum"] / n).tolist(),
            "layer_post_ge_fraction": (bucket["post_ge_sum"] / n).tolist(),
            "layer_surrogate_support_fraction": (bucket["sg_support_sum"] / n).tolist(),
            "prototype_spike_fraction": (bucket["prototype_spikes"] / n).tolist(),
            "prototype_post_ge_fraction": (bucket["prototype_post_ge"] / n).tolist(),
            "prototype_pre_high_fraction": (bucket["prototype_pre_high"] / n).tolist(),
            "prototype_pre_low_fraction": (bucket["prototype_pre_low"] / n).tolist(),
            "prototype_consecutive_spike_fraction": (bucket["streak_ge2"] / n).tolist(),
            "winning_prototype_count": bucket["winner"].tolist(),
            "mean_trace_readout_term": (bucket["trace_term_sum"] / n).tolist(),
            "mean_mem_readout_term": (bucket["mem_term_sum"] / n).tolist(),
            "signed_output_pre_quantiles_sample": quantile(np.concatenate(bucket["pre3_sample"])),
            "signed_output_post_quantiles_sample": quantile(np.concatenate(bucket["post3_sample"])),
            "output_trace_quantiles_sample": quantile(np.concatenate(bucket["trace_sample"])),
            "signed_layer_pre_quantiles_sample": [quantile(np.concatenate(rows)) if rows else None
                                                   for rows in bucket["layer_pre_sample"]],
            "signed_layer_post_quantiles_sample": [quantile(np.concatenate(rows)) if rows else None
                                                    for rows in bucket["layer_post_sample"]]}
    local = np.concatenate(local_all)
    input_sum = np.concatenate(input_all)
    post_max = np.concatenate(post_all)
    false_positive = np.concatenate(fp_all)
    target = np.concatenate(labels_all)
    background = ~target
    quartiles = np.quantile(local, [0, .25, .5, .75, 1]).tolist()
    bands = []
    for index, (lo, hi) in enumerate(zip(quartiles[:-1], quartiles[1:])):
        mask = background & (local >= lo) & (local <= hi if index == 0 else local < hi)
        if index:
            mask = background & (local > lo) & (local <= hi)
        bands.append({"local_count_range": [lo, hi], "background_candidates": int(mask.sum()),
                      "false_positives": int(np.count_nonzero(false_positive & mask)),
                      "false_positive_fraction": float(np.mean(false_positive[mask])) if np.any(mask) else None})
    association = {"candidate_local_count_quantiles": quartiles,
        "input_sum_quantiles": quantile(input_sum),
        "pearson_log1p_local_count_vs_output_post_max": float(np.corrcoef(np.log1p(local), post_max)[0, 1]),
        "pearson_input_sum_vs_output_post_max": float(np.corrcoef(input_sum, post_max)[0, 1]),
        "background_local_count_bands": bands,
        "warning": "discrete quantile bands can be empty; candidate-level correlations are descriptive, not causal"}
    return finished, association


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results" / "GEN2-S3-001-A")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only output exists: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    index, manifest = read_index()
    dev = role_records(index, manifest, "train_dev")
    main_subset = [role_records(index, manifest, "train_main", only_name=name)[0] for name in MAIN_DIAGNOSTIC]
    ann, snn = load_model("ann"), load_model("snn")
    started = time.perf_counter()
    ann_prob = replay_ann(ann, dev)
    snn_prob, snn_capture = replay_snn(snn, dev, capture=True)
    expected_ann = json.loads((RUN / "ann_history.json").read_text())[-1]["train_dev"]["average_precision"]
    expected_snn = json.loads((RUN / "snn_history.json").read_text())[-1]["train_dev"]["average_precision"]
    labels, ann_full = full_arrays(dev, ann_prob)
    _, snn_full = full_arrays(dev, snn_prob)
    assert abs(average_precision(labels, ann_full) - expected_ann) < 1e-9
    assert abs(average_precision(labels, snn_full) - expected_snn) < 1e-9
    np.savez_compressed(args.output / "pr_full.npz", labels_positive=int(labels.sum()),
                        **{f"{kind}_{key}": value for kind, lab, prob in
                           (("ann", labels, ann_full), ("snn", labels, snn_full))
                           for key, value in pr_arrays(lab, prob).items()})
    comparisons = {}
    for kind, probability, ap in (("ann", ann_prob, expected_ann), ("snn", snn_prob, expected_snn)):
        comparisons[kind] = {"average_precision": ap, "fp_budgets": {}}
        for budget in FP_BUDGETS:
            threshold = matched_fp_threshold(dev, probability, budget)
            comparisons[kind]["fp_budgets"][str(budget)] = {"threshold": threshold,
                "point": point(dev, probability, threshold),
                "candidate_target_subsets": subset_counts(dev, probability, threshold)}
    baseline_threshold = comparisons["snn"]["fp_budgets"]["4137"]["threshold"]
    state, association = captured_summary(dev, snn_prob, snn_capture, baseline_threshold)
    interventions = {}
    for mode in INTERVENTIONS:
        probabilities, _ = replay_snn(snn, dev, mode=mode)
        _, full = full_arrays(dev, probabilities)
        interventions[mode] = {"average_precision": average_precision(labels, full), "fp_budgets": {}}
        for budget in FP_BUDGETS:
            threshold = matched_fp_threshold(dev, probabilities, budget)
            interventions[mode]["fp_budgets"][str(budget)] = {"threshold": threshold,
                "point": point(dev, probabilities, threshold),
                "candidate_target_subsets": subset_counts(dev, probabilities, threshold)}
    main_prob, main_capture = replay_snn(snn, main_subset, capture=True)
    # Train-main is only a fixed phenomenon check; its labels cannot select S3 thresholds.
    main_state, main_association = captured_summary(main_subset, main_prob, main_capture, 0.6)
    output = {"experiment_id": "GEN2-S3-001-A", "status": "read_only_diagnostic",
              "train_main_diagnostic_names": list(MAIN_DIAGNOSTIC),
              "train_dev_names": [r["name"] for r in dev],
              "s2_checkpoint_sha256": {kind: sha256(RUN / f"{kind}_epoch05.pt") for kind in ("ann", "snn")},
              "index_sha256": manifest["index_sha256"], "s3_protocol_sha256": sha256(HERE / "PROTOCOL.md"),
              "models": comparisons, "snn_state_by_group_at_4137_fp": state,
              "snn_input_state_association": association,
              "same_checkpoint_interventions": interventions,
              "train_main_fixed_subset": {"names": list(MAIN_DIAGNOSTIC),
                                          "state_at_0.6": main_state,
                                          "input_state_association": main_association},
              "wall_s": time.perf_counter() - started,
              "caveat": "Train-dev thresholds and intervention results are diagnostic; no retrained state comparison or held-out validation"}
    (args.output / "diagnosis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "wall_s": output["wall_s"],
                      "ann_fp4137": comparisons["ann"]["fp_budgets"]["4137"]["point"]["groups"]["all"],
                      "snn_fp4137": comparisons["snn"]["fp_budgets"]["4137"]["point"]["groups"]["all"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
