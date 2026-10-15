"""Complete development evaluation and causal stress prefixes, without updates."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ..h2pro_snn.diagnostics import aggregate_counts, summarize_counts
from ..h2pro_snn.stream_data import encode_window, prefetch, stream_batches
from ..h2pro_snn.training import cpu_weights, selection_key
from ..h2pro_snn_v2_2.diagnostics import distribution
from ..h2pro_snn_v2_2.training import logit_counts, operating_point
from .probe import READOUTS, SCALE_NAMES

DETAIL_FIELDS = ("current_margin", "carry_margin", "bias_margin", "pre_winning_bg_fg")


class ScaleSummary:
    def __init__(self):
        self.steps = 0
        self.low = np.full((3, 8), np.inf)
        self.high = np.full((3, 8), -np.inf)
        self.sums = np.zeros((3, len(SCALE_NAMES)), np.float64)

    def add(self, scales, active):
        a = scales[:, active].cpu().numpy().reshape(-1, 3, len(SCALE_NAMES))
        if not len(a):
            return
        if not np.isfinite(a).all():
            raise FloatingPointError("Nonfinite scale diagnostic")
        self.steps += len(a)
        self.low = np.minimum(self.low, a[:, :, :8].min(0))
        self.high = np.maximum(self.high, a[:, :, :8].max(0))
        self.sums += a.sum(0, dtype=np.float64)

    def result(self):
        rows = {}
        for i, name in enumerate(("stem", "hidden", "head")):
            values = self.sums[i]
            rows[name] = {
                "global_min": dict(zip(SCALE_NAMES[:8], self.low[i].tolist())),
                "global_max": dict(zip(SCALE_NAMES[:8], self.high[i].tolist())),
                "mean_per_active_step": dict(zip(SCALE_NAMES, (values/max(self.steps, 1)).tolist())),
                "positive_current_blocked_fraction": float(values[10]/max(values[9], 1)),
                "negative_residual_grid_fraction": float(values[8]/max(values[11], 1)),
            }
        return {"active_stream_steps": self.steps, "layers": rows,
                "stem_current_note": "Stem current reconstructed as pre-carry (FP32 rounding); hidden/head use exact convolution outputs",
                "blocked_definition": "current>0 and pre<=0 and decayed residual<0; all grid units, not event accuracy"}


def step(model, batch, state, auxiliary, device):
    inputs = torch.from_numpy(batch["inputs"]).to(device)
    reset = torch.as_tensor(batch["reset"], device=device)
    if state is not None:
        state = model.reset_slots(state, reset)
    if auxiliary is not None:
        auxiliary = auxiliary * (~reset).to(auxiliary).view(-1, 1, 1, 1)
    result = model.probe(inputs, batch["maps"], state, auxiliary)
    scores, state, auxiliary, _, _, details, scales = result
    if (any(not torch.isfinite(v).all() for v in scores.values()) or
            any(not torch.isfinite(v).all() for v in state) or not torch.isfinite(auxiliary).all()):
        raise FloatingPointError("Nonfinite frozen diagnostic forward")
    return scores, state, auxiliary, details, scales


def metrics_for_scores(store, scores, thresholds, reference_threshold):
    def at(threshold):
        rows = []
        for name, values in scores.items():
            r = store.get("train_dev", name)
            counts = logit_counts(r.arrays["label"], values, r.arrays["candidate"], threshold)
            rows.append({"sequence": name, "high_density": r.metadata["high_density"], **summarize_counts(counts)})
        return {**operating_point(threshold), "overall": aggregate_counts(rows),
                "high_density": aggregate_counts([r for r in rows if r["high_density"]]),
                "ordinary": aggregate_counts([r for r in rows if not r["high_density"]]), "sequences": rows}
    scans = [at(float(t)) for t in thresholds]
    return {"selected": max(scans, key=selection_key), "fixed_reference_threshold": at(reference_threshold),
            "scan": [{k: r[k] for k in ("threshold", "overall", "high_density")} for r in scans],
            "scope": "Train-dev diagnostic only; counterfactual threshold selection does not replace the source checkpoint"}


def decomposition(values, labels, logits, threshold):
    labels = np.asarray(labels, bool)
    result = {}
    masks = {"all_candidates": np.ones(len(labels), bool),
             "targets": labels, "background": ~labels,
             "true_positives": labels & (logits >= threshold),
             "false_positives": ~labels & (logits >= threshold),
             "false_negatives": labels & (logits < threshold),
             "background_above80": ~labels & (logits > 80)}
    for name, mask in masks.items():
        v = values[mask]
        n = len(v)
        result[name] = {
            "count": n, "logit": distribution(logits[mask]),
            "current_margin": distribution(v[:, 0]), "carry_margin": distribution(v[:, 1]),
            "bias_margin": distribution(v[:, 2]), "winning_background_pre": distribution(v[:, 3]),
            "winning_foreground_pre": distribution(v[:, 4]),
            "negative_background_fraction": float((v[:, 3] < 0).mean()) if n else None,
            "positive_carry_margin_fraction": float((v[:, 1] > 0).mean()) if n else None,
            "carry_exceeds_half_absolute_logit_fraction": float((v[:, 1] > np.abs(logits[mask])*.5).mean()) if n else None,
        }
    return result


def detail_matrix(details):
    return torch.cat((details["current_margin"][:, None], details["carry_margin"][:, None],
                      details["bias_margin"][:, None], details["pre_winning_bg_fg"]), 1).cpu().numpy()


@torch.no_grad()
def evaluate_probe(model, store, device, config, reference_threshold, output_scores=None):
    before = cpu_weights(model)
    names = store.names("train_dev")
    scores = {key: {name: np.zeros(len(store.get("train_dev", name).arrays["t"]), np.float32) for name in names}
              for key in READOUTS}
    seen = {name: np.zeros(len(scores["original"][name]), bool) for name in names}
    details_per_sequence = {name: [] for name in names}
    labels_per_sequence = {name: [] for name in names}
    logits_per_sequence = {name: [] for name in names}
    state = auxiliary = None
    scale_summary = ScaleSummary()
    events = windows = 0
    started = time.perf_counter()
    for index, (batch, _) in enumerate(prefetch(iter(stream_batches(store, "train_dev", config["batch_size"]))), 1):
        values, state, auxiliary, details, scales = step(model, batch, state, auxiliary, device)
        scale_summary.add(scales, batch["active"])
        values = {k: v.cpu().numpy() for k, v in values.items()}
        matrix = detail_matrix(details)
        offset = 0
        for ref in batch["refs"]:
            name, ids = ref["sequence"], ref["ids"]
            stop = offset+len(ids)
            if seen[name][ids].any():
                raise AssertionError("Duplicate prediction")
            seen[name][ids] = True
            for key in READOUTS:
                scores[key][name][ids] = values[key][offset:stop]
            mask = batch["candidate"][offset:stop]
            details_per_sequence[name].append(matrix[offset:stop][mask])
            labels_per_sequence[name].append(batch["labels"][offset:stop][mask])
            logits_per_sequence[name].append(values["original"][offset:stop][mask])
            offset = stop
        if offset != len(batch["labels"]):
            raise AssertionError("Unmapped events")
        events += offset
        windows += int(batch["active"].sum())
        if index % 100 == 0:
            print({"stage": "readout_audit", "analog": model.analog, "batch": index, "events": events}, flush=True)
    if not all(a.all() for a in seen.values()):
        raise AssertionError("Incomplete development coverage")
    seq_details = {}
    all_values, all_labels, all_logits = [], [], []
    for name in names:
        v = np.concatenate(details_per_sequence[name])
        y = np.concatenate(labels_per_sequence[name])
        z = np.concatenate(logits_per_sequence[name])
        seq_details[name] = decomposition(v, y, z, reference_threshold)
        all_values.append(v); all_labels.append(y); all_logits.append(z)
    if output_scores is not None:
        payload: dict[str, Any] = {key+"__"+name.replace("/", "_"): a
                                  for key, rows in scores.items() for name, a in rows.items()}
        np.savez_compressed(output_scores, **payload)
    if not all(torch.equal(v, cpu_weights(model)[k]) for k, v in before.items()):
        raise AssertionError("Diagnostic changed parameters")
    return {"role": "train_dev", "events": events, "windows": windows,
            "candidate_events": sum(len(v) for v in all_labels),
            "cell": "continuous_charge" if model.analog else "integer_spike",
            "readouts": {key: metrics_for_scores(store, value, config["logit_thresholds"], reference_threshold)
                         for key, value in scores.items()},
            "decomposition_at_reference_threshold": decomposition(np.concatenate(all_values), np.concatenate(all_labels),
                                                                  np.concatenate(all_logits), reference_threshold),
            "sequence_decomposition": seq_details, "state_scales": scale_summary.result(),
            "parameters_unchanged": True, "parameter_updates": 0, "seconds": time.perf_counter()-started,
            "interpretation": "Same-weight readout counterfactuals; no adaptation. Transplanted ANN weights are not a trained ANN accuracy bound."}


def prefix_batch(record, window):
    inputs, coord, ids = encode_window(record, window)
    return {"inputs": inputs[:, None], "maps": np.stack((coord[0], np.zeros(len(ids), np.int64), *coord[1:])),
            "candidate": record.arrays["candidate"][ids], "labels": record.arrays["label"][ids],
            "teacher": record.arrays["teacher"][ids], "reset": np.array([window == 0]), "active": np.array([True]),
            "refs": [{"sequence": record.metadata["sequence"], "window": window, "slot": 0, "ids": ids}]}


@torch.no_grad()
def hard_prefixes(model, store, device, reference_threshold):
    prefixes = {"098": 5, "042": 5, "043": 41, "097": 6, "088": 32, "096": 76}
    result = []
    before = cpu_weights(model)
    for number, last in prefixes.items():
        name = "train/train_"+number+".npz"
        record = store.get("train_main", name)
        state = auxiliary = None
        scales_total = ScaleSummary()
        peak = np.zeros(3)
        for window in range(last+1):
            batch = prefix_batch(record, window)
            scores, state, auxiliary, details, scales = step(model, batch, state, auxiliary, device)
            scales_total.add(scales, batch["active"])
            layer_stats = model.last_diagnostics
            assert layer_stats is not None
            peak = np.maximum(peak, layer_stats[..., 4].amax((0, 1)).cpu().numpy())
        mask = batch["candidate"]
        z = scores["original"].cpu().numpy()
        matrix = detail_matrix(details)
        ids = np.flatnonzero(mask)
        extreme = []
        for i in ids[np.argsort(-np.abs(z[ids]), kind="stable")[:5]]:
            item = {"event_id": int(batch["refs"][0]["ids"][i]), "label": int(batch["labels"][i]),
                    "query": batch["maps"][:, i].tolist(), "readouts": {k: float(v[i]) for k, v in scores.items()}}
            for key in ("pre", "current", "carry", "bias", "winners"):
                item[key] = details[key][i].cpu().tolist()
            extreme.append(item)
        result.append({"sequence": name, "windows": last+1, "reported_window": last,
                       "layer_peak_activation_in_threshold_units": peak.tolist(),
                       "activation_is_integer_spike_count": not model.analog,
                       "state_scales": scales_total.result(), "extreme_candidates": extreme,
                       "decomposition": decomposition(matrix[mask], batch["labels"][mask], z[mask], reference_threshold),
                       "readout_at_reference_threshold": {k: summarize_counts(logit_counts(batch["labels"], v.cpu().numpy(), mask, reference_threshold))
                                                          for k, v in scores.items()}})
    assert all(torch.equal(v, cpu_weights(model)[k]) for k, v in before.items())
    return {"scope": "Known outlier slots, independently replayed from each sequence start; not a full Train-main sweep",
            "prefixes": result, "parameter_updates": 0, "parameters_unchanged": True}
