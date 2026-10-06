"""Observe the fixed SNN protocol without changing neurons, losses or training data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from wangdao.temporal_segmentation import TemporalSegmenter, collision_summary, event_loss
from wangdao.temporal_study import (PROTOCOL, RUN_ROOT, SequenceStore, checkpoint_save,
                                    choose_device, collate, frozen_inputs, read_json, report, sha256, update, write_json)


def describe(values: torch.Tensor) -> dict:
    a = values.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if not len(a):
        return {"count": 0}
    if not np.isfinite(a).all():
        raise FloatingPointError("nonfinite diagnostic observation")
    return {"count": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "p10_p50_p90_p99": np.quantile(a, [.1, .5, .9, .99]).tolist(),
            "nonzero_fraction": float(np.count_nonzero(a) / len(a))}


def masks(indices: torch.Tensor, labels: torch.Tensor, batch: int, dilated: bool) -> dict[str, torch.Tensor]:
    shape = (4, batch, 65, 87)
    positive, negative = torch.zeros(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.bool)
    i = indices.cpu()
    coarse = (i[:, 0], i[:, 1], i[:, 3] // 4, i[:, 4] // 4)
    for mask, member in ((positive, labels.cpu() > .5), (negative, labels.cpu() <= .5)):
        mask[tuple(c[member] for c in coarse)] = True
    occupied = positive | negative
    support = torch.zeros((4, 3, batch, 65, 87), dtype=torch.bool)
    for step in range(4):
        for layer in range(3):
            current = occupied[step] if layer == 0 else support[step, layer - 1]
            if layer > 0:
                dilation = 2 if dilated and layer == 2 else 1
                current = F.conv2d(current.float().unsqueeze(1), torch.ones(1, 1, 3, 3),
                                   padding=dilation, dilation=dilation).squeeze(1) > 0
            support[step, layer] = current | (support[step - 1, layer] if step else False)
    return {"target_only_cells": positive & ~negative, "background_only_cells": negative & ~positive,
            "mixed_cells": positive & negative, "empty_current_cells": ~occupied,
            "causal_event_support": support}


def code_audit(spikes: torch.Tensor, indices: torch.Tensor, labels: torch.Tensor) -> dict:
    i = indices.to(spikes.device)
    selected = spikes[i[:, 0], i[:, 1], :, i[:, 3] // 4, i[:, 4] // 4]
    if not torch.all((selected == 0) | (selected == 1)):
        raise ValueError("readout input must be binary")
    codes = (selected.long() * (2 ** torch.arange(8, device=spikes.device))).sum(1).cpu().numpy()
    # The 1x1 head shares the same deterministic function for each polarity/subpixel phase.
    phase = (indices[:, 2] * 16 + indices[:, 3] % 4 * 4 + indices[:, 4] % 4).cpu().numpy()
    keys = phase * 256 + codes
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    y = labels.cpu().numpy()
    pos = np.asarray(np.bincount(inverse, weights=y)).astype(np.int64)
    result = {"unique_codes": len(np.unique(codes)), "phase_code_oracle": collision_summary(pos, counts - pos),
              "scope": "label-assisted upper bound for any deterministic current-step readout of phase+8 spikes; not generalization accuracy"}
    for name, member in (("target_events", y > .5), ("background_events", y <= .5)):
        result[name] = {"events": int(member.sum()),
                        "zero_code_fraction": float(np.mean(codes[member] == 0)) if member.any() else None}
    return result


def snapshot(model: TemporalSegmenter, batch, device: torch.device) -> dict:
    x, idx, y = (v.to(device) for v in batch)
    traces: list[dict[str, Any]] = []
    def observe(cell, inputs, output):
        current, previous = inputs
        current.retain_grad()
        traces.append({"current": current, "pre_reset": (cell.beta * previous + current).detach(),
                       "post_reset": output[1].detach(), "spike": output[0].detach()})
    handle = model.cell.register_forward_hook(observe)
    model.zero_grad(set_to_none=True)
    try:
        out = model(x)
    finally:
        handle.remove()
    if len(traces) != 12:
        raise AssertionError("expected 4 time steps and 3 hidden layers")
    # Observer must not change the public forward output.
    with torch.no_grad():
        plain = model(x)
        torch.testing.assert_close(out.logits, plain.logits, rtol=0, atol=0)
    chosen = out.logits[tuple(idx.T)]
    loss = event_loss(chosen, y, out.rates, True)
    loss.backward()
    gradient = {name: describe(p.grad) for name, p in model.named_parameters() if p.grad is not None}
    region = masks(batch[1], batch[2], x.shape[1], model.arm.startswith("Dilated"))
    layers = []
    for layer in range(3):
        row: dict[str, Any] = {"layer": layer + 1, "regions": {}}
        stack = {k: torch.stack([traces[t * 3 + layer][k].detach().cpu() for t in range(4)])
                 for k in ("current", "pre_reset", "post_reset", "spike")}
        stack["current_gradient"] = torch.stack([traces[t * 3 + layer]["current"].grad.detach().cpu() for t in range(4)])
        stack["threshold_gap"] = stack["pre_reset"] - .25
        stack["surrogate_derivative"] = 1 / (1 + 10 * stack["threshold_gap"].abs()).square()
        selected_masks = {k: v for k, v in region.items() if k != "causal_event_support"}
        selected_masks["causal_event_support"] = region["causal_event_support"][:, layer]
        selected_masks["global"] = torch.ones_like(selected_masks["causal_event_support"])
        for name, mask in selected_masks.items():
            expanded = mask.unsqueeze(2).expand_as(stack["spike"])
            row["regions"][name] = {k: describe(v[expanded]) for k, v in stack.items()}
        ci = batch[1]
        for name, member in (("target_events", batch[2] > .5), ("background_events", batch[2] <= .5)):
            row["regions"][name] = {k: describe(v[ci[member, 0], ci[member, 1], :, ci[member, 3] // 4, ci[member, 4] // 4])
                                    for k, v in stack.items()}
        layers.append(row)
    last_spikes = torch.stack([traces[t * 3 + 2]["spike"] for t in range(4)])
    readout = code_audit(last_spikes, batch[1], batch[2])
    readout["by_label"] = {name: {"logits": describe(chosen[member]), "probability": describe(chosen[member].sigmoid())}
                            for name, member in (("target_events", y > .5), ("background_events", y <= .5))}
    assert model.head.bias is not None
    weights = model.head.weight.detach().flatten(1)
    low = model.head.bias.detach() + torch.minimum(weights, torch.zeros_like(weights)).sum(1)
    high = model.head.bias.detach() + torch.maximum(weights, torch.zeros_like(weights)).sum(1)
    readout["possible_logit_bounds_per_phase"] = torch.stack((low, high), 1).cpu().tolist()
    readout["maximum_possible_probability"] = float(high.sigmoid().max().cpu())
    phase = idx[:, 2] * 16 + idx[:, 3] % 4 * 4 + idx[:, 4] % 4
    optimistic = torch.where(y > .5, high[phase], low[phase])
    readout["optimistic_task_loss_fixed_head"] = float(event_loss(optimistic, y, out.rates.detach(), False).cpu())
    readout["bound_note"] = "Allows each event its own ideal 8-bit code, even conflicting events; lower bound on task loss with this fixed head, not achievable accuracy."
    # Split parameter gradients on a fresh graph, without observer retention or optimizer updates.
    model.zero_grad(set_to_none=True)
    out = model(x)
    main = event_loss(out.logits[tuple(idx.T)], y, out.rates, False)
    regularizer = .01 * (F.relu(.01 - out.rates).square() + F.relu(out.rates - .25).square()).mean()
    parameters = list(model.named_parameters())
    task_grad = torch.autograd.grad(main, [p for _, p in parameters], retain_graph=True)
    rate_grad = torch.autograd.grad(regularizer, [p for _, p in parameters], allow_unused=True)
    split = {}
    for (name, _), a, b in zip(parameters, task_grad, rate_grad):
        norm = float(a.norm().cpu())
        reg_norm = float(b.norm().cpu()) if b is not None else 0.0
        split[name] = {"task_l2": norm, "regularizer_l2": reg_norm, "ratio": reg_norm / norm if norm else None}
    model.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach().cpu()), "task_loss": float(main.detach().cpu()),
            "rate_penalty": float(regularizer.detach().cpu()), "layers": layers, "readout": readout,
            "parameter_gradients": gradient, "gradient_split": split, "observer_equivalence": True,
            "regions": "cell groups are unique 4x4 cells; event groups retain duplicates and can share a mixed cell; support includes causal spatial/temporal reach"}


def interpret(result: dict) -> None:
    observations = {}
    for arm, row in result["arms"].items():
        initial, trained = row["initial_fit"], row["trained_fit"]
        required = initial["loss"] * .8
        floor = trained["readout"]["optimistic_task_loss_fixed_head"]
        observations[arm] = {
            "fit_target_event_spike_rates": [l["regions"]["target_events"]["spike"]["mean"] for l in trained["layers"]],
            "probe_target_event_spike_rates": [l["regions"]["target_events"]["spike"]["mean"] for l in row["trained_probe"]["layers"]],
            "fit_global_spike_rates": [l["regions"]["global"]["spike"]["mean"] for l in trained["layers"]],
            "original_learning_gate_loss": required, "fixed_head_optimistic_task_loss": floor,
            "fixed_head_cannot_meet_original_learning_gate": floor > required,
            "fixed_head_maximum_loss_reduction_fraction": 1 - floor / initial["loss"],
            "fit_code_oracle_iou_before_after": [initial["readout"]["phase_code_oracle"]["oracle_event_iou"],
                                                 trained["readout"]["phase_code_oracle"]["oracle_event_iou"]],
            "trained_last_hidden_rate_regularizer_gradient_l2": trained["gradient_split"]["hidden.2.weight"]["regularizer_l2"],
        }
    result["observations"] = observations
    result["decision"] = {"hardware_mainline_changed": False, "architecture_changed": False,
                          "parameter_search_performed": False, "snn_remains_project_target": True,
                          "next": "controlled_readout_scale_and_causal_membrane_readout_comparison",
                          "do_not_infer_global_silence_from_global_average": True}
    result["conclusion"] = ("Target positions remain active; global means obscure local high firing. "
                            "Fixed readout bounds and code collisions warrant a readout-focused controlled experiment. "
                            "Observational diagnosis does not establish the best remedy or long-training accuracy.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=RUN_ROOT / "TEMPORAL-SNN-DIAG-v1")
    parser.add_argument("--source", type=Path, default=ROOT / "results/temporal_segmentation_study.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--arms", nargs="+", choices=("Local-SNN", "Dilated-SNN"), default=["Local-SNN", "Dilated-SNN"])
    parser.add_argument("--updates", type=int, choices=(3, 100), default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate architecture")
    directory = args.run_directory.resolve()
    if not directory.is_relative_to(RUN_ROOT.resolve()) or directory == RUN_ROOT.resolve():
        raise ValueError("diagnostic artifacts must be inside ignored results/runs")
    output = args.output or directory / "diagnostic.json"
    if output.suffix != ".json" or output.exists() or (args.output and (args.output.resolve().parent != ROOT / "results" or args.updates != 100 or set(args.arms) != {"Local-SNN", "Dilated-SNN"})):
        raise ValueError("output must be new; formal output requires both arms and full replay")
    if (directory / "diagnostic.json").exists():
        raise FileExistsError("choose a new diagnostic run directory")
    source = read_json(args.source)
    pre = source["preflight"]
    if source["protocol"] != PROTOCOL or source["frozen_inputs"] != frozen_inputs():
        raise ValueError("source protocol or frozen inputs mismatch")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = choose_device(args.device)
    started = time.perf_counter()
    store = SequenceStore(directory / "cache", 1024)
    fit = [tuple(v) for v in pre["learning_blocks"]]
    probe = []
    for path in store.paths["train_main"]:
        if path.name in {name for name, _ in fit}:
            continue
        record = store.get("train_main", path.name)
        for block in record.blocks:
            lo, hi = np.searchsorted(record.bins, [int(block) * 4, (int(block) + 1) * 4])
            labels = record.events["label"][lo:hi]
            if np.any(labels == 1) and np.any(labels == 0):
                probe.append((path.name, int(block)))
                break
        if len(probe) == 8:
            break
    if len(fit) != 8 or len(probe) != 8:
        raise ValueError("expected 8 fit blocks and 8 independent sequence probe blocks")
    fit_batch, probe_batch = collate(store, "train_main", fit), collate(store, "train_main", probe)
    preparation_s = time.perf_counter() - started
    result = report("snn_diagnosis", directory)
    result.update(data_role=["train_main"], pilot=args.updates != 100, source=str(args.source),
                  source_sha256=sha256(args.source),
                  frozen_inputs=source["frozen_inputs"], fit_blocks=fit, probe_blocks=probe, arms={},
                  preparation_s=preparation_s, updates=args.updates)
    result["environment"]["device"] = str(device)
    for arm in args.arms:
        arm_started = time.perf_counter()
        model = TemporalSegmenter(arm, 42, tuple(pre["priors"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        initial = snapshot(model, fit_batch, device)
        initial_probe = snapshot(model, probe_batch, device)
        history = []
        for number in range(args.updates):
            loss, rates = update(model, fit_batch, optimizer, device)
            expected = pre["arms"][arm]["history"][number]
            if abs(loss - expected["loss"]) > 1e-7 or not np.allclose(rates, expected["rates"], atol=1e-7, rtol=0):
                raise AssertionError(f"original learning trajectory mismatch at {arm} update {number + 1}")
            history.append({"update": number + 1, "loss": loss, "rates": rates})
        trained = snapshot(model, fit_batch, device)
        trained_probe = snapshot(model, probe_batch, device)
        if args.updates == 100 and abs(trained["loss"] - pre["arms"][arm]["final_loss"]) > 1e-7:
            raise AssertionError("final replay loss mismatch")
        checkpoint_save(directory / f"{arm}-diagnostic.pt", {"model": model.state_dict(), "updates": args.updates,
                         "arm": arm, "seed": 42, "priors": pre["priors"], "protocol": PROTOCOL})
        result["arms"][arm] = {"initial_fit": initial, "initial_probe": initial_probe, "trained_fit": trained,
                               "trained_probe": trained_probe, "history": history,
                               "trajectory_matches_source": True, "seconds": time.perf_counter() - arm_started}
        write_json(directory / "partial.json", result)
        print(arm, "diagnosed", round(time.perf_counter() - arm_started, 2), "seconds", flush=True)
    result.update(status="ok", elapsed_s=time.perf_counter() - started, cache=store.stats)
    result["primary_metrics"] = {a: {split: {"loss": row[split]["loss"], "readout": row[split]["readout"]}
                                         for split in ("initial_fit", "trained_fit", "trained_probe")}
                                 for a, row in result["arms"].items()}
    interpret(result)
    write_json(directory / "diagnostic.json", result)
    if output.resolve() != (directory / "diagnostic.json").resolve():
        write_json(output, result)
    print({"status": result["status"], "elapsed_s": result["elapsed_s"], "output": str(output)}, flush=True)


if __name__ == "__main__":
    main()
