"""Six fixed short SNN readout controls; no automatic full training or search."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from wangdao.metrics import event_metrics
from wangdao.temporal_segmentation import TemporalSegmenter, event_loss, structural_cost
from wangdao.temporal_study import (RUN_ROOT, PROTOCOL, SequenceStore, checkpoint_save, choose_device, collate,
                                    frozen_inputs, read_json, report, scan_thresholds, sha256, update, write_json,
                                    evaluate_model, preparation_timing, training_inventory)

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"head_std": .01, "readout": "instant"},
    "scaled_init": {"head_std": 8 ** -.5, "readout": "instant"},
    "membrane": {"head_std": .01, "readout": "membrane"},
}
READOUT_PROTOCOL = {
    "version": 1, "variants": VARIANTS, "seed": 42, "updates": 100,
    "base_training_protocol": PROTOCOL, "optimizer": "AdamW lr=1e-3 weight_decay=1e-4 constant for short check",
    "readout_equation": "v[t]=0.875*v[t-1]+W*s[t]; logits[t]=v[t]+bias; reset every 4 steps",
    "admission": "finite; all hidden layers fire; fit total loss drops >=20%; final fit task loss and probe task loss no worse than paired baseline",
    "scope": "short learning feasibility only; no model promotion or claim of full-data superiority",
    "no_combination_or_scale_search": True,
}


def measure(model: TemporalSegmenter, batch, device: torch.device, threshold: float | None = None) -> dict:
    x, indices, labels = (a.to(device) for a in batch)
    spikes = []
    def observe(module, inputs, output):
        spikes.append(output[0].detach())
    handle = model.cell.register_forward_hook(observe)
    try:
        with torch.no_grad():
            output = model(x)
    finally:
        handle.remove()
    if len(spikes) != 12 or not torch.isfinite(output.logits).all():
        raise AssertionError("incomplete or nonfinite SNN output")
    logits = output.logits[tuple(indices.T)]
    probabilities = logits.sigmoid().cpu().numpy()
    truth = labels.cpu().numpy()
    chosen = scan_thresholds(truth, probabilities)["best_threshold"] if threshold is None else threshold
    layers = []
    for layer in range(3):
        value = torch.stack(spikes[layer::3])
        at_events = value[indices[:, 0], indices[:, 1], :, indices[:, 3] // 4, indices[:, 4] // 4]
        layers.append({"global_rate": float(value.mean().cpu()),
                       "target_event_rate": float(at_events[labels > .5].mean().cpu()),
                       "background_event_rate": float(at_events[labels <= .5].mean().cpu())})
    return {"total_loss": float(event_loss(logits, labels, output.rates, True).cpu()),
            "task_loss": float(event_loss(logits, labels, output.rates, False).cpu()),
            "threshold": chosen, "event": event_metrics(truth, probabilities, chosen), "layers": layers,
            "target_probability_mean": float(probabilities[truth > .5].mean()),
            "background_probability_mean": float(probabilities[truth <= .5].mean()),
            "threshold_scope": "fit-selected threshold; probe uses fit threshold unchanged"}


def decide(arms: dict, complete: bool) -> dict:
    checks = {}
    for arm, variants in arms.items():
        if "baseline" not in variants:
            continue
        baseline = variants["baseline"]
        checks[arm] = {}
        for name, row in variants.items():
            if name == "baseline":
                continue
            reduction = 1 - row["final_fit"]["total_loss"] / row["initial_fit"]["total_loss"]
            active = all(l["global_rate"] > 0 for l in row["final_fit"]["layers"])
            fit_better = row["final_fit"]["task_loss"] <= baseline["final_fit"]["task_loss"]
            probe_better = row["final_probe"]["task_loss"] <= baseline["final_probe"]["task_loss"]
            checks[arm][name] = {"loss_reduction": reduction, "all_layers_active": active,
                                  "fit_no_worse_than_baseline": fit_better, "probe_no_worse_than_baseline": probe_better,
                                  "ready_for_full_training_plan": complete and reduction >= .20 and active and fit_better and probe_better}
    admitted = [f"{arm}/{name}" for arm, variants in checks.items() for name, row in variants.items()
                if row["ready_for_full_training_plan"]]
    return {"checks": checks, "admitted_short_check_candidates": admitted,
            "hardware_mainline_changed": False, "snn_remains_project_target": True,
            "next": "plan_full_training_for_admitted_snn_with_baseline" if admitted else
                    ("stop_this_readout_protocol_review_evidence" if complete else "complete_short_checks"),
            "full_training_started": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=RUN_ROOT / "SNN-READOUT-v1")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--updates", type=int, choices=(3, 100), default=100)
    parser.add_argument("--arms", nargs="+", choices=("Local-SNN", "Dilated-SNN"), default=["Local-SNN", "Dilated-SNN"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-dev", action="store_true", help="full Train-dev inference for timing only; not admission")
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    output = args.output or directory / "readout.json"
    complete = args.updates == 100 and set(args.arms) == {"Local-SNN", "Dilated-SNN"}
    if not directory.is_relative_to(RUN_ROOT.resolve()) or directory == RUN_ROOT.resolve():
        raise ValueError("run directory must be inside ignored results/runs")
    if len(set(args.arms)) != len(args.arms) or output.exists() or (directory / "readout.json").exists():
        raise ValueError("duplicate arm or existing result; use a fresh run directory")
    if output.suffix != ".json" or (args.output and (args.output.resolve().parent != ROOT / "results" or not complete)):
        raise ValueError("formal output requires both architectures and 100 updates")
    source_path = ROOT / "results/temporal_segmentation_study.json"
    diagnostic_path = ROOT / "results/temporal_snn_diagnosis.json"
    source, diagnostic = read_json(source_path), read_json(diagnostic_path)
    if source["protocol"] != PROTOCOL or source["frozen_inputs"] != frozen_inputs() or diagnostic["source_sha256"] != sha256(source_path):
        raise ValueError("source provenance mismatch")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = choose_device(args.device)
    started = time.perf_counter()
    store = SequenceStore(directory / "cache", 1024)
    fit = collate(store, "train_main", diagnostic["fit_blocks"])
    probe = collate(store, "train_main", diagnostic["probe_blocks"])
    preparation = time.perf_counter() - started
    result = report("snn_readout_controls", directory)
    result.update(protocol=READOUT_PROTOCOL, data_role=["train_main", "train_dev"] if args.full_dev else ["train_main"],
                  pilot=not complete, source_sha256=sha256(source_path), diagnostic_sha256=sha256(diagnostic_path),
                  fit_blocks=diagnostic["fit_blocks"], probe_blocks=diagnostic["probe_blocks"],
                  frozen_inputs=source["frozen_inputs"], arms={}, preparation_s=preparation, updates=args.updates)
    result["environment"]["device"] = str(device)
    # Persist the fixed protocol before observing outcomes.
    write_json(directory / "protocol.json", result)
    timing = None
    if args.full_dev:
        blocks, _, _ = training_inventory(store)
        timing = preparation_timing(store, blocks)
        result["training_blocks"] = len(blocks)
        result["batch_preparation_timing"] = timing
    for arm in args.arms:
        result["arms"][arm] = {}
        for variant, kwargs in VARIANTS.items():
            model = TemporalSegmenter(arm, 42, tuple(source["preflight"]["priors"]), **kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            initial_fit = measure(model, fit, device)
            row = {"configuration": kwargs, "initial_fit": initial_fit,
                   "initial_probe": measure(model, probe, device, initial_fit["threshold"]), "history": []}
            training_started = time.perf_counter()
            for number in range(args.updates):
                loss, rates = update(model, fit, optimizer, device)
                if variant == "baseline":
                    expected = source["preflight"]["arms"][arm]["history"][number]
                    if abs(loss - expected["loss"]) > 1e-7 or not np.allclose(rates, expected["rates"], rtol=0, atol=1e-7):
                        raise AssertionError("baseline trajectory does not reproduce")
                row["history"].append({"update": number + 1, "loss": loss, "rates": rates})
            row["training_s"] = time.perf_counter() - training_started
            row["final_fit"] = measure(model, fit, device)
            row["final_probe"] = measure(model, probe, device, row["final_fit"]["threshold"])
            row["finite_loss_and_gradients"] = True
            extra = 32 * 65 * 87 if variant == "membrane" else 0
            row["cost"] = {**structural_cost(True), "extra_readout_state_values": extra,
                           "total_persistent_state_float32_bytes": (135720 + extra) * 4,
                           "extra_readout_leak_updates_per_100ms": extra * 4,
                           "readout_scale_runtime_operations_added": 0}
            if args.full_dev:
                row["dev_timing_pilot"] = evaluate_model(model, store, device, localization=False)
                assert timing is not None
                row["estimated_30_epoch_s"] = 30 * (((result["training_blocks"] + 7) // 8) *
                    (row["training_s"] / args.updates + timing["seconds_per_batch"]) + row["dev_timing_pilot"]["seconds"])
            checkpoint_save(directory / f"{arm}-{variant}.pt", {"model": model.state_dict(), "arm": arm, "variant": variant,
                            "model_kwargs": kwargs, "priors": source["preflight"]["priors"], "seed": 42,
                            "protocol": READOUT_PROTOCOL, "updates": args.updates})
            result["arms"][arm][variant] = row
            write_json(directory / "partial.json", result)
            print({"arm": arm, "variant": variant, "initial_loss": row["initial_fit"]["total_loss"],
                   "final_loss": row["final_fit"]["total_loss"], "probe_loss": row["final_probe"]["task_loss"]}, flush=True)
    result.update(status="ok", elapsed_s=time.perf_counter() - started, cache=store.stats)
    result["decision"] = decide(result["arms"], complete)
    result["primary_metrics"] = {arm: {v: {"initial_loss": row["initial_fit"]["total_loss"],
                                  "final_loss": row["final_fit"]["total_loss"], "probe_loss": row["final_probe"]["task_loss"]}
                                      for v, row in rows.items()} for arm, rows in result["arms"].items()}
    result["conclusion"] = result["decision"]["next"] + "; short checks do not establish full-data accuracy or FPGA efficiency"
    write_json(directory / "readout.json", result)
    if output.resolve() != (directory / "readout.json").resolve():
        write_json(output, result)
    print({"status": "ok", "decision": result["decision"], "elapsed_s": result["elapsed_s"]}, flush=True)


if __name__ == "__main__":
    main()
