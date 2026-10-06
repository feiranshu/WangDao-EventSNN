"""Audit the completed readout pair and stop early when application accuracy excludes both paths."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from scripts.train_snn_readout import PAIR_PROTOCOL, VARIANTS
from wangdao.temporal_segmentation import TemporalSegmenter, structural_cost
from wangdao.temporal_study import (PROTOCOL, RUN_ROOT, SequenceStore, checkpoint_load, evaluate_model,
                                    frozen_inputs, read_json, report, sha256, training_inventory, write_json)


def selection_key(row: dict) -> list:
    evaluation = row["selection"]
    return [evaluation["event"]["iou"], evaluation["event"]["f1"], -row["epoch"], -evaluation["threshold"]]


def audit_history(record: dict, variant: str, identity: dict, blocks: int, events: int,
                  priors: tuple, dev_files: list[str], dev_events: int, dev_targets: int) -> dict:
    expected = {"status": "ok", "pilot": False, "epochs": 30, "variant": variant,
                "protocol": PAIR_PROTOCOL, "identity": identity, "model_kwargs": VARIANTS[variant], "priors": list(priors)}
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError(f"training record mismatch: {key}")
    history = record["history"]
    if len(history) != 30:
        raise ValueError("30 complete epochs required")
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or row["blocks"] != blocks or row["events"] != events or row["updates"] != (blocks + 7) // 8:
            raise ValueError("epoch order or sample/update budget mismatch")
        lr = 1e-5 + (1e-3 - 1e-5) * (1 + math.cos(math.pi * (epoch - 1) / 30)) / 2
        if not math.isclose(row["lr"], lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("learning-rate schedule mismatch")
        if not math.isfinite(row["loss"]) or row["loss"] < 0 or not 0 <= row["input_wait_s"] <= row["train_s"]:
            raise ValueError("invalid loss or training timing")
        selection = row["selection"]
        scan, event = selection["scan"], selection["event"]
        if selection["files"] != dev_files or scan["best_event"] != event or scan["best_threshold"] != selection["threshold"]:
            raise ValueError("full Train-dev selection mismatch")
        if [scan[k] for k in ("grid_min", "grid_max", "grid_step", "evaluated_thresholds")] != [.01, .99, .01, 99]:
            raise ValueError("threshold scan protocol mismatch")
        if sum(event[k] for k in ("true_positive", "false_positive", "false_negative", "true_negative")) != dev_events:
            raise ValueError("selection event count mismatch")
        if event["true_positive"] + event["false_negative"] != dev_targets:
            raise ValueError("selection foreground count mismatch")
    best = max(history, key=selection_key)
    if record["best"] != {"key": selection_key(best), "epoch": best["epoch"],
                          "threshold": best["selection"]["threshold"], "event": best["selection"]["event"]}:
        raise ValueError("checkpoint selection rule mismatch")
    if record["evaluation"]["event"] != best["selection"]["event"] or record["primary_metrics"] != record["evaluation"]["event"]:
        raise ValueError("final metric mismatch")
    return best


def accuracy_decision(evaluations: dict, reference: dict) -> dict:
    checks = {}
    for name, evaluation in evaluations.items():
        gain = evaluation["event"]["iou"] - reference["event"]["iou"]
        pd_gain = evaluation["localization"]["pd"] - reference["localization"]["pd"]
        fa = evaluation["localization"]["fa"]
        reference_fa = reference["localization"]["fa"]
        localization = pd_gain >= -PROTOCOL["cost_gate"]["pd_loss"] and fa <= reference_fa * PROTOCOL["cost_gate"]["fa_ratio"]
        checks[name] = {"iou_gain": gain, "pd_gain": pd_gain,
                        "fa_ratio": fa / reference_fa if reference_fa else None,
                        "localization_pass": localization,
                        "cost_path_iou_necessary_condition": gain >= -PROTOCOL["cost_gate"]["iou_loss"],
                        "accuracy_path_iou_necessary_condition": gain >= PROTOCOL["accuracy_gate"]["iou_gain"]}
    # Best(H2, hybrid) cannot have lower IoU than this verified frozen hybrid reference.
    excluded = all(not row["cost_path_iou_necessary_condition"] and not row["accuracy_path_iou_necessary_condition"]
                   for row in checks.values())
    return {"checks_vs_frozen_hybrid": checks, "both_paths_excluded_by_iou": excluded,
            "reference_bound": "max(H2, frozen hybrid) IoU >= verified frozen hybrid IoU",
            "quantization_admitted": False, "additional_training_authorized": False,
            "hardware_mainline_changed": False,
            "next": "stop_readout_pair_retain_snn_mainline" if excluded else "review_full_references_and_cost",
            "scope": "This fixed seed42 readout pair; not all SNN architectures."}


def timing_summary(history: list[dict]) -> dict:
    grouped = {}
    for name in sorted({row.get("loader", "original") for row in history}):
        rows = [r for r in history if r.get("loader", "original") == name]
        grouped[name] = {"epochs": [r["epoch"] for r in rows],
                         "median_train_s": statistics.median(r["train_s"] for r in rows),
                         "median_input_wait_s": statistics.median(r["input_wait_s"] for r in rows),
                         "median_selection_s": statistics.median(r["selection"]["seconds"] for r in rows),
                         "median_epoch_train_plus_selection_s": statistics.median(r["train_s"] + r["selection"]["seconds"] for r in rows)}
    return {"by_loader": grouped,
            "sum_completed_epoch_train_and_selection_s": sum(r["train_s"] + r["selection"]["seconds"] for r in history),
            "scope": "Recorded complete epochs only; excludes interrupted work and restart preparation; no controlled architecture timing."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=RUN_ROOT / "SNN-READOUT-PAIR-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/snn_readout_pair_selection.json")
    args = parser.parse_args()
    directory, output = args.run_directory.resolve(), args.output.resolve()
    if not directory.is_relative_to(RUN_ROOT.resolve()) or directory == RUN_ROOT.resolve():
        raise ValueError("run directory must be inside results/runs")
    if output.parent != ROOT / "results" or output.suffix != ".json" or output.exists():
        raise ValueError("output must be a new formal JSON directly in results")
    pair_path, reference_path = directory / "pair.json", ROOT / "results/refiner_study.json"
    pair, historical = read_json(pair_path), read_json(reference_path)
    short_path = ROOT / "results/snn_readout_study.json"
    identity = {"short_study_sha256": sha256(short_path), "frozen_inputs": frozen_inputs()}
    if pair["status"] != "ok" or pair["pilot"] or pair["protocol"] != PAIR_PROTOCOL or pair["identity"] != identity:
        raise ValueError("completed formal pair with matching provenance required")
    if historical["status"] != "ok" or historical["frozen_inputs"] != identity["frozen_inputs"]:
        raise ValueError("historical reference provenance mismatch")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    store = SequenceStore(directory / "cache", 1024)
    blocks, priors, _ = training_inventory(store)
    train_events = sum(len(store.get("train_main", p.name).events) for p in store.paths["train_main"])
    dev_files = [p.name for p in store.paths["train_dev"]]
    dev_records = [store.get("train_dev", name) for name in dev_files]
    dev_events = sum(len(r.events) for r in dev_records)
    dev_targets = sum(int(r.events["label"].sum()) for r in dev_records)
    selected = historical["baselines"]["frozen_ann"]["selected"]
    if [r["file"] for r in selected["per_sequence"]] != dev_files or historical["data"]["dev_events"] != dev_events or historical["data"]["dev_target_events"] != dev_targets:
        raise ValueError("reference data coverage mismatch")
    if [selected["scan"][k] for k in ("grid_min", "grid_max", "grid_step", "evaluated_thresholds")] != [.01, .99, .01, 99]:
        raise ValueError("reference threshold scan mismatch")
    reference = {"event": selected["scan"]["best_event"], "localization": selected["localization"]}
    result = report("readout_pair_selection", directory)
    result.update(protocol=PAIR_PROTOCOL, identity=identity, variants={},
                  reference={"kind": "frozen S0-P + H2 Refiner", "source": str(reference_path.relative_to(ROOT)),
                             "sha256": sha256(reference_path), "selected": reference,
                             "original_threshold": {"event": selected["fixed_event"], "localization": selected["fixed_localization"]},
                             "context": "offline application reference, not a causal ANN/SNN control; verified historical metrics reused"},
                  data={"train_blocks": len(blocks), "train_events": train_events, "dev_files": dev_files,
                        "dev_events": dev_events, "dev_targets": dev_targets},
                  source_pair_sha256=sha256(pair_path), last_invocation_elapsed_s=pair["elapsed_s"])
    for name in ("baseline", "membrane"):
        target = directory / name
        record = read_json(target / "training.json")
        if record != pair["variants"][name]:
            raise ValueError("pair summary and standalone training report disagree")
        best = audit_history(record, name, identity, len(blocks), train_events, priors, dev_files, dev_events, dev_targets)
        device = torch.device(record["backend"]["device"])
        if record["backend"] != {"device": str(device), "torch": str(torch.__version__), "threads": 1}:
            raise ValueError("audit must use original training backend")
        saved, last = checkpoint_load(target / "best.pt", device), checkpoint_load(target / "last.pt", device)
        for key in ("variant", "model_kwargs", "priors", "protocol", "identity"):
            if saved[key] != record[key]:
                raise ValueError(f"best checkpoint mismatch: {key}")
        if saved["epoch"] != best["epoch"] or saved["threshold"] != best["selection"]["threshold"]:
            raise ValueError("best checkpoint epoch/threshold mismatch")
        if last["report"]["history"] != record["history"] or last["report"]["identity"] != identity or last["scheduler"]["last_epoch"] != 30:
            raise ValueError("last checkpoint history/scheduler mismatch")
        expected_steps = sum(r["updates"] for r in record["history"])
        if not last["optimizer"]["state"] or not all(int(s["step"].item()) == expected_steps for s in last["optimizer"]["state"].values()):
            raise ValueError("optimizer update count mismatch")
        model = TemporalSegmenter("Local-SNN", 42, priors, **saved["model_kwargs"]).to(device)
        model.load_state_dict(saved["model"], strict=True)
        evaluation = evaluate_model(model, store, device, saved["threshold"])
        for key in ("event", "localization", "per_sequence", "files", "scan", "threshold"):
            if evaluation[key] != record["evaluation"][key]:
                raise ValueError(f"selected checkpoint does not reproduce {key}")
        extra = 32 * 65 * 87 if name == "membrane" else 0
        result["variants"][name] = {"epoch": saved["epoch"], "threshold": saved["threshold"], "evaluation": evaluation,
                                    "checkpoint": str((target / "best.pt").relative_to(ROOT)), "checkpoint_sha256": sha256(target / "best.pt"),
                                    "complete_epochs": 30, "optimizer_updates": expected_steps,
                                    "history": record["history"], "timing": timing_summary(record["history"]),
                                    "structure": {**structural_cost(True), "extra_output_state_values": extra,
                                                  "persistent_float32_state_bytes": (135720 + extra) * 4,
                                                  "extra_readout_leak_updates_per_100ms": extra * 4}}
        print({"variant": name, "checkpoint_reproduced": True, "iou": evaluation["event"]["iou"]}, flush=True)
    evaluations = {name: r["evaluation"] for name, r in result["variants"].items()}
    decision = accuracy_decision(evaluations, reference)
    gain = evaluations["membrane"]["event"]["iou"] - evaluations["baseline"]["event"]["iou"]
    if gain != pair["decision"]["iou_gain_vs_paired_baseline"]:
        raise ValueError("pair decision delta mismatch")
    decision["membrane_iou_gain_vs_baseline"] = gain
    result.update(status="ok", primary_metrics={n: {"event": e["event"], "localization": e["localization"]} for n, e in evaluations.items()},
                  decision=decision, conclusion=decision["next"],
                  cost_evaluation={"status": "not_run", "reason": "Both candidate IoUs exclude either original promotion path regardless of software speed."
                                   if decision["both_paths_excluded_by_iou"] else "Full reference and software cost review remains required.",
                                   "fpga_efficiency_claim": False}, cache=store.stats.copy())
    write_json(output, result)
    print({"status": "ok", "output": str(output), "decision": decision}, flush=True)


if __name__ == "__main__":
    main()
