"""Train the admitted Local-SNN readout pair from scratch, with fixed 30-epoch budgets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from scripts.study_snn_readout import READOUT_PROTOCOL, VARIANTS, decide
from wangdao.temporal_segmentation import TemporalSegmenter
from wangdao.temporal_study import (LOADING_IMPLEMENTATION, RUN_ROOT, SequenceStore, batches, checkpoint_load, checkpoint_save,
                                    choose_device, epoch_order, evaluate_model, frozen_inputs, read_json,
                                    report, sha256, training_inventory, update, write_json)

PAIR_PROTOCOL = {
    "version": 1, "arm": "Local-SNN", "variants": ["baseline", "membrane"], "seed": 42,
    "epochs": 30, "batch_size": 8, "max_formal_runs": 2, "initialization": "fresh seed42; never the 8-block pilot checkpoint",
    "optimizer": "AdamW", "learning_rate": 1e-3, "cosine_min_lr": 1e-5, "weight_decay": 1e-4,
    "gradient_clip": 5, "precision": "float32; no AMP/TF32", "prefetch_batches": 1,
    "selection": "full Train-dev every epoch; IoU,F1,earlier epoch,lower threshold",
    "readout_controls": READOUT_PROTOCOL,
    "endpoint": "paired full Train-dev results; no quantization or deployment admission without cost and reference review",
}


def run_variant(directory: Path, variant: str, store: SequenceStore, blocks, priors,
                device: torch.device, identity: dict, *, pilot: bool = False) -> dict:
    target = directory / variant
    path = target / "training.json"
    epochs = 2 if pilot else 30
    last = target / "last.pt"
    result = report("readout_pair_train", directory)
    result.update(protocol=PAIR_PROTOCOL, identity=identity, variant=variant, pilot=pilot, history=[], priors=list(priors),
                  model_kwargs=VARIANTS[variant], epochs=epochs,
                  backend={"device": str(device), "torch": str(torch.__version__), "threads": torch.get_num_threads()})
    if path.exists():
        saved_report = read_json(path)
        for key in ("protocol", "identity", "variant", "pilot", "priors", "model_kwargs", "epochs", "backend"):
            if saved_report[key] != result[key]:
                raise ValueError(f"resume identity mismatch: {key}")
        result = saved_report
        if result["status"] == "ok":
            return result
    model = TemporalSegmenter("Local-SNN", 42, tuple(priors), **VARIANTS[variant]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    if last.exists():
        saved = checkpoint_load(last, device)
        if saved["report"]["identity"] != identity or saved["report"]["protocol"] != PAIR_PROTOCOL:
            raise ValueError("resume checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        result = saved["report"]
    elif result["history"]:
        raise ValueError("history exists but resume checkpoint is absent")
    write_json(path, result)
    for epoch in range(len(result["history"]) + 1, epochs + 1):
        epoch_started = time.perf_counter()
        waiting = total_loss = 0.0
        samples = steps = 0
        model.train()
        iterator = iter(batches(store, epoch_order(blocks, 42, epoch)))
        while True:
            begin = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            waiting += time.perf_counter() - begin
            loss, _ = update(model, batch, optimizer, device)
            count = len(batch[2])
            total_loss += loss * count
            samples += count
            steps += 1
            if steps % 100 == 0:
                print({"variant": variant, "epoch": epoch, "step": steps,
                       "total_steps": (len(blocks) + 7) // 8,
                       "train_s": round(time.perf_counter() - epoch_started, 2),
                       "input_wait_s": round(waiting, 2), "loader": LOADING_IMPLEMENTATION}, flush=True)
        train_s = time.perf_counter() - epoch_started
        evaluation = evaluate_model(model, store, device, localization=False)
        entry = {"epoch": epoch, "loss": total_loss / samples, "updates": steps, "blocks": len(blocks),
                 "events": samples, "input_wait_s": waiting, "train_s": train_s,
                 "loader": LOADING_IMPLEMENTATION,
                 "lr": optimizer.param_groups[0]["lr"], "selection": evaluation}
        result["history"].append(entry)
        key = [evaluation["event"]["iou"], evaluation["event"]["f1"], -epoch, -evaluation["threshold"]]
        if "best" not in result or key > result["best"]["key"]:
            result["best"] = {"key": key, "epoch": epoch, "threshold": evaluation["threshold"], "event": evaluation["event"]}
            checkpoint_save(target / "best.pt", {"model": model.state_dict(), "variant": variant,
                            "model_kwargs": VARIANTS[variant], "priors": list(priors), "epoch": epoch,
                            "threshold": evaluation["threshold"], "protocol": PAIR_PROTOCOL, "identity": identity})
        scheduler.step()
        result["cache"] = store.stats.copy()
        checkpoint_save(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "report": result})
        write_json(path, result)
        print({"variant": variant, "epoch": epoch, "loss": entry["loss"], "iou": evaluation["event"]["iou"],
               "input_wait_s": waiting, "train_s": train_s, "selection_s": evaluation["seconds"],
               "seconds": train_s + evaluation["seconds"]}, flush=True)
    best = checkpoint_load(target / "best.pt", device)
    model.load_state_dict(best["model"])
    result["evaluation"] = evaluate_model(model, store, device, best["threshold"])
    if result["evaluation"]["event"] != result["best"]["event"]:
        raise AssertionError("best checkpoint does not reproduce its selection metrics")
    result.update(status="ok", primary_metrics=result["evaluation"]["event"],
                  conclusion="Fixed-budget training complete; checkpoint selected on Train-dev only.")
    write_json(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=RUN_ROOT / "SNN-READOUT-PAIR-v1")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--pilot", action="store_true", help="2 epochs of 16 Train-main blocks; never formal evidence")
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    if not directory.is_relative_to(RUN_ROOT.resolve()) or directory == RUN_ROOT.resolve():
        raise ValueError("run directory must be inside ignored results/runs")
    short_path = ROOT / "results/snn_readout_study.json"
    short = read_json(short_path)
    if short["protocol"] != READOUT_PROTOCOL or short["status"] != "ok" or short["pilot"] or short["updates"] != 100:
        raise ValueError("complete matching readout controls are required")
    admission = decide(short["arms"], True)
    if admission != short["decision"] or "Local-SNN/membrane" not in admission["admitted_short_check_candidates"]:
        raise ValueError("Local membrane readout was not admitted")
    frozen = frozen_inputs()
    if frozen != short["frozen_inputs"]:
        raise ValueError("frozen input mismatch")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = choose_device(args.device)
    identity = {"short_study_sha256": sha256(short_path), "frozen_inputs": frozen}
    store = SequenceStore(directory / "cache", 1024)
    started = time.perf_counter()
    blocks, priors, _ = training_inventory(store)
    if args.pilot:
        blocks = epoch_order(blocks, 42, 1)[:16]
    result: dict[str, Any] = report("readout_pair", directory)
    result.update(protocol=PAIR_PROTOCOL, identity=identity, pilot=args.pilot, variants={})
    for variant in ("baseline", "membrane"):
        result["variants"][variant] = run_variant(directory, variant, store, blocks, priors, device, identity, pilot=args.pilot)
        write_json(directory / "pair.partial.json", result)
    baseline, candidate = result["variants"]["baseline"], result["variants"]["membrane"]
    result.update(status="ok", elapsed_s=time.perf_counter() - started,
                  primary_metrics={v: r["evaluation"] for v, r in result["variants"].items()},
                  decision={"hardware_mainline_changed": False, "quantization_admitted": False,
                            "iou_gain_vs_paired_baseline": candidate["evaluation"]["event"]["iou"] - baseline["evaluation"]["event"]["iou"],
                            "next": "review_formal_accuracy_and_cost_before_further_training" if not args.pilot else "pilot_only"},
                  conclusion="Two SNN readout variants compared; no deployment replacement or FPGA efficiency claim.")
    write_json(directory / "pair.json", result)
    print({"status": "ok", "pilot": args.pilot, "decision": result["decision"], "elapsed_s": result["elapsed_s"],
           "output": str(directory / "pair.json")}, flush=True)


if __name__ == "__main__":
    main()
