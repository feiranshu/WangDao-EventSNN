"""S3 matched-objective research ANN / fixed-position SNN training.

Formal tasks are single-model, append-only runs. No held-out role is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
S2 = HERE.parent / "s2"
sys.path.insert(0, str(S2))
from core import ResearchANN, ResearchSNN, class_logits  # noqa: E402
from train import average_precision, read_index, role_records, windows  # noqa: E402
from runtime import DeviceSNNRunner  # noqa: E402

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
WEIGHT = (1.0, 3.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def training_groups(records):
    group = []
    for record in records:
        for segment in windows(record):
            group.append((record, segment))
            if len(group) == 8:
                yield group
                group = []
    if group:
        yield group


def update(optimizer, model, weighted, mass):
    optimizer.zero_grad()
    (weighted / mass).backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
    optimizer.step()
    return norm


def train_ann(model, optimizer, records, device, weight):
    model.train()
    plain_sum = weighted_sum = mass_sum = 0.0
    seen = steps = clipped = 0
    max_norm = 0.0
    for group in training_groups(records):
        features = np.concatenate([record["features"][segment] for record, segment in group])
        targets = np.concatenate([record["target"][segment] for record, segment in group])
        x = torch.as_tensor(features.astype(np.float32) / 255.0, device=device)
        y = torch.as_tensor(targets.astype(np.int64), device=device)
        ce = F.cross_entropy(class_logits(model(x)), y, reduction="none")
        weighted = (ce * weight[y]).sum()
        mass = weight[y].sum()
        norm = update(optimizer, model, weighted, mass)
        plain_sum += float(ce.detach().sum().cpu())
        weighted_sum += float(weighted.detach().cpu())
        mass_sum += float(mass.detach().cpu())
        seen += len(targets)
        steps += 1
        clipped += int(norm > 5.0)
        max_norm = max(max_norm, norm)
    return {"candidate_exposures": seen, "nonempty_windows": sum(len(list(windows(r))) for r in records),
            "optimizer_updates": steps, "clipped_updates": clipped,
            "gradient_norm_max_before_clip": max_norm,
            "unweighted_ce": plain_sum / seen,
            "weighted_ce_global_mass": weighted_sum / mass_sum,
            "weighted_loss_numerator": weighted_sum, "label_weight_mass": mass_sum}


def train_snn(model, optimizer, records, device, weight, mode):
    model.train()
    plain_sum = weighted_sum = mass_sum = 0.0
    seen = steps = clipped = nonempty = packed_slots = 0
    spikes = np.zeros(3, dtype=np.int64)
    max_norm = 0.0
    pending_weighted = pending_plain = pending_mass = None
    pending_windows = 0
    for record in records:
        runner = DeviceSNNRunner(model, record["t0"], mode=mode)
        for segment in windows(record):
            weighted, plain, mass, n = runner.run_window(
                record["features"][segment], record["x"][segment],
                record["y"][segment], record["t"][segment],
                record["target"][segment], weight)
            pending_weighted = weighted if pending_weighted is None else pending_weighted + weighted
            pending_plain = plain if pending_plain is None else pending_plain + plain
            pending_mass = mass if pending_mass is None else pending_mass + mass
            pending_windows += 1
            seen += n
            if pending_windows == 8:
                norm = update(optimizer, model, pending_weighted, pending_mass)
                plain_sum += float(pending_plain.detach().cpu())
                weighted_sum += float(pending_weighted.detach().cpu())
                mass_sum += float(pending_mass.detach().cpu())
                steps += 1
                clipped += int(norm > 5.0)
                max_norm = max(max_norm, norm)
                pending_weighted = pending_plain = pending_mass = None
                pending_windows = 0
        counters = runner.counters()
        nonempty += counters["nonempty_windows"]
        packed_slots += counters["packed_slots"]
        spikes += [counters["spikes_l1"], counters["spikes_l2"], counters["spikes_out"]]
    if pending_weighted is not None:
        norm = update(optimizer, model, pending_weighted, pending_mass)
        plain_sum += float(pending_plain.detach().cpu())
        weighted_sum += float(pending_weighted.detach().cpu())
        mass_sum += float(pending_mass.detach().cpu())
        steps += 1
        clipped += int(norm > 5.0)
        max_norm = max(max_norm, norm)
    return {"candidate_exposures": seen, "nonempty_windows": nonempty,
            "optimizer_updates": steps, "clipped_updates": clipped,
            "gradient_norm_max_before_clip": max_norm,
            "unweighted_ce": plain_sum / seen,
            "weighted_ce_global_mass": weighted_sum / mass_sum,
            "weighted_loss_numerator": weighted_sum, "label_weight_mass": mass_sum,
            "spike_fraction_l1": int(spikes[0]) / (seen * 48),
            "spike_fraction_l2": int(spikes[1]) / (seen * 32),
            "spike_fraction_out": int(spikes[2]) / (seen * 4),
            "packed_slots": packed_slots, "valid_slot_fraction": seen / packed_slots}


def evaluate(model, records, kind, mode, device):
    model.eval()
    started = time.perf_counter()
    labels_all, probability_all = [], []
    per_sequence = []
    counters = {key: 0 for key in ("candidate_events", "nonempty_windows", "packed_slots",
                                    "input_nonzero", "spikes_l1", "spikes_l2", "spikes_out")}
    ann_effective_mac = 0
    with torch.no_grad():
        for record in records:
            features = record["features"]
            probability = np.empty(len(features), dtype=np.float32)
            if kind == "ann":
                for start in range(0, len(features), 4096):
                    stop = min(start + 4096, len(features))
                    x = torch.as_tensor(features[start:stop].astype(np.float32) / 255.0, device=device)
                    a1 = F.hardtanh(model.fc1(x), 0, 1)
                    a2 = F.hardtanh(model.fc2(a1), 0, 1)
                    logits = model.fc3(a2)
                    probability[start:stop] = torch.softmax(class_logits(logits), 1)[:, 1].cpu().numpy()
                    ann_effective_mac += int(torch.count_nonzero(x)) * 48 + int(torch.count_nonzero(a1)) * 32 + int(torch.count_nonzero(a2)) * 4
            else:
                runner = DeviceSNNRunner(model, record["t0"], mode=mode)
                for segment in windows(record):
                    probability[segment] = runner.run_window(features[segment], record["x"][segment],
                                                              record["y"][segment], record["t"][segment])
                one = runner.counters()
                for key in counters:
                    counters[key] += one[key]
            full = np.zeros(len(record["labels"]), dtype=np.float32)
            full[record["indices"]] = probability
            labels_all.append(record["labels"].astype(bool))
            probability_all.append(full)
            per_sequence.append({"name": record["name"], "event_rate_eps": record["event_rate"],
                                 "high_density": bool(record["event_rate"] >= 20000),
                                 "events": len(full), "target_events": int(record["labels"].sum()),
                                 "candidate_events": len(probability),
                                 "target_candidates": int(record["target"].sum()),
                                 "probability": probability})
    labels = np.concatenate(labels_all)
    full_prob = np.concatenate(probability_all)
    curve = {}
    for threshold in THRESHOLDS:
        pred = full_prob >= threshold
        tp = int(np.count_nonzero(pred & labels))
        fp = int(np.count_nonzero(pred & ~labels))
        fn = int(np.count_nonzero(~pred & labels))
        high_tp = high_fn = high_fp = 0
        for record, row in zip(records, per_sequence):
            if row["high_density"]:
                y = record["target"].astype(bool)
                p = row["probability"] >= threshold
                high_tp += int(np.count_nonzero(p & y))
                high_fp += int(np.count_nonzero(p & ~y))
                high_fn += int(record["labels"].sum()) - int(np.count_nonzero(p & y))
        curve[str(threshold)] = {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "high_density_tp": high_tp, "high_density_fp": high_fp, "high_density_fn": high_fn,
            "high_density_recall": high_tp / (high_tp + high_fn) if high_tp + high_fn else None}
    selected = max(THRESHOLDS, key=lambda value: (
        curve[str(value)]["iou"], curve[str(value)]["high_density_recall"], -value))
    candidate_events = sum(row["candidate_events"] for row in per_sequence)
    target_events = int(labels.sum())
    target_candidates = sum(row["target_candidates"] for row in per_sequence)
    closed_windows = sum(record["closed_windows"] for record in records)
    cost = {"frozen_proposal_dense_mac": closed_windows * 2580480,
            "refiner_dense_mac": candidate_events * 3392,
            "state_bytes": 2035800 if kind == "snn" else 0}
    if kind == "ann":
        cost.update({"refiner_effective_mac_skip_zero": ann_effective_mac, "refiner_binary_ac": 0})
    else:
        cost.update(counters)
        cost.update({"refiner_effective_input_mac": counters["input_nonzero"] * 48,
                     "refiner_binary_ac": counters["spikes_l1"] * 32 + counters["spikes_l2"] * 4,
                     "state_decay_multiplications_upper": candidate_events * 88,
                     "threshold_comparisons": candidate_events * 84,
                     "readout_multiplications": candidate_events * 8,
                     "state_channel_reads": candidate_events * 88,
                     "state_channel_writes": candidate_events * 88,
                     "valid_slot_fraction": candidate_events / counters["packed_slots"]})
    for row in per_sequence:
        row.pop("probability")
    return {"events": len(labels), "target_events": target_events,
            "candidate_events": candidate_events, "candidate_fraction": candidate_events / len(labels),
            "proposal_target_recall": target_candidates / target_events,
            "average_precision": average_precision(labels, full_prob),
            "threshold_curve": curve, "selected_threshold": selected,
            "selected": curve[str(selected)], "fixed_0.35": curve["0.35"],
            "refiner_conditional_recall_selected": curve[str(selected)]["tp"] / target_candidates,
            "per_sequence": per_sequence, "cost": cost,
            "refiner_replay_wall_s": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ann", "snn"), required=True)
    parser.add_argument("--mode", choices=("persistent", "window_reset"), default="persistent")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.model == "ann" and args.mode != "persistent":
        raise ValueError("ANN mode must be persistent")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if args.smoke and args.resume:
        raise ValueError("smoke cannot resume")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"append-only output already exists: {args.output}")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device(args.device)
    index, manifest = read_index()
    train = role_records(index, manifest, "train_main", only_name="train_000.npz" if args.smoke else "")
    dev = role_records(index, manifest, "train_dev", only_name="train_002.npz" if args.smoke else "")
    if not args.smoke:
        assert len(train) == 79 and len(dev) == 10
    experiment_id = "GEN2-S3-PILOT" if args.smoke else ("GEN2-S3-002" if args.mode == "window_reset" else "GEN2-S3-001")
    provenance = {"experiment_id": experiment_id, "status": "pilot_not_stage_result" if args.smoke else "formal_training",
                  "model": args.model, "mode": args.mode, "seed": 42, "epochs": args.epochs,
                  "train_sequences": len(train), "dev_sequences": len(dev),
                  "train_candidate_events_per_epoch": sum(len(r["features"]) for r in train),
                  "index_sha256": manifest["index_sha256"],
                  "teacher_int8_sha256": manifest["teacher_int8_sha256"],
                  "feature_protocol_sha256": manifest["protocol_sha256"],
                  "s3_protocol_sha256": sha256(HERE / "PROTOCOL.md"),
                  "s2_core_sha256": sha256(S2 / "core.py"),
                  "s2_loader_sha256": sha256(S2 / "train.py"),
                  "s3_runtime_sha256": sha256(HERE / "runtime.py"),
                  "s3_train_sha256": sha256(Path(__file__).absolute()),
                  "torch_version": torch.__version__, "numpy_version": np.__version__,
                  "device": str(device), "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  "threads": torch.get_num_threads(),
                  "objective": "sum(weight[y] * CE) / sum(weight[y]) per 8 nonempty-window update",
                  "class_weights": list(WEIGHT)}
    model = (ResearchANN() if args.model == "ann" else ResearchSNN()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if args.model == "ann" else 5e-4,
                                  weight_decay=1e-4)
    history = []
    start_epoch = 1
    if args.resume:
        old = json.loads((args.output / "provenance.json").read_text(encoding="utf-8"))
        if old != provenance:
            raise RuntimeError("resume provenance mismatch")
        checkpoints = sorted(args.output.glob("epoch_*.pt"))
        if checkpoints:
            checkpoint = torch.load(checkpoints[-1], map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            history = checkpoint["history"]
            start_epoch = checkpoint["epoch"] + 1
            atomic_json(args.output / "history.json", history)
        print(f"resume from epoch {start_epoch}", flush=True)
    else:
        args.output.mkdir(parents=True)
        atomic_json(args.output / "provenance.json", provenance)
        (args.output / "PROTOCOL.lock.md").write_bytes((HERE / "PROTOCOL.md").read_bytes())
    weight = torch.tensor(WEIGHT, device=device)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        training = (train_ann(model, optimizer, train, device, weight) if args.model == "ann" else
                    train_snn(model, optimizer, train, device, weight, args.mode))
        training["wall_s"] = time.perf_counter() - started
        evaluation = evaluate(model, dev, args.model, args.mode, device)
        item = {"epoch": epoch, "train": training, "train_dev": evaluation}
        history.append(item)
        checkpoint = {"epoch": epoch, "model_kind": args.model, "mode": args.mode,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "history": history, "provenance": provenance}
        path = args.output / f"epoch_{epoch:03d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        atomic_json(args.output / "history.json", history)
        print(json.dumps({"epoch": epoch, "model": args.model, "mode": args.mode,
                          "weighted_ce": training["weighted_ce_global_mass"],
                          "unweighted_ce": training["unweighted_ce"],
                          "updates": training["optimizer_updates"],
                          "dev_ap": evaluation["average_precision"],
                          "dev_iou": evaluation["selected"]["iou"],
                          "threshold": evaluation["selected_threshold"],
                          "train_wall_s": training["wall_s"]}), flush=True)
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "epochs": args.epochs, "model": args.model, "mode": args.mode}), flush=True)


if __name__ == "__main__":
    main()
