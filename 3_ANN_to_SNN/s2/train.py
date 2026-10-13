"""Train and evaluate matched ANN/SNN research Refiner skeletons on frozen C1 caches.

Default: full Train-main 79 sequences, Train-dev 10 sequences, seed42, five epochs.
Pilot limits are explicitly marked non-stage results and cannot satisfy S2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from core import ResearchANN, ResearchSNN, SNNRunner, class_logits, CELLS, STATE_CHANNELS


HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
H2 = ROOT / "1_H2_model_improvement"
STUDY = H2 / "seven_step_improvement_study" / "H2-pro"
INDEX = H2 / "implementation" / "data" / "index.json"
CACHE = STUDY / "cache"
WEIGHT = STUDY / "frozen" / "H2-pro-Causal-ANN-v1" / "teacher_exact_int8.npz"
THRESHOLDS = [round(0.2 + 0.05 * i, 2) for i in range(9)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_index():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    manifest = json.loads((WEIGHT.parent / "model_manifest.json").read_text(encoding="utf-8"))
    assert sha256(INDEX) == manifest["index_sha256"]
    assert sha256(WEIGHT) == manifest["teacher_int8_sha256"]
    assert sha256(STUDY / "protocol.json") == manifest["protocol_sha256"]
    return index, manifest


def role_records(index: dict, manifest: dict, role: str, limit: int = 0, only_name: str = ""):
    if role not in ("train_main", "train_dev"):
        raise ValueError("only development roles are permitted")
    names = index["roles"][role][:limit or None]
    if only_name:
        if only_name not in index["roles"][role]:
            raise ValueError(f"{only_name} is not in {role}")
        names = [only_name]
    by_name = {Path(row["path"]).name: row for row in index["files"]}
    records = []
    for name in names:
        source = H2 / "implementation" / "data" / "EV-UAV-dataset" / "train" / name
        assert sha256(source) == by_name[name]["sha256"], f"source data changed: {name}"
        with np.load(source, allow_pickle=False) as archive:
            full_t = archive["ev"]["t"]
            event_rate = len(full_t) * 1000.0 / max(float(full_t[-1] - full_t[0]), 1e-9)
            closed_windows = int(np.floor((float(full_t[-1] - full_t[0])) / 100.0))
        path = CACHE / role / (Path(name).stem + "_C1-proposal-motion.npz")
        with np.load(path, allow_pickle=False) as archive:
            assert str(archive["protocol_sha256"].item()) == manifest["protocol_sha256"]
            assert str(archive["source_sha256"].item()) == by_name[name]["sha256"]
            features = archive["features"].copy()
            indices = archive["indices"].copy()
            labels = archive["labels"].copy()
            x = archive["x"].copy()[indices]
            y = archive["y"].copy()[indices]
            t = archive["t"].copy()[indices]
            t0 = float(archive["t0"].item())
        assert features.dtype == np.uint8 and features.shape == (len(indices), 36)
        assert len(indices) == 0 or (indices[0] >= 0 and indices[-1] < len(labels) and np.all(np.diff(indices) > 0))
        assert np.all(np.diff(t) >= 0)
        records.append({"name": name, "features": features, "indices": indices,
                        "labels": labels, "target": labels[indices], "x": x, "y": y,
                        "t": t, "t0": t0, "event_rate": event_rate,
                        "closed_windows": closed_windows})
    return records


def windows(record):
    t = record["t"]
    if not len(t):
        return
    bins = np.floor((t - record["t0"]) / 100.0).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
    ends = np.r_[starts[1:], len(t)]
    for start, end in zip(starts, ends):
        yield slice(int(start), int(end))


def average_precision(labels: np.ndarray, probability: np.ndarray) -> float | None:
    order = np.argsort(-probability, kind="stable")
    p, y = probability[order], labels[order].astype(bool)
    if not len(p) or not np.any(y):
        return None
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    recall = tp / int(y.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * tp / (ends + 1)))


def evaluate(model, records, kind: str):
    model.eval()
    started = time.perf_counter()
    all_labels, all_prob = [], []
    per_sequence = []
    total_cost = None
    ann_effective_mac = 0
    for record in records:
        features = record["features"]
        candidate_prob = np.empty(len(features), dtype=np.float32)
        if kind == "ann":
            with torch.no_grad():
                for start in range(0, len(features), 4096):
                    end = min(start + 4096, len(features))
                    x = torch.from_numpy(features[start:end].astype(np.float32) / 255.0)
                    a1 = F.hardtanh(model.fc1(x), 0, 1)
                    a2 = F.hardtanh(model.fc2(a1), 0, 1)
                    logits = model.fc3(a2)
                    candidate_prob[start:end] = torch.softmax(class_logits(logits), dim=1)[:, 1].numpy()
                    ann_effective_mac += (int(torch.count_nonzero(x)) * 48 +
                                          int(torch.count_nonzero(a1)) * 32 +
                                          int(torch.count_nonzero(a2)) * 4)
        else:
            runner = SNNRunner(model, record["t0"])
            with torch.no_grad():
                for segment in windows(record):
                    _, value = runner.run_window(features[segment], record["x"][segment],
                                                 record["y"][segment], record["t"][segment])
                    candidate_prob[segment] = value
            cost = runner.cost()
            if total_cost is None:
                total_cost = {key: value for key, value in cost.items() if key not in ("voltage_and_trace_bytes", "timestamp_bytes")}
                total_cost["voltage_and_trace_bytes"] = cost["voltage_and_trace_bytes"]
                total_cost["timestamp_bytes"] = cost["timestamp_bytes"]
            else:
                for key, value in cost.items():
                    if key not in ("voltage_and_trace_bytes", "timestamp_bytes"):
                        total_cost[key] += value
        probability = np.zeros(len(record["labels"]), dtype=np.float32)
        probability[record["indices"]] = candidate_prob
        labels = record["labels"].astype(bool)
        all_labels.append(labels)
        all_prob.append(probability)
        per_sequence.append({"name": record["name"], "event_rate_eps": record["event_rate"],
                             "high_density": record["event_rate"] >= 20000,
                             "events": len(labels), "target_events": int(labels.sum()),
                             "candidate_events": len(features),
                             "target_candidates": int(record["target"].sum()),
                             "closed_windows": record["closed_windows"]})
    labels = np.concatenate(all_labels)
    probability = np.concatenate(all_prob)
    curve = {}
    for threshold in THRESHOLDS:
        pred = probability >= threshold
        tp = int(np.count_nonzero(pred & labels))
        fp = int(np.count_nonzero(pred & ~labels))
        fn = int(np.count_nonzero(~pred & labels))
        curve[str(threshold)] = {"tp": tp, "fp": fp, "fn": fn,
                                 "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                                 "precision": tp / (tp + fp) if tp + fp else 1.0,
                                 "recall": tp / (tp + fn) if tp + fn else 1.0}
    high = [row for row in per_sequence if row["high_density"]]
    high_tp = high_fn = 0
    offset = 0
    for row in per_sequence:
        n = row["events"]
        if row["high_density"]:
            mask = labels[offset:offset+n]
            pred = probability[offset:offset+n] >= 0.35
            high_tp += int(np.count_nonzero(mask & pred))
            high_fn += int(np.count_nonzero(mask & ~pred))
        offset += n
    candidate_events = sum(row["candidate_events"] for row in per_sequence)
    target_candidates = sum(row["target_candidates"] for row in per_sequence)
    target_events = int(labels.sum())
    closed_windows = sum(row["closed_windows"] for row in per_sequence)
    n = len(per_sequence)
    cost_report = {"frozen_proposal_dense_mac": closed_windows * 2580480,
                   "research_refiner_dense_mac": candidate_events * 3392,
                   "total_dense_mac": closed_windows * 2580480 + candidate_events * 3392,
                   "per_sequence_total_dense_mac": (closed_windows * 2580480 + candidate_events * 3392) / n,
                   "frozen_proposal_effective_mac": None,
                   "scope": "candidate-cache replay; Proposal dense structural count, effective Proposal MAC not remeasured"}
    if kind == "ann":
        cost_report["research_refiner_effective_mac_skip_zero"] = ann_effective_mac
        cost_report["research_refiner_binary_ac"] = 0
    else:
        cost_report.update(total_cost or {})
    return {"role": "train_dev", "sequences": n, "events": len(labels),
            "candidate_events": candidate_events, "candidate_fraction": candidate_events / len(labels),
            "proposal_target_recall": target_candidates / target_events,
            "refiner_conditional_recall_at_0.35": curve["0.35"]["tp"] / target_candidates,
            "high_density_sequences": len(high),
            "high_density_recall_at_0.35": high_tp / (high_tp + high_fn) if high_tp + high_fn else None,
            "average_precision": average_precision(labels, probability),
            "fixed_0.35": curve["0.35"], "threshold_curve": curve,
            "cost": cost_report, "refiner_replay_wall_s": time.perf_counter() - started,
            "per_sequence": per_sequence}


def train_ann(model, records, optimizer, weight):
    model.train()
    loss_sum = events = gradient_max = 0.0
    for record in records:
        features = record["features"]
        order = np.random.permutation(len(features))
        for start in range(0, len(order), 4096):
            part = order[start:start+4096]
            x = torch.from_numpy(features[part].astype(np.float32) / 255.0)
            y = torch.from_numpy(record["target"][part].astype(np.int64))
            optimizer.zero_grad()
            loss = F.cross_entropy(class_logits(model(x)), y, weight=weight)
            loss.backward()
            gradient_max = max(gradient_max, float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(part)
            events += len(part)
    return {"weighted_ce": loss_sum / events, "gradient_norm_max_before_clip": gradient_max,
            "training_candidate_events": events}


def train_snn(model, records, optimizer, weight):
    model.train()
    loss_sum = 0.0
    events = windows_seen = gradient_max = 0
    counts = {"spikes_l1": 0, "spikes_l2": 0, "spikes_out": 0}
    optimizer.zero_grad()
    pending_loss = None
    pending_events = pending_windows = 0
    for record in records:
        runner = SNNRunner(model, record["t0"])
        for segment in windows(record):
            loss, _ = runner.run_window(record["features"][segment], record["x"][segment],
                                        record["y"][segment], record["t"][segment],
                                        record["target"][segment], weight)
            n = segment.stop - segment.start
            pending_loss = loss if pending_loss is None else pending_loss + loss
            pending_events += n
            pending_windows += 1
            loss_sum += float(loss.detach())
            events += n
            windows_seen += 1
            if pending_windows >= 8:
                (pending_loss / pending_events).backward()
                gradient_max = max(gradient_max, float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
                optimizer.step()
                optimizer.zero_grad()
                pending_loss = None
                pending_events = pending_windows = 0
        for key in counts:
            counts[key] += runner.counts[key]
    if pending_loss is not None:
        (pending_loss / pending_events).backward()
        gradient_max = max(gradient_max, float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
        optimizer.step()
    return {"weighted_ce": loss_sum / events, "gradient_norm_max_before_clip": gradient_max,
            "training_candidate_events": events, "nonempty_windows": windows_seen,
            "spike_fraction_l1": counts["spikes_l1"] / (events * 48),
            "spike_fraction_l2": counts["spikes_l2"] / (events * 32),
            "spike_fraction_out": counts["spikes_out"] / (events * 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit-train-sequences", type=int, default=0)
    parser.add_argument("--limit-dev-sequences", type=int, default=0)
    parser.add_argument("--train-name", default="", help="one Train-main sequence; pilot only")
    parser.add_argument("--dev-name", default="", help="one Train-dev sequence; pilot only")
    parser.add_argument("--model", choices=("ann", "snn", "both"), default="both")
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "GEN2-S2-001")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.output.exists():
        raise FileExistsError(f"append-only output already exists: {args.output}")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    index, manifest = read_index()
    train = role_records(index, manifest, "train_main", args.limit_train_sequences, args.train_name)
    dev = role_records(index, manifest, "train_dev", args.limit_dev_sequences, args.dev_name)
    pilot = bool(args.limit_train_sequences or args.limit_dev_sequences or args.train_name or args.dev_name)
    args.output.mkdir(parents=True)
    provenance = {"experiment_id": "GEN2-S2-001", "status": "pilot_not_stage_result" if pilot else "full_s2_health_check",
                  "seed": 42, "epochs": args.epochs, "models": args.model,
                  "train_sequences": len(train), "dev_sequences": len(dev),
                  "index_sha256": manifest["index_sha256"],
                  "teacher_int8_sha256": manifest["teacher_int8_sha256"],
                  "feature_protocol_sha256": manifest["protocol_sha256"],
                  "training_protocol_sha256": sha256(HERE / "PROTOCOL.md"),
                  "core_sha256": sha256(HERE / "core.py"), "train_sha256": sha256(Path(__file__).absolute()),
                  "torch_version": torch.__version__, "numpy_version": np.__version__,
                  "device": "cpu", "threads": torch.get_num_threads()}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    weight = torch.tensor([1.0, 3.0])
    for kind in (("ann", "snn") if args.model == "both" else (args.model,)):
        torch.manual_seed(42)
        np.random.seed(42)
        model = ResearchANN() if kind == "ann" else ResearchSNN()
        lr = 1e-3 if kind == "ann" else 5e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        history = []
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            training = (train_ann(model, train, optimizer, weight) if kind == "ann"
                        else train_snn(model, train, optimizer, weight))
            training["wall_s"] = time.perf_counter() - started
            evaluation = evaluate(model, dev, kind)
            item = {"epoch": epoch, "train": training, "train_dev": evaluation}
            history.append(item)
            (args.output / f"{kind}_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            torch.save({"model_kind": kind, "epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "provenance": provenance},
                       args.output / f"{kind}_epoch{epoch:02d}.pt")
            print(json.dumps({"model": kind, "epoch": epoch,
                              "train_ce": training["weighted_ce"],
                              "dev_iou_at_0.35": evaluation["fixed_0.35"]["iou"],
                              "dev_recall_at_0.35": evaluation["fixed_0.35"]["recall"],
                              "train_wall_s": training["wall_s"],
                              "dev_wall_s": evaluation["refiner_replay_wall_s"]}), flush=True)
    print(f"S2 run complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
