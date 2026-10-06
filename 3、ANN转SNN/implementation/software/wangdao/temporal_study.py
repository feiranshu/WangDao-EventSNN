"""Fixed, bounded research protocol for temporal event segmentation."""
from __future__ import annotations

import argparse
import ctypes
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import subprocess
import time
import threading
from typing import Any

import numpy as np
import torch

from .data import load_events, validate_events
from .h2.numpy_model import load_export, run_pipeline, quantized_mlp_with_activity
from .h2.representation import (build_temporal_patch_features, patch_scores_to_heatmaps,
                                 build_refiner_features, build_patch_features)
from .metrics import event_metrics, localization_metrics
from .refiner_study import (INDEX, ROOT, RUN_ROOT, choose_device, frozen_inputs,
                            scan_thresholds, sha256, study_paths, synchronize, weight_path)
from .s0r import build_s0r_candidates
from .snn import load_proposal_snn_export, profile_proposal_snn
from .temporal_segmentation import (ARMS, SparseSequence, TemporalSegmenter, collision_counts,
                                     collision_summary, event_loss, pack_blocks, represent, structural_cost)

LOADING_IMPLEMENTATION = "packed_scatter_v1"

PROTOCOL = {
    "version": 1, "micro_ms": 25, "steps": 4, "state_reset_ms": 100,
    "channels": ["count_p0", "count_p1", "last_time_p0", "last_time_p1"],
    "counts": "rint(32*log1p(n)), clip uint8, divide by 255",
    "last_time": "float32((t-bin_start)/25), empty=0", "origin": "first event",
    "epochs": 30, "batch_size": 8, "optimizer": "AdamW", "lr": 1e-3,
    "lr_min": 1e-5, "weight_decay": 1e-4, "gradient_clip": 5.0,
    "precision": "float32; no AMP or TF32", "beta": 0.875, "threshold": 0.25,
    "surrogate_slope": 10, "focal_alpha": 0.75, "focal_gamma": 2,
    "soft_iou_weight": 0.5, "spike_regularizer": [0.01, 0.01, 0.25],
    "hidden_initialization": "Kaiming normal fan_in relu", "head_std": 0.01,
    "bias": "logit Train-main polarity prior clipped [1e-6,1-1e-6]",
    "thresholds": [0.01, 0.99, 0.01], "selection": "IoU,F1,earlier epoch,lower threshold",
    "seeds": [42, 43, 44], "max_runs": 12, "arms": list(ARMS),
    "cost_gate": {"iou_loss": 0.005, "pd_loss": 0.005, "fa_ratio": 1.1, "cpu_ratio": 0.8},
    "accuracy_gate": {"iou_gain": 0.01, "pd_loss": 0.005, "fa_ratio": 1.1, "cpu_ratio": 2.0},
    "cpu_timing": "1 thread; min/median/max event-count Train-dev sequences; 1 warmup + 3 repeats",
    "learning_check": {"blocks": 8, "updates": 100, "loss_reduction": 0.20},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def report(stage: str, directory: Path, status: str = "running") -> dict:
    return {"schema_version": 1, "status": status, "experiment_id": directory.name,
            "stage": stage, "data_role": ["train_main", "train_dev"],
            "primary_metrics": {}, "decision": {"hardware_mainline_changed": False},
            "conclusion": "Research pending; frozen deployment unchanged.", "protocol": PROTOCOL,
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "cpu_threads": torch.get_num_threads(),
                            "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}}


class SequenceStore:
    """One producer uses a bounded LRU; disk records include source identity and mapping checks."""
    def __init__(self, directory: Path, memory_mb: int = 1024):
        if not 0 <= memory_mb <= 1024:
            raise ValueError("cache memory must be between 0 and 1024 MiB")
        self.directory = directory
        self.limit = memory_mb * 1024 ** 2
        self.memory: OrderedDict[str, SparseSequence] = OrderedDict()
        self.used = 0
        self.identity = sha256(INDEX)
        self.paths = {r: study_paths(r) for r in ("train_main", "train_dev")}
        self.stats = {"memory_hits": 0, "disk_hits": 0, "builds": 0, "load_build_s": 0.0,
                      "peak_cache_bytes": 0, "evictions": 0}

    def get(self, role: str, name: str) -> SparseSequence:
        if role not in self.paths:
            raise ValueError("only Train-main and Train-dev may be accessed")
        path = next((p for p in self.paths[role] if p.name == name), None)
        if path is None:
            raise ValueError("sequence is not in the requested role")
        stat = path.stat()
        identity = {"version": PROTOCOL["version"], "index": self.identity, "role": role,
                    "name": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        key = json.dumps(identity, sort_keys=True)
        if key in self.memory:
            self.stats["memory_hits"] += 1
            self.memory.move_to_end(key)
            return self.memory[key]
        started = time.perf_counter()
        target = self.directory / f"{role}-{name}"
        if target.exists():
            with np.load(target, allow_pickle=False) as archive:
                if json.loads(str(archive["identity"].item())) != identity:
                    raise ValueError(f"stale cache: {target}")
                record = SparseSequence(*(archive[k] for k in ("events", "bins", "keys", "counts", "last")))
            validate_events(record.events, require_labels=True)
            # Rebuild once per disk load: detects corrupted features and event maps.
            source = load_events(path, require_labels=True)
            expected = represent(source)
            for field in ("events", "bins", "keys", "counts", "last"):
                actual = getattr(record, field)
                wanted = getattr(expected, field)
                if actual.dtype != wanted.dtype or not np.array_equal(actual, wanted):
                    raise ValueError(f"inconsistent cached {field}: {name}")
            record.freeze()
            self.stats["disk_hits"] += 1
        else:
            record = represent(load_events(path, require_labels=True))
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, identity=np.array(json.dumps(identity)),
                                    **{k: getattr(record, k) for k in ("events", "bins", "keys", "counts", "last")})
            temporary.replace(target)
            self.stats["builds"] += 1
        if record.nbytes <= self.limit:
            while self.used + record.nbytes > self.limit:
                _, old = self.memory.popitem(last=False)
                self.used -= old.nbytes
                self.stats["evictions"] += 1
            self.memory[key] = record
            self.used += record.nbytes
        self.stats["peak_cache_bytes"] = max(self.stats["peak_cache_bytes"], self.used)
        self.stats["load_build_s"] += time.perf_counter() - started
        return record


def training_inventory(store: SequenceStore) -> tuple[list[tuple[str, int]], tuple[float, float], dict]:
    blocks, positives, negatives = [], [], []
    polarity_count, polarity_target = np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
    for path in store.paths["train_main"]:
        record = store.get("train_main", path.name)
        blocks.extend((path.name, int(b)) for b in record.blocks)
        p, n = collision_counts(record)
        positives.append(p)
        negatives.append(n)
        for polarity in range(2):
            selected = record.events["p"] == polarity
            polarity_count[polarity] += selected.sum()
            polarity_target[polarity] += record.events["label"][selected].sum()
    priors = np.clip(polarity_target / np.maximum(polarity_count, 1), 1e-6, 1 - 1e-6)
    return blocks, (float(priors[0]), float(priors[1])), collision_summary(np.concatenate(positives), np.concatenate(negatives))


def collate(store: SequenceStore, role: str, selected: list[tuple[str, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records = [(store.get(role, name), block) for name, block in selected]
    features, event_maps = pack_blocks(records)
    indices, labels = [], []
    for batch, ((record, _), event_indices) in enumerate(zip(records, event_maps)):
        events = record.events[event_indices]
        indices.append(np.column_stack((record.bins[event_indices] % 4, np.full(len(events), batch),
                                         events["p"], events["y"], events["x"])).astype(np.int64))
        labels.append(events["label"].astype(np.float32))
    return (torch.from_numpy(features), torch.from_numpy(np.concatenate(indices)),
            torch.from_numpy(np.concatenate(labels)))


def batches(store: SequenceStore, order: list[tuple[str, int]], prefetch: bool = True):
    groups = [order[i:i + 8] for i in range(0, len(order), 8)]
    if not prefetch:
        for group in groups:
            yield collate(store, "train_main", group)
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(collate, store, "train_main", groups[0]) if groups else None
        for index in range(len(groups)):
            assert future is not None
            batch = future.result()
            future = pool.submit(collate, store, "train_main", groups[index + 1]) if index + 1 < len(groups) else None
            yield batch


def epoch_order(blocks: list[tuple[str, int]], seed: int, epoch: int) -> list[tuple[str, int]]:
    generator = torch.Generator().manual_seed(seed + 100000 * epoch)
    return [blocks[i] for i in torch.randperm(len(blocks), generator=generator).tolist()]


def preparation_timing(store: SequenceStore, blocks: list[tuple[str, int]]) -> dict:
    order = epoch_order(blocks, 42, 1)[:128]
    started = time.perf_counter()
    count = 0
    for batch in batches(store, order):
        count += 1
        del batch
    seconds = time.perf_counter() - started
    return {"batches": count, "blocks": len(order), "seconds": seconds,
            "seconds_per_batch": seconds / max(count, 1), "prefetch_batches": 1,
            "scope": "representative epoch-1 batches; CPU sparse-to-dense collation including cache waits"}


def batch_loss(model: TemporalSegmenter, batch, device: torch.device):
    features, indices, labels = (v.to(device) for v in batch)
    output = model(features)
    if not torch.isfinite(output.logits).all():
        raise FloatingPointError("nonfinite spatial logits")
    chosen = output.logits[tuple(indices.T)]
    return event_loss(chosen, labels, output.rates, model.spiking), output.rates


def update(model: TemporalSegmenter, batch, optimizer, device: torch.device) -> tuple[float, list[float]]:
    optimizer.zero_grad(set_to_none=True)
    loss, rates = batch_loss(model, batch, device)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite gradient")
    optimizer.step()
    return float(loss.detach().cpu()), rates.detach().cpu().tolist()


def infer_record(model: TemporalSegmenter, record: SparseSequence, device: torch.device, batch_size: int = 8) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model.eval()
    probability = np.empty(len(record.events), dtype=np.float32)
    written = np.zeros(len(probability), dtype=bool)
    blocks = record.blocks
    # Empty blocks have no outputs and start/end with a reset. Empty steps inside a block still run.
    with torch.no_grad():
        for start in range(0, len(blocks), batch_size):
            packed, event_maps = pack_blocks([(record, int(b)) for b in blocks[start:start + batch_size]])
            tensor = torch.from_numpy(packed).to(device)
            output = model(tensor).logits
            for batch, indices in enumerate(event_maps):
                e = record.events[indices]
                index = tuple(torch.as_tensor(a.astype(np.int64), device=device) for a in
                              (record.bins[indices] % 4, np.full(len(indices), batch), e["p"], e["y"], e["x"]))
                if written[indices].any():
                    raise AssertionError("duplicate event output")
                probability[indices] = output[index].sigmoid().cpu().numpy()
                written[indices] = True
    if not written.all() or not np.isfinite(probability).all():
        raise AssertionError("missing or nonfinite event output")
    return probability


def evaluate(events: list[np.ndarray], probabilities: list[np.ndarray], fixed_threshold: float | None = None,
             with_localization: bool = True) -> dict:
    labels = np.concatenate([e["label"] for e in events])
    values = np.concatenate(probabilities)
    scan = scan_thresholds(labels, values)
    threshold = scan["best_threshold"] if fixed_threshold is None else fixed_threshold
    result = {"scan": scan, "event": event_metrics(labels, values, threshold), "threshold": threshold}
    if with_localization:
        totals: dict[str, float] = dict.fromkeys(("targets", "correct_targets", "false_components", "frame_denominator"), 0.0)
        per_sequence = []
        for e, p in zip(events, probabilities):
            loc = localization_metrics(e["t"], e["x"], e["y"], e["label"], e["name"], p, threshold)
            for key in totals:
                totals[key] += loc[key]
            per_sequence.append({"event": event_metrics(e["label"], p, threshold), "localization": loc})
        totals["pd"] = totals["correct_targets"] / totals["targets"] if totals["targets"] else 1.0
        denominator = totals["frame_denominator"] * 346 * 260
        totals["fa"] = totals["false_components"] / denominator if denominator else 0.0
        result.update(localization=totals, per_sequence=per_sequence)
    return result


def evaluate_model(model: TemporalSegmenter, store: SequenceStore, device: torch.device,
                   threshold: float | None = None, localization: bool = True) -> dict:
    started = time.perf_counter()
    events, probabilities = [], []
    for path in store.paths["train_dev"]:
        record = store.get("train_dev", path.name)
        events.append(record.events)
        probabilities.append(infer_record(model, record, device))
    result = evaluate(events, probabilities, threshold, localization)
    result["files"] = [p.name for p in store.paths["train_dev"]]
    result["seconds"] = time.perf_counter() - started
    return result


def preflight(args, store: SequenceStore, device: torch.device) -> dict:
    target = args.run_directory / "preflight.json"
    result = read_json(target) if target.exists() else report("preflight", args.run_directory)
    if result["protocol"] != PROTOCOL:
        raise ValueError("preflight protocol mismatch")
    started = time.perf_counter()
    blocks, priors, collisions = training_inventory(store)
    build_seconds = time.perf_counter() - started
    preparation = preparation_timing(store, blocks)
    selected = []
    for name, block in blocks:
        r = store.get("train_main", name)
        lo, hi = np.searchsorted(r.bins, [block * 4, (block + 1) * 4])
        labels = r.events["label"][lo:hi]
        if np.any(labels == 1) and np.any(labels == 0):
            selected.append((name, block))
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("learning check requires 8 mixed Train-main blocks")
    batch_started = time.perf_counter()
    batch = collate(store, "train_main", selected)
    wait_seconds = time.perf_counter() - batch_started
    result.update(frozen_inputs=frozen_inputs(), priors=list(priors), train_blocks=len(blocks),
                  collision_train_main=collisions, learning_blocks=selected,
                  representation_s=build_seconds, batch_wait_s=wait_seconds, pilot=True)
    result["batch_preparation_timing"] = preparation
    result["environment"] = {**report("preflight", args.run_directory)["environment"], "device": str(device)}
    if args.full_dev:
        groups = [collision_counts(store.get("train_dev", p.name)) for p in store.paths["train_dev"]]
        result["collision_train_dev"] = collision_summary(np.concatenate([p for p, _ in groups]),
                                                         np.concatenate([n for _, n in groups]))
    result.setdefault("arms", {})
    for arm in args.arms:
        model = TemporalSegmenter(arm, 42, priors).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        with torch.no_grad():
            initial = float(batch_loss(model, batch, device)[0].cpu())
        synchronize(device)
        started = time.perf_counter()
        history = []
        for step in range(args.pilot_updates):
            loss, rates = update(model, batch, optimizer, device)
            history.append({"update": step + 1, "loss": loss, "rates": rates})
        synchronize(device)
        seconds = time.perf_counter() - started
        with torch.no_grad():
            final_loss, final_rates = batch_loss(model, batch, device)
            final = float(final_loss.cpu())
        reduction = 1 - final / initial
        active = not model.spiking or bool(torch.all(final_rates > 0))
        entry = {"updates": args.pilot_updates, "initial_loss": initial, "final_loss": final,
                 "loss_reduction": reduction, "hidden_active": active, "finite_loss_gradients": True,
                 "learning_pass": args.pilot_updates == 100 and reduction >= 0.20 and active,
                 "training_s": seconds, "seconds_per_update": seconds / args.pilot_updates,
                 "history": history, "structure": structural_cost(model.spiking)}
        if args.full_dev:
            entry["dev_pilot"] = evaluate_model(model, store, device, localization=False)
        dev_s = entry.get("dev_pilot", {}).get("seconds")
        entry["estimated_30_epoch_s"] = (30 * ((len(blocks) + 7) // 8 * (seconds / args.pilot_updates + preparation["seconds_per_batch"]) + dev_s)
                                           if dev_s is not None else None)
        result["arms"][arm] = entry
        result["cache"] = store.stats.copy()
        write_json(target, result)
        print(json.dumps({"arm": arm, **{k: v for k, v in entry.items() if k not in ("history", "dev_pilot", "structure")}}), flush=True)
    complete = all(a in result["arms"] and result["arms"][a]["updates"] == 100
                   and "dev_pilot" in result["arms"][a] for a in ARMS)
    passed = complete and all(result["arms"][a]["learning_pass"] for a in ARMS)
    result["status"] = "ok" if passed else ("learning_failed" if complete else "pilot_incomplete")
    result["decision"].update(formal_training_allowed=passed)
    result["conclusion"] = "Ready for bounded formal training." if passed else "Formal training blocked until all fixed learning checks pass."
    result["cache"] = store.stats.copy()
    write_json(target, result)
    return result


def checkpoint_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_load(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def learning_input_audit(store: SequenceStore, blocks: list) -> dict:
    batch = collate(store, "train_main", blocks)
    features, indices, labels = batch
    expected_labels = []
    expected_indices = []
    for batch_index, (name, block) in enumerate(blocks):
        e = store.get("train_main", name).events
        relative = e["t"].astype(np.float64) - float(e["t"][0])
        mask = (relative >= 100 * block) & (relative < 100 * (block + 1))
        expected_labels.extend(e["label"][mask].tolist())
        expected_indices.extend(zip((np.floor(relative[mask] / 25).astype(np.int64) % 4).tolist(),
                                    [batch_index] * int(mask.sum()), e["p"][mask].tolist(),
                                    e["y"][mask].tolist(), e["x"][mask].tolist()))
    if labels.tolist() != expected_labels or indices.tolist() != [list(v) for v in expected_indices]:
        raise AssertionError("learning batch does not match raw event supervision")
    # An independent probability-space calculation checks the implemented focal/soft-IoU objective.
    logits = torch.linspace(-3, 3, len(labels))
    p = 1 / (1 + np.exp(-logits.numpy().astype(np.float64)))
    y = labels.numpy().astype(np.float64)
    pt = np.where(y > .5, p, 1 - p)
    focal = np.mean(np.where(y > .5, .75, .25) * (1 - pt) ** 2 * -np.log(pt))
    expected = focal + .5 * (1 - ((p * y).sum() + 1e-6) / ((p + y - p * y).sum() + 1e-6))
    actual = float(event_loss(logits, labels, torch.zeros(3), False))
    if abs(expected - actual) > 1e-6:
        raise AssertionError("event loss differs from independent formula")
    return {"event_mapping_matches_raw": True, "independent_loss_formula_matches": True,
            "events": len(labels), "positive_events": int(labels.sum()), "negative_events": int((labels == 0).sum()),
            "feature_min": float(features.min()), "feature_max": float(features.max()),
            "feature_nonzero_fraction": float((features != 0).float().mean()),
            "interpretation": "Very sparse spatial input; nonzero hidden activity alone is not evidence of adequate learning."}


def train(args, store: SequenceStore, device: torch.device) -> dict:
    pre = read_json(args.run_directory / "preflight.json")
    hashes = frozen_inputs()
    if pre["protocol"] != PROTOCOL or pre["frozen_inputs"] != hashes or not pre["decision"]["formal_training_allowed"]:
        raise ValueError("formal training requires matching, passed full preflight")
    if args.seed != 42:
        selection = read_json(args.run_directory / "selection.json")
        if any(arm not in selection["decision"]["replication_arms"] for arm in args.arms):
            raise ValueError("replication arm was not admitted by seed42 gates")
    blocks, priors, _ = training_inventory(store)
    for arm in args.arms:
        directory = args.run_directory / f"{arm}-seed{args.seed}"
        target = directory / "training.json"
        result = read_json(target) if target.exists() else report("train", args.run_directory)
        if target.exists() and (result["protocol"] != PROTOCOL or result["frozen_inputs"] != hashes
                                or result["arm"] != arm or result["seed"] != args.seed):
            raise ValueError("training identity mismatch")
        if result["status"] == "ok":
            print(f"{arm} seed {args.seed}: already complete", flush=True)
            continue
        result.update(arm=arm, seed=args.seed, pilot=False, frozen_inputs=hashes, priors=list(priors))
        backend = {"device": str(device), "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads()}
        if "backend" in result and result["backend"] != backend:
            raise ValueError("resume must use the original training backend")
        result["backend"] = backend
        result.setdefault("history", [])
        model = TemporalSegmenter(arm, args.seed, priors).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        last = directory / "last.pt"
        if last.exists():
            saved = checkpoint_load(last, device)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            result = saved["report"]
        elif result["history"]:
            raise ValueError("missing resume checkpoint")
        write_json(target, result)
        for epoch in range(len(result["history"]) + 1, 31):
            started = time.perf_counter()
            iterator = iter(batches(store, epoch_order(blocks, args.seed, epoch)))
            wait_seconds, total_loss, updates, total_events = 0.0, 0.0, 0, 0
            model.train()
            while True:
                waiting = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                wait_seconds += time.perf_counter() - waiting
                loss, _ = update(model, batch, optimizer, device)
                count = len(batch[2])
                total_loss += loss * count
                total_events += count
                updates += 1
            synchronize(device)
            training_seconds = time.perf_counter() - started
            evaluation = evaluate_model(model, store, device, localization=False)
            entry = {"epoch": epoch, "loss": total_loss / total_events, "updates": updates,
                     "blocks": len(blocks), "events": total_events, "input_wait_s": wait_seconds,
                     "train_s": training_seconds, "selection": evaluation, "lr": optimizer.param_groups[0]["lr"]}
            result["history"].append(entry)
            candidate_key = (evaluation["event"]["iou"], evaluation["event"]["f1"], -epoch, -evaluation["threshold"])
            previous = result.get("best")
            old_key = tuple(previous["key"]) if previous else (-1, -1, 0, 0)
            if candidate_key > old_key:
                result["best"] = {"key": list(candidate_key), "epoch": epoch, "threshold": evaluation["threshold"],
                                  "event": evaluation["event"]}
                checkpoint_save(directory / "best.pt", {"model": model.state_dict(), "arm": arm,
                                "seed": args.seed, "epoch": epoch, "priors": list(priors), "protocol": PROTOCOL,
                                "threshold": evaluation["threshold"], "frozen_inputs": hashes})
            scheduler.step()
            result["cache"] = store.stats.copy()
            checkpoint_save(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                  "scheduler": scheduler.state_dict(), "report": result})
            write_json(target, result)
            print(json.dumps({"arm": arm, "seed": args.seed, "epoch": epoch, "loss": entry["loss"],
                              "iou": evaluation["event"]["iou"], "seconds": training_seconds + evaluation["seconds"]}), flush=True)
        result["status"] = "ok"
        result["primary_metrics"] = result["best"]["event"]
        result["conclusion"] = "30 epochs complete; checkpoint requires unified selection and cost audit."
        write_json(target, result)
    return {"status": "ok", "stage": "train", "next": "select"}


class Reference:
    def __init__(self, kind: str, device: torch.device):
        self.kind, self.device = kind, device
        self.h2 = load_export(weight_path("h2"))
        self.proposal, _, metadata = load_proposal_snn_export(weight_path("proposal"))
        self.proposal.to(device).eval()
        # The frozen hybrid retains the H2 final threshold; S0-P exports only its routing threshold.
        self.threshold = self.h2.event_threshold
        if float(metadata["proposal_threshold"]) != 0.51:
            raise ValueError("frozen S0-P routing threshold mismatch")

    def __call__(self, events: np.ndarray) -> np.ndarray:
        if self.kind == "H2":
            return run_pipeline(events, self.h2)["probabilities"]
        temporal = build_temporal_patch_features(events, list(range(80)), 4)
        tensor = torch.from_numpy(temporal).permute(1, 0, 2, 3).reshape(4, -1, 128)
        pieces = []
        with torch.no_grad():
            for start in range(0, tensor.shape[1], 4096):
                pieces.append(self.proposal(tensor[:, start:start + 4096].to(self.device)).logits.cpu().numpy())
        logits = np.concatenate(pieces).astype(np.float64)
        scores = (1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))).astype(np.float32)
        heatmaps = patch_scores_to_heatmaps(scores.reshape(80, 336, 16))
        indices, _, _, values = build_s0r_candidates(events, self.h2, heatmaps=heatmaps, proposal_threshold=0.51)
        result = np.zeros(len(events), dtype=np.float32)
        result[indices] = values
        return result

    def activity(self, events: np.ndarray) -> dict:
        output = run_pipeline(events, self.h2) if self.kind == "H2" else None
        patches = 80 * 336
        spikes = [0, 0, 0]
        if self.kind == "H2":
            assert output is not None
            indices, heatmaps = output["candidate_indices"], output["heatmaps"]
            features = build_patch_features(events, list(range(80))).reshape(-1, 128)
            proposal = quantized_mlp_with_activity(features, self.h2.proposal_layers)
            nonzero = sum(n * width for n, width in zip(proposal.activation_nonzero, (48, 24, 16)))
            products, accumulations, updates, states = patches * 7680, 0, 0, 0
        else:
            features = build_temporal_patch_features(events, list(range(80)), 4)
            tensor = torch.from_numpy(features).permute(1, 0, 2, 3).reshape(4, -1, 128)
            logits = []
            nonzero = 0
            with torch.no_grad():
                for start in range(0, tensor.shape[1], 4096):
                    active = profile_proposal_snn(self.proposal, tensor[:, start:start + 4096].to(self.device))
                    logits.append(active.output.logits.cpu().numpy())
                    nonzero += active.input_nonzero * 48
                    spikes = [a + b for a, b in zip(spikes, active.spike_counts)]
            values = np.concatenate(logits).astype(np.float64)
            heatmaps = patch_scores_to_heatmaps((1 / (1 + np.exp(-np.clip(values, -60, 60)))).astype(np.float32).reshape(80, 336, 16))
            indices, _, _, _ = build_s0r_candidates(events, self.h2, heatmaps=heatmaps, proposal_threshold=0.51)
            products = patches * 4 * 6144
            accumulations = spikes[0] * 24 + spikes[1] * 16
            # Existing Python also updates a diagnostic output-spiking membrane, separately identified.
            updates, states = patches * 4 * (48 + 24 + 16), 48 + 24 + 16
        ref_features = build_refiner_features(events, indices, list(range(80)), heatmaps,
                                              self.h2.proposal_threshold if self.kind == "H2" else 0.51)
        ref = quantized_mlp_with_activity(ref_features, self.h2.refiner_layers)
        ref_nonzero = sum(n * width for n, width in zip(ref.activation_nonzero, (64, 32, 4)))
        return {"parameters": 11904, "events": len(events), "candidates": len(indices),
                "nonbinary_synaptic_products": products + len(indices) * 4224,
                "nonzero_nonbinary_product_proxy": nonzero + ref_nonzero,
                "active_spike_accumulations": accumulations, "hidden_and_readout_membrane_updates": updates,
                "diagnostic_extra_membrane_updates": patches * 4 * 16 if self.kind != "H2" else 0,
                "membrane_state_values_per_patch": states,
                "readout_accumulator_values_per_patch": 16 if self.kind != "H2" else 0,
                "scope": "functional operation counts; sequential per-patch state; Python batching and diagnostics cost extra"}


def working_set_bytes() -> int | None:
    """Current process resident memory on Windows; includes libraries and shared caches."""
    if platform.system() != "Windows":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def sampled_memory(predict, events: np.ndarray) -> dict:
    # Separate profiling invocation avoids perturbing the latency comparison with a sampling thread.
    baseline = working_set_bytes()
    if baseline is None:
        return {"status": "unavailable", "scope": "resident memory sampler is Windows-only"}
    samples = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.01):
            value = working_set_bytes()
            if value is not None:
                samples.append(value)
    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        predict(events)
    finally:
        final = working_set_bytes()
        if final is not None:
            samples.append(final)
        stop.set()
        worker.join()
    return {"status": "ok", "baseline_bytes": baseline, "sampled_peak_bytes": max(samples),
            "sampled_peak_increase_bytes": max(samples) - baseline, "sample_interval_ms": 10,
            "scope": "process working set, includes runtime/caches; sampled lower bound, not model-only allocation"}


def timing_sequences(store: SequenceStore) -> list[tuple[str, np.ndarray]]:
    names = sorted(((len(store.get("train_dev", p.name).events), p.name) for p in store.paths["train_dev"]))
    return [(names[i][1], store.get("train_dev", names[i][1]).events) for i in (0, len(names) // 2, len(names) - 1)]


def benchmark(predict, selected: list[tuple[str, np.ndarray]], threshold: float, device: torch.device) -> dict:
    rows = []
    for name, events in selected:
        # Includes feature construction, inference, event thresholding and connected-component postprocessing.
        def run():
            p = predict(events)
            _ = p >= threshold
            return localization_metrics(events["t"], events["x"], events["y"], events["label"], events["name"], p, threshold)
        run()
        samples = []
        for _ in range(3):
            synchronize(device)
            started = time.perf_counter()
            run()
            synchronize(device)
            samples.append(time.perf_counter() - started)
        rows.append({"file": name, "events": len(events), "seconds": samples, "median_s": float(np.median(samples))})
    # Each repetition is the same workload (sum of the 3 sequences), then take its median.
    totals = np.array([r["seconds"] for r in rows]).sum(axis=0)
    return {"sequences": rows, "repetition_total_s": totals.tolist(), "median_s": float(np.median(totals)),
            "device": str(device), "cpu_threads": torch.get_num_threads(),
            "scope": "in-memory raw events through representation, model, threshold, official component/metric postprocessing"}


def profile_activity(model: TemporalSegmenter, record: SparseSequence, device: torch.device) -> dict:
    counts = {"nonbinary_synaptic_products": 0, "active_spike_accumulations": 0,
              "hidden_updates": 0, "hidden_leak_multiplications": 0, "hidden_spikes": [0, 0, 0]}
    handles = []
    def hook(index):
        def count(layer, inputs, output):
            value = inputs[0]
            if model.spiking and index > 0:
                # Count only valid output fanout, excluding out-of-image convolution padding.
                kernel = torch.ones((1, value.shape[1], *layer.kernel_size), device=value.device)
                active = torch.nn.functional.conv2d(value, kernel, padding=layer.padding, dilation=layer.dilation)
                counts["active_spike_accumulations"] += int(active.sum().item()) * layer.out_channels
            else:
                counts["nonbinary_synaptic_products"] += output.numel() * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            if index < 3:
                counts["hidden_updates"] += output.numel()
                counts["hidden_leak_multiplications"] += output.numel()
        return count
    for i, layer in enumerate([*model.hidden, model.head]):
        handles.append(layer.register_forward_hook(hook(i)))
    # Count cell output spikes separately; cells are shared, calls are in layer order.
    calls = [0]
    def cell_hook(module, inputs, output):
        counts["hidden_spikes"][calls[0] % 3] += int(output[0].sum().item())
        calls[0] += 1
    if model.spiking:
        handles.append(model.cell.register_forward_hook(cell_hook))
    try:
        infer_record(model, record, device)
    finally:
        for handle in handles:
            handle.remove()
    return {"structure": structural_cost(model.spiking), "counts": counts,
            "events": len(record.events), "blocks": len(record.blocks)}


def gates(candidate: dict, reference: dict) -> dict:
    c, r = candidate["evaluation"], reference["evaluation"]
    gain = c["event"]["iou"] - r["event"]["iou"]
    guard = c["localization"]["pd"] >= r["localization"]["pd"] - 0.005 and c["localization"]["fa"] <= r["localization"]["fa"] * 1.1
    ratio = candidate["cpu"]["median_s"] / reference["cpu"]["median_s"]
    return {"iou_gain": gain, "localization_pass": guard, "cpu_ratio": ratio,
            "cost_pass": guard and gain >= -0.005 and ratio <= 0.8,
            "accuracy_pass": guard and gain >= 0.01 and ratio <= 2.0}


def decide(runs: dict, reference: dict) -> dict:
    first = runs.get("42", {})
    admission = {a: gates(r, reference) for a, r in first.items()}
    pairs = set()
    for arm, gate in admission.items():
        if gate["cost_pass"] or gate["accuracy_pass"]:
            pairs.add(arm.split("-")[0])
    replicate = [a for a in ARMS if a.split("-")[0] in pairs]
    fixed, software, repeated = [], [], {}
    for arm in replicate:
        if not all(arm in runs.get(str(s), {}) for s in (42, 43, 44)):
            continue
        checks = [gates(runs[str(s)][arm], reference) for s in (42, 43, 44)]
        cost = all(c["cost_pass"] for c in checks)
        accuracy = all(c["accuracy_pass"] for c in checks)
        dominated = False
        if arm.endswith("SNN"):
            ann = arm.replace("SNN", "ANN")
            dominated = all(ann in runs.get(str(s), {}) and
                            runs[str(s)][ann]["evaluation"]["event"]["iou"] >= runs[str(s)][arm]["evaluation"]["event"]["iou"] and
                            runs[str(s)][ann]["cpu"]["median_s"] <= runs[str(s)][arm]["cpu"]["median_s"] and
                            (runs[str(s)][ann]["evaluation"]["event"]["iou"] > runs[str(s)][arm]["evaluation"]["event"]["iou"] or
                             runs[str(s)][ann]["cpu"]["median_s"] < runs[str(s)][arm]["cpu"]["median_s"])
                            for s in (42, 43, 44))
        repeated[arm] = {"seeds": checks, "cost_repeated": cost, "accuracy_repeated": accuracy,
                         "dominated_by_paired_ann": dominated}
        if cost and not dominated:
            fixed.append(arm)
        if accuracy and not dominated:
            software.append(arm)
    complete42 = all(a in first for a in ARMS)
    complete = complete42 and all(all(a in runs.get(str(s), {}) for a in replicate) for s in (43, 44))
    return {"seed42": admission, "replication_arms": replicate if complete42 else [], "repeated": repeated,
            "fixed_point_research_candidates": fixed if complete else [],
            "further_software_candidates": software if complete else [],
            "complete": complete, "hardware_mainline_changed": False,
            "next": ("consider_admitted_candidates" if fixed or software else "retain_mainline_stop_search") if complete else "finish_required_runs"}


def select(args, store: SequenceStore, device: torch.device) -> dict:
    hashes = frozen_inputs()
    result = report("select", args.run_directory)
    result.update(frozen_inputs=hashes, references={}, runs={})
    result["environment"]["device"] = str(device)
    pre = read_json(args.run_directory / "preflight.json")
    if pre["protocol"] != PROTOCOL or pre["frozen_inputs"] != hashes:
        raise ValueError("preflight provenance mismatch")
    if pre["status"] == "learning_failed":
        pre["learning_input_audit"] = learning_input_audit(store, pre["learning_blocks"])
        if "batch_preparation_timing" not in pre:
            blocks, _, _ = training_inventory(store)
            pre["batch_preparation_timing"] = preparation_timing(store, blocks)
            for entry in pre["arms"].values():
                entry["estimated_30_epoch_s"] += 30 * ((len(blocks) + 7) // 8) * pre["batch_preparation_timing"]["seconds_per_batch"]
        execution = args.run_directory / "last-preflight-execution.json"
        if execution.exists():
            pre["execution"] = read_json(execution)
        groups = [collision_counts(store.get("train_dev", p.name)) for p in store.paths["train_dev"]]
        pre["collision_train_dev"] = collision_summary(np.concatenate([p for p, _ in groups]),
                                                       np.concatenate([n for _, n in groups]))
        result.update(status="stopped_preflight", preflight=pre,
                      primary_metrics={a: {k: pre["arms"][a][k] for k in ("loss_reduction", "learning_pass", "hidden_active")}
                                       for a in ARMS},
                      decision={"hardware_mainline_changed": False, "formal_training_allowed": False,
                                "replication_arms": [], "fixed_point_research_candidates": [],
                                "further_software_candidates": [], "next": "retain_mainline_stop_fixed_protocol"},
                      conclusion="Fixed SNN learning checks failed; no formal training or architecture superiority claim.")
        write_json(args.run_directory / "selection.json", result)
        if args.output:
            write_json(args.output, result)
        return result
    if pre["status"] != "ok":
        raise ValueError("complete preflight before selection")
    selected = timing_sequences(store)
    cpu = torch.device("cpu")
    for kind in ("H2", "S0P-H2"):
        baseline = Reference(kind, cpu)
        events, probabilities = [], []
        for path in store.paths["train_dev"]:
            e = store.get("train_dev", path.name).events
            events.append(e)
            probabilities.append(baseline(e))
        evaluation = evaluate(events, probabilities)
        fixed = evaluate(events, probabilities, baseline.threshold)
        timing = benchmark(baseline, selected, evaluation["threshold"], cpu)
        result["references"][kind] = {"evaluation": evaluation, "original_threshold": fixed, "cpu": timing,
                                          "context": "offline application reference; not a causal architecture control"}
        result["references"][kind]["activity_median_sequence"] = baseline.activity(selected[1][1])
        result["references"][kind]["cpu_memory"] = sampled_memory(baseline, selected[-1][1])
        if device.type == "cuda" and kind == "S0P-H2":
            accelerated = Reference(kind, device)
            torch.cuda.reset_peak_memory_stats(device)
            result["references"][kind]["gpu"] = benchmark(accelerated, selected, evaluation["threshold"], device)
            result["references"][kind]["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["references"][kind]["gpu_scope"] = "CUDA Proposal with NumPy CPU feature construction and Refiner"
        print(f"reference {kind}: IoU {evaluation['event']['iou']:.6f}", flush=True)
    reference_name = max(result["references"], key=lambda k: result["references"][k]["evaluation"]["event"]["iou"])
    result["reference"] = reference_name
    for seed in (42, 43, 44):
        for arm in ARMS:
            directory = args.run_directory / f"{arm}-seed{seed}"
            if not (directory / "training.json").exists():
                continue
            record = read_json(directory / "training.json")
            if record["status"] != "ok" or record["pilot"] or len(record["history"]) != 30 or record["protocol"] != PROTOCOL or record["frozen_inputs"] != hashes:
                raise ValueError(f"incomplete or incompatible formal training: {directory}")
            best = max(record["history"], key=lambda h: (h["selection"]["event"]["iou"], h["selection"]["event"]["f1"], -h["epoch"], -h["selection"]["threshold"]))
            saved = checkpoint_load(directory / "best.pt", cpu)
            if saved["arm"] != arm or saved["seed"] != seed or saved["epoch"] != best["epoch"] or saved["threshold"] != best["selection"]["threshold"] or saved["protocol"] != PROTOCOL or saved["frozen_inputs"] != hashes:
                raise ValueError("selected checkpoint provenance mismatch")
            model = TemporalSegmenter(arm, seed, tuple(saved["priors"]))
            model.load_state_dict(saved["model"], strict=True)
            # Re-evaluate on the training backend before timing CPU: reduction order may differ by backend.
            model.to(device)
            evaluation = evaluate_model(model, store, device, saved["threshold"])
            if evaluation["event"] != best["selection"]["event"]:
                raise ValueError("selected checkpoint metrics do not reproduce on this backend")
            model.to(cpu)
            timing = benchmark(lambda e: infer_record(model, represent(e), cpu), selected, saved["threshold"], cpu)
            activity = profile_activity(model, store.get("train_dev", selected[1][0]), cpu)
            entry = {"evaluation": evaluation, "cpu": timing, "activity_median_sequence": activity,
                     "checkpoint": str((directory / "best.pt").relative_to(ROOT)), "epoch": saved["epoch"]}
            entry["cpu_memory"] = sampled_memory(lambda e: infer_record(model, represent(e), cpu), selected[-1][1])
            if device.type == "cuda":
                model.to(device)
                torch.cuda.reset_peak_memory_stats(device)
                entry["gpu"] = benchmark(lambda e: infer_record(model, represent(e), device), selected, saved["threshold"], device)
                entry["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["runs"].setdefault(str(seed), {})[arm] = entry
            print(f"audited {arm} seed {seed}: IoU {evaluation['event']['iou']:.6f}", flush=True)
            write_json(args.run_directory / "selection.partial.json", result)
    result["decision"] = decide(result["runs"], result["references"][reference_name])
    result["status"] = "ok" if result["decision"]["complete"] else "awaiting_runs"
    result["primary_metrics"] = {s: {a: r["evaluation"]["event"] for a, r in arms.items()} for s, arms in result["runs"].items()}
    result["conclusion"] = result["decision"]["next"]
    result["cache"] = store.stats.copy()
    write_json(args.run_directory / "selection.json", result)
    if args.output:
        if args.output.exists():
            raise FileExistsError("formal result already exists; choose a new output path")
        write_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "select"))
    parser.add_argument("--run-directory", type=Path, default=RUN_ROOT / "TEMPORAL-SEG-v1")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache-mb", type=int, default=1024)
    parser.add_argument("--pilot-updates", type=int, default=100)
    parser.add_argument("--full-dev", action="store_true", help="time full Train-dev selection during preflight")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.run_directory = args.run_directory.resolve()
    if not args.run_directory.is_relative_to(RUN_ROOT.resolve()) or args.run_directory == RUN_ROOT.resolve():
        raise ValueError("experiment directory must be inside ignored results/runs")
    if not 1 <= args.pilot_updates <= 100 or len(set(args.arms)) != len(args.arms):
        raise ValueError("invalid pilot length or duplicate arms")
    if args.stage == "preflight" and args.seed != 42:
        raise ValueError("fixed preflight uses seed 42")
    if args.stage != "preflight" and (args.pilot_updates != 100 or args.full_dev):
        raise ValueError("pilot options cannot change formal runs")
    if args.output and (args.stage != "select" or args.output.resolve().parent != (ROOT / "results").resolve()):
        raise ValueError("formal output must be a new JSON directly in results/")
    if args.output and (args.output.exists() or args.output.suffix != ".json"):
        raise ValueError("output must be a new .json file")
    device = choose_device(args.device)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.run_directory.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
                   "cpu_threads": 1, "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    store = SequenceStore(args.run_directory / "cache", args.cache_mb)
    started = time.perf_counter()
    result = {"status": "failed"}
    try:
        result = {"preflight": preflight, "train": train, "select": select}[args.stage](args, store, device)
    finally:
        write_json(args.run_directory / f"last-{args.stage}-execution.json", {**report(args.stage, args.run_directory, result["status"]),
                   "environment": environment, "elapsed_s": time.perf_counter() - started, "cache": store.stats})
    print(json.dumps({"status": result["status"], "decision": result.get("decision"),
                      "elapsed_s": time.perf_counter() - started}), flush=True)
