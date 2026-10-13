"""Select exploratory Train-dev thresholds and replay S1 target subsets for S2.

The target subsets use labels and identities only after inference. They are
diagnostics, never model inputs. This script reads no held-out split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from core import ResearchANN, ResearchSNN, SNNRunner, class_logits
from train import CACHE, THRESHOLDS, read_index, role_records, windows

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "s1"))
from analyze import GROUPS, motion_and_history  # noqa: E402


HERE = Path(__file__).absolute().parent
TARGET_GROUPS = ("cross_cell", "first_appearance", "reappearance", "turn", "speed_change")
EXTENDED_DIAGNOSTIC_THRESHOLDS = [round(0.65 + 0.05 * i, 2) for i in range(7)]


def predict(kind: str, checkpoint: Path, records: list[dict]) -> list[np.ndarray]:
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["model_kind"] == kind and saved["epoch"] == 5
    model = ResearchANN() if kind == "ann" else ResearchSNN()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    results = []
    with torch.no_grad():
        for record in records:
            features = record["features"]
            probability = np.empty(len(features), dtype=np.float32)
            if kind == "ann":
                for start in range(0, len(features), 4096):
                    end = min(start + 4096, len(features))
                    x = torch.from_numpy(features[start:end].astype(np.float32) / 255.0)
                    probability[start:end] = torch.softmax(class_logits(model(x)), 1)[:, 1].numpy()
            else:
                runner = SNNRunner(model, record["t0"])
                for segment in windows(record):
                    _, probability[segment] = runner.run_window(
                        features[segment], record["x"][segment],
                        record["y"][segment], record["t"][segment])
            assert np.all(np.isfinite(probability))
            results.append(probability)
    return results


def counts(records: list[dict], probability: list[np.ndarray], threshold: float,
           high_only: bool = False) -> dict:
    tp = fp = fn = 0
    for record, candidate_probability in zip(records, probability):
        if high_only and record["event_rate"] < 20000:
            continue
        pred = candidate_probability >= threshold
        target = record["target"].astype(bool)
        tp += int(np.count_nonzero(pred & target))
        fp += int(np.count_nonzero(pred & ~target))
        fn += int(record["labels"].sum()) - int(np.count_nonzero(pred & target))
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def select(records: list[dict], probabilities: list[np.ndarray]) -> dict:
    curve = {}
    for threshold in THRESHOLDS:
        overall = counts(records, probabilities, threshold)
        high = counts(records, probabilities, threshold, high_only=True)
        curve[str(threshold)] = {"overall": overall, "high_density": high}
    best = max(THRESHOLDS, key=lambda t: (
        curve[str(t)]["overall"]["iou"],
        curve[str(t)]["high_density"]["recall"], -t))
    return {"threshold": best, "selection_rule": "max overall Event IoU; tie higher high-density Recall; tie lower threshold",
            "curve": curve}


def subset_summary(record: dict, candidate_probability: np.ndarray,
                   threshold: float, exposed: np.ndarray, eligible: np.ndarray) -> dict:
    target = record["labels"].astype(bool)
    target_group = target & exposed & eligible
    candidate_group = target_group[record["indices"]]
    tp = int(np.count_nonzero(candidate_group & (candidate_probability >= threshold)))
    target_n = int(target_group.sum())
    candidate_n = int(candidate_group.sum())
    return {"target_events": target_n, "target_candidates": candidate_n,
            "proposal_misses": target_n - candidate_n,
            "refiner_tp": tp, "refiner_misses": candidate_n - tp,
            "full_recall": tp / target_n if target_n else None,
            "conditional_recall": tp / candidate_n if candidate_n else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "GEN2-S2-001-revision1")
    parser.add_argument("--output", type=Path, default=HERE / "analysis.json")
    parser.add_argument("--extended-diagnostic", action="store_true",
                        help="record post-hoc 0.65–0.95 curve; do not change preregistered selection")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only result already exists: {args.output}")
    torch.set_num_threads(1)
    index, manifest = read_index()
    records = role_records(index, manifest, "train_dev")
    models = {}
    for kind in ("ann", "snn"):
        checkpoint = args.run / f"{kind}_epoch05.pt"
        saved = torch.load(checkpoint, map_location="cpu")
        assert saved["provenance"]["index_sha256"] == manifest["index_sha256"]
        probabilities = predict(kind, checkpoint, records)
        selection = select(records, probabilities)
        history = json.loads((args.run / f"{kind}_history.json").read_text(encoding="utf-8"))[-1]
        for threshold in THRESHOLDS:
            observed = selection["curve"][str(threshold)]["overall"]
            expected = history["train_dev"]["threshold_curve"][str(threshold)]
            assert (observed["tp"], observed["fp"], observed["fn"]) == (
                expected["tp"], expected["fp"], expected["fn"])
        models[kind] = {"checkpoint": checkpoint.as_posix(), "probability": probabilities,
                        "selection": selection,
                        "fixed_0.35": selection["curve"]["0.35"]}
        if args.extended_diagnostic:
            models[kind]["extended_diagnostic_curve"] = {
                str(threshold): {"overall": counts(records, probabilities, threshold),
                                 "high_density": counts(records, probabilities, threshold, high_only=True)}
                for threshold in EXTENDED_DIAGNOSTIC_THRESHOLDS}
            candidate_labels = np.concatenate([record["target"].astype(bool) for record in records])
            candidate_probability = np.concatenate(probabilities)
            models[kind]["candidate_probability_quantiles"] = {
                label: np.quantile(candidate_probability[candidate_labels == target],
                                   [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist()
                for label, target in (("background", False), ("target", True))}
    groups = {kind: {"fixed_0.35": {}, "selected": {}} for kind in models}
    for record_i, record in enumerate(records):
        path = CACHE / "train_dev" / (Path(record["name"]).stem + "_C1-proposal-motion.npz")
        with np.load(path, allow_pickle=False) as cache:
            labels = cache["labels"].copy().astype(bool)
            identities = cache["identities"].copy()
            x, y, t = cache["x"].copy(), cache["y"].copy(), cache["t"].copy()
        history = motion_and_history(labels, identities, x, y, t, record["t0"])
        for kind, model in models.items():
            for label, threshold in (("fixed_0.35", 0.35),
                                     ("selected", model["selection"]["threshold"])):
                for group in TARGET_GROUPS:
                    result = subset_summary(record, model["probability"][record_i], threshold,
                                            history["exposed"][group], history["eligible"][group])
                    total = groups[kind][label].setdefault(group, {
                        "target_events": 0, "target_candidates": 0,
                        "proposal_misses": 0, "refiner_tp": 0, "refiner_misses": 0})
                    for key in total:
                        total[key] += result[key]
    for kind in groups:
        for label in groups[kind]:
            for result in groups[kind][label].values():
                result["full_recall"] = result["refiner_tp"] / result["target_events"] if result["target_events"] else None
                result["conditional_recall"] = result["refiner_tp"] / result["target_candidates"] if result["target_candidates"] else None
    s1_result = json.loads((HERE.parent / "s1" / "results.json").read_text(encoding="utf-8"))
    for group in TARGET_GROUPS:
        assert groups["ann"]["fixed_0.35"][group]["target_candidates"] == s1_result["summary"][group]["raw_exposed"]
    output = {"experiment_id": "GEN2-S2-001", "role": "train_dev",
              "purpose": "preregistered threshold selection and offline target-subset diagnosis",
              "extended_diagnostic_is_post_hoc": args.extended_diagnostic,
              "run": args.run.as_posix(), "models": {
                  kind: {key: value for key, value in model.items() if key != "probability"}
                  for kind, model in models.items()}, "target_subsets": groups}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"S2 Train-dev analysis complete: {args.output}")
    for kind, model in models.items():
        print(kind, "selected threshold", model["selection"]["threshold"],
              "IoU", model["selection"]["curve"][str(model["selection"]["threshold"])]["overall"]["iou"])


if __name__ == "__main__":
    main()
