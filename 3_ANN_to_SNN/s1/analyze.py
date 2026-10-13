"""S1 read-only, sequence-aware H2-pro Train-dev error grouping.

Run with the evuav Python environment from any working directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
H2 = ROOT / "1_H2_model_improvement"
STUDY = H2 / "seven_step_improvement_study" / "H2-pro"
INDEX = H2 / "implementation" / "data" / "index.json"
FROZEN = STUDY / "frozen" / "H2-pro-Causal-ANN-v1"
CACHE = STUDY / "cache" / "train_dev"
OUT = HERE / "results.json"
sys.path.insert(0, str(H2 / "implementation" / "software"))
from wangdao.h2.numpy_model import _refiner_probability, quantized_mlp  # noqa: E402

GROUPS = ("cross_cell", "first_appearance", "reappearance", "turn", "speed_change", "high_background")
BOOTSTRAPS = 2000
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motion_and_history(labels: np.ndarray, identities: np.ndarray, x: np.ndarray,
                       y: np.ndarray, t: np.ndarray, t0: float) -> dict:
    n = len(labels)
    result = {key: np.zeros(n, dtype=bool) for key in GROUPS}
    eligible = {key: np.ones(n, dtype=bool) for key in GROUPS}
    eligible["cross_cell"][:] = False
    eligible["turn"][:] = False
    eligible["speed_change"][:] = False
    target = np.flatnonzero(labels)
    for identity in np.unique(identities[target]):
        ix = target[identities[target] == identity]
        if not len(ix):
            continue
        result["first_appearance"][ix] = t[ix] - t[ix[0]] < 100.0
        previous = None
        last_reappear = -np.inf
        for j in ix:
            if previous is not None:
                gap = float(t[j] - t[previous])
                if 20.0 <= gap <= 200.0:
                    last_reappear = float(t[j])
                result["reappearance"][j] = float(t[j]) - last_reappear <= 20.0
            previous = int(j)
        bins = np.floor((t[ix] - t0) / 25.0).astype(np.int64)
        centers = {}
        for b in np.unique(bins):
            member = ix[bins == b]
            if len(member) >= 3:
                centers[int(b)] = np.array([x[member].mean(), y[member].mean()])
        for b in centers:
            if b - 1 in centers:
                member = ix[bins == b]
                displacement = float(np.linalg.norm(centers[b] - centers[b-1]))
                previous_cell = np.floor(centers[b-1] / 4).astype(np.int64)
                current_cell = np.floor(centers[b] / 4).astype(np.int64)
                eligible["cross_cell"][member] = True
                result["cross_cell"][member] = bool(displacement >= 4.0 and
                                                       np.any(previous_cell != current_cell))
            if b - 1 not in centers or b - 2 not in centers:
                continue
            previous_v = centers[b-1] - centers[b-2]
            current_v = centers[b] - centers[b-1]
            speed0 = float(np.linalg.norm(previous_v))
            speed1 = float(np.linalg.norm(current_v))
            if min(speed0, speed1) < 2.0:
                continue
            member = ix[bins == b]
            eligible["turn"][member] = True
            eligible["speed_change"][member] = True
            result["turn"][member] = float(np.dot(previous_v, current_v) / (speed0 * speed1)) <= 0.5
            ratio = speed1 / speed0
            result["speed_change"][member] = ratio >= 2.0 or ratio <= 0.5
    return {"exposed": result, "eligible": eligible}


def matched_cells(exposure: np.ndarray, miss: np.ndarray, eligible: np.ndarray,
                  strata: np.ndarray) -> tuple[list, dict]:
    valid = eligible
    exposed = exposure & valid
    control = (~exposure) & valid
    raw = {"eligible": int(valid.sum()), "exposed": int(exposed.sum()),
           "control": int(control.sum()), "exposed_fn": int(miss[exposed].sum()),
           "control_fn": int(miss[control].sum())}
    raw["exposed_fn_rate"] = raw["exposed_fn"] / raw["exposed"] if raw["exposed"] else None
    raw["control_fn_rate"] = raw["control_fn"] / raw["control"] if raw["control"] else None
    cells = []
    for s in range(8):
        one = exposed & (strata == s)
        zero = control & (strata == s)
        n1, n0 = int(one.sum()), int(zero.sum())
        if min(n1, n0) >= 5:
            cells.append((n1, n0, int(miss[one].sum()), int(miss[zero].sum())))
    raw["matched_exposed_events"] = sum(c[0] for c in cells)
    raw["matched_control_events"] = sum(c[1] for c in cells)
    raw["matched_strata"] = len(cells)
    return cells, raw


def aggregate(cells_by_sequence: list[list[tuple]]) -> dict:
    weighted_one = weighted_zero = weight_total = 0.0
    for seq_cells in cells_by_sequence:
        for n1, n0, fn1, fn0 in seq_cells:
            weight = min(n1, n0)
            weighted_one += weight * fn1 / n1
            weighted_zero += weight * fn0 / n0
            weight_total += weight
    if weight_total == 0:
        return {"exposed_rate": None, "control_rate": None, "risk_difference": None, "weight": 0}
    return {"exposed_rate": weighted_one / weight_total,
            "control_rate": weighted_zero / weight_total,
            "risk_difference": (weighted_one - weighted_zero) / weight_total,
            "weight": weight_total}


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    manifest = json.loads((FROZEN / "model_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((STUDY / "results" / "formal_seed42_local_20261010" / "train_dev_selection_v2.json").read_text(encoding="utf-8"))
    assert sha256(INDEX) == manifest["index_sha256"] == selection["index_sha256"]
    assert sha256(STUDY / "protocol.json") == manifest["protocol_sha256"]
    weight_path = FROZEN / "teacher_exact_int8.npz"
    assert sha256(weight_path) == manifest["teacher_int8_sha256"]
    with np.load(weight_path, allow_pickle=False) as weight:
        assert str(weight["variant"].item()) == "C1-proposal-motion"
        assert tuple(weight["dimensions"]) == (36, 64, 32, 4)
        assert int(weight["epoch"].item()) == 10
        assert float(weight["threshold"].item()) == 0.35
        layers = tuple((weight[f"layer{i}_weight"].copy(), weight[f"layer{i}_scale"].copy()) for i in range(3))
    by_name = {Path(item["path"]).name: item for item in index["files"]}
    sequences = []
    cell_store = {key: [] for key in GROUPS}
    totals = {"events": 0, "target_events": 0, "candidate_events": 0,
              "target_candidates": 0, "tp": 0, "fp": 0, "fn": 0}
    for name in index["roles"]["train_dev"]:
        source = H2 / "implementation" / "data" / "EV-UAV-dataset" / "train" / name
        assert sha256(source) == by_name[name]["sha256"]
        with np.load(source, allow_pickle=False) as data:
            full_t = data["ev"]["t"]
            rate = len(full_t) * 1000.0 / max(float(full_t[-1] - full_t[0]), 1e-9)
        cache = CACHE / (Path(name).stem + "_C1-proposal-motion.npz")
        with np.load(cache, allow_pickle=False) as data:
            assert str(data["source_sha256"].item()) == by_name[name]["sha256"]
            assert str(data["protocol_sha256"].item()) == manifest["protocol_sha256"]
            f = data["features"].copy()
            ix = data["indices"].copy()
            labels = data["labels"].copy().astype(bool)
            ids = data["identities"].copy()
            x, y, t = data["x"].copy(), data["y"].copy(), data["t"].copy()
            t0 = float(data["t0"].item())
        assert f.shape == (len(ix), 36) and f.dtype == np.uint8
        assert len(ix) == 0 or (ix[0] >= 0 and ix[-1] < len(labels) and np.all(np.diff(ix) > 0))
        probability = np.empty(len(ix), dtype=np.float32)
        for start in range(0, len(ix), 4096):
            stop = min(start + 4096, len(ix))
            probability[start:stop] = _refiner_probability(quantized_mlp(f[start:stop], layers), 2)
        positive = probability >= 0.35
        tp = int(np.count_nonzero(positive & labels[ix]))
        fp = int(np.count_nonzero(positive & ~labels[ix]))
        fn = int(np.count_nonzero(labels)) - tp
        counts = {"events": len(labels), "target_events": int(labels.sum()),
                  "candidate_events": len(ix), "target_candidates": int(labels[ix].sum()),
                  "tp": tp, "fp": fp, "fn": fn}
        for key in totals:
            totals[key] += counts[key]
        win = np.floor((t - t0) / 100.0).astype(np.int64)
        assert np.all(win >= 0)
        window_events = np.bincount(win)
        window_candidates = np.bincount(win[ix], minlength=len(window_events))
        candidate_fraction = window_candidates / np.maximum(window_events, 1)
        # Offline label-assisted diagnostic. Never use this as an inference feature.
        cell = (y.astype(np.int64) // 4) * 87 + x.astype(np.int64) // 4
        background = np.bincount((win[~labels] * (65 * 87) + cell[~labels]),
                                 minlength=len(window_events) * 65 * 87)
        history = motion_and_history(labels, ids, x, y, t, t0)
        target_slot = np.flatnonzero(labels[ix])
        target_ix = ix[target_slot]
        miss = ~positive[target_slot]
        w = win[target_ix]
        score = f[target_slot, 28]
        local_background = background[w * (65 * 87) + cell[target_ix]]
        history["exposed"]["high_background"][target_ix] = local_background > np.quantile(local_background, 0.75)
        # Match within sequence on three coarse candidate/context variables.
        window_volume = window_events[w]
        candidate_share = candidate_fraction[w]
        strata = ((window_volume > np.median(window_volume)).astype(np.uint8) * 4 +
                  (candidate_share > np.median(candidate_share)).astype(np.uint8) * 2 +
                  (score > np.median(score)).astype(np.uint8))
        group_result = {}
        for key in GROUPS:
            exposure = history["exposed"][key][target_ix]
            eligible = history["eligible"][key][target_ix]
            cells, raw = matched_cells(exposure, miss, eligible, strata)
            cell_store[key].append(cells)
            group_result[key] = raw
        sequences.append({"name": name, "event_rate_eps": rate,
                          "high_density": bool(rate >= 20000), **counts,
                          "candidate_fraction": len(ix) / len(labels),
                          "proposal_target_recall": counts["target_candidates"] / counts["target_events"],
                          "refiner_conditional_recall": tp / counts["target_candidates"],
                          "groups": group_result})
    winner = selection["candidates"]["C1-proposal-motion"]["winner"]["metrics"]["event"]
    assert totals["tp"] == int(winner["true_positive"])
    assert totals["fp"] == int(winner["false_positive"])
    assert totals["fn"] == int(winner["false_negative"])
    selected = selection["candidates"]["C1-proposal-motion"]["winner"]
    metrics = {
        "event_iou": totals["tp"] / (totals["tp"] + totals["fp"] + totals["fn"]),
        "average_precision_historical": selected["ap"],
        "precision": totals["tp"] / (totals["tp"] + totals["fp"]),
        "recall": totals["tp"] / totals["target_events"],
        "high_density_recall": selected["metrics"]["high_density_recall"],
        "candidate_fraction": totals["candidate_events"] / totals["events"],
        "proposal_target_recall": totals["target_candidates"] / totals["target_events"],
        "refiner_conditional_recall": totals["tp"] / totals["target_candidates"],
    }
    assert abs(metrics["event_iou"] - selected["metrics"]["event"]["iou"]) < 1e-12
    rng = np.random.default_rng(SEED)
    high_ids = [i for i, row in enumerate(sequences) if row["high_density"]]
    ordinary_ids = [i for i, row in enumerate(sequences) if not row["high_density"]]
    assert len(high_ids) == 2 and len(ordinary_ids) == 8
    summary = {}
    for key in GROUPS:
        cells = cell_store[key]
        point = aggregate(cells)
        draws = []
        if point["weight"]:
            for _ in range(BOOTSTRAPS):
                sampled = np.concatenate((rng.choice(high_ids, len(high_ids), replace=True),
                                          rng.choice(ordinary_ids, len(ordinary_ids), replace=True)))
                value = aggregate([cells[int(i)] for i in sampled])["risk_difference"]
                if value is not None:
                    draws.append(value)
        seq_diffs = [aggregate([item])["risk_difference"] for item in cells]
        summary[key] = {
            "raw_exposed": sum(row["groups"][key]["exposed"] for row in sequences),
            "raw_control": sum(row["groups"][key]["control"] for row in sequences),
            "raw_exposed_fn": sum(row["groups"][key]["exposed_fn"] for row in sequences),
            "raw_control_fn": sum(row["groups"][key]["control_fn"] for row in sequences),
            "matched": point,
            "matched_sequences": sum(bool(item) for item in cells),
            "matched_exposed_events": sum(row["groups"][key]["matched_exposed_events"] for row in sequences),
            "matched_control_events": sum(row["groups"][key]["matched_control_events"] for row in sequences),
            "sequence_positive_difference": sum(v is not None and v > 0 for v in seq_diffs),
            "sequence_negative_difference": sum(v is not None and v < 0 for v in seq_diffs),
            "bootstrap_95_interval": [float(v) for v in np.quantile(draws, [0.025, 0.975])] if len(draws) >= 100 else None,
            "bootstrap_valid_draws": len(draws),
            "high_density_matched": aggregate([cells[i] for i in high_ids]),
            "ordinary_density_matched": aggregate([cells[i] for i in ordinary_ids]),
            "leave_one_sequence_out_difference": {
                sequences[i]["name"]: aggregate([cells[j] for j in range(len(cells)) if j != i])["risk_difference"]
                for i in range(len(cells))
            },
        }
    result = {"experiment_id": "GEN2-S1-001", "role": "train_dev", "training_epochs": 0,
              "model_id": manifest["model_id"], "threshold": 0.35,
              "index_sha256": manifest["index_sha256"],
              "teacher_int8_sha256": manifest["teacher_int8_sha256"],
              "protocol_sha256": manifest["protocol_sha256"],
              "analysis_protocol": "s1/PROTOCOL.md",
              "analysis_protocol_sha256": sha256(HERE / "PROTOCOL.md"),
              "analysis_code_sha256": sha256(Path(__file__).absolute()),
              "bootstrap_seed": SEED,
              "bootstrap_resamples": BOOTSTRAPS, "overall": totals,
              "metrics": metrics,
              "proposal_false_negatives": totals["target_events"] - totals["target_candidates"],
              "refiner_false_negatives": totals["target_candidates"] - totals["tp"],
              "summary": summary, "sequences": sequences}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"S1 analysis complete: {OUT}")
    print(json.dumps({"overall": totals, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
