"""Verify the H2-pro Train-dev Proposal/Refiner error split from frozen caches.

Requires NumPy. Reads only Train-dev caches and the frozen selection record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
STUDY = ROOT / "1_H2_model_improvement" / "seven_step_improvement_study" / "H2-pro"
INDEX = ROOT / "1_H2_model_improvement" / "implementation" / "data" / "index.json"
SELECTION = STUDY / "results" / "formal_seed42_local_20261010" / "train_dev_selection_v2.json"
CACHE = STUDY / "cache" / "train_dev"
OUT = HERE / "train_dev_errors.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    manifest = json.loads((STUDY / "frozen" / "H2-pro-Causal-ANN-v1" / "model_manifest.json").read_text(encoding="utf-8"))
    assert sha256(INDEX) == manifest["index_sha256"] == selection["index_sha256"]
    assert sha256(STUDY / "protocol.json") == manifest["protocol_sha256"]
    by_name = {Path(item["path"]).name: item["sha256"] for item in index["files"]}
    total = targets = candidates = target_candidates = 0
    for name in index["roles"]["train_dev"]:
        path = CACHE / (Path(name).stem + "_C1-proposal-motion.npz")
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["labels"]
            indices = archive["indices"]
            assert str(archive["source_sha256"].item()) == by_name[name]
            assert str(archive["protocol_sha256"].item()) == manifest["protocol_sha256"]
            assert archive["features"].shape == (len(indices), 36)
            assert archive["features"].dtype == np.uint8
            assert np.all(np.diff(indices) > 0) and (len(indices) == 0 or indices[-1] < len(labels))
            total += len(labels)
            targets += int(np.count_nonzero(labels))
            candidates += len(indices)
            target_candidates += int(np.count_nonzero(labels[indices]))
    winner = selection["candidates"]["C1-proposal-motion"]["winner"]
    assert winner["epoch"] == 10 and winner["threshold"] == 0.35
    checkpoint = STUDY / "results" / "formal_seed42_local_20261010" / "C1-proposal-motion" / "epoch010_int8.npz"
    assert sha256(checkpoint) == winner["int8_sha256"]
    with np.load(checkpoint, allow_pickle=False) as source, np.load(
        STUDY / "frozen" / "H2-pro-Causal-ANN-v1" / "teacher_exact_int8.npz", allow_pickle=False
    ) as frozen:
        assert all(np.array_equal(source[key], frozen[key]) for key in source.files)
    event = winner["metrics"]["event"]
    false_negatives = int(event["false_negative"])
    proposal_fn = targets - target_candidates
    refiner_fn = false_negatives - proposal_fn
    assert targets == int(event["true_positive"] + false_negatives)
    assert 0 <= proposal_fn <= false_negatives and refiner_fn >= 0
    result = {
        "scope": "H2-pro frozen C1 exact-int8 seed42 epoch10 threshold 0.35, Train-dev complete 100 ms prefixes",
        "source": [INDEX.relative_to(ROOT).as_posix(), SELECTION.relative_to(ROOT).as_posix()],
        "sequences": len(index["roles"]["train_dev"]), "events": total,
        "target_events": targets, "candidate_events": candidates,
        "candidate_fraction": candidates / total,
        "target_candidates": target_candidates,
        "proposal_false_negatives": proposal_fn,
        "proposal_target_recall": target_candidates / targets,
        "refiner_false_negatives_within_candidates": refiner_fn,
        "refiner_conditional_recall": int(event["true_positive"]) / target_candidates,
        "total_false_negatives": false_negatives,
        "note": "The Proposal split is recomputed from frozen Train-dev caches; total false negatives come from the frozen Train-dev selection record. No Validation/Test events opened.",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
