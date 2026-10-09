"""Score a trained causal checkpoint on development roles and freeze working points."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from wangdao.causal_training import (choose_device, load_checkpoint, evaluate_paths,
                                    select_evaluation, environment, FINAL_THRESHOLDS)
from wangdao.phase1_common import ROOT, role_paths, sha256, write_json, PROTOCOL_ID


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--role", choices=["train_dev", "hw_calibration"], default="train_dev")
    p.add_argument("--baseline", type=Path, help="Complete C0 Train-dev evaluation JSON")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists; choose a new filename")
    if a.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(a.threads)
    device = choose_device(a.device)
    model, config, metadata = load_checkpoint(a.checkpoint, device)
    if metadata.get("data_index_sha256") != sha256(ROOT / "data/index.json"):
        p.error("checkpoint and current dataset index differ")
    smoke = a.smoke or bool(metadata.get("smoke"))
    paths = role_paths(a.role, limit=1 if smoke else None)
    predictions = a.output.with_name(a.output.stem + "_predictions")
    result = evaluate_paths(model, paths, device, config, thresholds=FINAL_THRESHOLDS,
                            max_blocks=8 if smoke else None, prediction_dir=predictions)
    baseline = json.loads(a.baseline.read_text(encoding="utf-8")) if a.baseline else None
    selection = (select_evaluation(result, baseline, config.variant) if a.role == "train_dev"
                 else {"eligible": False, "threshold": None, "metrics": None,
                       "reason": "HW-calibration must not select event thresholds"})
    report = {
        "schema_version": 1, "status": "complete", "experiment_id": a.output.stem,
        "stage": "phase1_causal_evaluation", "data_role": a.role,
        "primary_metrics": selection["metrics"] or {}, "decision": "smoke_only" if smoke else
        ("working_point_selected" if selection["eligible"] else "not_eligible"),
        "conclusion": "Software development evaluation; no Test or board measurements.",
        "test_used": False, "validation_used": False, "smoke": smoke,
        "checkpoint": str(a.checkpoint.resolve()), "checkpoint_sha256": sha256(a.checkpoint),
        "checkpoint_metadata": metadata, "model_config": config.to_dict(),
        "data_index_sha256": sha256(ROOT / "data/index.json"), "metric_protocol": PROTOCOL_ID,
        "environment": environment(device), "evaluation": result, "selection": selection,
        "baseline": ({"path": str(a.baseline.resolve()), "sha256": sha256(a.baseline)}
                     if a.baseline else None),
        "prediction_directory": str(predictions.resolve()), "resources": model.resource_summary(),
    }
    write_json(a.output, report)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
