"""Freeze a full Train-dev selection, then allow one recorded Validation run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from wangdao.phase1_common import ROOT, PROTOCOL_ID, sha256, write_json
from wangdao.causal_training import load_checkpoint, evaluate_paths
from wangdao.runtime import choose_device, environment


def freeze(evaluation_path: Path, output: Path):
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("data_role") != "train_dev" or report.get("smoke", True):
        raise ValueError("Freeze requires a complete, non-smoke Train-dev evaluation")
    if report.get("metric_protocol") != PROTOCOL_ID:
        raise ValueError("Metric protocol mismatch")
    index_path = ROOT / "data/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    evaluation = report["evaluation"]
    expected = set(index["roles"]["train_dev"])
    actual = [row["sequence"] for row in evaluation["per_sequence"]]
    if evaluation.get("partial") or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Freeze requires all ten indexed Train-dev sequences")
    records = {record["path"]: record for record in index["files"]}
    for row in evaluation["per_sequence"]:
        if row.get("partial") or row["events"] != records["train/" + row["sequence"]]["events"]:
            raise ValueError("Partial/mismatched sequence in Train-dev evaluation")
    if report["data_index_sha256"] != sha256(index_path):
        raise ValueError("Evaluation data index changed")
    selection = report["selection"]
    if not selection.get("eligible") or selection.get("metrics") is None:
        raise ValueError("No feasible Pd/Fa-constrained working point was selected")
    checkpoint = Path(report["checkpoint"])
    if sha256(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after evaluation")
    _, config, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    if metadata.get("smoke", True) or metadata.get("data_index_sha256") != sha256(index_path):
        raise ValueError("Checkpoint is smoke-only or has a different dataset index")
    manifest = {
        "schema_version": 1, "status": "frozen", "experiment_id": output.stem,
        "stage": "phase1_frozen_causal_model", "data_role": "train_dev",
        "primary_metrics": selection["metrics"], "decision": "ready_for_stage_validation",
        "conclusion": "A Train-dev working point is frozen; no Validation/Test score yet.",
        "checkpoint": os.path.relpath(checkpoint.resolve(), output.parent.resolve()),
        "checkpoint_sha256": sha256(checkpoint), "model_config": config.to_dict(),
        "event_threshold": selection["threshold"], "metric_protocol": PROTOCOL_ID,
        "data_index_sha256": sha256(index_path),
        "selection_evidence_sha256": sha256(evaluation_path),
        "test_used": False, "validation_used": False,
        "post_historical_test_development": True,
        "numeric_mode": "floating_ANN",
    }
    write_json(output, manifest)
    return manifest


def validate(manifest_path: Path, output: Path, device):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("metric_protocol") != PROTOCOL_ID:
        raise ValueError("Expected a frozen phase-one manifest")
    checkpoint = (manifest_path.parent / manifest["checkpoint"]).resolve()
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    index_path = ROOT / "data/index.json"
    if sha256(index_path) != manifest["data_index_sha256"]:
        raise ValueError("Frozen dataset index changed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    model, config, metadata = load_checkpoint(checkpoint, device)
    # JSON canonicalization treats checkpoint tuples and JSON lists identically.
    if json.dumps(config.to_dict(), sort_keys=True) != json.dumps(manifest["model_config"], sort_keys=True):
        raise ValueError("Frozen inference configuration changed")
    if metadata.get("smoke", True):
        raise ValueError("Smoke checkpoints cannot enter Validation")
    paths = []
    indexed = {record["path"]: record for record in index["files"]}
    for name in index["roles"]["validation"]:
        if Path(name).name != name or not name.startswith("val_"):
            raise ValueError("Invalid Validation filename")
        path = ROOT / "data/EV-UAV-dataset/val" / name
        if sha256(path) != indexed["val/" + name]["sha256"]:
            raise ValueError("Validation data hash mismatch: " + name)
        paths.append(path)
    if len(paths) != 24 or len(set(paths)) != 24:
        raise ValueError("Expected exactly 24 Validation sequences")
    if output.exists():
        raise FileExistsError(output)
    claim = manifest_path.with_name(manifest_path.stem + "_validation_started.json")
    write_json(claim, {"manifest_sha256": sha256(manifest_path), "output": str(output),
                       "status": "started", "note": "Keep this record after interruptions; do not silently rerun."})
    thresholds = sorted(set([.28, .9, float(manifest["event_threshold"])]))
    result = evaluate_paths(model, paths, device, config, thresholds=thresholds,
                            prediction_dir=output.with_name(output.stem + "_predictions"))
    selected = result["curve"][format(float(manifest["event_threshold"]), ".8g")]
    report = {"schema_version": 1, "status": "complete", "experiment_id": output.stem,
              "stage": "phase1_stage_validation", "data_role": "validation",
              "primary_metrics": selected, "decision": "report_without_retuning",
              "conclusion": "Frozen model stage evaluation; this is not a new blind Test.",
              "test_used": False, "validation_used": True, "post_historical_test_development": True,
              "manifest_sha256": sha256(manifest_path), "environment": environment(device),
              "evaluation": result, "event_threshold": manifest["event_threshold"]}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "freeze":
        result = freeze(args.evaluation, args.output)
    else:
        result = validate(args.manifest, args.output, choose_device(args.device))
    print(json.dumps(result["primary_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
