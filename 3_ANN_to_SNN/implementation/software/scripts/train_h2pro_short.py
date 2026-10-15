#!/usr/bin/env python3
"""Continue only ANN-linear/SNN-linear to a small cumulative epoch budget."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/"software"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "train", "pause", "status"))
    p.add_argument("--teacher", type=Path, default=ROOT/"results/runs/h2pro_teacher_v1")
    p.add_argument("--encoded", type=Path, default=ROOT/"results/runs/h2pro_encoded_sparse_v1")
    p.add_argument("--suite", type=Path, default=ROOT/"results/runs/h2pro_fulltrain_v1_1_s42")
    p.add_argument("--arms", nargs="+", choices=("ann_linear", "snn_linear"), default=["ann_linear", "snn_linear"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--checkpoint-batches", type=int, default=100)
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    from wangdao.h2pro_snn.stream_data import run_path
    a.suite = run_path(a.suite)
    if a.action == "pause":
        if not a.suite.exists():
            p.error("Training directory does not exist")
        (a.suite/"PAUSE").write_text("Short trial pause requested by user\n", encoding="utf-8")
        print("Pause requested; wait for status=paused.")
    elif a.action == "status":
        rows = {}
        for arm in (".", "ann_state", "ann_linear", "snn_linear"):
            path = a.suite/arm/"progress.json"
            rows["suite" if arm == "." else arm] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        from wangdao.h2pro_shorttrain import execute
        result = execute(a.teacher, a.encoded, a.suite, a.epochs, a.arms,
                         a.checkpoint_batches, a.retry_failed, check_only=a.action == "check")
        if a.action == "train":
            print(json.dumps({"status": result["status"], "suite": str(a.suite),
                              "selected_controls": result["selected_controls"],
                              "requested_total_epochs": a.epochs}, indent=2))
            if result["status"] == "incomplete":
                sys.exit(2)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
