#!/usr/bin/env python3
"""Prepare/check/train/pause/status the independent three-control 30-epoch suite."""
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
    p.add_argument("action", choices=("prepare", "check", "train", "pause", "status"))
    p.add_argument("--teacher", type=Path, default=ROOT/"results/runs/h2pro_teacher_v1")
    p.add_argument("--encoded", type=Path, default=ROOT/"results/runs/h2pro_encoded_sparse_v1")
    p.add_argument("--suite", type=Path, default=ROOT/"results/runs/h2pro_fulltrain_v1_s42")
    p.add_argument("--output", type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--checkpoint-batches", type=int, default=100)
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    from wangdao.h2pro_snn.stream_data import run_path
    a.suite = run_path(a.suite)
    if not 1 <= a.epochs <= 30 or not 1 <= a.checkpoint_batches <= 1000:
        p.error("Epochs must be 1..30; checkpoint interval 1..1000")
    if a.action == "pause":
        a.suite.mkdir(parents=True, exist_ok=True)
        (a.suite/"PAUSE").write_text("Pause requested by user\n", encoding="utf-8")
        print("Pause requested. Wait for status=paused before closing the training terminal.")
    elif a.action == "status":
        result = {}
        for name in (".", "ann_state", "ann_linear", "snn_linear"):
            f = a.suite/name/"progress.json"
            result[name] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"status": "not_started"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif a.action == "prepare":
        from wangdao.h2pro_longtrain_v1.data import prepare
        r = prepare(a.teacher, a.encoded)
        print(json.dumps({"status": r["status"], "bytes": r["total_bytes"], "checked_windows": r["bitwise_checked_windows"]}))
    elif a.action == "check":
        from wangdao.h2pro_longtrain_v1.checks import check
        if a.output is None:
            p.error("check requires a new --output path")
        run_path(a.output.parent)
        if a.output.exists():
            p.error("Check output exists")
        print(json.dumps(check(a.teacher, a.encoded, a.output), ensure_ascii=False, indent=2))
    else:
        from wangdao.h2pro_longtrain_v1.runtime import run_suite
        r = run_suite(a.teacher, a.encoded, a.suite, a.epochs, a.checkpoint_batches, a.retry_failed)
        print(json.dumps({"status": r["status"], "suite": str(a.suite)}, indent=2))
        if any(v["status"].startswith("failed") for v in r["models"].values()):
            sys.exit(2)


if __name__ == "__main__":
    main()
