#!/usr/bin/env python3
"""Continue verified v1 checkpoints; finite numerical excursions are advisory."""
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
    p.add_argument("--source-suite", type=Path, default=ROOT/"results/runs/h2pro_fulltrain_v1_s42")
    p.add_argument("--suite", type=Path, default=ROOT/"results/runs/h2pro_fulltrain_v1_1_s42")
    p.add_argument("--output", type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--checkpoint-batches", type=int, default=100)
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    from wangdao.h2pro_snn.stream_data import run_path
    a.suite, a.source_suite = run_path(a.suite), run_path(a.source_suite)
    if not 1 <= a.epochs <= 30 or a.checkpoint_batches < 1:
        p.error("Use total epochs 1..30 and a positive checkpoint interval")
    if a.action == "pause":
        if not a.suite.exists():
            p.error("Training suite has not been prepared")
        (a.suite/"PAUSE").write_text("Pause requested by user\n", encoding="utf-8")
        print("Pause requested; wait for status=paused before closing.")
    elif a.action == "status":
        result = {}
        for name in (".", "ann_state", "ann_linear", "snn_linear"):
            f = a.suite/name/"progress.json"
            result["suite" if name == "." else name] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"status": "not_prepared"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif a.action == "prepare":
        from wangdao.h2pro_snn.training import setup_device
        from wangdao.h2pro_longtrain_v1.data import ResidentData
        from wangdao.h2pro_longtrain_v1_1.models import profile_for
        from wangdao.h2pro_longtrain_v1_1.migration import migrate_suite
        device = setup_device("cuda", profile_for("ann_linear")["training"])
        data = ResidentData(a.teacher, a.encoded, device)
        print(json.dumps(migrate_suite(data, a.source_suite, a.suite), ensure_ascii=False, indent=2))
    elif a.action == "check":
        from wangdao.h2pro_longtrain_v1_1.checks import check
        if a.output is None:
            p.error("check requires --output")
        run_path(a.output.parent)
        if a.output.exists():
            p.error("Check output exists; select a fresh path")
        print(json.dumps(check(a.teacher, a.encoded, a.source_suite, a.output), ensure_ascii=False, indent=2))
    else:
        from wangdao.h2pro_longtrain_v1_1.runtime import run_suite
        r = run_suite(a.teacher, a.encoded, a.suite, a.epochs, a.checkpoint_batches, a.retry_failed)
        print(json.dumps({"status": r["status"], "suite": str(a.suite)}, indent=2))
        if any(v["status"].startswith("failed") for v in r["models"].values()):
            sys.exit(2)


if __name__ == "__main__":
    main()
