#!/usr/bin/env python3
"""Manual H2-pro Proposal + trainable stream SNN Refiner workflow (single GPU)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Required by deterministic CUDA GEMM; set before importing torch or creating CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/"software"))

from wangdao.h2pro_snn.foundation import ANN_ROOT

BOOTSTRAP = """import sys,runpy
ann,snn=sys.argv[1:3]
sys.argv=['h2pro_snn.cache_worker']+sys.argv[3:]
sys.path.insert(0,ann)
import wangdao
wangdao.__path__.append(snn)
runpy.run_module('wangdao.h2pro_snn.cache_worker',run_name='__main__')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="CPU frozen teacher inference only; no training")
    prepare.add_argument("--dataset-root", type=Path, default=ROOT/"data/EV-UAV-dataset")
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--limit", type=int, help="Per-role sequence limit; marks cache as pilot only")
    prepare.add_argument("--max-windows", type=int, help="Closed prefix limit per sequence; pilot only")
    for name in ("check", "train", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        child.add_argument("--cache", type=Path, required=name != "check")
        if name != "evaluate":
            child.add_argument("--profile", type=Path)
            child.add_argument("--mode", choices=("fixed", "spike_route", "membrane_route"))
            child.add_argument("--batch-size", type=int)
        if name in ("check", "evaluate"):
            child.add_argument("--output", type=Path, required=True)
        if name in ("train", "evaluate"):
            child.add_argument("--run-directory", type=Path, required=True)
        if name == "train":
            child.add_argument("--epochs", type=int, default=5, help="Total epoch budget, including completed epochs")
            child.add_argument("--resume", action="store_true", help="Resume committed epoch, with unchanged code/config/cache")
            child.add_argument("--pilot", action="store_true", help="Allow partial cache; always labels the run as pilot")
    args = parser.parse_args()
    if args.command == "prepare":
        command = [sys.executable, "-B", "-c", BOOTSTRAP, str(ANN_ROOT/"software"),
                   str(ROOT/"software/wangdao"), "--dataset-root", str(args.dataset_root.absolute()),
                   "--cache", str(args.cache.absolute())]
        for name in ("limit", "max_windows"):
            value = getattr(args, name)
            if value is not None:
                if value < 1:
                    parser.error("Limits must be positive")
                command += ["--"+name.replace("_", "-"), str(value)]
        completed = subprocess.run(command, cwd=ROOT,
            env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"))
        raise SystemExit(completed.returncode)
    from wangdao.h2pro_snn.training import check, evaluate_checkpoint, load_profile, train
    from wangdao.h2pro_snn.stream_data import run_path
    if args.command in ("check", "evaluate"):
        run_path(args.output.parent)
        if args.output.exists():
            parser.error("Output exists; choose a new report path")
    if args.command == "evaluate":
        result = evaluate_checkpoint(args.cache, args.run_directory, args.device, args.output)
    else:
        profile = load_profile(args.profile, args.mode, args.batch_size)
        if args.command == "check":
            result = check(profile, args.device, args.output, args.cache)
        else:
            result = train(args.cache, args.run_directory, profile, args.device,
                           args.epochs, resume=args.resume, pilot=args.pilot)
    print(json.dumps({"command": args.command, "status": result.get("status", "evaluated"),
                      "training_started": args.command == "train",
                      "output": str(args.run_directory if args.command == "train" else args.output)}, indent=2))


if __name__ == "__main__":
    main()
