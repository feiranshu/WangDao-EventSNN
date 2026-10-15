#!/usr/bin/env python3
"""Manual fixed v2 stability experiment. Defaults never start training."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "software"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "train", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        child.add_argument("--cache", type=Path, required=True, help="Reuse the verified v1 teacher cache")
        if name != "evaluate":
            child.add_argument("--profile", type=Path)
            child.add_argument("--batch-size", type=int)
        if name in ("check", "evaluate"):
            child.add_argument("--output", type=Path, required=True)
        if name in ("train", "evaluate"):
            child.add_argument("--run-directory", type=Path, required=True)
        if name == "train":
            child.add_argument("--epochs", type=int, default=1, help="Total epochs including completed ones; first health check is one")
            child.add_argument("--resume", action="store_true", help="Only resume the same v2 identity and directory")
            child.add_argument("--pilot", action="store_true", help="Partial cache only; not a full development result")
    args = parser.parse_args()
    from wangdao.h2pro_snn.stream_data import run_path
    from wangdao.h2pro_snn_v2.training import check, evaluate_checkpoint, load_profile, train
    if args.command in ("check", "evaluate"):
        run_path(args.output.parent)
        if args.output.exists():
            parser.error("Output exists; choose a new path")
    if args.command == "evaluate":
        result = evaluate_checkpoint(args.cache, args.run_directory, args.device, args.output)
    else:
        profile = load_profile(args.profile, batch_size=args.batch_size)
        if args.command == "check":
            result = check(profile, args.device, args.output, args.cache)
        else:
            result = train(args.cache, args.run_directory, profile, args.device, args.epochs,
                           resume=args.resume, pilot=args.pilot)
    print(json.dumps({"command": args.command, "status": result.get("status", "evaluated"),
                      "training_started": args.command == "train",
                      "output": str(args.run_directory if args.command == "train" else args.output)}, indent=2))


if __name__ == "__main__":
    main()
