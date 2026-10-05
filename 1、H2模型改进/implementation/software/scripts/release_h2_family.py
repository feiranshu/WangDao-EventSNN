"""Freeze H2-family candidates, then perform one protected Validation/Test pass."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from wangdao.runtime import choose_device
from wangdao.h2_release import evaluate_frozen, freeze_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--accuracy-summary", type=Path, nargs=3, required=True,
                        metavar=("SEED42", "SEED43", "SEED44"))
    freeze.add_argument("--student-summary", type=Path, nargs=3,
                        metavar=("SEED42", "SEED43", "SEED44"))
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--role", choices=["validation", "test"], required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--validation-result", type=Path,
                          help="Required for Test and must match this frozen manifest")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.add_argument("--batch-size", type=int, default=4096)
    evaluate.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_manifest(args.accuracy_summary, args.output, args.student_summary)
    else:
        if args.batch_size < 1 or args.threads < 1:
            parser.error("batch size and threads must be positive")
        if args.role == "test" and args.validation_result is None:
            parser.error("Test requires --validation-result")
        torch.set_num_threads(args.threads)
        result = evaluate_frozen(args.manifest, args.role, args.output,
                                 choose_device(args.device), args.validation_result,
                                 args.batch_size)
    print(json.dumps(result["primary_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
