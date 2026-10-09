"""Validate formal software/board model admission coverage."""

import argparse
from pathlib import Path

from rebuild_evaluation_tables import build_coverage


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4] / "5_model_evaluation"); parser.add_argument("--allow-incomplete", action="store_true"); args = parser.parse_args()
    path, ok = build_coverage(args.root, fail_on_incomplete=not args.allow_incomplete); print(f"{path}: {'complete' if ok else 'incomplete'}")


if __name__ == "__main__": main()
