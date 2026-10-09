"""Rebuild A01-A22 comparison CSVs from registered model records."""

from pathlib import Path

from rebuild_evaluation_tables import build_accuracy, build_coverage


def main() -> None:
    root = Path(__file__).resolve().parents[4] / "5_model_evaluation"
    for path in build_accuracy(root): print(path)
    build_coverage(root)


if __name__ == "__main__": main()
