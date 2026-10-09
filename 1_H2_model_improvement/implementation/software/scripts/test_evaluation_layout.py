"""Regression tests for the metric-by-split evaluation layout."""

from __future__ import annotations

import csv
import hashlib
import shutil
import tempfile
from pathlib import Path

from rebuild_evaluation_tables import build_accuracy, build_complexity, build_coverage


ROOT = Path(__file__).resolve().parents[4] / "5_model_evaluation"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metric_csvs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/01_metric_comparisons/**/*.csv") if "archive" not in path.parts)


def hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_active_layout() -> None:
    files = metric_csvs(ROOT)
    assert len(files) == 54, len(files)
    assert len(list((ROOT / "A_model_accuracy" / "01_metric_comparisons").glob("**/A??_*/README.md"))) == 22
    for path in files:
        rows = csv_rows(path)
        assert [row["evaluation_id"] for row in rows] == [
            "h2__software_exact_int8", "h2_plus__software_exact_int8", "EV-SpSegNet", "EMS-YOLO"
        ]
        for row in rows:
            if row["value"] == "": assert row["status"] != "complete"
        split = path.name.removesuffix(".csv")
        if split == "validation":
            assert not any("/test/" in row["source_file"].lower() for row in rows)
        if split == "test":
            assert not any("/validation/" in row["source_file"].lower() for row in rows)
    forbidden = []
    for path in ROOT.rglob("*"):
        if "archive" in path.parts or not path.is_file(): continue
        if path.name in {"metrics_summary.csv", "per_sequence_details.csv"} or path.suffix.lower() in {".png", ".svg", ".xlsx", ".xls"}:
            forbidden.append(path)
    assert not forbidden, forbidden
    empty = [path for path in ROOT.rglob("*") if path.is_dir() and "archive" not in path.parts and not any(path.iterdir())]
    assert not empty, empty


def test_b04c_identity() -> None:
    base = ROOT / "B_model_complexity" / "01_metric_comparisons" / "B04_effective_operations"
    for split in ("validation", "test"):
        tables = {}
        for code, folder in (("a", "B04a_effective_mac"), ("b", "B04b_effective_ac"), ("c", "B04c_mac_plus_ac")):
            tables[code] = {row["evaluation_id"]: row for row in csv_rows(base / folder / f"{split}.csv")}
        for evaluation_id in ("h2__software_exact_int8", "h2_plus__software_exact_int8"):
            assert float(tables["c"][evaluation_id]["value"]) == float(tables["a"][evaluation_id]["value"]) + float(tables["b"][evaluation_id]["value"])
        ev_ac = tables["b"]["EV-SpSegNet"]
        assert ev_ac["status"] == "complete" and float(ev_ac["value"]) == 0.0
        ems_ac = tables["b"]["EMS-YOLO"]
        assert ems_ac["status"] == "pending" and ems_ac["value"] == ""


def test_deterministic_rebuild() -> None:
    paths = metric_csvs(ROOT) + [ROOT / "evaluation_coverage.csv"]
    before = hashes(paths)
    build_accuracy(ROOT); build_complexity(ROOT); build_coverage(ROOT, fail_on_incomplete=False)
    after = hashes(paths)
    assert before == after


def test_coverage_rules_for_new_models() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        temp_root = Path(tmp) / "5_model_evaluation"; temp_root.mkdir()
        for category in ("A_model_accuracy", "B_model_complexity", "C_deployment_efficiency_and_hardware_cost"):
            source = ROOT / category / "01_metric_comparisons"
            if source.exists(): shutil.copytree(source, temp_root / category / "01_metric_comparisons")
        registry = csv_rows(ROOT / "model_registry.csv")
        fields = list(registry[0])
        mock = dict(registry[0]); mock.update(evaluation_id="mock_software", model_name="Mock software", evaluation_track="software", model_role="formal", display_order="99")
        with (temp_root / "model_registry.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(registry + [mock])
        _, ok = build_coverage(temp_root, fail_on_incomplete=False); assert not ok
        coverage = {row["evaluation_id"]: row for row in csv_rows(temp_root / "evaluation_coverage.csv")}
        assert coverage["mock_software"]["overall_status"] == "incomplete"
        assert coverage["mock_software"]["c_common"] == "not_required"
        mock.update(evaluation_id="mock_board", model_name="Mock board", evaluation_track="board")
        with (temp_root / "model_registry.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(registry + [mock])
        _, ok = build_coverage(temp_root, fail_on_incomplete=False); assert not ok
        coverage = {row["evaluation_id"]: row for row in csv_rows(temp_root / "evaluation_coverage.csv")}
        assert coverage["mock_board"]["overall_status"] == "incomplete"
        assert coverage["mock_board"]["b_common"] == "not_required"
        assert coverage["mock_board"]["c_common"] == "incomplete"


def main() -> None:
    test_active_layout(); test_b04c_identity(); test_deterministic_rebuild(); test_coverage_rules_for_new_models()
    print("evaluation layout tests: PASS")


if __name__ == "__main__": main()
