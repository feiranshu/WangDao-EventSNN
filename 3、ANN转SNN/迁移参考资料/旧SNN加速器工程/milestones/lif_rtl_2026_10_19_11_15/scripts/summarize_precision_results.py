from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = (
    "fp32_exact",
    "fp32_approx",
    "fp16_exact",
    "fp16_approx",
    "int8_exact",
    "int8_approx",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report_root = args.root / "reports" / "precision_compare"
    rows = []
    for name in RUNS:
        path = report_root / name / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        row["run"] = name
        rows.append(row)

    csv_path = report_root / "hardware_metrics.csv"
    fields = ["run", "top", "lut", "ff", "dsp", "bram", "wns_ns", "estimated_fmax_mhz"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    comparisons = {}
    by_name = {row["run"]: row for row in rows}
    for precision in ("fp32", "fp16", "int8"):
        exact = by_name[f"{precision}_exact"]
        approximate = by_name[f"{precision}_approx"]
        comparisons[precision] = {
            "lut_change_pct": 100.0 * (approximate["lut"] - exact["lut"]) / max(exact["lut"], 1),
            "ff_change_pct": 100.0 * (approximate["ff"] - exact["ff"]) / max(exact["ff"], 1),
            "dsp_change": approximate["dsp"] - exact["dsp"],
            "fmax_change_pct": 100.0
            * (approximate["estimated_fmax_mhz"] - exact["estimated_fmax_mhz"])
            / max(exact["estimated_fmax_mhz"], 1e-9),
        }
    (report_root / "hardware_comparison.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(csv_path)


if __name__ == "__main__":
    main()
