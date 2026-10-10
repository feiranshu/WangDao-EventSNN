"""Export frozen same-prefix H2/H2-pro A/B records into the formal layout.

This transforms existing post-freeze reports; it does not rerun or select models.
The H2-plus reference remains in the report but is not exported as formal.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "5_model_evaluation"
DELIVERY = ROOT / "1_H2_model_improvement" / "implementation" / "results" / "final_H2-pro"
PROTOCOL_A = "evuav-33e2aff-official-compatible-v1+step3-density-20000+closed-prefix-h2pro-v1"
PROTOCOL_B = "formal-ann-algorithm-complexity-closed-prefix-h2pro-v1"
PROTOCOL_TRACK = "evuav-observable-event-track-hota-closed-prefix-h2pro-v1"
WEIGHTS = {
    "H2": "21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2",
    "H2-pro": "6e9221829459825413c58604a40da3582137a555aaa508d84d9da8325aa5368d",
}
IDS = {"H2": "h2__software_exact_int8", "H2-pro": "h2_pro__software_exact_int8"}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export() -> None:
    for split in ("validation", "test"):
        report_path = HERE / "reports" / f"ab_same_prefix_{split}_v2.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for name, evaluation_id in IDS.items():
            a = report["A"][name]
            b = report["B"][name]
            ev = a["event"]
            event = {
                "evaluation_id": evaluation_id,
                "source_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
                "historical_prediction_policy": report["historical_prediction_policy"] if name == "H2" else "causal streaming closed windows",
                "evaluation": {"metric_protocol": PROTOCOL_A, "role": split, "scope": report["scope"]},
                "working_point": {
                    "threshold": a["threshold"], "average_precision": a["ap"],
                    "macro_sequence_recall": a["macro_sequence_recall"],
                    "overall": {
                        "event": {"tp": int(ev["true_positive"]), "fp": int(ev["false_positive"]),
                                  "fn": int(ev["false_negative"]), "tn": int(ev["true_negative"]),
                                  **{key: ev[key] for key in ("iou", "precision", "recall", "f1")}},
                        "localization": a["localization"],
                    },
                    "high_density": {"event": {"recall": a["density_strata"]["high"]["recall"]}},
                    "ordinary_density": {"event": {"recall": a["density_strata"]["ordinary"]["recall"]}},
                    "per_sequence": {},
                },
            }
            track = {
                "evaluation_id": evaluation_id, "source_report": event["source_report"],
                "metric_protocol": PROTOCOL_TRACK,
                "metrics": {"averaged": a["target_track"]["averaged"]},
                "longest_continuous_miss_p95_ms": a["target_track"]["longest_continuous_miss_p95_ms"],
            }
            base_a = EVAL / "A_model_accuracy" / "02_model_records" / "formal" / evaluation_id / split
            write(base_a / "event_level" / "summary.json", event)
            write(base_a / "target_track_level" / "summary.json", track)
            if name == "H2-pro":
                write(DELIVERY / split / "event_summary.json", event)
                write(DELIVERY / split / "track_summary.json", track)
            base_b = EVAL / "B_model_complexity" / "02_model_records" / "formal" / evaluation_id
            profile = {
                "evaluation_id": evaluation_id, "model": name, "protocol": PROTOCOL_B,
                "numeric_path": "exact_int8_replay", "source_weights_sha256": WEIGHTS[name],
                "summary": {"total_weight_parameters": b["B01_parameters"],
                            "total_tensor_payload_bytes": b["B02_tensor_payload_bytes"],
                            "persistent_proposal_history_bytes": b["persistent_proposal_history_bytes"]},
            }
            write(base_b / "common" / "model_profile.json", profile)
            if name == "H2-pro":
                write(DELIVERY / "model_profile.json", profile)
            complexity = {
                "evaluation_id": evaluation_id, "model": name, "protocol": PROTOCOL_B,
                "source_report": event["source_report"],
                "summary": {"sequences": b["sequences"], "closed_windows": b["closed_windows"],
                            "candidate_events": b["candidates"],
                            "dense_macs_per_sequence": b["B03_dense_MAC_per_sequence"],
                            "effective_macs_per_sequence": b["B04a_effective_MAC_per_sequence"],
                            "effective_acs_per_sequence": b["B04b_effective_AC_per_sequence"]},
            }
            write(base_b / split / "summary.json", complexity)
            if name == "H2-pro":
                write(DELIVERY / split / "complexity_summary.json", complexity)


if __name__ == "__main__":
    export()
