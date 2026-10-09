from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step5_common import (  # noqa: E402
    REGISTERED_EPOCHS, corrected_noninferiority, enumerate_combinations,
    required_metrics_are_finite, revised_candidate_gates, select_original,
    select_revised,
)


def metric(iou: float, recall: float, pd: float, fa: float) -> dict:
    return {
        "event": {"iou": iou, "recall": recall},
        "localization": {"pd": pd, "fa": fa},
    }


def view(iou: float, threshold: float = 0.3, ap: float = 0.93) -> dict:
    return {
        "eligible": True, "threshold": threshold, "average_precision": ap,
        "overall": metric(iou, 0.86, 0.85, 1.0e-5),
        "high_density": metric(0.48, 0.57, 0.52, 4.0e-5),
        "ordinary_density": metric(0.946, 0.989, 0.978, 6.5e-6),
        "per_sequence": {
            "train_089.npz": metric(0.18, 0.2, 0.3, 2.0e-5),
            "train_092.npz": metric(0.60, 0.7, 0.7, 5.0e-5),
        },
    }


def baselines() -> tuple[dict, dict]:
    h2_r = view(0.7805, ap=0.9274)
    h2_r["high_density"] = metric(0.4753, 0.5691, 0.519, 4.17e-5)
    h2_r["ordinary_density"] = metric(0.9465, 0.9896, 0.978, 6.8e-6)
    h2_r["per_sequence"]["train_089.npz"]["event"]["iou"] = 0.167
    h2_r["per_sequence"]["train_092.npz"]["event"]["iou"] = 0.591
    h2_f = copy.deepcopy(h2_r)
    h2_f["overall"]["localization"].update(pd=0.846, fa=1.3e-5)
    return h2_f, h2_r


def evaluation(epoch: int, low_iou: float, high_iou: float) -> dict:
    low, high = view(low_iou, 0.3), view(high_iou, 0.4)
    def evaluation_payload(row: dict) -> dict:
        return {
            "curve": {"0.3": row["overall"], "0.4": row["overall"]},
            "average_precision": row["average_precision"],
            "density_curves": {
                "high_density": {"0.3": row["high_density"], "0.4": row["high_density"]},
                "ordinary_density": {"0.3": row["ordinary_density"], "0.4": row["ordinary_density"]},
            },
            "per_sequence": [
                {"file": name, "scores": {"0.3": data, "0.4": data}}
                for name, data in row["per_sequence"].items()
            ],
        }
    payload = evaluation_payload(low)
    payload["curve"]["0.4"] = high["overall"]
    payload["density_curves"]["high_density"]["0.4"] = high["high_density"]
    payload["density_curves"]["ordinary_density"]["0.4"] = high["ordinary_density"]
    return {"epoch": epoch, "int8": payload}


class Step5Tests(unittest.TestCase):
    def test_revised_gate_requires_strict_iou_gain(self) -> None:
        h2_f, h2_r = baselines()
        equal = copy.deepcopy(h2_r)
        self.assertFalse(revised_candidate_gates(equal, h2_f, h2_r)["passes"])
        improved = copy.deepcopy(h2_r)
        improved["overall"]["event"]["iou"] += 0.001
        self.assertTrue(revised_candidate_gates(improved, h2_f, h2_r)["passes"])

    def test_missing_or_nan_metric_is_ineligible(self) -> None:
        value = view(0.79)
        value["high_density"]["localization"]["fa"] = float("nan")
        self.assertFalse(required_metrics_are_finite(value))

    def test_noninferiority_enforces_high_density_fa(self) -> None:
        h2_f, h2_r = baselines()
        value = copy.deepcopy(h2_r)
        value["high_density"]["localization"]["fa"] *= 1.11
        result = corrected_noninferiority(value, h2_f, h2_r)
        self.assertFalse(result["passes"])
        self.assertFalse(result["checks"]["high_fa_rise_within_10pct"])

    def test_eligibility_precedes_ranking(self) -> None:
        good = {"epoch": 8, "threshold": 0.4, "view": view(0.786),
                "original_full": {"passes": False}, "original_noninferior": {"passes": True},
                "revised": {"passes": True}}
        bad = {"epoch": 10, "threshold": 0.5, "view": view(0.90),
               "original_full": {"passes": False}, "original_noninferior": {"passes": False},
               "revised": {"passes": False}}
        self.assertEqual(select_original([good, bad])["epoch"], 8)
        self.assertEqual(select_revised([good, bad])["epoch"], 8)

    def test_all_registered_epoch_threshold_pairs_are_enumerated(self) -> None:
        h2_f, h2_r = baselines()
        summary = {
            "configuration": {"registered_epochs": list(REGISTERED_EPOCHS)},
            "h2_f": h2_f, "h2_r": h2_r,
            "evaluations": [evaluation(epoch, 0.781, 0.782) for epoch in REGISTERED_EPOCHS],
        }
        rows = enumerate_combinations(summary)
        self.assertEqual(len(rows), len(REGISTERED_EPOCHS) * 2)
        self.assertEqual({row["epoch"] for row in rows}, set(REGISTERED_EPOCHS))

    def test_unregistered_epoch_cannot_enter_enumeration(self) -> None:
        h2_f, h2_r = baselines()
        rows = [evaluation(epoch, 0.781, 0.782) for epoch in REGISTERED_EPOCHS]
        rows.append(evaluation(6, 0.99, 0.99))
        summary = {"configuration": {"registered_epochs": list(REGISTERED_EPOCHS)},
                   "h2_f": h2_f, "h2_r": h2_r, "evaluations": rows}
        result = enumerate_combinations(summary)
        self.assertNotIn(6, {row["epoch"] for row in result})

    def test_tie_break_prefers_earlier_epoch_then_lower_threshold(self) -> None:
        rows = []
        for epoch, threshold in ((10, 0.4), (8, 0.4), (8, 0.3)):
            rows.append({"epoch": epoch, "threshold": threshold, "view": view(0.786, threshold),
                         "original_full": {"passes": False}, "original_noninferior": {"passes": True},
                         "revised": {"passes": True}})
        self.assertEqual((select_revised(rows)["epoch"], select_revised(rows)["threshold"]), (8, 0.3))

    def test_readme_commands_have_no_backslash_continuations(self) -> None:
        for name in ("README.md", "README_SERVER.md"):
            path = ROOT / name
            if path.exists():
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"\\\s*\n")


if __name__ == "__main__":
    unittest.main()
