from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    variants = {}
    for name in ("exact_int8", "pot_approx"):
        path = args.root / "reports" / name / "metrics.json"
        if not path.exists():
            raise SystemExit(f"missing metrics file: {path}")
        variants[name] = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vivado": "2025.1",
        "variants": variants,
        "checks": {
            "exact_int8_routed": (args.root / "reports/exact_int8/post_route.dcp").exists(),
            "pot_approx_routed": (args.root / "reports/pot_approx/post_route.dcp").exists(),
            "pot_datapath_zero_dsp": variants["pot_approx"]["dsp"] == 0,
            "target_150mhz_met": all(item["wns_ns"] >= 0 for item in variants.values()),
        },
    }
    (args.root / "reports/summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

