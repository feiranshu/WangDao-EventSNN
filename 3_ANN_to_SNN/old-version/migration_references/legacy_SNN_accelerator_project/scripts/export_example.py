from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software"))

from snn_accel.export import LayerSpec, export_model, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=ROOT / "results" / "generated" / "example")
    args = parser.parse_args()
    rng = np.random.default_rng(17)
    layer = LayerSpec(
        name="stem",
        input_channels=2,
        output_channels=8,
        kernel=3,
        stride=1,
        threshold=512,
        leak_shift=3,
        accumulator_bits=20,
        approx_lsb_bits=4,
    )
    weights = {"stem": rng.normal(0, 0.125, size=(8, 2, 3, 3)).astype(np.float32)}
    manifest = export_model(weights, [layer], args.output, model_hash="example")
    verify_manifest(args.output)
    print(f"Exported {len(manifest['layers'])} layer(s) to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

