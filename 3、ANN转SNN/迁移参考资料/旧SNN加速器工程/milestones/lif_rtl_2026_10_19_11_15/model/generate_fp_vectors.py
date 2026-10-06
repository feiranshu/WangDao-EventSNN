from __future__ import annotations

import argparse
from pathlib import Path
import random

from precision_models import mitchell_multiply_bits


def write_vectors(path: Path, fmt: str, width: int, count: int, seed: int) -> None:
    rng = random.Random(seed)
    mask = (1 << width) - 1
    path.parent.mkdir(parents=True, exist_ok=True)
    digits = width // 4
    directed = {
        "fp32": [
            (0x3FC00000, 0x3FC00000),
            (0x40000000, 0x3F000000),
            (0x00000000, 0x7F800000),
            (0x7F7FFFFF, 0x40000000),
        ],
        "fp16": [
            (0x3E00, 0x3E00),
            (0x4000, 0x3800),
            (0x0000, 0x7C00),
            (0x7BFF, 0x4000),
        ],
    }[fmt]
    pairs = directed + [(rng.getrandbits(width), rng.getrandbits(width)) for _ in range(count)]
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for a_bits, b_bits in pairs:
            result = mitchell_multiply_bits(a_bits & mask, b_bits & mask, fmt)
            handle.write(f"{a_bits:0{digits}x} {b_bits:0{digits}x} {result:0{digits}x}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2048)
    args = parser.parse_args()
    write_vectors(
        args.output_dir / "fp32_mitchell_vectors.mem",
        "fp32",
        32,
        args.count,
        24307140011 + 32,
    )
    write_vectors(
        args.output_dir / "fp16_mitchell_vectors.mem",
        "fp16",
        16,
        args.count,
        24307140011 + 16,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
