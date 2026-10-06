from __future__ import annotations

import argparse
import random
from pathlib import Path

from bittrue import approximate_add, exact_add


def twos(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--approx-bits", type=int, default=4)
    args = parser.parse_args()

    rng = random.Random(24307140011)
    minimum = -(1 << (args.width - 1))
    maximum = (1 << (args.width - 1)) - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as stream:
        for _ in range(args.count):
            a = rng.randint(minimum, maximum)
            b = rng.randint(minimum, maximum)
            exact, exact_sat = exact_add(a, b, args.width)
            approx, approx_sat = approximate_add(a, b, args.width, args.approx_bits)
            stream.write(
                f"{twos(a,args.width):05x} {twos(b,args.width):05x} "
                f"{twos(exact,args.width):05x} {int(exact_sat):x} "
                f"{twos(approx,args.width):05x} {int(approx_sat):x}\n"
            )


if __name__ == "__main__":
    main()

