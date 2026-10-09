#!/usr/bin/env python3
import argparse
import array
import pathlib
import sys


def write_i32(path: pathlib.Path, values):
    data = array.array("i", values)
    if data.itemsize != 4:
        raise RuntimeError("当前Python平台的int不是32位")
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def main():
    parser = argparse.ArgumentParser(description="生成KV260 vadd的SD卡输入和黄金结果")
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data"))
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("count必须大于0")

    args.out.mkdir(parents=True, exist_ok=True)
    a = [i % 1000 for i in range(args.count)]
    b = [1000 - (i % 1000) for i in range(args.count)]
    golden = [x + y for x, y in zip(a, b)]

    write_i32(args.out / "input_a.bin", a)
    write_i32(args.out / "input_b.bin", b)
    write_i32(args.out / "golden.bin", golden)
    print(f"已生成 {args.count} 个int32元素；正确输出全部应为1000")


if __name__ == "__main__":
    main()
