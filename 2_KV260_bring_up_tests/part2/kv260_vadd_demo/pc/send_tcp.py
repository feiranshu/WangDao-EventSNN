#!/usr/bin/env python3
import argparse
import array
import socket
import struct
import sys
import time

MAGIC_REQUEST = 0x56414444
MAGIC_RESPONSE = 0x56415253
VERSION = 1


def recv_all(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("连接提前关闭")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def i32_bytes(values):
    data = array.array("i", values)
    if data.itemsize != 4:
        raise RuntimeError("当前Python平台的int不是32位")
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def bytes_i32(payload):
    data = array.array("i")
    data.frombytes(payload)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tolist()


def main():
    parser = argparse.ArgumentParser(description="向KV260发送一次vadd请求并校验结果")
    parser.add_argument("--host", default="192.168.137.2")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--request-id", type=int, default=1)
    args = parser.parse_args()
    if not (0 < args.count <= 16 * 1024 * 1024):
        raise SystemExit("count必须在1到16777216之间")

    a = [i % 1000 for i in range(args.count)]
    b = [1000 - (i % 1000) for i in range(args.count)]
    expected = [x + y for x, y in zip(a, b)]
    header = struct.pack("!4I", MAGIC_REQUEST, VERSION, args.count, args.request_id)

    started = time.perf_counter()
    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        sock.sendall(header)
        sock.sendall(i32_bytes(a))
        sock.sendall(i32_bytes(b))
        response = recv_all(sock, 20)
        magic, version, count, request_id, status = struct.unpack("!5I", response)
        if (magic, version, count, request_id) != (
            MAGIC_RESPONSE,
            VERSION,
            args.count,
            args.request_id,
        ):
            raise RuntimeError("响应头不匹配")
        result = bytes_i32(recv_all(sock, count * 4))
    elapsed = time.perf_counter() - started

    mismatches = sum(x != y for x, y in zip(result, expected))
    mib = (args.count * 3 * 4) / (1024 * 1024)
    print(f"服务端状态={status}，错误元素={mismatches}")
    print(f"端到端时间={elapsed * 1000:.3f} ms，双向载荷={mib:.3f} MiB")
    print("TEST PASSED" if status == 0 and mismatches == 0 else "TEST FAILED")


if __name__ == "__main__":
    main()
