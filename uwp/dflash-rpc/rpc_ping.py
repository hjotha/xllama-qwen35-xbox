"""Probe the ggml RPC wire protocol and record LAN round trips."""

import argparse
import json
import socket
import struct
import time
from pathlib import Path


def recv_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("ggml RPC socket closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def response(sock, expected):
    size = struct.unpack("<Q", recv_exact(sock, 8))[0]
    if size != expected:
        raise ValueError(f"ggml RPC response size {size}, expected {expected}")
    return recv_exact(sock, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.26")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    rows = []
    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        sock.settimeout(5)
        start = time.perf_counter_ns()
        sock.sendall(struct.pack("<BQ", 14, 24) + bytes(24))
        hello = response(sock, 28)
        hello_us = (time.perf_counter_ns() - start) // 1000
        major, minor, patch = hello[:3]
        if (major, minor, patch) != (6, 0, 1):
            raise ValueError(f"unexpected ggml RPC version {major}.{minor}.{patch}")
        for index in range(args.samples):
            start = time.perf_counter_ns()
            sock.sendall(struct.pack("<BQ", 15, 0))
            count = struct.unpack("<I", response(sock, 4))[0]
            elapsed_us = (time.perf_counter_ns() - start) // 1000
            rows.append({"sample": index, "rtt_us": elapsed_us, "device_count": count})
        if count < 1:
            raise ValueError("Xbox RPC reported zero devices")

    values = sorted(row["rtt_us"] for row in rows)
    result = {
        "host": args.host,
        "port": args.port,
        "version": f"{major}.{minor}.{patch}",
        "hello_us": hello_us,
        "device_count": count,
        "samples": len(values),
        "min_us": values[0],
        "p50_us": values[len(values) // 2],
        "p95_us": values[min(len(values) - 1, int(len(values) * 0.95))],
        "max_us": values[-1],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
