"""Benchmark the Xbox DFR2 binary transport probe over one persistent TCP socket.

This measures transport only. It does not run or emulate the DFlash2 algorithm.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import time

SIZES = (0, 4096, 16384, 65536, 131072, 262144, 524288, 1048576)
REQUEST = struct.Struct("<4sHHQQQII")
RESPONSE = struct.Struct("<4sHHQQQII32s")


def quantile(sorted_values, percent):
    index = (len(sorted_values) - 1) * percent / 100
    lower = int(index)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[min(lower + 1, len(sorted_values) - 1)] * weight


def summary(samples):
    values = sorted(samples)
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": quantile(values, 50),
        "p90_ms": quantile(values, 90),
        "p95_ms": quantile(values, 95),
        "p99_ms": quantile(values, 99),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def read_exact(sock, size):
    out = bytearray(size)
    view = memoryview(out)
    offset = 0
    while offset < size:
        count = sock.recv_into(view[offset:])
        if count == 0:
            raise ConnectionError("Xbox closed the connection")
        offset += count
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.26")
    parser.add_argument("--port", type=int, default=43125)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--sizes", type=lambda value: tuple(int(part) for part in value.split(",")),
                        default=SIZES, help="comma-separated byte counts")
    parser.add_argument("--pairs", help="comma-separated request:response byte counts")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if not args.sizes or any(size < 0 or size > 1048576 for size in args.sizes):
        parser.error("sizes must be between 0 and 1048576 bytes")
    try:
        cases = ([(int(request), int(response)) for request, response in
                  (part.split(":") for part in args.pairs.split(","))]
                 if args.pairs else [(size, 0) for size in args.sizes])
    except ValueError:
        parser.error("pairs must be request:response byte counts")
    if not cases or any(request < 0 or request > 1048576 or response < 0 or response > 1048576
                        for request, response in cases):
        parser.error("pair byte counts must be between 0 and 1048576")

    result = {"host": args.host, "port": args.port, "iterations": args.iterations,
              "warmup": args.warmup, "samples": {}}
    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(15)
        request_id = 0
        for size, response_size in cases:
            payload = os.urandom(size)
            digest = hashlib.sha256(payload).digest() if size else bytes(32)
            full, serialize, wire, xbox_read, xbox_hash = [], [], [], [], []
            for index in range(args.warmup + args.iterations):
                request_id += 1
                start = time.perf_counter_ns()
                command = 3 if response_size else (1 if size == 0 else 2)
                header = REQUEST.pack(b"DFR2", 1, command,
                                      request_id, 0, 0, size, response_size)
                frame = header + payload
                packed = time.perf_counter_ns()
                sock.sendall(frame)
                response = RESPONSE.unpack(read_exact(sock, RESPONSE.size))
                response_payload = read_exact(sock, response_size) if response_size else b""
                done = time.perf_counter_ns()
                magic, version, status, returned_id, read_us, hash_us, length, returned_size, remote_digest = response
                if (magic, version, status, returned_id, length, returned_size, remote_digest) != (
                    b"DFR2", 1, 0, request_id, size, response_size, digest
                ) or any(response_payload):
                    raise ValueError(f"invalid Xbox reply at size={size} iteration={index}: {response[:7]}")
                if index >= args.warmup:
                    full.append((done - start) / 1e6)
                    serialize.append((packed - start) / 1e6)
                    wire.append((done - packed) / 1e6)
                    xbox_read.append(read_us / 1000)
                    xbox_hash.append(hash_us / 1000)
            key = f"{size}:{response_size}" if response_size else str(size)
            result["samples"][key] = {
                "full_cycle": summary(full), "host_serialize": summary(serialize),
                "socket_roundtrip": summary(wire), "xbox_payload_read": summary(xbox_read),
                "xbox_hash": summary(xbox_hash),
            }
            print(json.dumps({"request_bytes": size, "response_bytes": response_size,
                              **result["samples"][key]}, sort_keys=True), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
