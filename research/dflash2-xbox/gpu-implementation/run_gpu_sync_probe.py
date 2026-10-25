"""Run the Xbox SYNC diagnostic with a recorded DFL2 input row batch."""

import argparse
import hashlib
import json
import socket
import struct
import time
from pathlib import Path


CAPTURE_MAGIC = b"DFLCAP01"
RECORD = struct.Struct("<II")
REQUEST = struct.Struct("<IHHQII")
RESPONSE = struct.Struct("<IHHQiIQQQQ")
ROW_BYTES = 4 + 25600 * 4


def read_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        part = sock.recv(count - len(data))
        if not part:
            raise EOFError(f"short response {len(data)}/{count}")
        data.extend(part)
    return bytes(data)


def recorded_sync(path, count):
    with path.open("rb") as source:
        if source.read(8) != CAPTURE_MAGIC:
            raise ValueError("invalid capture magic")
        while lengths := source.read(RECORD.size):
            if len(lengths) != RECORD.size:
                raise ValueError("truncated capture record")
            request_bytes, response_bytes = RECORD.unpack(lengths)
            request = source.read(request_bytes)
            response = source.read(response_bytes)
            if len(request) != request_bytes or len(response) != response_bytes:
                raise ValueError("truncated capture frame")
            magic, version, op, cycle, tokens, payload_bytes = REQUEST.unpack_from(request)
            if magic != 0x324C4644 or version != 2 or payload_bytes != request_bytes - REQUEST.size:
                raise ValueError("invalid DFL2 request")
            if op in (3, 7) and tokens == count:
                rows = request[REQUEST.size + (8 if op == 7 else 0):]
                if len(rows) != tokens * ROW_BYTES:
                    raise ValueError("invalid SYNC row count")
                return rows, cycle, op
    raise ValueError(f"no recorded SYNC with {count} positions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.26")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(3, 5, 11), default=11)
    parser.add_argument("--synthetic-count", type=int, choices=(64,))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, cycle, source_op = recorded_sync(args.input, args.count)
    tokens = args.count
    if args.synthetic_count:
        original = [rows[i * ROW_BYTES:(i + 1) * ROW_BYTES] for i in range(tokens)]
        rows = b"".join(struct.pack("<i", i) + original[i % tokens][4:]
                        for i in range(args.synthetic_count))
        tokens = args.synthetic_count
    started = time.perf_counter()
    with socket.create_connection((args.host, args.port), timeout=300) as sock:
        sock.settimeout(300)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(REQUEST.pack(0x324C4644, 2, 14, cycle, tokens, len(rows)) + rows)
        fields = RESPONSE.unpack(read_exact(sock, RESPONSE.size))
        response = read_exact(sock, fields[5])
    if fields[:4] != (0x324C4644, 2, 14, cycle):
        raise ValueError(f"unexpected DFL2 response {fields[:4]}")
    result = {
        "host": args.host, "port": args.port, "status": fields[4],
        "wall_ms": (time.perf_counter() - started) * 1000,
        "xbox_compute_us": fields[8], "count": tokens,
        "source_count": args.count, "source_op": source_op,
        "synthetic": bool(args.synthetic_count),
        "capture_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "probe": json.loads(response),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if fields[4] or not result["probe"].get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
