"""Call the opt-in DFL2 GPU probe and preserve its exact response."""

import argparse
import json
import socket
import struct
import time
from pathlib import Path


REQUEST = struct.Struct("<IHHQII")
RESPONSE = struct.Struct("<IHHQiIQQQQ")


def read_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise EOFError(f"short response {len(data)}/{size}")
        data.extend(chunk)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.26")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--opcode", type=int, choices=(8, 9, 10, 11, 12), default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > 64 or (args.batch != 1 and args.opcode not in (11, 12)):
        parser.error("batch 2..64 is supported only for Q3_K/Q5_K diagnostics")
    start = time.perf_counter()
    with socket.create_connection((args.host, args.port), timeout=300) as conn:
        conn.settimeout(300)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        payload = struct.pack("<I", args.batch) if args.batch != 1 else b""
        conn.sendall(REQUEST.pack(0x324C4644, 2, args.opcode, 1, 0, len(payload)) + payload)
        header = RESPONSE.unpack(read_exact(conn, RESPONSE.size))
        payload = read_exact(conn, header[5])
    wall_ms = (time.perf_counter() - start) * 1000
    if header[:4] != (0x324C4644, 2, args.opcode, 1):
        raise ValueError(f"incorrect response header: {header[:4]}")
    result = {"host": args.host, "port": args.port, "wall_ms": wall_ms,
              "status": header[4], "response_bytes": header[5],
              "xbox_receive_us": header[6], "xbox_prepare_us": header[7],
              "xbox_compute_us": header[8], "xbox_prior_response_us": header[9],
              "probe": json.loads(payload)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if header[4] != 0 or not result["probe"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
