"""Capture and replay complete DFL2 request/response frames for CPU parity."""

import argparse
import hashlib
import io
import json
import socket
import struct
from pathlib import Path


MAGIC = b"DFLCAP01"
REQUEST = struct.Struct("<IHHQII")
RESPONSE = struct.Struct("<IHHQiIQQQQ")
RECORD = struct.Struct("<II")
WIRE_MAGIC = 0x324C4644
MAX_FRAME = 8 * 1024 * 1024


def read_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        part = conn.recv(size - len(data))
        if not part:
            raise EOFError(f"short frame: {len(data)}/{size}")
        data.extend(part)
    return bytes(data)


def address(value):
    host, port = value.rsplit(":", 1)
    return host, int(port)


def frame(conn, header_format, payload_index):
    header = read_exact(conn, header_format.size)
    fields = header_format.unpack(header)
    if fields[0] != WIRE_MAGIC or fields[1] != 2:
        raise ValueError("invalid DFL2 header")
    size = fields[payload_index]
    if size > MAX_FRAME:
        raise ValueError(f"oversized DFL2 payload: {size}")
    return header + read_exact(conn, size)


def capture(args):
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    operations = {}
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address(args.listen))
        listener.listen(1)
        print(json.dumps({"phase": "listening", "listen": args.listen}), flush=True)
        client, _ = listener.accept()
        with client, socket.create_connection(address(args.target), timeout=30) as target, destination.open("wb") as output:
            client.settimeout(180)
            target.settimeout(180)
            output.write(MAGIC)
            while True:
                try:
                    request = frame(client, REQUEST, 5)
                except EOFError as error:
                    if str(error) != "short frame: 0/24":
                        raise
                    break
                target.sendall(request)
                response = frame(target, RESPONSE, 5)
                expected = REQUEST.unpack_from(request)
                actual = RESPONSE.unpack_from(response)
                if actual[:4] != expected[:4]:
                    raise ValueError("response does not match request")
                client.sendall(response)
                output.write(RECORD.pack(len(request), len(response)))
                output.write(request)
                output.write(response)
                count += 1
                operations[str(expected[2])] = operations.get(str(expected[2]), 0) + 1
            output.flush()
    print(json.dumps({"phase": "complete", "records": count, "operations": operations,
                      "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}), flush=True)


def replay(args):
    source = Path(args.input)
    count = 0
    with source.open("rb") as capture_file, socket.create_connection(address(args.target), timeout=30) as target:
        target.settimeout(180)
        if capture_file.read(len(MAGIC)) != MAGIC:
            raise ValueError("invalid capture header")
        while lengths := capture_file.read(RECORD.size):
            if len(lengths) != RECORD.size:
                raise ValueError("truncated record header")
            request_size, response_size = RECORD.unpack(lengths)
            if request_size < REQUEST.size or response_size < RESPONSE.size or request_size > MAX_FRAME or response_size > MAX_FRAME:
                raise ValueError("invalid record lengths")
            request = capture_file.read(request_size)
            expected = capture_file.read(response_size)
            if len(request) != request_size or len(expected) != response_size:
                raise ValueError("truncated record")
            target.sendall(request)
            actual = frame(target, RESPONSE, 5)
            # Server timing fields depend on the replay. ABI/status/output must match.
            if actual[:24] != expected[:24] or actual[RESPONSE.size:] != expected[RESPONSE.size:]:
                expected_header = RESPONSE.unpack_from(expected)
                actual_header = RESPONSE.unpack_from(actual)
                raise AssertionError(f"record {count}: DFL2 mismatch: expected={expected_header[:6]} actual={actual_header[:6]}")
            count += 1
    print(json.dumps({"passed": True, "records": count, "capture_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))


def rerecord(args):
    # Pin the full trusted source before sending any request to the executor.
    raw = Path(args.input).read_bytes()
    if len(raw) != 131419032 or hashlib.sha256(raw).hexdigest() != (
        "b55ef2a104eb37a89beb26810a31dad7a956c2039329bd41987b0b77ce58bd1f"
    ):
        raise ValueError("source differs from trusted reference capture")
    source = io.BytesIO(raw)
    if source.read(8) != MAGIC:
        raise ValueError("invalid capture header")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    # Keep incomplete/error responses out of the named fixture.
    output = bytearray(MAGIC)
    phase = count = 0
    with socket.create_connection(address(args.target), timeout=30) as target:
        target.settimeout(180)
        while phase < 4 and count < 64:
            lengths = source.read(RECORD.size)
            if len(lengths) != RECORD.size:
                raise ValueError("capture lacks SYNC/DRAFT/SYNC_TRIM/DRAFT")
            request_size, response_size = RECORD.unpack(lengths)
            if not (REQUEST.size <= request_size <= MAX_FRAME and
                    RESPONSE.size <= response_size <= MAX_FRAME):
                raise ValueError("invalid record lengths")
            request, previous = source.read(request_size), source.read(response_size)
            if len(request) != request_size or len(previous) != response_size:
                raise ValueError("truncated record")
            hdr = REQUEST.unpack_from(request)
            old = RESPONSE.unpack_from(previous)
            if (hdr[:2] != (WIRE_MAGIC, 2) or hdr[5] != request_size - REQUEST.size or
                    old[:4] != hdr[:4] or old[4] or old[5] != response_size - RESPONSE.size):
                raise ValueError("invalid source DFL2 frame")
            target.sendall(request)
            response = frame(target, RESPONSE, 5)
            actual = RESPONSE.unpack_from(response)
            if actual[:4] != hdr[:4] or actual[4] or actual[5] != old[5]:
                raise ValueError(f"invalid CPU response: {actual[:6]}")
            output.extend(RECORD.pack(len(request), len(response)))
            output.extend(request)
            output.extend(response)
            count += 1
            if hdr[2] == (3, 4, 7, 4)[phase] and (phase != 2 or hdr[4] > 0):
                phase += 1
    if phase != 4:
        raise ValueError("capture lacks SYNC/DRAFT/SYNC_TRIM/DRAFT")
    with destination.open("xb") as file:
        file.write(output)
    print(json.dumps({"passed": True, "records": count, "bytes": len(output),
                      "sha256": hashlib.sha256(output).hexdigest()}))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("capture")
    save.add_argument("--listen", default="127.0.0.1:50054")
    save.add_argument("--target", default="192.168.1.26:50053")
    save.add_argument("--output", required=True)
    check = commands.add_parser("replay")
    check.add_argument("--target", default="192.168.1.26:50053")
    check.add_argument("--input", required=True)
    record = commands.add_parser("rerecord", help="recompute trusted reference outputs on CPU")
    record.add_argument("--target", default="192.168.1.26:50053")
    record.add_argument("--input", required=True)
    record.add_argument("--output", required=True)
    args = parser.parse_args()
    {"capture": capture, "replay": replay, "rerecord": rerecord}[args.command](args)


if __name__ == "__main__":
    main()
