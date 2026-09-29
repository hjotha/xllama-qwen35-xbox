"""Extract reproducible DFL2 state and tensor fixtures from the F0 capture.

Payloads stay in reference.cap. Offsets and SHA-256 hashes identify each input
and output without copying 131 MB of target activations into another artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
F0 = HERE / "f0-2026-09-28"
CAPTURE_MAGIC = b"DFLCAP01"
WIRE_MAGIC = 0x324C4644
WIRE_VERSION = 2
EXPECTED_CAPTURE_SHA256 = "b55ef2a104eb37a89beb26810a31dad7a956c2039329bd41987b0b77ce58bd1f"
MAX_FRAME = 8 * 1024 * 1024
LENGTHS = struct.Struct("<II")
REQUEST = struct.Struct("<IHHQII")
RESPONSE = struct.Struct("<IHHQiIQQQQ")
HELLO = struct.Struct("<IIIIiIIQ")
DRAFT = struct.Struct("<iii")
SYNC_TRIM = struct.Struct("<iI")
OPS = {1: "HELLO", 2: "RESET", 3: "SYNC", 4: "DRAFT", 5: "TRIM", 6: "MOCK", 7: "SYNC_TRIM"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def jsonl(rows: list[dict]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def decode_rows(payload: bytes, count: int, width: int, base_offset: int) -> list[dict]:
    row_size = 4 + width * 4
    require(len(payload) == count * row_size, f"SYNC length {len(payload)} != {count} * {row_size}")
    result = []
    for i in range(count):
        offset = i * row_size
        vector = payload[offset + 4 : offset + row_size]
        position = struct.unpack_from("<i", payload, offset)[0]
        require(position >= 0 and (not result or position == result[-1]["position"] + 1), "SYNC positions not contiguous")
        result.append({
            "position": position,
            "vector_offset": base_offset + offset + 4,
            "vector_bytes": len(vector),
            "vector_sha256": sha256(vector),
        })
    return result


def decode_vectors(payload: bytes, count: int, width: int, base_offset: int) -> list[dict]:
    row_size = width * 4
    require(len(payload) == count * row_size, f"vector length {len(payload)} != {count} * {row_size}")
    return [
        {
            "offset": base_offset + i * row_size,
            "bytes": row_size,
            "sha256": sha256(payload[i * row_size : (i + 1) * row_size]),
        }
        for i in range(count)
    ]


def read_capture(path: Path) -> tuple[list[dict], list[dict], dict, dict]:
    frames: list[dict] = []
    drafts: list[dict] = []
    hello: dict | None = None
    pending_sync: dict | None = None
    with path.open("rb") as stream:
        require(stream.read(8) == CAPTURE_MAGIC, "capture magic invalid")
        while (length_bytes := stream.read(LENGTHS.size)):
            index = len(frames)
            require(len(length_bytes) == LENGTHS.size, f"record {index}: truncated lengths")
            req_len, rsp_len = LENGTHS.unpack(length_bytes)
            require(REQUEST.size <= req_len <= MAX_FRAME, f"record {index}: request length {req_len}")
            require(RESPONSE.size <= rsp_len <= MAX_FRAME, f"record {index}: response length {rsp_len}")
            request_offset = stream.tell()
            request = stream.read(req_len)
            response_offset = stream.tell()
            response = stream.read(rsp_len)
            require(len(request) == req_len and len(response) == rsp_len, f"record {index}: truncated frame")
            r_magic, r_ver, op, cycle_id, n_tokens, req_payload_len = REQUEST.unpack_from(request)
            s_magic, s_ver, s_op, s_cycle, status, rsp_payload_len, *_timing = RESPONSE.unpack_from(response)
            req_payload, rsp_payload = request[REQUEST.size:], response[RESPONSE.size:]
            require((r_magic, r_ver, s_magic, s_ver) == (WIRE_MAGIC, WIRE_VERSION, WIRE_MAGIC, WIRE_VERSION), f"record {index}: wire magic/version")
            require(op in OPS and (s_op, s_cycle) == (op, cycle_id), f"record {index}: response mismatch")
            require(req_payload_len == len(req_payload) and rsp_payload_len == len(rsp_payload), f"record {index}: payload length mismatch")
            require(status == 0, f"record {index}: status {status}")
            frame = {
                "index": index,
                "op": OPS[op],
                "cycle_id": cycle_id,
                "n_tokens": n_tokens,
                "request_offset": request_offset,
                "request_bytes": req_len,
                "response_offset": response_offset,
                "response_bytes": rsp_len,
                "request_payload_sha256": sha256(req_payload),
                "response_payload_sha256": sha256(rsp_payload),
                # Response timestamps occupy bytes 24..55 and vary across runs.
                "response_semantic_sha256": sha256(response[:24] + rsp_payload),
                "status": status,
            }
            if op == 1:
                require(hello is None and n_tokens == 0 and not req_payload and len(rsp_payload) == HELLO.size, "HELLO shape")
                keys = ("n_ctx", "encoder_width", "decoder_width", "top_k", "mask_token", "cache_k_bits", "cache_v_bits", "model_bytes")
                hello = dict(zip(keys, HELLO.unpack(rsp_payload)))
                frame["hello"] = hello
            else:
                require(hello is not None, f"record {index}: missing HELLO")
                if op in (2, 6):
                    require(n_tokens == 0 and not req_payload and not rsp_payload, f"record {index}: control shape")
                    if op == 2:
                        pending_sync = None
                elif op == 3:
                    require(not rsp_payload, f"record {index}: SYNC response payload")
                    frame["rows"] = decode_rows(req_payload, n_tokens, hello["encoder_width"], request_offset + REQUEST.size)
                    require(frame["rows"] and frame["rows"][-1]["position"] < hello["n_ctx"], f"record {index}: SYNC context bounds")
                    require(pending_sync is None, f"record {index}: unconsumed SYNC")
                    pending_sync = frame
                elif op == 7:
                    require(not rsp_payload, f"record {index}: SYNC_TRIM response payload")
                    if n_tokens == 0 and not req_payload:
                        frame["capability_probe"] = True
                    else:
                        require(len(req_payload) >= SYNC_TRIM.size, f"record {index}: SYNC_TRIM prefix")
                        pos0, reset = SYNC_TRIM.unpack_from(req_payload)
                        require(reset in (0, 1), f"record {index}: invalid reset flag")
                        frame.update({"trim_pos0": pos0, "reset": reset})
                        frame["rows"] = decode_rows(req_payload[SYNC_TRIM.size:], n_tokens, hello["encoder_width"], request_offset + REQUEST.size + SYNC_TRIM.size)
                        require(frame["rows"] and frame["rows"][0]["position"] == pos0, f"record {index}: trim/row position")
                        require(frame["rows"][-1]["position"] < hello["n_ctx"], f"record {index}: SYNC_TRIM context bounds")
                        require(pending_sync is None, f"record {index}: unconsumed SYNC_TRIM")
                        pending_sync = frame
                elif op == 4:
                    require(len(req_payload) >= DRAFT.size, f"record {index}: DRAFT prefix")
                    pos0, id_last, n_max = DRAFT.unpack_from(req_payload)
                    require(n_max > 0 and n_tokens == n_max + 1, f"record {index}: DRAFT proposal count")
                    frame.update({"pos0": pos0, "id_last": id_last, "n_max": n_max})
                    frame["anchors"] = decode_vectors(req_payload[DRAFT.size:], n_tokens, hello["decoder_width"], request_offset + REQUEST.size + DRAFT.size)
                    frame["outputs"] = decode_vectors(rsp_payload, n_tokens, hello["decoder_width"], response_offset + RESPONSE.size)
                    require(pending_sync is not None, f"record {index}: DRAFT without preceding SYNC")
                    require(pending_sync["index"] == index - 1, f"record {index}: SYNC and DRAFT not adjacent")
                    require(0 <= pos0 < hello["n_ctx"], f"record {index}: DRAFT context position")
                    drafts.append({"sync": pending_sync, "draft": frame})
                    pending_sync = None
                elif op == 5:
                    # This reference capture has no standalone TRIM. Reject an
                    # unexpected one until its payload is decoded explicitly.
                    raise ValueError(f"record {index}: standalone TRIM is not supported by this fixture format")
            frames.append(frame)
    require(hello is not None and pending_sync is not None and drafts, "incomplete capture state")
    require(pending_sync["op"] == "SYNC_TRIM" and pending_sync["index"] == len(frames) - 1, "unexpected terminal operation")
    require(pending_sync["cycle_id"] == drafts[-1]["draft"]["cycle_id"] + 1, "terminal SYNC_TRIM cycle ID")
    require(pending_sync["trim_pos0"] == drafts[-1]["draft"]["pos0"], "terminal SYNC_TRIM trim position")
    return frames, drafts, hello, pending_sync


def correlate_cycles(path: Path, drafts: list[dict]) -> list[dict]:
    traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(traces) == len(drafts), f"trace cycles {len(traces)} != captured drafts {len(drafts)}")
    cycles = []
    for i, (trace, pair) in enumerate(zip(traces, drafts)):
        sync, draft = pair["sync"], pair["draft"]
        require(trace["cycle_id"] == draft["cycle_id"], f"cycle {i}: ID mismatch")
        require(trace["target_position"] == draft["pos0"], f"cycle {i}: target position mismatch")
        require(trace["n_proposed"] == draft["n_max"], f"cycle {i}: proposal count mismatch")
        require(0 <= trace["n_accepted"] <= draft["n_max"], f"cycle {i}: acceptance count")
        cycles.append({
            "index": i,
            "cycle_id": draft["cycle_id"],
            "sync_frame": sync["index"],
            "sync_op": sync["op"],
            "draft_frame": draft["index"],
            "target_position": draft["pos0"],
            "id_last": draft["id_last"],
            "n_proposed": draft["n_max"],
            "n_accepted": trace["n_accepted"],
            "sync_input_sha256": sync["request_payload_sha256"],
            "draft_input_sha256": draft["request_payload_sha256"],
            "draft_output_sha256": draft["response_payload_sha256"],
        })
    return cycles


def timing_summary(trace: Path) -> dict:
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {}
    for key in ("xbox_compute_us", "sync_xbox_compute_us", "draft_rpc_us", "sync_rpc_us"):
        values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
        require(len(values) == 182, f"incomplete timing field: {key}")
        result[key] = {
            "count": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": statistics.median(values),
            "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1],
            "min": values[0],
            "max": values[-1],
        }
    return result


def log_stage_counts(path: Path) -> dict[str, int]:
    stages: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = re.search(r"DFTRACE stage=([^ ]+)", line)
            if match:
                stages[match.group(1)] += 1
    return dict(sorted(stages.items()))


def build_files(capture: Path, trace: Path, server_log: Path) -> dict[str, bytes]:
    capture_hash = file_sha256(capture)
    require(capture_hash == EXPECTED_CAPTURE_SHA256, f"unexpected capture SHA-256: {capture_hash}")
    frames, drafts, hello, terminal_sync = read_capture(capture)
    cycles = correlate_cycles(trace, drafts)
    counts = dict(sorted(Counter(frame["op"] for frame in frames).items()))
    fused = [frame for frame in frames if frame["op"] == "SYNC_TRIM" and not frame.get("capability_probe")]
    require(len(frames) == 369 and len(cycles) == 182, f"F0 frame/cycle count: {len(frames)}/{len(cycles)}")
    require(counts == {"DRAFT": 182, "HELLO": 1, "MOCK": 1, "RESET": 1, "SYNC": 1, "SYNC_TRIM": 183}, f"F0 op counts: {counts}")
    require(sum(bool(f.get("capability_probe")) for f in frames) == 1, "F0 capability probe count")
    require(sum(c["n_proposed"] for c in cycles) == 728 and sum(c["n_accepted"] for c in cycles) == 329, "F0 proposal/acceptance totals")
    frame_bytes, cycle_bytes = jsonl(frames), jsonl(cycles)
    manifest = {
        "format": "dflash-f0-reference-fixtures-v1",
        "capture": capture.name,
        "capture_bytes": capture.stat().st_size,
        "capture_sha256": capture_hash,
        "trace": trace.name,
        "trace_sha256": file_sha256(trace),
        "server_log_sha256": file_sha256(server_log),
        "frames": len(frames),
        "cycles": len(cycles),
        "op_counts": counts,
        "hello": hello,
        "proposals": sum(c["n_proposed"] for c in cycles),
        "accepted": sum(c["n_accepted"] for c in cycles),
        "terminal_sync_frame": terminal_sync["index"],
        "terminal_sync_cycle_id": terminal_sync["cycle_id"],
        "terminal_sync_rows": terminal_sync["n_tokens"],
        "fused_sync_trim": {
            "count": len(fused),
            "reset_flag_counts": {str(k): v for k, v in sorted(Counter(frame["reset"] for frame in fused).items())},
            "row_count_distribution": {str(k): v for k, v in sorted(Counter(frame["n_tokens"] for frame in fused).items())},
            "trim_pos0_min": min(frame["trim_pos0"] for frame in fused),
            "trim_pos0_max": max(frame["trim_pos0"] for frame in fused),
        },
        "frames_jsonl_sha256": sha256(frame_bytes),
        "cycles_jsonl_sha256": sha256(cycle_bytes),
        "cpu_profile": {
            "status": "operator_groups_unavailable_in_existing_instrumentation",
            "xbox_source": "uwp/dflash-rpc/dedicated.cpp:249-252,285-287 times a whole llama_decode per SYNC/DRAFT",
            "target_log_stages": log_stage_counts(server_log),
            "available_aggregate_microseconds": timing_summary(trace),
            "limitation": "Neither the DFL2 response nor server.log includes Xbox matmul, attention, FFN or conv per-operator durations.",
        },
        "state_cases_present": ["HELLO", "RESET", "MOCK disabled", "empty SYNC_TRIM capability probe", "initial SYNC", "fused SYNC_TRIM before subsequent DRAFT", "182 DRAFT output payloads", "terminal SYNC_TRIM without following DRAFT"],
        "state_cases_absent": ["standalone TRIM", "SYNC_TRIM reset=1", "nonzero status/error responses", "disconnect and reconnect", "rollback after failed call", "arbitrary acceptance sequences", "multiple RESET sessions"],
        "limitations": [
            "Inputs and outputs are original F32 wire bytes; hashes and offsets reference the external capture file.",
            "Accepted counts come from cycles.jsonl, not the RPC response; no target acceptance result is encoded in DRAFT.",
            "Capture proves one concrete 182-cycle sequence and does not cover other prompt, seed, shape, or cache trajectories.",
            "Semantic response hashes exclude the four 64-bit timing fields at response bytes 24..55.",
            "No per-operator Xbox CPU profiling data is present in this capture or cycle trace; only aggregate RPC/compute fields exist.",
        ],
    }
    return {
        "frames.jsonl": frame_bytes,
        "cycles.jsonl": cycle_bytes,
        "manifest.json": canonical_json(manifest) + b"\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=F0 / "reference.cap")
    parser.add_argument("--trace", type=Path, default=F0 / "cycles.jsonl")
    parser.add_argument("--server-log", type=Path, default=F0 / "server.log")
    parser.add_argument("--output-dir", type=Path, default=F0 / "reference-fixtures")
    parser.add_argument("--check", action="store_true", help="regenerate in memory and compare existing files byte for byte")
    args = parser.parse_args()
    outputs = build_files(args.capture, args.trace, args.server_log)
    if args.check:
        for name, content in outputs.items():
            path = args.output_dir / name
            require(path.is_file() and path.read_bytes() == content, f"fixture mismatch: {path}")
        print(f"PASS: {len(outputs)} fixture files match regenerated source byte for byte")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (args.output_dir / name).write_bytes(content)
        print(f"wrote {len(outputs)} fixture files to {args.output_dir}")
    print(f"capture={EXPECTED_CAPTURE_SHA256} frames=369 cycles=182 proposals=728 accepted=329")


if __name__ == "__main__":
    main()
