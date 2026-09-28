"""Measure the experimental DFL2 DFlash RPC server without changing its state format.

The current Xbox server runs GGML on CPU. Each sample resets the draft KV,
injects synthetic target features, and runs one draft block. Three RPC calls
per sample make this a baseline for the implementation, not the planned
one-request speculative cycle or a GPU benchmark.
"""

import argparse
import hashlib
import json
import socket
import struct
import sys
import time
from pathlib import Path


MAGIC = 0x324C4644  # DFL2 on little-endian x64
VERSION = 2
HELLO, RESET, SYNC, DRAFT, TRIM = 1, 2, 3, 4, 5
REQUEST = struct.Struct("<IHHQII")
RESPONSE = struct.Struct("<IHHQiIQQQQ")
HELLO_PAYLOAD = struct.Struct("<IIIIiIIQ")
DRAFT_PAYLOAD = struct.Struct("<iii")


def read_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        data = sock.recv(remaining)
        if not data:
            raise ConnectionError(f"socket closed with {remaining} bytes unread")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def call(sock, op, cycle_id, n_tokens=0, payload=b""):
    packet = REQUEST.pack(MAGIC, VERSION, op, cycle_id, n_tokens, len(payload)) + payload
    start = time.perf_counter_ns()
    sock.sendall(packet)
    fields = RESPONSE.unpack(read_exact(sock, RESPONSE.size))
    magic, version, response_op, response_id, status, size, receive_us, prepare_us, compute_us, prior_response_us = fields
    if magic != MAGIC or version != VERSION or response_op != op or response_id != cycle_id:
        raise ValueError(f"invalid response header: {fields}")
    if size > 64 * 1024 * 1024:
        raise ValueError(f"unreasonable response payload: {size}")
    data = read_exact(sock, size)
    rtt_us = (time.perf_counter_ns() - start) / 1000
    if status:
        raise RuntimeError(f"DFL2 op={op} cycle={cycle_id} status={status}")
    return data, {
        "rtt_us": rtt_us,
        "xbox_receive_us": receive_us,
        "xbox_prepare_us": prepare_us,
        "xbox_compute_us": compute_us,
        # The server reports this for the PREVIOUS response, not this call.
        "prior_response_us": prior_response_us,
    }


def stats(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample set")

    def percentile(p):
        index = (len(ordered) - 1) * p / 100
        low = int(index)
        fraction = index - low
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * fraction

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
    }


def benchmark(args):
    phases = {name: [] for name in (
        "reset_rtt_us", "sync_rtt_us", "draft_rtt_us", "cycle_rtt_us",
        "sync_xbox_compute_us", "draft_xbox_compute_us", "sync_xbox_prepare_us",
        "draft_xbox_prepare_us", "sync_xbox_receive_us", "draft_xbox_receive_us",
    )}
    first_hash = None
    differing_outputs = 0
    cycle_rows = []
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello_data, hello_timing = call(sock, HELLO, 0)
        if len(hello_data) != HELLO_PAYLOAD.size:
            raise ValueError(f"HELLO returned {len(hello_data)} bytes; expected {HELLO_PAYLOAD.size}")
        n_ctx, enc_width, dec_width, top_k, mask_id, k_bits, v_bits, model_bytes = HELLO_PAYLOAD.unpack(hello_data)
        if not (n_ctx > 0 and enc_width == 25600 and dec_width >= 5120 and top_k == 16 and model_bytes > 0):
            raise ValueError(f"unexpected DFlash2 model shape: {HELLO_PAYLOAD.unpack(hello_data)}")
        if args.sync_tokens < 1 or args.sync_tokens > 64 or args.drafts < 1 or args.drafts > 7:
            raise ValueError("sync_tokens must be 1..64 and drafts must be 1..7")

        # One deterministic F32 row per target position, prepared once and reused.
        features = struct.pack("<f", 0.125) * enc_width
        sync_payload = b"".join(struct.pack("<i", position) + features
                                for position in range(args.sync_tokens))
        # Synthetic logical target embeddings exercise the split backbone only.
        embeddings = struct.pack("<f", 0.03125) * ((args.drafts + 1) * dec_width)
        draft_payload = DRAFT_PAYLOAD.pack(args.sync_tokens, args.anchor_token, args.drafts) + embeddings
        expected_draft_bytes = (args.drafts + 1) * dec_width * 4
        first_cycle = None
        for cycle in range(1, args.warmup + args.iterations + 1):
            _, reset_t = call(sock, RESET, cycle)
            _, sync_t = call(sock, SYNC, cycle, args.sync_tokens, sync_payload)
            output, draft_t = call(sock, DRAFT, cycle, args.drafts + 1, draft_payload)
            if len(output) != expected_draft_bytes:
                raise ValueError(f"cycle {cycle}: {len(output)} draft bytes; expected {expected_draft_bytes}")
            digest = hashlib.sha256(output).hexdigest()
            if first_hash is None:
                first_hash = digest
            elif digest != first_hash:
                differing_outputs += 1
            total = reset_t["rtt_us"] + sync_t["rtt_us"] + draft_t["rtt_us"]
            if cycle == 1:
                first_cycle = {"cycle_rtt_us": total, "reset": reset_t, "sync": sync_t, "draft": draft_t}
            if cycle <= args.warmup:
                continue
            samples = {
                "reset_rtt_us": reset_t["rtt_us"],
                "sync_rtt_us": sync_t["rtt_us"],
                "draft_rtt_us": draft_t["rtt_us"],
                "cycle_rtt_us": total,
                "sync_xbox_compute_us": sync_t["xbox_compute_us"],
                "draft_xbox_compute_us": draft_t["xbox_compute_us"],
                "sync_xbox_prepare_us": sync_t["xbox_prepare_us"],
                "draft_xbox_prepare_us": draft_t["xbox_prepare_us"],
                "sync_xbox_receive_us": sync_t["xbox_receive_us"],
                "draft_xbox_receive_us": draft_t["xbox_receive_us"],
            }
            for name, value in samples.items():
                phases[name].append(value)
            cycle_rows.append({
                "cycle_id": cycle,
                "target_position": None,
                "n_proposed": None,
                "n_accepted": None,
                "draft_block_positions": args.drafts,
                "capture_hidden_us": None,
                "serialize_us": None,
                "send_us": None,
                "network_rtt_us": None,
                "xbox_receive_us": sync_t["xbox_receive_us"] + draft_t["xbox_receive_us"],
                "xbox_prepare_us": sync_t["xbox_prepare_us"] + draft_t["xbox_prepare_us"],
                "xbox_gpu_us": 0,
                "xbox_cpu_compute_us": sync_t["xbox_compute_us"] + draft_t["xbox_compute_us"],
                "xbox_sync_us": None,
                "xbox_response_us": None,
                "target_verify_us": None,
                "commit_us": None,
                "rpc_total_rtt_us": total,
                "total_cycle_us": total,
                "output_sha256": digest,
            })

    result = {
        "kind": "dflash2_xbox_cpu_dedicated_rpc_baseline",
        "limitations": "CPU GGML; synthetic features and embeddings; RESET+SYNC+DRAFT are three round trips; no target verification or acceptance",
        "host": args.host,
        "port": args.port,
        "model": {"bytes": model_bytes, "n_ctx": n_ctx, "encoder_width": enc_width,
                  "decoder_width": dec_width, "top_k": top_k, "mask_id": mask_id,
                  "kv_bits_k": k_bits, "kv_bits_v": v_bits},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "sync_tokens": args.sync_tokens,
        "drafts": args.drafts,
        "hello_rtt_us": hello_timing["rtt_us"],
        "first_cycle_after_model_load": first_cycle,
        "first_output_sha256": first_hash,
        "differing_outputs": differing_outputs,
        "phases_us": {name: stats(values) for name, values in phases.items()},
    }
    return result, cycle_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.26")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--sync-tokens", type=int, default=1)
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--anchor-token", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cycles-output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.timeout <= 0:
        parser.error("warmup >= 0, iterations >= 1, timeout > 0 are required")
    result, cycle_rows = benchmark(args)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if args.cycles_output:
        args.cycles_output.parent.mkdir(parents=True, exist_ok=True)
        args.cycles_output.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in cycle_rows),
            encoding="utf-8",
        )
    sys.stdout.write(encoded + "\n")


if __name__ == "__main__":
    main()
