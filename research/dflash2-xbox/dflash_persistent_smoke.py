"""Exercise DFL2 KV continuity and rollback with deterministic synthetic inputs."""

import argparse
import hashlib
import json
import socket
import struct
from pathlib import Path

from dflash_dedicated_bench import DRAFT, DRAFT_PAYLOAD, HELLO, HELLO_PAYLOAD, RESET, SYNC, TRIM, call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.26")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--drafts", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 2 or not 1 <= args.drafts <= 7:
        parser.error("cycles must be >= 2 and drafts must be 1..7")

    with socket.create_connection((args.host, args.port), timeout=120) as sock:
        sock.settimeout(120)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello, _ = call(sock, HELLO, 0)
        n_ctx, enc_width, dec_width, top_k, mask_id, k_bits, v_bits, model_bytes = HELLO_PAYLOAD.unpack(hello)
        if enc_width != 25600 or dec_width != 5120 or top_k != 16:
            raise ValueError("unexpected DFlash2 model geometry")
        features = struct.pack("<f", 0.125) * enc_width
        embeddings = struct.pack("<f", 0.03125) * ((args.drafts + 1) * dec_width)
        rows = []

        def run_sequence(start_cycle):
            call(sock, RESET, start_cycle)
            output_hashes = []
            for pos in range(args.cycles):
                cycle_id = start_cycle + pos + 1
                _, sync_t = call(sock, SYNC, cycle_id, 1, struct.pack("<i", pos) + features)
                payload = DRAFT_PAYLOAD.pack(pos + 1, 1, args.drafts) + embeddings
                output, draft_t = call(sock, DRAFT, cycle_id, args.drafts + 1, payload)
                if len(output) != (args.drafts + 1) * dec_width * 4:
                    raise ValueError("wrong draft output size")
                digest = hashlib.sha256(output).hexdigest()
                output_hashes.append(digest)
                _, trim_t = call(sock, TRIM, cycle_id, 0, struct.pack("<i", pos + 1))
                rows.append({"cycle_id": cycle_id, "target_position": pos, "output_sha256": digest,
                             "sync_rtt_us": sync_t["rtt_us"], "draft_rtt_us": draft_t["rtt_us"],
                             "trim_rtt_us": trim_t["rtt_us"],
                             "draft_xbox_cpu_us": draft_t["xbox_compute_us"]})
            return output_hashes

        first = run_sequence(0)
        second = run_sequence(args.cycles)
        result = {"kind": "dflash2_persistent_synthetic_smoke", "model_bytes": model_bytes,
                  "n_ctx": n_ctx, "mask_id": mask_id, "kv_bits": [k_bits, v_bits],
                  "cycles_per_pass": args.cycles, "n_max": args.drafts,
                  "one_connection": True, "resets": 2, "trims": 2 * args.cycles,
                  "same_output_after_reset": first == second, "rows": rows,
                  "limitations": "Synthetic features/embeddings; no target verification or acceptance"}
        if not result["same_output_after_reset"]:
            raise AssertionError("replay after RESET produced different hidden states")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("cycles_per_pass", "n_max", "one_connection", "resets", "trims", "same_output_after_reset")}))


if __name__ == "__main__":
    main()
