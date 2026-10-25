#!/usr/bin/env python3
"""RKVA hardware parity test — validates the Xbox D3D12 F16 attention kernel.

Connects to the RKVA service, negotiates F16 geometry, creates a session, then
runs ATTN_PREFILL and ATTN_DECODE with synthetic Q/K/V and compares the console
GPU output against a NumPy CPU reference (causal GQA, F16 KV, F32 accumulate).
This isolates the Xbox attention kernel from the 4070/model entirely.
"""
import socket
import struct
import sys
import numpy as np

MAGIC = 0x41564B52
VERSION = 1
OP_HELLO = 1
OP_CREATE_SESSION = 2
OP_ATTN_DECODE = 4
OP_ATTN_PREFILL = 5
OP_PING = 9
OP_GET_STATS = 10

N_HEAD = 24
N_HEAD_KV = 4
HEAD_DIM = 256
KQ_SCALE = 1.0 / np.sqrt(HEAD_DIM)

HOST = "192.168.1.26"
PORT = 50054


def recv_all(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("connection closed")
        buf += chunk
    return buf


class Rkva:
    def __init__(self, host, port):
        self.s = socket.create_connection((host, port), timeout=30)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.cycle = 0

    def call(self, op, session_id, layer_id, n_tokens, payload, expected_bytes):
        self.cycle += 1
        req = struct.pack("<IHHQIIII", MAGIC, VERSION, op, self.cycle,
                          session_id, layer_id, n_tokens, len(payload))
        self.s.sendall(req + payload)
        rsp = recv_all(self.s, 56)
        magic, ver, rop, cycle, status, pbytes = struct.unpack("<IHHQIi", rsp[:24])
        recv_us, prep_us, comp_us, prev_us = struct.unpack("<QQQQ", rsp[24:56])
        assert magic == MAGIC and ver == VERSION and rop == op and cycle == self.cycle, \
            f"header mismatch: {magic:x} {ver} {rop} {cycle}"
        body = recv_all(self.s, pbytes) if pbytes else b""
        return status, body, dict(recv_us=recv_us, prep_us=prep_us, comp_us=comp_us)

    def hello(self):
        # rkva_hello: 16 u32 + f32 + 2 u64
        geo = struct.pack("<IIIIIIIIIIIIIIII f QQ",
                          (1 << 0) | (1 << 1),  # proto_flags: F16+F32
                          16,                   # n_layer_remote
                          N_HEAD, N_HEAD_KV, HEAD_DIM,
                          512,                  # max_ctx
                          0, 0,                 # cache_bits k/v = 0 -> F16 path
                          128, 0,               # group_tokens, sinkhorn_iters
                          0, 2, 0,              # tail_tokens, tail_groups, tail_type
                          0,                    # domain AUTO
                          0, 0,                 # has_sinks, swa
                          KQ_SCALE, 0, 0)
        status, body, _ = self.call(OP_HELLO, 0xFFFFFFFF, 0, 0, geo, 40)
        assert status == 0, f"HELLO status {status}"
        supported, wire, max_sess, srv_ver = struct.unpack("<IIII", body[:16])
        mem_budget = struct.unpack("<Q", body[16:24])[0]
        return dict(supported=supported, wire=wire, max_sess=max_sess,
                    srv_ver=srv_ver, mem_budget=mem_budget)

    def create_session(self, sid, capacity):
        payload = struct.pack("<IIII", sid, 0, capacity, 0)
        status, _, _ = self.call(OP_CREATE_SESSION, sid, 0, 0, payload, 0)
        assert status == 0, f"CREATE_SESSION status {status}"

    def attend(self, sid, layer, positions, q, k, v, decode):
        n = len(positions)
        attn_req = struct.pack("<IIII", 1, 0, 0, 0)  # wire=F32
        payload = attn_req + struct.pack(f"<{n}i", *positions)
        payload += q.astype("<f4").tobytes()
        payload += k.astype("<f4").tobytes()
        payload += v.astype("<f4").tobytes()
        op = OP_ATTN_DECODE if decode else OP_ATTN_PREFILL
        expected = n * N_HEAD * HEAD_DIM * 4
        status, body, timing = self.call(op, sid, layer, n, payload, expected)
        assert status == 0, f"ATTN status {status}"
        out = np.frombuffer(body, dtype="<f4").reshape(n, N_HEAD, HEAD_DIM)
        return out, timing


def cpu_reference(q, k, v, positions, stored_k, stored_v):
    """Causal GQA attention. q [n,24,256] F32; k/v [n,4,256] F32.
    stored_k/v: dict pos -> [4,256] F16 (the KV cache so far, incl. this call)."""
    n = q.shape[0]
    out = np.zeros((n, N_HEAD, HEAD_DIM), dtype=np.float32)
    for i in range(n):
        qp = positions[i]
        for h in range(N_HEAD):
            kvh = h // (N_HEAD // N_HEAD_KV)
            qv = q[i, h].astype(np.float32)
            keys = [p for p in sorted(stored_k) if p <= qp]
            if not keys:
                continue
            K = np.stack([stored_k[p][kvh].astype(np.float32) for p in keys])  # [m,256]
            V = np.stack([stored_v[p][kvh].astype(np.float32) for p in keys])
            scores = (K @ qv) * KQ_SCALE
            m = scores.max()
            e = np.exp(scores - m)
            p = e / e.sum()
            out[i, h] = (p[:, None] * V).sum(0)
    return out


def main():
    np.random.seed(1234)
    r = Rkva(HOST, PORT)
    info = r.hello()
    print(f"HELLO ok: wire={info['wire']} (1=F32) supported=0x{info['supported']:x} "
          f"mem_budget={info['mem_budget']>>20}MiB srv_ver={info['srv_ver']}")
    if not (info["supported"] & (1 << OP_ATTN_DECODE)):
        print("server does not advertise ATTN_DECODE"); sys.exit(2)

    sid = 0
    r.create_session(sid, 512)
    print("CREATE_SESSION ok")

    stored_k, stored_v = {}, {}
    failures = 0

    def run_case(name, positions, decode):
        nonlocal failures
        n = len(positions)
        q = (np.random.randn(n, N_HEAD, HEAD_DIM) * 0.5).astype(np.float32)
        k = (np.random.randn(n, N_HEAD_KV, HEAD_DIM) * 0.5).astype(np.float32)
        v = (np.random.randn(n, N_HEAD_KV, HEAD_DIM) * 0.5).astype(np.float32)
        # The engine stores K/V as F16; mirror that in the reference.
        for i, p in enumerate(positions):
            stored_k[p] = k[i].astype(np.float16)
            stored_v[p] = v[i].astype(np.float16)
        out, timing = r.attend(sid, 3, positions, q, k, v, decode)
        ref = cpu_reference(q, k, v, positions, stored_k, stored_v)
        cos = float((out.flatten() @ ref.flatten()) /
                    (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
        maxabs = float(np.max(np.abs(out - ref)))
        ok = cos > 0.9999 and maxabs < 1e-2
        failures += 0 if ok else 1
        print(f"{name}: n={n} cos={cos:.6f} maxabs={maxabs:.3e} "
              f"xbox_compute_us={timing['comp_us']} {'OK' if ok else 'FAIL'}")

    # Prefill 8 tokens at positions 0..7, then decode tokens 8,9,10 one at a time.
    run_case("prefill[0:8]", list(range(8)), decode=False)
    for p in (8, 9, 10):
        run_case(f"decode[{p}]", [p], decode=True)

    # A second prefill chunk (positions 11..18) to exercise prefill-after-decode.
    run_case("prefill[11:19]", list(range(11, 19)), decode=False)

    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
