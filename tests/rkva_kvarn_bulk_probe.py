#!/usr/bin/env python3
"""Native bulk vs serial op16/op14 GPU parity; --execute requires console lease.

Same immutable native upload/query bytes in two sessions, covering maximum
256-query envelopes, seals, tail wrap, resident rollback/reappend, a rejected
last segment followed by retry, and both native tail policies. Kernels stay
identical: every OUT/LSE byte must match. Offline default opens no connection.
"""
import argparse
import json
import os
import struct
import time


# Keep the independent NumPy oracle within the single-worker test budget.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"


def run(host, port):
    import numpy as np
    import rkva_kvarn_serial_prefix_probe as p
    import rkva_kvarn_serial_prefix_model as sp
    from rkva_range_probe import Client
    p.CAPACITY = 2048
    client = Client(host, port)
    results = []
    try:
        flags = p.PROTO_F16 | p.PROTO_F32 | p.PROTO_NATIVE_REWIND_V2 | p.PROTO_SERIAL_PREFIX
        ack = struct.unpack('<IIIIQQQ', client.call(1, p.hello_payload(1, flags)))
        assert ack[0] & (1 << 22) and ack[5] & (1 << 8), 'native bulk not negotiated'
        for tail in (0, 128):
            rng = np.random.default_rng(270 + tail)
            layer = p.Layer(np, sp, rng, tail)
            generation = {}
            for session in (0, 1):
                client.call(2, struct.pack('<4I', session, session, p.CAPACITY, 1))
                body = client.call(15, struct.pack('<18I', *p.descriptor(tail)), session=session)
                generation[session] = struct.unpack('<Q', body)[0]
            client.call(20, struct.pack('<IIQ', 0, 2, 0))  # qualified fine/mq2 kernel

            def batch(n, label, negative=False):
                chunks = []
                first = layer.live
                layer.canon.append(n)
                limit = n if n <= 9 else 8
                for offset in range(0, n, limit):
                    count = min(limit, n - offset)
                    live = first + offset + count
                    upload = layer.upload(live, live == count)
                    positions = list(range(p.ORIGIN + live - count, p.ORIGIN + live))
                    q = rng.normal(size=(count, p.H, p.D)).astype('<f4')
                    query = struct.pack('<4I', 1, 0, 0, 0) + struct.pack('<%di' % count, *positions) + q.tobytes()
                    chunks.append((upload, query, count))
                bulk = struct.pack('<4I', 1, len(chunks), n, 0) + b''.join(
                    struct.pack('<2I', len(upload), count) + upload + query
                    for upload, query, count in chunks)
                expected_o, expected_l = [], []
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                start = time.monotonic()
                for upload, query, count in chunks:
                    client.call(16, upload, session=0)
                    generation[0] += 1
                    reply = bytes(client.call(14, query, session=0, nq=count))
                    size = count * p.H * p.D * 4
                    expected_o.append(reply[:size]); expected_l.append(reply[size:])
                serial_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                serial_counters = profile.get('engine', profile)['counters']
                assert serial_counters['submits'] == 2 * len(chunks), serial_counters
                if negative:
                    # Corrupt the last query position: admission must reject
                    # before the first segment mutates the second session.
                    corrupted = bytearray(bulk)
                    last = chunks[-1]
                    pos_offset = len(bulk) - len(last[1]) + 16
                    struct.pack_into('<i', corrupted, pos_offset, -1)
                    client.call(22, corrupted, session=1, nq=n, success=False)
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                start = time.monotonic()
                got = bytes(client.call(22, bulk, session=1, nq=n))
                bulk_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                bulk_counters = profile.get('engine', profile)['counters']
                assert bulk_counters['submits'] == 1, bulk_counters
                assert bulk_counters['fence_blocking_waits'] + bulk_counters['fence_already_complete'] == 1
                generation[1] += len(chunks)
                expected = b''.join(expected_o) + b''.join(expected_l)
                assert got == expected, '%s tail%d OUT/LSE byte mismatch' % (label, tail)
                results.append(dict(label=label, tail=tail, nq=n, live=layer.live,
                                    exact=True, serial_ms=serial_ms, bulk_ms=bulk_ms,
                                    serial_submits=serial_counters['submits'], bulk_submits=bulk_counters['submits'],
                                    bulk_fence_blocking_waits=bulk_counters['fence_blocking_waits'],
                                    bulk_fence_already_complete=bulk_counters['fence_already_complete'],
                                    serial_transactions=2 * len(chunks), bulk_transactions=1))

            batch(256, 'initial maximum envelope, two seals and ring wrap')
            batch(9, 'verification nine')
            batch(64, 'late malformed segment rejected then valid retry', True)
            batch(256, 'append maximum envelope')
            batch(55, 'reach seal640 before rollback')
            for back in (1, 8, 9):
                keep = layer.live - back
                block = layer.rewind_layer(keep)
                for session in (0, 1):
                    head = struct.pack('<QIIII', generation[session], p.ORIGIN + keep, 1, tail, 0)
                    client.call(21, head + block, session=session)
                    generation[session] += 1
                layer.canon.rewind(keep)
                batch(back, 'rollback%d and new-content reappend' % back)
            for session in (0, 1):
                client.call(3, struct.pack('<4I', session, 0, 0, 0), session=session)
    finally:
        client.socket.close()
    return dict(ok=True, server_version=ack[3], checks=results)


def run_cuda_fixtures(host, port, directory):
    import numpy as np
    import rkva_kvarn_serial_prefix_probe as p
    from rkva_kvarn_gpu_fixture import load_fixture
    from rkva_range_probe import Client
    results = []
    oracle_indices = {0, 14, 15, 16, 30, 31, 32, 46, 47, 48, 62, 63, 64, 78, 79}
    for tail in (0, 128):
        fixture = load_fixture(directory, tail)
        p.ORIGIN, p.CAPACITY = fixture.origin, fixture.capacity
        client = Client(host, port)
        try:
            flags = p.PROTO_F16 | p.PROTO_F32 | p.PROTO_NATIVE_REWIND_V2 | p.PROTO_SERIAL_PREFIX
            ack = struct.unpack('<IIIIQQQ', client.call(1, p.hello_payload(1, flags)))
            assert ack[0] & (1 << 22) and ack[5] & (1 << 8), 'native bulk not negotiated'
            for session in (0, 1):
                client.call(2, struct.pack('<4I', session, session, fixture.capacity, 1))
                client.call(15, struct.pack('<18I', *p.descriptor(tail)), session=session)
            for first in range(0, len(fixture.segments), 32):
                segments = fixture.segments[first:first + 32]
                n = 8 * len(segments)
                chunks = []
                for segment in segments:
                    query = struct.pack('<4I', 1, 0, 0, 0) + segment['positions'].tobytes() + segment['q'].tobytes()
                    chunks.append((segment['upload'], query))
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                expected_o, expected_l = [], []
                start = time.monotonic()
                for upload, query in chunks:
                    client.call(16, upload, session=0)
                    reply = bytes(client.call(14, query, session=0, nq=8))
                    expected_o.append(reply[:8 * 24 * 256 * 4])
                    expected_l.append(reply[8 * 24 * 256 * 4:])
                serial_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                serial_counters = profile.get('engine', profile)['counters']
                assert serial_counters['submits'] == 2 * len(chunks)
                bulk = struct.pack('<4I', 1, len(chunks), n, 0) + b''.join(
                    struct.pack('<2I', len(upload), 8) + upload + query for upload, query in chunks)
                client.call(20, struct.pack('<IIQ', 3, 2, 0))
                start = time.monotonic()
                reply = bytes(client.call(22, bulk, session=1, nq=n))
                bulk_ms = (time.monotonic() - start) * 1000
                profile = json.loads(bytes(client.call(20, struct.pack('<IIQ', 8, 0, 0))))
                bulk_counters = profile.get('engine', profile)['counters']
                assert bulk_counters['submits'] == 1
                assert bulk_counters['fence_blocking_waits'] + bulk_counters['fence_already_complete'] == 1
                assert reply == b''.join(expected_o) + b''.join(expected_l), 'CUDA fixture serial/bulk byte mismatch'
                out = np.frombuffer(reply, dtype='<f4', count=n * 24 * 256).reshape(n, 24, 256)
                lse = np.frombuffer(reply, dtype='<f4', offset=n * 24 * 256 * 4).reshape(n, 24)
                checks = []
                for i in range(len(segments)):
                    if first + i not in oracle_indices: continue
                    check = p.compare(np, out[i * 8:(i + 1) * 8], lse[i * 8:(i + 1) * 8],
                                      *fixture.expected(first + i))
                    assert check['ok'], (tail, first + i, check)
                    checks.append(dict(index=first + i, vs_fp64=check))
                results.append(dict(tail=tail, first_segment=first, queries=n, exact=True,
                                    serial_ms=serial_ms, bulk_ms=bulk_ms,
                                    serial_submits=serial_counters['submits'], bulk_submits=bulk_counters['submits'],
                                    bulk_fence_blocking_waits=bulk_counters['fence_blocking_waits'],
                                    bulk_fence_already_complete=bulk_counters['fence_already_complete'],
                                    checks=checks))
        finally:
            client.socket.close()
    return dict(ok=True, cuda_fixtures=str(directory), checks=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.1.26')
    parser.add_argument('--port', type=int, default=50054)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cuda-fixtures', help='real GPU op16 packet fixture directory')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = (run_cuda_fixtures(args.host, args.port, args.cuda_fixtures) if args.cuda_fixtures
              else run(args.host, args.port)) if args.execute else dict(
        execute=False, plan='2 tail policies; exact bulk/serial bytes; seal/wrap/rollback/retry gates')
    text = json.dumps(result, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
