"""Host arithmetic check for the SYNC WV Q4_K/Q5_K x Q8_K dispatch.

Replays the shader's GGUF byte decoding and lane reduction against the GGML
generic CPU dot. This does not execute DXIL; Xbox physical KV parity is the
hardware gate.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import struct
from pathlib import Path


K = 5120
BLOCK = 256
Q8_BYTES = 292


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def weight_value(data: bytes, base: int, lane: int, typ: int) -> float:
    d, dmin = struct.unpack_from("<ee", data, base)
    group = lane >> 5
    if group < 4:
        scale = data[base + 4 + group] & 63
        minimum = data[base + 8 + group] & 63
    else:
        scale = (data[base + 8 + group] & 15) | ((data[base + group] >> 6) << 4)
        minimum = (data[base + 8 + group] >> 4) | ((data[base + 4 + group] >> 6) << 4)
    packed = data[base + (16 if typ == 4 else 48) + (lane >> 6) * 32 + (lane & 31)]
    quant = (packed >> (4 if lane & 32 else 0)) & 15
    if typ == 5:
        quant |= ((data[base + 16 + (lane & 31)] >> (lane >> 5)) & 1) << 4
    return f32(f32(f32(d * scale) * quant) - f32(dmin * minimum))


def shader_dot(weight: bytes, q8: bytes, typ: int) -> float:
    weight_bytes = 144 if typ == 4 else 176
    lanes = [0.0] * BLOCK
    for block in range(K // BLOCK):
        wb = block * weight_bytes
        xb = block * Q8_BYTES
        activation_scale = struct.unpack_from("<f", q8, xb)[0]
        for lane in range(BLOCK):
            quant = struct.unpack_from("<b", q8, xb + 4 + lane)[0]
            activation = f32(activation_scale * quant)
            weight_float = weight_value(weight, wb, lane, typ)
            lanes[lane] = f32(lanes[lane] + f32(weight_float * activation))
    stride = BLOCK // 2
    while stride:
        for lane in range(stride):
            lanes[lane] = f32(lanes[lane] + lanes[lane + stride])
        stride //= 2
    return lanes[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-dll", required=True, type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.ggml_dll.resolve()))
    lib.dflash_sync_quantize.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_void_p, ctypes.c_int)
    lib.dflash_sync_dot.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int)
    lib.dflash_sync_dot.restype = ctypes.c_float
    rng = random.Random(0xDF1A5)
    worst = (0.0, 0, 0, 0.0, 0.0)
    outputs = 0
    for typ in (4, 5):
        for row in range(4):
            values = [f32(rng.uniform(-0.2, 0.2)) for _ in range(K)]
            if row == 3:
                values = [f32(value * ((i % 7) - 3)) for i, value in enumerate(values)]
            source = (ctypes.c_float * K)(*values)
            weight = ctypes.create_string_buffer(K // BLOCK * (144 if typ == 4 else 176))
            lib.dflash_sync_quantize(typ, source, weight, K)
            for batch in range(3):
                features = [f32(rng.uniform(-2.0, 2.0)) for _ in range(K)]
                if batch == 1:
                    features = [f32(x * 16.0) for x in features]
                if batch == 2:
                    features = [0.0 if i % 3 else f32(x * 0.125)
                                for i, x in enumerate(features)]
                activation = ctypes.create_string_buffer(K // BLOCK * Q8_BYTES)
                lib.dflash_sync_quantize(8, (ctypes.c_float * K)(*features), activation, K)
                expected = float(lib.dflash_sync_dot(typ, weight, activation, K))
                actual = shader_dot(weight.raw, activation.raw, typ)
                error = abs(expected - actual)
                if error > worst[0]:
                    worst = (error, typ, row, expected, actual)
                if not math.isclose(expected, actual, rel_tol=2e-5, abs_tol=2e-4):
                    raise AssertionError(
                        f"Q{typ}_K row={row} batch={batch}: GGML={expected} shader={actual} "
                        f"abs={error}"
                    )
                outputs += 1
    print(f"Q4_K/Q5_K x Q8_K GGML generic parity: {outputs} outputs; "
          f"maxAbs={worst[0]:.9g} type={worst[1]} row={worst[2]} "
          f"GGML={worst[3]:.9g} shader={worst[4]:.9g}")


if __name__ == "__main__":
    main()
