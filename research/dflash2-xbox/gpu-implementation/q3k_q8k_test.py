"""Host parity for DFlash F32->Q8_K and Q3_K x Q8_K shaders.

The --ggml-dll is a native test wrapper built from the same ggml checkout. It
exports dflash_ggml_q8_quantize and dflash_ggml_q3q8_dot. This test emulates
the shader's arithmetic and packed byte addresses; it does not execute DXIL.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import runpy
import struct
from pathlib import Path


QK = 256
Q3_BYTES = 110
Q8_BYTES = 292
FC_K = 25600
FC_SAMPLE_ROWS = (0, 1, 127, 512, 1024, 2048, 4095, 5119)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def quantize_shader_order(values: tuple[float, ...]) -> bytes:
    if len(values) % QK:
        raise ValueError("Q8_K input length must divide 256")
    output = bytearray(len(values) // QK * Q8_BYTES)
    for block in range(len(values) // QK):
        segment = values[block * QK : (block + 1) * QK]
        amax = 0.0
        signed_max = 0.0
        for value in segment:
            if abs(value) > amax:
                amax = abs(value)
                signed_max = value
        scale = 0.0 if amax == 0.0 else f32(-127.0 / signed_max)
        d = 0.0 if scale == 0.0 else f32(1.0 / scale)
        base = block * Q8_BYTES
        struct.pack_into("<f", output, base, d)
        quant = [0 if scale == 0.0 else min(127, round(f32(scale * value)))
                 for value in segment]
        struct.pack_into("<256b", output, base + 4, *quant)
        sums = [sum(quant[i * 16 : (i + 1) * 16]) for i in range(16)]
        struct.pack_into("<16h", output, base + 260, *sums)
    return bytes(output)


def ggml_quantize(lib: ctypes.CDLL, raw_f32: bytes) -> bytes:
    if len(raw_f32) % (QK * 4):
        raise ValueError("F32 byte length must divide 1024")
    length = len(raw_f32) // 4
    source = (ctypes.c_float * length).from_buffer_copy(raw_f32)
    target = ctypes.create_string_buffer(length // QK * Q8_BYTES)
    lib.dflash_ggml_q8_quantize(source, target, length)
    return target.raw


def compare_q8(lib: ctypes.CDLL, raw_f32: bytes, label: str) -> bytes:
    values = struct.unpack(f"<{len(raw_f32) // 4}f", raw_f32)
    expected = ggml_quantize(lib, raw_f32)
    actual = quantize_shader_order(values)
    if actual != expected:
        at = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise AssertionError(
            f"{label} Q8 byte {at}: shader arithmetic {actual[at]:02x}, "
            f"GGML {expected[at]:02x}"
        )
    print(f"{label}: {len(values) // QK} Q8_K blocks, {len(actual)} bytes exact")
    return expected


def q3_scaled_integer(raw: bytes, base: int, element: int) -> int:
    half = element >> 7
    quarter = (element >> 5) & 3
    lane32 = element & 31
    low2 = (raw[base + 32 + half * 32 + lane32] >> (quarter * 2)) & 3
    high_bit = (raw[base + lane32] >> (half * 4 + quarter)) & 1
    quant = low2 - (0 if high_bit else 4)
    scale_index = half * 8 + quarter * 2 + (lane32 >> 4)
    lowbyte = raw[base + 96 + (scale_index & 7)]
    highbyte = raw[base + 104 + (scale_index & 3)]
    low4 = (lowbyte >> (4 if scale_index >= 8 else 0)) & 15
    high2 = (highbyte >> (2 * (scale_index >> 2))) & 3
    return quant * ((low4 | (high2 << 4)) - 32)


def dot_shader_order(q3: bytes, q8: bytes, k: int) -> float:
    nb = k // QK
    result = 0.0
    for block in range(nb):
        wbase = block * Q3_BYTES
        xbase = block * Q8_BYTES
        integer_sum = 0
        for element in range(QK):
            q8_value = struct.unpack_from("<b", q8, xbase + 4 + element)[0]
            integer_sum += q3_scaled_integer(q3, wbase, element) * q8_value
        d3 = struct.unpack_from("<e", q3, wbase + 108)[0]
        d8 = struct.unpack_from("<f", q8, xbase)[0]
        result = f32(result + f32(f32(d8 * d3) * integer_sum))
    return result


def check_dot(lib: ctypes.CDLL, weights: bytes, activations_q8: list[bytes]) -> None:
    row_bytes = FC_K // QK * Q3_BYTES
    if len(weights) != len(FC_SAMPLE_ROWS) * row_bytes:
        raise ValueError("FC fixture has wrong row count or K")
    worst = (0.0, 0, 0, 0.0, 0.0)
    errors: list[float] = []
    first_difference: tuple[int, int, float, float, float] | None = None
    first_failure: tuple[int, int, float, float, float] | None = None
    for batch_index, q8 in enumerate(activations_q8):
        if len(q8) != FC_K // QK * Q8_BYTES:
            raise ValueError("Q8 activation has wrong K")
        xbuf = ctypes.create_string_buffer(q8)
        for sample_index, row in enumerate(FC_SAMPLE_ROWS):
            w = weights[sample_index * row_bytes : (sample_index + 1) * row_bytes]
            wbuf = ctypes.create_string_buffer(w)
            expected = float(lib.dflash_ggml_q3q8_dot(wbuf, xbuf, FC_K))
            actual = dot_shader_order(w, q8, FC_K)
            error = abs(actual - expected)
            errors.append(error)
            if error != 0.0 and first_difference is None:
                first_difference = (batch_index, row, error, actual, expected)
            if error > worst[0]:
                worst = (error, batch_index, row, actual, expected)
            if not math.isclose(actual, expected, abs_tol=0.02, rel_tol=2e-5) and first_failure is None:
                first_failure = (batch_index, row, error, actual, expected)
    ordered = sorted(errors)
    p50 = ordered[math.ceil(0.50 * len(ordered)) - 1]
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    print(
        f"FC GGML CPU dot: {len(activations_q8) * len(FC_SAMPLE_ROWS)} outputs, "
        f"p50_abs_err={p50:.9g}, p95_abs_err={p95:.9g}, "
        f"max_abs_err={worst[0]:.9g} at batch={worst[1]} row={worst[2]} "
        f"(shader={worst[3]:.9g}, ggml={worst[4]:.9g})"
    )
    print(f"first nonzero difference: {first_difference}; first tolerance failure: {first_failure}")
    if first_failure is not None:
        raise AssertionError(f"FC Q3_K x Q8_K parity failed: {first_failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-dll", required=True, type=Path)
    parser.add_argument("--capture", type=Path, help="DFLCAP01 with 11 real SYNC positions")
    parser.add_argument("--fc-rows", type=Path, help="eight FC Q3_K rows, indices in FC_SAMPLE_ROWS")
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.ggml_dll.resolve()))
    lib.dflash_ggml_q8_quantize.argtypes = (
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int
    )
    lib.dflash_ggml_q8_quantize.restype = None
    lib.dflash_ggml_q3q8_dot.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    )
    lib.dflash_ggml_q3q8_dot.restype = ctypes.c_float

    rng = random.Random(0xD5A53)
    random_values = [f32(rng.uniform(-5, 5)) for _ in range(QK * 4)]
    random_values[QK : 2 * QK] = [0.0] * QK
    random_values[2 * QK] = 2.0
    random_values[2 * QK + 1] = -2.0
    for index in range(2 * QK + 2, 3 * QK):
        random_values[index] *= 0.1
    compare_q8(lib, struct.pack(f"<{len(random_values)}f", *random_values), "random/zero/tie")

    real_q8: list[bytes] = []
    if args.capture:
        reader = runpy.run_path(str(Path(__file__).with_name("run_gpu_sync_probe.py")),
                                run_name="capture_reader")
        rows, cycle, op = reader["recorded_sync"](args.capture, 11)
        row_bytes = 4 + FC_K * 4
        for index in range(11):
            raw = rows[index * row_bytes + 4 : (index + 1) * row_bytes]
            packed = compare_q8(lib, raw, f"capture cycle={cycle} op={op} row={index}")
            real_q8.append(packed)
    if args.fc_rows and real_q8:
        check_dot(lib, args.fc_rows.read_bytes(), real_q8)


if __name__ == "__main__":
    main()
