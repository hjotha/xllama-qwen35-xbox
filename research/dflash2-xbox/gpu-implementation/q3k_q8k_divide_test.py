"""Check candidate FP64/RNE Q8_K scales against ggml's F32 divisions.

Uses the same native ggml wrapper and 11 real SYNC rows as q3k_q8k_test.py.
The shader implementation must still be read back on Xbox separately.
"""

from __future__ import annotations

import argparse
import ctypes
import random
import runpy
import struct
from pathlib import Path


QK = 256
Q8_BYTES = 292
K = 25600


def bits_f32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def double_div_to_f32_rne(numerator: float, denominator: float) -> int:
    """Mirror the HLSL normal-number path using splitDouble and integer RNE."""
    low, high = struct.unpack("<II", struct.pack("<d", numerator / denominator))
    exponent = ((high >> 20) & 0x7FF) - 896
    if not 1 <= exponent <= 254:
        return bits_f32(numerator / denominator)
    mantissa = ((high & 0xFFFFF) << 3) | (low >> 29)
    remainder = low & 0x1FFFFFFF
    if remainder > 0x10000000 or (remainder == 0x10000000 and mantissa & 1):
        mantissa += 1
    if mantissa == 0x800000:
        mantissa = 0
        exponent += 1
    return (high & 0x80000000) | (exponent << 23) | mantissa


def test_capture(lib: ctypes.CDLL, capture: Path) -> None:
    reader = runpy.run_path(str(Path(__file__).with_name("run_gpu_sync_probe.py")),
                            run_name="capture_reader")
    rows, cycle, op = reader["recorded_sync"](capture, 11)
    row_bytes = 4 + K * 4
    checked = 0
    for row in range(11):
        raw = rows[row * row_bytes + 4 : (row + 1) * row_bytes]
        values = struct.unpack(f"<{K}f", raw)
        source = (ctypes.c_float * K).from_buffer_copy(raw)
        output = ctypes.create_string_buffer(K // QK * Q8_BYTES)
        lib.dflash_ggml_q8_quantize(source, output, K)
        for block in range(K // QK):
            segment = values[block * QK : (block + 1) * QK]
            signed_max = max(segment, key=abs)
            iscale = f32_from_bits(double_div_to_f32_rne(-127.0, signed_max))
            actual = double_div_to_f32_rne(1.0, iscale)
            expected = struct.unpack_from("<I", output.raw, block * Q8_BYTES)[0]
            if actual != expected:
                raise AssertionError(
                    f"cycle={cycle} op={op} row={row} block={block}: "
                    f"signed_max={signed_max}, candidate={actual:08x}, ggml={expected:08x}"
                )
            checked += 1
    print(f"capture cycle={cycle} op={op}: {checked} Q8_K scales exact vs ggml")


def test_random() -> None:
    rng = random.Random(0xD5A53)
    for _ in range(100_000):
        numerator = f32_from_bits((rng.getrandbits(1) << 31) |
                                  (rng.randrange(90, 160) << 23) | rng.getrandbits(23))
        denominator = f32_from_bits((rng.getrandbits(1) << 31) |
                                    (rng.randrange(90, 160) << 23) | rng.getrandbits(23))
        actual = double_div_to_f32_rne(numerator, denominator)
        expected = bits_f32(numerator / denominator)
        if actual != expected:
            raise AssertionError(f"random division {numerator} / {denominator}: "
                                 f"candidate={actual:08x}, f32={expected:08x}")
    print("100000 random normal-ratio divisions: RNE bit conversion exact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-dll", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.ggml_dll.resolve()))
    lib.dflash_ggml_q8_quantize.argtypes = (
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int
    )
    lib.dflash_ggml_q8_quantize.restype = None
    test_random()
    test_capture(lib, args.capture)


if __name__ == "__main__":
    main()
