"""Check DFlash Q3_K/Q5_K shader layouts against this checkout's GGML DLL.

Build native ggml-base from the current llama.cpp/ggml source and pass its DLL:
  python test_quant_kernels.py --ggml-base C:/path/to/ggml-base.dll

Optional --q3/--q5 files hold one or more raw GGUF tensor blocks. This tests
the byte-level decode contract; DXC compilation is checked separately and this
does not claim that an Xbox shader has executed.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import struct
from pathlib import Path

QK_K = 256
Q3_BYTES = 110
Q5_BYTES = 176


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def load_u8(raw: bytes, offset: int) -> int:
    aligned = offset & ~3
    word = int.from_bytes(raw[aligned : aligned + 4], "little")
    return (word >> ((offset & 3) * 8)) & 255


def load_u16(raw: bytes, even_offset: int) -> int:
    aligned = even_offset & ~3
    word = int.from_bytes(raw[aligned : aligned + 4], "little")
    return (word >> ((even_offset & 2) * 8)) & 65535


def q3_element(raw: bytes, base: int, element: int) -> float:
    half = element >> 7
    quarter = (element >> 5) & 3
    lane32 = element & 31
    low2 = (load_u8(raw, base + 32 + half * 32 + lane32) >> (quarter * 2)) & 3
    high_bit = (load_u8(raw, base + lane32) >> (half * 4 + quarter)) & 1
    quant = low2 - (0 if high_bit else 4)

    scale_index = half * 8 + quarter * 2 + (lane32 >> 4)
    scale_byte = load_u8(raw, base + 96 + (scale_index & 7))
    high_byte = load_u8(raw, base + 104 + (scale_index & 3))
    low4 = (scale_byte >> (4 if scale_index >= 8 else 0)) & 15
    high2 = (high_byte >> (2 * (scale_index >> 2))) & 3
    scale = (low4 | (high2 << 4)) - 32
    d = struct.unpack("<e", load_u16(raw, base + 108).to_bytes(2, "little"))[0]
    return f32(f32(d * scale) * quant)


def q5_element(raw: bytes, base: int, element: int) -> float:
    d, dmin = struct.unpack_from("<ee", raw, base)
    scale_index = element >> 5
    if scale_index < 4:
        scale = load_u8(raw, base + 4 + scale_index) & 63
        minimum = load_u8(raw, base + 8 + scale_index) & 63
    else:
        scale = (load_u8(raw, base + 8 + scale_index) & 15) | (
            (load_u8(raw, base + scale_index) >> 6) << 4
        )
        minimum = (load_u8(raw, base + 8 + scale_index) >> 4) | (
            (load_u8(raw, base + 4 + scale_index) >> 6) << 4
        )

    lane32 = element & 31
    packed = load_u8(raw, base + 48 + (element >> 6) * 32 + lane32)
    low4 = (packed >> (4 if element & 32 else 0)) & 15
    high1 = (load_u8(raw, base + 16 + lane32) >> (element >> 5)) & 1
    quant = low4 | (high1 << 4)
    return f32(f32(f32(d * scale) * quant) - f32(dmin * minimum))


def ggml_decode(dll: ctypes.CDLL, kind: str, raw: bytes) -> list[float]:
    block_bytes = Q3_BYTES if kind == "q3" else Q5_BYTES
    if len(raw) == 0 or len(raw) % block_bytes:
        raise ValueError(f"{kind} length must be a nonzero multiple of {block_bytes}")
    func = getattr(dll, f"dequantize_row_{kind}_K")
    func.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
    func.restype = None
    output = (ctypes.c_float * (len(raw) // block_bytes * QK_K))()
    source = ctypes.create_string_buffer(raw)
    func(ctypes.cast(source, ctypes.c_void_p), output, len(output))
    return list(output)


def check_case(dll: ctypes.CDLL, kind: str, raw: bytes, label: str) -> tuple[int, float]:
    block_bytes = Q3_BYTES if kind == "q3" else Q5_BYTES
    decode = q3_element if kind == "q3" else q5_element
    reference = ggml_decode(dll, kind, raw)
    exact = 0
    max_error = 0.0
    for index, expected in enumerate(reference):
        actual = decode(raw, index // QK_K * block_bytes, index % QK_K)
        if struct.pack("<f", actual) == struct.pack("<f", expected):
            exact += 1
        error = abs(actual - expected)
        max_error = max(error, max_error)
        if not math.isclose(actual, expected, abs_tol=5e-5, rel_tol=1e-6):
            raise AssertionError(
                f"{label}: {kind} element {index}: shader layout={actual} ggml={expected}"
            )
    print(
        f"{label}: {kind} {len(raw) // block_bytes} blocks, "
        f"{exact}/{len(reference)} exact f32, max_abs_err={max_error:.9g}"
    )
    return exact, max_error


def random_blocks(kind: str, count: int, seed: int) -> bytes:
    rng = random.Random(seed)
    block_bytes = Q3_BYTES if kind == "q3" else Q5_BYTES
    data = bytearray(rng.randbytes(block_bytes * count))
    for block in range(count):
        base = block * block_bytes
        if kind == "q3":
            struct.pack_into("<e", data, base + 108, rng.uniform(-2, 2))
        else:
            struct.pack_into("<ee", data, base, rng.uniform(-2, 2), rng.uniform(-2, 2))
    return bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-base", required=True, type=Path)
    parser.add_argument("--q3", type=Path, help="raw GGUF Q3_K tensor blocks")
    parser.add_argument("--q5", type=Path, help="raw GGUF Q5_K tensor blocks")
    args = parser.parse_args()
    dll = ctypes.CDLL(str(args.ggml_base.resolve()))
    for kind in ("q3", "q5"):
        check_case(dll, kind, random_blocks(kind, 19, 0xD5A53), "random")
        block_bytes = Q3_BYTES if kind == "q3" else Q5_BYTES
        edge = bytearray(block_bytes * 2)
        # Exercise zero/all-one high masks and the alternating 2-byte Q3_K
        # block alignment while keeping fp16 scales finite.
        edge[block_bytes : block_bytes * 2] = b"\xff" * block_bytes
        for block in range(2):
            base = block * block_bytes
            if kind == "q3":
                struct.pack_into("<e", edge, base + 108, 0.5)
            else:
                struct.pack_into("<ee", edge, base, 0.5, -0.25)
        check_case(dll, kind, bytes(edge), "edge")
        sample = args.q3 if kind == "q3" else args.q5
        if sample:
            check_case(dll, kind, sample.read_bytes(), sample.name)


if __name__ == "__main__":
    main()
