"""Check DFlash Q2_K/Q3_K/Q4_K/Q5_K shader layouts against this checkout's GGML DLL.

Build native ggml-base from the current llama.cpp/ggml source and pass its DLL:
  python test_quant_kernels.py --ggml-base C:/path/to/ggml-base.dll

Optional --q2/--q3/--q4/--q5 files hold one or more raw GGUF tensor blocks.
Optional --f32 is a real [5120,2,2] convolution-base tensor, tested as a
4x5120 diagnostic matrix. The byte-level decode and host reduction checks do
not claim that an Xbox shader or the complete DFlash convolution has executed.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import struct
from pathlib import Path

QK_K = 256
Q2_BYTES = 84
Q3_BYTES = 110
Q4_BYTES = 144
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


def q2_element(raw: bytes, base: int, element: int) -> float:
    quant, minimum = q2_integers(raw, base, element)
    d, dmin = struct.unpack_from("<ee", raw, base + 80)
    return f32(f32(d * quant) - f32(dmin * minimum))


def q2_integers(raw: bytes, base: int, element: int) -> tuple[int, int]:
    packed = load_u8(raw, base + 16 + (element >> 7) * 32 + (element & 31))
    quant = (packed >> (((element >> 5) & 3) * 2)) & 3
    scale = load_u8(raw, base + (element >> 4))
    return (scale & 15) * quant, scale >> 4


def check_q2_dot(dll: ctypes.CDLL, raw: bytes) -> None:
    # Exercise signed Q8 values, both affine terms, multiple blocks and wave sizes.
    reference = ggml_decode(dll, "q2", raw)
    rng = random.Random(20260928)
    for batch in (1, 5, 8):
        for _ in range(batch):
            x = [rng.randrange(-128, 128) for _ in reference]
            expected = sum(w * q * 0.03125 for w, q in zip(reference, x))
            for wave_size in (32, 64):
                actual = 0.0
                for block in range(len(raw) // Q2_BYTES):
                    base = block * Q2_BYTES
                    partial = []
                    for wave in range(256 // wave_size):
                        sq = sm = 0
                        for lane in range(wave * wave_size, (wave + 1) * wave_size):
                            q, m = q2_integers(raw, base, lane)
                            sq += q * x[block * 256 + lane]
                            sm += m * x[block * 256 + lane]
                        partial.append((sq, sm))
                    d, dmin = struct.unpack_from("<ee", raw, base + 80)
                    actual += 0.03125 * (d * sum(q for q, _ in partial) -
                                        dmin * sum(m for _, m in partial))
                assert math.isclose(actual, expected, abs_tol=5e-5, rel_tol=1e-6), (actual, expected)
    print("q2 integer dot: batch 1/5/8, wave32/64 match GGML dequantized reference")


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


def q4_element(raw: bytes, base: int, element: int) -> float:
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
    packed = load_u8(raw, base + 16 + (element >> 6) * 32 + (element & 31))
    quant = (packed >> (4 if element & 32 else 0)) & 15
    return f32(f32(f32(d * scale) * quant) - f32(dmin * minimum))


def ggml_decode(dll: ctypes.CDLL, kind: str, raw: bytes) -> list[float]:
    block_bytes = {"q2": Q2_BYTES, "q3": Q3_BYTES, "q4": Q4_BYTES, "q5": Q5_BYTES}[kind]
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
    block_bytes = {"q2": Q2_BYTES, "q3": Q3_BYTES, "q4": Q4_BYTES, "q5": Q5_BYTES}[kind]
    decode = {"q2": q2_element, "q3": q3_element, "q4": q4_element, "q5": q5_element}[kind]
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
    block_bytes = {"q2": Q2_BYTES, "q3": Q3_BYTES, "q4": Q4_BYTES, "q5": Q5_BYTES}[kind]
    data = bytearray(rng.randbytes(block_bytes * count))
    for block in range(count):
        base = block * block_bytes
        if kind == "q2":
            struct.pack_into("<ee", data, base + 80, rng.uniform(-2, 2), rng.uniform(-2, 2))
        elif kind == "q3":
            struct.pack_into("<e", data, base + 108, rng.uniform(-2, 2))
        else:
            struct.pack_into("<ee", data, base, rng.uniform(-2, 2), rng.uniform(-2, 2))
    return bytes(data)


def check_f32_conv_base(raw: bytes) -> None:
    k, n = 5120, 4  # Flatten GGUF [5120,2,2] outer axes for this diagnostic.
    if len(raw) != n * k * 4:
        raise ValueError(f"F32 conv-base needs {n * k * 4} bytes, got {len(raw)}")
    weights = struct.unpack(f"<{n * k}f", raw)
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError("F32 conv-base contains non-finite weights")
    rng = random.Random(0xD5A53)
    activations = [[f32(rng.uniform(-1, 1)) for _ in range(k)] for _ in range(8)]
    for batch in (1, 2, 4, 5, 7, 8):
        max_error = 0.0
        for b in range(batch):
            x = activations[b]
            for row in range(n):
                base = row * k
                partial = [0.0] * 256
                for lane in range(256):
                    acc = 0.0
                    for column in range(lane, k, 256):
                        acc = f32(acc + f32(weights[base + column] * x[column]))
                    partial[lane] = acc
                for stride in (128, 64, 32, 16, 8, 4, 2, 1):
                    for lane in range(stride):
                        partial[lane] = f32(partial[lane] + partial[lane + stride])
                reference = math.fsum(weights[base + column] * x[column]
                                      for column in range(k))
                error = abs(partial[0] - reference)
                max_error = max(max_error, error)
                if not math.isclose(partial[0], reference, abs_tol=1e-4, rel_tol=1e-4):
                    raise AssertionError(
                        f"F32 host reduction batch={batch} row={row}: "
                        f"shader order={partial[0]} reference={reference}"
                    )
        print(f"f32 conv-base host reduction: batch={batch}, max_abs_err={max_error:.9g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-base", required=True, type=Path)
    parser.add_argument("--q2", type=Path, help="raw GGUF Q2_K tensor blocks")
    parser.add_argument("--q3", type=Path, help="raw GGUF Q3_K tensor blocks")
    parser.add_argument("--q4", type=Path, help="raw GGUF Q4_K tensor blocks")
    parser.add_argument("--q5", type=Path, help="raw GGUF Q5_K tensor blocks")
    parser.add_argument("--f32", type=Path, help="raw GGUF [5120,2,2] F32 conv-base tensor")
    args = parser.parse_args()
    dll = ctypes.CDLL(str(args.ggml_base.resolve()))
    for kind in ("q2", "q3", "q4", "q5"):
        check_case(dll, kind, random_blocks(kind, 19, 0xD5A53), "random")
        block_bytes = {"q2": Q2_BYTES, "q3": Q3_BYTES, "q4": Q4_BYTES, "q5": Q5_BYTES}[kind]
        edge = bytearray(block_bytes * 2)
        # Exercise zero/all-one high masks and the alternating 2-byte Q3_K
        # block alignment while keeping fp16 scales finite.
        edge[block_bytes : block_bytes * 2] = b"\xff" * block_bytes
        for block in range(2):
            base = block * block_bytes
            if kind == "q2":
                struct.pack_into("<ee", edge, base + 80, 0.5, -0.25)
            elif kind == "q3":
                struct.pack_into("<e", edge, base + 108, 0.5)
            else:
                struct.pack_into("<ee", edge, base, 0.5, -0.25)
        check_case(dll, kind, bytes(edge), "edge")
        if kind == "q2":
            check_q2_dot(dll, random_blocks(kind, 19, 0xD5A53))
            check_q2_dot(dll, bytes(edge))
        sample = {"q2": args.q2, "q3": args.q3, "q4": args.q4, "q5": args.q5}[kind]
        if sample:
            check_case(dll, kind, sample.read_bytes(), sample.name)
    if args.f32:
        check_f32_conv_base(args.f32.read_bytes())


if __name__ == "__main__":
    main()
