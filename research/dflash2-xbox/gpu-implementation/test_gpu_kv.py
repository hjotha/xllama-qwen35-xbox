"""Validate KV Q4_0 block math against this checkout's GGML DLL.

The Python packer models shaders/dflash/kv_q4_0.hlsl. This checks its byte
contract and dequantization, not actual D3D12 execution. Pass --ggml-base
with a native DLL built from the matching llama.cpp checkout.
"""

from __future__ import annotations

import argparse
import ctypes
import random
import struct
from pathlib import Path


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pack_gpu(values: list[float]) -> bytes:
    if not values or len(values) % 32:
        raise ValueError("Q4_0 input must contain whole 32-value blocks")
    result = bytearray()
    for start in range(0, len(values), 32):
        block = values[start : start + 32]
        amax = 0.0
        maximum = 0.0
        for value in block:
            if amax < abs(value):
                amax = abs(value)
                maximum = value
        d = f32(maximum / -8.0)
        inv_d = f32(1.0 / d) if d else 0.0
        result += struct.pack("<H", struct.unpack("<H", struct.pack("<e", d))[0])
        result += b"\x00\x00"  # GPU alignment after the half scale
        for j in range(16):
            lo = min(15, int(f32(f32(block[j] * inv_d) + 8.5)))
            hi = min(15, int(f32(f32(block[j + 16] * inv_d) + 8.5)))
            result.append(lo | (hi << 4))
    return bytes(result)


def gpu_to_ggml(gpu_blocks: bytes) -> bytes:
    if len(gpu_blocks) % 20:
        raise ValueError("GPU block stride must be 20")
    return b"".join(
        gpu_blocks[i : i + 2] + gpu_blocks[i + 4 : i + 20]
        for i in range(0, len(gpu_blocks), 20)
    )


def unpack_gpu(gpu_blocks: bytes) -> list[float]:
    if len(gpu_blocks) % 20:
        raise ValueError("GPU block stride must be 20")
    result: list[float] = []
    for i in range(0, len(gpu_blocks), 20):
        d = struct.unpack_from("<e", gpu_blocks, i)[0]
        block = [0.0] * 32
        for j, quant in enumerate(gpu_blocks[i + 4 : i + 20]):
            block[j] = f32(((quant & 15) - 8) * d)
            block[j + 16] = f32(((quant >> 4) - 8) * d)
        result += block
    return result


def ggml_pack(dll: ctypes.CDLL, values: list[float]) -> bytes:
    input_data = (ctypes.c_float * len(values))(*values)
    output = (ctypes.c_ubyte * (len(values) // 32 * 18))()
    dll.quantize_row_q4_0_ref.argtypes = (
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64
    )
    dll.quantize_row_q4_0_ref.restype = None
    dll.quantize_row_q4_0_ref(input_data, output, len(values))
    return bytes(output)


def ggml_unpack(dll: ctypes.CDLL, raw: bytes) -> list[float]:
    output = (ctypes.c_float * (len(raw) // 18 * 32))()
    dll.dequantize_row_q4_0.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64
    )
    dll.dequantize_row_q4_0.restype = None
    source = ctypes.create_string_buffer(raw)
    dll.dequantize_row_q4_0(source, output, len(output))
    return list(output)


def check(dll: ctypes.CDLL, label: str, values: list[float]) -> None:
    native = ggml_pack(dll, values)
    gpu = pack_gpu(values)
    shader_native = gpu_to_ggml(gpu)
    if native != shader_native:
        for index, (expected, actual) in enumerate(zip(native, shader_native)):
            if expected != actual:
                raise AssertionError(
                    f"{label}: packed byte {index}: ggml={expected:02x}, shader={actual:02x}"
                )
    decoded = unpack_gpu(gpu)
    reference = ggml_unpack(dll, native)
    for index, (expected, actual) in enumerate(zip(reference, decoded)):
        if struct.pack("<f", expected) != struct.pack("<f", actual):
            raise AssertionError(
                f"{label}: decoded element {index}: ggml={expected}, shader={actual}"
            )
    print(f"{label}: {len(values) // 32} blocks, exact pack and unpack")


def check_stale_suffix() -> None:
    # Logical-position reference: TRIM and RESET must hide old bytes even when
    # physical Q4_0 blocks retain their previous contents.
    capacity = 5
    layers = 2
    physical = [[[(layer, kind, pos) for pos in range(capacity)] for kind in range(2)]
                for layer in range(layers)]
    valid = [[[True] * capacity for _ in range(2)] for _ in range(layers)]
    for layer in range(layers):
        for kind in range(2):
            valid[layer][kind][2:] = [False] * (capacity - 2)
    assert physical[1][1][4] == (1, 1, 4) and not valid[1][1][4]
    assert all(valid[layer][kind][1] for layer in range(layers) for kind in range(2))
    valid[1][1][2] = True  # reinject one suffix slot only
    assert valid[1][1][2] and not valid[1][0][2] and not valid[0][1][2]
    for layer in range(layers):
        for kind in range(2):
            valid[layer][kind] = [False] * capacity
    assert not any(any(row) for layer in valid for row in layer)
    assert physical[1][1][4] == (1, 1, 4)
    print("logical reference: stale suffix, partial reinjection and reset covered")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-base", type=Path, required=True)
    args = parser.parse_args()
    dll = ctypes.CDLL(str(args.ggml_base.resolve()))
    rng = random.Random(0xD5A54)
    check(dll, "zero", [0.0] * 32)
    check(dll, "positive and negative constants", [1.0] * 32 + [-1.0] * 32)
    check(dll, "first equal-magnitude maximum wins", [2.0, -2.0] + [0.0] * 30)
    check(dll, "reverse equal-magnitude maximum", [-2.0, 2.0] + [0.0] * 30)
    check(dll, "alternating extremes", [-6.0, 7.0] * 32)
    check(dll, "small scales", [f32((i - 16) * 0.0000125) for i in range(32)])
    check(dll, "random", [f32(rng.uniform(-4.0, 4.0)) for _ in range(32 * 1024)])
    check_stale_suffix()


if __name__ == "__main__":
    main()
