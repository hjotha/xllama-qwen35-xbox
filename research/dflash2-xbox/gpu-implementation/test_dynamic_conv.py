"""Check the isolated shader indexing against dflash.cpp:549-614 semantics.

Provide --gguf with the original model, or --base-raw with exactly one unmodified
GGUF F32 [5120,2,2] conv-base tensor. This is a CPU layout/reference test; a
D3D12 dispatch and Xbox execution require integration by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
from pathlib import Path


HIDDEN = 5120
GROUP_SIZE = 16
GROUPS = HIDDEN // GROUP_SIZE
TAPS = 2
SIDES = 2
BASE_BYTES = HIDDEN * TAPS * SIDES * 4
ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "research/dflash2-xbox/gpu-implementation/f0-2026-09-28/tensor-inventory.json"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def read_base(args: argparse.Namespace) -> tuple[bytes, str]:
    if args.base_raw:
        raw = args.base_raw.read_bytes()
        label = str(args.base_raw)
    else:
        inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
        if args.gguf.stat().st_size != inventory["model_bytes"]:
            raise ValueError("GGUF size differs from recorded real-model inventory")
        tensor = next(item for item in inventory["tensors"] if item["name"] == args.tensor)
        if tensor["type"] != "F32" or tensor["shape"] != [5120, 2, 2] or tensor["bytes"] != BASE_BYTES:
            raise ValueError(f"unexpected tensor shape/type: {tensor}")
        with args.gguf.open("rb") as source:
            source.seek(tensor["offset"])
            raw = source.read(tensor["bytes"])
        label = f"{args.gguf}:{args.tensor}@{tensor['offset']}"
    if len(raw) != BASE_BYTES:
        raise ValueError(f"expected {BASE_BYTES} base bytes, got {len(raw)}")
    return raw, label


def ggml_reference(hidden: list[float], dynamic: list[float], base: list[float],
                   n_tokens: int, n_blocks: int, side: int) -> list[float]:
    """Materialize GGML's reshape/view/repeat and concat with token-major axes."""
    block_size = n_tokens // n_blocks
    # GGML reshape_4d(dynamic, n_groups, kernel, 2, n_tokens).
    coeffs = [[[[dynamic[((t * SIDES + s) * TAPS + k) * GROUPS + g]
                 for g in range(GROUPS)] for k in range(TAPS)]
               for s in range(SIDES)] for t in range(n_tokens)]
    # GGML reshape_4d(base_side, group_size, n_groups, kernel, 1).
    bases = [[[base[((s * TAPS + k) * HIDDEN) + c]
               for c in range(HIDDEN)] for k in range(TAPS)] for s in range(SIDES)]
    result = [0.0] * len(hidden)
    for block in range(n_blocks):
        for local in range(block_size):
            token = block * block_size + local
            for channel in range(HIDDEN):
                group = channel // GROUP_SIZE
                value = 0.0
                for tap in range(TAPS):
                    # GGML concat([zeros, previous], axis=token) within each block.
                    src = 0.0 if local < tap else hidden[(token - tap) * HIDDEN + channel]
                    weight = f32(bases[side][tap][channel] + coeffs[token][side][tap][group])
                    term = f32(weight * src)
                    value = term if tap == 0 else f32(value + term)
                result[token * HIDDEN + channel] = value
    return result


def shader_addressing(hidden: list[float], dynamic: list[float], base: list[float],
                      n_tokens: int, n_blocks: int, side: int) -> list[float]:
    """Host emulation of the HLSL buffer addresses, independent of reshape arrays."""
    block_size = n_tokens // n_blocks
    output = [0.0] * (n_tokens * HIDDEN)
    for element in range(len(output)):
        token, channel = divmod(element, HIDDEN)
        group = channel // GROUP_SIZE
        coeff_base = token * (SIDES * TAPS * GROUPS) + side * (TAPS * GROUPS) + group
        weight_base = side * (TAPS * HIDDEN) + channel
        weight0 = f32(base[weight_base] + dynamic[coeff_base])
        value = f32(weight0 * hidden[element])
        if token % block_size:
            weight1 = f32(base[weight_base + HIDDEN] + dynamic[coeff_base + GROUPS])
            value = f32(value + f32(weight1 * hidden[element - HIDDEN]))
        output[element] = value
    return output


def check(base: list[float], batch: int, n_blocks: int, rng: random.Random) -> None:
    assert batch % n_blocks == 0
    hidden = [f32(rng.uniform(-2, 2)) for _ in range(batch * HIDDEN)]
    # Projection emits 2 sides x 2 taps x 320 groups per token, GGML contiguous.
    dynamic = [f32(rng.uniform(-0.7, 0.7)) for _ in range(batch * SIDES * TAPS * GROUPS)]
    for side in range(SIDES):
        expected = ggml_reference(hidden, dynamic, base, batch, n_blocks, side)
        actual = shader_addressing(hidden, dynamic, base, batch, n_blocks, side)
        mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        if mismatches:
            index = mismatches[0]
            raise AssertionError(f"batch={batch} blocks={n_blocks} side={side} "
                                 f"element={index}: shader={actual[index]} GGML={expected[index]}")
        # Changing the final hidden row of a block must not affect the first
        # output row of the next block. This catches causal bleed across blocks.
        if n_blocks > 1:
            boundary = batch // n_blocks
            changed = hidden.copy()
            changed[(boundary - 1) * HIDDEN:boundary * HIDDEN] = [100.0] * HIDDEN
            boundary_output = shader_addressing(changed, dynamic, base, batch, n_blocks, side)
            if boundary_output[boundary * HIDDEN:(boundary + 1) * HIDDEN] != \
                    actual[boundary * HIDDEN:(boundary + 1) * HIDDEN]:
                raise AssertionError(f"block boundary bleed at token {boundary}")
        print(f"PASS batch={batch} blocks={n_blocks} side={side} elements={len(actual)} exact_f32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gguf", type=Path)
    source.add_argument("--base-raw", type=Path)
    parser.add_argument("--tensor", default="blk.0.attn_conv_base")
    args = parser.parse_args()
    raw, label = read_base(args)
    base = list(struct.unpack(f"<{HIDDEN * TAPS * SIDES}f", raw))
    if not all(math.isfinite(value) for value in base):
        raise ValueError("GGUF conv-base has non-finite values")
    print(f"base={label} sha256={hashlib.sha256(raw).hexdigest()} "
          f"min={min(base):.9g} max={max(base):.9g}")
    rng = random.Random(0xDFA52026)
    for batch in (2, 3, 5, 7, 8):
        check(base, batch, 1, rng)
        check(base, batch, batch, rng)  # block_size=1: tap 1 is always zero
    for blocks in (2, 4):
        check(base, 8, blocks, rng)  # multi-token, multi-block boundaries


if __name__ == "__main__":
    main()
