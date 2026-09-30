"""Host reference for DFlash2 DRAFT attention and DflashGpuKv Q4_0 bytes.

This checks shader indexing, mask and online-softmax math without claiming a
D3D12 run. DflashGpuKv validity remains CPU-side; the active position upload
is part of the caller contract and must be made after each trim/reinjection.
"""

from __future__ import annotations

import math
import random
import struct
import unittest

Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SWA = 2048
CAPACITY = 8192


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def row(seed: int) -> bytes:
    rng = random.Random(seed)
    result = bytearray()
    for _ in range(KV_HEADS * HEAD_DIM // 32):
        result += struct.pack("<eH", rng.uniform(0.01, 0.08), 0)
        result += bytes(rng.randrange(256) for _ in range(16))
    return bytes(result)


def unpack_gpu(data: bytes) -> list[float]:
    """Independent scalar Q4_0 decoder for the reference calculation."""
    result = []
    for block in range(len(data) // 20):
        offset = block * 20
        scale = struct.unpack_from("<e", data, offset)[0]
        lower = [(byte & 15) - 8 for byte in data[offset + 4:offset + 20]]
        upper = [(byte >> 4) - 8 for byte in data[offset + 4:offset + 20]]
        result += [value * scale for value in lower + upper]
    return result


class Cache:
    def __init__(self) -> None:
        self.physical: dict[int, tuple[bytes, bytes]] = {}
        self.decoded: dict[int, tuple[list[float], list[float]]] = {}
        self.valid: set[int] = set()

    def write(self, position: int, seed: int) -> None:
        assert 0 <= position < CAPACITY
        self.physical[position] = row(seed), row(seed + 10000)
        self.decoded[position] = tuple(unpack_gpu(data) for data in self.physical[position])
        self.valid.add(position)

    def trim(self, keep: int) -> None:
        self.valid = {p for p in self.valid if p < keep}

    def reset(self) -> None:
        self.valid.clear()

    def active(self) -> list[int]:
        return sorted(self.valid)


def element(data: bytes, head: int, d: int) -> float:
    """Port of kv_element's aligned 20-byte-block addressing."""
    offset = (head * 4 + d // 32) * 20
    scale = struct.unpack_from("<e", data, offset)[0]
    within = d & 31
    aligned = offset + 4 + ((within & 15) >> 2) * 4
    packed = struct.unpack_from("<I", data, aligned)[0]
    byte_value = (packed >> ((within & 3) * 8)) & 255
    nibble = (byte_value >> ((within >> 4) * 4)) & 15
    return (nibble - 8) * scale


def shader_port(q: list[float], qp: int, h: int, cache: Cache,
                active: list[int], sink: float) -> list[float]:
    """One compute group's online softmax, using packed GPU bytes."""
    max_score = sink
    denominator = 1.0
    numerator = [0.0] * HEAD_DIM
    kv_head = h // 4
    for kp in active:
        if kp >= CAPACITY or (kp <= qp and qp - kp >= SWA):
            continue
        key, value = cache.physical[kp]
        score = sum(q[d] * element(key, kv_head, d) for d in range(HEAD_DIM)) / math.sqrt(HEAD_DIM)
        next_max = max(max_score, score)
        old_weight = math.exp(max_score - next_max)
        new_weight = math.exp(score - next_max)
        denominator = denominator * old_weight + new_weight
        numerator = [numerator[d] * old_weight + new_weight * element(value, kv_head, d)
                     for d in range(HEAD_DIM)]
        max_score = next_max
    return [v / denominator for v in numerator]


def reference(q: list[float], qp: int, h: int, cache: Cache,
              sink: float) -> list[float]:
    """Independent decoded-vector, two-pass masked-softmax reference."""
    visible = [p for p in cache.valid if p > qp or qp - p < SWA]
    kv_head = h // 4
    scores = []
    values = []
    for position in visible:
        key, value = cache.decoded[position]
        decoded_key = key[kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM]
        decoded_value = value[kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM]
        scores.append(sum(a * b for a, b in zip(q, decoded_key)) / math.sqrt(HEAD_DIM))
        values.append(decoded_value)
    maximum = max([sink, *scores])
    weights = [math.exp(score - maximum) for score in scores]
    denominator = math.exp(sink - maximum) + sum(weights)
    return [sum(weight * value[d] for weight, value in zip(weights, values)) / denominator
            for d in range(HEAD_DIM)]


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = Cache()
        for position in (2045, 2046, 2047, 2048, 4093):
            self.cache.write(position, position)
        self.sinks = [f32(-0.4 + 0.02 * h) for h in range(Q_HEADS)]

    def check_batch(self, positions: list[int]) -> None:
        rng = random.Random(731)
        active = self.cache.active()
        for qi, position in enumerate(positions):
            for head in range(Q_HEADS):
                q = [f32(rng.uniform(-0.7, 0.7)) for _ in range(HEAD_DIM)]
                got = shader_port(q, position, head, self.cache, active, self.sinks[head])
                want = reference(q, position, head, self.cache, self.sinks[head])
                for d, (a, b) in enumerate(zip(got, want)):
                    self.assertAlmostEqual(a, b, delta=2e-6,
                                           msg=f"q={qi}, position={position}, head={head}, d={d}")

    def test_draft_shapes_and_future_visibility(self) -> None:
        for count in (2, 3, 5, 7, 8):
            with self.subTest(count=count):
                positions = list(range(4094, 4094 + count))
                for position in positions:
                    self.cache.write(position, position + count)
                self.assertIn(positions[-1], self.cache.active())
                self.check_batch(positions)
                self.cache.trim(4094)

    def test_window_exact_boundaries_and_absolute_positions(self) -> None:
        # At query 4095, 2047 is excluded, 2048 is included. A future
        # position 4096 is included even though this is not causal attention.
        self.cache.write(4096, 4096)
        visible = [p for p in self.cache.active() if p > 4095 or 4095 - p < SWA]
        self.assertEqual(visible, [2048, 4093, 4096])
        self.check_batch([4095, 4096])

    def test_partial_rejection_trim_and_reinject(self) -> None:
        for position in range(4094, 4102):
            self.cache.write(position, position)
        old = self.cache.physical[4100]
        self.cache.trim(4097)
        self.assertEqual(self.cache.active()[-1], 4096)
        self.assertEqual(self.cache.physical[4100], old)  # bytes persist but are invalid
        for position in range(4097, 4102):
            self.cache.write(position, position + 20000)
        self.assertNotEqual(self.cache.physical[4100], old)
        self.check_batch([4097, 4098, 4099, 4100, 4101])

    def test_full_rejection_and_reset_hide_stale_bytes(self) -> None:
        for position in (4094, 4095, 4096):
            self.cache.write(position, position)
        self.cache.trim(4094)
        self.assertTrue(all(p not in self.cache.active() for p in (4094, 4095, 4096)))
        self.check_batch([4094, 4095])
        self.cache.reset()
        self.assertEqual(self.cache.active(), [])
        self.assertIn(4096, self.cache.physical)
        self.check_batch([4094, 4095])  # only virtual sink remains; output zero

    def test_q4_block_offsets_and_gqa_mapping(self) -> None:
        packed = row(19)
        decoded = unpack_gpu(packed)
        self.assertEqual(len(packed), KV_HEADS * 4 * 20)
        for head in range(KV_HEADS):
            for d in range(HEAD_DIM):
                self.assertEqual(element(packed, head, d), decoded[head * HEAD_DIM + d])
        self.assertEqual([h // 4 for h in range(Q_HEADS)],
                         [kv for kv in range(KV_HEADS) for _ in range(4)])


if __name__ == "__main__":
    unittest.main()
