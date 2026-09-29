"""Host checks for the GGML x86 Q8_0 query contract used by DRAFT attention.

This is a scalar port of ggml-cpu/arch/x86/quants.c:quantize_row_q8_0 and the
Q4_0 x Q8_0 integer dot. It checks fixed byte vectors and scale bits; the GPU
shader still requires the Xbox hidden-state parity probe for validation.
"""

import math
import random
import struct


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def shader_half_bits(value):
    """Scalar port of ggml_half_bits shared by Q4 KV pack and Q8 Q scales."""
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    sign = (bits >> 16) & 0x8000
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7fffff
    if exponent == 255:
        return sign | (0x7c00 if mantissa == 0 else 0x7e00)
    half_exponent = exponent - 127 + 15
    if half_exponent <= 0:
        if half_exponent < -10:
            return sign
        mantissa |= 0x800000
        shift = 14 - half_exponent
        rounded = mantissa >> shift
        remainder = mantissa & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and rounded & 1):
            rounded += 1
        return sign | rounded
    if half_exponent >= 31:
        return sign | 0x7c00
    rounded = mantissa >> 13
    remainder = mantissa & 0x1fff
    if remainder > 0x1000 or (remainder == 0x1000 and rounded & 1):
        rounded += 1
    if rounded == 0x400:
        rounded = 0
        half_exponent += 1
    if half_exponent >= 31:
        return sign | 0x7c00
    return sign | (half_exponent << 10) | rounded


def quantize_q8_0(block):
    assert len(block) == 32
    block = [f32(value) for value in block]
    maximum = max(abs(value) for value in block)
    d = f32(maximum / 127.0)
    inv = f32(127.0 / maximum) if maximum else 0.0
    packed = bytearray(struct.pack("<H", shader_half_bits(d)))
    for value in block:
        # Python round implements the x86 _MM_ROUND_NEAREST tie-to-even rule.
        integer = round(f32(value * inv))
        assert -127 <= integer <= 127
        packed += struct.pack("<b", integer)
    return bytes(packed)


def dot_q4_q8(q4, q8):
    assert len(q4) == 20 and len(q8) == 34
    dk = struct.unpack_from("<e", q4)[0]
    dq = struct.unpack_from("<e", q8)[0]
    ints4 = [(byte & 15) - 8 for byte in q4[4:20]]
    ints4 += [(byte >> 4) - 8 for byte in q4[4:20]]
    ints8 = struct.unpack_from("<32b", q8, 2)
    return sum(a * b for a, b in zip(ints4, ints8)) * dk * dq


def test_fixed_vectors():
    assert quantize_q8_0([0.0] * 32) == bytes(34)

    ties = [127.0, -127.0, 0.5, -0.5, 1.5, -1.5,
            2.5, -2.5, 126.5, -126.5] + [0.0] * 22
    expected = [127, -127, 0, 0, 2, -2, 2, -2, 126, -126] + [0] * 22
    assert quantize_q8_0(ties) == struct.pack("<e32b", 1.0, *expected)

    half = [63.5, -63.5, 1.25, -1.25, 0.75, -0.75] + [0.0] * 26
    expected_half = [127, -127, 2, -2, 2, -2] + [0] * 26
    assert quantize_q8_0(half) == struct.pack("<e32b", 0.5, *expected_half)

    # The stored FP16 scale must not be inverted to create the Q8 integers.
    small = [1.0, 0.1, -0.1] + [0.0] * 29
    encoded = quantize_q8_0(small)
    assert encoded[:2] == struct.pack("<e", f32(1.0 / 127.0))
    assert struct.unpack_from("<3b", encoded, 2) == (127, 13, -13)

    key_values = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5] + [0] * 22
    packed_q4 = bytes((key_values[j] + 8) | ((key_values[j + 16] + 8) << 4)
                      for j in range(16))
    key = struct.pack("<eH", 0.25, 0) + packed_q4
    assert dot_q4_q8(key, quantize_q8_0(ties)) == 385.5
    assert abs(385.5 / math.sqrt(128) - 34.07370801842676) < 1e-10


def test_random_bytes_and_dot():
    rng = random.Random(31197)
    count = 0
    for _ in range(512):
        values = [f32(rng.uniform(-6.0, 6.0)) for _ in range(32)]
        q8 = quantize_q8_0(values)
        scale = struct.unpack_from("<e", q8)[0]
        qints = struct.unpack_from("<32b", q8, 2)
        assert all(-127 <= value <= 127 for value in qints)
        assert abs(scale - max(map(abs, values)) / 127.0) <= 1e-4

        q4 = [rng.randint(-8, 7) for _ in range(32)]
        packed = bytes((q4[j] + 8) | ((q4[j + 16] + 8) << 4)
                       for j in range(16))
        key_scale = 0.125
        key = struct.pack("<eH", key_scale, 0) + packed
        integer_dot = sum(a * b for a, b in zip(q4, qints))
        assert dot_q4_q8(key, q8) == integer_dot * key_scale * scale
        count += 1
    return count


def test_half_scale_bits():
    rng = random.Random(99317)
    cases = [0.0, -0.0, 1.0, 0.5, 1.0 / 127.0, 2**-24, 2**-14]
    # Both parity branches at the F32 values exactly halfway between FP16s.
    cases += [1.0 + 2**-11, 1.0 + 3 * 2**-11]
    for _ in range(10000):
        bits = (rng.randrange(90, 141) << 23) | rng.getrandbits(23)
        if rng.getrandbits(1):
            bits |= 0x80000000
        cases.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    for value in cases:
        expected = struct.unpack("<H", struct.pack("<e", f32(value)))[0]
        assert shader_half_bits(value) == expected, (value, shader_half_bits(value), expected)
    return len(cases)


if __name__ == "__main__":
    test_fixed_vectors()
    cases = test_random_bytes_and_dot()
    half_cases = test_half_scale_bits()
    print(f"Q8_0 fixed bytes/scales and Q4_0 x Q8_0 dot: PASS; "
          f"random cases={cases}; FP16 conversions={half_cases}")
