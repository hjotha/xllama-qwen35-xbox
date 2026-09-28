"""Independent scalar checks for the DFlash H64/H128 shader algorithm.

The source kernel is compiled separately by draft_compile_hadamard.py; these
checks compare its butterfly indexing with ggml_gen_hadamard's matrix formula.
"""

import math
import random


def butterfly(values, width):
    result = [value / math.sqrt(width) for value in values]
    stride = 1
    while stride < width:
        old = result.copy()
        for lane in range(128):
            other = old[lane ^ stride]
            result[lane] = other - old[lane] if lane & stride else old[lane] + other
        stride *= 2
    return result


def matrix_entry(i, j, width):
    return (-1.0 if (i & j).bit_count() & 1 else 1.0) / math.sqrt(width)


def main():
    rng = random.Random(341791)
    checks = 0
    for width in (64, 128):
        for batch in (2, 3, 5, 7, 8):
            for head in (0, 3, 7, 31):
                values = [rng.uniform(-2.0, 2.0) for _ in range(128)]
                rotated = butterfly(values, width)
                restored = butterfly(rotated, width)
                for lane in range(128):
                    group_start = (lane // width) * width
                    row = lane - group_start
                    expected = sum(matrix_entry(row, col, width) * values[group_start + col]
                                   for col in range(width))
                    assert abs(rotated[lane] - expected) < 2e-14, (
                        width, batch, head, lane, rotated[lane], expected)
                    assert abs(restored[lane] - values[lane]) < 2e-14, (
                        width, batch, head, lane, restored[lane], values[lane])
                    checks += 1
    print(f"Hadamard host matrix/involution: PASS {checks} elements (H64 and H128)")


if __name__ == "__main__":
    main()
