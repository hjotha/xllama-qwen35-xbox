"""Deterministic host parity for the DRAFT dynamic-conv call shapes."""

import random

from test_dynamic_conv import BASE_BYTES, HIDDEN, SIDES, TAPS, check, f32


def main():
    rng = random.Random(0xDFA52026)
    base = [f32(rng.uniform(-0.5, 0.5))
            for _ in range(BASE_BYTES // 4)]
    assert len(base) == HIDDEN * TAPS * SIDES
    cases = 0
    for batch in (2, 3, 5, 7, 8):
        for blocks in (1, batch):
            check(base, batch, blocks, rng)
            cases += 2  # each check validates both sides
    for blocks in (2, 4):
        check(base, 8, blocks, rng)
        cases += 2
    print(f"DRAFT conv host parity: {cases} side/batch/block cases passed")


if __name__ == "__main__":
    main()
