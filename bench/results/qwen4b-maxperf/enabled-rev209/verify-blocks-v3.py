#!/usr/bin/env python3
"""verify-blocks-v3.py <blocks/bN> — strict acceptance-block validator.

Fixes the rev207 validator reproducibility gaps flagged by review:
- REQUIRES all 12 cells x 2 arms = 24 CSVs (a truncated battery fails; the v2
  glob-based pass could not see missing cells such as std64/std256).
- REQUIRES rows run_index {2,3} exactly (2 measured lines; warmup excluded)
  and token sidecars run2+run3 for every cell-arm.
- EXPLICIT baseline digest per cell (by prompt group). MTP cells are checked
  against the same baseline as their prompt group (MTP must not change the
  greedy output). std256 / std256-mtp have NO historical baseline — parity
  only, stated explicitly as BASELINE=none (documented, not silent).
- EXPLICIT arm parity sw0==sw1 for every cell and every measured run.

Exit 0 only when every required check passes; prints BASELINE-checked vs
parity-only classification per cell.
"""
import csv
import glob
import hashlib
import os
import sys

B64 = "84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2"  # spec-chat-open, 64
B_CODE64 = "ff42c5beec71ff491b458d0e48cbac4d98f8cdcf8153c78868c64aee687a4010"  # spec-code-edit, 64
B_CHAT256 = "f78ce8372ccc7281aaad5e726bf7c88a0317212e3b2ed397fb0fd83ffd53066c"  # spec-chat-open, 256
B_CODE256 = "c7d84255e08e20776c1ec6743d23161d32d08dcba642036b305250f10841ec0c"  # spec-code-edit, 256
B_STD64 = "44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732"  # standard-512, 64

CELLS = [
    "chat64", "code64", "std64", "chat256", "code256", "std256",
    "chat64-mtp", "code64-mtp", "std64-mtp", "chat256-mtp", "code256-mtp",
    "std256-mtp",
]
BASE = {
    "chat64": B64, "chat64-mtp": B64,
    "code64": B_CODE64, "code64-mtp": B_CODE64,
    "std64": B_STD64, "std64-mtp": B_STD64,
    "chat256": B_CHAT256, "chat256-mtp": B_CHAT256,
    "code256": B_CODE256, "code256-mtp": B_CODE256,
    "std256": None, "std256-mtp": None,  # no historical baseline: parity only
}


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(d):
    fails = []
    dig = {}
    dec = {0: {}, 1: {}}
    expected = [f"{cell}-sw{sw}" for cell in CELLS for sw in (0, 1)]
    for name in expected:
        f = os.path.join(d, "csv", name + ".csv")
        cell, sws = name.rsplit("-sw", 1)
        sw = int(sws)
        if not os.path.exists(f):
            fails.append(f"CSV_MISSING {name} (24-CSV requirement)")
            continue
        rows = list(csv.DictReader(open(f)))
        idx = [r["run_index"] for r in rows]
        if not (len(rows) == 2 and set(idx) == {"2", "3"}):
            fails.append(f"ROW_FAIL {name} rows={len(rows)} idx={idx}")
        for run in (2, 3):
            p = f.replace(".csv", f".run{run}.tokens")
            if not os.path.exists(p):
                fails.append(f"TOK_MISSING {name} run{run}")
                continue
            h = sha256(p)
            dig[(cell, sw, run)] = h
            b = BASE[cell]
            if b is None:
                continue  # parity-only cell, stated below
            if h != b:
                fails.append(f"BASE_FAIL {name} run{run} got={h[:16]} want={b[:16]}")
        try:
            dec[sw].setdefault(cell, []).extend(
                float(r["decode_tok_s"]) for r in rows)
        except (KeyError, ValueError) as exc:
            fails.append(f"DECODE_COL_FAIL {name} {exc}")
    # arm parity for every cell (explicit, incl. MTP cells)
    for cell in CELLS:
        for run in (2, 3):
            a = dig.get((cell, 0, run))
            b = dig.get((cell, 1, run))
            if a is None or b is None:
                continue  # already reported as missing above
            if a != b:
                fails.append(f"PARITY_FAIL {cell} run{run} {a[:16]} vs {b[:16]}")
    print(f"== {d}: failures={len(fails)} (cells={len(CELLS)}, "
          f"csvs required={len(expected)})")
    for x in fails:
        print("  ", x)
    print("-- baseline classification (explicit):")
    for cell in CELLS:
        b = BASE[cell]
        tag = f"BASELINE={b[:16]}" if b else "BASELINE=none (no historical run; arm-parity gate only)"
        mtp = " [MTP]" if cell.endswith("-mtp") else ""
        print(f"  {cell:<14}{mtp:<7} {tag}")
    print("-- decode means (measured run_index 2,3 only):")
    hdr = f"{'cell':<14} {'sw0_mean':>8} {'sw1_mean':>8} {'delta%':>7}"
    print(hdr)
    deltas = []
    for cell in CELLS:
        a = dec[0].get(cell, [])
        b = dec[1].get(cell, [])
        if len(a) == 2 and len(b) == 2:
            da = sum(a) / len(a)
            db = sum(b) / len(b)
            pct = 100 * (db / da - 1)
            deltas.append(pct)
            print(f"{cell:<14} {da:8.2f} {db:8.2f} {pct:+6.2f}%")
        else:
            print(f"{cell:<14} INCOMPLETE sw0={len(a)} sw1={len(b)}")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: verify-blocks-v3.py <blocks/bN>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
