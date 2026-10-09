#!/usr/bin/env python3
"""report-stats.py <blocks/b1> [blocks/b2 ...] — OFF/ON comparison report.

Counts COMPARISONS (cell x block mean pairs), not individual gains:
with 12 cells and 2 blocks there are 24 OFF/ON comparisons. Reports the
median/dispersion of the comparison deltas and flags every comparison whose
individual measured lines OVERLAP (max(ON lines) >= min(OFF lines) or vice
versa) — overlapping comparisons are positive at the mean level but are NOT
claimed to be above noise.
"""
import csv
import glob
import os
import statistics
import sys


def load(block):
    out = {}
    for f in sorted(glob.glob(os.path.join(block, "csv", "*.csv"))):
        name = os.path.basename(f)[:-4]
        cell, sws = name.rsplit("-sw", 1)
        sw = int(sws[0])
        rows = list(csv.DictReader(open(f)))
        vals = [float(r["decode_tok_s"]) for r in rows if r.get("run_index") in ("2", "3")]
        out.setdefault(cell, {})[sw] = vals
    return out


def main(blocks):
    comparisons = []  # (block_label, cell, off_vals, on_vals, delta_pct, overlap)
    missing = []
    for bpath in blocks:
        blabel = os.path.basename(os.path.normpath(bpath))
        data = load(bpath)
        for cell in sorted(data):
            off = data[cell].get(0, [])
            on = data[cell].get(1, [])
            if len(off) != 2 or len(on) != 2:
                missing.append(f"{blabel}/{cell} off={len(off)} on={len(on)}")
                continue
            d = 100 * (statistics.mean(on) / statistics.mean(off) - 1)
            overlap = max(on) >= min(off) and max(off) >= min(on)
            comparisons.append((blabel, cell, off, on, d, overlap))
    print(f"comparisons={len(comparisons)} (cells x blocks; each = mean of 2 "
          f"measured lines OFF vs ON)")
    if missing:
        print("INCOMPLETE:")
        for m in missing:
            print("  ", m)
    pos = [c for c in comparisons if c[4] > 0]
    ovl = [c for c in comparisons if c[5]]
    print(f"positive={len(pos)}/{len(comparisons)}  overlapping_individual_lines="
          f"{len(ovl)}/{len(comparisons)}")
    deltas = sorted(c[4] for c in comparisons)
    if deltas:
        print(f"delta_pct: median={statistics.median(deltas):+.2f} "
              f"min={deltas[0]:+.2f} max={deltas[-1]:+.2f} "
              f"mean={statistics.mean(deltas):+.2f} sd={statistics.pstdev(deltas):.2f} "
              f"q1={statistics.median(deltas[:len(deltas)//2]):+.2f} "
              f"q3={statistics.median(deltas[len(deltas)//2:]):+.2f}")
    print(f"{'block':<5} {'cell':<14} {'OFF lines':>16} {'ON lines':>16} "
          f"{'delta%':>7} {'overlap':>8}")
    for blabel, cell, off, on, d, ovl in comparisons:
        print(f"{blabel:<5} {cell:<14} "
              f"{off[0]:7.2f}/{off[1]:<7.2f} {on[0]:7.2f}/{on[1]:<7.2f} "
              f"{d:+6.2f}% {'YES' if ovl else '-':>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
