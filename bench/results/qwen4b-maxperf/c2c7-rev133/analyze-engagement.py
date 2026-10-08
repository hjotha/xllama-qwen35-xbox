#!/usr/bin/env python3
"""Per-run engagement extraction for the rev133 C2/C7 grid.

The device log accumulates across launches, so each runN.log is parsed as its
own final segment: the last knob/effective line and the last main-context
summary before/after it belong to that run. Also re-checks full-token parity
per measured run. Writes engagement-per-run.txt.
"""
import csv
import hashlib
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
CHAT = "84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2"
STD = "44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732"

CELLS = {
    "verify-mtp-b0": ("chat", "effective threads:", "decode=2 batch=2", [2]),
    "verify-mtp-b4": ("chat", "effective threads:", "decode=2 batch=4", [2]),
    "c2-mtp-t1": ("chat", "draft CPU threads=", "draft CPU threads=1", [2, 3]),
    "c2-mtp-t2": ("chat", "draft CPU threads=", "draft CPU threads=2", [2, 3]),
    "c2-mtp-t2-b2": ("chat", "draft CPU threads=", "draft CPU threads=2", [2, 3]),
    "c2-mtp-t1-b2": ("chat", "draft CPU threads=", "draft CPU threads=1", [2, 3]),
    "c7-std-u32": ("std", "prefill batch override:", "n_batch=32 n_ubatch=32", [2, 3]),
    "c7-std-u64": ("std", "prefill batch override:", "n_batch=64 n_ubatch=64", [2, 3]),
    "c7-std-u64-b2": ("std", "prefill batch override:", "n_batch=64 n_ubatch=64", [2, 3]),
    "c7-std-u32-b2": ("std", "prefill batch override:", "n_batch=32 n_ubatch=32", [2, 3]),
}
summary_re = re.compile(r"\[xllama\] d3d12: \d+ graph_compute calls")

out = []
failures = 0
for cell, (kind, needle, expect, runs) in CELLS.items():
    for run in runs:
        log = (BASE / "logs" / cell / f"run{run}.log").read_text(errors="replace").splitlines()
        # Last occurrence of the expected effective line in this run's segment.
        hit = None
        hit_idx = -1
        for i, line in enumerate(log):
            if needle in line:
                hit = line
                hit_idx = i
        eff_ok = hit is not None and expect in hit
        # A main-context summary exists after it (the same run's free log).
        sum_ok = False
        if hit_idx >= 0:
            for line in log[hit_idx + 1 :]:
                if summary_re.search(line):
                    sum_ok = True
        token_file = BASE / "csv" / f"{cell}.run{run}.tokens"
        want = CHAT if kind == "chat" else STD
        parity = token_file.exists() and hashlib.sha256(token_file.read_bytes()).hexdigest() == want
        status = "PASS" if (eff_ok and sum_ok and parity) else "FAIL"
        if status == "FAIL":
            failures += 1
        out.append(
            f"{status} cell={cell} run={run} effective={eff_ok} summary={sum_ok} parity={parity} "
            f"line={(hit or 'none').strip()[:110]}"
        )
out.append(f"total={sum(len(v[3]) for v in CELLS.values())} failures={failures}")
(BASE / "engagement-per-run.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
raise SystemExit(1 if failures else 0)
