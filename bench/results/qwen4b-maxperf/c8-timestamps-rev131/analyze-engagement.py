#!/usr/bin/env python3
"""Authoritative per-cell extraction of the C8 A/B engagement.

The runbook's earlier grep counts are cumulative: the device log accumulates
every app launch, so 3/6/9/... lines are not per-cell proof. This script reads
each cell's LAST run log, takes the LAST effective knob line, and requires an
availability summary emitted AFTER it (main context, >0 calls) whose pattern
matches the knob. Writes engagement-final.txt next to the logs.
"""
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
CELLS = {
    "seq-tson": "on",
    "seq-tsoff": "off",
    "seq-tson-b2": "on",
    "seq-tsoff-b2": "off",
    "seq-tson-b3": "on",
    "mtp-tson": "on",
    "mtp-tsoff": "off",
    "mtp-tson-b2": "on",
    "mtp-tsoff-b2": "off",
    "mtp-tson-b3": "on",
}
knob_re = re.compile(r"gpu timestamps=(on|off)")
summary_re = re.compile(r"\[xllama\] d3d12: (\d+) graph_compute calls.*")
lines_out = []
failures = 0
for cell, want in CELLS.items():
    path = BASE / "logs" / cell / "run3.log"
    lines = path.read_text(errors="replace").splitlines()
    knob_idx = knob_val = None
    for i, line in enumerate(lines):
        m = knob_re.search(line)
        if m:
            knob_idx, knob_val = i, m.group(1)
    status = "FAIL"
    detail = "no knob line"
    if knob_idx is not None:
        if knob_val != want:
            detail = f"knob={knob_val} expected={want}"
        else:
            summary = None
            for line in lines[knob_idx + 1 :]:
                m = summary_re.search(line)
                if m and int(m.group(1)) > 0:
                    summary = line
            if summary is None:
                detail = "no main-context summary after the knob line"
            elif want == "on":
                ok = "ts_valid=" in summary and "timing unavailable:" not in summary
                detail = ("summary complete ts_valid" if ok else f"bad ON summary: {summary[:90]}")
                status = "PASS" if ok else "FAIL"
            else:
                ok = "timing unavailable:" in summary
                detail = (
                    "summary unavailable" if ok else f"bad OFF summary: {summary[:90]}"
                )
                status = "PASS" if ok else "FAIL"
    if status == "FAIL":
        failures += 1
    lines_out.append(f"{status} cell={cell} expected={want} {detail}")
lines_out.append(f"total={len(CELLS)} failures={failures}")
(BASE / "engagement-final.txt").write_text("\n".join(lines_out) + "\n")
print("\n".join(lines_out))
raise SystemExit(1 if failures else 0)
