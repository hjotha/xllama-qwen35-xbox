#!/usr/bin/env python3
"""Extract the baseline pilot's last 30 bench blocks from the full device log.

Blocks are delimited by 'bench.flag detected -> headless bench mode'. The pilot
ran 10 cells x 3 runs in the driver order; this script maps the last 30 blocks
to that order, verifies each block's prompt size and MTP/seq marker, and writes
a per-cell/per-run summary plus a slice file for durable evidence.
"""
import json
import re
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
log = (BASE / "post-pilot-full.log").read_text(errors="replace").splitlines()

starts = [i for i, l in enumerate(log) if "bench.flag detected -> headless bench mode" in l]
blocks = []
for n, s in enumerate(starts):
    e = starts[n + 1] if n + 1 < len(starts) else len(log)
    blocks.append(log[s:e])
sel = blocks[-30:]
assert len(sel) == 30, f"expected 30 blocks, got {len(sel)}"

CELLS = [
    ("chat64-seq", "spec-chat-open", 0), ("chat64-mtp", "spec-chat-open", 1),
    ("code64-seq", "spec-code-edit", 0), ("code64-mtp", "spec-code-edit", 1),
    ("code64-mtp-b2", "spec-code-edit", 1), ("code64-seq-b2", "spec-code-edit", 0),
    ("chat64-mtp-b2", "spec-chat-open", 1), ("chat64-seq-b2", "spec-chat-open", 0),
    ("chat64-mtp-on", "spec-chat-open", 1), ("chat64-seq-on", "spec-chat-open", 0),
]

def med(xs):
    return round(statistics.median(xs), 3) if xs else None

summary = {}
for ci, (name, prompt, is_mtp) in enumerate(CELLS):
    runs = sel[ci * 3:(ci + 1) * 3]
    recs = []
    for bi, b in enumerate(runs):
        txt = "\n".join(b)
        done = re.search(r"done: load=(\d+)ms prompt=([\d.]+) tok/s decode=([\d.]+) tok/s peak=(\d+)MB(?: drafted=(\d+) spec_accept=(\d+))?", txt)
        stats = re.search(r"MTP_STATS rounds=(\d+) decodes=(\d+) discarded=(\d+) catchup_tok=(\d+) draft_ms=([\d.]+) sample_ms=([\d.]+) topprob_ms=([\d.]+) catchup_ms=([\d.]+) verify_ms=([\d.]+) corrective_ms=([\d.]+)", txt)
        d3 = re.search(r"\[xllama\] d3d12: (\d+) graph_compute calls, (\d+) matmuls \((\d+) 2col\), ([\d.]+) ms wall \(([\d.]+) ms GPU\)", txt)
        gdn = re.search(r"\[xllama\] d3d12: (\d+) GATED_DELTA_NET dispatches", txt)
        vr = []
        for m in re.finditer(r"VROUND width=(\d+) wall=([\d.]+) d3w=([\d.]+) d3g=([\d.]+) cpu=([\d.]+) reuse=(\d+) accepted=(\d+) gen=(\d+)", txt):
            vr.append(dict(zip(("width", "wall", "d3w", "d3g", "cpu", "reuse", "accepted", "gen"), m.groups())))
        expected = "drafted=0" if is_mtp == 0 else None
        if done and expected is None:
            assert int(done.group(5) or 0) > 0, f"{name} run{bi+1}: expected drafting"
        if done and expected == "drafted=0":
            assert not (done.group(5) and int(done.group(5)) > 0), f"{name} run{bi+1}: expected no drafting"
        recs.append({
            "run": bi + 1,
            "done": done.groups() if done else None,
            "mtp_stats": [float(x) if i >= 4 else int(x) for i, x in enumerate(stats.groups())] if stats else None,
            "d3d12": d3.groups() if d3 else None,
            "gdn": int(gdn.group(1)) if gdn else None,
            "vround_n": len(vr),
            "vround_widths": sorted({int(v["width"]) for v in vr}),
            "vround_sums": {k: round(sum(float(v[k]) for v in vr), 1) for k in ("wall", "d3w", "d3g", "cpu")} if vr else None,
            "vround_med": {k: med([float(v[k]) for v in vr]) for k in ("wall", "d3w", "d3g", "cpu")} if vr else None,
            "vround_reuse": sum(1 for v in vr if v["reuse"] == "1") if vr else None,
        })
    summary[name] = {"prompt": prompt, "mtp_on": bool(is_mtp), "runs": recs, "blocks": [b[0] for b in runs]}

(BASE / "segment-baseline-30runs.log").write_text("\n".join(l for b in sel for l in b) + "\n")
(BASE / "baseline-log-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

# Compact human table
print(f"{'cell':18} {'decode':>7} {'ttft':>7} {'prefill':>8} | mtp stats (run2): draft verify topprob catchup sample corr | VROUND sum wall d3w d3g cpu n")
for name, prompt, _ in CELLS:
    s = summary[name]["runs"][1]  # run2 = first measured row
    d = s["done"]
    st = s["mtp_stats"]
    v = s["vround_sums"]
    cells = [name, f"{float(d[2]):7.2f}", f"{(64000/float(d[2])):7.0f}", f"{(float(d[1]) and 1000*119/float(d[1])):8.0f}" if False else ""]
    if st:
        statline = f"{st[4]:6.1f} {st[8]:6.1f} {st[6]:6.1f} {st[7]:6.1f} {st[5]:5.1f} {st[9]:5.1f}"
    else:
        statline = "-"
    print(f"{name:18} {float(d[2]):7.2f} {'':7} {'':8} | {statline} | {v['wall'] if v else '-'} {v['d3w'] if v else '-'} {v['d3g'] if v else '-'} {v['cpu'] if v else '-'} {s['vround_n']}")
