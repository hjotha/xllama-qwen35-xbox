#!/usr/bin/env python3
"""Check the existing Xbox termination harness receipts and all 48 token dumps."""
import csv
import json
from pathlib import Path
import re
import sys

base = Path(sys.argv[1])
q6 = int(sys.argv[2])
expected = {"cancel@6", "cancel@13", "stop@probe", "stop@run", *(f"eog@{i}" for i in range(4))}
summary = {}

def ids(arm, name):
    p = base / arm / name
    assert p.is_file() and p.stat().st_size, f"missing or empty dump: {p}"
    v = [int(x) for x in p.read_text().split()]
    assert v, f"no IDs: {p}"
    return v

for arm in ("seq", "cand"):
    assert (base / f"{arm}.done").is_file(), f"missing completion marker: {arm}"
    rows = list(csv.DictReader((base / f"{arm}.csv").open()))
    assert len(rows) == 8 and {r["scenario"] for r in rows} == expected, f"incomplete CSV: {arm}"
    assert all(r["ok"] == "1" and r["arm"] == arm for r in rows), f"device failure: {arm}"
    assert len(list((base / arm).glob("termgate-*.txt*"))) == 24, f"dump coverage: {arm}"
    for p in (base / arm).glob("termgate-*.txt*"):
        ids(arm, p.name)
    raw = (base / f"{arm}.log").read_text()
    raw = raw[raw.rfind("[xllama] termgate: Q6_K LM-head columns="):]
    assert f"Q6_K LM-head columns={q6}" in raw and "Q8_0 matmul=D3D12" in raw
    assert "GATED_DELTA_NET=D3D12\n" in raw and "fused attention=forced" in raw
    assert "attention split-KV=disabled for consistent query reduction" in raw, f"FA2 receipt missing: {arm}"
    cfg = re.search(r"TERM_GATE_CONFIG arm=" + arm + r" threads=2 n_ctx=2048 gpu_layers=34 mtp=(\d+) p_min=0.50 twocol=2 repack_scope=2", raw)
    assert cfg and int(cfg[1]) == (0 if arm == "seq" else 2), f"effective configuration: {arm}"
    assert "termgate done rows=8 fail=-" in raw, f"completion failure: {arm}"
    for n in (6, 13):
        prefix = f"termgate-cancel{n}"
        t1 = ids(arm, prefix + "-t1.txt")
        reference = ids(arm, prefix + "-seqref.txt")
        resumed = ids(arm, prefix + "-resume.txt")
        assert len(t1) == n and t1 == reference[:n], f"cancel boundary: {arm}/{n}"
        assert resumed == reference[n:n + len(resumed)], f"cancel resume: {arm}/{n}"
        assert ids(arm, prefix + "-resume.txt.prefill") == ids(arm, prefix + "-t1.txt.prefill") + t1
    stopped = ids(arm, "termgate-stop-run.txt")
    assert stopped == ids(arm, "termgate-stop-probe.txt")[:len(stopped)], f"stop prefix: {arm}"
    for suffix in ("", ".prefill"):
        assert ids(arm, "termgate-stop-resume.txt" + suffix) == ids(arm, "termgate-stop-cold.txt" + suffix), f"stop resume: {arm}"
    stop = next(r for r in rows if r["scenario"] == "stop@run")
    assert stop["ews"] == "1" and stop["stop_branch"] != "-" and int(stop["stop_round"]) > 0
    eos = [r for r in rows if r["scenario"].startswith("eog@") and r["reason"] == "eos"]
    assert eos and all(int(r["eog_tok"]) >= 0 and r["eog_branch"] != "-" for r in eos), f"natural EOG coverage: {arm}"
    if arm == "cand":
        assert all(r["active"] == "1" and int(r["drafted"]) > 0 and int(r["rounds"]) > 0 for r in rows), "MTP engagement"
    for i in range(4):
        row = next(r for r in rows if r["scenario"] == f"eog@{i}")
        assert len(ids(arm, f"termgate-eog{i}.txt")) == int(row["n_eval"])
    summary[arm] = {"rows": len(rows), "dumps": 24, "natural_eog": [r["scenario"] for r in eos], "stop_branch": stop["stop_branch"]}

for i in range(4):
    assert ids("seq", f"termgate-eog{i}.txt") == ids("cand", f"termgate-eog{i}.txt"), f"EOG full IDs: {i}"
(base / "checked-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS: 16/16 termination scenarios,48 token dumps, cancel/stop state proofs and natural EOG parity")
