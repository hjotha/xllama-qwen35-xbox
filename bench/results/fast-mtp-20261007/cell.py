#!/usr/bin/env python3
"""Run one Xbox benchmark cell through the existing runner; retain its exact profile and receipts."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("stem")
p.add_argument("--threads", type=int, default=3)
p.add_argument("--draft", type=int, default=1)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--pmin", type=int, default=75)
p.add_argument("--q6", type=int, choices=(0, 1, 2, 4), default=1)
p.add_argument("--gdn", type=int, choices=(0, 1, 2), default=1)
p.add_argument("--fa", type=int, choices=(0, 1, 2), default=0)
p.add_argument("--runs", type=int, default=2)
p.add_argument("--predict", type=int, default=64)
p.add_argument("--prompt", default="bench/prompts/spec-chat-open.txt")
p.add_argument("--reference")
p.add_argument("--ignore-eog", action="store_true")
p.add_argument("--profile", type=int, choices=(0, 1), default=0)
a = p.parse_args()
if not re.fullmatch(r"[a-zA-Z0-9_.-]+", a.stem) or not (1 <= a.draft <= a.threads <= 6):
    p.error("invalid stem or draft/target thread counts")
if not (0 <= a.depth <= 16 and 0 <= a.pmin <= 100 and a.runs >= 2):
    p.error("invalid depth, probability or run count")
sha = os.environ["XLLAMA_MSIX_SHA256"]
pfn = os.environ["XLLAMA_EXPECTED_PFN"]
out = BASE / f"{a.stem}.csv"
if out.exists():
    p.error("cell already exists; retain it and use a new stem")
receipt = BASE / f"{a.stem}.json"
profile = {
    "d3d12q8.txt": "1", "d3d12q6tile.txt": str(a.q6),
    "d3d12gdn.txt": str(a.gdn), "flashattn.txt": str(a.fa),
    "mtp_threads.txt": str(a.draft), "mtp_catchup_logits.txt": "0",
    "cpurepackforcegemv.txt": "2", "d3d12twocol.txt": "auto",
}
meta = {"requested": vars(a), "profile": profile, "package_sha256": sha, "package_pfn": pfn,
        "classification": "diagnostic" if a.profile else ("pilot" if a.runs == 2 else "repeated"), "status": "started"}
receipt.write_text(json.dumps(meta, indent=2) + "\n")
try:
    with tempfile.TemporaryDirectory(prefix="xllama-cell-") as scratch:
        tmp = Path(scratch)
        for name, value in profile.items():
            f = tmp / name
            f.write_text(value)
            subprocess.run([str(ROOT / "scripts/deploy.sh"), "upload-file", str(f), pfn, "", name],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        runlog = tmp / "logs"
        runlog.mkdir()
        env = dict(os.environ, RUN_LOG_DIR=str(runlog))
        cmd = [str(ROOT / "scripts/bench-xbox-ort.sh"), "qwen35-4b-mtp", "--threads", str(a.threads),
               "--ctx", "2048", "--gpu-layers", "34", "--batch", "64", "--ubatch", "64",
               "--twocol", "auto", "--greedy", "--seed", "1", "--tokens", "--profile-phases", str(a.profile),
               "--mtp", str(a.depth), "--mtp-pmin", str(a.pmin), "--runs", str(a.runs),
               "--n-predict", str(a.predict), "--prompt", str(ROOT / a.prompt), "--out", str(out)]
        if a.ignore_eog:
            cmd.append("--ignore-eog")
        with (BASE / f"{a.stem}.driver.log").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = list(csv.DictReader(out.open()))
        if len(rows) != a.runs - 1:
            raise RuntimeError(f"missing measurements: {len(rows)} != {a.runs - 1}")
        stats = []
        hashes = []
        for run in range(1, a.runs + 1):
            raw = (runlog / f"run{run}.log").read_text(errors="replace")
            pos = raw.rfind("[xllama] bench: Q6_K LM-head columns=")
            if pos < 0:
                raise RuntimeError("run-scoped configuration receipt missing")
            scoped = raw[pos:]
            (BASE / f"{a.stem}.run{run}.log").write_text(scoped)
            expected = [f"Q6_K LM-head columns={a.q6}", "Q8_0 matmul=D3D12",
                        "prefill batch override: n_batch=64 n_ubatch=64", "offloaded 34/34"]
            expected.append("fused attention=" + ("forced" if a.fa else "auto"))
            if a.fa == 2:
                expected.append("attention split-KV=disabled for consistent query reduction")
            expected.append("GATED_DELTA_NET=" + {0: "CPU control", 1: "D3D12", 2: "D3D12 (T>=2)"}[a.gdn] + "\n")
            if a.depth:
                expected.append(f"draft CPU threads={a.draft}")
            for needle in expected:
                if needle not in scoped:
                    raise RuntimeError(f"effective configuration missing: {needle}")
            match = re.search(r"done:.*drafted=(\d+) spec_accept=(\d+)", scoped)
            if not match:
                raise RuntimeError("draft/accept receipt missing")
            drafted, accepted = map(int, match.groups())
            stats.append({"run": run, "drafted": drafted, "accepted": accepted})
            tokens = BASE / f"{a.stem}.run{run}.tokens"
            digest = hashlib.sha256(tokens.read_bytes()).hexdigest()
            hashes.append(digest)
            if a.reference:
                ref = BASE / f"{a.reference}.run{run}.tokens"
                if not ref.exists():
                    ref = BASE / f"{a.reference}.run1.tokens"
                if tokens.read_bytes() != ref.read_bytes():
                    raise RuntimeError(f"full-token mismatch vs {ref.name}, run {run}")
        if len(set(hashes)) != 1:
            raise RuntimeError("greedy repetitions emitted different full-token dumps")
        meta.update(status="complete", rows=len(rows), stats=stats, token_sha256=hashes[0],
                    median={k: statistics.median(float(r[k]) for r in rows)
                            for k in ("decode_tok_s", "ttft_ms", "peak_ws_mb")})
        if a.depth and not any(s["accepted"] > 0 for s in stats if s["run"] > 1):
            meta["mtp_acceptance"] = "none: exclude from MTP winner selection"
        print(json.dumps({"cell": a.stem, "n": len(rows), **meta["median"], "stats": stats}))
except Exception as e:
    meta.update(status="failed", error=str(e))
    raise
finally:
    receipt.write_text(json.dumps(meta, indent=2) + "\n")
