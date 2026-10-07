#!/usr/bin/env python3
"""Install the selected normal profile, prove actual API MTP engagement, retain receipts."""
import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
OUT = BASE / sys.argv[1]
assert not OUT.exists(), "retain historical evidence; choose a fresh output directory"
OUT.mkdir()
PFN = os.environ["XLLAMA_EXPECTED_PFN"]
SHA = os.environ["XLLAMA_MSIX_SHA256"]
DEPLOY = str(ROOT / "scripts/deploy.sh")
def deploy(*args):
    return subprocess.run([DEPLOY, *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
assert deploy("pfn").strip() == PFN, "installed package mismatch"
(OUT / "llama.ini.before").write_text(deploy("fetch-file", PFN, "llama.ini", str(OUT / "llama.ini.before.raw")))
profile = {
 "llama.ini": "n_gpu_layers=34\nn_ctx=2048\nn_threads=2\nn_batch=64\nn_ubatch=64\nkv_q8=0\nmtp=2\nmtp_pmin=50\n",
 "d3d12q6tile.txt":"0", "d3d12gdn.txt":"1", "d3d12q8.txt":"1", "flashattn.txt":"2",
 "mtp_threads.txt":"1", "mtp_catchup_logits.txt":"0", "d3d12twocol.txt":"auto",
 "cpurepackforcegemv.txt":"2", "gguf_gpu_layers.txt":"34", "kv_q8.txt":"0",
 "model.txt":"qwen35-4b-mtp", "api.flag":"go",
}
previous = (OUT / "llama.ini.before.raw").read_text(encoding="utf-8-sig")
owned = {line.partition("=")[0] for line in profile["llama.ini"].splitlines()}
profile["llama.ini"] = "\n".join(line for line in previous.splitlines() if line.partition("=")[0].strip() not in owned) + "\n" + profile["llama.ini"]
deploy("stop-app", PFN)
for name, value in profile.items():
    path = OUT / name
    path.write_text(value)
    deploy("upload-file", str(path), PFN, "", name)
    got = OUT / (name + ".verified")
    deploy("fetch-file", PFN, name, str(got))
    assert got.read_bytes() == path.read_bytes(), "durable profile mismatch: " + name
transient = ["bench.flag", "diffuse.flag", "diffuse-inproc.flag", "gpustep-inproc.flag", "membw.flag", "diskbw.flag", "gpubw.flag", "gpugemv.flag", "tttarget.flag", "d3d12be.flag", "gpustep.flag", "ramceil.flag", "mic.flag", "logits.flag", "replay.flag", "diverge.flag", "termgate.flag", "native-capture.flag", "bench_mtp_session.txt", "bench_turns.txt", "mtp_pool.txt", "verify_trace.txt"]
for name in transient:
    deploy("delete-file", PFN, name)
(OUT / "start-app.txt").write_text(deploy("start-app", PFN))
api = "http://" + os.environ["XBOX_IP"] + ":11434"
deadline = time.monotonic() + 120
while True:
    try:
        with urlopen(api + "/", timeout=5) as r:
            health = r.read().decode()
        break
    except OSError:
        if time.monotonic() >= deadline: raise
        time.sleep(2)
(OUT / "health.json").write_text(health)
assert json.loads(health)["status"] == "ok"
requests = ["Explain in three short sentences why speculative decoding verifies draft tokens.", "Write a concise Python function that returns the greatest common divisor of two positive integers."]
for i, prompt in enumerate(requests, 1):
    payload = {"model":"qwen35-4b-mtp", "messages":[{"role":"user", "content":prompt}], "temperature":0, "seed":1, "max_tokens":64, "stream":False}
    (OUT / f"request{i}.json").write_text(json.dumps(payload, indent=2)+"\n")
    with urlopen(Request(api + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}), timeout=300) as r:
        body = r.read().decode()
    (OUT / f"response{i}.json").write_text(body)
    result = json.loads(body)
    assert result["choices"][0]["message"]["content"].strip(), "empty API response"
raw = deploy("get-log", PFN)
(OUT / "normal-api-full.log").write_text(raw)
raw = raw[raw.rfind("[xllama] startup: Q6_K LM-head columns="):]
(OUT / "normal-api.log").write_text(raw)
for needle in ["Q6_K LM-head columns=0", "Q8_0 matmul=D3D12", "attention split-KV=disabled for consistent query reduction", "GATED_DELTA_NET=D3D12", "mtp=1 depth=2 pmin=0.50", "n_batch=64 n_ubatch=64", "draft CPU threads=1"]:
    assert needle in raw, "missing effective normal profile: " + needle
stats = re.findall(r"session generate:.*drafted=(\d+) \(mtp=(\d+) lookup=(\d+)\) spec_accept=(\d+) \(mtp=(\d+) lookup=(\d+)\)", raw)
assert len(stats) >= 2 and all(int(s[1])>0 and int(s[4])>0 for s in stats[-2:]), "normal API did not accept MTP tokens"
app_status = deploy("status", PFN).strip()
assert "running" in app_status.lower() and "not running" not in app_status.lower(), "normal API app stopped"
summary = {"status":"PASS", "package_pfn":PFN, "package_sha256":SHA, "profile":profile, "requests":2, "stats":[list(map(int,s)) for s in stats[-2:]], "app_status":app_status}
(OUT / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
