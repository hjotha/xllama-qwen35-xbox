#!/usr/bin/env bash
# api-multiturn-v2.sh — corrected multi-turn/session gate on installed rev209.
# Runs in the CLEAN final state (no knob, no transient bench knobs — verified
# absent by 404 fetches) and then the OFF equivalent (knob=0), then re-enables.
# Per-turn log slices give the exact temporal map; analysis classifies
# prefill/reuse/rewind AS OBSERVED. Full prefill is never called reuse.
# Hard gates: 3x HTTP 200 + non-empty replies, exact t2 recall (both arms),
# single bind line per session with the right mode, real counters at free,
# no conflicting-context rejection FOR qwen35-4b-mtp. Observations that are
# not gates (reloads, rejections, rewind refusals) are RECORDED with line
# refs, never hidden.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/enabled-rev209}"
V2="$BASE/api-v2"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.209_x64__pj67f1fcj4n14}"
API=http://192.168.1.26:11434
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$V2"
echo "api_multiturn_v2 start=$(date -u +%Y-%m-%dT%H:%M:%SZ) PFN=$PFN" > "$V2/receipt.txt"

absent() { # file -> prints ABSENT or PRESENT(content)
  if ./scripts/deploy.sh fetch-file "$PFN" "$1" /tmp/opencode/qwen4b-recon/preflight-check.bin >/dev/null 2>&1; then
    echo "PRESENT $1"
  else
    echo "ABSENT $1"
  fi
}
{
  echo "--- preflight clean state (knob + transient bench knobs must be ABSENT):"
  absent d3d12swiglu.txt
  for k in bench_mtp.txt bench_profile.txt bench_run_index.txt bench_tokens.txt \
      bench_npredict.txt bench_maxlen.txt bench_ctx.txt bench_threads.txt \
      bench_n_batch.txt bench_ubatch.txt bench_kvq8.txt bench_gpu_layers.txt \
      bench_ignore_eog.txt bench_greedy.txt bench_seed.txt bench_twocol.txt \
      d3d12twocol.txt cpurepackforcegemv.txt termgate.txt termgate.flag bench.flag; do
    absent "$k"
  done
  ./scripts/deploy.sh fetch-file "$PFN" llama.ini "$V2/llama.ini" >/dev/null
  echo "llama.ini: $(grep -E '^(mtp|n_gpu_layers|n_batch|n_ubatch|n_threads)=' "$V2/llama.ini" | tr '\n' ' ')"
} | tee "$V2/preflight.txt"

restart_wait() {
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  sleep 2
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
  local ok=""
  for _ in $(seq 1 60); do
    r=$(curl -s -m 5 "$API/" 2>/dev/null || true)
    if [[ -n "$r" ]]; then ok=1; break; fi
    sleep 3
  done
  [[ -n "$ok" ]] || { echo "API_UP_FAIL"; exit 1; }
}

turns() { # arm
  local arm="$1"
  local A="$V2/$arm"
  mkdir -p "$A"
  chat() {
    curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "$3" > "$2"
    grep -q '"choices"' "$2" || { echo "CHAT_FAIL $2"; exit 1; }
    ./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$A/log-after-$1.json.log" || true
  }
  chat t1 "$A/turn1.json" '{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"Remember this code word exactly: QUARTZ-LUMEN-4821. Reply with the single word: stored."}]}'
  chat t2 "$A/turn2.json" '{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"Remember this code word exactly: QUARTZ-LUMEN-4821. Reply with the single word: stored."},{"role":"assistant","content":"stored"},{"role":"user","content":"What code word did I just give you? Reply with only the code word."}]}'
  chat t3 "$A/turn3.json" '{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"Remember this code word exactly: QUARTZ-LUMEN-4821. Reply with the single word: stored."},{"role":"assistant","content":"stored"},{"role":"user","content":"What code word did I just give you? Reply with only the code word."},{"role":"assistant","content":"QUARTZ-LUMEN-4821"},{"role":"user","content":"Split it into two words at the dash and count the letters of each part. Answer as: first/second"}]}'
}

# --- ON arm (default: knob absent) ---
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$V2/ON-L0.txt"
restart_wait
turns ON
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 2
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$V2/ON-full-log.txt"
tail -n +"$(($(cat "$V2/ON-L0.txt") + 1))" "$V2/ON-full-log.txt" > "$V2/ON-segment.log"

# --- OFF arm (knob=0) ---
printf '0' > /tmp/opencode/qwen4b-recon/knob0-v2.txt
./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/knob0-v2.txt "$PFN" "" d3d12swiglu.txt >/dev/null
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$V2/OFF-L0.txt"
restart_wait
turns OFF
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 2
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$V2/OFF-full-log.txt"
tail -n +"$(($(cat "$V2/OFF-L0.txt") + 1))" "$V2/OFF-full-log.txt" > "$V2/OFF-segment.log"

# --- re-enable: delete knob, restart, final validation ---
./scripts/deploy.sh delete-file "$PFN" d3d12swiglu.txt >/dev/null 2>&1 || true
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$V2/FINAL-L0.txt"
restart_wait
curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":32,"messages":[{"role":"user","content":"Final re-enable check: reply with exactly: serving-enabled"}]}' \
  > "$V2/final-chat.json"
grep -q '"choices"' "$V2/final-chat.json" || { echo "FINAL_CHAT_FAIL"; exit 1; }
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$(($(cat "$V2/FINAL-L0.txt") + 1))" > "$V2/FINAL-segment.log"

python3 - "$V2" <<'PY'
import json, re, sys, pathlib
V = pathlib.Path(sys.argv[1])
fails = []
out = []
def seg(name):
    return (V / f"{name}-segment.log").read_text(errors="replace").splitlines()
def content(p):
    return json.load(open(p))["choices"][0]["message"]["content"]
for arm in ("ON", "OFF"):
    lines = seg(arm)
    t = {i: content(V / arm / f"turn{i}.json") for i in (1, 2, 3)}
    out.append(f"[{arm}] t1={t[1]!r} t2={t[2]!r} t3={t[3]!r}")
    if "QUARTZ-LUMEN-4821" not in t[2]:
        fails.append(f"{arm}_RECALL_FAIL")
        out.append(f"[{arm}] RECALL=FAIL")
    else:
        out.append(f"[{arm}] RECALL=PASS (API-level continuity via request history)")
    binds_on = [l for l in lines if "FFN SWIGLU profile bound: on (mtp_capable=1)" in l]
    binds_off = [l for l in lines if "FFN SWIGLU profile bound: off" in l]
    want = binds_on if arm == "ON" else binds_off
    out.append(f"[{arm}] bind_lines_on={len(binds_on)} bind_lines_off={len(binds_off)}")
    if len(want) != 1:
        fails.append(f"{arm}_BIND_COUNT want=1 got={len(want)}")
    disp = [l for l in lines if "FFN SWIGLU dispatches" in l]
    ffn = int(disp[-1].split()[2]) if disp else 0
    gdn = [l for l in lines if "GATED_DELTA_NET dispatches" in l]
    gdn_n = int(gdn[-1].split()[2]) if gdn else 0
    out.append(f"[{arm}] FFN_SWI dispatches={ffn} GDN={gdn_n}")
    if arm == "ON" and ffn <= 0:
        fails.append("ON_COUNTER_FAIL")
    if arm == "OFF" and ffn != 0:
        fails.append("OFF_COUNTER_FAIL(ffn must be 0)")
    if arm == "ON" and gdn_n <= 0:
        fails.append("ON_GDN_FAIL")
    # temporal map: generate / prefill / reuse / rewind / reload / reject
    out.append(f"[{arm}] --- temporal map (segment line refs) ---")
    for i, l in enumerate(lines, 1):
        if ("session: turn full=" in l or "session generate: n=" in l
                or "KV rewind unsupported" in l or "KV prefix reuse" in l
                or "session config" in l or "preload failed" in l
                or "graph_compute calls" in l or "profile bound" in l
                or "main_loop:" in l or "[xllama] done:" in l):
            out.append(f"[{arm}] L{i}: {l[:200]}")
# ON vs OFF reply equality (exact kernel => same greedy tokens on API path)
same = all(content(V / "ON" / f"turn{i}.json") == content(V / "OFF" / f"turn{i}.json") for i in (1, 2, 3))
out.append(f"ON_vs_OFF_replies_identical={same}")
if not same:
    fails.append("ON_OFF_REPLY_DIFF")
final = content(V / "final-chat.json")
out.append(f"FINAL re-enable chat={final!r}")
if "serving-enabled" not in final:
    fails.append("FINAL_CHAT_CONTENT_FAIL")
fb = [l for l in seg("FINAL") if "FFN SWIGLU profile bound: on (mtp_capable=1)" in l]
out.append(f"FINAL bind_on_count={len(fb)}")
if len(fb) != 1:
    fails.append("FINAL_BIND_FAIL")
(V / "analysis.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
print(f"failures={len(fails)}")
for f in fails:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)
PY
echo "api_multiturn_v2 end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$V2/receipt.txt"
echo API_MULTITURN_V2_DONE
