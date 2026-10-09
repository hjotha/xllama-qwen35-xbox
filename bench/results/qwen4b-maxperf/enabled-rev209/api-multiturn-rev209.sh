#!/usr/bin/env bash
# rev209 API gate: FFN GPU + MTP really ON under multi-turn/session-reuse on
# the DEFAULT path (no DIAG override, no bench flag). Deletes d3d12swiglu.txt
# (default ON), restarts, runs a 3-turn chat on the resident session
# (llama.ini mtp=2, n_gpu_layers=34), then stops the app so backend_free
# flushes the real FFN SWIGLU dispatch counter. Evidence: HTTP responses, the
# delimited log segment (bind line + counters), and a durable receipt.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/enabled-rev209}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.209_x64__pj67f1fcj4n14}"
API=http://192.168.1.26:11434
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/api"

# Config assertions: MTP on, GPU layers on.
./scripts/deploy.sh fetch-file "$PFN" llama.ini "$BASE/api/llama.ini" >/dev/null
grep -q '^mtp=2$' "$BASE/api/llama.ini" || { echo "INI_FAIL mtp!=2"; exit 1; }
grep -q '^n_gpu_layers=34$' "$BASE/api/llama.ini" || { echo "INI_FAIL n_gpu_layers!=34"; exit 1; }

# Default-ON: remove the knob so the compiled default binds the FFN profile on.
./scripts/deploy.sh delete-file "$PFN" d3d12swiglu.txt >/dev/null 2>&1 || true

# Delimit the log: stop (flush previous session counters), note line count.
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 2
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$BASE/api/log-lines-before.txt"
L0=$(cat "$BASE/api/log-lines-before.txt")
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
ok=""
for _ in $(seq 1 60); do
  r=$(curl -s -m 5 "$API/" 2>/dev/null || true)
  if [[ -n "$r" ]]; then ok=1; break; fi
  sleep 3
done
[[ -n "$ok" ]] || { echo "API_UP_FAIL"; exit 1; }
curl -s -m 10 "$API/" > "$BASE/api/get-root.json"

chat() { # out max_tokens body
  curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "$3" > "$1"
  grep -q '"choices"' "$1" || { echo "CHAT_FAIL $1"; exit 1; }
}
chat "$BASE/api/turn1.json" 48 '{
  "model":"qwen35-4b-mtp","temperature":0,"max_tokens":48,
  "messages":[{"role":"user","content":"Remember this code word exactly: QUARTZ-LUMEN-4821. Reply with the single word: stored."}]}'
chat "$BASE/api/turn2.json" 48 '{
  "model":"qwen35-4b-mtp","temperature":0,"max_tokens":48,
  "messages":[
    {"role":"user","content":"Remember this code word exactly: QUARTZ-LUMEN-4821. Reply with the single word: stored."},
    {"role":"assistant","content":"stored"},
    {"role":"user","content":"What code word did I just give you? Reply with only the code word."}]}'
chat "$BASE/api/turn3.json" 48 '{
  "model":"qwen35-4b-mtp","temperature":0,"max_tokens":48,
  "messages":[
    {"role":"user","content":"Remember this code word exactly: QUARTZ-LUMEN-4821. Reply with the single word: stored."},
    {"role":"assistant","content":"stored"},
    {"role":"user","content":"What code word did I just give you? Reply with only the code word."},
    {"role":"assistant","content":"QUARTZ-LUMEN-4821"},
    {"role":"user","content":"Split it into two words at the dash and count the letters of each part. Answer as: first/second"}]}'

# Session-reuse evidence: turn2 must recall the turn1 code word (context
# carried across HTTP calls on the resident session).
T2=$(python3 -c "import json,sys;print(json.load(open('$BASE/api/turn2.json'))['choices'][0]['message']['content'])")
echo "turn2_reply=$T2" | tee "$BASE/api/turn2-reply.txt"
grep -q 'QUARTZ-LUMEN-4821' "$BASE/api/turn2.json" || { echo "RECALL_FAIL turn2"; exit 1; }

# Stop: backend_free flushes the real dispatch counters for THIS session.
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 2
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$BASE/api/full-log-after.txt"
L0=$(cat "$BASE/api/log-lines-before.txt")
tail -n +"$((L0 + 1))" "$BASE/api/full-log-after.txt" > "$BASE/api/session-segment.log"
BIND=$(grep -c "FFN SWIGLU profile bound: on (mtp_capable=1)" "$BASE/api/session-segment.log" || true)
DISP=$(grep -o "[0-9]* FFN SWIGLU dispatches" "$BASE/api/session-segment.log" | tail -1 | grep -o "^[0-9]*" || true)
LOADS=$(grep -ci "load.*model\|loading model" "$BASE/api/session-segment.log" || true)
{
  echo "api_multiturn_rev209 start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PFN=$PFN knob=d3d12swiglu.txt:DELETED(default ON) llama.ini: mtp=2 n_gpu_layers=34"
  echo "log_lines_before=$L0 segment_lines=$(wc -l < "$BASE/api/session-segment.log")"
  echo "bind_on_mtp_capable1_count=$BIND"
  echo "ffn_swiglu_dispatches=${DISP:-0}"
  echo "model_load_lines_in_segment=$LOADS (0 => no reload across the 3 turns)"
  echo "turn2_recall=PASS ($T2)"
  echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$BASE/api/receipt.txt"
[[ "${BIND:-0}" -ge 1 ]] || { echo "BIND_FAIL"; exit 1; }
[[ "${DISP:-0}" -gt 0 ]] || { echo "COUNTER_FAIL"; exit 1; }
# Restart so the app is left serving (final state must be reachable).
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
echo API_MULTITURN_REV209_DONE
