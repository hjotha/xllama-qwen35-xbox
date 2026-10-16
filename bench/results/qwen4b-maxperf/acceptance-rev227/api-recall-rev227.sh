#!/usr/bin/env bash
# api-recall-rev227.sh v3 — multi-turn recall + binding + counters on the final
# package. Self-contained: restarts first so the delimited segment holds the
# startup bind and both turns. Counter capture WITHOUT process death (whose
# TerminateProcess flush is racy): after the qwen turns, one lfm25-350m chat
# swaps the single resident session, freeing the qwen backend (flush with its
# counters) while the app stays up; then swaps back to qwen and revalidates.
# Asserts API-level continuity (exact recall), single bind on (mtp_capable=1),
# real qwen counters at the swap flush. Records per-turn prefill/reuse lines.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/acceptance-rev227}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.227_x64__pj67f1fcj4n14}"
API=http://192.168.1.26:11434
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/api"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$BASE/api/L0.txt"
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 2
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
ok=""
for _ in $(seq 1 60); do
  r=$(curl -s -m 5 "$API/" 2>/dev/null || true)
  if [[ -n "$r" ]]; then ok=1; break; fi
  sleep 3
done
[[ -n "$ok" ]] || { echo "API_UP_FAIL"; ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true; exit 1; }
chat() { # out max_tokens model messages_json
  curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$3\",\"temperature\":0,\"max_tokens\":$2,\"messages\":$4}" > "$1"
  grep -q '"choices"' "$1" || { echo "CHAT_FAIL $1"; exit 1; }
}
chat "$BASE/api/t1.json" 16 qwen35-4b-mtp '[{"role":"user","content":"The gate cipher is RECALL-5. Reply with: got-it."}]'
T1B='[{"role":"user","content":"The gate cipher is RECALL-5. Reply with: got-it."},{"role":"assistant","content":"got-it"},{"role":"user","content":"Repeat the cipher back exactly."}]'
chat "$BASE/api/t2.json" 32 qwen35-4b-mtp "$T1B"
T2=$(python3 -c "import json;print(json.load(open('$BASE/api/t2.json'))['choices'][0]['message']['content'])")
echo "t2_reply=$T2" | tee "$BASE/api/t2-reply.txt"
grep -q 'RECALL-5' "$BASE/api/t2.json" || { echo "RECALL_FAIL"; exit 1; }
# swap out to lfm (frees the qwen backend -> flush with counters), then back
chat "$BASE/api/swap-lfm.json" 16 lfm25-350m '[{"role":"user","content":"Reply with the single word away."}]'
chat "$BASE/api/swap-back.json" 16 qwen35-4b-mtp '[{"role":"user","content":"Reply with the single word home."}]'
SB=$(python3 -c "import json;print(json.load(open('$BASE/api/swap-back.json'))['choices'][0]['message']['content'])")
[[ -n "$SB" ]] || { echo "SWAPBACK_FAIL"; exit 1; }
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$BASE/api/full-log.txt"
tail -n +"$(($(cat "$BASE/api/L0.txt") + 1))" "$BASE/api/full-log.txt" > "$BASE/api/segment.txt"
BIND=$(grep -c "FFN SWIGLU profile bound: on (mtp_capable=1)" "$BASE/api/segment.txt" || true)
DISP=$(grep -o "[0-9]* FFN SWIGLU dispatches" "$BASE/api/segment.txt" | tail -1 | grep -o "^[0-9]*" || true)
GDN=$(grep -o "[0-9]* GATED_DELTA_NET dispatches" "$BASE/api/segment.txt" | tail -1 | grep -o "^[0-9]*" || true)
{
  echo "api_recall_rev227 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) PFN=$PFN mode=final"
  echo "recall=PASS t2=[$T2] swapback=[$SB]"
  echo "bind_on_mtp1=$BIND ffn_disp=${DISP:-0} gdn=${GDN:-0}"
  grep -E "session: turn full|KV prefix reuse|KV rewind unsupported|session generate: n=|session config:" "$BASE/api/segment.txt" | head -12
} | tee "$BASE/api/receipt.txt"
rm -f "$BASE/api/full-log.txt"
[[ "${BIND:-0}" -ge 1 ]] || { echo "BIND_FAIL"; exit 1; }
[[ "${DISP:-0}" -gt 0 ]] || { echo "COUNTER_FAIL"; exit 1; }
echo API_RECALL_DONE
