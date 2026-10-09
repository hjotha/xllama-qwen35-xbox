#!/usr/bin/env bash
# Immediate fallback: FFN SWIGLU knob forced OFF + controlled restart + API
# validation. Keeps the running package and every other config untouched.
# Run on demand; re-enable afterwards by deleting d3d12swiglu.txt and
# restarting (api-multiturn-rev209.sh does exactly that).
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
mkdir -p "$BASE/fallback"
printf '0' > /tmp/opencode/qwen4b-recon/knob0.txt
./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/knob0.txt "$PFN" "" d3d12swiglu.txt >/dev/null
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$BASE/fallback/log-lines-before.txt"
L0=$(cat "$BASE/fallback/log-lines-before.txt")
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 2
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
ok=""
for _ in $(seq 1 60); do
  r=$(curl -s -m 5 "$API/" 2>/dev/null || true)
  if [[ -n "$r" ]]; then ok=1; break; fi
  sleep 3
done
[[ -n "$ok" ]] || { echo "FALLBACK_API_UP_FAIL"; exit 1; }
curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"reply with the single word ok"}]}' \
  > "$BASE/fallback/chat.json"
grep -q '"choices"' "$BASE/fallback/chat.json" || { echo "FALLBACK_CHAT_FAIL"; exit 1; }
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$BASE/fallback/session-segment.log"
OFF=$(grep -c "FFN SWIGLU profile bound: off" "$BASE/fallback/session-segment.log" || true)
{
  echo "fallback_knob0 start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "knob=d3d12swiglu.txt=0 restart=controlled api=PASS chat=PASS"
  echo "bind_off_count=$OFF"
  echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$BASE/fallback/receipt.txt"
[[ "${OFF:-0}" -ge 1 ]] || { echo "FALLBACK_BIND_FAIL"; exit 1; }
echo FALLBACK_KNOB0_DONE
