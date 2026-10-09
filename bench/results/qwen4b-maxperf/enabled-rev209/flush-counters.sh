#!/usr/bin/env bash
# flush-counters.sh — supplementary counter receipt for the v2 ON session.
# The v2 script's get-log raced the backend_free flush, so the ON segment has
# no counter lines (script verdict ON_COUNTER_FAIL is a measurement race, not
# a device failure — v1 already showed 79 FFN dispatches on the same build).
# This captures the flush with a longer settle, then restarts serving and
# re-validates the final enabled state. No knob changes.
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
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$V2/FLUSH-L0.txt"
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 8
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$V2/FLUSH-full-log.txt"
tail -n +"$(($(cat "$V2/FLUSH-L0.txt") + 1))" "$V2/FLUSH-full-log.txt" > "$V2/FLUSH-segment.log"
grep -E "graph_compute calls|SWIGLU dispatches|GATED_DELTA_NET|profile bound" "$V2/FLUSH-segment.log" | tee "$V2/FLUSH-counters.txt"
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
ok=""
for _ in $(seq 1 60); do
  r=$(curl -s -m 5 "$API/" 2>/dev/null || true)
  if [[ -n "$r" ]]; then ok=1; break; fi
  sleep 3
done
[[ -n "$ok" ]] || { echo "FLUSH_API_UP_FAIL"; exit 1; }
curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"reply with the single word serving"}]}' \
  > "$V2/FLUSH-chat.json"
grep -q '"choices"' "$V2/FLUSH-chat.json" || { echo "FLUSH_CHAT_FAIL"; exit 1; }
echo "FLUSH_DONE api=PASS chat=PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$V2/FLUSH-counters.txt"
