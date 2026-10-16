#!/usr/bin/env bash
# slot-smoke-rev227.sh — slot save/restore/continue + refusal + list on the
# final package. Token-exact continuation vs fresh prefill (relative, same
# build). Fast smoke, not the full battery (which ran on rev215-iteration).
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
mkdir -p "$BASE/slotsmoke"
fails=0
chat() {
  curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen35-4b-mtp\",\"temperature\":0,\"max_tokens\":$2,\"messages\":$3}" > "$1"
  grep -q '"choices"' "$1" || { echo "CHAT_FAIL $1"; exit 1; }
}
C() { python3 -c "import json,sys;print(json.load(open('$1'))['choices'][0]['message']['content'])"; }
P='[{"role":"user","content":"Smoke cipher SEAL-11. Reply: sealed."}]'
PC='[{"role":"user","content":"Smoke cipher SEAL-11. Reply: sealed."},{"role":"assistant","content":"sealed"},{"role":"user","content":"Repeat the cipher back exactly."}]'
chat "$BASE/slotsmoke/seed.json" 16 "$P"
curl -s -m 120 "$API/v1/slots/save" -H 'Content-Type: application/json' -d '{"slot":"finalsmoke"}' > "$BASE/slotsmoke/save.json"
grep -q '"ok":true' "$BASE/slotsmoke/save.json" || { echo "SAVE_FAIL"; exit 1; }
curl -s -m 120 "$API/v1/slots/restore" -H 'Content-Type: application/json' -d '{"slot":"finalsmoke"}' > "$BASE/slotsmoke/restore.json"
grep -q '"ok":true' "$BASE/slotsmoke/restore.json" || { echo "RESTORE_FAIL"; exit 1; }
chat "$BASE/slotsmoke/cont.json" 16 "$PC"
T_R="$(C "$BASE/slotsmoke/cont.json")"
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true; sleep 2
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
for _ in $(seq 1 40); do curl -s -m 5 "$API/" >/dev/null 2>&1 && break; sleep 3; done
chat "$BASE/slotsmoke/seed2.json" 16 "$P"
chat "$BASE/slotsmoke/cont-fresh.json" 16 "$PC"
T_F="$(C "$BASE/slotsmoke/cont-fresh.json")"
[[ "$T_R" == "$T_F" ]] && echo "SMOKE restore-continue==fresh [$T_R]" || { echo "SMOKE_FAIL [$T_R] vs [$T_F]"; exit 1; }
printf 'junk' > /tmp/opencode/qwen4b-recon/junk.kv
./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/junk.kv "$PFN" "slots" "junk.kv" >/dev/null
if curl -s -m 60 "$API/v1/slots/restore" -H 'Content-Type: application/json' -d '{"slot":"junk"}' | grep -q '"ok":true'; then
  echo "SMOKE_REFUSAL_FAIL"; exit 1
else echo "SMOKE refusal ok"; fi
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"finalsmoke"}' >/dev/null
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"junk"}' >/dev/null
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"nomtp"}' >/dev/null
curl -s -m 20 "$API/v1/slots" | tee "$BASE/slotsmoke/list.json"; echo
echo SLOT_SMOKE_DONE
