#!/usr/bin/env bash
# T16-final-gaps.sh — focused closure of the three audit gaps on the FINAL
# artifact (rev244, LTCG), no broad rerun:
#  G1 memory/stability ON rev244: WS (device portal) sampled per phase
#     around restore + multi-turn generation; app must stay serving.
#  G2 failed save leaves NO partial file: invalid slot id, and save with no
#     resident KV — both must refuse AND leave the store unchanged (no
#     entry, no leftover file for the attempted ids).
#  G3 fingerprint refusal beyond wrong-version: cited from the rev244
#     battery (B9 edits the fingerprint field of a valid slot) — verified
#     here by reading the verdict, no rerun.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T16"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.244_x64__pj67f1fcj4n14"
API=http://192.168.1.26:11434
cd "$REPO"
API=http://192.168.1.26:11434
set -a
source ~/.config/xllama/xbox-env
set +a
mkdir -p "$D"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
# shellcheck disable=SC1091
source "$D/../T2-lib.sh"
: > "$D/verdicts.txt"
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'

# ---------- G2: failed saves leave nothing behind
echo "== G2 failed-save atomicity" | tee -a "$D/verdicts.txt"
list_ids() { curl -s -m 30 "$API/v1/slots" | python3 -c "import json,sys;print(sorted(s['slot'] for s in json.load(sys.stdin)['slots']))"; }
BEFORE=$(list_ids)
echo "store before: $BEFORE" | tee -a "$D/verdicts.txt"
code=$(curl -s -m 30 -o "$D/badid.json" -w "%{http_code}" -X POST "$API/v1/slots/save" \
  -H 'Content-Type: application/json' -d '{"slot":"bad/id"}')
echo "invalid-id save: http=$code body=$(head -c 120 "$D/badid.json")" | tee -a "$D/verdicts.txt"
restart_verified "no-kv"
code=$(curl -s -m 30 -o "$D/nokv.json" -w "%{http_code}" -X POST "$API/v1/slots/save" \
  -H 'Content-Type: application/json' -d '{"slot":"t16-nokv"}')
echo "no-resident-KV save: http=$code body=$(head -c 120 "$D/nokv.json")" | tee -a "$D/verdicts.txt"
AFTER=$(list_ids)
echo "store after: $AFTER" | tee -a "$D/verdicts.txt"
[[ "$BEFORE" == "$AFTER" ]] && pass "G2: store unchanged after both failed saves" || fail "G2: store changed"
for id in bad/id t16-nokv; do
  ./scripts/deploy.sh fetch-file "$PFN" "$id.kv" "$D/leftover-$id" "slots" >/dev/null 2>&1 \
    && { fail "G2: file exists for failed save $id"; } || pass "G2: no file for failed save '$id'"
  ./scripts/deploy.sh fetch-file "$PFN" "$id.kv.tmp" "$D/leftover-tmp-$id" "slots" >/dev/null 2>&1 \
    && fail "G2: leftover .tmp for $id" || pass "G2: no leftover .tmp for '$id'"
done

# ---------- G1: memory/stability on rev244
echo "== G1 memory/stability (rev244 Final)" | tee -a "$D/verdicts.txt"
restart_verified "g1"
ws_sample "g1-after-restart"
chat "$D/g1-seed.json" 16 "$P1"
slot_save "$D/g1-save.json" "t16"
ws_sample "g1-after-seed+save"
restart_verified "g1-restore"
chat "$D/g1-pre.json" 1 '[{"role":"user","content":"x"}]'
ws_sample "g1-after-preload"
slot_restore "$D/g1-restore.json" "t16" || fail "g1 restore"
chat "$D/g1-c1.json" 24 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]' || fail "g1 t1"
ws_sample "g1-after-restore-turn"
chat "$D/g1-c2.json" 48 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."},{"role":"assistant","content":"BRAVO-7"},{"role":"user","content":"Explain RoPE attention in two sentences."}]' || fail "g1 t2"
ws_sample "g1-after-t2"
chat "$D/g1-c3.json" 32 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."},{"role":"assistant","content":"BRAVO-7"},{"role":"user","content":"Explain RoPE attention in two sentences."},{"role":"assistant","content":"x"},{"role":"user","content":"Now spell the cipher backwards."}]' || fail "g1 t3"
ws_sample "g1-after-t3"
curl -s -m 30 "$API/v1/slots" | head -c 120 | tee -a "$D/verdicts.txt"; echo
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t16"}' >/dev/null || true
echo "== G3 fingerprint refusal (existing evidence, not rerun)" | tee -a "$D/verdicts.txt"
grep -E "B9 wrong-version" "$REPO/bench/results/qwen4b-maxperf/slots-rev211/slots-on/verdicts.txt" | tee -a "$D/verdicts.txt"
grep -n "edited-fp" "$REPO/bench/results/qwen4b-maxperf/slots-rev211/slot-battery.sh" | head -2 | tee -a "$D/verdicts.txt"
echo "== T16 done"
