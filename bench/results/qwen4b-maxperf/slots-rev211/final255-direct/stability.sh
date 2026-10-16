#!/usr/bin/env bash
# Final247 bounded memory cycle: restart, preload, restore, three turns, six WS samples.
# No long-soak or leak-free claim.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/stability"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14"
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

# ---------- G1: memory/stability on rev255
echo "== G1 memory/stability (rev255 Final)" | tee -a "$D/verdicts.txt"
restart_verified "g1"
ws_sample "g1-after-restart"
chat "$D/g1-seed.json" 16 "$P1"
slot_save "$D/g1-save.json" "r255stability"
ws_sample "g1-after-seed+save"
restart_verified "g1-restore"
chat "$D/g1-pre.json" 1 '[{"role":"user","content":"x"}]'
ws_sample "g1-after-preload"
slot_restore "$D/g1-restore.json" "r255stability" || fail "g1 restore"
chat "$D/g1-c1.json" 24 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]' || fail "g1 t1"
ws_sample "g1-after-restore-turn"
chat "$D/g1-c2.json" 48 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."},{"role":"assistant","content":"BRAVO-7"},{"role":"user","content":"Explain RoPE attention in two sentences."}]' || fail "g1 t2"
ws_sample "g1-after-t2"
chat "$D/g1-c3.json" 32 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."},{"role":"assistant","content":"BRAVO-7"},{"role":"user","content":"Explain RoPE attention in two sentences."},{"role":"assistant","content":"x"},{"role":"user","content":"Now spell the cipher backwards."}]' || fail "g1 t3"
ws_sample "g1-after-t3"
curl -s -m 30 "$API/v1/slots" | head -c 120 | tee -a "$D/verdicts.txt"; echo
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"r255stability"}' >/dev/null || true

python3 - "$D" <<'CHECK'
import json,pathlib,re,sys
p=pathlib.Path(sys.argv[1]);v=(p/"verdicts.txt").read_text();ws=[int(x) for x in re.findall(r"raw=(\d+)",v)];assert len(ws)==6,(len(ws),v)
for name in ("g1-save.json","g1-restore.json"):
 j=json.loads((p/name).read_text());assert j["ok"] is True and j["tokens"]>0
print("PASS bounded stability cycle; WS samples bytes:",ws,"loaded range:",min(ws[1:]),max(ws[1:]))
CHECK
