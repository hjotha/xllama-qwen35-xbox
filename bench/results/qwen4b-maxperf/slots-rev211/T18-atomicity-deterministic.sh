#!/usr/bin/env bash
# T18-atomicity-deterministic.sh — plan 009 atomicity subgate, WITHOUT
# timing kills (T17/T17b could not reach a partial write). Uses two
# DETERMINISTIC, isolated and reversible failure points that already exist
# in the write routine, aimed only at test ids:
#   F1  the tmp PATH is a directory  -> open fails (cannot open tmp).
#       Proves the PREVIOUS VALID SNAPSHOT survives a failing save.
#   F2  the TARGET PATH is a directory -> the whole write completes and the
#       RENAME fails -> the routine removes the tmp. Proves deterministic
#       temporary treatment on a real write-routine failure and that no
#       partial file appears.
#   F3  reproduces the post-restart "save/restore failed" observed in T17:
#       right after a restart the session is not resident yet, so the API
#       refuses the save with "chat first" until a chat loads the model.
# No disk filling, no data damage, no production change: only test slots
# t18/t19 and the two directories created here.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T18"
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
BASE_URL_="${XBOX_IP}"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

ensure_chat() { chat "$D/warm.json" 8 '[{"role":"user","content":"Reply with the single word ready."}]' >/dev/null 2>&1 || true; }
rm_dir() { # relpath under LocalState (best-effort)
  curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:$(curl "${CURL_AUTH[@]}" "${BASE_URL_}/" -o /dev/null -D - 2>/dev/null | sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' | tr -d '\r' | head -n1)" \
    -X DELETE "${BASE_URL_}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5C$1&filename=$2" >/dev/null 2>&1 || true
}
list_dir() { ./scripts/deploy.sh list-localstate "$PFN" "$1" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(sorted((i.get('Name') or '') for i in d.get('Items') or []))"; }

ensure_chat
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
chat "$D/seed.json" 16 "$P1" >/dev/null

echo "== baseline: a valid snapshot of t18" | tee -a "$D/verdicts.txt"
slot_save "$D/f0-save.json" "t18" && pass "baseline save t18" || fail "baseline save"
SNAP_SHA=$(./scripts/deploy.sh fetch-file "$PFN" "t18.kv" "$D/f0-snap.kv" "slots" >/dev/null 2>&1 && sha256sum "$D/f0-snap.kv" | cut -c1-16)
echo "snapshot t18.kv sha=$SNAP_SHA bytes=$(stat -c%s "$D/f0-snap.kv")" | tee -a "$D/verdicts.txt"
CHA=$(content_of "$D/seed.json")
PA="[{\"role\":\"user\",\"content\":\"The northern depot cipher is BRAVO-7. Acknowledge with: noted.\"},{\"role\":\"assistant\",\"content\":\"$CHA\"},{\"role\":\"user\",\"content\":\"Repeat the cipher back exactly.\"}]"
chat "$D/f0-chat.json" 16 "$PA" >/dev/null && echo "baseline restore+chat: $(content_of "$D/f0-chat.json")" | tee -a "$D/verdicts.txt"

echo "== F1: tmp path is a directory -> open fails; snapshot must survive" | tee -a "$D/verdicts.txt"
./scripts/deploy.sh mkdir-localstate "$PFN" 'slots\t18.kv.tmp' >/dev/null 2>&1
code=$(curl -s -m 120 -o "$D/f1-out.json" -w "%{http_code}" -X POST "$API/v1/slots/save" \
  -H 'Content-Type: application/json' -d '{"slot":"t18"}')
echo "F1 save: http=$code body=$(head -c 140 "$D/f1-out.json")" | tee -a "$D/verdicts.txt"
if slot_restore "$D/f1-restore.json" "t18"; then
  chat "$D/f1-chat.json" 16 "$PA" >/dev/null 2>&1
  A=$(content_of "$D/f1-chat.json")
  [[ "$A" == "BRAVO-7" ]] && pass "F1: previous valid snapshot preserved through the failing save (restore+chat ok)" || fail "F1: snapshot not preserved"
else
  fail "F1: snapshot restore failed"
fi
F1SHA=$(./scripts/deploy.sh fetch-file "$PFN" "t18.kv" "$D/f1-snap.kv" "slots" >/dev/null 2>&1 && sha256sum "$D/f1-snap.kv" | cut -c1-16)
echo "F1 post-failure t18.kv sha=$F1SHA (baseline was $SNAP_SHA)" | tee -a "$D/verdicts.txt"
[[ "$F1SHA" == "$SNAP_SHA" ]] && pass "F1: snapshot bytes unchanged by the failed save" || fail "F1: snapshot bytes changed"
echo "slots dir during F1: $(list_dir slots)" | tee -a "$D/verdicts.txt"

echo "== F2: target path is a directory -> write completes, rename fails" | tee -a "$D/verdicts.txt"
./scripts/deploy.sh mkdir-localstate "$PFN" 'slots\t19.kv' >/dev/null 2>&1
code=$(curl -s -m 300 -o "$D/f2-out.json" -w "%{http_code}" -X POST "$API/v1/slots/save" \
  -H 'Content-Type: application/json' -d '{"slot":"t19"}')
echo "F2 save: http=$code body=$(head -c 140 "$D/f2-out.json")" | tee -a "$D/verdicts.txt"
echo "slots dir after F2: $(list_dir slots)" | tee -a "$D/verdicts.txt"
if ./scripts/deploy.sh fetch-file "$PFN" "t19.kv.tmp" "$D/f2-tmp" "slots" >/dev/null 2>&1; then
  fail "F2: leftover .tmp remains ($(stat -c%s "$D/f2-tmp") bytes)"
else
  pass "F2: no leftover .tmp after the failed rename (temporary cleaned)"
fi
curl -s -m 30 -o "$D/f2-list.json" "$API/v1/slots" && python3 -c "
import json
d = json.load(open('$D/f2-list.json'))
ids = sorted(s['slot'] for s in d['slots'])
print('F2 store listing:', ids)" | tee -a "$D/verdicts.txt"
python3 -c "
import json
d = json.load(open('$D/f2-list.json'))
ids = sorted(s['slot'] for s in d['slots'])
print('F2_VERDICT', 'PASS_no_partial_entry' if 't19' not in ids else 'FAIL_partial_entry')" | tee -a "$D/verdicts.txt"

echo "== F3: post-restart save/restore reproduction (T17 explanation)" | tee -a "$D/verdicts.txt"
restart_verified "F3"
code=$(curl -s -m 60 -o "$D/f3-save.json" -w "%{http_code}" -X POST "$API/v1/slots/save" \
  -H 'Content-Type: application/json' -d '{"slot":"t18"}')
echo "F3 save right after restart: http=$code body=$(head -c 140 "$D/f3-save.json")" | tee -a "$D/verdicts.txt"
code=$(curl -s -m 60 -o "$D/f3-restore.json" -w "%{http_code}" -X POST "$API/v1/slots/restore" \
  -H 'Content-Type: application/json' -d '{"slot":"t18"}')
echo "F3 restore right after restart: http=$code body=$(head -c 140 "$D/f3-restore.json")" | tee -a "$D/verdicts.txt"
ensure_chat
slot_save "$D/f3-save2.json" "t18" && pass "F3: save succeeds after a chat loads the model" || fail "F3: save still failing"
echo "== cleanup" | tee -a "$D/verdicts.txt"
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t19"}' >/dev/null || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t18"}' >/dev/null || true
./scripts/deploy.sh delete-file "$PFN" "t19.kv" "slots" >/dev/null 2>&1 || true
rm_dir 'slots' 't19.kv'
rm_dir 'slots' 't18.kv.tmp'
echo "slots dir final: $(list_dir slots)" | tee -a "$D/verdicts.txt"
echo "store final: $(curl -s -m 30 "$API/v1/slots")" | tee -a "$D/verdicts.txt"
echo "== T18 done"
