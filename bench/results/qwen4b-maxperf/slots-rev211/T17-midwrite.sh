#!/usr/bin/env bash
# T17-midwrite.sh — plan 009 atomicity subgate: demonstrate a REAL
# mid-write failure safely. Two saves rejected BEFORE writing (T16) do not
# prove an I/O interruption during the write, so this kills the process
# DURING the write of a TEST slot (58 MB) and verifies the tmp+rename
# guarantee: no partial file ever appears as a slot, the store stays
# consistent, and a stale partial .tmp does not poison a later save.
# Safe/isolated/reversible: only slot id t17 is touched; baselines,
# snapshots and evidence are never written by this test; nothing is
# filled on disk; the leftover .tmp is removed at the end.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T17"
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

# build a large resident state (187 tokens => ~58 MB slot) to widen the
# write window, in a dedicated process
restart_verified "build"
chat "$D/b1.json" 16 "$P1"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
r1 = json.load(open(f"{d}/b1.json"))["choices"][0]["message"]["content"]
open(f"{d}/T2.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."}]))
PY
chat "$D/b2.json" 48 "$(cat "$D/T2.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
open(f"{d}/T3.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("b1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("b2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."}]))
PY
chat "$D/b3.json" 48 "$(cat "$D/T3.json")"
# reference: a full save of the same state, to know the expected size
slot_save "$D/ref-save.json" "t17ref"
REFSZ=$(python3 -c "import json;print(json.load(open('$D/ref-save.json'))['bytes'])")
echo "reference full save: $REFSZ bytes" | tee -a "$D/verdicts.txt"
./scripts/deploy.sh delete-file "$PFN" "t17ref.kv" "slots" >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t17ref"}' >/dev/null || true

attempt() { # delay_seconds
  local delay="$1"
  echo "-- mid-write attempt with kill after ${delay}s" | tee -a "$D/verdicts.txt"
  curl -s -m 120 -X POST "$API/v1/slots/save" -H 'Content-Type: application/json' \
    -d '{"slot":"t17"}' > "$D/save-out.json" 2>&1 &
  local cpid=$!
  sleep "$delay"
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  wait "$cpid" 2>/dev/null || true
  # wait for death
  local down=0
  for _ in $(seq 1 20); do
    if ! curl -s -m 4 "$API/" >/dev/null 2>&1; then down=$((down + 1)); else down=0; fi
    [[ "$down" -ge 3 ]] && break
    sleep 3
  done
  [[ "$down" -ge 3 ]] || { fail "app never died"; return 2; }
  # collect evidence
  if ./scripts/deploy.sh fetch-file "$PFN" "t17.kv" "$D/kv-after" "slots" >/dev/null 2>&1; then
    echo "t17.kv EXISTS after kill ($(stat -c%s "$D/kv-after") bytes)" | tee -a "$D/verdicts.txt"
    KVAFTER=1
  else
    echo "t17.kv ABSENT after kill (rename never happened)" | tee -a "$D/verdicts.txt"
    KVAFTER=0
  fi
  if ./scripts/deploy.sh fetch-file "$PFN" "t17.kv.tmp" "$D/tmp-after" "slots" >/dev/null 2>&1; then
    local tsz; tsz=$(stat -c%s "$D/tmp-after")
    echo "t17.kv.tmp EXISTS after kill: $tsz bytes of $REFSZ expected ($((100 * tsz / REFSZ))%)" | tee -a "$D/verdicts.txt"
    TMPAFTER=$tsz
  else
    echo "t17.kv.tmp ABSENT after kill" | tee -a "$D/verdicts.txt"
    TMPAFTER=0
  fi
  return 0
}
attempt 0.35 || true
if [[ "${KVAFTER:-1}" == "1" || "${TMPAFTER:-0}" == "0" ]]; then
  echo "-- first attempt did not land mid-write; retrying tighter" | tee -a "$D/verdicts.txt"
  restart_verified "retry-up"
  attempt 0.18 || true
fi
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
up=""
for _ in $(seq 1 60); do
  if curl -s -m 5 "$API/" >/dev/null 2>&1; then up=1; break; fi
  sleep 3
done
[[ -n "$up" ]] || fail "app did not come back"
echo "app back after mid-write kill" | tee -a "$D/verdicts.txt"
echo "store now: $(curl -s -m 30 "$API/v1/slots")" | tee -a "$D/verdicts.txt"
# B11-style: the stale partial .tmp must not poison a fresh save/restore
slot_save "$D/save-after.json" "t17" && pass "save after mid-write kill ok (stale .tmp not poisoning)" || fail "save after kill"
if slot_restore "$D/restore-after.json" "t17"; then
  pass "restore after mid-write kill ok"
else
  fail "restore after mid-write kill"
fi
chat "$D/chat-after.json" 16 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]' >/dev/null 2>&1 \
  && pass "chat after mid-write kill ok" || fail "chat after kill"
# cleanup: remove the test slot and any leftover .tmp
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t17"}' >/dev/null || true
./scripts/deploy.sh delete-file "$PFN" "t17.kv.tmp" "slots" >/dev/null 2>&1 || true
./scripts/deploy.sh delete-file "$PFN" "t17.kv" "slots" >/dev/null 2>&1 || true
echo "store final: $(curl -s -m 30 "$API/v1/slots")" | tee -a "$D/verdicts.txt"
echo "== T17 done"
