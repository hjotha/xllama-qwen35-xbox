#!/usr/bin/env bash
# T17b-midwrite-adaptive.sh — plan 009 atomicity: land a REAL mid-write
# failure with an ADAPTIVE kill. Time-based kills (0.35s/0.18s) always
# landed after the rename because a 58 MB write completes in <0.18s; this
# polls the Device Portal directory listing (no download) and kills the
# process the moment the tmp file shows a PARTIAL size (0 < size < full).
# Safe/isolated: only test slots t17/t17ref are touched; the reference slot
# lets each attempt rebuild the state by restore (no re-chat); the leftover
# partial .tmp and both slots are removed at the end.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T17b"
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
tmp_size() { # prints the current size of slots/t17.kv.tmp (0 if absent, -1 if the listing failed)
  ./scripts/deploy.sh list-localstate "$PFN" "slots" 2>/dev/null \
    | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(-1); raise SystemExit
for f in (d.get('Items') or []):
    if (f.get('Name') or '').lower() == 't17.kv.tmp':
        print(int(f.get('FileSize') or 0)); raise SystemExit
print(0)
"
}

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
slot_save "$D/ref-save.json" "t17ref"
REFSZ=$(python3 -c "import json;print(json.load(open('$D/ref-save.json'))['bytes'])")
echo "reference slot: $REFSZ bytes" | tee -a "$D/verdicts.txt"

attempt() { # n
  local n="$1"
  # rebuild state by restore (instant) in a fresh process
  restart_verified "att$n"
  chat "$D/att$n-pre.json" 1 '[{"role":"user","content":"x"}]' || { fail "att$n preload"; return 2; }
  slot_restore "$D/att$n-restore.json" "t17ref" || { fail "att$n restore ref"; return 2; }
  curl -s -m 120 -X POST "$API/v1/slots/save" -H 'Content-Type: application/json' \
    -d '{"slot":"t17"}' > "$D/att$n-out.json" 2>&1 &
  local cpid=$!
  local killed=0
  for _ in $(seq 1 200); do
    local sz; sz=$(tmp_size)
    if [[ "$sz" -gt 0 && "$sz" -lt "$REFSZ" ]]; then
      ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
      killed=1
      echo "attempt $n: killed with t17.kv.tmp at $sz / $REFSZ bytes ($((100 * sz / REFSZ))%)" | tee -a "$D/verdicts.txt"
      break
    fi
    sleep 0.03
  done
  wait "$cpid" 2>/dev/null || true
  if [[ "$killed" == "0" ]]; then
    echo "attempt $n: tmp never observed partial (write too fast for polling); save finished" | tee -a "$D/verdicts.txt"
    return 1
  fi
  local down=0
  for _ in $(seq 1 20); do
    if ! curl -s -m 4 "$API/" >/dev/null 2>&1; then down=$((down + 1)); else down=0; fi
    [[ "$down" -ge 3 ]] && break
    sleep 3
  done
  [[ "$down" -ge 3 ]] || { fail "att$n: app never died"; return 2; }
  if ./scripts/deploy.sh fetch-file "$PFN" "t17.kv" "$D/att$n-kv" "slots" >/dev/null 2>&1; then
    echo "attempt $n: t17.kv EXISTS ($(stat -c%s "$D/att$n-kv")) — rename completed" | tee -a "$D/verdicts.txt"
    return 1
  fi
  echo "attempt $n: t17.kv ABSENT — rename never happened; partial tmp is the only artifact" | tee -a "$D/verdicts.txt"
  return 0
}
LANDED=0
for n in 1 2 3; do
  if attempt "$n"; then LANDED=1; break; fi
done
# restart + verify no poisoning
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
up=""
for _ in $(seq 1 60); do
  if curl -s -m 5 "$API/" >/dev/null 2>&1; then up=1; break; fi
  sleep 3
done
[[ -n "$up" ]] || fail "app did not come back"
chat "$D/warm.json" 8 '[{"role":"user","content":"Reply with the single word ready."}]' >/dev/null 2>&1 || true
echo "store after mid-write: $(curl -s -m 30 "$API/v1/slots")" | tee -a "$D/verdicts.txt"
if [[ "$LANDED" == "1" ]]; then
  pass "MID-WRITE FAILURE LANDED: no partial file became a slot (t17.kv absent)"
  # a later save over the stale partial tmp must succeed and load
  slot_restore "$D/after-restore2.json" "t17ref" || fail "restore ref after kill"
  slot_save "$D/after-save.json" "t17" && pass "save over stale partial tmp ok" || fail "save over stale tmp"
  slot_restore "$D/after-restore.json" "t17" && pass "restore of the re-saved slot ok" || fail "restore re-saved"
  chat "$D/after-chat.json" 16 '[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]' >/dev/null 2>&1 \
    && pass "chat after re-save ok" || fail "chat after re-save"
else
  fail "MID-WRITE NOT LANDED: write completed before any poll could see a partial tmp"
fi
# cleanup (always)
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t17"}' >/dev/null || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t17ref"}' >/dev/null || true
./scripts/deploy.sh delete-file "$PFN" "t17.kv.tmp" "slots" >/dev/null 2>&1 || true
./scripts/deploy.sh delete-file "$PFN" "t17.kv" "slots" >/dev/null 2>&1 || true
echo "store final: $(curl -s -m 30 "$API/v1/slots")" | tee -a "$D/verdicts.txt"
echo "landed=$LANDED" | tee -a "$D/verdicts.txt"
echo "== T17b done"
