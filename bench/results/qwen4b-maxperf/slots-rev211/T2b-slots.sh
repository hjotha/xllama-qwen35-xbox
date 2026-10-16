#!/usr/bin/env bash
# T2b-slots.sh — plan 009 T2 expanded (cancel/isolation/concurrency/corrupt).
# API-level, reversible, MTP on, rev233 Iteration, hook dormant (no flags).
# CANCEL = stop-app mid-generation + recovery. ISOLATION = interleaved slots
# A/B across a restart. CONCURRENCY = parallel save/save/list/chat.
# CORRUPT = missing id, invalid id, truncated file, wrong-version header +
# session usable afterwards. Thresholds: exact texts/ok-flags, verified
# restarts — no epsilon.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T2b"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.233_x64__pj67f1fcj4n14"
API=http://192.168.1.26:11434
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
mkdir -p "$D"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
# shellcheck disable=SC1091
source "$D/../T2-lib.sh"
: > "$D/verdicts.txt"

PA='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
PAA='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'
PB='[{"role":"user","content":"The southern harbor key is DELTA-3. Acknowledge with: noted."}]'
PBB='[{"role":"user","content":"The southern harbor key is DELTA-3. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the key back exactly."}]'
PLONG='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Explain in detail what the northern depot cipher is, who issued it, how it is structured, and repeat it three times with commentary."}]'

# CANCEL: long generation killed mid-flight via stop-app, then recovery
restart_verified "cancel-fresh"
chat "$D/k-pre.json" 1 '[{"role":"user","content":"x"}]'
curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen35-4b-mtp\",\"temperature\":0,\"max_tokens\":256,\"messages\":$PLONG}" \
  > "$D/k-killed.json" &
KPID=$!
sleep 12
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
wait "$KPID" || true
down=0
for _ in $(seq 1 20); do
  if ! curl -s -m 4 "$API/" >/dev/null 2>&1; then down=$((down + 1)); else down=0; fi
  [[ "$down" -ge 3 ]] && break
  sleep 3
done
[[ "$down" -ge 3 ]] && pass "cancel: mid-generation kill observed (death 3x GET fail)" || fail "cancel: app never died"
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
ok=""
for _ in $(seq 1 60); do
  if curl -s -m 5 "$API/" >/dev/null 2>&1; then ok=1; break; fi
  sleep 3
done
[[ -n "$ok" ]] && pass "cancel: app recovered" || fail "cancel: no recovery"
chat "$D/k-after.json" 16 "$PAA" && pass "cancel: fresh chat works after kill" || fail "cancel: broken after kill"
echo "killed-reply-bytes=$(wc -c < "$D/k-killed.json") after-reply=$(content_of "$D/k-after.json")" | tee -a "$D/verdicts.txt"
slot_save "$D/k-save.json" "t2b-k" && pass "cancel: slot save works after kill" || fail "cancel: save broken"
slot_delete_q t2b-k

# ISOLATION: A/B interleaved across a restart, no cross-talk
restart_verified "iso-build"
chat "$D/ia-seed.json" 16 "$PA"
slot_save "$D/ia-save.json" "t2b-ia"
chat "$D/ib-seed.json" 16 "$PB"
slot_save "$D/ib-save.json" "t2b-ib"
restart_verified "iso-newproc"
chat "$D/i-pre.json" 1 '[{"role":"user","content":"x"}]'
slot_restore "$D/ia-restore.json" "t2b-ia" || fail "iso: restore A"
chat "$D/ia-cont.json" 16 "$PAA" || fail "iso: continue A"
slot_restore "$D/ib-restore.json" "t2b-ib" || fail "iso: restore B"
chat "$D/ib-cont.json" 16 "$PBB" || fail "iso: continue B"
{ echo "== ISOLATION A/B"
  echo "A: $(content_of "$D/ia-cont.json") (want BRAVO-7)"
  echo "B: $(content_of "$D/ib-cont.json") (want DELTA-3)"; } | tee -a "$D/verdicts.txt"
[[ "$(content_of "$D/ia-cont.json")" == "BRAVO-7" ]] && pass "isolation A intact" || fail "isolation A crosstalk"
[[ "$(content_of "$D/ib-cont.json")" == "DELTA-3" ]] && pass "isolation B intact" || fail "isolation B crosstalk"
slot_delete_q t2b-ia; slot_delete_q t2b-ib

# CONCURRENCY: parallel save/save/list/chat — hub mutex serializes, all must succeed
chat "$D/n-seed.json" 16 "$PA"
curl -s -m 120 "$API/v1/slots/save" -H 'Content-Type: application/json' -d '{"slot":"t2b-n1"}' > "$D/n-s1.json" &
P1=$!
curl -s -m 120 "$API/v1/slots/save" -H 'Content-Type: application/json' -d '{"slot":"t2b-n2"}' > "$D/n-s2.json" &
P2=$!
curl -s -m 60 "$API/v1/slots" > "$D/n-list.json" &
P3=$!
curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen35-4b-mtp\",\"temperature\":0,\"max_tokens\":8,\"messages\":$PAA}" > "$D/n-chat.json" &
P4=$!
wait $P1 $P2 $P3 $P4 || true
okc=0
grep -q '"ok":true' "$D/n-s1.json" && okc=$((okc + 1))
grep -q '"ok":true' "$D/n-s2.json" && okc=$((okc + 1))
grep -q '"ok":true' "$D/n-list.json" && okc=$((okc + 1))
grep -q '"choices"' "$D/n-chat.json" && okc=$((okc + 1))
[[ "$okc" == "4" ]] && pass "concurrency: 4/4 parallel ops ok" || fail "concurrency: $okc/4 ok"
slot_restore "$D/n-r1.json" "t2b-n1" && pass "concurrency: slot n1 intact" || fail "concurrency: n1 broken"
slot_restore "$D/n-r2.json" "t2b-n2" && pass "concurrency: slot n2 intact" || fail "concurrency: n2 broken"
slot_delete_q t2b-n1; slot_delete_q t2b-n2

# CORRUPT / incompatible in a new process + session usable afterwards
restart_verified "corrupt-newproc"
chat "$D/g-pre.json" 1 '[{"role":"user","content":"x"}]'
if slot_restore "$D/g-missing.json" "t2b-nope"; then fail "corrupt: missing id accepted"; else pass "corrupt: missing id refused"; fi
if slot_restore "$D/g-badid.json" "t2b/../evil"; then fail "corrupt: invalid id accepted"; else pass "corrupt: invalid id refused"; fi
chat "$D/g-seed.json" 16 "$PA"
slot_save "$D/g-save.json" "t2b-g"
./scripts/deploy.sh fetch-file "$PFN" "t2b-g.kv" "$D/real.kv" "slots" || fail "corrupt: fetch real slot file"
head -c 100 "$D/real.kv" > "$D/trunc.kv"
./scripts/deploy.sh upload-file "$D/trunc.kv" "$PFN" "slots" "t2b-trunc.kv" >/dev/null
if slot_restore "$D/g-trunc.json" "t2b-trunc"; then fail "corrupt: truncated accepted"; else pass "corrupt: truncated refused"; fi
python3 - "$D/real.kv" "$D/v9.kv" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
nl = data.index(b"\n")
hdr = data[:nl].decode("utf-8", "replace").split("\t")
hdr[3] = "v9 edited-fp"
open(sys.argv[2], "wb").write(("\t".join(hdr) + "\n").encode() + data[nl + 1:])
print("v9 header written")
PY
./scripts/deploy.sh upload-file "$D/v9.kv" "$PFN" "slots" "t2b-v9.kv" >/dev/null
if slot_restore "$D/g-v9.json" "t2b-v9"; then fail "corrupt: wrong-version accepted"; else pass "corrupt: wrong-version refused"; fi
chat "$D/g-after.json" 16 "$PAA" && [[ "$(content_of "$D/g-after.json")" == "BRAVO-7" ]] \
  && pass "corrupt: session usable after refusals" || fail "corrupt: session broken"
slot_delete_q t2b-g; slot_delete_q t2b-trunc; slot_delete_q t2b-v9
echo "== T2b done"
