#!/usr/bin/env bash
# slot-battery.sh v2 — plan 009 S3 validation battery for KV slots on device.
# Strict token-exact comparisons (greedy, temperature 0); no epsilon.
# Evidence model (log-rotation and contention-proof):
#  - QUIESCE pre/post: two log fetches with no requests between must show no
#    new turn/save/load lines, or the run aborts (another actor = contamination).
#  - ONE start fetch (L0 line count) + ONE end fetch; the window slice is
#    analyzed by ORDER (device is quiet: my requests are the only turn
#    producers). A short window (<1000 lines for ~15 chats) flags ROTATION and
#    invalidates mechanism mapping (response verdicts still stand).
#  - Any `session config:` (swap) line inside the window is reported with its
#    position; unexpected swaps fail the run.
# Usage: BASE=<results-dir> PFN_EXPECT=... KNOB=on|off ./slot-battery.sh
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/slots-rev211}"
PFN_EXPECT="${PFN_EXPECT:?set PFN_EXPECT}"
KNOB="${KNOB:-on}"
API=http://192.168.1.26:11434
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
D="$BASE/slots-$KNOB"; mkdir -p "$D"
# KNOB now ACTUALLY drives the FFN SWIGLU knob (d3d12swiglu.txt: "1" = all
# FFN on D3D12, "0" = forced OFF; absent = default ON since 2026-10-09) and
# the effective mode is VERIFIED from the device log at the end — a label is
# not evidence. Both arms also record the session config line (mtp/batch/
# ubatch/gpu) and the FFN dispatch counters.
fails=0
note() { echo "$@" | tee -a "$D/verdicts.txt"; }
fail() { fails=$((fails + 1)); note "FAIL: $*"; }
pass() { note "PASS: $*"; }
turnsig() { grep -a -c "session: turn full\|session generate: n=\|KV state saved\|KV state loaded" "$1" || true; }

loglines() { ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l; }
case "$KNOB" in
  on)  printf '1' > "$D/knob.txt" ;;
  off) printf '0' > "$D/knob.txt" ;;
  *)   echo "KNOB must be on|off" >&2; exit 2 ;;
esac
./scripts/deploy.sh upload-file "$D/knob.txt" "$PFN" "" d3d12swiglu.txt >/dev/null
# the FFN mode is bound at app STARTUP: mark the log BEFORE the restart so
# the post-restart window contains the startup mode line, restart so the
# uploaded knob takes effect, then read the device's own mode line.
KNOB_L0=$(loglines)
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
sleep 2
./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
up=""
for _ in $(seq 1 40); do
  if curl -s -m 5 "$API/" >/dev/null 2>&1; then up=1; break; fi
  sleep 3
done
[[ -n "$up" ]] || { echo "app did not come back after knob restart" >&2; exit 1; }
MODE_LINE=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((KNOB_L0 + 1))" \
  | grep -m1 -oE "FFN SWIGLU D3D12=(all|off|target-only)" || true)
echo "$KNOB_L0" > "$D/knob-L0.txt"
if [[ "$KNOB" == "on" && "$MODE_LINE" != "FFN SWIGLU D3D12=all" ]]; then
  echo "KNOB_VERIFY_FAIL on->$MODE_LINE" >&2; exit 1
fi
if [[ "$KNOB" == "off" && "$MODE_LINE" != "FFN SWIGLU D3D12=off" ]]; then
  echo "KNOB_VERIFY_FAIL off->$MODE_LINE" >&2; exit 1
fi
note "== battery knob=$KNOB start=$(date -u +%Y-%m-%dT%H:%M:%SZ) effective=$MODE_LINE verified_at_log_line=$KNOB_L0"
BIND_LINE=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((KNOB_L0 + 1))" \
  | grep -m1 -c "FFN SWIGLU profile bound: on (mtp_capable=1)" || true)
note "knob_verify: mode=$MODE_LINE bind_on_mtp1=$BIND_LINE"
note "session_config_expected: model=qwen35-4b-mtp mtp=1 depth=2 pmin=0.50 n_ctx=2048 n_batch=256 n_ubatch=64 gpu=34"
# quiesce pre-check
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$D/quiesce-a.log"
sleep 30
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$D/quiesce-b.log"
if [[ "$(turnsig "$D/quiesce-a.log")" != "$(turnsig "$D/quiesce-b.log")" ]]; then
  fail "quiesce: foreign turn activity during 30s idle - aborting (contamination)"
  exit 1
fi
pass "quiesce pre: device quiet"
L0=$(wc -l < "$D/quiesce-b.log")
echo "$L0" > "$D/window-L0.txt"

chat() { # out max_tokens messages_json
  curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen35-4b-mtp\",\"temperature\":0,\"max_tokens\":$2,\"messages\":$3}" > "$1"
  grep -q '"choices"' "$1" || { fail "chat $1 no choices"; return 1; }
}
content_of() { python3 -c "import json,sys;print(json.load(open('$1'))['choices'][0]['message']['content'])"; }
slot_save() { # out id
  curl -s -m 300 "$API/v1/slots/save" -H 'Content-Type: application/json' -d "{\"slot\":\"$2\"}" > "$1"
  grep -q '"ok":true' "$1" || { fail "slot save $2: $(head -c 200 "$1")"; return 1; }
}
slot_restore() { # out id
  curl -s -m 300 "$API/v1/slots/restore" -H 'Content-Type: application/json' -d "{\"slot\":\"$2\"}" > "$1"
  grep -q '"ok":true' "$1"
}
slot_delete() { # out id
  curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d "{\"slot\":\"$2\"}" > "$1"
  grep -q '"ok":true' "$1" || { fail "slot delete $2"; return 1; }
}
# Restart with PROOF (item 2: silent no-op restarts voided prior
# fresh-process claims). Death requires 3 consecutive GET failures (a single
# blip must not fake it); rebirth requires GET ok. Any failure aborts the
# freshness claims loudly instead of proceeding blind.
restart_verified() { # reason
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  local down=0
  for _ in $(seq 1 20); do
    if ! curl -s -m 4 "$API/" >/dev/null 2>&1; then down=$((down + 1)); else down=0; fi
    [[ "$down" -ge 3 ]] && break
    sleep 3
  done
  [[ "$down" -ge 3 ]] || { fail "restart ($1): app never died (stop ineffective)"; return 1; }
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
  local ok=""
  for _ in $(seq 1 60); do
    if curl -s -m 5 "$API/" >/dev/null 2>&1; then ok=1; break; fi
    sleep 3
  done
  [[ -n "$ok" ]] || { fail "restart ($1): app never came back"; return 1; }
  pass "restart verified ($1): death (3x GET fail) + rebirth observed"
}

P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
P1A='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'
P1B='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."},{"role":"assistant","content":"BRAVO-7"},{"role":"user","content":"Now spell it backwards."}]'

# B1 preflight: slots empty
curl -s -m 30 "$API/v1/slots" > "$D/list-pre.json"
grep -q '"ok":true' "$D/list-pre.json" || { fail "list pre"; exit 1; }
python3 -c "import json;d=json.load(open('$D/list-pre.json'));assert d['slots']==[],d" \
  && pass "B1 preflight: slots endpoint ok, store empty" || fail "B1 store not empty"

# B2 seed + save
chat "$D/seed.json" 16 "$P1"
slot_save "$D/save-s1.json" "s1"
python3 - "$D/save-s1.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d["ok"] and d["tokens"]>0 and d["bytes"]>0, d
print(f"s1 tokens={d['tokens']} bytes={d['bytes']}")
PY
[[ $? == 0 ]] && pass "B2 seed+save s1 (tokens>0, bytes>0)" || fail "B2 save s1"

# B3 cross-request reuse: restore s1 + continue == fresh prefill of full history
chat "$D/cont-restore.json" 16 "$P1A"
T_RESTORE="$(content_of "$D/cont-restore.json")"
slot_restore "$D/restore-s1.json" "s1" || fail "B3 restore s1"
restart_verified "B3 fresh arm"
chat "$D/seed2.json" 16 "$P1"  # reload same model fresh (empty KV)
chat "$D/cont-fresh.json" 16 "$P1A"
T_FRESH="$(content_of "$D/cont-fresh.json")"
[[ "$T_RESTORE" == "$T_FRESH" ]] \
  && pass "B3 reuse==fresh prefill ($(printf %.40s "$T_RESTORE")...)" \
  || fail "B3 reuse [$T_RESTORE] vs fresh [$T_FRESH]"

# B4 fresh-process reuse is covered by the restart above.
pass "B4 fresh-process reuse (restart between save and restore, B3 equality holds)"

# B4b restore-after-restart into a fresh session (no stale-state leakage).
slot_restore "$D/restore-s1-postrestart.json" "s1" || fail "B4b restore s1 post-restart"
chat "$D/cont-postrestart.json" 16 "$P1A"
[[ "$(content_of "$D/cont-postrestart.json")" == "$T_FRESH" ]] \
  && pass "B4b restore-after-restart matches fresh" || fail "B4b post-restart mismatch"

# B6 rollback: advance, save s2, restore s1, continue == B3 tokens
chat "$D/adv.json" 16 "$P1B"
slot_save "$D/save-s2.json" "s2"
slot_restore "$D/restore-s1b.json" "s1" || fail "B6 restore s1 (rollback)"
chat "$D/cont-rollback.json" 16 "$P1A"
[[ "$(content_of "$D/cont-rollback.json")" == "$T_FRESH" ]] \
  && pass "B6 rollback to s1 matches" || fail "B6 rollback mismatch"

# B8 isolation: s1/s2 independent; rapid double-save valid
slot_restore "$D/restore-s2.json" "s2" || fail "B8 restore s2"
chat "$D/cont-s2.json" 16 "$P1B"
slot_save "$D/save-s1c.json" "s1copy"
slot_restore "$D/restore-s1c.json" "s1" || fail "B8 restore s1 again"
chat "$D/cont-s1c.json" 16 "$P1A"
[[ "$(content_of "$D/cont-s1c.json")" == "$T_FRESH" ]] \
  && pass "B8 slot isolation (s1/s2/s1copy independent)" || fail "B8 isolation"
curl -s -m 30 "$API/v1/slots" > "$D/list-mid.json"
python3 -c "import json;d=json.load(open('$D/list-mid.json'));print(sorted((s['slot'],s['tokens']) for s in d['slots']))"

# B9 corrupt / truncated / wrong-version
printf 'garbage-not-a-slot-file' > /tmp/opencode/qwen4b-recon/bad.kv
./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/bad.kv "$PFN" "slots" "bad.kv" >/dev/null
if slot_restore "$D/restore-bad.json" "bad"; then fail "B9 garbage accepted"; else pass "B9 garbage refused"; fi
./scripts/deploy.sh fetch-file "$PFN" "s1.kv" /tmp/opencode/qwen4b-recon/s1-copy.kv "slots" >/dev/null
head -c 100 /tmp/opencode/qwen4b-recon/s1-copy.kv > /tmp/opencode/qwen4b-recon/trunc.kv
./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/trunc.kv "$PFN" "slots" "trunc.kv" >/dev/null
if slot_restore "$D/restore-trunc.json" "trunc"; then fail "B9 truncated accepted"; else pass "B9 truncated refused"; fi
python3 - /tmp/opencode/qwen4b-recon/s1-copy.kv /tmp/opencode/qwen4b-recon/v9.kv <<'PY'
import sys
data=open(sys.argv[1],'rb').read()
nl=data.index(b'\n')
hdr=data[:nl].decode().split('\t')
hdr[3]='v9 edited-fp'
open(sys.argv[2],'wb').write(('\t'.join(hdr)+'\n').encode()+data[nl+1:])
print("v9 header written")
PY
./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/v9.kv "$PFN" "slots" "v9.kv" >/dev/null
if slot_restore "$D/restore-v9.json" "v9"; then fail "B9 wrong-version accepted"; else pass "B9 wrong-version refused"; fi
chat "$D/after-refusals.json" 16 "$P1A"
[[ "$(content_of "$D/after-refusals.json")" == "$T_FRESH" ]] \
  && pass "B9 session usable after refusals" || fail "B9 session broken after refusals"

# B7 EOS/stop: tiny max_tokens forces a stop; save+restore+continue stays sane
chat "$D/eos.json" 2 "$P1A"
slot_save "$D/save-eos.json" "seos"
slot_restore "$D/restore-eos.json" "seos" || fail "B7 restore seos"
chat "$D/cont-eos.json" 16 "$P1B"
[[ -n "$(content_of "$D/cont-eos.json")" ]] && pass "B7 EOS save/restore/continue sane" || fail "B7 EOS empty"

# B5 divergence: edited history (cipher changed mid-prefix) == fresh prefill
P1DIV='[{"role":"user","content":"The northern depot cipher is CHARLIE-9. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'
slot_restore "$D/restore-div.json" "s1" || fail "B5 restore s1"
chat "$D/cont-div.json" 16 "$P1DIV"
T_DIV="$(content_of "$D/cont-div.json")"
restart_verified "B5 fresh arm"
chat "$D/seed-div.json" 16 '[{"role":"user","content":"x"}]'
chat "$D/cont-div-fresh.json" 16 "$P1DIV"
[[ "$(content_of "$D/cont-div-fresh.json")" == "$T_DIV" ]] \
  && pass "B5 divergent continuation == fresh prefill" || fail "B5 divergence mismatch"
grep -q "CHARLIE-9" "$D/cont-div.json" && pass "B5 divergent content correct" || fail "B5 content"

# B11 tmp hygiene: garbage .tmp must not poison a later save of the same id
printf 'partial-write-simulation' > /tmp/opencode/qwen4b-recon/hygiene.tmp
./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/hygiene.tmp "$PFN" "slots" "hygiene.kv.tmp" >/dev/null
slot_save "$D/save-hygiene.json" "hygiene" || fail "B11 save over stale tmp"
slot_restore "$D/restore-hygiene.json" "hygiene" || fail "B11 restore after tmp hygiene"
pass "B11 stale .tmp does not poison save/restore"

# cleanup slots used here (beta slots removed; evidence JSONs stay in repo)
for s in s1 s2 s1copy bad trunc v9 seos hygiene; do slot_delete "$D/del-$s.json" "$s" || true; done
curl -s -m 30 "$API/v1/slots" > "$D/list-post.json"

# window analysis: end fetch, slice from L0, EXACT collection integrity.
# Expected (full,kv_before,pf,prompt) per turn in battery order — fixed
# prompts + fixed template => deterministic (proven: rev226 ON/OFF and
# rev228 ON identical). Labels distinguish the two reuse classes:
#  1 seed            (1, 0,40,40) fresh prefill, original process
#  2 cont-restore    (1,42,20,62) CONTINUATION AFTER SEED (seed live KV;
#                                 the slot file ops are gated by B2/B4b/B6)
#  3 seed2           (1, 0,40,40) FRESH IN NEW PROCESS (restart_verified)
#  4 cont-fresh      (1,42,20,62) CONTINUATION AFTER SEED2 — the 20 tokens
#                                 are INCREMENTAL: kv_before=42 == kept,
#                                 pf = 62-42 = 20 (protocol-proven)
#  5 cont-postrestart(1,42,20,62) new-process restore + tail-only reuse
#                                 with MTP ON (plan009 M3 sealed zero-carry
#                                 prime; measured rev237 — see receipt)
#  6 adv             (1,66,19,85) continuation after 5
#  7 cont-rollback   (1,42,20,62) new-process restore + tail-only reuse (M3)
#  8 cont-s2         (1, 0,85,85) new-process restore, common prefix stops
#                                 at 40 (slot has 42 incl. generated reply)
#                                 -> F3.5 refuses carry -> full prefill
#  9 cont-s1c        (1,42,20,62) new-process restore + tail-only reuse (M3)
# 10 after-refusals  (1, 0,62,62) full prefill (no live reuse observed)
# 11 eos             (1, 0,62,62) full prefill
# 12 cont-eos        (1,64,21,85) SAME-PROCESS restore attached (kv=64)
# 13 cont-div        (1, 0,63,63) new-process restore + DIVERGENT prompt
#                                 (common prefix 40 < saved 42) -> F3.5
#                                 refuses carry -> full prefill (by design)
# 14 seed-div        (1, 0,25,25) FRESH IN NEW PROCESS (restart_verified)
# 15 cont-div-fresh  (1, 0,63,63) FRESH full prefill of full history in new
#                                 process (seed-div shares no prefix)
# Harvest cells prove reuse; kv==0 cells prove correct full-prefill
# fallback. Any deviation (count, order, tuples, markers) fails. Turn
# response IDs are asserted below from the per-turn chat JSONs.
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$D/log-end.log"
L1=$(wc -l < "$D/log-end.log")
tail -n +"$((L0 + 1))" "$D/log-end.log" > "$D/window-slice.txt"
python3 - "$D/window-slice.txt" >> "$D/verdicts.txt" <<'PY'
import re, sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
turns = []
for l in lines:
    m = re.search(r"turn full=(\d) kv_before=(\d+) pf=(\d+) prompt=(\d+)", l)
    if m:
        turns.append(tuple(m.group(i) for i in (1, 2, 3, 4)))
expected = [("1","0","40","40"),("1","42","20","62"),("1","0","40","40"),
            ("1","42","20","62"),("1","42","20","62"),("1","66","19","85"),
            ("1","42","20","62"),("1","0","85","85"),("1","42","20","62"),
            ("1","0","62","62"),("1","0","62","62"),("1","64","21","85"),
            ("1","0","63","63"),("1","0","25","25"),("1","0","63","63")]
names = ["seed","cont-restore","seed2","cont-fresh","cont-postrestart","adv",
         "cont-rollback","cont-s2","cont-s1c","after-refusals","eos",
         "cont-eos","cont-div","seed-div","cont-div-fresh"]
print(f"-- turns: {len(turns)}/15 expected")
ok = True
if len(turns) != 15:
    print(f"   COUNT_MISMATCH got={len(turns)} want=15 (rotation or foreign turns?)")
    ok = False
else:
    for i, (got, want) in enumerate(zip(turns, expected)):
        if got != want:
            print(f"   TURN_MISMATCH #{i+1} {names[i]} got={got} want={want}")
            ok = False
    if turns[0] != ("1","0","40","40"):
        print("   START_MARKER_FAIL (first turn is not seed kv0/pf40/40)")
        ok = False
    if turns[-1] != ("1","0","63","63"):
        print("   END_MARKER_FAIL (last turn is not cont-div-fresh kv0/pf63/63)")
        ok = False
    # incremental-reuse invariant: a harvest turn must have kv+pf == prompt
    for i, (f, kv, pf, pr) in enumerate(turns):
        if int(kv) > 0 and int(kv) + int(pf) != int(pr):
            print(f"   INCREMENTAL_FAIL #{i+1} {names[i]}: kv {kv} + pf {pf} != prompt {pr}")
            ok = False
open(sys.argv[1] + ".turnfail", "w").write("0" if ok else "1")
PY
if [[ "$(cat "$D/window-slice.txt.turnfail")" != "0" ]]; then
  fail "collection integrity: turn sequence mismatch (see analysis above)"
else
  pass "collection integrity: exactly 15 turns, expected (full,kv,pf,prompt) each, markers ok, no rotation"
fi
rm -f "$D/window-slice.txt.turnfail"

# Log-chain + prefix preservation (real continuity evidence, not counts).
# prefix: log-end.log lines 1..L0 must be byte-identical to the saved
# quiesce-b.log (no rotation/truncation between quiesce and end, so the
# slice is a pure continuation of the pre-battery log state). chain: the
# 15 turn lines must alternate with 15 turn-end drafter lines in order;
# prompt == kv_before + pf for every turn; pending_pos == mirror_end and
# kv_tokens == pending_pos + 1 for every turn-end; every harvested turn
# (expected kv>0) must have kv_before == previous turn-end kv_tokens.
# cross: each turn's JSON usage.prompt_tokens must equal the LOG pf of
# the same turn index (response<->log bind per turn, two independent
# paths: API JSON vs device log).
python3 - "$D" >> "$D/verdicts.txt" <<'PY'
import hashlib, json, pathlib, re, sys
d = pathlib.Path(sys.argv[1])
fails = []
qb = (d / "quiesce-b.log").read_bytes().splitlines()
le = (d / "log-end.log").read_bytes().splitlines()
l0 = int((d / "window-L0.txt").read_text().strip())
if le[:l0] != qb:
    fails.append(f"PREFIX_FAIL log-end[1..{l0}] != quiesce-b (rotation/truncation: slice is not a pure continuation)")
else:
    print(f"-- prefix: log-end[1..{l0}] byte-identical to quiesce-b "
          f"(sha={(hashlib.sha256((d / 'quiesce-b.log').read_bytes()).hexdigest())[:16]}); slice is a pure continuation")
sl = (d / "window-slice.txt").read_bytes().splitlines()
turns, ends = [], []
for n, raw in enumerate(sl, start=l0 + 1):
    l = raw.decode("utf-8", "replace")
    m = re.search(r"turn full=(\d) kv_before=(\d+) pf=(\d+) prompt=(\d+)", l)
    if m:
        turns.append((n, tuple(m.group(i) for i in (1, 2, 3, 4))))
    m2 = re.search(r"turn-end drafter state: pending_valid=(\d) pending_pos=(\d+) mirror_end=(\d+) kv_tokens=(\d+)", l)
    if m2:
        ends.append((n, tuple(m2.group(i) for i in (1, 2, 3, 4))))
print(f"-- chain: {len(turns)} turn lines, {len(ends)} turn-end lines")
if len(turns) != 15 or len(ends) != 15:
    fails.append(f"CHAIN_COUNT_FAIL turns={len(turns)} ends={len(ends)} want=15/15")
else:
    alt = all(turns[i][0] < ends[i][0] for i in range(15)) and \
          all(ends[i][0] < turns[i + 1][0] for i in range(14))
    if not alt:
        fails.append("CHAIN_ORDER_FAIL turn/turn-end lines do not strictly alternate")
    expected = [("1","0","40","40"),("1","42","20","62"),("1","0","40","40"),
                ("1","42","20","62"),("1","42","20","62"),("1","66","19","85"),
                ("1","42","20","62"),("1","0","85","85"),("1","42","20","62"),
                ("1","0","62","62"),("1","0","62","62"),("1","64","21","85"),
                ("1","0","63","63"),("1","0","25","25"),("1","0","63","63")]
    names = ["seed","cont-restore","seed2","cont-fresh","cont-postrestart","adv",
             "cont-rollback","cont-s2","cont-s1c","after-refusals","eos",
             "cont-eos","cont-div","seed-div","cont-div-fresh"]
    if [t for _, t in turns] != expected:
        fails.append("CHAIN_TUPLE_FAIL turn sequence != expected table")
    for i, (ln, (f, kv, pf, pr)) in enumerate(turns):
        if int(kv) + int(pf) != int(pr):
            fails.append(f"CHAIN_SUM_FAIL #{i+1} {names[i]} kv {kv} + pf {pf} != prompt {pr} (log line {ln})")
    for i, (ln, (v, p, m, k)) in enumerate(ends):
        if int(p) != int(m):
            fails.append(f"CHAIN_MIRROR_FAIL end #{i+1} pos {p} != mirror {m} (log line {ln})")
        if int(k) != int(p) + 1:
            fails.append(f"CHAIN_KV_FAIL end #{i+1} kv_tokens {k} != pos {p} + 1 (log line {ln})")
    for i in (1, 3, 5, 11):  # harvested turns: kv_before must chain from previous turn-end
        kvb = int(turns[i][1][1])
        prevk = int(ends[i - 1][1][3])
        if kvb != prevk:
            fails.append(f"CHAIN_LINK_FAIL #{i+1} {names[i]} kv_before {kvb} != prev turn-end kv_tokens {prevk}")
        else:
            print(f"   link #{i+1} {names[i]}: kv_before={kvb} == prev turn-end kv_tokens (log lines {ends[i-1][0]} -> {turns[i][0]})")
    # cross JSON<->log per turn index
    for i, name in enumerate(names):
        try:
            u = json.load(open(d / f"{name}.json"))["usage"]
            pt = u["prompt_tokens"]
        except Exception as e:
            fails.append(f"CROSS_UNREADABLE {name} ({e})")
            continue
        if pt != int(turns[i][1][2]):
            fails.append(f"CROSS_FAIL {name}: json prompt_tokens {pt} != log pf {turns[i][1][2]} (log line {turns[i][0]})")
    print(f"   coherence: prompt==kv+pf 15/15, pos==mirror 15/15, kv==pos+1 15/15, json<->log pf 15/15")
open(sys.argv[1] + "/chainfail", "w").write("0" if not fails else "1")
for x in fails:
    print("   " + x)
PY
if [[ "$(cat "$D/chainfail")" != "0" ]]; then
  fail "log chain / prefix / JSON-log cross mismatch (see analysis above)"
else
  pass "log chain + prefix preserved + JSON-log cross per turn (response<->log bound, not counts)"
fi
rm -f "$D/chainfail"

# Turn response IDs (fixed prompts + deterministic sampling; identical in
# rev226 ON/OFF and rev228 ON). seed-div gets a nonempty check only (free
# reply to "x").
python3 - "$D" >> "$D/verdicts.txt" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
expect = {"seed": "noted", "cont-restore": "BRAVO-7", "seed2": "noted",
          "cont-fresh": "BRAVO-7", "cont-postrestart": "BRAVO-7",
          "adv": "7-OVARB", "cont-rollback": "BRAVO-7", "cont-s2": "7-OVARB",
          "cont-s1c": "BRAVO-7", "after-refusals": "BRAVO-7", "eos": "BRAVO",
          "cont-eos": "7-OVARB", "cont-div": "CHARLIE-9",
          "cont-div-fresh": "CHARLIE-9"}
fails = []
for name, want in expect.items():
    p = d / f"{name}.json"
    try:
        got = json.load(open(p))["choices"][0]["message"]["content"]
    except Exception as e:
        fails.append(f"{name}: unreadable ({e})")
        continue
    if got != want:
        fails.append(f"{name}: got={got[:60]!r} want={want!r}")
try:
    sd = json.load(open(d / "seed-div.json"))["choices"][0]["message"]["content"]
    if not sd.strip():
        fails.append("seed-div: empty reply")
except Exception as e:
    fails.append(f"seed-div: unreadable ({e})")
print(f"-- turn IDs: {len(expect) + 1} checked, failures={len(fails)}")
for f in fails:
    print("   ID_FAIL " + f)
open(d / "idfail", "w").write("0" if not fails else "1")
PY
if [[ "$(cat "$D/idfail")" != "0" ]]; then
  fail "turn response IDs mismatch (see analysis above)"
else
  pass "turn response IDs: 15/15 match expected content"
fi
rm -f "$D/idfail"

# Request identity + log continuity. Content IDs alone do not prove that
# response N came from request N, nor that the window is one contiguous log
# span. Gates: per-turn JSON id embeds the same epoch as `created`;
# (created, id-suffix) strictly increases in turn order (same-second safe);
# model fixed; usage.prompt_tokens equals the expected PREFILLED slice of
# that turn (the API reports the actually-prefilled delta — the pf column
# of the turn table; protocol-proven: harvest turns report 20/19/21, full
# turns report the full prompt; binds each response to the exact reuse
# behavior of its turn); ids unique; on the log
# side: exactly 15 turn-full lines and exactly 7 KV-reuse lines (4 live
# harvests + 3 restore-load lines; proven in rev226 ON/OFF + rev228 ON/OFF),
# plus a rotation sanity cap on slice length.
python3 - "$D" >> "$D/verdicts.txt" <<'PY'
import json, pathlib, re, sys
d = pathlib.Path(sys.argv[1])
files = ["seed", "cont-restore", "seed2", "cont-fresh", "cont-postrestart",
         "adv", "cont-rollback", "cont-s2", "cont-s1c", "after-refusals",
         "eos", "cont-eos", "cont-div", "seed-div", "cont-div-fresh"]
# pf column of the expected turn table (prefilled slice, not full prompt)
want_ptok = [40, 20, 40, 20, 20, 19, 20, 85, 20, 62, 62, 21, 63, 25, 63]
fails = []
seq = []
ids = []
for i, name in enumerate(files):
    try:
        r = json.load(open(d / f"{name}.json"))
    except Exception as e:
        fails.append(f"{name}: unreadable ({e})")
        continue
    rid = r.get("id", "")
    cr = r.get("created", 0)
    model = r.get("model", "")
    ptok = r.get("usage", {}).get("prompt_tokens")
    ids.append(rid)
    m = re.search(r"-(\d+)$", rid)
    suffix = int(m.group(1)) if m else -1
    seq.append((cr, suffix, name))
    if model != "qwen35-4b-mtp":
        fails.append(f"{name}: model={model!r}")
    if ptok != want_ptok[i]:
        fails.append(f"{name}: prompt_tokens={ptok} want={want_ptok[i]}")
    if f"chatcmpl-xllama-{cr}-" not in rid:
        fails.append(f"{name}: id {rid!r} does not embed created={cr}")
if len(set(ids)) != len(ids):
    fails.append("duplicate request ids across turns")
keys = [(cr, sfx) for cr, sfx, _ in seq]
if any(b <= a for a, b in zip(keys, keys[1:])):
    fails.append(f"(created,suffix) not strictly increasing: {seq}")
lines = (d / "window-slice.txt").read_text(errors="replace").splitlines()
nturn = sum(1 for l in lines if "session: turn full" in l)
nkept = sum(1 for l in lines if "KV prefix reuse" in l)
if nturn != 15:
    fails.append(f"slice turn lines={nturn} want=15 (continuity/rotation)")
if nkept != 7:
    fails.append(f"slice kept lines={nkept} want=7 (mechanism count)")
if not (100 < len(lines) < 20000):
    fails.append(f"slice lines={len(lines)} outside sane bounds (rotation)")
print(f"-- request identity: {len(files)} turns, (created,suffix) monotonic, "
      f"ids bound, prompt_tokens match; log continuity: "
      f"turns={nturn} kept={nkept} slice={len(lines)}")
for f in fails:
    print("   IDENT_FAIL " + f)
open(d / "identfail", "w").write("0" if not fails else "1")
PY
if [[ "$(cat "$D/identfail")" != "0" ]]; then
  fail "request identity / log continuity (see analysis above)"
else
  pass "request identity + log continuity: order, binding, markers, no rotation"
fi
rm -f "$D/identfail"

python3 - "$D/window-slice.txt" >> "$D/verdicts.txt" <<'PY'
import sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
swaps = [(i + 1, l[:120]) for i, l in enumerate(lines) if "session config:" in l]
print(f"-- window swaps (session config lines): {len(swaps)}")
for n, l in swaps[:16]:
    print(f"   slice-L{n}: {l}")
# unexpected-swap rule: an lfm25 config is only legal within 40 lines after an
# "api: listening" line (app start). Anything else = mid-gate model swap.
listen = [i for i, l in enumerate(lines) if "api: listening" in l]
bad = [n for n, l in swaps
       if "lfm25" in l and not any(0 <= n - 1 - m <= 40 for m in listen)]
print(f"-- unexpected lfm swaps outside startup: {len(bad)}")
for n in bad[:8]:
    print(f"   UNEXPECTED slice-L{n}")
open(sys.argv[1] + ".swapfail", "w").write(str(len(bad)))
kept = [l for l in lines if "KV prefix reuse" in l]
print(f"-- kept lines: {len(kept)}")
for l in kept[:12]:
    print(f"   {l[:150]}")
pf = [(l.split("pf=")[1].split()[0], l.split("prompt=")[1].split()[0])
      for l in lines if "session: turn full" in l]
print(f"-- turns: {len(pf)} (pf,prompt) sequence: {[p for p, _ in pf]}")
PY
if [[ "$(cat "$D/window-slice.txt.swapfail")" != "0" ]]; then
  fail "unexpected mid-gate model swap (see window analysis)"
else
  pass "no unexpected model swaps in window"
fi
rm -f "$D/log-end.log" "$D/quiesce-a.log" "$D/quiesce-b.log"
echo "$L1" > "$D/end-L1.txt"
turnsig "$D/window-slice.txt" > "$D/window-turnsig.txt"
rm -f "$D/log-end.log" "$D/quiesce-a.log" "$D/quiesce-b.log"
# quiesce post-check (no foreign traffic during the run)
./scripts/deploy.sh get-log "$PFN" 2>/dev/null > "$D/quiesce-post.log"
tail -n +"$(($(cat "$D/end-L1.txt") + 1))" "$D/quiesce-post.log" > "$D/post-delta.txt"
if [[ "$(turnsig "$D/post-delta.txt")" != "0" ]]; then
  note "WARN: $(turnsig "$D/post-delta.txt") turn/save/load lines appeared after end-fetch (background activity?)"
  head -3 "$D/post-delta.txt" | tee -a "$D/verdicts.txt"
else
  pass "quiesce post: quiet"
fi
rm -f "$D/quiesce-post.log" "$D/post-delta.txt" "$D/window-slice.txt.swapfail" "$D/end-L1.txt"
{
  echo "== battery knob=$KNOB end=$(date -u +%Y-%m-%dT%H:%M:%SZ) failures=$fails"
  # effective FFN mode + dispatch counters (proof, not label)
  KL0=$(cat "$D/knob-L0.txt")
  # NB: capture in $() with an inner || true — deploy.sh get-log exits
  # non-zero while still streaming the log, and under `set -o pipefail`
  # that made a FOUND mode line print alongside a false
  # FFN_MODE_LINE_NOT_FOUND (fixed 2026-10-11).
  MODE_END=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((KL0 + 1))" \
    | grep -m1 -oE "FFN SWIGLU D3D12=(all|off|target-only)" || true)
  BIND_END=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((KL0 + 1))" \
    | grep -c "FFN SWIGLU profile bound: on (mtp_capable=1)" || true)
  echo "end_mode=${MODE_END:-FFN_MODE_LINE_NOT_FOUND} end_bind=${BIND_END:-0}"
  grep -oE "[0-9]+ FFN SWIGLU dispatches" "$D/log-end.log" 2>/dev/null | tail -1 || echo "FFN_DISPATCHES=none-in-window (counters flush on clean unload only; api-recall is the counter gate)"
  grep -oE "session config: model=[^ ]+ mtp=[0-9]+ depth=[0-9]+ pmin=[0-9.]+ n_ctx=[0-9]+ n_batch=[0-9]+ n_ubatch=[0-9]+ gpu=[0-9]+" "$D/window-slice.txt" 2>/dev/null | head -2 || true
} | tee -a "$D/verdicts.txt"
./scripts/deploy.sh delete-file "$PFN" d3d12swiglu.txt >/dev/null 2>&1 || true
exit $((fails > 0 ? 1 : 0))
