#!/usr/bin/env bash
# T3b-M3x.sh — plan 009 M3 extension: zerocarry on DIVERGENT tail + LONG
# prefix (same rev234 build). Discriminators the short/long same-prefix
# arms cannot provide: (a) divergent tail with zero-carry (does partial
# reuse survive without any true prefix knowledge?); (b) 151-token prefix
# (does the ~1.5s harvest saving materialize at scale with drafting?).
# References: T2a DIV-control/DIV-full + LONG-FULL (same prompts).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T3b-M3x"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.234_x64__pj67f1fcj4n14"
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

P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
P1A='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'
PDIV='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Forget the cipher. The southern depot cipher is CHARLIE-9. Repeat the southern cipher back exactly."}]'

# DIV-ZERO: same seed slot as T2a/T3 short arms, divergent continuation
set_flag zerocarry
restart_verified "divz-fresh"
chat "$D/seed.json" 16 "$P1"
slot_save "$D/save.json" "t3b-dz"
restart_verified "divz-newproc"
chat "$D/pre.json" 1 '[{"role":"user","content":"x"}]'
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/restore.json" "t3b-dz" || fail "divz restore"
chat "$D/div.json" 16 "$PDIV" || fail "divz continue"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/div-slice.log"
{ echo "== DIV-ZERO (flag=zerocarry)"
  grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$D/div-slice.log" || echo T1_LINE_ABSENT
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/div-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/div-slice.log" | tail -1
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/div-slice.log" | tail -1
  echo "response: $(content_of "$D/div.json")"; } | tee -a "$D/verdicts.txt"

# LONG-ZERO: rebuild the ~150tok history, then zerocarry restore + continue
restart_verified "longz-build"
chat "$D/l1.json" 16 "$P1"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def reply(name):
    return json.load(open(f"{d}/{name}.json"))["choices"][0]["message"]["content"]
r1 = reply("l1")
open(f"{d}/m2.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Spell it backwards and explain each character."}]))
PY
chat "$D/l2.json" 32 "$(cat "$D/m2.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def reply(name):
    return json.load(open(f"{d}/{name}.json"))["choices"][0]["message"]["content"]
r1 = reply("l1"); r2 = reply("l2")
open(f"{d}/m3.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Spell it backwards and explain each character."},
    {"role": "assistant", "content": r2},
    {"role": "user", "content": "Now spell it forwards and count the characters."}]))
PY
chat "$D/l3.json" 32 "$(cat "$D/m3.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def reply(name):
    return json.load(open(f"{d}/{name}.json"))["choices"][0]["message"]["content"]
r1 = reply("l1"); r2 = reply("l2"); r3 = reply("l3")
open(f"{d}/long-prompt.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Spell it backwards and explain each character."},
    {"role": "assistant", "content": r2},
    {"role": "user", "content": "Now spell it forwards and count the characters."},
    {"role": "assistant", "content": r3},
    {"role": "user", "content": "Repeat the cipher three times with indices."}]))
PY
slot_save "$D/lsave.json" "t3b-lz"
restart_verified "longz-newproc"
chat "$D/lpre.json" 1 '[{"role":"user","content":"x"}]'
ws_sample "longz-pre"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/lrestore.json" "t3b-lz" || fail "longz restore"
chat "$D/long.json" 32 "$(cat "$D/long-prompt.json")" || fail "longz continue"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/long-slice.log"
{ echo "== LONG-ZERO (flag=zerocarry)"
  grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$D/long-slice.log" || echo T1_LINE_ABSENT
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/long-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/long-slice.log" | tail -1
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/long-slice.log" | tail -1
  grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$D/long-slice.log" | tail -1
  echo "response: $(content_of "$D/long.json") $(usage_of "$D/long.json")"; } | tee -a "$D/verdicts.txt"
ws_sample "longz-post"
set_flag none
slot_delete_q t3b-dz; slot_delete_q t3b-lz
echo "== T3b-M3x done"
