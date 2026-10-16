#!/usr/bin/env bash
# T2a-slots.sh — plan 009 T2 expanded (mechanism/scale/divergence/stop).
# API-level, reversible, MTP on, rev233 Iteration (all arms same build).
# Arms: FRESH (no restore baseline) / CONTROL (restore, no flag) / FULL
# (flag=full) / LONG fresh+full (~200tok prefix, scaling) / DIV
# fresh+control+full (divergent tail) / ROLLBACK (older slot, new process) /
# EOSSTOP (max_tokens=2 stop + save/restore/continue, flag=full).
# Memory (portal WS) sampled per phase. Thresholds: exact texts, exact
# tuples, verified restarts — no epsilon.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T2a"
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

P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
P1A='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'
PDIV='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Forget the cipher. The southern depot cipher is CHARLIE-9. Repeat the southern cipher back exactly."}]'

# FRESH: no save/restore — apples-to-apples baseline for CONTROL
restart_verified "fresh"
chat "$D/fresh-pre.json" 1 '[{"role":"user","content":"x"}]'
ws_sample "fresh-pre"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/fresh.json" 16 "$P1A"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/fresh-slice.log"
{ echo "== FRESH (no restore)"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/fresh-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/fresh-slice.log" | tail -1
  echo "response: $(content_of "$D/fresh.json") $(usage_of "$D/fresh.json")"; } | tee -a "$D/verdicts.txt"
ws_sample "fresh-post"

# CONTROL: save/restore/continue, no flag (T1-control rerun + WS + rounds)
set_flag none
restart_verified "control-fresh"
chat "$D/c-seed.json" 16 "$P1"
slot_save "$D/c-save.json" "t2a-c"
restart_verified "control-newproc"
chat "$D/c-pre.json" 1 '[{"role":"user","content":"x"}]'
ws_sample "control-post-restore-pre"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/c-restore.json" "t2a-c" || fail "control restore"
chat "$D/c-cont.json" 16 "$P1A" || fail "control continue"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/c-slice.log"
{ echo "== CONTROL (restore, no flag)"
  grep -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$D/c-slice.log" | head -1
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/c-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/c-slice.log" | tail -1
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/c-slice.log" | tail -1
  grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$D/c-slice.log" | tail -1
  echo "response: $(content_of "$D/c-cont.json") $(usage_of "$D/c-cont.json")"; } | tee -a "$D/verdicts.txt"
ws_sample "control-post"

# FULL: flag=full (T1-full rerun + WS + rounds)
set_flag full
restart_verified "full-fresh"
chat "$D/f-seed.json" 16 "$P1"
slot_save "$D/f-save.json" "t2a-f"
restart_verified "full-newproc"
chat "$D/f-pre.json" 1 '[{"role":"user","content":"x"}]'
ws_sample "full-post-restore-pre"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/f-restore.json" "t2a-f" || fail "full restore"
chat "$D/f-cont.json" 16 "$P1A" || fail "full continue"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/f-slice.log"
{ echo "== FULL (flag=full)"
  grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$D/f-slice.log" || echo T1_LINE_ABSENT
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/f-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/f-slice.log" | tail -1
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/f-slice.log" | tail -1
  grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$D/f-slice.log" | tail -1
  echo "response: $(content_of "$D/f-cont.json") $(usage_of "$D/f-cont.json")"; } | tee -a "$D/verdicts.txt"
ws_sample "full-post"
set_flag none

# LONG prefix (~200tok): buildup with 32tok replies, save, fresh-long text
# baseline vs full-restore continuation
set_flag none
restart_verified "long-build"
chat "$D/l1.json" 16 "$P1"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def reply(name):
    return json.load(open(f"{d}/{name}.json"))["choices"][0]["message"]["content"]
def msgs(*parts):
    # parts: sequence of ("user"|"assistant", text) starting with the seed user turn
    return json.dumps([{"role": r, "content": t} for r, t in parts])
r1 = reply("l1")
open(f"{d}/m2.json", "w").write(msgs(
    ("user", "The northern depot cipher is BRAVO-7. Acknowledge with: noted."),
    ("assistant", r1),
    ("user", "Spell it backwards and explain each character.")))
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
print("long prompt built")
PY
P4L="$(cat "$D/long-prompt.json")"
usage_of "$D/l3.json" | tee -a "$D/verdicts.txt"
slot_save "$D/l-save.json" "t2a-l"
# fresh-long baseline: full long prompt, no restore
restart_verified "longfresh"
chat "$D/lf-pre.json" 1 '[{"role":"user","content":"x"}]'
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/long-fresh.json" 32 "$P4L"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/lf-slice.log"
{ echo "== LONG-FRESH (no restore)"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/lf-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/lf-slice.log" | tail -1
  echo "response: $(content_of "$D/long-fresh.json") $(usage_of "$D/long-fresh.json")"; } | tee -a "$D/verdicts.txt"
# long-full: restore (flag=full) + same continuation
set_flag full
restart_verified "longfull-newproc"
chat "$D/ll-pre.json" 1 '[{"role":"user","content":"x"}]'
ws_sample "long-pre"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/l-restore.json" "t2a-l" || fail "long restore"
chat "$D/long-full.json" 32 "$P4L" || fail "long continue"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/ll-slice.log"
{ echo "== LONG-FULL (flag=full)"
  grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$D/ll-slice.log" || echo T1_LINE_ABSENT
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/ll-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/ll-slice.log" | tail -1
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/ll-slice.log" | tail -1
  echo "response: $(content_of "$D/long-full.json") $(usage_of "$D/long-full.json")"; } | tee -a "$D/verdicts.txt"
ws_sample "long-post"
set_flag none

# DIVERGENT tail: fresh vs control-restore vs full-restore (same divergent prompt)
restart_verified "divfresh"
chat "$D/df-pre.json" 1 '[{"role":"user","content":"x"}]'
chat "$D/div-fresh.json" 16 "$PDIV" || fail "div fresh"
for mode in control full; do
  if [[ "$mode" == "full" ]]; then set_flag full; else set_flag none; fi
  restart_verified "div-$mode-newproc"
  chat "$D/d-$mode-pre.json" 1 '[{"role":"user","content":"x"}]'
  L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  slot_restore "$D/d-$mode-restore.json" "t2a-c" || fail "div-$mode restore"
  chat "$D/div-$mode.json" 16 "$PDIV" || fail "div-$mode continue"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/d-$mode-slice.log"
  { echo "== DIV-$mode"
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/d-$mode-slice.log" | tail -1
    echo "response: $(content_of "$D/div-$mode.json")"; } | tee -a "$D/verdicts.txt"
done
set_flag none

# ROLLBACK in new process: seed/saveA, diverge/saveB, restart, restore A
restart_verified "rollback-build"
chat "$D/rb-seed.json" 16 "$P1"
slot_save "$D/rb-saveA.json" "t2a-rbA"
chat "$D/rb-div.json" 16 "$PDIV"
slot_save "$D/rb-saveB.json" "t2a-rbB"
restart_verified "rollback-newproc"
chat "$D/rb-pre.json" 1 '[{"role":"user","content":"x"}]'
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/rb-restoreA.json" "t2a-rbA" || fail "rollback restore A"
chat "$D/rb-contA.json" 16 "$P1A" || fail "rollback continue A"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/rb-slice.log"
{ echo "== ROLLBACK (restore older slot A in new process)"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/rb-slice.log" | tail -1
  echo "response: $(content_of "$D/rb-contA.json") (want BRAVO-7, A's line)"; } | tee -a "$D/verdicts.txt"

# EOSSTOP under rebuild: max_tokens=2 stop, then save/restore/continue
set_flag full
restart_verified "eosstop-fresh"
chat "$D/e-seed.json" 16 "$P1"
slot_save "$D/e-save.json" "t2a-e"
restart_verified "eosstop-newproc"
chat "$D/e-pre.json" 1 '[{"role":"user","content":"x"}]'
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/e-restore.json" "t2a-e" || fail "eosstop restore"
chat "$D/e-stop.json" 2 "$P1A" || fail "eosstop stopped turn"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/e-slice.log"
slot_save "$D/e-save2.json" "t2a-e2" || fail "eosstop re-save"
restart_verified "eosstop-newproc2"
chat "$D/e-pre2.json" 1 '[{"role":"user","content":"x"}]'
slot_restore "$D/e-restore2.json" "t2a-e2" || fail "eosstop restore2"
chat "$D/e-cont2.json" 16 "$P1A" || fail "eosstop continue2"
{ echo "== EOSSTOP (max_tokens=2 stop + save/restore/continue, flag=full)"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/e-slice.log" | tail -1
  python3 -c "import json;d=json.load(open('$D/e-stop.json'));print('stop_completion:',repr(d['choices'][0]['message']['content']),'finish:',d['choices'][0].get('finish_reason'))"
  echo "post-restore response: $(content_of "$D/e-cont2.json")"; } | tee -a "$D/verdicts.txt"
set_flag none

for s in t2a-c t2a-f t2a-l t2a-rbA t2a-rbB t2a-e t2a-e2; do slot_delete_q "$s"; done
echo "== T2a done"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
pairs = [("fresh", "fresh.json", "c-cont", "c-cont.json"), ("fresh", "fresh.json", "f-cont", "f-cont.json"),
         ("long-fresh", "long-fresh.json", "long-full", "long-full.json"),
         ("div-fresh", "div-fresh.json", "div-control", "div-control.json"),
         ("div-fresh", "div-fresh.json", "div-full", "div-full.json")]
for an, af, bn, bf in pairs:
    try:
        a = json.load(open(f"{d}/{af}"))["choices"][0]["message"]["content"]
        b = json.load(open(f"{d}/{bf}"))["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"TEXT_{an}_vs_{bn}=UNREADABLE {e}"); continue
    print(f"TEXT_{an}_vs_{bn}={a == b}")
PY