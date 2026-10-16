#!/usr/bin/env bash
# T1-rebuild-test.sh — plan 009 T1: disk save/restore + continuation with MTP
# on in a NEW process on the rev233 Iteration build (T1 hook, flag-gated).
# Three arms, identical prompts (greedy => identical expected text):
#   control   no flag   -> full prefill (1,0,62,62), response BRAVO-7
#   full      flag=full -> harvest+process rebuild; expect (1,42,20,62),
#                          drafted>0, sane acceptance, text == control
#   carryonly flag      -> bare carry, no draft history; expect carry
#                          accepted (kv_before=42) but ~0 acceptance, text ==
#                          control (correctness via target verification)
# Fail-closed: any hook failure degrades to control behavior. Evidence:
# per-arm JSONs + log slices with T1/turn/prefill lines. No format change.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T1"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.233_x64__pj67f1fcj4n14"
API=http://192.168.1.26:11434
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
mkdir -p "$D"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
: > "$D/verdicts.txt"
pass() { echo "PASS: $1" | tee -a "$D/verdicts.txt"; }
fail() { echo "FAIL: $1" | tee -a "$D/verdicts.txt"; }

chat() {
  curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen35-4b-mtp\",\"temperature\":0,\"max_tokens\":$2,\"messages\":$3}" > "$1"
  grep -q '"choices"' "$1" || { fail "chat $1 no choices"; return 1; }
}
slot_save() {
  curl -s -m 300 "$API/v1/slots/save" -H 'Content-Type: application/json' -d "{\"slot\":\"$2\"}" > "$1"
  grep -q '"ok":true' "$1" || { fail "slot save $2"; return 1; }
}
slot_restore() {
  curl -s -m 300 "$API/v1/slots/restore" -H 'Content-Type: application/json' -d "{\"slot\":\"$2\"}" > "$1"
  grep -q '"ok":true' "$1"
}
restart_verified() {
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  local down=0
  for _ in $(seq 1 20); do
    if ! curl -s -m 4 "$API/" >/dev/null 2>&1; then down=$((down + 1)); else down=0; fi
    [[ "$down" -ge 3 ]] && break
    sleep 3
  done
  [[ "$down" -ge 3 ]] || { fail "restart ($1): app never died"; return 1; }
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
  local ok=""
  for _ in $(seq 1 60); do
    if curl -s -m 5 "$API/" >/dev/null 2>&1; then ok=1; break; fi
    sleep 3
  done
  [[ -n "$ok" ]] || { fail "restart ($1): app never came back"; return 1; }
  pass "restart verified ($1)"
}
set_flag() { # mode | "none"
  ./scripts/deploy.sh delete-file "$PFN" mtp-rebuild-test.flag >/dev/null 2>&1 || true
  if [[ "$1" != "none" ]]; then
    printf '%s' "$1" > "$D/flagbody.txt"
    ./scripts/deploy.sh upload-file "$D/flagbody.txt" "$PFN" "" mtp-rebuild-test.flag >/dev/null
  fi
}
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
P1A='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'
content_of() { python3 -c "import json,sys;print(json.load(open('$1'))['choices'][0]['message']['content'])"; }

run_arm() { # name flagmode
  local arm="$1" mode="$2"
  local ad="$D/$arm"
  mkdir -p "$ad"
  set_flag "$mode"
  restart_verified "$arm-fresh"
  chat "$ad/seed.json" 16 "$P1"
  slot_save "$ad/save.json" "t1-$arm"
  restart_verified "$arm-newproc"
  # restore needs a resident session (model load); this pre-chat's KV is
  # fully replaced by set_data below (same as the battery's restores, which
  # always run with a loaded model).
  chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]' || { fail "$arm: preload chat failed"; return 1; }
  local L0
  L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  slot_restore "$ad/restore.json" "t1-$arm" || { fail "$arm: restore failed"; return 1; }
  chat "$ad/cont.json" 16 "$P1A" || { fail "$arm: continue failed"; return 1; }
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  {
    echo "== arm $arm (flag=$mode)"
    grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$ad/slice.log" || echo "T1_LINE_ABSENT"
    grep -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$ad/slice.log" | head -2
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1
    grep -oE "reuse=[01] drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1
    echo "response: $(content_of "$ad/cont.json")"
  } | tee -a "$D/verdicts.txt"
}

run_arm control none
run_arm full full
run_arm carryonly carryonly
set_flag none
for s in control full carryonly; do
  curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d "{\"slot\":\"t1-$s\"}" >/dev/null || true
done
echo "== T1 arms done (see verdicts.txt + per-arm slice.log)"
