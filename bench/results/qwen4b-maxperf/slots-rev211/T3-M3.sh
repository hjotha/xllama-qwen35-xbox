#!/usr/bin/env bash
# T3-M3.sh — plan 009 M3: harvest-free zero-carry vs full vs control, SAME
# build (rev234 Iteration). Question: can drafting engage and accept from a
# zero carry + tail history, with NO harvest at all? If M3 acceptance ~ full,
# even the carry row is dispensable (efficient v2 path: restore ms only).
# If ~0, the carry row is load-bearing (harvest floor stands).
# Arms: control/full/zerocarry (16tok) + full64/zerocarry64 (64tok long).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T3-M3"
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
PLONG='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Explain in detail what the northern depot cipher is, who issued it, how it is structured, and repeat it three times with commentary."}]'

run_arm() { # name flagmode prompt max
  local arm="$1" mode="$2" prompt="$3" max="$4"
  local ad="$D/$arm"
  mkdir -p "$ad"
  set_flag "$mode"
  restart_verified "$arm-fresh"
  chat "$ad/seed.json" 16 "$P1"
  slot_save "$ad/save.json" "t3-$arm"
  restart_verified "$arm-newproc"
  chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]' || { fail "$arm: preload failed"; return 1; }
  ws_sample "$arm-pre"
  local L0
  L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  slot_restore "$ad/restore.json" "t3-$arm" || { fail "$arm: restore failed"; return 1; }
  chat "$ad/cont.json" "$max" "$prompt" || { fail "$arm: continue failed"; return 1; }
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  {
    echo "== arm $arm (flag=$mode)"
    grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$ad/slice.log" || echo "T1_LINE_ABSENT"
    grep -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$ad/slice.log" | head -1
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$ad/slice.log" | tail -1
    python3 -c "import json;d=json.load(open('$ad/cont.json'));c=d['choices'][0]['message']['content'];u=d['usage'];print(f\"response={c[:40]!r} prompt_tok={u['prompt_tokens']} completion_tok={u['completion_tokens']}\")"
  } | tee -a "$D/verdicts.txt"
  ws_sample "$arm-post"
}
run_arm control none "$P1A" 16
run_arm full full "$P1A" 16
run_arm zerocarry zerocarry "$P1A" 16
run_arm full64 full "$PLONG" 64
run_arm zerocarry64 zerocarry "$PLONG" 64
set_flag none
for s in control full zerocarry full64 zerocarry64; do slot_delete_q "t3-$s"; done
echo "== T3-M3 done"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
refs = {}
for arm in ("control", "full", "zerocarry"):
    try:
        refs[arm] = json.load(open(f"{d}/{arm}/cont.json"))["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"{arm}=UNREADABLE {e}")
print(f"TEXT_control_vs_full={refs.get('control') == refs.get('full')}")
print(f"TEXT_control_vs_zerocarry={refs.get('control') == refs.get('zerocarry')}")
try:
    a = json.load(open(f"{d}/full64/cont.json"))["choices"][0]["message"]["content"]
    b = json.load(open(f"{d}/zerocarry64/cont.json"))["choices"][0]["message"]["content"]
    print(f"TEXT_full64_vs_zerocarry64={a == b} len={len(a)}/{len(b)}")
except Exception as e:
    print(f"long=UNREADABLE {e}")
PY