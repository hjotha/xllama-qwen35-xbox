#!/usr/bin/env bash
# T4a-slots.sh — plan 009 amplification batch 1: prompt-family matrix on ONE
# build (rev236 Iteration). Families: A cipher-shallow, B code-deep,
# C std512-opening, F minimal-suffix. Arms per family: FRESH (no restore
# reference) / CONTROL (restore, no flag) / ZEROCARRY (restore, flag).
# Cutover protocol per family: fresh seed+save, then one restart per arm.
# Records: restore ms, reuse line, turn tuple, prefill, drafted/accept,
# rounds, WS, text + equality vs FRESH. No flag => control identical to
# rev235 Final behavior. MTP on throughout; thresholds unchanged.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T4a"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.236_x64__pj67f1fcj4n14"
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

pyjson() { python3 -c "import json,sys;print(json.dumps([{'role':'user','content':open(sys.argv[1]).read()}]))" "$1"; }

family() { # name seedfile contmsg maxtok
  local fam="$1" seedfile="$2" contmsg="$3" maxtok="$4"
  local sjson cjson reply
  sjson=$(python3 -c "import json,sys;print(json.dumps([{'role':'user','content':open(sys.argv[1],'rb').read().decode('utf-8')}]))" "$seedfile")
  # seed + save
  restart_verified "$fam-seed"
  chat "$D/$fam-seed.json" 8 "$sjson"
  reply=$(python3 -c "import json;print(json.load(open('$D/$fam-seed.json'))['choices'][0]['message']['content'])")
  echo "seed_tokens=$(usage_of "$D/$fam-seed.json")" | tee -a "$D/verdicts.txt"
  slot_save "$D/$fam-save.json" "t4-$fam"
  # continuation prompt: seed + REAL assistant reply + cont (so kv_keep
  # covers the saved tokens and F3.5 can_carry_from is testable)
  cjson=$(python3 - "$seedfile" "$reply" "$contmsg" <<'PY'
import json, sys
seed, reply, cont = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps([{"role": "user", "content": open(seed, "rb").read().decode("utf-8")},
                  {"role": "assistant", "content": reply},
                  {"role": "user", "content": cont}]))
PY
)
  # FRESH arm: same continuation prompt, fresh process, no restore
  restart_verified "$fam-fresh"
  chat "$D/$fam-pre.json" 1 '[{"role":"user","content":"x"}]'
  L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  chat "$D/$fam-fresh.json" "$maxtok" "$cjson"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/$fam-fresh-slice.log"
  arm_block() { # arm
    local arm="$1" ad="$D/$fam-$1" L1
    mkdir -p "$ad"
    ws_sample "$fam-$arm-pre"
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" 2>/dev/null | tail -1
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" 2>/dev/null | tail -1
    grep -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$ad/slice.log" 2>/dev/null | head -1
    grep -oE "KV prefix reuse — kept [0-9]+ of [0-9]+ tokens" "$ad/slice.log" 2>/dev/null | head -1
    grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$ad/slice.log" 2>/dev/null || echo "T1_LINE_ABSENT"
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" 2>/dev/null | tail -1
    grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$ad/slice.log" 2>/dev/null | tail -1
    python3 -c "import json;d=json.load(open('$ad/cont.json'));c=d['choices'][0]['message']['content'];u=d['usage'];print(f\"CTOK={u['prompt_tokens']}/{u['completion_tokens']} sha={__import__('hashlib').sha256(c.encode()).hexdigest()[:12]}\")" 2>/dev/null || echo "CONT_UNREADABLE"
    ws_sample "$fam-$arm-post"
  }
  { echo "== $fam FRESH"; grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/$fam-fresh-slice.log" | tail -1
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/$fam-fresh-slice.log" | tail -1
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/$fam-fresh-slice.log" | tail -1
    python3 -c "import json;d=json.load(open('$D/$fam-fresh.json'));c=d['choices'][0]['message']['content'];u=d['usage'];print(f\"CTOK={u['prompt_tokens']}/{u['completion_tokens']} sha={__import__('hashlib').sha256(c.encode()).hexdigest()[:12]}\")"; } | tee -a "$D/verdicts.txt"
  for arm in control zero; do
    local mode="none" fad="$D/$fam-$arm"
    mkdir -p "$fad"
    [[ "$arm" == "zero" ]] && mode="zerocarry"
    set_flag "$mode"
    restart_verified "$fam-$arm-newproc"
    chat "$fad/pre.json" 1 '[{"role":"user","content":"x"}]'
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    slot_restore "$fad/restore.json" "t4-$fam" || fail "$fam-$arm restore"
    chat "$fad/cont.json" "$maxtok" "$cjson" || fail "$fam-$arm cont"
    ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$fad/slice.log"
    { echo "== $fam $arm (flag=$mode)"; arm_block "$arm"; } | tee -a "$D/verdicts.txt"
  done
  set_flag none
  slot_delete_q "t4-$fam"
  python3 - "$D" "$fam" <<'PY'
import hashlib, json, sys
d, fam = sys.argv[1], sys.argv[2]
def sha(p):
    try:
        return hashlib.sha256(json.load(open(f"{d}/{p}.json"))["choices"][0]["message"]["content"].encode()).hexdigest()[:12]
    except Exception:
        return "ERR"
fr, ct, zr = f"{fam}-fresh.json", f"{fam}-control/cont.json", f"{fam}-zero/cont.json"
print(f"TEXT[{fam}] fresh={sha(fr)} control={sha(ct)} zero={sha(zr)} equal_control={sha(fr)==sha(ct)} equal_zero={sha(fr)==sha(zr)}")
PY
}

family A "$REPO/bench/results/qwen4b-maxperf/slots-rev211/prompts/famA.txt" "Repeat the cipher back exactly." 16
family B "$REPO/bench/prompts/spec-code-edit.txt" "In the first version, which element is dropped by the off-by-one and where is the bug?" 64
family C "$REPO/bench/prompts/standard-512.txt" "State the opening claim of the text in one sentence." 32
family F "$REPO/bench/results/qwen4b-maxperf/slots-rev211/prompts/famA.txt" "Cipher?" 4
echo "== T4a done"