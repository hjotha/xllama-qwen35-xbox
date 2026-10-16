#!/usr/bin/env bash
# T4b-slots.sh — plan 009 amplification batch 2 on the SAME build (rev236):
# family E DEEP-PREFIX (cipher at the START, asked at the END after ~200
# tokens of filler) with FOUR arms (fresh / control / zerocarry / full —
# full is the correct-draft-history reference), plus the MULTI-TURN chain:
# 3 post-restore turns in one new process, control vs zerocarry (does the
# empty deep-prefix mirror hole cost acceptance on turns 2-3?).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T4b"
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
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'

# ---- family E buildup: seed + 3 filler turns -> ~200tok history
restart_verified "E-build"
chat "$D/e1.json" 16 "$P1"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
r1 = json.load(open(f"{d}/e1.json"))["choices"][0]["message"]["content"]
open(f"{d}/e2m.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."}]))
PY
chat "$D/e2.json" 48 "$(cat "$D/e2m.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
open(f"{d}/e3m.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("e1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("e2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."}]))
PY
chat "$D/e3.json" 48 "$(cat "$D/e3m.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
msgs = [{"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
        {"role": "assistant", "content": rp("e1")},
        {"role": "user", "content": "Describe how a sorted merge works, step by step."},
        {"role": "assistant", "content": rp("e2")},
        {"role": "user", "content": "Now explain RoPE attention to a junior engineer."},
        {"role": "assistant", "content": rp("e3")},
        {"role": "user", "content": "Going back to the very first thing in this conversation: what cipher was stated, exactly? Reply with just the cipher."}]
open(f"{d}/E-cont.json", "w").write(json.dumps(msgs))
print("deep cont built:", len(msgs), "msgs")
PY
slot_save "$D/E-save.json" "t4b-E"
echo "E-seed-saved $(usage_of "$D/e3.json")" | tee -a "$D/verdicts.txt"
ECONT="$(cat "$D/E-cont.json")"

restart_verified "E-fresh"
chat "$D/E-pre.json" 1 '[{"role":"user","content":"x"}]'
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/E-fresh.json" 16 "$ECONT"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/E-fresh-slice.log"
{ echo "== E FRESH (deep prefix, no restore)"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/E-fresh-slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/E-fresh-slice.log" | tail -1
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/E-fresh-slice.log" | tail -1
  python3 -c "import json;print('response:',repr(json.load(open('$D/E-fresh.json'))['choices'][0]['message']['content']))"; } | tee -a "$D/verdicts.txt"
for arm in control zero full; do
  local_mode=none; [[ "$arm" == "zero" ]] && local_mode=zerocarry; [[ "$arm" == "full" ]] && local_mode=full
  set_flag "$local_mode"
  ad="$D/E-$arm"; mkdir -p "$ad"
  restart_verified "E-$arm-newproc"
  chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
  ws_sample "E-$arm-pre"
  L1=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  slot_restore "$ad/restore.json" "t4b-E" || fail "E-$arm restore"
  chat "$ad/cont.json" 16 "$ECONT" || fail "E-$arm cont"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L1 + 1))" > "$ad/slice.log"
  { echo "== E $arm (flag=$local_mode)"
    grep -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$ad/slice.log" | head -1
    grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+ pending=[01] pos=[0-9]+ carry_from_n=[01]" "$ad/slice.log" || echo "T1_LINE_ABSENT"
    grep -oE "KV prefix reuse — kept [0-9]+ of [0-9]+ tokens" "$ad/slice.log" | head -1
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$ad/slice.log" | tail -1
    python3 -c "import json;print('response:',repr(json.load(open('$ad/cont.json'))['choices'][0]['message']['content']))"; } | tee -a "$D/verdicts.txt"
  ws_sample "E-$arm-post"
done
set_flag none
python3 - "$D" <<'PY'
import hashlib, json, sys
d = sys.argv[1]
def sha(n):
    try:
        return hashlib.sha256(json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"].encode()).hexdigest()[:12]
    except Exception:
        return "ERR"
print(f"TEXT[E] fresh={sha('E-fresh.json')} control={sha('E-control/cont.json')} zero={sha('E-zero/cont.json')} full={sha('E-full/cont.json')}")
print(f"EQUAL_E control={sha('E-fresh.json')==sha('E-control/cont.json')} zero={sha('E-fresh.json')==sha('E-zero/cont.json')} full={sha('E-fresh.json')==sha('E-full/cont.json')}")
PY
slot_delete_q t4b-E

# ---- multi-turn chain: family A seed, 3 post-restore turns per arm
for arm in mcontrol mzero; do
  mode=none; [[ "$arm" == "mzero" ]] && mode=zerocarry
  set_flag "$mode"
  ad="$D/$arm"; mkdir -p "$ad"
  restart_verified "$arm-build"
  chat "$ad/seed.json" 16 "$P1"
  slot_save "$ad/save.json" "t4b-$arm"
  restart_verified "$arm-newproc"
  chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
  L2=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  slot_restore "$ad/restore.json" "t4b-$arm" || fail "$arm restore"
  PA="[{\"role\":\"user\",\"content\":\"The northern depot cipher is BRAVO-7. Acknowledge with: noted.\"},{\"role\":\"assistant\",\"content\":\"noted\"},{\"role\":\"user\",\"content\":\"Repeat the cipher back exactly.\"}]"
  chat "$ad/t1.json" 24 "$PA" || fail "$arm t1"
  TC1=$(python3 -c "import json;print(json.load(open('$ad/t1.json'))['choices'][0]['message']['content'])")
  PB="[{\"role\":\"user\",\"content\":\"The northern depot cipher is BRAVO-7. Acknowledge with: noted.\"},{\"role\":\"assistant\",\"content\":\"noted\"},{\"role\":\"user\",\"content\":\"Repeat the cipher back exactly.\"},{\"role\":\"assistant\",\"content\":\"$TC1\"},{\"role\":\"user\",\"content\":\"Now spell it backwards letter by letter.\"}]"
  chat "$ad/t2.json" 24 "$PB" || fail "$arm t2"
  TC2=$(python3 -c "import json;print(json.load(open('$ad/t2.json'))['choices'][0]['message']['content'])")
  PC="[{\"role\":\"user\",\"content\":\"The northern depot cipher is BRAVO-7. Acknowledge with: noted.\"},{\"role\":\"assistant\",\"content\":\"noted\"},{\"role\":\"user\",\"content\":\"Repeat the cipher back exactly.\"},{\"role\":\"assistant\",\"content\":\"$TC1\"},{\"role\":\"user\",\"content\":\"Now spell it backwards letter by letter.\"},{\"role\":\"assistant\",\"content\":\"$TC2\"},{\"role\":\"user\",\"content\":\"How many characters does the cipher have? Answer with one number.\"}]"
  chat "$ad/t3.json" 8 "$PC" || fail "$arm t3"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L2 + 1))" > "$ad/slice.log"
  { echo "== $arm MULTI-TURN (post-restore t1..t3 same process)"
    grep -oE "KV state loaded — [0-9]+ tokens" "$ad/slice.log" | head -1
    grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+" "$ad/slice.log" || echo "T1_LINE_ABSENT"
    i=1; while IFS= read -r line; do
      t=$(echo "$line" | grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+")
      p=$(echo "$line" | grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]")
      a=$(echo "$line" | grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+")
      echo "  t$i: $t | $p | $a"; i=$((i + 1))
    done < <(grep -E "turn full=|prefill=" "$ad/slice.log")
    for t in 1 2 3; do python3 -c "import json;print(f'  t$t response:',repr(json.load(open('$ad/t$t.json'))['choices'][0]['message']['content'][:60]))"; done; } | tee -a "$D/verdicts.txt"
  ws_sample "$arm-post"
  slot_delete_q "t4b-$arm"
done
set_flag none
python3 - "$D" <<'PY'
import hashlib, json, sys
d = sys.argv[1]
def rp(arm, t):
    try:
        return json.load(open(f"{d}/{arm}/t{t}.json"))["choices"][0]["message"]["content"]
    except Exception:
        return "ERR"
for t in (1, 2, 3):
    a, b = rp("mcontrol", t), rp("mzero", t)
    print(f"MULTI t{t} equal={a == b} control={a[:40]!r} zero={b[:40]!r}")
PY
echo "== T4b done"