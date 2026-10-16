#!/usr/bin/env bash
# T4c-slots.sh — multi-turn chain only (rev236, same build as T4a/T4b):
# post-restore t1..t3 in ONE new process, control vs zerocarry. Question:
# does the empty deep-prefix mirror hole (zerocarry) cost acceptance on
# turns 2-3, and do texts stay identical? (T4b died on a set -e grep bug
# inside its per-turn printer, not on device evidence.)
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T4c"
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
PA='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'

for arm in mcontrol mzero; do
  mode=none; [[ "$arm" == "mzero" ]] && mode=zerocarry
  set_flag "$mode"
  ad="$D/$arm"; mkdir -p "$ad"
  restart_verified "$arm-build"
  chat "$ad/seed.json" 16 "$P1"
  slot_save "$ad/save.json" "t4-$arm"
  restart_verified "$arm-newproc"
  chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
  L2=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  slot_restore "$ad/restore.json" "t4-$arm" || fail "$arm restore"
  chat "$ad/t1.json" 24 "$PA" || fail "$arm t1"
  TC1=$(python3 -c "import json;print(json.load(open('$ad/t1.json'))['choices'][0]['message']['content'])")
  TC2=$(python3 -c "import json,sys;print(json.dumps(sys.argv[1])[1:-1])" "$(content_of "$ad/t1.json")")
  PB=$(python3 - "$TC1" <<'PY'
import json, sys
print(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": "noted"},
    {"role": "user", "content": "Repeat the cipher back exactly."},
    {"role": "assistant", "content": sys.argv[1]},
    {"role": "user", "content": "Now spell it backwards letter by letter."}]))
PY
)
  chat "$ad/t2.json" 24 "$PB" || fail "$arm t2"
  TC2=$(content_of "$ad/t2.json")
  PC=$(python3 - "$TC1" "$TC2" <<'PY'
import json, sys
print(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": "noted"},
    {"role": "user", "content": "Repeat the cipher back exactly."},
    {"role": "assistant", "content": sys.argv[1]},
    {"role": "user", "content": "Now spell it backwards letter by letter."},
    {"role": "assistant", "content": sys.argv[2]},
    {"role": "user", "content": "How many characters does the cipher have? Answer with one number."}]))
PY
)
  chat "$ad/t3.json" 8 "$PC" || fail "$arm t3"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L2 + 1))" > "$ad/slice.log"
  {
    echo "== $arm MULTI-TURN (post-restore t1..t3 same process)"
    grep -oE "KV state loaded — [0-9]+ tokens" "$ad/slice.log" | head -1 || true
    grep -oE "T1: rebuild mode=[a-z]+ n=[0-9]+ harvest=[0-9.]+ms process=[0-9.]+ms [A-Z]+" "$ad/slice.log" || echo "T1_LINE_ABSENT"
    # NB: no `grep | python <<heredoc` under pipefail — the heredoc feeds
    # stdin to python and grep dies of SIGPIPE (141) which pipefail turns
    # into a script kill (diagnosed 2026-10-10, T4c first attempt).
    python3 - "$ad" <<'PY'
import re, sys
import pathlib
ad = sys.argv[1]
text = pathlib.Path(f"{ad}/slice.log").read_text(errors="replace")
turns = re.findall(r"turn full=(\d) kv_before=(\d+) pf=(\d+) prompt=(\d+)", text)
pf = re.findall(r"prefill=([\d.]+)ms \((\d+) tok\) reuse=(\d)", text)
ac = re.findall(r"drafted=(\d+) \(mtp=(\d+) lookup=(\d+)\) spec_accept=(\d+)", text)
for i in range(3):
    t = turns[i] if i < len(turns) else ("?", "?", "?", "?")
    p = pf[i] if i < len(pf) else ("?", "?", "?")
    a = ac[i] if i < len(ac) else ("?", "?", "?", "?")
    print(f"  t{i+1}: kv_before={t[1]} pf={t[2]}/{t[3]} prefill={p[0]}ms({p[1]}tok) "
          f"drafted={a[0]} accept={a[3]}")
PY
    for t in 1 2 3; do python3 -c "import json;print(f'  t$t response:',repr(json.load(open('$ad/t$t.json'))['choices'][0]['message']['content'][:70]))"; done
  } | tee -a "$D/verdicts.txt"
  ws_sample "$arm-post"
  slot_delete_q "t4-$arm"
done
set_flag none
python3 - "$D" <<'PY'
import hashlib, json, sys
d = sys.argv[1]
def sha(a, t):
    try:
        return hashlib.sha256(json.load(open(f"{d}/{a}/t{t}.json"))["choices"][0]["message"]["content"].encode()).hexdigest()[:12]
    except Exception:
        return "ERR"
for t in (1, 2, 3):
    a, b = sha("mcontrol", t), sha("mzero", t)
    print(f"MULTI t{t} equal={a == b} control={a} zero={b}")
PY
echo "== T4c done"