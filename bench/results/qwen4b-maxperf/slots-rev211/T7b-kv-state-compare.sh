#!/usr/bin/env bash
# T7b-kv-state-compare.sh — state-level comparison without a new build
# (plan 009 G-L): slot files ARE the post-turn KV blob. Arms on rev241:
#   rt        restore t7b then save it back with no chat  (blob roundtrip
#             fidelity — must be byte-identical to the source)
#   freshcont chat cont in a fresh process, then save        (fresh state)
#   restorecont restore t7b, chat cont, then save           (restore state)
# Byte-compares the slot files (same tokens => any difference is KV blob
# bytes) and reports the first differing offset + counts.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T7b"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.241_x64__pj67f1fcj4n14"
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

# same deep-history build sequence as T7 (deterministic token content)
restart_verified "hist"
chat "$D/h1.json" 16 "$P1"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
r1 = json.load(open(f"{d}/h1.json"))["choices"][0]["message"]["content"]
open(f"{d}/h2m.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."}]))
PY
chat "$D/h2.json" 48 "$(cat "$D/h2m.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
open(f"{d}/h3m.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("h1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("h2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."}]))
PY
chat "$D/h3.json" 48 "$(cat "$D/h3m.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
open(f"{d}/deep-cont.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("h1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("h2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."},
    {"role": "assistant", "content": rp("h3")},
    {"role": "user", "content": "Write a short operational briefing on the northern depot cipher from the first message, then the sorted-merge summary, then RoPE in two lines."}]))
PY
slot_save "$D/src-save.json" "t7b"
./scripts/deploy.sh fetch-file "$PFN" "t7b.kv" "$D/src.kv" "slots" >/dev/null 2>&1
echo "source slot fetched: $(python3 -c "import json;print(json.load(open('$D/src-save.json'))['tokens'], json.load(open('$D/src-save.json'))['bytes'])")" | tee -a "$D/verdicts.txt"
CONT="$(cat "$D/deep-cont.json")"

# rt: restore then save back, no chat
restart_verified "rt"
chat "$D/rt-pre.json" 1 '[{"role":"user","content":"x"}]'
slot_restore "$D/rt-restore.json" "t7b" || fail "rt restore"
slot_save "$D/rt-save.json" "t7b-rt" || fail "rt save"
./scripts/deploy.sh fetch-file "$PFN" "t7b-rt.kv" "$D/rt.kv" "slots" >/dev/null 2>&1
echo "rt done" | tee -a "$D/verdicts.txt"
slot_delete_q t7b-rt

# freshcont: fresh process, cont, save
restart_verified "freshcont"
chat "$D/fc-cont.json" 32 "$CONT" || fail "freshcont"
slot_save "$D/fc-save.json" "t7b-fc" || fail "freshcont save"
./scripts/deploy.sh fetch-file "$PFN" "t7b-fc.kv" "$D/fc.kv" "slots" >/dev/null 2>&1
echo "freshcont done: $(usage_of "$D/fc-cont.json")" | tee -a "$D/verdicts.txt"
slot_delete_q t7b-fc

# restorecont: new process, restore, cont, save
restart_verified "restorecont"
chat "$D/rc-pre.json" 1 '[{"role":"user","content":"x"}]'
slot_restore "$D/rc-restore.json" "t7b" || fail "restorecont restore"
chat "$D/rc-cont.json" 32 "$CONT" || fail "restorecont cont"
slot_save "$D/rc-save.json" "t7b-rc" || fail "restorecont save"
./scripts/deploy.sh fetch-file "$PFN" "t7b-rc.kv" "$D/rc.kv" "slots" >/dev/null 2>&1
echo "restorecont done: $(usage_of "$D/rc-cont.json")" | tee -a "$D/verdicts.txt"
slot_delete_q t7b-rc
slot_delete_q t7b

python3 - "$D" <<'PY'
import hashlib, pathlib, sys
d = pathlib.Path(sys.argv[1])
def firstdiff(p, q):
    a, b = p.read_bytes(), q.read_bytes()
    if a == b:
        return "identical", len(a), 0
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            nd = sum(1 for j in range(i, n) if a[j] != b[j])
            return i, n, nd
    return n, n, 0
pairs = [("src.kv", "rt.kv", "ROUNDTRIP src_vs_restored-then-saved"),
         ("fc.kv", "rc.kv", "STATE freshcont_vs_restorecont"),
         ("src.kv", "fc.kv", "PROVENANCE srcslot_vs_freshcont")]
for x, y, tag in pairs:
    p, q = d / x, d / y
    if not p.exists() or not q.exists():
        print(f"{tag}: MISSING {x}={p.exists()} {y}={q.exists()}")
        continue
    off, n, nd = firstdiff(p, q)
    sa = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    sb = hashlib.sha256(q.read_bytes()).hexdigest()[:16]
    print(f"{tag}: sizes={p.stat().st_size}/{q.stat().st_size} sha={sa}/{sb} first_diff_offset={off} differing_bytes={nd}")
PY
echo "== T7b done"
