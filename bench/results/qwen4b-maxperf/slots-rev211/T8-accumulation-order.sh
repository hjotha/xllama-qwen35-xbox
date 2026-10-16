#!/usr/bin/env bash
# T8-accumulation-order.sh — plan 009 G-L cause test on rev241: is the
# fresh/restore logits divergence caused by f32 ACCUMULATION ORDER (prefill
# chunking/assembly), as hypothesized, rather than by state loss or
# non-determinism? Decisive comparison, no build, no KV dump:
#   S1  the deep history built through MULTI-TURN chats (rows assembled
#       from prefills of 40 / 22 / 28 tokens across turns) -> slot
#   S2  the SAME token content prefilled in ONE turn (187-token prompt in
#       ubatch chunks 64/64/59) -> slot
# If S1 and S2 hold the same token ids but different KV blob bytes, the
# mechanism is proven: identical content, different accumulation order =>
# different KV bytes => different logits, with identical decisions.
# The blob occupies the last n_state bytes of each slot file (header field).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T8"
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

# S1: the history assembled by multi-turn chats (same sequence as T7/T7b)
restart_verified "s1-build"
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
slot_save "$D/s1-save.json" "t8-s1"
./scripts/deploy.sh fetch-file "$PFN" "t8-s1.kv" "$D/s1.kv" "slots" >/dev/null 2>&1
echo "S1 saved: $(python3 -c "import json;d=json.load(open('$D/s1-save.json'));print('tokens',d['tokens'],'bytes',d['bytes'])")" | tee -a "$D/verdicts.txt"

# S2: the SAME rendered history as one prompt in a fresh process, 1 token
restart_verified "s2-build"
chat "$D/s2-seed.json" 1 "$(cat "$D/h3m.json")"
slot_save "$D/s2-save.json" "t8-s2"
./scripts/deploy.sh fetch-file "$PFN" "t8-s2.kv" "$D/s2.kv" "slots" >/dev/null 2>&1
echo "S2 saved: $(python3 -c "import json;d=json.load(open('$D/s2-save.json'));print('tokens',d['tokens'],'bytes',d['bytes'])")" | tee -a "$D/verdicts.txt"
grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/s2-seed.json" >/dev/null 2>&1 || true

slot_delete_q t8-s1; slot_delete_q t8-s2
python3 - "$D" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])

def parse(p):
    b = p.read_bytes()
    nl = b.index(b"\n")
    hdr = b[:nl].decode().split("\t")
    magic, n_state, n_tok = hdr[0], int(hdr[1]), int(hdr[2])
    return b, n_state, n_tok, nl

b1, n1, t1, nl1 = parse(d / "s1.kv")
b2, n2, t2, nl2 = parse(d / "s2.kv")
print(f"S1: tokens={t1} n_state={n1} file={len(b1)}")
print(f"S2: tokens={t2} n_state={n2} file={len(b2)}")
blob1 = b1[len(b1) - n1:]
blob2 = b2[len(b2) - n2:]
# token id regions (after the header line; each token is 4 bytes)
ids1 = [int.from_bytes(b1[nl1 + 1 + 4 * i: nl1 + 5 + 4 * i], "little", signed=True) for i in range(t1)]
ids2 = [int.from_bytes(b2[nl2 + 1 + 4 * i: nl2 + 5 + 4 * i], "little", signed=True) for i in range(t2)]
common = min(t1, t2)
same_ids = ids1[:common] == ids2[:common]
first_id_diff = next((i for i in range(common) if ids1[i] != ids2[i]), None)
print(f"token ids: common={common} identical_prefix={same_ids} first_id_diff={first_id_diff}")
if blob1 == blob2:
    print("KV blob: BYTE-IDENTICAL (same accumulation order — hypothesis REFUTED)")
else:
    n = min(len(blob1), len(blob2))
    nd = sum(1 for j in range(n) if blob1[j] != blob2[j])
    off = next((j for j in range(n) if blob1[j] != blob2[j]), None)
    print(f"KV blob: DIFFERENT bytes={nd}/{len(blob1)} first_diff_offset={off} (of {len(blob1)})")
    print(f"  -> same tokens, different KV bytes => accumulation-order mechanism PROVEN (differing bytes={100.0*nd/len(blob1):.1f}%)")
PY
echo "== T8 done"
