#!/usr/bin/env bash
# T11b-decomposition.sh — turn-boundary vs resident+tail decomposition on
# rev242 (T11 follow-up, no restore involved):
#   F   one turn, 230-token prompt            (single full prefill, 4 ubatches)
#   H2  fresh process: one 187-token turn, then the SAME tail as turn 2
#       (resident prefix + 43-token tail; total 2 turns)
#   C   the T11 control trajectory (4 turns: 40 + 20 + 80 + 43 tail)
# If H2_tail == C_t4 byte-wise, the number of turn boundaries does not
# change the bytes; if H2_tail != F_tail while C_tail == H2_tail, the
# resident+tail structure (not the boundary count) explains F/C.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T11b"
T11="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T11"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.242_x64__pj67f1fcj4n14"
cd "$REPO"
API=http://192.168.1.26:11434
set -a
source ~/.config/xllama/xbox-env
set +a
mkdir -p "$D"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
# shellcheck disable=SC1091
source "$D/../T2-lib.sh"
: > "$D/verdicts.txt"
printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"
T3="$(cat "$T11/T3.json")"
TAIL="$(cat "$T11/TAIL.json")"
R3_REF="$(content_of "$T11/p3.json")"

# H2: one 187-token turn (no restore), then the same tail as turn 2
restart_verified "H2"
mkdir -p "$D/H2"
chat "$D/h2-turn1.json" 48 "$T3" || fail "H2 turn1"
R3_H2="$(content_of "$D/h2-turn1.json")"
echo "H2 turn1 reply == producer r3: $([ "$R3_H2" = "$R3_REF" ] && echo YES || echo NO)" | tee -a "$D/verdicts.txt"
python3 - "$D" "$T3" "$R3_H2" "$R3_REF" <<'PY'
import json, sys
d, t3, r3_h2, r3_ref = sys.argv[1:5]
# tail prompt built with H2's own reply (equal to the producer's by determinism)
m = json.loads(t3) + [{"role": "assistant", "content": r3_h2},
                      {"role": "user", "content": "Write a short operational briefing on the northern depot cipher from the first message, then the sorted-merge summary, then RoPE in two lines."}]
open(f"{d}/H2-tail.json", "w").write(json.dumps(m))
PY
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/h2-turn2.json" 32 "$(cat "$D/H2-tail.json")" || fail "H2 turn2"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/h2-turn2-slice.log"
for f in chat-dump.logits chat-dump.tokens chat-dump.kv chat-dump.mirror chat-dump.mtp; do
  ./scripts/deploy.sh fetch-file "$PFN" "$f" "$D/H2/$f" >/dev/null 2>&1 || echo "MISSING H2/$f" | tee -a "$D/verdicts.txt"
done
{ echo "== H2 tail (turn 2)"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/h2-turn2-slice.log" | tail -1 || true
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/h2-turn2-slice.log" | tail -1 || true
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/h2-turn2-slice.log" | tail -1 || true; } | tee -a "$D/verdicts.txt"

./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
python3 - "$D" "$T11" "$REPO/bench/results/qwen4b-maxperf/slots-rev211/T10" <<'PY'
import hashlib, pathlib, sys
d, t11, t10 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
pairs = [
    ("H2_tail vs C_t4 (turn boundaries: 2 vs 4)", d / "H2" / "chat-dump.kv", t11 / "C" / "t4-chat-dump.kv"),
    ("H2_tail vs F (resident+tail vs single full prefill)", d / "H2" / "chat-dump.kv", t10 / "F" / "chat-dump.kv"),
    ("H2_tail vs C_t4 logits", d / "H2" / "chat-dump.logits", t11 / "C" / "t4-chat-dump.logits"),
    ("H2_tail vs F logits", d / "H2" / "chat-dump.logits", t11 / "F" / "t4-chat-dump.logits"),
    ("H2_tail tokens vs C_t4 tokens", d / "H2" / "chat-dump.tokens", t11 / "C" / "t4-chat-dump.tokens"),
]
for tag, a, b in pairs:
    print(f"{tag}: EQUAL={sha(a)==sha(b)} sha={sha(a)[:20]}")
import struct
for arm, p in (("H2", d / "H2" / "chat-dump.mirror"), ("C_t4", t11 / "C" / "t4-chat-dump.mirror")):
    if p.exists():
        h = struct.unpack("<13I", p.read_bytes()[:52])
        print(f"mirror {arm}: cells={h[3]} pos_min={h[4]}")
PY
echo "== T11b done"
