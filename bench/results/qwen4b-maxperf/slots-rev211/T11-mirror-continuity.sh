#!/usr/bin/env bash
# T11-mirror-continuity.sh — plan 009 MTP-mirror investigation on rev242.
# The mirror blob header decodes the VALID region (u32 x13):
#   field[3]=cells, field[4]=pos_min  (C/F: 230 cells @0; R: 43 cells @187)
# so the partial mirror is proven by the blob itself. This script:
#  (1) C vs R aligned comparison at the SAME point (tail turn), including
#      the pending row (=h_rows[n-1], byte-equal in T10 => the target's
#      hidden rows match; the mirror difference is the decoy's blast radius
#      inside the causal catch-up batch).
#  (2) MULTI-TURN continuity: t4 (tail) + t5 + t6 in the same process for
#      C and R, per-turn drafted/accept/discarded/rounds (acceptance,
#      rejection, rollback) and mirror/kv/logits dumps after each turn.
#  (3) H arm: fresh process, ONE 187-token turn then the tail as turn 2 —
#      decomposes turn-boundary count (C: 4 turns) from resident+tail
#      structure (H: 2 turns) for the F/C explanation.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T11"
T10="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T10"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.242_x64__pj67f1fcj4n14"
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
printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"

# fixed prompts (rebuilt from the T10 artifacts so both arms see identical input)
R1=$(content_of "$T10/t1.json"); R2=$(content_of "$T10/t2.json"); R3=$(content_of "$T10/t3.json")
python3 - "$D" "$R1" "$R2" "$R3" <<'PY'
import json, sys
d, r1, r2, r3 = sys.argv[1:5]
msgs = lambda: [
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": r2},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."},
    {"role": "assistant", "content": r3}]
TAIL = msgs() + [{"role": "user", "content": "Write a short operational briefing on the northern depot cipher from the first message, then the sorted-merge summary, then RoPE in two lines."}]
open(f"{d}/T1.json", "w").write(json.dumps([msgs()[0]]))
open(f"{d}/T2.json", "w").write(json.dumps(msgs()[:3]))
open(f"{d}/T3.json", "w").write(json.dumps(msgs()[:5]))  # ends with the ask3 USER turn
open(f"{d}/TAIL.json", "w").write(json.dumps(TAIL))
PY
T4=$(cat "$D/TAIL.json")

fetch() { # ad tag
  local ad="$1" tag="$2"
  mkdir -p "$ad"
  for f in chat-dump.logits chat-dump.tokens chat-dump.kv chat-dump.mirror chat-dump.mtp; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$tag-$f" >/dev/null 2>&1 || echo "MISSING $tag/$f" | tee -a "$D/verdicts.txt"
  done
}
metrics() { # ad tag slicename
  local sl="$1/$3"
  grep -m1 -oE "S3: state dumped — kv_bytes=[0-9]+ mirror_bytes=[0-9]+ mirror_pos=[0-9-]+ pending=[01] kv_tokens=[0-9]+" "$sl" || echo "STATE_LINE_MISSING"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$sl" | tail -1 || true
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$sl" | tail -1 || true
  grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$sl" | tail -1 || true
  grep -cE "mtp: catch-up seq_rm refused|abort: mid-round trim|stop sequence after" "$sl" || true
}

# ---- P: producer (same trajectory) -> slot
restart_verified "P"
chat "$D/p1.json" 16 "$P1"
chat "$D/p2.json" 48 "$(cat "$D/T2.json")"
chat "$D/p3.json" 48 "$(cat "$D/T3.json")"
slot_save "$D/save.json" "t11"
R3L=$(content_of "$D/p3.json")
# t5/t6 prompts built from the ACTUAL tail reply
T4R=$(content_of "$D/pafter.json" 2>/dev/null || true)
python3 - "$D" "$T4" "$R3L" <<'PY'
import json, sys
d, t4, r3 = sys.argv[1], sys.argv[2], sys.argv[3]
# tail prompt is T4 (fixed); the t5/t6 prompts are built after the tail reply
open(f"{d}/_t4.json", "w").write(t4)
PY
echo "P done" | tee -a "$D/verdicts.txt"

# ---- C: control — replay trajectory, then t4/t5/t6
restart_verified "C"
chat "$D/c1.json" 16 "$P1"
chat "$D/c2.json" 48 "$(cat "$D/T2.json")"
chat "$D/c3.json" 48 "$(cat "$D/T3.json")"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/c4.json" 32 "$T4" || fail "C t4"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/c4-slice.log"
fetch "$D/C" t4; metrics "$D" "C" "c4-slice.log" | tee -a "$D/verdicts.txt"
C4=$(content_of "$D/c4.json")
python3 - "$D" "$T4" "$C4" <<'PY'
import json, sys
d, t4, c4 = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.loads(t4) + [{"role": "assistant", "content": c4},
                      {"role": "user", "content": "Summarize each of the three topics we discussed in one sentence, starting with the cipher."}]
open(f"{d}/C5.json", "w").write(json.dumps(m))
PY
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/c5.json" 48 "$(cat "$D/C5.json")" || fail "C t5"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/c5-slice.log"
fetch "$D/C" t5; metrics "$D" "C" "c5-slice.log" | tee -a "$D/verdicts.txt"
C5=$(content_of "$D/c5.json")
python3 - "$D" "$(cat "$D/C5.json")" "$C5" <<'PY'
import json, sys
d, c5, r5 = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.loads(c5) + [{"role": "assistant", "content": r5},
                      {"role": "user", "content": "Now: state the cipher, then spell it backwards."}]
open(f"{d}/C6.json", "w").write(json.dumps(m))
PY
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/c6.json" 24 "$(cat "$D/C6.json")" || fail "C t6"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/c6-slice.log"
fetch "$D/C" t6; metrics "$D" "C" "c6-slice.log" | tee -a "$D/verdicts.txt"

# ---- R: restore the SAME slot, then the SAME t4/t5/t6
restart_verified "R"
mkdir -p "$D/R"
chat "$D/r-pre.json" 1 '[{"role":"user","content":"x"}]'
slot_restore "$D/r-restore.json" "t11" || fail "R restore"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/r4.json" 32 "$T4" || fail "R t4"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/r4-slice.log"
fetch "$D/R" t4; metrics "$D" "R" "r4-slice.log" | tee -a "$D/verdicts.txt"
R4=$(content_of "$D/r4.json")
python3 - "$D" "$T4" "$R4" <<'PY'
import json, sys
d, t4, r4 = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.loads(t4) + [{"role": "assistant", "content": r4},
                      {"role": "user", "content": "Summarize each of the three topics we discussed in one sentence, starting with the cipher."}]
open(f"{d}/R5.json", "w").write(json.dumps(m))
PY
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/r5.json" 48 "$(cat "$D/R5.json")" || fail "R t5"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/r5-slice.log"
fetch "$D/R" t5; metrics "$D" "R" "r5-slice.log" | tee -a "$D/verdicts.txt"
R5=$(content_of "$D/r5.json")
python3 - "$D" "$(cat "$D/R5.json")" "$R5" <<'PY'
import json, sys
d, r5, rr5 = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.loads(r5) + [{"role": "assistant", "content": rr5},
                      {"role": "user", "content": "Now: state the cipher, then spell it backwards."}]
open(f"{d}/R6.json", "w").write(json.dumps(m))
PY
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/r6.json" 24 "$(cat "$D/R6.json")" || fail "R t6"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/r6-slice.log"
fetch "$D/R" t6; metrics "$D" "R" "r6-slice.log" | tee -a "$D/verdicts.txt"

# ---- H: fresh, ONE 187-token turn, then the tail as turn 2 (decomposition)
restart_verified "H"
mkdir -p "$D/H"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/h1.json" 8 "$(cat "$D/T3.json")" || fail "H t1"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/h1-slice.log"
H1=$(content_of "$D/h1.json")
python3 - "$D" "$(cat "$D/T3.json")" "$H1" "$R3L" <<'PY'
import json, sys
d, t3, h1, r3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# H's first turn generated h1; to keep the SAME tail prompt the history must
# match, so H's t2 uses the PRODUCER's r3 as the assistant turn (H's own h1
# is a 1-token generation and is replaced by the reference trajectory's r3
# via restore of the SAME slot instead) -> see H2 below.
print("h1:", h1[:40])
PY
# H2: restore the producer slot in the SAME process gives the identical
# history (replacing h1's 1-token divergence) with the tail as turn 2
slot_restore "$D/h2-restore.json" "t11" || fail "H restore"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/h2.json" 32 "$T4" || fail "H t2"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/h2-slice.log"
fetch "$D/H" t2; metrics "$D" "H" "h2-slice.log" | tee -a "$D/verdicts.txt"

./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t11"}' >/dev/null || true
echo "== compare" | tee -a "$D/verdicts.txt"
python3 - "$D" "$T10" <<'PY'
import hashlib, pathlib, struct, sys
d, t10 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
def mirror_meta(p):
    b = pathlib.Path(p).read_bytes()
    h = struct.unpack("<13I", b[:52])
    return h[3], h[4], len(b)   # cells, pos_min, bytes
# per-turn C vs R
for tag in ("t4", "t5", "t6"):
    for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.kv", "chat-dump.mirror", "chat-dump.mtp"):
        c, r = sha(d / "C" / f"{tag}-{f}"), sha(d / "R" / f"{tag}-{f}")
        print(f"{tag} {f}: C={c[:16]} R={r[:16]} EQUAL={c==r}")
# mirror region per turn
for arm in ("C", "R"):
    for tag in ("t4", "t5", "t6"):
        p = d / arm / f"{tag}-chat-dump.mirror"
        if p.exists():
            cells, pos_min, n = mirror_meta(p)
            print(f"mirror {arm}/{tag}: cells={cells} pos_min={pos_min} bytes={n} rows={(n-52)//4120}")
# H vs C vs F at the tail turn
for arm, path in (("H_t2", d / "H" / "t2-chat-dump.kv"), ("C_t4", d / "C" / "t4-chat-dump.kv")):
    print(f"KV {arm}: sha={sha(path)[:24]}")
print("KV C_t4 vs H_t2:", sha(d / "C" / "t4-chat-dump.kv") == sha(d / "H" / "t2-chat-dump.kv"))
print("KV C_t4 vs T10 C_t4:", sha(d / "C" / "t4-chat-dump.kv") == sha(t10 / "C" / "chat-dump.kv"))
PY
echo "== T11 done"
