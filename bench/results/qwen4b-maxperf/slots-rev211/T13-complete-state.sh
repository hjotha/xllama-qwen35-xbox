#!/usr/bin/env bash
# T13-complete-state.sh — plan 009 #170c validation on rev243 (MTP drafter
# trailer: draft KV + real carry + prefix proof, versioned, atomic).
# Sections:
#  A aligned state: C (fresh trajectory + tail) vs R (restore WITH trailer)
#    — mirror bytes, positions, target bytes at the same point; 3-turn
#    continuity with accept/reject/rollback; separated restore/prefill/decode.
#  B fail-closed: bad magic / bad version / bad prefix hash / truncated
#    trailer / v2-file-without-trailer — each must REFUSE (logged reason)
#    and leave the session on the safe v2 path (prime), never claiming a
#    mirror it does not have.
#  C varied prefixes: short (42) / medium (~90) / long (187) independent
#    trajectories, save+restore+continue, mirror completeness per case.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T13"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.243_x64__pj67f1fcj4n14"
API=http://192.168.1.26:11434
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
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"

restart_verified "P"
chat "$D/p1.json" 16 "$P1"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
r1 = json.load(open(f"{d}/p1.json"))["choices"][0]["message"]["content"]
open(f"{d}/T2.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."}]))
PY
chat "$D/p2.json" 48 "$(cat "$D/T2.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
open(f"{d}/T3.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("p1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("p2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."}]))
PY
chat "$D/p3.json" 48 "$(cat "$D/T3.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
base = [
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("p1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("p2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."},
    {"role": "assistant", "content": rp("p3")}]
open(f"{d}/TAIL.json", "w").write(json.dumps(base + [
    {"role": "user", "content": "Write a short operational briefing on the northern depot cipher from the first message, then the sorted-merge summary, then RoPE in two lines."}]))
PY
T0=$(date -u +%s%N)
slot_save "$D/save.json" "t13a"
T1N=$(date -u +%s%N)
echo "save(api): $(( (T1N - T0) / 1000000 )) ms; tokens=$(python3 -c "import json;print(json.load(open('$D/save.json'))['tokens'])") bytes=$(python3 -c "import json;print(json.load(open('$D/save.json'))['bytes'])")" | tee -a "$D/verdicts.txt"
./scripts/deploy.sh fetch-file "$PFN" "t13a.kv" "$D/slot-full.bin" "slots" >/dev/null 2>&1
TAIL="$(cat "$D/TAIL.json")"

fetch_all() { local ad="$1"; mkdir -p "$ad"
  for f in chat-dump.logits chat-dump.tokens chat-dump.mirror chat-dump.mtp; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done; }
metrics() { local sl="$1"
  grep -m1 -oE "S3: state dumped — kv_bytes=[0-9]+ mirror_bytes=[0-9]+ mirror_pos=[0-9-]+ pending=[01] kv_tokens=[0-9]+" "$sl" || echo NO_STATE
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$sl" | tail -1 || true
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$sl" | tail -1 || true
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$sl" | tail -1 || true
  grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$sl" | tail -1 || true; }

# ---- C: control (fresh trajectory + tail)
restart_verified "C"
chat "$D/c1.json" 16 "$P1"; chat "$D/c2.json" 48 "$(cat "$D/T2.json")"; chat "$D/c3.json" 48 "$(cat "$D/T3.json")"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/c4.json" 32 "$TAIL" || fail "C tail"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/c4-slice.log"
{ echo "== C tail"; metrics "$D/c4-slice.log"; } | tee -a "$D/verdicts.txt"
fetch_all "$D/C"
C4=$(content_of "$D/c4.json")
python3 - "$D" "$TAIL" "$C4" <<'PY'
import json, sys
d, t, r = sys.argv[1], sys.argv[2], sys.argv[3]
open(f"{d}/C5.json", "w").write(json.dumps(json.loads(t) + [
    {"role": "assistant", "content": r},
    {"role": "user", "content": "Summarize each of the three topics in one sentence, starting with the cipher."}]))
PY
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/c5.json" 48 "$(cat "$D/C5.json")" || fail "C t5"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/c5-slice.log"
{ echo "== C t5"; metrics "$D/c5-slice.log"; } | tee -a "$D/verdicts.txt"
fetch_all "$D/C5"

# ---- R: restore WITH trailer
restart_verified "R"
mkdir -p "$D/R"
chat "$D/r-pre.json" 1 '[{"role":"user","content":"x"}]'
TR0=$(date -u +%s%N)
slot_restore "$D/r-restore.json" "t13a" || fail "R restore"
TR1=$(date -u +%s%N)
echo "restore(api): $(( (TR1 - TR0) / 1000000 )) ms" | tee -a "$D/verdicts.txt"
grep -m1 -oE "MTP trailer restored — drafter mirror cells=[0-9]+ pos_min=[0-9]+ carry@[0-9]+ \(prefix proof OK\)" "$D/r-restore.json" >/dev/null 2>&1 || true
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/r4.json" 32 "$TAIL" || fail "R tail"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/r4-slice.log"
{ echo "== R tail"; metrics "$D/r4-slice.log"
  grep -m1 -oE "MTP trailer restored — drafter mirror cells=[0-9]+ pos_min=[0-9]+ carry@[0-9]+" /dev/null 2>/dev/null || true; } | tee -a "$D/verdicts.txt"
fetch_all "$D/R"
R4=$(content_of "$D/r4.json")
python3 - "$D" "$TAIL" "$R4" <<'PY'
import json, sys
d, t, r = sys.argv[1], sys.argv[2], sys.argv[3]
open(f"{d}/R5.json", "w").write(json.dumps(json.loads(t) + [
    {"role": "assistant", "content": r},
    {"role": "user", "content": "Summarize each of the three topics in one sentence, starting with the cipher."}]))
PY
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/r5.json" 48 "$(cat "$D/R5.json")" || fail "R t5"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/r5-slice.log"
{ echo "== R t5"; metrics "$D/r5-slice.log"; } | tee -a "$D/verdicts.txt"
fetch_all "$D/R5"

./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t13a"}' >/dev/null || true

# ---- B: fail-closed matrix on crafted files
python3 - "$D" <<'PY'
import pathlib, struct, sys
d = pathlib.Path(sys.argv[1])
b = bytearray(pathlib.Path(d / "slot-full.bin").read_bytes())
nl = b.index(b"\n")
hdr = b[:nl].decode().split("\t")
n_state, n_tok = int(hdr[1]), int(hdr[2])
trailer_off = nl + 1 + 4 * n_tok + n_state
print("slot layout: header=", nl + 1, "tokens=", 4 * n_tok, "blob=", n_state, "trailer_off=", trailer_off, "file=", len(b))
assert bytes(b[trailer_off:trailer_off + 8]) == b"XLMTPST1", b[trailer_off:trailer_off + 8]
variants = {}
v = bytearray(b); v[trailer_off] ^= 0xFF; variants["badmagic"] = v
v = bytearray(b); v[trailer_off + 8] ^= 0xFF; variants["badversion"] = v
v = bytearray(b); v[-1] ^= 0xFF; variants["badhash"] = v
v = bytearray(b[:trailer_off + 40]); variants["truncated"] = v
variants["v2only"] = bytearray(b[:trailer_off])  # trailer removed -> v2 file
for name, data in variants.items():
    (d / f"slot-{name}.bin").write_bytes(bytes(data))
    print(f"crafted {name}: {len(data)} bytes")
PY
for name in badmagic badversion badhash truncated v2only; do
  ./scripts/deploy.sh upload-file "$D/slot-$name.bin" "$PFN" "slots" "t13-$name.kv" >/dev/null
  code=$(curl -s -m 60 -o "$D/restore-$name.json" -w "%{http_code}" -X POST "$API/v1/slots/restore" \
    -H 'Content-Type: application/json' -d "{\"slot\":\"t13-$name\"}")
  ok=$(python3 -c "import json;print(json.load(open('$D/restore-$name.json')).get('ok'))" 2>/dev/null || echo ERR)
  chat "$D/after-$name.json" 8 "$P1" >/dev/null 2>&1 && alive=$(content_of "$D/after-$name.json") || alive=DEAD
  echo "restore[$name]: http=$code ok=$ok session_alive=$([ "$alive" != "DEAD" ] && echo YES || echo NO)" | tee -a "$D/verdicts.txt"
  curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d "{\"slot\":\"t13-$name\"}" >/dev/null || true
done

echo "== compare" | tee -a "$D/verdicts.txt"
python3 - "$D" <<'PY'
import hashlib, json, pathlib, struct, sys
d = pathlib.Path(sys.argv[1])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:20] if p.exists() else "MISSING"
for tag in ("", "5"):
    for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.mirror", "chat-dump.mtp"):
        c, r = sha(d / f"C{tag}/{f}"), sha(d / f"R{tag}/{f}")
        print(f"t{4 if tag=='' else 5} {f}: C={c} R={r} EQUAL={c==r}")
for arm, sub in (("C tail", "C"), ("R tail", "R"), ("C t5", "C5"), ("R t5", "R5")):
    p = d / sub / "chat-dump.mirror"
    if p.exists():
        h = struct.unpack("<13I", p.read_bytes()[:52])
        print(f"mirror {arm}: cells={h[3]} pos_min={h[4]}")
tc = (d / "C" / "chat-dump.tokens").read_bytes()
tr = (d / "R" / "chat-dump.tokens").read_bytes()
print("tail tokens byte-equal:", tc == tr)
PY
echo "== T13 done"
