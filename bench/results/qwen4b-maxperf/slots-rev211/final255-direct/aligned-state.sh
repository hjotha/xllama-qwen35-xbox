#!/usr/bin/env bash
# T15 five-file gate adapted for rev255 Final/LTCG; not Final acceptance.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/aligned-state"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14"
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
source "$REPO/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/T2-lib.sh"
: > "$D/verdicts.txt"
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"

restart_verified "build"
chat "$D/b1.json" 16 "$P1"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
r1 = json.load(open(f"{d}/b1.json"))["choices"][0]["message"]["content"]
open(f"{d}/T2.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."}]))
PY
chat "$D/b2.json" 48 "$(cat "$D/T2.json")"
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
r1 = json.load(open(f"{d}/b1.json"))["choices"][0]["message"]["content"]
r2 = json.load(open(f"{d}/b2.json"))["choices"][0]["message"]["content"]
open(f"{d}/TAIL.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": r2},
    {"role": "user", "content": "Repeat the cipher back exactly, then name the topics we discussed."}]))
PY
slot_save "$D/save.json" "t15"
echo "slot: $(python3 -c "import json;d=json.load(open('$D/save.json'));print(d['tokens'],'tokens',d['bytes'],'bytes')")" | tee -a "$D/verdicts.txt"
TAIL="$(cat "$D/TAIL.json")"
fetch_all() { local ad="$1"; mkdir -p "$ad"
  for f in chat-dump.logits chat-dump.tokens chat-dump.kv chat-dump.mirror chat-dump.mtp; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done; }
restart_verified "R"
mkdir -p "$D/R"
chat "$D/R-pre.json" 1 '[{"role":"user","content":"x"}]'
TR0=$(date -u +%s%N); slot_restore "$D/R-restore.json" "t15" || fail "R restore"; TR1=$(date -u +%s%N)
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/R-tail.json" 24 "$TAIL" || fail "R tail"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/R-slice.log"
fetch_all "$D/R"
{ echo "R: restore_api_ms=$(( (TR1 - TR0) / 1000000 )) $(grep -oE 'turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+' "$D/R-slice.log" | tail -1) $(grep -oE 'prefill=[0-9.]+ms \([0-9]+ tok\)' "$D/R-slice.log" | tail -1) $(grep -oE 'drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+' "$D/R-slice.log" | tail -1) $(grep -oE 'MTP_STATS rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+' "$D/R-slice.log" | tail -1)"; } | tee -a "$D/verdicts.txt"
restart_verified "C"
mkdir -p "$D/C"
chat "$D/C1.json" 16 "$P1"
chat "$D/C2.json" 48 "$(cat "$D/T2.json")"
LC=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/C-tail.json" 24 "$TAIL" || fail "C tail"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((LC + 1))" > "$D/C-slice.log"
fetch_all "$D/C"
{ echo "C: $(grep -oE 'turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+' "$D/C-slice.log" | tail -1) $(grep -oE 'prefill=[0-9.]+ms \([0-9]+ tok\)' "$D/C-slice.log" | tail -1) $(grep -oE 'drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+' "$D/C-slice.log" | tail -1) $(grep -oE 'MTP_STATS rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+' "$D/C-slice.log" | tail -1)"; } | tee -a "$D/verdicts.txt"
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t15"}' >/dev/null || true
python3 - "$D" <<'PY'
import hashlib, json, pathlib, struct, sys
d = pathlib.Path(sys.argv[1])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:20] if p.exists() else "MISSING"
for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.kv", "chat-dump.mirror", "chat-dump.mtp"):
    c, r = sha(d / "C" / f), sha(d / "R" / f)
    print(f"{f}: C={c} R={r} BYTE_EQUAL={c == r}")
for arm in ("C", "R"):
    h = struct.unpack("<13I", (d / arm / "chat-dump.mirror").read_bytes()[:52])
    print(f"mirror {arm}: cells={h[3]} pos_min={h[4]}")
a = json.load(open(d / "C-tail.json"))["choices"][0]["message"]["content"]
b = json.load(open(d / "R-tail.json"))["choices"][0]["message"]["content"]
print("tail text equal:", a == b)
PY
python3 - "$D" <<'CHECK'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
for f in ('save.json','R-restore.json'):
    j=json.loads((p/f).read_text()); assert j['ok'] is True and j['tokens']==115,(f,j)
for name in ('chat-dump.logits','chat-dump.tokens','chat-dump.kv','chat-dump.mirror','chat-dump.mtp'):
    assert (p/'C'/name).read_bytes() == (p/'R'/name).read_bytes(), name
assert json.loads((p/'C-tail.json').read_text())['choices'][0]['message']['content'] == json.loads((p/'R-tail.json').read_text())['choices'][0]['message']['content']
print('PASS: five full files byte-identical and continuation text identical')
CHECK
echo "== rev255 Final/LTCG aligned-state gate done "
