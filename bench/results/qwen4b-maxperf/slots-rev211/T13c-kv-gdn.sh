#!/usr/bin/env bash
# T13c-kv-gdn.sh — the remaining T13 gate: strict byte equality of the
# TARGET KV blob (which carries every cache tensor, the recurrent/GDN cells
# included) between a fresh trajectory and a restore WITH the MTP trailer,
# on rev243. Medium prefix; the five artifact classes compared at the same
# point (post-prefill, pre-decode): logits, token ids, mirror, pending row,
# target KV.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T13c"
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
open(f"{d}/TAIL.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "user", "content": "Repeat the cipher back exactly, then name the topics we discussed."}]))
PY
chat "$D/b2.json" 48 "$(cat "$D/T2.json")"
slot_save "$D/save.json" "t13c"
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
print("tail prompt fixed")
PY
TAIL="$(cat "$D/TAIL.json")"
echo "slot: $(python3 -c "import json;d=json.load(open('$D/save.json'));print(d['tokens'],'tokens',d['bytes'],'bytes')")" | tee -a "$D/verdicts.txt"

run_arm() { # tag restore
  local tag="$1" do_restore="$2"
  local ad="$D/$1"
  mkdir -p "$ad"
  restart_verified "$tag"
  if [[ "$do_restore" == "1" ]]; then
    chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
    TR0=$(date -u +%s%N); slot_restore "$ad/restore.json" "t13c" || fail "$tag restore"; TR1=$(date -u +%s%N)
    echo "[$tag] restore_api_ms=$(( (TR1 - TR0) / 1000000 ))" | tee -a "$D/verdicts.txt"
  else
    # ALIGNED control: replay the exact same trajectory, then the same tail.
    chat "$ad/b1.json" 16 "$P1"
    chat "$ad/b2.json" 48 "$(cat "$D/T2.json")"
  fi
  local L0; L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  chat "$ad/tail.json" 24 "$TAIL" || fail "$tag tail"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  for f in chat-dump.logits chat-dump.tokens chat-dump.mirror chat-dump.mtp chat-dump.kv; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done
  { echo "[$tag] $(grep -oE 'turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+' "$ad/slice.log" | tail -1)"
    echo "[$tag] $(grep -oE 'prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]' "$ad/slice.log" | tail -1)"
    echo "[$tag] $(grep -oE 'drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+' "$ad/slice.log" | tail -1)"
    echo "[$tag] $(grep -oE 'MTP_STATS rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+ draft_ms=[0-9.]+ sample_ms=[0-9.]+ topprob_ms=[0-9.]+ catchup_ms=[0-9.]+ verify_ms=[0-9.]+' "$ad/slice.log" | tail -1)"
    echo "[$tag] kv_dump=$(stat -c%s "$ad/chat-dump.kv" 2>/dev/null || echo MISSING) mirror_dump=$(stat -c%s "$ad/chat-dump.mirror" 2>/dev/null || echo MISSING)"; } | tee -a "$D/verdicts.txt"
}
run_arm C 0
run_arm R 1
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t13c"}' >/dev/null || true
python3 - "$D" <<'PY'
import hashlib, pathlib, struct, sys
d = pathlib.Path(sys.argv[1])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:20] if p.exists() else "MISSING"
for f in ("chat-dump.kv", "chat-dump.logits", "chat-dump.tokens", "chat-dump.mirror", "chat-dump.mtp"):
    c, r = sha(d / "C" / f), sha(d / "R" / f)
    print(f"{f}: C={c} R={r} BYTE_EQUAL={c == r}")
for arm in ("C", "R"):
    p = d / arm / "chat-dump.mirror"
    h = struct.unpack("<13I", p.read_bytes()[:52])
    print(f"mirror {arm}: cells={h[3]} pos_min={h[4]}")
print("tail text equal:", (d / "C" / "tail.json").read_text() and
      __import__("json").load(open(d / "C" / "tail.json"))["choices"][0]["message"]["content"] ==
      __import__("json").load(open(d / "R" / "tail.json"))["choices"][0]["message"]["content"])
PY
echo "== T13c done"
