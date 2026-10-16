#!/usr/bin/env bash
# T6c-logits-byteprobe-deep.sh — S3 byte-probe, DEEP prefix (n=187) and a
# longer multi-token continuation (max_tokens=32), same capture (rev240
# Iteration, chat-dump.flag armed). Same question, harder case: the draft
# mirror hole spans 187 positions and the run is long enough to expose any
# per-step divergence. fresh full-prefill vs restore-in-new-process.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T6c"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.240_x64__pj67f1fcj4n14"
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

# build the deep history (~187 tokens) + the long continuation prompt
restart_verified "deep-build"
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
slot_save "$D/save.json" "t6c"
echo "seed saved: $(usage_of "$D/h3.json")" | tee -a "$D/verdicts.txt"
CONT="$(cat "$D/deep-cont.json")"

run_arm() { # tag do_restore
  local tag="$1" do_restore="$2"
  local ad="$D/$1"
  mkdir -p "$ad"
  restart_verified "$tag-new"
  if [[ "$do_restore" == "1" ]]; then
    chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    ws_sample "$tag-pre"
    slot_restore "$ad/restore.json" "t6c" || fail "$tag restore"
  else
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    ws_sample "$tag-pre"
  fi
  chat "$ad/cont.json" 32 "$CONT" || fail "$tag cont"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  {
    echo "== $tag"
    grep -m1 -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$ad/slice.log" || echo "no-restore"
    grep -oE "mtp: post-restore carry primed at pos=[0-9]+" "$ad/slice.log" || echo "no-prime"
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1 || true
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1 || true
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1 || true
    python3 -c "import json;d=json.load(open('$ad/cont.json'));u=d['usage'];print(f\"CTOK={u['prompt_tokens']}/{u['completion_tokens']} sha={__import__('hashlib').sha256(d['choices'][0]['message']['content'].encode()).hexdigest()[:16]}\")"
  } | tee -a "$D/verdicts.txt"
  for f in chat-dump.logits chat-dump.tokens chat-dump.tokens.prefill; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done
  ws_sample "$tag-post"
}
run_arm fresh 0
run_arm restore 1
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t6c"}' >/dev/null || true
echo "== compare" | tee -a "$D/verdicts.txt"
python3 - "$D" <<'PY'
import hashlib, pathlib, sys
d = pathlib.Path(sys.argv[1])
for f in ("chat-dump.logits", "chat-dump.tokens"):
    a, b = d / "fresh" / f, d / "restore" / f
    if a.exists() and b.exists():
        ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
        print(f"{f}: BYTE_EQUAL={ha == hb} sha={ha[:32]} bytes={a.stat().st_size}")
    else:
        print(f"{f}: MISSING")
pf = d / "fresh/chat-dump.tokens.prefill"
pr = d / "restore/chat-dump.tokens.prefill"
if pf.exists() and pr.exists():
    print(f"prefill_ids: fresh={sum(1 for _ in open(pf))} restore={sum(1 for _ in open(pr))} (reuse class, not a defect)")
PY
echo "== T6c done"
