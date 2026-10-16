#!/usr/bin/env bash
# T5-longdeep.sh — closes documented gap 1 on the SEALED candidate
# (rev238 Final, no flags, MTP-on): (a) LONG generation (256 tok) post
# slot-restore in a NEW process vs fresh; (b) DEEP-PREFIX LONG-ANSWER
# (answer must reference the cipher stated ~200 tokens earlier and be
# long) fresh vs restore. The sealed prime path runs on every restore.
# Discriminators: drafted/acceptance per turn, kv_before/pf (real reuse),
# separated prefill/rebuild/decode times, WS, and text sha (equality is
# NECESSARY, never sufficient).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T5"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.238_x64__pj67f1fcj4n14"
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

build_hist() { # outfile — cipher seed + 3 filler turns (~200tok), then the deep long-answer ask
  local out="$1"
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
  python3 - "$D" "$out" <<'PY'
import json, sys
d, out = sys.argv[1], sys.argv[2]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
open(out, "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("h1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("h2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."},
    {"role": "assistant", "content": rp("h3")},
    {"role": "user", "content": "Write a 200-word operational briefing on the northern depot cipher from the first message: who issued it, how it is structured, and a step-by-step verification protocol. Start with the cipher itself."}]))
PY
}

arm() { # tag promptfile maxtok
  local tag="$1" promptfile="$2" maxtok="$3"
  local pf="$promptfile" ad="$D/$1"
  mkdir -p "$ad"
  restart_verified "$tag-fresh"
  if [[ "$tag" == restore ]]; then
    chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    slot_restore "$ad/restore.json" "t5-h" || fail "$tag restore"
  else
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  fi
  ws_sample "$tag-pre"
  chat "$ad/cont.json" "$maxtok" "$(cat "$promptfile")" || fail "$tag cont"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  {
    echo "== $tag"
    grep -oE "mtp: post-restore carry primed at pos=[0-9]+ \([^)]*\)" "$ad/slice.log" || echo "PRIME_LINE_ABSENT"
    grep -m1 -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$ad/slice.log" || true
    grep -m1 -oE "KV prefix reuse — kept [0-9]+ of [0-9]+ tokens" "$ad/slice.log" || true
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$ad/slice.log" | tail -1
    grep -oE "decode: [0-9.]+ tok/s" "$ad/slice.log" | tail -1 || true
    python3 -c "import json;d=json.load(open('$ad/cont.json'));c=d['choices'][0]['message']['content'];u=d['usage'];print(f\"CTOK={u['prompt_tokens']}/{u['completion_tokens']} sha={__import__('hashlib').sha256(c.encode()).hexdigest()[:12]}\")"
  } | tee -a "$D/verdicts.txt"
  ws_sample "$tag-post"
}

restart_verified "hist-build"
build_hist "$D/deep-cont.json"
slot_save "$D/save.json" "t5-h"
arm fresh "$D/deep-cont.json" 256
arm restore "$D/deep-cont.json" 256
slot_delete_q t5-h
python3 - "$D" <<'PY'
import hashlib, json, sys
d = sys.argv[1]
def sha(p):
    try:
        return hashlib.sha256(json.load(open(f"{d}/{p}/cont.json"))["choices"][0]["message"]["content"].encode()).hexdigest()[:12]
    except Exception:
        return "ERR"
fr, rs = sha("fresh"), sha("restore")
print(f"TEXT fresh={fr} restore={rs} equal={fr == rs}")
PY
echo "== T5 done"