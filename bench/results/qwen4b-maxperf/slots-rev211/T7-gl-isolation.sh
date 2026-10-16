#!/usr/bin/env bash
# T7-gl-isolation.sh — plan 009 G-L multi-ubatch isolation on rev241
# Iteration (S3 probe with KV/MTP state dumps). Matrix, all with the SAME
# prompt file and mode (model qwen35-4b-mtp, greedy, max_tokens=32):
#   F1/F2  fresh full-prefill, two new processes   (fresh determinism)
#   R1/R2  restore the SAME slot file, two new processes (restore determinism)
#   F1 vs R1 (the target comparison) and R1 vs R2 separate EXECUTION
#   non-determinism from STATE provenance. Compares bytes of the per-step
#   logits, sampled token ids, the post-prefill TARGET KV blob and the MTP
#   mirror meta; reports the first differing byte offset of the KV blob.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T7"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.241_x64__pj67f1fcj4n14"
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

# ONE deep-history build (same chat sequence every arm reuses), then the
# fixed continuation prompt file + ONE saved slot.
restart_verified "hist-build"
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
print("deep prompt fixed")
PY
slot_save "$D/save.json" "t7"
echo "slot saved once: $(usage_of "$D/h3.json")" | tee -a "$D/verdicts.txt"
CONT="$(cat "$D/deep-cont.json")"

fetch_all() { # ad
  local ad="$1"
  for f in chat-dump.logits chat-dump.tokens chat-dump.kv chat-dump.mtp chat-dump.tokens.prefill; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done
}
arm() { # tag do_restore
  local tag="$1" do_restore="$2"
  local ad="$D/$1"
  mkdir -p "$ad"
  restart_verified "$tag"
  if [[ "$do_restore" == "1" ]]; then
    chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    ws_sample "$tag-pre"
    slot_restore "$ad/restore.json" "t7" || fail "$tag restore"
  else
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    ws_sample "$tag-pre"
  fi
  chat "$ad/cont.json" 32 "$CONT" || fail "$tag cont"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  {
    echo "== $tag"
    grep -m1 -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$ad/slice.log" || echo "no-restore"
    grep -m1 -oE "S3: state dumped — kv_bytes=[0-9]+ mirror_pos=[0-9-]+ pending=[01] kv_tokens=[0-9]+" "$ad/slice.log" || echo "no-kvdump"
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1 || true
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1 || true
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1 || true
    python3 -c "import json;d=json.load(open('$ad/cont.json'));u=d['usage'];print(f\"CTOK={u['prompt_tokens']}/{u['completion_tokens']} sha={__import__('hashlib').sha256(d['choices'][0]['message']['content'].encode()).hexdigest()[:16]}\")"
  } | tee -a "$D/verdicts.txt"
  fetch_all "$ad"
  ws_sample "$tag-post"
}
arm fresh1 0
arm fresh2 0
arm restore1 1
arm restore2 1
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t7"}' >/dev/null || true
echo "== compare" | tee -a "$D/verdicts.txt"
python3 - "$D" <<'PY'
import hashlib, pathlib, sys
d = pathlib.Path(sys.argv[1])
arms = ("fresh1", "fresh2", "restore1", "restore2")
for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.kv", "chat-dump.mtp"):
    shas = {}
    for a in arms:
        p = d / a / f
        shas[a] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
    for x, y, tag in (("fresh1", "fresh2", "F1_vs_F2"), ("restore1", "restore2", "R1_vs_R2"),
                      ("fresh1", "restore1", "F1_vs_R1")):
        eq = shas[x] == shas[y]
        print(f"{f} {tag}: BYTE_EQUAL={eq} sha_x={shas[x][:20]} sha_y={shas[y][:20]}")
# first differing byte offset in the KV blob
def firstdiff(p, q):
    a, b = pathlib.Path(p).read_bytes(), pathlib.Path(q).read_bytes()
    if a == b:
        return None, len(a), 0
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            nd = sum(1 for j in range(i, n) if a[j] != b[j])
            return i, n, nd
    return n, n, 0
for x, y, tag in (("fresh1", "restore1", "KV F1_vs_R1"), ("restore1", "restore2", "KV R1_vs_R2"),
                  ("fresh1", "fresh2", "KV F1_vs_F2")):
    p, q = d / x / "chat-dump.kv", d / y / "chat-dump.kv"
    if p.exists() and q.exists():
        off, n, nd = firstdiff(p, q)
        print(f"{tag}: size={n} first_diff_offset={off} differing_bytes={nd}")
PY
echo "== T7 done"
