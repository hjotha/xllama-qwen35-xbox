#!/usr/bin/env bash
# T9-fullprefill-armed.sh — plan 009 G-L closure test on rev241: does a FULL
# re-prefill after restore (no shared prefix -> F3.5 keeps nothing) produce
# byte-identical per-step logits to a fresh run of the same multi-ubatch
# prompt? If yes, byte-logits equality is attainable exactly when both arms
# compute the shared rows with the SAME prefill geometry — which is what the
# sealed M3 reuse path deliberately trades for speed (measured). Arms:
#   A  fresh process, prompt Q (multi-ubatch)                -> dump
#   B  new process, restore an UNRELATED slot, then prompt Q (kv_keep=0 ->
#      full prefill, same geometry as A)                     -> dump
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T9"
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
printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"

# an unrelated slot (different content) + the multi-ubatch prompt Q
chat_seed='[{"role":"user","content":"The southern harbor key is DELTA-3. Acknowledge with: noted."}]'
Q=$(python3 - "$REPO/bench/prompts/spec-code-edit.txt" <<'PY'
import json, sys
print(json.dumps([{"role": "user", "content": open(sys.argv[1], "rb").read().decode("utf-8")},
                  {"role": "user", "content": "Summarize the bug and the fix in three sentences."}]))
PY
)
restart_verified "slot-build"
chat "$D/unrelated.json" 8 "$chat_seed"
slot_save "$D/save.json" "t9-x"
echo "$Q" > "$D/Q.json"

run_arm() { # tag restore_slot
  local tag="$1" slotid="$2"
  local ad="$D/$1"
  mkdir -p "$ad"
  restart_verified "$tag"
  if [[ -n "$slotid" ]]; then
    chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    slot_restore "$ad/restore.json" "$slotid" || fail "$tag restore"
  else
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  fi
  chat "$ad/cont.json" 32 "$(cat "$D/Q.json")" || fail "$tag cont"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  {
    echo "== $tag"
    grep -m1 -oE "KV state loaded — [0-9]+ tokens" "$ad/slice.log" || echo "no-restore"
    grep -m1 -oE "KV prefix reuse — kept [0-9]+ of [0-9]+ tokens" "$ad/slice.log" || echo "no-reuse-line (kv_keep=0)"
    grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1 || true
    grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1 || true
    grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1 || true
    python3 -c "import json;d=json.load(open('$ad/cont.json'));print('CTOK',d['usage']['prompt_tokens'],d['usage']['completion_tokens'])"
  } | tee -a "$D/verdicts.txt"
  for f in chat-dump.logits chat-dump.tokens; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done
}
run_arm freshA ""
run_arm restoreB "t9-x"
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t9-x"}' >/dev/null || true
python3 - "$D" <<'PY'
import hashlib, pathlib, sys
d = pathlib.Path(sys.argv[1])
for f in ("chat-dump.logits", "chat-dump.tokens"):
    a, b = d / "freshA" / f, d / "restoreB" / f
    if a.exists() and b.exists():
        ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
        print(f"{f}: freshA_sha={ha[:24]} restoreB_sha={hb[:24]} BYTE_EQUAL={ha == hb}")
    else:
        print(f"{f}: MISSING {a.exists()} {b.exists()}")
PY
echo "== T9 done"
