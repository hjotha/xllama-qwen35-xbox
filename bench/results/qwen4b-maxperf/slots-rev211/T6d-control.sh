#!/usr/bin/env bash
# T6d-control.sh — determinism control for the S3 byte-probe (rev240):
# (1) a SECOND fresh run of the deep prompt in a new process must be
# byte-identical to the first fresh run; (2) a SECOND restore run must be
# byte-identical to the first restore run. If both hold, the fresh-vs-restore
# logits difference in the deep case is a property of the ARITHMETIC PATH
# (full multi-ubatch prefill vs tail-only prefill), not of the disk restore
# itself — the short case (single-ubatch both arms) already produced
# byte-identical logits under restore.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T6d"
T6C="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T6c"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.240_x64__pj67f1fcj4n14"
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
CONT="$(cat "$T6C/deep-cont.json")"

# rebuild the same deep slot (deterministic seeds: greedy)
restart_verified "seed"
chat "$D/seed.json" 48 "$(cat "$T6C/h2m.json")"
chat "$D/seed2.json" 48 "$(cat "$T6C/h3m.json")"
slot_save "$D/save.json" "t6d"

run_arm() { # tag do_restore
  local tag="$1" do_restore="$2"
  local ad="$D/$tag"
  mkdir -p "$ad"
  restart_verified "$tag-new"
  if [[ "$do_restore" == "1" ]]; then
    chat "$ad/pre.json" 1 '[{"role":"user","content":"x"}]'
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
    slot_restore "$ad/restore.json" "t6d" || fail "$tag restore"
  else
    L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  fi
  chat "$ad/cont.json" 32 "$CONT" || fail "$tag cont"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$ad/slice.log"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1 | tee -a "$D/verdicts.txt" || true
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1 | tee -a "$D/verdicts.txt" || true
  for f in chat-dump.logits chat-dump.tokens; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done
}
run_arm fresh2 0
run_arm restore2 1
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 http://192.168.1.26:11434/v1/slots/delete -H 'Content-Type: application/json' -d '{"slot":"t6d"}' >/dev/null || true
echo "== compare (vs T6c first runs)" | tee -a "$D/verdicts.txt"
python3 - "$D" "$T6C" <<'PY'
import hashlib, pathlib, sys
d, c = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for tag, ref in (("fresh2", "fresh"), ("restore2", "restore")):
    for f in ("chat-dump.logits", "chat-dump.tokens"):
        a, b = d / tag / f, c / ref / f
        if a.exists() and b.exists():
            ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
            print(f"{f}: {tag}_vs_{ref} BYTE_EQUAL={ha == hb} sha={ha[:24]}")
        else:
            print(f"{f}: {tag} MISSING")
PY
echo "== T6d done"
