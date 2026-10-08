#!/usr/bin/env bash
# Resume the missing SWIGLU extension cells (rev154): 256-token timing for the
# cells lost to the disk-full fetch failure, plus both Session scenario arms.
# Knob default OFF; OFF/ON pairs compared by full IDs and session dumps.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/swiglu-ext-rev154"
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.154_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/csv" "$BASE/knobs"
echo "resume_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"

set_sw() {
  printf '%s' "$1" > "$BASE/knobs/d3d12swiglu.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null
}

run_cell() { # name prompt mtp sw profile predict
  local name="$1" prompt="$2" mtp="$3" sw="$4" profile="$5" predict="$6"
  set_sw "$sw"
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "resume-cell=$name sw=$sw utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$profile" --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver-resume.log" 2>&1
  echo "resume-done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

run_cell chat256-mtp-sw1 spec-chat-open.txt 2 1 0 256
run_cell code256-mtp-sw1 spec-code-edit.txt 2 1 0 256
run_cell code256-mtp-sw0 spec-code-edit.txt 2 0 0 256

run_session() { # name sw
  local name="$1" sw="$2"
  set_sw "$sw"
  export EVIDENCE_DIR="$BASE/$name"
  mkdir -p "$EVIDENCE_DIR"
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "resume-session=$name sw=$sw utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --mtp-session \
    --mtp 2 --mtp-pmin 50 >> "$BASE/driver-resume.log" 2>&1
  echo "resume-session-done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}
run_session session-sw0 0
run_session session-sw1 1

check_eq() {
  local a="$1" b="$2" got_a got_b
  got_a=$(sha256sum "$a" | cut -d' ' -f1)
  got_b=$(sha256sum "$b" | cut -d' ' -f1)
  [[ "$got_a" == "$got_b" ]] || { echo "PARITY_FAIL $a vs $b" | tee -a "$BASE/receipt.txt"; return 1; }
}
for r in 2 3; do
  check_eq "$BASE/csv/chat256-mtp-sw0.run$r.tokens" "$BASE/csv/chat256-mtp-sw1.run$r.tokens"
  check_eq "$BASE/csv/code256-mtp-sw0.run$r.tokens" "$BASE/csv/code256-mtp-sw1.run$r.tokens"
done
echo "resume-parity=OK" >> "$BASE/receipt.txt"
echo "resume_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo SWIGLU_EXT_RESUME_DONE
