#!/usr/bin/env bash
# shellcheck disable=SC1090  # env file path is resolved at run time
# Plan 005 baseline part 2: 256-token pairs (prose/code) and a 512-token
# long-prompt prefill pair. MTP toggle only, same qwen35-4b-mtp GGUF.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/baseline-rev126"
cd "$REPO"
# shellcheck source=/dev/null
set -a; source ~/.config/xllama/xbox-env; set +a
export XLLAMA_MSIX_SHA256=5bde8869ab587350fa42fb4a81ef1d47744697db672abb0d9bb6ed521682105f
export XLLAMA_EXPECTED_PFN=GianlucaMazza.xllama_1.6.0.126_x64__pj67f1fcj4n14
mkdir -p "$BASE/csv"
echo "part2_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"

run_cell() { # name prompt mtp predict
  local name="$1" prompt="$2" mtp="$3" predict="$4"
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver-part2.log" 2>&1
  echo "cell=$name done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

run_cell chat256-seq spec-chat-open.txt 0 256
run_cell chat256-mtp spec-chat-open.txt 2 256
run_cell code256-mtp spec-code-edit.txt 2 256
run_cell code256-seq spec-code-edit.txt 0 256
run_cell std512-seq  standard-512.txt  0 64
run_cell std512-mtp  standard-512.txt  2 64

echo "part2_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo BASELINE_PART2_DONE
