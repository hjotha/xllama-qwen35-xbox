#!/usr/bin/env bash
# shellcheck disable=SC1090  # env file path is resolved at run time
# Plan 005 baseline pilot on rev126: MTP toggle only, same qwen35-4b-mtp GGUF.
# Interleaved seq/mtp cells, prose+code @64, profile OFF timings and one
# profile ON attribution pair. Absolute cwd; XLLAMA_EXPECTED_PFN guards.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/baseline-rev126"
cd "$REPO"
# shellcheck source=/dev/null
set -a; source ~/.config/xllama/xbox-env; set +a
export XLLAMA_MSIX_SHA256=5bde8869ab587350fa42fb4a81ef1d47744697db672abb0d9bb6ed521682105f
export XLLAMA_EXPECTED_PFN=GianlucaMazza.xllama_1.6.0.126_x64__pj67f1fcj4n14
mkdir -p "$BASE/csv"

{
  echo "sha256=$XLLAMA_MSIX_SHA256"
  echo "pfn=$XLLAMA_EXPECTED_PFN"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$BASE/receipt.txt"

# Profile knob bundle (rev124 delivered profile minus per-run bench_* knobs).
declare -A KNOBS=(
  [d3d12q6tile.txt]=0 [d3d12gdn.txt]=1 [d3d12q8.txt]=1 [flashattn.txt]=2
  [mtp_threads.txt]=1 [mtp_catchup_logits.txt]=0 [d3d12twocol.txt]=auto
  [cpurepackforcegemv.txt]=2
)
for name in "${!KNOBS[@]}"; do
  printf '%s' "${KNOBS[$name]}" > "$BASE/knobs/$name"
  ./scripts/deploy.sh upload-file "$BASE/knobs/$name" "$XLLAMA_EXPECTED_PFN" "" "$name" >/dev/null
done
sha256sum "$BASE"/knobs/* > "$BASE/knobs-sha256.txt"

run_cell() { # name prompt mtp profile
  local name="$1" prompt="$2" mtp="$3" profile="$4"
  local tag="-mtp${mtp}-pmin50"
  [[ "$profile" == "0" ]] && tag="$tag-prof0"
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$profile" --mtp "$mtp" --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver.log" 2>&1
  echo "cell=$name done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

# Block 1: chat seq, chat mtp, code seq, code mtp (profile OFF)
run_cell chat64-seq      spec-chat-open.txt 0 0
run_cell chat64-mtp      spec-chat-open.txt 2 0
run_cell code64-seq      spec-code-edit.txt 0 0
run_cell code64-mtp      spec-code-edit.txt 2 0
# Block 2: reversed order (profile OFF)
run_cell code64-mtp-b2   spec-code-edit.txt 2 0
run_cell code64-seq-b2   spec-code-edit.txt 0 0
run_cell chat64-mtp-b2   spec-chat-open.txt 2 0
run_cell chat64-seq-b2   spec-chat-open.txt 0 0
# Attribution pair (profile ON, not used for timing claims)
run_cell chat64-mtp-on   spec-chat-open.txt 2 1
run_cell chat64-seq-on   spec-chat-open.txt 0 1

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo BASELINE_PILOT_DONE
