#!/usr/bin/env bash
# n_batch/n_ubatch 128/256 sweep on the final 1.6.0.163 package: does it answer
# or OOM, and what prefill/decode tok/s. Joint b/ub (as rev127 did), profile
# OFF, MTP arm, two prompts (std-512 = 298-token prefill, chat64 = decode),
# interleaved configs so drift hits both equally.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/batchubatch-rev163"
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.163_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/csv" "$BASE/logs"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/receipt.txt"
echo "pfn=$PFN" >> "$BASE/receipt.txt"
printf 'xbox-series-x' > "$BASE/tag.txt"
./scripts/deploy.sh upload-file "$BASE/tag.txt" "$PFN" "" bench_host_tag.txt >/dev/null

cell() { # cfg prompt npredict
  local cfg="$1" prompt="$2" npredict="$3"
  RUN_LOG_DIR="$BASE/logs" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch "$cfg" --ubatch "$cfg" \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp 2 --mtp-pmin 50 --n-predict "$npredict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/${prompt%.txt}-bu$cfg.csv" --runs 1 >> "$BASE/driver.log" 2>&1 \
    && echo "ok $prompt $cfg" >> "$BASE/receipt.txt" \
    || echo "FAIL $prompt $cfg" >> "$BASE/receipt.txt"
}

for prompt in standard-512.txt spec-chat-open.txt; do
  npredict=64
  [[ "$prompt" == standard-512.txt ]] && npredict=64
  for _round in 1 2; do
    for cfg in 64 128 256; do
      cell "$cfg" "$prompt" "$npredict"
    done
  done
done
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo SWEEP_DONE