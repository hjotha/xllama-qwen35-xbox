#!/usr/bin/env bash
# Owner audit step 2+3 (batchfactorial-rev165): per-cell/per-repeat raw logs, a
# same-boundary census pass with node-level scheduling (ggmlprof=sched2) and
# prefill AND decode stage profiling. Every cell gets its own RUN_LOG_DIR so
# earlier evidence can never be overwritten (the first factorial pass shared one
# directory and kept only run1.log).
#
# Cells: 64/64 (reference), 128/128 and 256/256 (the regressing corners) and
# 256/64 (best measured prefill). Profile ON for attribution; timing comes from
# the profile-OFF factorial CSVs already in csv/.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/batchfactorial-rev165"
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.165_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }

# Node-level scheduling census (per-node op, shape and backend assignment).
# Profile OFF on the clock side: sched2 only redirects ggml's scheduler debug,
# it is not the decode-phase instrumentation.
printf 'sched2' > "$BASE/ggmlprof.txt"
./scripts/deploy.sh upload-file "$BASE/ggmlprof.txt" "$PFN" "" ggmlprof.txt >/dev/null

census() { # nb nu repeat
  local nb="$1" nu="$2" rep="$3"
  local dir="$BASE/logs/std512-b$nb-u$nu-r$rep"
  rm -rf "$dir"
  mkdir -p "$dir"
  echo "census std512 b=$nb u=$nu r=$rep -> $dir" >> "$BASE/receipt.txt"
  RUN_LOG_DIR="$dir" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch "$nb" --ubatch "$nu" \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp 2 --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/standard-512.txt" \
    --out "$BASE/csv/census-b$nb-u$nu-r$rep.csv" --runs 1 >> "$BASE/driver.log" 2>&1 \
    && echo "ok census b=$nb u=$nu r=$rep" >> "$BASE/receipt.txt" \
    || echo "FAIL census b=$nb u=$nu r=$rep" >> "$BASE/receipt.txt"
}

for combo in "64 64" "128 128" "256 256" "256 64"; do
  # shellcheck disable=SC2086
  set -- $combo
  census "$1" "$2" 1
done

./scripts/deploy.sh delete-file "$PFN" ggmlprof.txt >/dev/null 2>&1 || true

# Same-boundary stage profiling (profile ON): DSTEP covers every decode call
# including the prefill path, CSTEP the classic step, VROUND the verify rounds,
# DSPLIT the whole decode split. Prefill DSTEP lines are the same-boundary
# prefill evidence; the decode half comes from VROUND/CSTEP.
attr() { # nb nu
  local nb="$1" nu="$2"
  local dir="$BASE/logs/attr-std512-b$nb-u$nu"
  rm -rf "$dir"
  mkdir -p "$dir"
  echo "attr std512 b=$nb u=$nu -> $dir" >> "$BASE/receipt.txt"
  RUN_LOG_DIR="$dir" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch "$nb" --ubatch "$nu" \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 1 --mtp 2 --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/standard-512.txt" \
    --out "$BASE/attr/stage-b$nb-u$nu.csv" --runs 1 >> "$BASE/driver.log" 2>&1 \
    && echo "ok attr b=$nb u=$nu" >> "$BASE/receipt.txt" \
    || echo "FAIL attr b=$nb u=$nu" >> "$BASE/receipt.txt"
}

for combo in "64 64" "128 128" "256 256" "256 64"; do
  # shellcheck disable=SC2086
  set -- $combo
  attr "$1" "$2"
done

printf '' > /tmp/opencode/qwen4b-recon/done.flag
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo CENSUS_DONE