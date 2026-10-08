#!/usr/bin/env bash
# Plan 006 C5 on-console A/B (rev161): scalar vs AVX max scan.
#   - profile-ON trace runs: per-round VROUND feed/accepted sequences + final IDs
#   - profile-OFF paired timing, interleaved arms
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/c5-rev161"
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.161_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/csv" "$BASE/traces" "$BASE/knobs"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/receipt.txt"
echo "pfn=$PFN" >> "$BASE/receipt.txt"
echo "msix_sha256=823263EA5E57213769708AECCAA0F8D1FE669254DEC1271439AD3E20159ADC04" >> "$BASE/receipt.txt"

set_knob() { printf '%s' "$1" > "$BASE/knobs/bench_topprob_avx.txt"; ./scripts/deploy.sh upload-file "$BASE/knobs/bench_topprob_avx.txt" "$PFN" "" bench_topprob_avx.txt >/dev/null; }

run_trace() { # arm cell prompt
  local arm="$1" cell="$2" prompt="$3"
  set_knob "$arm"
  local logs="/tmp/opencode/c5-logs/$cell-$arm"
  rm -rf "$logs"; mkdir -p "$logs"
  RUN_LOG_DIR="$logs" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 1 --mtp 2 --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/trace-$cell-$arm.csv" --runs 1 >> "$BASE/driver.log" 2>&1
  grep -a "VROUND" "$logs/run1.log" | grep -a "feed=" > "$BASE/traces/$cell-$arm.vround.txt" || true
  grep -a "MTP_STATS" "$logs/run1.log" | tail -1 > "$BASE/traces/$cell-$arm.stats.txt" || true
  grep -a "topprob: max scan" "$logs/run1.log" | tail -1 > "$BASE/traces/$cell-$arm.knob.txt" || true
  rm -rf "$logs"
  echo "trace cell=$cell arm=$arm utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

run_timing() { # arm cell prompt index
  local arm="$1" cell="$2" prompt="$3" idx="$4"
  set_knob "$arm"
  ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp 2 --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/t-$cell-$arm-$idx.csv" --runs 1 >> "$BASE/driver.log" 2>&1
}

for cell in chat64 code64; do
  prompt=spec-chat-open.txt; [[ "$cell" == code64 ]] && prompt=spec-code-edit.txt
  run_trace 0 "$cell" "$prompt"
  run_trace 1 "$cell" "$prompt"
done

# Paired interleaved timing: scalar,avx,scalar,avx,scalar,avx per cell.
for cell in chat64 code64; do
  prompt=spec-chat-open.txt; [[ "$cell" == code64 ]] && prompt=spec-code-edit.txt
  for i in 1 2 3; do
    run_timing 0 "$cell" "$prompt" "$i"
    run_timing 1 "$cell" "$prompt" "$i"
  done
  echo "timing cell=$cell utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
done
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo C5_AB_DONE
