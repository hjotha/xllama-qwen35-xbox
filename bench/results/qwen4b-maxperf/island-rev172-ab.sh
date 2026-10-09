#!/usr/bin/env bash
# Plan 008 island paired A/B on rev172 (same build): d3d12island.txt 0/1,
# interleaved, profile OFF; MTP and seq cells; profile-ON traces per arm.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/island-rev172"
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.172_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/csv"
printf 'xbox-series-x' > "$BASE/tag.txt"
./scripts/deploy.sh upload-file "$BASE/tag.txt" "$PFN" "" bench_host_tag.txt >/dev/null

set_knob() {
  if [[ "$1" == "1" ]]; then printf '1' > "$BASE/knob.txt"; ./scripts/deploy.sh upload-file "$BASE/knob.txt" "$PFN" "" d3d12island.txt >/dev/null;
  else ./scripts/deploy.sh delete-file "$PFN" d3d12island.txt >/dev/null 2>&1 || true; fi
}

run_cell() { # knob prompt mtp prof tag
  local knob="$1" prompt="$2" mtp="$3" prof="$4" tag="$5"
  set_knob "$knob"
  local logs="$BASE/logs/${tag}"
  rm -rf "$logs"; mkdir -p "$logs"
  RUN_LOG_DIR="$logs" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$prof" --mtp "$mtp" --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/${tag}.csv" --runs 1 >> "$BASE/driver.log" 2>&1 \
    && echo "ok $tag" >> "$BASE/receipt.txt" || echo "FAIL $tag" >> "$BASE/receipt.txt"
}

echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/receipt.txt"
# MTP cells: 3 interleaved rounds.
for round in 1 2 3; do
  for knob in 0 1; do
    run_cell "$knob" standard-512.txt 2 0 "t-std512-mtp-k$knob-r$round"
    run_cell "$knob" spec-chat-open.txt 2 0 "t-chat64-mtp-k$knob-r$round"
  done
done
# Seq cells: 2 rounds.
for round in 1 2; do
  for knob in 0 1; do
    run_cell "$knob" standard-512.txt 0 0 "t-std512-seq-k$knob-r$round"
    run_cell "$knob" spec-chat-open.txt 0 0 "t-chat64-seq-k$knob-r$round"
  done
done
# Profile-ON traces per arm (MTP).
for knob in 0 1; do
  run_cell "$knob" spec-chat-open.txt 2 1 "trace-chat64-mtp-k$knob"
done
./scripts/deploy.sh delete-file "$PFN" d3d12island.txt >/dev/null 2>&1 || true
./scripts/deploy.sh delete-file "$PFN" bench_host_tag.txt >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo ISLAND_AB_DONE