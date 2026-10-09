#!/usr/bin/env bash
# Plan 007 paired A/B on rev167 (same build): runtime knob bench_normmul.txt
# 0 = chain on CPU (control), 1 = chain on D3D12. Profile OFF for timing;
# sched2 census and profile-ON traces in their own runs with unique log dirs.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/normmul-rev167}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.167_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/csv" "$BASE/logs"
printf 'xbox-series-x' > "$BASE/tag.txt"
./scripts/deploy.sh upload-file "$BASE/tag.txt" "$PFN" "" bench_host_tag.txt >/dev/null
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/receipt.txt"

set_normmul() { printf '%s' "$1" > "$BASE/knob.txt"; ./scripts/deploy.sh upload-file "$BASE/knob.txt" "$PFN" "" bench_normmul.txt >/dev/null; }

run_cell() { # knob cell prompt mtp npred profile tag
  local knob="$1" prompt="$3" mtp="$4" npred="$5" prof="$6" tag="$7"
  set_normmul "$knob"
  local logs="$BASE/logs/${tag}"
  rm -rf "$logs"; mkdir -p "$logs"
  RUN_LOG_DIR="$logs" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$prof" --mtp "$mtp" --mtp-pmin 50 --n-predict "$npred" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/${tag}.csv" --runs 1 >> "$BASE/driver.log" 2>&1 \
    && echo "ok $tag" >> "$BASE/receipt.txt" || echo "FAIL $tag" >> "$BASE/receipt.txt"
}

# Paired timing: interleaved knob 0/1, three rounds per MTP cell, two for seq.
for round in 1 2 3; do
  for knob in 0 1; do
    run_cell "$knob" std512 standard-512.txt 2 64 0 "t-std512-mtp-k$knob-r$round"
    run_cell "$knob" chat64 spec-chat-open.txt 2 64 0 "t-chat64-mtp-k$knob-r$round"
  done
done
for round in 1 2; do
  for knob in 0 1; do
    run_cell "$knob" chat64seq spec-chat-open.txt 0 64 0 "t-chat64-seq-k$knob-r$round"
  done
done

# Census (sched2 ON) one run per arm: splits + dispatch counters.
printf 'sched2' > "$BASE/ggmlprof.txt"
./scripts/deploy.sh upload-file "$BASE/ggmlprof.txt" "$PFN" "" ggmlprof.txt >/dev/null
for knob in 0 1; do
  run_cell "$knob" census standard-512.txt 2 64 0 "census-k$knob"
done
./scripts/deploy.sh delete-file "$PFN" ggmlprof.txt >/dev/null 2>&1 || true

# Profile-ON MTP traces per arm (feed/acceptance parity).
for knob in 0 1; do
  run_cell "$knob" trace64 spec-chat-open.txt 2 64 1 "trace-chat64-mtp-k$knob"
done

./scripts/deploy.sh delete-file "$PFN" bench_normmul.txt >/dev/null 2>&1 || true
./scripts/deploy.sh delete-file "$PFN" bench_host_tag.txt >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo NORMMUL_AB_DONE