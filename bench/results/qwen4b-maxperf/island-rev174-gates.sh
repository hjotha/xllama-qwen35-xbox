#!/usr/bin/env bash
# Plan 008 extended gates on rev174: chat256/code256, seq+MTP, island OFF/ON,
# full-token digests + per-round VROUND feed traces (profile ON), plus census
# engagement/split reduction and a same-build timing re-check (profile OFF).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/island-rev174"
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.174_x64__pj67f1fcj4n14
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

set_knob() {
  if [[ "$1" == "1" ]]; then printf '1' > "$BASE/knob.txt"; ./scripts/deploy.sh upload-file "$BASE/knob.txt" "$PFN" "" d3d12island.txt >/dev/null;
  else ./scripts/deploy.sh delete-file "$PFN" d3d12island.txt >/dev/null 2>&1 || true; fi
}

run() { # knob prompt mtp npred prof tag
  local knob="$1" prompt="$2" mtp="$3" npred="$4" prof="$5" tag="$6"
  set_knob "$knob"
  local logs="$BASE/logs/$tag"
  rm -rf "$logs"; mkdir -p "$logs"
  RUN_LOG_DIR="$logs" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$prof" --mtp "$mtp" --mtp-pmin 50 --n-predict "$npred" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/${tag}.csv" --runs 1 >> "$BASE/driver.log" 2>&1 \
    && echo "ok $tag" >> "$BASE/receipt.txt" || echo "FAIL $tag" >> "$BASE/receipt.txt"
}

# Trace gates: 256-token chat and code, seq and MTP, OFF/ON.
for prompt in spec-chat-open.txt spec-code-edit.txt; do
  tag=${prompt#spec-}; tag=${tag%.txt}
  for mtp in 2 0; do
    for knob in 0 1; do
      run "$knob" "$prompt" "$mtp" 256 1 "trace${tag}256-mtp$mtp-k$knob"
    done
  done
done
# Census ON/OFF (sched2) for engagement + split reduction on this build.
printf 'sched2' > "$BASE/ggmlprof.txt"
./scripts/deploy.sh upload-file "$BASE/ggmlprof.txt" "$PFN" "" ggmlprof.txt >/dev/null
for knob in 0 1; do
  run "$knob" standard-512.txt 2 32 0 "census-k$knob"
done
./scripts/deploy.sh delete-file "$PFN" ggmlprof.txt >/dev/null 2>&1 || true
# Same-build timing spot-check (3 interleaved rounds, MTP + seq chat64).
for round in 1 2 3; do
  for knob in 0 1; do
    run "$knob" spec-chat-open.txt 2 64 0 "t-chat64-mtp-k$knob-r$round"
    run "$knob" spec-chat-open.txt 0 64 0 "t-chat64-seq-k$knob-r$round"
  done
done
./scripts/deploy.sh delete-file "$PFN" d3d12island.txt >/dev/null 2>&1 || true
./scripts/deploy.sh delete-file "$PFN" bench_host_tag.txt >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo EXTENDED_GATES_DONE