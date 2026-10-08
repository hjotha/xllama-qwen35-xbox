#!/usr/bin/env bash
# shellcheck disable=SC1090  # env file path is resolved at run time
# Plan 005 experiment 1 (rev127): Q6_K LM-head tile 0 vs 2 at t2/d2/p50.
# Interleaved paired blocks, profile OFF, full-ID parity gate per run.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/q6tile-rev127"
cd "$REPO"
# shellcheck source=/dev/null
set -a; source ~/.config/xllama/xbox-env; set +a
export XLLAMA_MSIX_SHA256=801f50eb5f4cf6a43fef4039563acc41a2e7b0f2096d4d39f0b8b706dc86a99e
export XLLAMA_EXPECTED_PFN=GianlucaMazza.xllama_1.6.0.127_x64__pj67f1fcj4n14
mkdir -p "$BASE/csv" "$BASE/knobs"

printf 'xbox-series-x' > "$BASE/knobs/bench_host_tag.txt"
printf '2' > "$BASE/knobs/cpurepackforcegemv.txt"
printf 'auto' > "$BASE/knobs/d3d12twocol.txt"
printf '1' > "$BASE/knobs/d3d12gdn.txt"
printf '1' > "$BASE/knobs/d3d12q8.txt"
printf '2' > "$BASE/knobs/flashattn.txt"
printf '1' > "$BASE/knobs/mtp_threads.txt"
printf '0' > "$BASE/knobs/mtp_catchup_logits.txt"
for name in bench_host_tag cpurepackforcegemv d3d12twocol d3d12gdn d3d12q8 flashattn mtp_threads mtp_catchup_logits; do
  ./scripts/deploy.sh upload-file "$BASE/knobs/$name.txt" "$XLLAMA_EXPECTED_PFN" "" "$name.txt" >/dev/null
done
sha256sum "$BASE"/knobs/* > "$BASE/knobs-sha256.txt"
{ echo "sha256=$XLLAMA_MSIX_SHA256"; echo "pfn=$XLLAMA_EXPECTED_PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name prompt q6
  local name="$1" prompt="$2" q6="$3"
  printf '%s' "$q6" > "$BASE/knobs/d3d12q6tile.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12q6tile.txt" "$XLLAMA_EXPECTED_PFN" "" d3d12q6tile.txt >/dev/null
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name q6=$q6 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp 2 --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

run_cell chat64-q6c0 spec-chat-open.txt 0
run_cell chat64-q6c2 spec-chat-open.txt 2
run_cell code64-q6c0 spec-code-edit.txt 0
run_cell code64-q6c2 spec-code-edit.txt 2
run_cell code64-q6c2-b2 spec-code-edit.txt 2
run_cell code64-q6c0-b2 spec-code-edit.txt 0
run_cell chat64-q6c2-b2 spec-chat-open.txt 2
run_cell chat64-q6c0-b2 spec-chat-open.txt 0

# Parity gate per run against the frozen baseline digests.
check() { local f="$1" want="$2"; local got; got=$(sha256sum "$f" | cut -d' ' -f1); [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }; }
for c in chat64-q6c0 chat64-q6c2 chat64-q6c0-b2 chat64-q6c2-b2; do
  for r in 2 3; do check "$BASE/csv/$c.run$r.tokens" 84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2; done
done
for c in code64-q6c0 code64-q6c2 code64-q6c0-b2 code64-q6c2-b2; do
  for r in 2 3; do check "$BASE/csv/$c.run$r.tokens" ff42c5beec71ff491b458d0e48cbac4d98f8cdcf8153c78868c64aee687a4010; done
done
echo "parity=OK" >> "$BASE/receipt.txt"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo Q6TILE_AB_DONE
