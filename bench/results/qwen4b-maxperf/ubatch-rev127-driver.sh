#!/usr/bin/env bash
# Plan 005 experiment 2 (rev127): prefill n_batch/n_ubatch sweep.
# Arms u64 (baseline) / u128 / u256, std-512 prompt, seq and MTP, two
# opposite-order blocks, --runs 2 (warmup + 1 measured), profile OFF.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/ubatch-rev127"
cd "$REPO"
set -a; source ~/.config/xllama/xbox-env; set +a
export XLLAMA_MSIX_SHA256=801f50eb5f4cf6a43fef4039563acc41a2e7b0f2096d4d39f0b8b706dc86a99e
export XLLAMA_EXPECTED_PFN=GianlucaMazza.xllama_1.6.0.127_x64__pj67f1fcj4n14
mkdir -p "$BASE/csv" "$BASE/knobs"
printf 'xbox-series-x' > "$BASE/knobs/bench_host_tag.txt"
printf '0' > "$BASE/knobs/d3d12q6tile.txt"
printf '2' > "$BASE/knobs/cpurepackforcegemv.txt"
printf 'auto' > "$BASE/knobs/d3d12twocol.txt"
printf '1' > "$BASE/knobs/d3d12gdn.txt"
printf '1' > "$BASE/knobs/d3d12q8.txt"
printf '2' > "$BASE/knobs/flashattn.txt"
printf '1' > "$BASE/knobs/mtp_threads.txt"
printf '0' > "$BASE/knobs/mtp_catchup_logits.txt"
for name in bench_host_tag d3d12q6tile cpurepackforcegemv d3d12twocol d3d12gdn d3d12q8 flashattn mtp_threads mtp_catchup_logits; do
  ./scripts/deploy.sh upload-file "$BASE/knobs/$name.txt" "$XLLAMA_EXPECTED_PFN" "" "$name.txt" >/dev/null
done
sha256sum "$BASE"/knobs/* > "$BASE/knobs-sha256.txt"
{ echo "sha256=$XLLAMA_MSIX_SHA256"; echo "pfn=$XLLAMA_EXPECTED_PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name ubatch mtp
  local name="$1" ub="$2" mtp="$3"
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name u=$ub mtp=$mtp utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch "$ub" --ubatch "$ub" \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/standard-512.txt" \
    --out "$BASE/csv/$name.csv" --runs 2 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

run_cell std512-seq-u64 64 0
run_cell std512-seq-u128 128 0
run_cell std512-seq-u256 256 0
run_cell std512-mtp-u64 64 2
run_cell std512-mtp-u128 128 2
run_cell std512-mtp-u256 256 2
run_cell std512-mtp-u256-b2 256 2
run_cell std512-mtp-u128-b2 128 2
run_cell std512-mtp-u64-b2 64 2
run_cell std512-seq-u256-b2 256 0
run_cell std512-seq-u128-b2 128 0
run_cell std512-seq-u64-b2 64 0

check() { local f="$1" want="$2" got; got=$(sha256sum "$f" | cut -d' ' -f1); [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }; }
for c in std512-seq-u64 std512-seq-u128 std512-seq-u256 std512-seq-u64-b2 std512-seq-u128-b2 std512-seq-u256-b2 std512-mtp-u64 std512-mtp-u128 std512-mtp-u256 std512-mtp-u64-b2 std512-mtp-u128-b2 std512-mtp-u256-b2; do
  check "$BASE/csv/$c.run2.tokens" 44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732
done
echo "parity=OK" >> "$BASE/receipt.txt"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo UBATCH_SWEEP_DONE
