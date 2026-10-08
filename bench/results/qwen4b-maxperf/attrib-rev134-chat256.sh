#!/usr/bin/env bash
# Plan 006 rev134 correction run: the two chat256 cells with n_predict=256
# (the first pass mislabeled 64-token runs; those are preserved under
# mislabelled-64/). No deploy; rev134 already installed.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/attrib-rev134"
SHA=d7842e262e07daea1c14907f41838d82e3ffbec7dc3070fb75a65a3651cbc681
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.134_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
export XLLAMA_MSIX_SHA256="$SHA" XLLAMA_EXPECTED_PFN="$PFN"
echo "chat256-correction start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"

run_cell() { # name profile
  local name="$1" profile="$2"
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=2 prof=$profile predict=256 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$profile" --mtp 2 --mtp-pmin 50 --n-predict 256 \
    --prompt "$REPO/bench/prompts/spec-chat-open.txt" \
    --out "$BASE/csv/$name.csv" --runs 2 >> "$BASE/driver-chat256.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}
run_cell chat256-mtp-prof1 1
run_cell chat256-mtp-off 0

check() {
  local f="$1" want="$2" got
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }
}
check "$BASE/csv/chat256-mtp-prof1.run2.tokens" f78ce8372ccc7281aaad5e726bf7c88a0317212e3b2ed397fb0fd83ffd53066c
check "$BASE/csv/chat256-mtp-off.run2.tokens" f78ce8372ccc7281aaad5e726bf7c88a0317212e3b2ed397fb0fd83ffd53066c
echo "chat256-parity=OK" >> "$BASE/receipt.txt"
last_line() { grep -a "$2" "$BASE/logs/$1/run2.log" 2>/dev/null | tail -1; }
for c in chat256-mtp-prof1 chat256-mtp-off; do
  echo "engage=$c '$(last_line "$c" "gpu timestamps=" | grep -ao "gpu timestamps=[a-z]* ([a-z-]* [a-z]*)" | tail -1)'" >> "$BASE/receipt.txt"
done
echo "chat256-correction end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo CHAT256_DONE
