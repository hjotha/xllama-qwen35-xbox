#!/usr/bin/env bash
# Numeric gate on the INSTALLED rev255 package: runs the device d3d12be
# selftest (50-row numeric CSV incl. silu_avx2/swiglu_avx2) plus the shape-cost
# diagnostic, then slices the log over the selftest window for the fused
# ord/edge counters. No relabeling: output files are rev255-named.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/slots-rev211/final255-direct}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l > "$BASE/numeric-log-lines-before.txt"
L0=$(cat "$BASE/numeric-log-lines-before.txt")
./scripts/bench-d3d12-selftest.sh \
  --out "$BASE/d3d12be-rev255.csv" \
  --shapecost-out "$BASE/d3d12sc-rev255.csv" \
  --timeout 1200
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$BASE/numeric-session-segment.log"
{
  echo "numeric_selftest_rev255 start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "csv=d3d12be-rev255.csv rows=$(( $(wc -l < "$BASE/d3d12be-rev255.csv") - 1 ))"
  echo "silu_fused=$(grep -c 'silu_avx2.*ok=1\|\[fused\].*silu' "$BASE/numeric-session-segment.log" || true)"
  grep -E "silu|swiglu|D2a=" "$BASE/numeric-session-segment.log" | tail -20
  echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$BASE/numeric-receipt.txt"
echo NUMERIC_REV255_DONE
