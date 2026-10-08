#!/usr/bin/env bash
# Plan 006 C4 seq-candidate termination gates (rev159): termgate cancel/stop/EOS
# with the FFN knob OFF vs ON. Each arm runs in its own process (bench_mtp per
# arm), so the process profile binds per arm: seq -> FFN on, MTP -> forced off.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/seqgates-rev159}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.159_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }

run_arm() { # sw arm mtp
  local sw="$1" arm="$2" mtp="$3"
  printf '%s' "$sw" > "$BASE/knobs/d3d12swiglu.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null
  printf '%s' "$mtp" > "$BASE/knobs/bench_mtp.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/bench_mtp.txt" "$PFN" "" bench_mtp.txt >/dev/null
  printf 'cancel,stop,eog' > "$BASE/knobs/termgate.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/termgate.txt" "$PFN" "" termgate.txt >/dev/null
  ./scripts/deploy.sh delete-file "$PFN" termgate-result.csv >/dev/null 2>&1 || true
  ./scripts/deploy.sh delete-file "$PFN" termgate-result.csv.done >/dev/null 2>&1 || true
  printf '' > "$BASE/knobs/termgate.flag"
  ./scripts/deploy.sh upload-file "$BASE/knobs/termgate.flag" "$PFN" "" termgate.flag >/dev/null
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
  local deadline ok=0
  deadline=$(( $(date +%s) + 300 ))
  while [[ "$(date +%s)" -lt "$deadline" ]]; do
    if ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv.done" "/tmp/opencode/qwen4b-recon/${arm}.done" 2>&1 | grep -q "^Fetched"; then ok=1; break; fi
    sleep 5
  done
  [[ "$ok" == "1" ]] || echo "TIMEOUT arm=sw$sw-$arm"
  ./scripts/deploy.sh fetch-file "$PFN" termgate-result.csv "$BASE/term-sw$sw-$arm.csv" >/dev/null 2>&1 || true
  echo "arm=sw$sw-$arm mtp=$mtp done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

for sw in 0 1; do
  run_arm "$sw" seq 0
  run_arm "$sw" ref 4
  run_arm "$sw" cand 4
done
echo TERMGATE_SEQ_DONE
