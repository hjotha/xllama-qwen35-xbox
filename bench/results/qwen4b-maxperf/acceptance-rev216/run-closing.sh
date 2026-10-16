#!/usr/bin/env bash
# run-closing.sh — closing gates on the FINAL package (rev216 LTCG).
# Order: numeric, acceptance b1+b2, termgate, vround, api-recall, slot-smoke.
# Stops at first failure. Mode/evidence recorded per stage; no TPS from
# iteration builds is reported here (this package IS the final binary).
set -uo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/acceptance-rev216"
export BASE PFN_EXPECT=GianlucaMazza.xllama_1.6.0.216_x64__pj67f1fcj4n14
STATUS="$BASE/run-status.txt"
E209="$REPO/bench/results/qwen4b-maxperf/enabled-rev209"
START_STAGE="${1:-numeric}"   # resume support: skip stages before this one
: > "$STATUS"
echo "start_stage=$START_STAGE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
SKIP=0
[[ "$START_STAGE" != "numeric" ]] && SKIP=1
run_stage() {
  local name="$1"; shift
  if [[ "$SKIP" == 1 ]]; then
    if [[ "$name" == "$START_STAGE" ]]; then SKIP=0
    else echo "STAGE $name SKIP (before start_stage=$START_STAGE)" >> "$STATUS"; return 0
    fi
  fi
  echo "STAGE $name start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  if "$@" >> "$BASE/stages.txt" 2>&1; then
    echo "STAGE $name OK end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  else
    local rc=$?
    echo "STAGE $name FAIL rc=$rc end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
    echo "STAGE_FAILED name=$name rc=$rc"
    exit "$rc"
  fi
}
run_stage numeric  "$BASE/numeric-rev216.sh"
run_stage block-b1 "$E209/acceptance-rev209-v2.sh" b1
run_stage block-b2 "$E209/acceptance-rev209-v2.sh" b2
run_stage termgate "$E209/termgate-rev209.sh"
run_stage vround   "$E209/vround-rev209.sh"
run_stage api      "$BASE/api-recall-rev216.sh"
run_stage slotsmoke "$BASE/slot-smoke-rev216.sh"
echo "ALL_CLOSING_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
echo ALL_CLOSING_DONE
