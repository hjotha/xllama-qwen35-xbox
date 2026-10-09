#!/usr/bin/env bash
# Sequential gate runner for the installed rev209 package. Stops at the first
# failing stage; one status line per stage in run-status.txt, raw output in
# stages.log. Final stage (api) leaves FFN GPU+MTP enabled (default knob) and
# the API serving.
set -uo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/enabled-rev209"
STATUS="$BASE/run-status.txt"
START_STAGE="${1:-numeric}"   # skip stages before this one (resume support)
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
  if "$@" >> "$BASE/stages.log" 2>&1; then
    echo "STAGE $name OK end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  else
    local rc=$?
    echo "STAGE $name FAIL rc=$rc end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
    echo "STAGE_FAILED name=$name rc=$rc"
    exit "$rc"
  fi
}
run_stage numeric  "$BASE/numeric-rev209.sh"
run_stage block-b1 "$BASE/acceptance-rev209-v2.sh" b1
run_stage block-b2 "$BASE/acceptance-rev209-v2.sh" b2
run_stage vround   "$BASE/vround-rev209.sh"
run_stage termgate "$BASE/termgate-rev209.sh"
run_stage fallback "$BASE/fallback-knob0.sh"
run_stage api      "$BASE/api-multiturn-rev209.sh"
echo "ALL_GATES_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
echo ALL_GATES_DONE
