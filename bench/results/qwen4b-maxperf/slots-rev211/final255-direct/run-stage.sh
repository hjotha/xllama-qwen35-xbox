#!/usr/bin/env bash
set -euo pipefail
# An older unacknowledged campaign lacks this explicit invocation marker.
[[ "${2:-}" == "confirmed-20261013" ]] || { echo "Unacknowledged legacy invocation refused before device access" >&2; exit 75; }
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
export BASE="$REPO/bench/results/qwen4b-maxperf/slots-rev211/final255-direct"
export PFN_EXPECT=GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
[[ "$(./scripts/deploy.sh pfn)" == "$PFN_EXPECT" ]]
[[ -f "$BASE/package-verification.json" ]]
mkdir -p "$BASE/scratch"
STAGE="${1:?stage}"
echo "$STAGE start $(date -u +%FT%TZ)" >> "$BASE/stages.txt"
case "$STAGE" in
cold) bash "$BASE/cold-prefill.sh" ;;
fingerprint) python3 "$BASE/fingerprint-device.py" ;;
numeric) bash "$BASE/numeric.sh" ;;
b1|b2) bash bench/results/qwen4b-maxperf/enabled-rev209/acceptance-rev209-v2.sh "$STAGE" ;;
termgate) bash "$BASE/termgate.sh" ;;
vround)
 bash "$BASE/vround-rev209.sh"
 python3 bench/results/qwen4b-maxperf/enabled-rev209/extract-segments.py "$BASE/vround"
 python3 "$BASE/check-vround.py" ;;
api) bash "$BASE/api-recall.sh" ;;
slots-on) KNOB=on bash "$BASE/slot-battery.sh" ;;
slots-off) KNOB=off bash "$BASE/slot-battery.sh" ;;
stability) bash "$BASE/stability.sh" ;;
aligned) bash "$BASE/aligned-state.sh" ;;
concurrency) python3 "$BASE/concurrency.py" ;;
fault) python3 "$BASE/fault-contention.py" ;;
trailer) python3 "$BASE/trailer-matrix.py" ;;
*) exit 2 ;;
esac
./scripts/deploy.sh status "$PFN_EXPECT" >> "$BASE/stages.txt"
echo "$STAGE PASS $(date -u +%FT%TZ)" >> "$BASE/stages.txt"

