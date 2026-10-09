#!/usr/bin/env bash
# rev209 delimited VROUND traces: profile-ENABLED runs for the schedule gate.
# Segment extraction (last "main_loop:" start .. last "[xllama] done:" end of
# the freshly fetched accumulated log) happens in extract-segments.py after
# the runs. Performance timings elsewhere stay profile OFF.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/enabled-rev209}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.209_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/vround"
set_sw() { printf '%s' "$1" > "$BASE/vround/sw.txt"; ./scripts/deploy.sh upload-file "$BASE/vround/sw.txt" "$PFN" "" d3d12swiglu.txt >/dev/null; }
one() { # cell prompt mtp predict sw
  local cell="$1" prompt="$2" mtp="$3" predict="$4" sw="$5"
  set_sw "$sw"
  RUN_LOG_DIR="$BASE/vround/$cell-sw$sw" ./scripts/bench-xbox-ort.sh qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 256 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 1 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/vround/$cell-sw$sw.csv" --runs 2 >/dev/null 2>&1
  echo "done $cell-sw$sw utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
for sw in 0 1; do one chat256-mtp spec-chat-open.txt 2 256 "$sw"; done
for sw in 0 1; do one code256-mtp spec-code-edit.txt 2 256 "$sw"; done
echo VROUND_RUN_REV209_DONE
