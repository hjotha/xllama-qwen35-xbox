#!/usr/bin/env bash
# rev209 acceptance: 2 interleaved blocks, 12 cells x 2 arms, --runs 3 =>
# run1 warmup dropped, run2+run3 = 2 real measured lines per arm per cell per
# block. Idempotent: skips a cell-arm already having 2 rows. Post-check is the
# STRICT verify-blocks-v3.py (24 CSVs required, explicit baseline/parity).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/enabled-rev209}"
BLOCK="${1:?block id (b1|b2)}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.209_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN want=$PFN_EXPECT" >&2; exit 1; }
D="$BASE/blocks/$BLOCK"; mkdir -p "$D/csv" "$D/knobs"
set_sw() { printf '%s' "$1" > "$D/knobs/d3d12swiglu.txt"; ./scripts/deploy.sh upload-file "$D/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null; }
measured_rows() { local f="$1"; [[ -f "$f" ]] && tail -n +2 "$f" | wc -l || echo 0; }
run_cell() { # name prompt mtp predict
  local name="$1" prompt="$2" mtp="$3" predict="$4" sw out n
  for sw in 0 1; do
    out="$D/csv/$name-sw$sw.csv"
    n=$(measured_rows "$out")
    if [[ "$n" -ge 2 ]]; then echo "skip $name-sw$sw (already $n rows)"; continue; fi
    set_sw "$sw"
    "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
      --threads 2 --ctx 2048 --gpu-layers 34 --batch 256 --ubatch 64 \
      --twocol auto --greedy --seed 1 --tokens --ignore-eog \
      --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
      --prompt "$REPO/bench/prompts/$prompt" \
      --out "$out" --runs 3 >> "$D/driver.log" 2>&1
    echo "cell=$name sw=$sw rows=$(measured_rows "$out") utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
}
if [[ ! -f "$D/done" ]]; then
  echo "block=$BLOCK start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_cell chat64     spec-chat-open.txt 0 64
  run_cell code64     spec-code-edit.txt 0 64
  run_cell std64      standard-512.txt   0 64
  run_cell chat256    spec-chat-open.txt 0 256
  run_cell code256    spec-code-edit.txt 0 256
  run_cell std256     standard-512.txt   0 256
  run_cell chat64-mtp spec-chat-open.txt 2 64
  run_cell code64-mtp spec-code-edit.txt 2 64
  run_cell std64-mtp  standard-512.txt   2 64
  run_cell chat256-mtp spec-chat-open.txt 2 256
  run_cell code256-mtp spec-code-edit.txt 2 256
  run_cell std256-mtp  standard-512.txt   2 256
  touch "$D/done"
  echo "block=$BLOCK end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
python3 "$BASE/verify-blocks-v3.py" "$D"
