#!/usr/bin/env bash
# Plan 006 C4 sequential-candidate gates (rev159): longer-prompt IDs, API chat,
# termgate (cancel/stop/EOS) and the MTP baseline regression with FFN forced
# OFF. Knob default OFF; the seq profile binds at the first context.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/seqgates-rev159"
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.159_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/csv" "$BASE/knobs"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/receipt.txt"

set_sw() { printf '%s' "$1" > "$BASE/knobs/d3d12swiglu.txt"; ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null; }

run_cell() { # name prompt mtp sw predict
  local name="$1" prompt="$2" mtp="$3" sw="$4" predict="$5"
  set_sw "$sw"
  echo "cell=$name mtp=$mtp sw=$sw predict=$predict utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 2 >> "$BASE/driver.log" 2>&1
}

# A. Longer-prompt sequential IDs (256 tok) + seq std-512.
run_cell chat256-seq-sw0 spec-chat-open.txt 0 0 256
run_cell chat256-seq-sw1 spec-chat-open.txt 0 1 256
run_cell code256-seq-sw1 spec-code-edit.txt 0 1 256
run_cell code256-seq-sw0 spec-code-edit.txt 0 0 256
run_cell std512-seq-sw0 standard-512.txt 0 0 64
run_cell std512-seq-sw1 standard-512.txt 0 1 64
# C. MTP baseline regression with FFN forced OFF (knob 1 must engage nothing).
run_cell chat64-mtp-sw0 spec-chat-open.txt 2 0 64
run_cell chat64-mtp-sw1 spec-chat-open.txt 2 1 64

check_eq() {
  local a="$1" b="$2" ga gb
  ga=$(sha256sum "$a" | cut -d' ' -f1); gb=$(sha256sum "$b" | cut -d' ' -f1)
  [[ "$ga" == "$gb" ]] || { echo "PARITY_FAIL $a vs $b" | tee -a "$BASE/receipt.txt"; return 1; }
}
for c in chat256-seq code256-seq std512-seq; do
  check_eq "$BASE/csv/$c-sw0.run2.tokens" "$BASE/csv/$c-sw1.run2.tokens"
done
check_eq "$BASE/csv/chat64-mtp-sw0.run2.tokens" "$BASE/csv/chat64-mtp-sw1.run2.tokens"
echo "parity=OK" >> "$BASE/receipt.txt"

# B. API chat gate with a sequential session (llama.ini mtp=0, knob ON).
./scripts/deploy.sh fetch-file "$PFN" llama.ini "$BASE/llama.ini.backup" >/dev/null 2>&1 || true
if [[ -f "$BASE/llama.ini.backup" ]]; then
  sed 's/^mtp=2$/mtp=0/' "$BASE/llama.ini.backup" > "$BASE/llama.ini.seq"
  grep -q '^mtp=0$' "$BASE/llama.ini.seq" || printf '\nmtp=0\n' >> "$BASE/llama.ini.seq"
  ./scripts/deploy.sh upload-file "$BASE/llama.ini.seq" "$PFN" "" llama.ini >/dev/null
fi
set_sw 1
echo "api_gate_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
if timeout 300 ./scripts/validate-api.sh chat > "$BASE/api-chat.log" 2>&1; then echo "api=OK" >> "$BASE/receipt.txt"; else echo "api=FAIL" >> "$BASE/receipt.txt"; fi
cp -f "$BASE/api-chat.log" "$BASE/api-chat-evidence.log" 2>/dev/null || true
if [[ -f "$BASE/llama.ini.backup" ]]; then
  ./scripts/deploy.sh upload-file "$BASE/llama.ini.backup" "$PFN" "" llama.ini >/dev/null
fi
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo SEQGATES_DONE
