#!/usr/bin/env bash
# Plan 006 C4 product-trial gates (rev153): FFN split-SWIGLU on D3D12 behind
# the default-OFF d3d12swiglu.txt knob. Engagement was smoke-verified
# (ON 2134 dispatches, OFF zero, token parity). This run does:
#   - profile-ON MTP pairs: per-round VROUND accepted sequences + full IDs
#   - profile-OFF pairs: timing (seq + MTP, chat64 + code64)
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/swiglu-gates-rev154"
MSIX=/tmp/opencode/qwen4b-builds/rev154/xllama-rev154.msix
SHA=$(sha256sum "$MSIX" | cut -d' ' -f1)
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.154_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a

if ! ping -c 1 -W 2 "$XBOX_IP" >/dev/null 2>&1; then
  echo "XBOX_OFFLINE: not starting" >&2
  exit 1
fi
http_code=$(curl --basic -u "$XBOX_USER:$XBOX_PASS" -k -sS --connect-timeout 5 --max-time 10 \
  "https://$XBOX_IP:11443/" -o /dev/null -w '%{http_code}' 2>/dev/null) || http_code=000
case "$http_code" in
2*) ;;
*)
  echo "WDP_NOT_READY: http=${http_code} (need authenticated 2xx); not starting" >&2
  exit 1
  ;;
esac
mkdir -p "$BASE/csv" "$BASE/knobs" "$BASE/logs"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN want=$PFN_EXPECT" >&2; exit 1; }

export XLLAMA_MSIX_SHA256="$SHA" XLLAMA_EXPECTED_PFN="$PFN"
{ echo "sha256=$SHA"; echo "pfn=$PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name prompt mtp sw profile
  local name="$1" prompt="$2" mtp="$3" sw="$4" profile="$5"
  printf '%s' "$sw" > "$BASE/knobs/d3d12swiglu.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=$mtp sw=$sw prof=$profile utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$profile" --mtp "$mtp" --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

# Profile-ON MTP pairs: per-round acceptance + full IDs.
run_cell chat64-mtp-p1-sw0 spec-chat-open.txt 2 0 1
run_cell chat64-mtp-p1-sw1 spec-chat-open.txt 2 1 1
run_cell code64-mtp-p1-sw1 spec-code-edit.txt 2 1 1
run_cell code64-mtp-p1-sw0 spec-code-edit.txt 2 0 1
# Profile-OFF timing pairs.
run_cell chat64-mtp-sw0 spec-chat-open.txt 2 0 0
run_cell chat64-mtp-sw1 spec-chat-open.txt 2 1 0
run_cell code64-mtp-sw1 spec-code-edit.txt 2 1 0
run_cell code64-mtp-sw0 spec-code-edit.txt 2 0 0
run_cell chat64-seq-sw0 spec-chat-open.txt 0 0 0
run_cell chat64-seq-sw1 spec-chat-open.txt 0 1 0
run_cell code64-seq-sw1 spec-code-edit.txt 0 1 0
run_cell code64-seq-sw0 spec-code-edit.txt 0 0 0

check_eq() {
  local a="$1" b="$2" got_a got_b
  got_a=$(sha256sum "$a" | cut -d' ' -f1)
  got_b=$(sha256sum "$b" | cut -d' ' -f1)
  [[ "$got_a" == "$got_b" ]] || { echo "PARITY_FAIL $a vs $b ($got_a vs $got_b)" | tee -a "$BASE/receipt.txt"; return 1; }
}
for r in 2 3; do
  check_eq "$BASE/csv/chat64-mtp-p1-sw0.run$r.tokens" "$BASE/csv/chat64-mtp-p1-sw1.run$r.tokens"
  check_eq "$BASE/csv/code64-mtp-p1-sw0.run$r.tokens" "$BASE/csv/code64-mtp-p1-sw1.run$r.tokens"
  check_eq "$BASE/csv/chat64-mtp-sw0.run$r.tokens" "$BASE/csv/chat64-mtp-sw1.run$r.tokens"
  check_eq "$BASE/csv/code64-mtp-sw0.run$r.tokens" "$BASE/csv/code64-mtp-sw1.run$r.tokens"
  check_eq "$BASE/csv/chat64-seq-sw0.run$r.tokens" "$BASE/csv/chat64-seq-sw1.run$r.tokens"
  check_eq "$BASE/csv/code64-seq-sw0.run$r.tokens" "$BASE/csv/code64-seq-sw1.run$r.tokens"
done
echo "parity=OK" >> "$BASE/receipt.txt"

# Engagement + MTP aggregates from the last segment per cell.
last_line() { grep -a "$2" "$BASE/logs/$1/run3.log" 2>/dev/null | tail -1; }
for c in chat64-mtp-p1-sw0 chat64-mtp-p1-sw1 code64-mtp-p1-sw0 code64-mtp-p1-sw1; do
  echo "engage=$c knob='$(last_line "$c" "FFN SWIGLU D3D12=" | grep -ao "FFN SWIGLU D3D12=[a-z]*")' disp='$(last_line "$c" "FFN SWIGLU dispatches" | grep -ao "[0-9]* FFN SWIGLU dispatches")'" >> "$BASE/receipt.txt"
  echo "mtp=$c '$(last_line "$c" "MTP_STATS" | grep -ao "rounds=[0-9]* decodes=[0-9]* discarded=[0-9]*")'" >> "$BASE/receipt.txt"
done
for c in chat64-mtp-sw0 chat64-mtp-sw1 code64-mtp-sw0 code64-mtp-sw1; do
  echo "knob=$c '$(last_line "$c" "FFN SWIGLU D3D12=" | grep -ao "FFN SWIGLU D3D12=[a-z]*")'" >> "$BASE/receipt.txt"
done
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo SWIGLU_GATES_REV154_DONE
