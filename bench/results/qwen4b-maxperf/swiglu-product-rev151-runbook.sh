#!/usr/bin/env bash
# Plan 006 C4 product trial gates (rev151): FFN split-SWIGLU on D3D12,
# default-OFF knob d3d12swiglu.txt. Same-model deterministic OFF/ON full
# tokens + MTP proposal/acceptance comparison. Parity must be exact or the
# wiring is reverted.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/swiglu-product-rev151"
MSIX=/tmp/opencode/qwen4b-builds/rev151/xllama-rev151.msix
SHA=$(sha256sum "$MSIX" | cut -d' ' -f1)
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.151_x64__pj67f1fcj4n14
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
[[ "$SHA" != "" ]] || { echo "MSIX_MISSING" >&2; exit 1; }
mkdir -p "$BASE/csv" "$BASE/knobs"
"$REPO/scripts/deploy.sh" "$MSIX" 2>&1 | tail -3
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN want=$PFN_EXPECT" >&2; exit 1; }

export XLLAMA_MSIX_SHA256="$SHA" XLLAMA_EXPECTED_PFN="$PFN"
printf 'xbox-series-x' > "$BASE/knobs/bench_host_tag.txt"
printf '0' > "$BASE/knobs/d3d12q6tile.txt"
printf '2' > "$BASE/knobs/cpurepackforcegemv.txt"
printf 'auto' > "$BASE/knobs/d3d12twocol.txt"
printf '1' > "$BASE/knobs/d3d12gdn.txt"
printf '1' > "$BASE/knobs/d3d12q8.txt"
printf '2' > "$BASE/knobs/flashattn.txt"
printf '1' > "$BASE/knobs/mtp_threads.txt"
printf '0' > "$BASE/knobs/mtp_catchup_logits.txt"
for name in bench_host_tag d3d12q6tile cpurepackforcegemv d3d12twocol d3d12gdn d3d12q8 flashattn mtp_threads mtp_catchup_logits; do
  ./scripts/deploy.sh upload-file "$BASE/knobs/$name.txt" "$PFN" "" "$name.txt" >/dev/null
done
{ echo "sha256=$SHA"; echo "pfn=$PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name prompt mtp sw
  local name="$1" prompt="$2" mtp="$3" sw="$4"
  printf '%s' "$sw" > "$BASE/knobs/d3d12swiglu.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=$mtp swiglu=$sw utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

run_cell chat64-mtp-sw0 spec-chat-open.txt 2 0
run_cell chat64-mtp-sw1 spec-chat-open.txt 2 1
run_cell code64-mtp-sw1 spec-code-edit.txt 2 1
run_cell code64-mtp-sw0 spec-code-edit.txt 2 0
run_cell chat64-seq-sw0 spec-chat-open.txt 0 0
run_cell chat64-seq-sw1 spec-chat-open.txt 0 1
run_cell code64-seq-sw1 spec-code-edit.txt 0 1
run_cell code64-seq-sw0 spec-code-edit.txt 0 0

check_eq() {
  local a="$1" b="$2" got_a got_b
  got_a=$(sha256sum "$a" | cut -d' ' -f1)
  got_b=$(sha256sum "$b" | cut -d' ' -f1)
  [[ "$got_a" == "$got_b" ]] || { echo "PARITY_FAIL $a vs $b ($got_a vs $got_b)" | tee -a "$BASE/receipt.txt"; return 1; }
}
for r in 2 3; do
  check_eq "$BASE/csv/chat64-mtp-sw0.run$r.tokens" "$BASE/csv/chat64-mtp-sw1.run$r.tokens"
  check_eq "$BASE/csv/code64-mtp-sw0.run$r.tokens" "$BASE/csv/code64-mtp-sw1.run$r.tokens"
  check_eq "$BASE/csv/chat64-seq-sw0.run$r.tokens" "$BASE/csv/chat64-seq-sw1.run$r.tokens"
  check_eq "$BASE/csv/code64-seq-sw0.run$r.tokens" "$BASE/csv/code64-seq-sw1.run$r.tokens"
done
echo "parity=OK" >> "$BASE/receipt.txt"

# MTP proposal/acceptance + knob engagement from the last segment per cell.
last_line() { grep -a "$2" "$BASE/logs/$1/run3.log" 2>/dev/null | tail -1; }
for c in chat64-mtp-sw0 chat64-mtp-sw1 code64-mtp-sw0 code64-mtp-sw1; do
  echo "mtp=$c '$(last_line "$c" "MTP_STATS" | grep -ao "rounds=[0-9]* decodes=[0-9]* discarded=[0-9]* catchup_tok=[0-9]*")'" >> "$BASE/receipt.txt"
  echo "knob=$c '$(last_line "$c" "FFN SWIGLU D3D12=" | grep -ao "FFN SWIGLU D3D12=[a-z]*")'" >> "$BASE/receipt.txt"
done
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo SWIGLU_GATES_DONE
