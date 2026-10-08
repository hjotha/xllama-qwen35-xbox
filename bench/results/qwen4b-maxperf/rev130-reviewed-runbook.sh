#!/usr/bin/env bash
# Plan 005 rev130-reviewed runbook: deploy + same-boundary decode-split measurement.
# Run only when the console is back. Single reachability gate, no retries;
# never relaunches an uncertain run. Profile-OFF cells are the timing pair;
# profile-ON cells exist only for attribution and their overhead is measured
# against the OFF arm on the same package.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/rev130-reviewed-dsplit"
MSIX=/tmp/opencode/qwen4b-builds/rev130/xllama-rev130.msix
SHA=8e14aa6f99434d5da954c6cfb5f461c9b8ce65ff77ebe742b5110b70d0d59112
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.130_x64__pj67f1fcj4n14
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a

# Single reachability gate: ICMP then WDP with an authenticated HTTP 2xx.
# A completed TCP/curl transfer is not enough: auth failure (401) or any
# non-2xx must stop the runbook before it touches the device.
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
[[ "$(sha256sum "$MSIX" | cut -d' ' -f1)" == "$SHA" ]] || { echo "MSIX_SHA_MISMATCH" >&2; exit 1; }

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
sha256sum "$BASE"/knobs/* > "$BASE/knobs-sha256.txt"
{ echo "sha256=$SHA"; echo "pfn=$PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name prompt mtp profile
  local name="$1" prompt="$2" mtp="$3" profile="$4"
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=$mtp prof=$profile utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$profile" --mtp "$mtp" --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

run_cell chat64-seq-off spec-chat-open.txt 0 0
run_cell chat64-mtp-off spec-chat-open.txt 2 0
run_cell chat64-seq-on  spec-chat-open.txt 0 1
run_cell chat64-mtp-on  spec-chat-open.txt 2 1
run_cell chat64-mtp-on-b2 spec-chat-open.txt 2 1
run_cell chat64-seq-on-b2 spec-chat-open.txt 0 1
run_cell chat64-mtp-off-b2 spec-chat-open.txt 2 0
run_cell chat64-seq-off-b2 spec-chat-open.txt 0 0

check() { local f="$1" want="$2" got; got=$(sha256sum "$f" | cut -d' ' -f1); [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }; }
for c in chat64-seq-off chat64-mtp-off chat64-seq-on chat64-mtp-on chat64-mtp-on-b2 chat64-seq-on-b2 chat64-mtp-off-b2 chat64-seq-off-b2; do
  for r in 2 3; do
    check "$BASE/csv/$c.run$r.tokens" 84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2
  done
done
echo "parity=OK" >> "$BASE/receipt.txt"

# One full device log after the session; the per-run logs under logs/ are the
# same file at earlier points (the device log accumulates).
./scripts/deploy.sh fetch-file "$PFN" xllama.log "$BASE/post-run-full.log" >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo REV129_RUNBOOK_DONE
