#!/usr/bin/env bash
# Plan 006 C1 A/B: batch/decode thread split. Decode threads fixed at 2;
# batch (prefill/verify/catch-up) threads 2 vs 4 vs 6. Timestamps fixed OFF
# (d3d12timestamps.txt=0), profile OFF everywhere, full-ID parity required.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/c1-threads-rev132"
MSIX=/tmp/opencode/qwen4b-builds/rev132/xllama-rev132.msix
SHA=c97a109be5f12f8b2a44d017a15424b799b1cb74f795a73a09aec9cccfaa5914
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.132_x64__pj67f1fcj4n14
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
printf '0' > "$BASE/knobs/d3d12timestamps.txt"
for name in bench_host_tag d3d12q6tile cpurepackforcegemv d3d12twocol d3d12gdn d3d12q8 flashattn mtp_threads mtp_catchup_logits d3d12timestamps; do
  ./scripts/deploy.sh upload-file "$BASE/knobs/$name.txt" "$PFN" "" "$name.txt" >/dev/null
done
sha256sum "$BASE"/knobs/* > "$BASE/knobs-sha256.txt"
{ echo "sha256=$SHA"; echo "pfn=$PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name mtp batch_threads predict
  local name="$1" mtp="$2" bt="$3" predict="$4"
  printf '%s' "$bt" > "$BASE/knobs/bench_batch_threads.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/bench_batch_threads.txt" "$PFN" "" bench_batch_threads.txt >/dev/null
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=$mtp t=2 tb=$bt predict=$predict utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$5" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

# Block 1 (chat64): seq then MTP, batch 2/4/6.
run_cell chat64-seq-t2b2 0 2 64 spec-chat-open.txt
run_cell chat64-seq-t2b4 0 4 64 spec-chat-open.txt
run_cell chat64-seq-t2b6 0 6 64 spec-chat-open.txt
run_cell chat64-mtp-t2b2 2 2 64 spec-chat-open.txt
run_cell chat64-mtp-t2b4 2 4 64 spec-chat-open.txt
run_cell chat64-mtp-t2b6 2 6 64 spec-chat-open.txt
# Block 2 (chat64) reversed.
run_cell chat64-mtp-t2b6-b2 2 6 64 spec-chat-open.txt
run_cell chat64-mtp-t2b4-b2 2 4 64 spec-chat-open.txt
run_cell chat64-mtp-t2b2-b2 2 2 64 spec-chat-open.txt
run_cell chat64-seq-t2b6-b2 0 6 64 spec-chat-open.txt
run_cell chat64-seq-t2b4-b2 0 4 64 spec-chat-open.txt
run_cell chat64-seq-t2b2-b2 0 2 64 spec-chat-open.txt
# Long prompt prefill (std-512).
run_cell std512-seq-t2b2 0 2 64 standard-512.txt
run_cell std512-seq-t2b4 0 4 64 standard-512.txt
run_cell std512-seq-t2b6 0 6 64 standard-512.txt

check() {
  local f="$1" want="$2" got
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }
}
for c in chat64-seq-t2b2 chat64-seq-t2b4 chat64-seq-t2b6 chat64-mtp-t2b2 chat64-mtp-t2b4 chat64-mtp-t2b6 \
  chat64-mtp-t2b6-b2 chat64-mtp-t2b4-b2 chat64-mtp-t2b2-b2 chat64-seq-t2b6-b2 chat64-seq-t2b4-b2 chat64-seq-t2b2-b2; do
  for r in 2 3; do
    check "$BASE/csv/$c.run$r.tokens" 84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2
  done
done
for c in std512-seq-t2b2 std512-seq-t2b4 std512-seq-t2b6; do
  for r in 2 3; do
    check "$BASE/csv/$c.run$r.tokens" 44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732
  done
done
echo "parity=OK" >> "$BASE/receipt.txt"

# Engagement from the LAST segment of each cell's final run log (the device log
# accumulates): the effective batch-thread line and the CSV host tag.
for c in chat64-seq-t2b2 chat64-seq-t2b4 chat64-seq-t2b6 chat64-mtp-t2b2 chat64-mtp-t2b4 chat64-mtp-t2b6 \
  chat64-mtp-t2b6-b2 chat64-mtp-t2b4-b2 chat64-mtp-t2b2-b2 chat64-seq-t2b6-b2 chat64-seq-t2b4-b2 chat64-seq-t2b2-b2 \
  std512-seq-t2b2 std512-seq-t2b4 std512-seq-t2b6; do
  bt=$(printf %s "$c" | grep -o -- "-t2b[0-9]" | head -1 | tr -d -- "-t2b")
  eff=$(grep -a "bench: batch threads=" "$BASE/logs/$c/run3.log" 2>/dev/null | tail -1 | grep -ao "batch threads=[0-9]*" | tail -1)
  tag=$(tail -1 "$BASE/csv/$c.csv" | awk -F, '{print $15}' | grep -o -- "-tb[0-9]*" | tail -1)
  echo "engage=$c effective='$eff' host_tag='$tag' expected_tb=$bt" >> "$BASE/receipt.txt"
done

./scripts/deploy.sh fetch-file "$PFN" xllama.log "$BASE/post-run-full.log" >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo C1_AB_DONE
