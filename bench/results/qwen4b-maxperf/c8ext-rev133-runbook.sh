#!/usr/bin/env bash
# Plan 006 C8 extension on rev133 (no rebuild): profile-bound timestamp default
# (knob absent => OFF at profile OFF) vs explicit ON, on code64 and chat256,
# opposite-order blocks, parity required. Two profile-ON cells at the end feed
# the same-boundary attribution for the next bottleneck.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/c8ext-rev133"
MSIX=/tmp/opencode/qwen4b-builds/rev133/xllama-rev133.msix
SHA=47b274695a1882eec2d02d1829af8d2e4117748ce70dd471914d1ad55d561b41
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.133_x64__pj67f1fcj4n14
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

run_cell() { # name prompt mtp ts predict profile
  local name="$1" prompt="$2" mtp="$3" ts="$4" predict="$5" profile="$6"
  if [[ "$ts" == "default" ]]; then
    ./scripts/deploy.sh delete-file "$PFN" d3d12timestamps.txt >/dev/null 2>&1 || true
  else
    printf '%s' "$ts" > "$BASE/knobs/d3d12timestamps.txt"
    ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12timestamps.txt" "$PFN" "" d3d12timestamps.txt >/dev/null
  fi
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=$mtp ts=$ts predict=$predict prof=$profile utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases "$profile" --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 2 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

# Block 1.
run_cell code64-seq-def spec-code-edit.txt 0 default 64 0
run_cell code64-seq-on spec-code-edit.txt 0 1 64 0
run_cell code64-mtp-def spec-code-edit.txt 2 default 64 0
run_cell code64-mtp-on spec-code-edit.txt 2 1 64 0
run_cell chat256-seq-def spec-chat-open.txt 0 default 256 0
run_cell chat256-seq-on spec-chat-open.txt 0 1 256 0
run_cell chat256-mtp-def spec-chat-open.txt 2 default 256 0
run_cell chat256-mtp-on spec-chat-open.txt 2 1 256 0
# Block 2 reversed.
run_cell chat256-mtp-on-b2 spec-chat-open.txt 2 1 256 0
run_cell chat256-mtp-def-b2 spec-chat-open.txt 2 default 256 0
run_cell chat256-seq-on-b2 spec-chat-open.txt 0 1 256 0
run_cell chat256-seq-def-b2 spec-chat-open.txt 0 default 256 0
run_cell code64-mtp-on-b2 spec-code-edit.txt 2 1 64 0
run_cell code64-mtp-def-b2 spec-code-edit.txt 2 default 64 0
run_cell code64-seq-on-b2 spec-code-edit.txt 0 1 64 0
run_cell code64-seq-def-b2 spec-code-edit.txt 0 default 64 0
# Same-boundary attribution cells (profile ON; absent knob => timestamps ON).
run_cell code64-mtp-prof1 spec-code-edit.txt 2 default 64 1
run_cell chat256-mtp-prof1 spec-chat-open.txt 2 default 256 1

check() {
  local f="$1" want="$2" got
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }
}
for c in code64-seq-def code64-seq-on code64-mtp-def code64-mtp-on code64-mtp-on-b2 code64-mtp-def-b2 code64-seq-on-b2 code64-seq-def-b2 code64-mtp-prof1; do
  check "$BASE/csv/$c.run2.tokens" ff42c5beec71ff491b458d0e48cbac4d98f8cdcf8153c78868c64aee687a4010
done
for c in chat256-seq-def chat256-seq-on chat256-mtp-def chat256-mtp-on chat256-mtp-on-b2 chat256-mtp-def-b2 chat256-seq-on-b2 chat256-seq-def-b2 chat256-mtp-prof1; do
  check "$BASE/csv/$c.run2.tokens" f78ce8372ccc7281aaad5e726bf7c88a0317212e3b2ed397fb0fd83ffd53066c
done
echo "parity=OK" >> "$BASE/receipt.txt"

last_line() { grep -a "$2" "$BASE/logs/$1/run2.log" 2>/dev/null | tail -1; }
for c in code64-seq-def code64-mtp-def chat256-seq-def chat256-mtp-def chat256-mtp-def-b2 chat256-seq-def-b2 code64-mtp-def-b2 code64-seq-def-b2 code64-mtp-prof1 chat256-mtp-prof1; do
  echo "engage=$c knob='$(last_line "$c" "gpu timestamps=" | grep -ao "gpu timestamps=[a-z]* ([a-z-]* [a-z]*)" | tail -1)'" >> "$BASE/receipt.txt"
done
for c in code64-seq-on code64-mtp-on chat256-seq-on chat256-mtp-on chat256-mtp-on-b2 chat256-seq-on-b2 code64-mtp-on-b2 code64-seq-on-b2; do
  echo "engage=$c knob='$(last_line "$c" "gpu timestamps=" | grep -ao "gpu timestamps=[a-z]* ([a-z-]* [a-z]*)" | tail -1)'" >> "$BASE/receipt.txt"
done

./scripts/deploy.sh fetch-file "$PFN" xllama.log "$BASE/post-run-full.log" >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo C8EXT_DONE
