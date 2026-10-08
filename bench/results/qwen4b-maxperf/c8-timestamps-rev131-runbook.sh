#!/usr/bin/env bash
# Plan 006 C8 A/B: GPU timestamp queries ON (default) vs OFF, profile OFF
# everywhere. Paired cells, ON-OFF-ON sequences per workload, plus a
# reverse-order block; full-ID parity required; knob engagement and
# unavailable-counter semantics checked from the run logs.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/c8-timestamps-rev131"
MSIX=/tmp/opencode/qwen4b-builds/rev131/xllama-rev131.msix
SHA=f97e95e3fc777c240d547104cd6cd11933ffecc3c81e6268cd0952f1f3ecedb2
PFN_EXPECT=GianlucaMazza.xllama_1.6.0.131_x64__pj67f1fcj4n14
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
for name in bench_host_tag d3d12q6tile cpurepackforcegemv d3d12twocol d3d12gdn d3d12q8 flashattn mtp_threads mtp_catchup_logits; do
  ./scripts/deploy.sh upload-file "$BASE/knobs/$name.txt" "$PFN" "" "$name.txt" >/dev/null
done
sha256sum "$BASE"/knobs/* > "$BASE/knobs-sha256.txt"
{ echo "sha256=$SHA"; echo "pfn=$PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name mtp ts
  local name="$1" mtp="$2" ts="$3"
  printf '%s' "$ts" > "$BASE/knobs/d3d12timestamps.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12timestamps.txt" "$PFN" "" d3d12timestamps.txt >/dev/null
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=$mtp ts=$ts utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict 64 \
    --prompt "$REPO/bench/prompts/spec-chat-open.txt" \
    --out "$BASE/csv/$name.csv" --runs 3 >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

# ON-OFF-ON sequences per workload, then a reverse-order paired block.
run_cell seq-tson 0 1
run_cell seq-tsoff 0 0
run_cell seq-tson-b2 0 1
run_cell mtp-tson 2 1
run_cell mtp-tsoff 2 0
run_cell mtp-tson-b2 2 1
run_cell mtp-tsoff-b2 2 0
run_cell mtp-tson-b3 2 1
run_cell seq-tsoff-b2 0 0
run_cell seq-tson-b3 0 1

check() {
  local f="$1" want="$2" got
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }
}
for c in seq-tson seq-tsoff seq-tson-b2 mtp-tson mtp-tsoff mtp-tson-b2 mtp-tsoff-b2 mtp-tson-b3 seq-tsoff-b2 seq-tson-b3; do
  for r in 2 3; do
    check "$BASE/csv/$c.run$r.tokens" 84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2
  done
done
echo "parity=OK" >> "$BASE/receipt.txt"

# Knob engagement + counter semantics, extracted from the LAST log segment in
# each cell's final run log. The device log ACCUMULATES across launches, so
# grep counts (3/6/9...) are not per-cell proof; only the last knob line and
# the last main-context summary after it are interpreted.
last_knob() { grep -ao "gpu timestamps=[a-z]*" "$1" 2>/dev/null | tail -1; }
last_summary_after_knob() {
  awk '/gpu timestamps=/{kn=NR} /d3d12: [0-9]+ graph_compute calls/{if(NR>kn){s=$0;sn=NR}} END{print s}' \
    "$1" 2>/dev/null
}
for c in seq-tson seq-tson-b2 mtp-tson mtp-tson-b2 mtp-tson-b3 seq-tson-b3; do
  k=$(last_knob "$BASE/logs/$c/run3.log")
  s=$(last_summary_after_knob "$BASE/logs/$c/run3.log")
  ok=0; [[ "$k" == *"=on"* && "$s" == *"ts_valid="* && "$s" != *"timing unavailable:"* ]] && ok=1
  echo "engage=$c knob='$k' complete=$ok" >> "$BASE/receipt.txt"
done
for c in seq-tsoff seq-tsoff-b2 mtp-tsoff mtp-tsoff-b2; do
  k=$(last_knob "$BASE/logs/$c/run3.log")
  s=$(last_summary_after_knob "$BASE/logs/$c/run3.log")
  ok=0; [[ "$k" == *"=off"* && "$s" == *"timing unavailable:"* ]] && ok=1
  echo "engage=$c knob='$k' unavailable=$ok" >> "$BASE/receipt.txt"
done

./scripts/deploy.sh fetch-file "$PFN" xllama.log "$BASE/post-run-full.log" >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo C8_AB_DONE
