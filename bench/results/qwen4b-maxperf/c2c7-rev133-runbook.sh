#!/usr/bin/env bash
# Plan 006 rev133: (1) verify ACTUAL effective thread counts from the context,
# (2) C2 draft-thread A/B (mtp_threads 1 vs 2), (3) C7 ubatch 32 vs 64 on the
# long prompt. Timestamps fixed OFF, profile OFF, parity required.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/c2c7-rev133"
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
for name in bench_host_tag d3d12q6tile cpurepackforcegemv d3d12twocol d3d12gdn d3d12q8 flashattn mtp_catchup_logits d3d12timestamps; do
  ./scripts/deploy.sh upload-file "$BASE/knobs/$name.txt" "$PFN" "" "$name.txt" >/dev/null
done
sha256sum "$BASE"/knobs/* > "$BASE/knobs-sha256.txt"
{ echo "sha256=$SHA"; echo "pfn=$PFN"; echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$BASE/receipt.txt"

run_cell() { # name prompt mtp mtp_threads batch_threads ubatch predict runs
  local name="$1" prompt="$2" mtp="$3" mt="$4" bt="$5" ub="$6" predict="$7" runs="$8"
  printf '%s' "$mt" > "$BASE/knobs/mtp_threads.txt"
  ./scripts/deploy.sh upload-file "$BASE/knobs/mtp_threads.txt" "$PFN" "" mtp_threads.txt >/dev/null
  if [[ "$bt" == "absent" ]]; then
    ./scripts/deploy.sh delete-file "$PFN" bench_batch_threads.txt >/dev/null 2>&1 || true
  else
    printf '%s' "$bt" > "$BASE/knobs/bench_batch_threads.txt"
    ./scripts/deploy.sh upload-file "$BASE/knobs/bench_batch_threads.txt" "$PFN" "" bench_batch_threads.txt >/dev/null
  fi
  export RUN_LOG_DIR="$BASE/logs/$name"
  mkdir -p "$RUN_LOG_DIR"
  echo "cell=$name mtp=$mtp mtp_threads=$mt batch=$bt ubatch=$ub predict=$predict runs=$runs utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch "$ub" --ubatch "$ub" \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs "$runs" >> "$BASE/driver.log" 2>&1
  echo "done=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
}

# 1) Effective thread verification: knob absent -> batch == decode; knob 4 -> 4.
run_cell verify-mtp-b0 spec-chat-open.txt 2 1 absent 64 64 2
run_cell verify-mtp-b4 spec-chat-open.txt 2 1 4 64 64 2
# 2) C2: draft threads 1 vs 2 (t2 decode, t2 batch, timestamps OFF).
run_cell c2-mtp-t1 spec-chat-open.txt 2 1 absent 64 64 3
run_cell c2-mtp-t2 spec-chat-open.txt 2 2 absent 64 64 3
run_cell c2-mtp-t2-b2 spec-chat-open.txt 2 2 absent 64 64 3
run_cell c2-mtp-t1-b2 spec-chat-open.txt 2 1 absent 64 64 3
# 3) C7: ubatch 32 vs 64, long prompt, seq.
run_cell c7-std-u32 standard-512.txt 0 1 absent 32 64 3
run_cell c7-std-u64 standard-512.txt 0 1 absent 64 64 3
run_cell c7-std-u64-b2 standard-512.txt 0 1 absent 64 64 3
run_cell c7-std-u32-b2 standard-512.txt 0 1 absent 32 64 3

check() {
  local f="$1" want="$2" got
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [[ "$got" == "$want" ]] || { echo "PARITY_FAIL $f got=$got want=$want" | tee -a "$BASE/receipt.txt"; return 1; }
}
for c in verify-mtp-b0 verify-mtp-b4; do
  check "$BASE/csv/$c.run2.tokens" 84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2
done
for c in c2-mtp-t1 c2-mtp-t2 c2-mtp-t2-b2 c2-mtp-t1-b2; do
  for r in 2 3; do
    check "$BASE/csv/$c.run$r.tokens" 84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2
  done
done
for c in c7-std-u32 c7-std-u64 c7-std-u64-b2 c7-std-u32-b2; do
  for r in 2 3; do
    check "$BASE/csv/$c.run$r.tokens" 44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732
  done
done
echo "parity=OK" >> "$BASE/receipt.txt"

# Engagement from the LAST segment only (the device log accumulates).
last_line() { grep -a "$2" "$BASE/logs/$1/run3.log" 2>/dev/null | tail -1; }
for c in verify-mtp-b0 verify-mtp-b4; do
  echo "engage=$c eff='$(last_line "$c" "effective threads: decode=" | grep -ao "decode=[0-9]* batch=[0-9]*")'" >> "$BASE/receipt.txt"
done
for c in c2-mtp-t1 c2-mtp-t2 c2-mtp-t2-b2 c2-mtp-t1-b2; do
  echo "engage=$c draft='$(last_line "$c" "draft CPU threads=" | grep -ao "draft CPU threads=[0-9]*")'" >> "$BASE/receipt.txt"
done
for c in c7-std-u32 c7-std-u64 c7-std-u64-b2 c7-std-u32-b2; do
  echo "engage=$c pref='$(last_line "$c" "prefill batch override:" | grep -ao "n_batch=[0-9]* n_ubatch=[0-9]*")'" >> "$BASE/receipt.txt"
done

./scripts/deploy.sh fetch-file "$PFN" xllama.log "$BASE/post-run-full.log" >/dev/null 2>&1 || true
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
echo C2C7_DONE
