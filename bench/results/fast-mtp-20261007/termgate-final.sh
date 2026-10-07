#!/usr/bin/env bash
# Final profile termination gate: sequential control and selected MTP, same GPU/CPU knobs.
# Usage: termgate-final.sh <msix_sha256> <fresh-output-directory> [q6-columns=0]
set -euo pipefail
cd /home/hjotha/worktrees/xllama-mtp
set -a
# shellcheck source=/dev/null
source "$HOME/.config/xllama/xbox-env"
set +a
export XLLAMA_MSIX_SHA256="${1:?msix sha required}"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "${XLLAMA_EXPECTED_PFN:?expected pfn required}" ]] || { echo 'PFN mismatch' >&2; exit 1; }
OUT="${2:?fresh output directory required}"
[[ ! -e "$OUT" ]] || { echo 'Output directory exists; retain it and choose a new path' >&2; exit 1; }
mkdir -p "$OUT"
Q6="${3:-0}"
SCRATCH=$(mktemp -d)
BACKUP=$(mktemp -d)
for knob in d3d12twocol.txt cpurepackforcegemv.txt; do
  ./scripts/deploy.sh fetch-file "$PFN" "$knob" "$BACKUP/$knob" >/dev/null 2>&1 || true
 done
restore_profile() {
  for knob in d3d12twocol.txt cpurepackforcegemv.txt; do
    if [[ -f "$BACKUP/$knob" ]]; then
      ./scripts/deploy.sh upload-file "$BACKUP/$knob" "$PFN" "" "$knob" >/dev/null 2>&1 || true
    else
      ./scripts/deploy.sh delete-file "$PFN" "$knob" >/dev/null 2>&1 || true
    fi
  done
  rm -rf "$BACKUP" "$SCRATCH"
}
trap restore_profile EXIT
for entry in bench_threads.txt:2 bench_ctx.txt:2048 bench_gpu_layers.txt:34 bench_n_batch.txt:64 bench_ubatch.txt:64 bench_mtp_pmin.txt:50 bench_profile.txt:0 mtp_threads.txt:1 d3d12q8.txt:1 d3d12gdn.txt:1 flashattn.txt:2 mtp_catchup_logits.txt:0 d3d12twocol.txt:auto cpurepackforcegemv.txt:2 "d3d12q6tile.txt:$Q6" model.txt:qwen35-4b-mtp; do
  name="${entry%%:*}"; value="${entry#*:}"
  printf '%s' "$value" > "$SCRATCH/$name"
  ./scripts/deploy.sh upload-file "$SCRATCH/$name" "$PFN" "" "$name" >/dev/null 2>&1
done

mkdir -p "$OUT"
printf '%s' "$1" > "$OUT/msix.txt"
./scripts/deploy.sh upload-file "$OUT/msix.txt" "$PFN" "" "replay_msix_sha.txt" >/dev/null 2>&1
mkdir -p "$OUT/seq" "$OUT/cand"
./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
for arm in seq cand; do
  # arm knobs
  case $arm in
    seq)  echo 0 > "$OUT/bench_mtp.txt" ;;
    cand) echo 2 > "$OUT/bench_mtp.txt" ;;
  esac
  ./scripts/deploy.sh upload-file "$OUT/bench_mtp.txt" "$PFN" "" "bench_mtp.txt" >/dev/null 2>&1
  printf 'cancel,stop,eog' > "$OUT/termgate.txt"
  ./scripts/deploy.sh upload-file "$OUT/termgate.txt" "$PFN" "" "termgate.txt" >/dev/null 2>&1
  ./scripts/deploy.sh delete-file "$PFN" "termgate-result.csv" >/dev/null 2>&1 || true
  ./scripts/deploy.sh delete-file "$PFN" "termgate-result.csv.done" >/dev/null 2>&1 || true
  printf '' > "$OUT/termgate.flag"
  ./scripts/deploy.sh upload-file "$OUT/termgate.flag" "$PFN" "" "termgate.flag" >/dev/null 2>&1
  echo "=== ARM=$arm starting"
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1
  for i in $(seq 1 90); do
    if ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv.done" "$OUT/${arm}.done" 2>&1 | grep -q "^Fetched"; then
      echo "ARM=$arm done at poll $i"
      break
    fi
    ST=$(./scripts/deploy.sh status "$PFN" 2>&1 | head -n 1)
    if echo "$ST" | grep -q "not running"; then
      echo "ARM=$arm APP_EXIT at poll $i"
      ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv.done" "$OUT/${arm}.done" >/dev/null 2>&1 || true
      ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv" "$OUT/${arm}.csv" >/dev/null 2>&1 || true
      break
    fi
    sleep 20
    if [ $((i % 6)) -eq 0 ]; then echo "  waiting... $i ($ST)"; fi
  done
  [[ -f "$OUT/${arm}.done" ]] || { echo "Marker missing: preserve running app and stop the driver" >&2; exit 1; }
  # Fetch this arm's result + all its dumps BEFORE the next arm overwrites them.
  ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv" "$OUT/${arm}.csv" >/dev/null 2>&1 || true
  for f in termgate-cancel6-t1.txt termgate-cancel6-t1.txt.prefill termgate-cancel6-resume.txt \
           termgate-cancel6-resume.txt.prefill termgate-cancel6-seqref.txt termgate-cancel6-seqref.txt.prefill \
           termgate-cancel13-t1.txt termgate-cancel13-t1.txt.prefill termgate-cancel13-resume.txt \
           termgate-cancel13-resume.txt.prefill termgate-cancel13-seqref.txt termgate-cancel13-seqref.txt.prefill \
           termgate-stop-probe.txt termgate-stop-probe.txt.prefill termgate-stop-run.txt termgate-stop-run.txt.prefill \
           termgate-stop-resume.txt termgate-stop-resume.txt.prefill termgate-stop-cold.txt termgate-stop-cold.txt.prefill \
           termgate-eog0.txt termgate-eog1.txt termgate-eog2.txt termgate-eog3.txt; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$OUT/${arm}/$f" >/dev/null 2>&1 || true
  done
  ./scripts/deploy.sh get-log "$PFN" > "$OUT/${arm}.log" 2>/dev/null || true
  # Strip to this arm's segment (TERM_GATE_CONFIG .. termgate done) later on host.
  echo "=== ARM=$arm fetched ($(python3 -c 'import os,sys; print(len(os.listdir(sys.argv[1])))' "$OUT/${arm}") dumps, csv $(wc -c < "$OUT/${arm}.csv" 2>/dev/null || echo 0) bytes)"
done
python3 bench/results/fast-mtp-20261007/check-termgate.py "$OUT" "$Q6"
# The EXIT trap restores the user profile that existed before the test.
echo "TERMGATE SELECTED-PROFILE COMPLETE"
