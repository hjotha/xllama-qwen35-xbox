#!/usr/bin/env bash
# Termgate 3-arm driver: seq / ref / cand, preserving each arm's dumps.
# Usage: termgate-run.sh <msix_sha256>
set -euo pipefail
cd /home/hjotha/worktrees/xllama-mtp
set -a; source ~/.config/xllama/xbox-env; set +a
export XLLAMA_MSIX_SHA256="${1:?msix sha required}"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
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
  rm -rf "$BACKUP"
}
trap restore_profile EXIT
mkdir -p /tmp/xllama-rev95
printf '%s' "$1" > /tmp/xllama-rev95/msix.txt
./scripts/deploy.sh upload-file /tmp/xllama-rev95/msix.txt "$PFN" "" "replay_msix_sha.txt" >/dev/null 2>&1
rm -rf /tmp/xllama-rev95/seq /tmp/xllama-rev95/ref /tmp/xllama-rev95/cand
mkdir -p /tmp/xllama-rev95/seq /tmp/xllama-rev95/ref /tmp/xllama-rev95/cand
for arm in seq ref cand; do
  # arm knobs
  case $arm in
    seq)  echo 0 > /tmp/xllama-rev95/bench_mtp.txt ;;
    ref|cand) echo 4 > /tmp/xllama-rev95/bench_mtp.txt ;;
  esac
  ./scripts/deploy.sh upload-file /tmp/xllama-rev95/bench_mtp.txt "$PFN" "" "bench_mtp.txt" >/dev/null 2>&1
  ./scripts/deploy.sh delete-file "$PFN" "d3d12twocol.txt" >/dev/null 2>&1 || true
  ./scripts/deploy.sh delete-file "$PFN" "cpurepackforcegemv.txt" >/dev/null 2>&1 || true
  if [ "$arm" = "cand" ]; then
    printf 'auto' > /tmp/xllama-rev95/twocol.txt
    ./scripts/deploy.sh upload-file /tmp/xllama-rev95/twocol.txt "$PFN" "" "d3d12twocol.txt" >/dev/null 2>&1
    printf '2' > /tmp/xllama-rev95/scope.txt
    ./scripts/deploy.sh upload-file /tmp/xllama-rev95/scope.txt "$PFN" "" "cpurepackforcegemv.txt" >/dev/null 2>&1
  fi
  printf 'cancel,stop,eog' > /tmp/xllama-rev95/termgate.txt
  ./scripts/deploy.sh upload-file /tmp/xllama-rev95/termgate.txt "$PFN" "" "termgate.txt" >/dev/null 2>&1
  ./scripts/deploy.sh delete-file "$PFN" "termgate-result.csv" >/dev/null 2>&1 || true
  ./scripts/deploy.sh delete-file "$PFN" "termgate-result.csv.done" >/dev/null 2>&1 || true
  printf '' > /tmp/xllama-rev95/termgate.flag
  ./scripts/deploy.sh upload-file /tmp/xllama-rev95/termgate.flag "$PFN" "" "termgate.flag" >/dev/null 2>&1
  echo "=== ARM=$arm starting"
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1
  for i in $(seq 1 90); do
    if ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv.done" "/tmp/xllama-rev95/${arm}.done" 2>&1 | grep -q "^Fetched"; then
      echo "ARM=$arm done at poll $i"
      break
    fi
    ST=$(./scripts/deploy.sh status "$PFN" 2>&1 | head -n 1)
    if echo "$ST" | grep -q "not running"; then
      echo "ARM=$arm APP_EXIT at poll $i"
      ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv.done" "/tmp/xllama-rev95/${arm}.done" >/dev/null 2>&1 || true
      ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv" "/tmp/xllama-rev95/${arm}.csv" >/dev/null 2>&1 || true
      break
    fi
    sleep 20
    if [ $((i % 6)) -eq 0 ]; then echo "  waiting... $i ($ST)"; fi
  done
  # Fetch this arm's result + all its dumps BEFORE the next arm overwrites them.
  ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv" "/tmp/xllama-rev95/${arm}.csv" >/dev/null 2>&1 || true
  for f in termgate-cancel6-t1.txt termgate-cancel6-t1.txt.prefill termgate-cancel6-resume.txt \
           termgate-cancel6-resume.txt.prefill termgate-cancel6-seqref.txt termgate-cancel6-seqref.txt.prefill \
           termgate-cancel13-t1.txt termgate-cancel13-t1.txt.prefill termgate-cancel13-resume.txt \
           termgate-cancel13-resume.txt.prefill termgate-cancel13-seqref.txt termgate-cancel13-seqref.txt.prefill \
           termgate-stop-probe.txt termgate-stop-probe.txt.prefill termgate-stop-run.txt termgate-stop-run.txt.prefill \
           termgate-stop-resume.txt termgate-stop-resume.txt.prefill termgate-stop-cold.txt termgate-stop-cold.txt.prefill \
           termgate-eog0.txt termgate-eog1.txt termgate-eog2.txt termgate-eog3.txt; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "/tmp/xllama-rev95/${arm}/$f" >/dev/null 2>&1 || true
  done
  ./scripts/deploy.sh get-log "$PFN" > "/tmp/xllama-rev95/${arm}.log" 2>/dev/null || true
  # Strip to this arm's segment (TERM_GATE_CONFIG .. termgate done) later on host.
  echo "=== ARM=$arm fetched ($(ls /tmp/xllama-rev95/${arm} | wc -l) dumps, csv $(wc -c < /tmp/xllama-rev95/${arm}.csv 2>/dev/null || echo 0) bytes)"
done
# The EXIT trap restores the user profile that existed before the test.
echo "TERMGATE 3-ARM COMPLETE"
