#!/usr/bin/env bash
# bench-xbox-ttarget.sh — real T_target(B) on the console (plan 003 stage 2).
#
# Uploads ttarget.flag plus the same bench knobs the session gate uses, waits for
# ttarget-result.csv.done, fetches the CSV and prints the per-B medians. The
# device measures ONE llama_decode per row on a live context; this script only
# drives it and reports. Not to be confused with the isolated matmul
# microbenchmark in bench-d3d12-selftest.sh.
#
# The wait loop is not a blind poll for the .done file: every cycle it reads the
# process state (deploy.sh status) and the console log, reports new stage lines
# (stage=reference / stage=measure / stage=done), and stops WITH a diagnosis as
# soon as the app is no longer running or the log shows a failure — instead of
# sitting on a 15-minute timeout for a file that will never appear.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-xbox-ttarget.sh [--out FILE] [--reps N] [--threads N] [--ctx N]
#                                   [--ubatch N] [--widths 1,2,3,5] [--evidence DIR]
#                                   [--twocol off|auto]
# --twocol auto uploads d3d12twocol.txt ("auto") so real decode in this run uses
# the allowlisted two-column tile (variant 2); off (default) deletes any stale
# knob file for a clean OLD arm. Same package, paired arms, no rebuild.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/tttarget.csv"
REPS=3
THREADS=6
CTX=2048
UBATCH=64
EVIDENCE=""
WIDTHS=""
TWOCOL="off"
while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		OUT="$2"
		shift 2
		;;
	--reps)
		REPS="$2"
		shift 2
		;;
	--threads)
		THREADS="$2"
		shift 2
		;;
	--ctx)
		CTX="$2"
		shift 2
		;;
	--ubatch)
		UBATCH="$2"
		shift 2
		;;
	--evidence)
		EVIDENCE="$2"
		shift 2
		;;
	--widths)
		WIDTHS="$2"
		shift 2
		;;
	--twocol)
		TWOCOL="$2"
		shift 2
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done
[[ "${TWOCOL:-off}" == off || "$TWOCOL" == auto ]] || {
	echo "--twocol must be off|auto" >&2
	exit 2
}

PFN=$("${SCRIPT_DIR}/deploy.sh" pfn | tail -1)
if [[ -n "${XLLAMA_EXPECTED_PFN:-}" && "$PFN" != "$XLLAMA_EXPECTED_PFN" ]]; then
	echo "Error: installed package $PFN differs from XLLAMA_EXPECTED_PFN" >&2
	exit 1
fi
TMP=$(mktemp -d)
cleanup() {
	rm -rf "$TMP"
	"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "tttarget.flag" >/dev/null 2>&1 || true
	"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "tttarget_reps.txt" >/dev/null 2>&1 || true
	"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "tttarget_msix_sha.txt" >/dev/null 2>&1 || true
	"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "tttarget_widths.txt" >/dev/null 2>&1 || true
	"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "d3d12twocol.txt" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf '%d' "$REPS" >"$TMP/tttarget_reps.txt"
if [[ -n "$WIDTHS" ]]; then
	printf '%s' "$WIDTHS" >"$TMP/tttarget_widths.txt"
fi
printf '%s' "${XLLAMA_MSIX_SHA256:-unknown}" >"$TMP/tttarget_msix_sha.txt"
printf '%d' "$THREADS" >"$TMP/bench_threads.txt"
printf '%d' "$CTX" >"$TMP/bench_ctx.txt"
printf '%d' "$UBATCH" >"$TMP/bench_ubatch.txt"
: >"$TMP/tttarget.flag"

echo "  Uploading ttarget artifacts..."
"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "tttarget-result.csv.done" >/dev/null 2>&1 || true
"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "tttarget-result.csv" >/dev/null 2>&1 || true
"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "d3d12twocol.txt" >/dev/null 2>&1 || true
if [[ "$TWOCOL" == auto ]]; then
	printf 'auto' >"$TMP/d3d12twocol.txt"
	"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/d3d12twocol.txt" "$PFN" "" "d3d12twocol.txt"
	echo "  twocol=auto: real decode in this run uses the allowlisted NEW tile"
else
	echo "  twocol=off: clean OLD arm (stale knob deleted)"
fi
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/tttarget_reps.txt" "$PFN" "" "tttarget_reps.txt"
if [[ -n "$WIDTHS" ]]; then
	"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/tttarget_widths.txt" "$PFN" "" "tttarget_widths.txt"
fi
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/tttarget_msix_sha.txt" "$PFN" "" "tttarget_msix_sha.txt"
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/bench_threads.txt" "$PFN" "" "bench_threads.txt"
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/bench_ctx.txt" "$PFN" "" "bench_ctx.txt"
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/bench_ubatch.txt" "$PFN" "" "bench_ubatch.txt"
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/tttarget.flag" "$PFN" "" "tttarget.flag"

"${SCRIPT_DIR}/deploy.sh" stop-app || true
sleep 1
"${SCRIPT_DIR}/deploy.sh" start-app

# Stage-aware wait. Each cycle answers three questions through the existing WDP
# API: has the result landed, what stage has the log reached, and is the process
# still there. A row error is REPORTED but does NOT end the wait: the app writes
# the CSV with that row marked and only then the .done. A process that is gone
# with no .done does end it, with a diagnosis, instead of running the clock down
# on a file that cannot come. Nothing is killed.
WAIT_TIMEOUT="${XLLAMA_TTTARGET_TIMEOUT:-900}"
SEEN="$TMP/stages-seen"
: >"$SEEN"
run_start=$SECONDS
saw_stage=""
dead_reason=""
echo "  Waiting for ttarget-result.csv.done (timeout ${WAIT_TIMEOUT}s) ..."
while ((SECONDS - run_start < WAIT_TIMEOUT)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "tttarget-result.csv.done" "$TMP/done" 2>/dev/null; then
		break
	fi
	if "${SCRIPT_DIR}/deploy.sh" get-log "$PFN" >"$TMP/log.txt" 2>/dev/null; then
		# xllama.log is cumulative across runs, so everything below must only see
		# THIS run. run_tttarget emits 'stage=load start' before any work, so the
		# slice after its last occurrence is this run; before it lands the slice is
		# empty and a previous run's rows cannot leak in as false alarms.
		awk '/ttarget stage=load start/{n=NR} {a[NR]=$0} END{for(i=n;i<=NR;i++)print a[i]}' \
			"$TMP/log.txt" >"$TMP/cur.txt" || true
		# Stage progress: each new stage line is printed exactly once.
		grep -a 'ttarget stage=' "$TMP/cur.txt" | tr -d '\r' | sort -u >"$TMP/stages-now" || true
		while IFS= read -r line; do
			[[ -z "$line" ]] && continue
			grep -Fxq "$line" "$SEEN" 2>/dev/null && continue
			printf '%s\n' "$line" >>"$SEEN"
			echo "  $line"
			saw_stage=1
		done <"$TMP/stages-now"
		# Row errors are visible before .done exists; surfaced, not fatal.
		if grep -a 'ttarget: .*err=[^ -]' "$TMP/cur.txt" 2>/dev/null | head -1 | grep -q .; then
			if [[ -z "${row_err_seen:-}" ]]; then
				row_err_seen=1
				echo "  (row error reported above; waiting for the CSV to land)"
			fi
		fi
		# A crash never writes .done: stop with the reason. Per-row errors are NOT
		# here on purpose — they are recorded in the CSV and the run continues.
		if grep -aqiE 'unhandled|access violation|std::terminate' "$TMP/cur.txt" 2>/dev/null; then
			dead_reason="the log reports a fatal error"
			break
		fi
	fi
	# Liveness. The app exits on its own after writing, and it can also crash; the
	# two are only distinguishable by whether .done is there — which was checked
	# first this cycle. Missing .done + gone process = finished without result.
	# The grace period keeps the post-start race (start-app returns before the
	# process is registered) from reading as a crash.
	status_line="$("${SCRIPT_DIR}/deploy.sh" status 2>/dev/null | head -1 || true)"
	app_running=""
	if grep -qi 'running=true' <<<"$status_line"; then
		app_running=1
	fi
	if [[ -z "$app_running" ]]; then
		if [[ -n "$saw_stage" ]] || ((SECONDS - run_start > 60)); then
			dead_reason="xllama is no longer running (${status_line})"
			break
		fi
	fi
	sleep 5
done
# The app exits on its own once it has written the result, so a cycle can observe
# "process gone" while the .done is already there. Retry once before failing.
if [[ ! -f "$TMP/done" && -n "$dead_reason" ]]; then
	"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "tttarget-result.csv.done" "$TMP/done" 2>/dev/null || true
fi
if [[ ! -f "$TMP/done" ]]; then
	if [[ -n "$dead_reason" ]]; then
		echo "stopped: ${dead_reason}" >&2
	else
		echo "timeout waiting for ttarget-result.csv.done after ${WAIT_TIMEOUT}s" >&2
	fi
	echo "--- stages seen ---" >&2
	cat "$SEEN" >&2 || true
	echo "--- process ---" >&2
	"${SCRIPT_DIR}/deploy.sh" status >&2 || true
	echo "--- log tail ---" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log "$PFN" 2>&1 | tail -40 || true
	exit 1
fi
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "tttarget-result.csv" "$TMP/r.csv"
mkdir -p "$(dirname "$OUT")"
tr -d '\r' <"$TMP/r.csv" >"$OUT"
echo "Wrote $OUT"
cat "$OUT"
if [[ -n "$EVIDENCE" ]]; then
	mkdir -p "$EVIDENCE"
	cp "$OUT" "$EVIDENCE/tttarget-result.csv"
	"${SCRIPT_DIR}/deploy.sh" get-log >"$EVIDENCE/xllama.log" 2>&1 || true
	{
		echo "package_pfn=$PFN"
		echo "msix_sha256=${XLLAMA_MSIX_SHA256:-unknown}"
		echo "reps=$REPS threads=$THREADS ctx=$CTX ubatch=$UBATCH"
		echo "recorded_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	} >"$EVIDENCE/config.txt"
	echo "Evidence in $EVIDENCE"
fi

python3 - "$OUT" <<'PY'
import csv, statistics, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))]
bad = [r for r in rows if r["error"] != "-"]
meas = [r for r in rows if r["rep"] != "-1" and r["error"] == "-"]
print(f"--- T_target(B): {len(rows)} rows, {len(meas)} measured, {len(bad)} error(s) ---")
for r in bad[:8]:
    print(f"  B={r['b']} rep={r['rep']} error={r['error']}")
present = sorted({r["b"] for r in meas}, key=int)
for b in present:
    rs = [r for r in meas if r["b"] == b and r["argmax_match"] == "1"]
    if not rs:
        print(f"B={b}: no valid row")
        continue
    w = sorted(float(r["wall_ms"]) for r in rs)
    g = sorted(float(r["gpu_ms"]) for r in rs)
    mm = sorted(int(r["matmuls"]) for r in rs)
    print(f"B={b}: n={len(rs)} wall med={statistics.median(w):.3f} min={w[0]:.3f} "
          f"max={w[-1]:.3f} | gpu med={statistics.median(g):.3f} | matmuls med={statistics.median(mm)}")
mismatch = [r for r in rows if r["argmax_match"] == "0"]
if mismatch:
    print(f"FAIL: {len(mismatch)} row(s) whose logits/argmax did not match the reference")
    sys.exit(1)
if not meas:
    print("FAIL: no measured rows (warm-up only or every row errored)")
    sys.exit(1)
print("OK: every row matched the sequential reference on logits/argmax")
PY
