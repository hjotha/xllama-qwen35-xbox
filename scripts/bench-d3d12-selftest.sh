#!/usr/bin/env bash
# bench-d3d12-selftest.sh — d3d12 ggml backend selftest on the console (D2a,
# docs/gguf-gpu-decode.md). Uploads d3d12be.flag, waits for d3d12be-result.csv,
# and checks the D2a gate: every Q4_0 / Q4_K / Q6_K case within tolerance of
# ggml's dequantizers, and the decode (ncols=1) cases at >= 100 GB/s packed.
#
# Prerequisites: CI MSVC package (unified or llamacpp) installed on Series S.
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-d3d12-selftest.sh [--out bench/results/d2a-d3d12-selftest.csv] [--force]
#       [--timeout 600] [--shapecost-out <file>]
# The shape-cost diagnostic (real tttarget triples at B=1/2/3/5, d3d12sc-result.csv)
# is fetched to --shapecost-out (default: <out>-shapecost.csv) after the D2a gate
# below passes its unchanged verdict. Reported, never gated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/d2a-d3d12-selftest.csv"
TIMEOUT_S=1800
SHAPECOST_OUT=""
FORCE=0
while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		OUT="$2"
		shift 2
		;;
	--shapecost-out)
		SHAPECOST_OUT="$2"
		shift 2
		;;
	--timeout)
		TIMEOUT_S="$2"
		shift 2
		;;
	--force)
		FORCE=1
		shift
		;;
	-h | --help)
		sed -n '2,11p' "$0"
		exit 0
		;;
	*)
		echo "unknown: $1" >&2
		exit 2
		;;
	esac
done
[[ "$TIMEOUT_S" =~ ^[0-9]+$ && "$TIMEOUT_S" -gt 0 ]] || {
	echo "--timeout needs a positive number of seconds" >&2
	exit 2
}
if [[ -z "$SHAPECOST_OUT" ]]; then
	SHAPECOST_OUT="${OUT%.csv}-shapecost.csv"
fi
if [[ -e "$OUT" && "$FORCE" != 1 ]]; then
	echo "refusing to overwrite recorded $OUT. Pass --force to override." >&2
	exit 2
fi
if [[ -e "$SHAPECOST_OUT" && "$FORCE" != 1 ]]; then
	echo "refusing to overwrite recorded $SHAPECOST_OUT. Pass --force to override." >&2
	exit 2
fi

: "${XBOX_IP:?source ~/.config/xllama/xbox-env}"
PFN=$("${SCRIPT_DIR}/deploy.sh" pfn 2>/dev/null || true)
[[ -n "$PFN" ]] || {
	echo "xllama not installed on console" >&2
	exit 1
}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "d3d12be-result.csv.done" >/dev/null 2>&1 || true
"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "d3d12sc-result.csv.done" >/dev/null 2>&1 || true
: >"$TMP/d3d12be.flag"
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/d3d12be.flag" "$PFN" "" "d3d12be.flag"
"${SCRIPT_DIR}/deploy.sh" stop-app || true
sleep 1
"${SCRIPT_DIR}/deploy.sh" start-app
echo "Waiting for d3d12be-result.csv.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12be-result.csv.done" "$TMP/done" 2>/dev/null; then
		break
	fi
	sleep 3
done
[[ -f "$TMP/done" ]] || {
	echo "timeout waiting for d3d12be-result.csv.done" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12be-result.csv" "$TMP/r.csv"
mkdir -p "$(dirname "$OUT")"
tr -d '\r' <"$TMP/r.csv" >"$OUT"
echo "Wrote $OUT"
cat "$OUT"

python3 - "$OUT" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r for r in rows if r["ok"] != "1" or r["d3d12_ran"] != "1"]
slow = [r for r in rows if r["ncols"] == "1" and float(r["packed_gbs"]) < 100.0]
print(f"--- D2a gate: {len(rows) - len(bad)}/{len(rows)} cases correct; "
      f"{len(slow)} decode case(s) under 100 GB/s ---")
for r in bad:
    print(f"FAIL {r['type']} n={r['n']} k={r['k']} ncols={r['ncols']} rel_err={r['rel_err']} {r['error']}")
for r in slow:
    print(f"SLOW {r['type']} n={r['n']} k={r['k']} gbs={r['packed_gbs']}")
print("D2a=" + ("PASS" if rows and not bad and not slow else "FAIL"))

# Plan 003 stage 2: isolated matmul cost on the verify batch sizes (NOT T_target(B):
# these are isolated matmuls with a fixed seed independent of B — no KV attention,
# no recurrent state, no decode scheduling. Only the live-context matrix is
# T_target(B)). Same type/n/k across the group and one timing per shape, so the
# B's are comparable and the profile can say whether a weight-reuse kernel is
# worth building. Reported, not gated: the D2a gate above is the pass/fail contract.
print("")
print("--- isolated matmul cost on the Qwen shapes (plan 003 stage 2) ---")
print(f"{'type':>6} {'n':>6} {'k':>6} {'B':>3} {'gpu_ms':>10} {'GB/s':>9} {'rel_err':>10}")
groups = {}
for r in rows:
    if r["d3d12_ran"] != "1" or not r["gpu_ms"]:
        continue
    b = int(r["ncols"])
    if b in (1, 2, 3, 5):
        groups.setdefault((r["type"], r["n"], r["k"]), []).append(r)
for (t, n, k), rs in groups.items():
    if len(rs) < 2:
        continue
    base = None
    for r in sorted(rs, key=lambda x: int(x["ncols"])):
        if base is None:
            base = float(r["gpu_ms"])
        print(f"{t:>6} {n:>6} {k:>6} {r['ncols']:>3} {float(r['gpu_ms']):>10.4f} "
              f"{float(r['packed_gbs']):>9.2f} {r['rel_err']:>10}")
    print(f"       per-column cost vs B=1: " + ", ".join(
        f"B={r['ncols']}:{float(r['gpu_ms']) / float(base):.2f}x"
        for r in sorted(rs, key=lambda x: int(x["ncols"]))))
if not rows or bad or slow:
    sys.exit(1)
PY

# Plan 004: the shape-cost bench runs after the gate rows are done, in the same
# app launch. It gets its own wait window: up to TIMEOUT_S more seconds.
echo "Waiting for d3d12sc-result.csv.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12sc-result.csv.done" "$TMP/sc-done" 2>/dev/null; then
		break
	fi
	sleep 3
done
[[ -f "$TMP/sc-done" ]] || {
	echo "timeout waiting for d3d12sc-result.csv.done" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12sc-result.csv" "$TMP/sc.csv"
mkdir -p "$(dirname "$SHAPECOST_OUT")"
tr -d '\r' <"$TMP/sc.csv" >"$SHAPECOST_OUT"
echo "Wrote $SHAPECOST_OUT"

python3 - "$SHAPECOST_OUT" <<'PY'
import csv, sys
from statistics import median
rows = list(csv.DictReader(open(sys.argv[1])))
fail = [r for r in rows if r["ok"] != "1" or r["d3d12_ran"] != "1"]
print("")
print(f"--- shape cost: real tttarget triples at B=1/2/3/5 "
      f"({len(rows) - len(fail)}/{len(rows)} rows correct; REPORTED, not gated) ---")
print(f"{'type':>6} {'n':>6} {'k':>6} {'B':>3} {'gpu_ms':>10} {'grange':>8} {'wall_ms':>10} "
      f"{'wrange':>8} {'GB/s':>9} {'wMB':>8} {'peakMB':>8} {'var':>3} {'pair':>4} {'tc':>2} "
      f"{'pads':>8} {'rel_err':>10}")
for r in rows:
    if r["pair"] != "-1":
        continue
    print(f"{r['type']:>6} {r['n']:>6} {r['k']:>6} {r['ncols']:>3} "
          f"{float(r['gpu_ms'] or 0):>10.4f} {float(r['gpu_ms_range'] or 0):>8.4f} "
          f"{float(r['wall_ms'] or 0):>10.4f} {float(r['wall_ms_range'] or 0):>8.4f} "
          f"{float(r['packed_gbs'] or 0):>9.2f} {float(r['weight_mb'] or 0):>8.3f} "
          f"{float(r['peak_ws_mb'] or 0):>8.1f} {r['variant']:>3} {r['pair']:>4} {r['twocol']:>2} "
          f"{r['pads']:>8} {r['rel_err']:>10}")
for r in fail:
    print(f"COSTFAIL var={r['variant']} pair={r['pair']} tc={r['twocol']} pads={r['pads']} "
          f"{r['type']} n={r['n']} k={r['k']} ncols={r['ncols']} rel_err={r['rel_err']} {r['error']}")

# Paired OLD/NEW blocks (plan 004 q4_k experiment): same tensors, alternating
# order, individual pairs kept. Median pair delta (NEW-OLD) with its range is
# the decision number; a speedup is claimed only from these same-run pairs.
# tc is the RECORDED two-column dispatch count, not the request: a NEW pair
# row with tc=0 means the new kernel never ran and must not count as coverage.
pairs = {}
for r in rows:
    if r["pair"] == "-1" or r["d3d12_ran"] != "1":
        continue
    pairs.setdefault((r["type"], r["n"], r["k"], r["ncols"], r["pads"], r["pair"]),
                     {})[r["variant"]] = r
groups = {}
tcmismatch = []
for (t, n, k, b, pads, p), vs in pairs.items():
    if "0" in vs and "1" in vs:
        if vs["0"]["twocol"] != "0" or vs["1"]["twocol"] == "0":
            tcmismatch.append((t, n, k, b, pads, p, vs["0"]["twocol"], vs["1"]["twocol"]))
            continue
        groups.setdefault((t, n, k, b, pads), []).append(
            (float(vs["0"]["gpu_ms"]), float(vs["1"]["gpu_ms"])))
if tcmismatch:
    print("TCMISMATCH (request != actual dispatch — excluded from pairs):")
    for t, n, k, b, pads, p, tc0, tc1 in tcmismatch:
        print(f"  {t} n={n} k={k} B={b} pads={pads} pair={p} tc_old={tc0} tc_new={tc1}")
print("")
print("--- OLD/NEW pairs (same tensors, alternating order; +delta means NEW slower) ---")
print(f"{'type':>6} {'n':>6} {'k':>6} {'B':>3} {'pairs':>5} {'old_med':>9} {'new_med':>9} "
      f"{'d_med':>9} {'d_range':>9} {'x':>6} {'pads':>8}")
for (t, n, k, b, pads), ps in groups.items():
    olds = [o for o, _ in ps]
    news = [w for _, w in ps]
    ds = [w - o for o, w in ps]
    om, nm = median(olds), median(news)
    dm = median(ds)
    dr = max(ds) - min(ds) if len(ds) > 1 else 0.0
    print(f"{t:>6} {n:>6} {k:>6} {b:>3} {len(ps):>5} {om:>9.4f} {nm:>9.4f} "
          f"{dm:>+9.4f} {dr:>9.4f} {om / nm if nm else 0:>6.3f} {pads:>8}")

# Required experiment coverage: a truncated/older CSV must not promote a tile
# or recurrent kernel that never ran. Timing wins remain reported separately.
keys = {(r['type'], r['n'], r['k'], r['ncols'], r['pads'], r['variant'], r['pair']) for r in rows}
required = set()
for cols in (2, 4):
    for b, pads in ((2, '0/0/0'), (3, '0/0/0'), (5, '0/0/0'), (5, '13/7/64')):
        for variant in (0, 1):
            for pair in range(4):
                required.add((f'q6_k_c{cols}', '248320', '2560', str(b), pads, str(variant), str(pair)))
for slots in (1, 5):
    for tokens in (1, 2, 3, 4, 5, 64):
        required.add((f'gdn_k{slots}', '128', '32', str(tokens), '0/0/0', '0', '-1'))
for tokens in (3, 4):
    required.add(('gdn_k5', '128', '32', str(tokens), '3/5/7', '0', '-1'))
missing = sorted(required - keys)
if missing:
    print(f'COSTFAIL: {len(missing)} required Q6/GDN cases missing')
if not rows or fail or tcmismatch or missing:
    sys.exit(1)
PY
