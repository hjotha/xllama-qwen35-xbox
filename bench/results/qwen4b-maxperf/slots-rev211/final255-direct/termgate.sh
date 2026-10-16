#!/usr/bin/env bash
# Termgate with labels reflecting the real invocations (plan 008/009 correction).
# Legacy script (seqgates-rev159-termgate.sh) ran 3 arms per knob: seq, "ref"
# and "cand" — but "ref" and "cand" were the SAME invocation (bench_mtp=4),
# byte-identical outputs (sha 12c2fc8ff84cc9dd... in rev192 baselines too).
# The device-side `arm` column does NOT denote the invocation: it encodes the
# twocol/repack scope (uwp/inference-bridge.cpp:1992 — "cand" only when
# twocol>=2 and repack_scope==2, else "ref"), i.e. ambient knob state. It is
# the ONLY column excluded from comparison; every other field (including
# active/drafted/rounds/lookup/ews/n_eval) is gated.
# Real matrix (this script): bench_mtp=0 -> term-sw<sw>-mtp0.csv;
# bench_mtp=4 -> term-sw<sw>-mtp4.csv. Gates: within-arm sw0==sw1
# byte-identical mod arm (FFN off vs on termination parity), each vs the
# rev192 legacy baseline of the SAME invocation (mtp0->seq; mtp4->cand and
# ref, which are duplicates of one invocation). TERM_GATE_ONLY=1 skips
# probing and gates the CSVs already collected in $BASE/termgate.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/slots-rev211/final255-direct}"
LEGACY="$REPO/bench/results/qwen4b-maxperf/restore-rev192"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/termgate"
echo "termgate start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) PFN=$PFN" > "$BASE/termgate/receipt.txt"

run_arm() { # sw label mtp
  local sw="$1" label="$2" mtp="$3"
  printf '%s' "$sw" > /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg-sw.txt
  ./scripts/deploy.sh upload-file /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg-sw.txt "$PFN" "" d3d12swiglu.txt >/dev/null
  printf '%s' "$mtp" > /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg-mtp.txt
  ./scripts/deploy.sh upload-file /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg-mtp.txt "$PFN" "" bench_mtp.txt >/dev/null
  printf 'cancel,stop,eog' > /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg-term.txt
  ./scripts/deploy.sh upload-file /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg-term.txt "$PFN" "" termgate.txt >/dev/null
  ./scripts/deploy.sh delete-file "$PFN" termgate-result.csv >/dev/null 2>&1 || true
  ./scripts/deploy.sh delete-file "$PFN" termgate-result.csv.done >/dev/null 2>&1 || true
  printf '' > /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg.flag
  ./scripts/deploy.sh upload-file /home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg.flag "$PFN" "" termgate.flag >/dev/null
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
  local deadline ok=0 out="$BASE/termgate/term-sw$sw-$label.csv"
  deadline=$(( $(date +%s) + 300 ))
  while [[ "$(date +%s)" -lt "$deadline" ]]; do
    if ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv.done" "/home/hjotha/worktrees/xllama-qwen35-4b-maxperf/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/scratch/tg.done" 2>&1 | grep -q "^Fetched"; then ok=1; break; fi
    sleep 5
  done
  [[ "$ok" == "1" ]] || { echo "TIMEOUT sw=$sw label=$label"; echo "TIMEOUT sw=$sw label=$label" >> "$BASE/termgate/receipt.txt"; return 1; }
  ./scripts/deploy.sh fetch-file "$PFN" termgate-result.csv "$out" >/dev/null 2>&1 || true
  echo "arm=sw$sw-$label invocation=bench_mtp=$mtp knob=$sw device_arm_column=$(awk -F, 'NR==2{print $2}' "$out") done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/termgate/receipt.txt"
}

if [[ "${TERM_GATE_ONLY:-0}" == "1" ]]; then
  echo "termgate mode=gates-only (CSVs already collected)" >> "$BASE/termgate/receipt.txt"
else
  for sw in 0 1; do
    run_arm "$sw" mtp0 0
    run_arm "$sw" mtp4 4
  done
fi

# Gates: within-arm FFN parity + legacy-baseline parity per invocation.
# Every CSV field is compared; only column 2 (`arm`) is excluded.
python3 - "$BASE/termgate" "$LEGACY" <<'PY'
import hashlib, sys, pathlib
base, legacy = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
fails = []
def sha_xarm(p):
    # sha256 of the CSV with column index 1 (`arm`) removed from every line.
    rows = []
    for l in p.read_text().splitlines():
        if not l:
            continue
        f = l.split(",")
        rows.append(",".join(x for i, x in enumerate(f) if i != 1))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()
def arm_col(p):
    lines = p.read_text().splitlines()
    if len(lines) < 2:
        return "?"
    f = lines[1].split(",")
    return f[1] if len(f) > 1 else "?"
def activity(p):
    out = []
    for l in p.read_text().splitlines()[1:]:
        if not l:
            continue
        f = l.split(",")
        out.append((f[0], f[1] if len(f) > 1 else "", f[8] if len(f) > 8 else "",
                    f[9] if len(f) > 9 else "", f[10] if len(f) > 10 else "",
                    f[11] if len(f) > 11 else ""))
    return out
for f in [base / f"term-sw{sw}-{arm}.csv" for sw in (0, 1) for arm in ("mtp0", "mtp4")]:
    if not f.exists():
        fails.append(f"MISSING {f.name}")
for label in ("mtp0", "mtp4"):
    a = base / f"term-sw0-{label}.csv"; b = base / f"term-sw1-{label}.csv"
    if a.exists() and b.exists() and sha_xarm(a) != sha_xarm(b):
        fails.append(f"SW_PARITY_FAIL {label}")
pairs = [(base / "term-sw0-mtp0.csv", legacy / "term-sw0-seq.csv"),
         (base / "term-sw0-mtp4.csv", legacy / "term-sw0-cand.csv"),
         (base / "term-sw0-mtp4.csv", legacy / "term-sw0-ref.csv"),
         (base / "term-sw1-mtp0.csv", legacy / "term-sw1-seq.csv"),
         (base / "term-sw1-mtp4.csv", legacy / "term-sw1-cand.csv"),
         (base / "term-sw1-mtp4.csv", legacy / "term-sw1-ref.csv")]
for mine, ref in pairs:
    if mine.exists() and ref.exists() and sha_xarm(mine) != sha_xarm(ref):
        fails.append(f"LEGACY_BASE_FAIL {mine.name} vs {ref.name} (all fields but arm)")
print(f"== termgate full-field parity (all fields except arm): failures={len(fails)}")
for f in fails: print("  ", f)
print("   arm column (excluded, reported): mine vs legacy per pair:")
for mine, ref in pairs:
    if mine.exists() and ref.exists():
        print(f"     {mine.name}: arm mine={arm_col(mine)} legacy={arm_col(ref)}")
print("   activity counters (gated; scene,arm,active,drafted,rounds,lookup,ews):")
for label in ("mtp0", "mtp4"):
    a = base / f"term-sw0-{label}.csv"
    if a.exists():
        print(f"   {a.name}: {activity(a)}")
print("   note: legacy ref/cand were one identical bench_mtp=4 invocation;")
print("   device arm column encodes twocol/repack scope, not the invocation.")
sys.exit(1 if fails else 0)
PY
echo "termgate end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/termgate/receipt.txt"
echo TERMGATE_DONE
