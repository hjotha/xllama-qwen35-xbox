#!/usr/bin/env bash
# rev209 termgate with CORRECTED labels reflecting the real invocations.
# Legacy script (seqgates-rev159-termgate.sh) ran 3 arms per knob: seq, "ref"
# and "cand" — but "ref" and "cand" were the SAME invocation (bench_mtp=4) and
# the device itself labels both outputs arm=cand (legacy ref file was a
# duplicate, byte-identical to legacy cand, sha 12c2fc8ff84cc9dd...).
# Real matrix: bench_mtp=0 -> device arm=seq; bench_mtp=4 -> device arm=cand.
# Gate: within-arm sw0==sw1 byte-identical (FFN off vs on termination
# parity), each vs the rev192 legacy baseline of the SAME real arm, and the
# device arm column must match the label.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/enabled-rev209}"
LEGACY="$REPO/bench/results/qwen4b-maxperf/restore-rev192"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.209_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/termgate"
echo "termgate_rev209 start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/termgate/receipt.txt"

run_arm() { # sw label mtp
  local sw="$1" label="$2" mtp="$3"
  printf '%s' "$sw" > /tmp/opencode/qwen4b-recon/tg-sw.txt
  ./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/tg-sw.txt "$PFN" "" d3d12swiglu.txt >/dev/null
  printf '%s' "$mtp" > /tmp/opencode/qwen4b-recon/tg-mtp.txt
  ./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/tg-mtp.txt "$PFN" "" bench_mtp.txt >/dev/null
  printf 'cancel,stop,eog' > /tmp/opencode/qwen4b-recon/tg-term.txt
  ./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/tg-term.txt "$PFN" "" termgate.txt >/dev/null
  ./scripts/deploy.sh delete-file "$PFN" termgate-result.csv >/dev/null 2>&1 || true
  ./scripts/deploy.sh delete-file "$PFN" termgate-result.csv.done >/dev/null 2>&1 || true
  printf '' > /tmp/opencode/qwen4b-recon/tg.flag
  ./scripts/deploy.sh upload-file /tmp/opencode/qwen4b-recon/tg.flag "$PFN" "" termgate.flag >/dev/null
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
  local deadline ok=0 out="$BASE/termgate/term-sw$sw-$label.csv"
  deadline=$(( $(date +%s) + 300 ))
  while [[ "$(date +%s)" -lt "$deadline" ]]; do
    if ./scripts/deploy.sh fetch-file "$PFN" "termgate-result.csv.done" "/tmp/opencode/qwen4b-recon/tg.done" 2>&1 | grep -q "^Fetched"; then ok=1; break; fi
    sleep 5
  done
  [[ "$ok" == "1" ]] || { echo "TIMEOUT sw=$sw label=$label"; echo "TIMEOUT sw=$sw label=$label" >> "$BASE/termgate/receipt.txt"; return 1; }
  ./scripts/deploy.sh fetch-file "$PFN" termgate-result.csv "$out" >/dev/null 2>&1 || true
  echo "arm=sw$sw-$label invocation=bench_mtp=$mtp knob=$sw device_arm=$(awk -F, 'NR==2{print $2}' "$out") done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/termgate/receipt.txt"
}

for sw in 0 1; do
  run_arm "$sw" seq  0
  run_arm "$sw" cand 4
done

# Gates: real-label match, within-arm FFN parity, legacy-baseline parity.
python3 - "$BASE/termgate" "$LEGACY" <<'PY'
import hashlib, sys, pathlib
base, legacy = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
fails = []
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for sw in (0, 1):
    for label, want_arm in (("seq", "seq"), ("cand", "cand")):
        f = base / f"term-sw{sw}-{label}.csv"
        if not f.exists():
            fails.append(f"MISSING {f.name}"); continue
        arm = [l.split(",")[1] for l in f.read_text().splitlines()[1:] if l]
        if not arm or any(a != want_arm for a in arm):
            fails.append(f"DEVICE_ARM_MISMATCH {f.name} arms={sorted(set(arm))}")
# within-arm FFN off/on byte parity
for label in ("seq", "cand"):
    a = base / f"term-sw0-{label}.csv"; b = base / f"term-sw1-{label}.csv"
    if a.exists() and b.exists() and sha(a) != sha(b):
        fails.append(f"SW_PARITY_FAIL {label}")
# legacy baseline of the SAME real arm
pairs = [("seq", legacy / "term-sw0-seq.csv"), ("cand", legacy / "term-sw0-cand.csv"),
         ("cand", legacy / "term-sw0-ref.csv")]  # legacy ref == legacy cand (dup invocation)
for label, ref in pairs:
    mine = base / "term-sw0-cand.csv" if label == "cand" else base / "term-sw0-seq.csv"
    if mine.exists() and ref.exists() and sha(mine) != sha(ref):
        fails.append(f"LEGACY_BASE_FAIL {label} vs {ref.name}")
print(f"== termgate rev209 corrected labels: failures={len(fails)}")
for f in fails: print("  ", f)
print("   note: legacy ref/cand were one identical bench_mtp=4 invocation;")
print("   corrected matrix = {seq: mtp0, cand: mtp4} x {sw0, sw1}.")
sys.exit(1 if fails else 0)
PY
echo "termgate_rev209 end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/termgate/receipt.txt"
echo TERMGATE_REV209_DONE
