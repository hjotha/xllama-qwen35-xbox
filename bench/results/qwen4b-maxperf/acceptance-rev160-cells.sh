#!/usr/bin/env bash
# Plan 006 acceptance grid on the final LTCG build (rev160): parity vs the
# recorded baseline digests + timing for both arms, FFN knob OFF vs ON.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/acceptance-rev160}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.160_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
# shellcheck source=/dev/null
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
mkdir -p "$BASE/csv" "$BASE/knobs" "$BASE/session"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/receipt.txt"
echo "msix_sha256=74C9DEDDA15493A6673353F08E29A77EE16D21A124783BA023DC8956A03EF1C7" >> "$BASE/receipt.txt"
echo "pfn=$PFN" >> "$BASE/receipt.txt"

set_sw() { printf '%s' "$1" > "$BASE/knobs/d3d12swiglu.txt"; ./scripts/deploy.sh upload-file "$BASE/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null; }

run_cell() { # name prompt mtp sw predict
  local name="$1" prompt="$2" mtp="$3" sw="$4" predict="$5"
  set_sw "$sw"
  "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
    --threads 2 --ctx 2048 --gpu-layers 34 --batch 64 --ubatch 64 \
    --twocol auto --greedy --seed 1 --tokens --ignore-eog \
    --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
    --prompt "$REPO/bench/prompts/$prompt" \
    --out "$BASE/csv/$name.csv" --runs 2 >> "$BASE/driver.log" 2>&1
}

for spec in "chat64:spec-chat-open.txt:0:64" "code64:spec-code-edit.txt:0:64" \
            "chat256:spec-chat-open.txt:0:256" "code256:spec-code-edit.txt:0:256" \
            "std512:standard-512.txt:0:64" \
            "chat64-mtp:spec-chat-open.txt:2:64" "code64-mtp:spec-code-edit.txt:2:64" \
            "chat256-mtp:spec-chat-open.txt:2:256"; do
  IFS=: read -r name prompt mtp predict <<< "$spec"
  for sw in 0 1; do run_cell "$name-sw$sw" "$prompt" "$mtp" "$sw" "$predict"; done
  echo "cell=$name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BASE/receipt.txt"
done

declare -A DIGEST=(
  [chat64]=84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2
  [code64]=ff42c5beec71ff491b458d0e48cbac4d98f8cdcf8153c78868c64aee687a4010
  [chat256]=f78ce8372ccc7281aaad5e726bf7c88a0317212e3b2ed397fb0fd83ffd53066c
  [code256]=c7d84255e08e20776c1ec6743d23161d32d08dcba642036b305250f10841ec0c
  [std512]=44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732
)
PARITY_ALL=1
for base in chat64 code64 chat256 code256 std512 chat64-mtp code64-mtp chat256-mtp; do
  stem=${base%-mtp}
  for sw in 0 1; do
    for r in run1 run2; do
      got=$(sha256sum "$BASE/csv/$base-sw$sw.$r.tokens" | cut -d' ' -f1)
      if [[ "$got" != "${DIGEST[$stem]}" ]]; then PARITY_ALL=0; echo "PARITY_FAIL $base-sw$sw.$r got=$got" >> "$BASE/receipt.txt"; fi
    done
  done
done
[[ "$PARITY_ALL" == "1" ]] && echo "parity=OK" >> "$BASE/receipt.txt" || echo "parity=FAIL" >> "$BASE/receipt.txt"

python3 - "$BASE" <<'EOF'
import csv, statistics, sys
base = sys.argv[1]
def med(p):
    return statistics.median(float(x["decode_tok_s"]) for x in csv.DictReader(open(p)))
print("cell,med_off,med_on,delta_pct")
for name in ("chat64","code64","chat256","code256","std512","chat64-mtp","code64-mtp","chat256-mtp"):
    off = med(f"{base}/csv/{name}-sw0.csv"); on = med(f"{base}/csv/{name}-sw1.csv")
    print(f"{name},{off:.3f},{on:.3f},{(on/off-1)*100:+.2f}")
EOF
echo ACCEPTANCE_CELLS_DONE
