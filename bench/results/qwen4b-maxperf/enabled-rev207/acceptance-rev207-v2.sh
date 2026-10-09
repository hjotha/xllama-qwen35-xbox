#!/usr/bin/env bash
# rev207 acceptance completion v2: 2 interleaved blocks, 12 cells x 2 arms,
# --runs 3 => run1 warmup dropped, run2+run3 = 2 real measured lines per arm
# per cell per block. Idempotent: skips a cell-arm already having 2 rows.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="${BASE:-$REPO/bench/results/qwen4b-maxperf/enabled-rev207}"
BLOCK="${1:?block id (b1|b2)}"
PFN_EXPECT="${PFN_EXPECT:-GianlucaMazza.xllama_1.6.0.207_x64__pj67f1fcj4n14}"
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
D="$BASE/blocks/$BLOCK"; mkdir -p "$D/csv" "$D/knobs"
set_sw() { printf '%s' "$1" > "$D/knobs/d3d12swiglu.txt"; ./scripts/deploy.sh upload-file "$D/knobs/d3d12swiglu.txt" "$PFN" "" d3d12swiglu.txt >/dev/null; }
measured_rows() { local f="$1"; [[ -f "$f" ]] && tail -n +2 "$f" | wc -l || echo 0; }
run_cell() { # name prompt mtp predict
  local name="$1" prompt="$2" mtp="$3" predict="$4" sw out n
  for sw in 0 1; do
    out="$D/csv/$name-sw$sw.csv"
    n=$(measured_rows "$out")
    if [[ "$n" -ge 2 ]]; then echo "skip $name-sw$sw (already $n rows)"; continue; fi
    set_sw "$sw"
    "$REPO/scripts/bench-xbox-ort.sh" qwen35-4b-mtp \
      --threads 2 --ctx 2048 --gpu-layers 34 --batch 256 --ubatch 64 \
      --twocol auto --greedy --seed 1 --tokens --ignore-eog \
      --profile-phases 0 --mtp "$mtp" --mtp-pmin 50 --n-predict "$predict" \
      --prompt "$REPO/bench/prompts/$prompt" \
      --out "$out" --runs 3 >> "$D/driver.log" 2>&1
    echo "cell=$name sw=$sw rows=$(measured_rows "$out") utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
}
if [[ ! -f "$D/done" ]]; then
  echo "block=$BLOCK start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_cell chat64     spec-chat-open.txt 0 64
  run_cell code64     spec-code-edit.txt 0 64
  run_cell std64      standard-512.txt   0 64
  run_cell chat256    spec-chat-open.txt 0 256
  run_cell code256    spec-code-edit.txt 0 256
  run_cell std256     standard-512.txt   0 256
  run_cell chat64-mtp spec-chat-open.txt 2 64
  run_cell code64-mtp spec-code-edit.txt 2 64
  run_cell std64-mtp  standard-512.txt   2 64
  run_cell chat256-mtp spec-chat-open.txt 2 256
  run_cell code256-mtp spec-code-edit.txt 2 256
  run_cell std256-mtp  standard-512.txt   2 256
  touch "$D/done"
  echo "block=$BLOCK end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
python3 - "$D" <<'PY'
import csv, glob, hashlib, os, sys, statistics
D=sys.argv[1]
DIG={"chat64":"84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2",
"code64":"ff42c5beec71ff491b458d0e48cbac4d98f8cdcf8153c78868c64aee687a4010",
"chat256":"f78ce8372ccc7281aaad5e726bf7c88a0317212e3b2ed397fb0fd83ffd53066c",
"code256":"c7d84255e08e20776c1ec6743d23161d32d08dcba642036b305250f10841ec0c",
"std512":"44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732"}
fails=0; t0={}; t1={}
for f in sorted(glob.glob(D+"/csv/*.csv")):
    name=os.path.basename(f)[:-4]
    stem=name.split("-sw")[0]; sw=int(name.split("-sw")[1][0]); stem0=stem.replace("-mtp","")
    rows=list(csv.DictReader(open(f)))
    idx=[r["run_index"] for r in rows]
    okrows=(len(rows)==2 and set(idx)=={"2","3"})
    if not okrows: fails+=1; print(f"ROW_FAIL {name} rows={len(rows)} idx={idx}")
    toks=[]
    for r in (2,3):
        p=f.replace(".csv",f".run{r}.tokens")
        if os.path.exists(p):
            h=hashlib.sha256(open(p,'rb').read()).hexdigest()
            toks.append(h)
            if h!=DIG[stem0]: fails+=1; print(f"TOK_FAIL {name} run{r}")
        else:
            fails+=1; print(f"TOK_MISSING {name} run{r}")
    dec=[float(r["decode_tok_s"]) for r in rows]
    (t0 if sw==0 else t1).setdefault(stem,[]).extend(dec)
print(f"blocks/{os.path.basename(D)}: failures={fails}")
print(f"{'cell':<12} {'sw0_mean':>8} {'sw1_mean':>8} {'delta%':>7}")
for k in sorted(set(t0)|set(t1)):
    a=t0.get(k,[]); b=t1.get(k,[])
    if a and b:
        print(f"{k:<12} {statistics.mean(a):8.2f} {statistics.mean(b):8.2f} {100*(statistics.mean(b)/statistics.mean(a)-1):+6.2f}%")
PY
