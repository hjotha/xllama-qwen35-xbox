#!/usr/bin/env bash
# T13b-varied-prefixes.sh — plan 009 #170c across INDEPENDENT trajectories
# and prefix lengths on rev243: short (42, seed only), medium (seed+1 turn),
# long (seed+2 turns, 187). Per case: fresh control trajectory vs restore
# with trailer; strict byte comparison at the tail turn; mirror completeness
# (pos_min) and separated restore/prefill/decode times.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T13b"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.243_x64__pj67f1fcj4n14"
API=http://192.168.1.26:11434
cd "$REPO"
API=http://192.168.1.26:11434
set -a
source ~/.config/xllama/xbox-env
set +a
mkdir -p "$D"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
# shellcheck disable=SC1091
source "$D/../T2-lib.sh"
: > "$D/verdicts.txt"
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"

fetch_one() { local ad="$1"; mkdir -p "$ad"
  for f in chat-dump.logits chat-dump.tokens chat-dump.mirror chat-dump.mtp; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done; }
one_case() { # tag nturns
  local tag="$1" nturns="$2"
  local cd="$D/$tag"
  mkdir -p "$cd"
  # trajectory build
  restart_verified "$tag-build"
  chat "$cd/b1.json" 16 "$P1"
  if [[ "$nturns" -ge 2 ]]; then
    python3 - "$cd" <<'PY'
import json, sys
d = sys.argv[1]
r1 = json.load(open(f"{d}/b1.json"))["choices"][0]["message"]["content"]
open(f"{d}/T2.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."}]))
PY
    chat "$cd/b2.json" 48 "$(cat "$cd/T2.json")"
  fi
  if [[ "$nturns" -ge 3 ]]; then
    python3 - "$cd" <<'PY'
import json, sys
d = sys.argv[1]
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
open(f"{d}/T3.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": rp("b1")},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": rp("b2")},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."}]))
PY
    chat "$cd/b3.json" 48 "$(cat "$cd/T3.json")"
  fi
  python3 - "$cd" "$nturns" <<'PY'
import json, sys
d, nturns = sys.argv[1], int(sys.argv[2])
def rp(n): return json.load(open(f"{d}/{n}.json"))["choices"][0]["message"]["content"]
msgs = [{"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]
if nturns >= 2:
    msgs += [{"role": "assistant", "content": rp("b1")},
             {"role": "user", "content": "Describe how a sorted merge works, step by step."}]
if nturns >= 3:
    msgs += [{"role": "assistant", "content": rp("b2")},
             {"role": "user", "content": "Now explain RoPE attention to a junior engineer."},
             {"role": "assistant", "content": rp("b3")}]
msgs += [{"role": "user", "content": "Repeat the cipher back exactly, then name the topics we discussed."}]
open(f"{d}/TAIL.json", "w").write(json.dumps(msgs))
PY
  TAIL="$(cat "$cd/TAIL.json")"
  slot_save "$cd/save.json" "t13b-$tag"
  local ntok
  ntok=$(python3 -c "import json;print(json.load(open('$cd/save.json'))['tokens'])")
  echo "[$tag] slot tokens=$ntok" | tee -a "$D/verdicts.txt"
  # R: restore with trailer + tail
  restart_verified "$tag-R"
  mkdir -p "$cd/R"
  chat "$cd/R-pre.json" 1 '[{"role":"user","content":"x"}]'
  TR0=$(date -u +%s%N); slot_restore "$cd/R-restore.json" "t13b-$tag" || fail "$tag R restore"; TR1=$(date -u +%s%N)
  local L0; L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  chat "$cd/R-tail.json" 24 "$TAIL" || fail "$tag R tail"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$cd/R-slice.log"
  fetch_one "$cd/R"
  { echo "[$tag] R: restore_api=$(( (TR1 - TR0) / 1000000 ))ms $(grep -oE 'turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+' "$cd/R-slice.log" | tail -1) $(grep -oE 'prefill=[0-9.]+ms \([0-9]+ tok\)' "$cd/R-slice.log" | tail -1) $(grep -oE 'drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+' "$cd/R-slice.log" | tail -1)"; } | tee -a "$D/verdicts.txt"
  # C: fresh trajectory + tail
  restart_verified "$tag-C"
  mkdir -p "$cd/C"
  chat "$cd/C1.json" 16 "$P1"
  if [[ "$nturns" -ge 2 ]]; then chat "$cd/C2.json" 48 "$(cat "$cd/T2.json")"; fi
  if [[ "$nturns" -ge 3 ]]; then chat "$cd/C3.json" 48 "$(cat "$cd/T3.json")"; fi
  local LC; LC=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  chat "$cd/C-tail.json" 24 "$TAIL" || fail "$tag C tail"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((LC + 1))" > "$cd/C-slice.log"
  fetch_one "$cd/C"
  { echo "[$tag] C: $(grep -oE 'turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+' "$cd/C-slice.log" | tail -1) $(grep -oE 'prefill=[0-9.]+ms \([0-9]+ tok\)' "$cd/C-slice.log" | tail -1) $(grep -oE 'drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+' "$cd/C-slice.log" | tail -1)"; } | tee -a "$D/verdicts.txt"
  curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d "{\"slot\":\"t13b-$tag\"}" >/dev/null || true
}
one_case short 1
one_case medium 2
one_case long 3
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
echo "== compare" | tee -a "$D/verdicts.txt"
python3 - "$D" <<'PY'
import hashlib, pathlib, struct, sys
d = pathlib.Path(sys.argv[1])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"
for tag in ("short", "medium", "long"):
    print(f"[{tag}]")
    for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.mirror", "chat-dump.mtp"):
        c, r = sha(d / tag / "C" / f), sha(d / tag / "R" / f)
        print(f"   {f}: C={c} R={r} EQUAL={c == r}")
    for arm, sub in (("C", "C"), ("R", "R")):
        p = d / tag / sub / "chat-dump.mirror"
        if p.exists():
            h = struct.unpack("<13I", p.read_bytes()[:52])
            print(f"   mirror {arm}: cells={h[3]} pos_min={h[4]}")
PY
set -a; source ~/.config/xllama/xbox-env; set +a
./scripts/deploy.sh get-log "$(./scripts/deploy.sh pfn 2>/dev/null)" 2>/dev/null | grep -c "MTP trailer restored" | xargs -I{} echo "trailer_restored_count_in_log={}"
echo "== T13b done"
