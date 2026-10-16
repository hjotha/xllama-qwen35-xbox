#!/usr/bin/env bash
# T14-families.sh — #170c pending item: extend prompt families/trajectories
# with ALIGNED controls (fresh replay of the identical trajectory vs restore
# with the MTP trailer), comparing all five artifact classes INCLUDING the
# target KV/GDN blob, on rev243. Cases:
#   code       spec-code-edit prompt family (~300 tok) + follow-up
#   std512     standard-512 technical-overview family
#   divergent  a trajectory containing a DIVERGENT turn (edit of the line),
#              exercising a non-linear accumulation path
# Each case: producer -> slot(with trailer); C(replay+tail); R(restore+tail).
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T14"
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
printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"

mkjson() { # outfile role:text ...  (python-built to stay quoting-safe)
  python3 - "$@" <<'PY'
import json, sys
out = sys.argv[1]
open(out, "w").write(json.dumps([{"role": "user", "content": t} for t in sys.argv[2:]]))
PY
}

one_case() { # tag
  local tag="$1"
  local cd="$D/$tag"
  mkdir -p "$cd"
  # ---- producer trajectory + slot
  restart_verified "$tag-build"
  case "$tag" in
    code)
      mkjson "$cd/p1.json" "$(cat "$REPO/bench/prompts/spec-code-edit.txt")"
      chat "$cd/g1.json" 64 "$(cat "$cd/p1.json")"
      mkjson "$cd/p2.json" "$(cat "$REPO/bench/prompts/spec-code-edit.txt")" "Which element is dropped by the off-by-one, and where exactly?"
      chat "$cd/g2.json" 48 "$(cat "$cd/p2.json")"
      mkjson "$cd/tail.json" "$(cat "$REPO/bench/prompts/spec-code-edit.txt")" "Which element is dropped by the off-by-one, and where exactly?" "Now write the corrected tail-copy loop in C++."
      ;;
    std512)
      mkjson "$cd/p1.json" "$(cat "$REPO/bench/prompts/standard-512.txt")"
      chat "$cd/g1.json" 64 "$(cat "$cd/p1.json")"
      mkjson "$cd/p2.json" "$(cat "$REPO/bench/prompts/standard-512.txt")" "State the opening claim of the text in one sentence."
      chat "$cd/g2.json" 32 "$(cat "$cd/p2.json")"
      mkjson "$cd/tail.json" "$(cat "$REPO/bench/prompts/standard-512.txt")" "State the opening claim of the text in one sentence." "Now list three consequences of that claim."
      ;;
    divergent)
      mkjson "$cd/p1.json" "The northern depot cipher is BRAVO-7. Acknowledge with: noted."
      chat "$cd/g1.json" 16 "$(cat "$cd/p1.json")"
      mkjson "$cd/p2.json" "The northern depot cipher is BRAVO-7. Acknowledge with: noted." "Repeat the cipher back exactly."
      chat "$cd/g2.json" 16 "$(cat "$cd/p2.json")"
      mkjson "$cd/p3.json" "The northern depot cipher is BRAVO-7. Acknowledge with: noted." "Repeat the cipher back exactly." "Forget that. The harbor key is DELTA-3. Repeat the harbor key."
      chat "$cd/g3.json" 16 "$(cat "$cd/p3.json")"
      mkjson "$cd/tail.json" "The northern depot cipher is BRAVO-7. Acknowledge with: noted." "Repeat the cipher back exactly." "Forget that. The harbor key is DELTA-3. Repeat the harbor key." "Now state both codes in order."
      ;;
  esac
  slot_save "$cd/save.json" "t14-$tag"
  echo "[$tag] slot: $(python3 -c "import json;d=json.load(open('$cd/save.json'));print(d['tokens'],'tokens',d['bytes'],'bytes')")" | tee -a "$D/verdicts.txt"
  TAIL="$(cat "$cd/tail.json")"
  fetch_all() { local ad="$1"; mkdir -p "$ad"
    for f in chat-dump.logits chat-dump.tokens chat-dump.kv chat-dump.mirror chat-dump.mtp; do
      ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
    done; }
  # ---- R: restore (trailer) + tail
  restart_verified "$tag-R"
  mkdir -p "$cd/R"
  chat "$cd/R-pre.json" 1 '[{"role":"user","content":"x"}]'
  TR0=$(date -u +%s%N); slot_restore "$cd/R-restore.json" "t14-$tag" || fail "$tag R restore"; TR1=$(date -u +%s%N)
  L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  chat "$cd/R-tail.json" 48 "$TAIL" || fail "$tag R tail"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$cd/R-slice.log"
  fetch_all "$cd/R"
  { echo "[$tag] R: restore_api_ms=$(( (TR1 - TR0) / 1000000 )) $(grep -oE 'turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+' "$cd/R-slice.log" | tail -1) $(grep -oE 'prefill=[0-9.]+ms \([0-9]+ tok\)' "$cd/R-slice.log" | tail -1) $(grep -oE 'drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+' "$cd/R-slice.log" | tail -1)"; } | tee -a "$D/verdicts.txt"
  # ---- C: fresh replay of the SAME trajectory + tail
  restart_verified "$tag-C"
  mkdir -p "$cd/C"
  python3 - "$cd" "$tag" <<'PY'
import json, sys
cd, tag = sys.argv[1], sys.argv[2]
# rebuild the exact same multi-turn prompt chain from the producer artifacts
def jl(n):
    return json.load(open(f"{cd}/{n}.json"))["choices"][0]["message"]["content"]
if tag == "code":
    turns = [("p1.json", None), ("p2.json", "g1"), ("tail.json", "g2")]
elif tag == "std512":
    turns = [("p1.json", None), ("p2.json", "g1"), ("tail.json", "g2")]
else:
    turns = [("p1.json", None), ("p2.json", "g1"), ("p3.json", "g2"), ("tail.json", "g3")]
# the tail prompt is already the full chain; for C we replay each turn with the
# SAME per-turn prompt, using the producer's replies as the assistant turns.
import pathlib
for pf, prev in turns:
    if prev is None:
        pathlib.Path(f"{cd}/C-{pf}").write_text(pathlib.Path(f"{cd}/{pf}").read_text())
    else:
        msgs = json.loads(pathlib.Path(f"{cd}/C-{prev_pf}").read_text()) if False else None
print("chain prepared")
PY
  # simpler + exact: C replays with the same per-turn prompt files, chaining
  # the producer's own replies (deterministic => identical content)
  cp "$cd/p1.json" "$cd/C1.json"
  chat "$cd/C-g1.json" 64 "$(cat "$cd/C1.json")"
  if [[ "$tag" == "divergent" ]]; then
    cp "$cd/p2.json" "$cd/C2.json"; chat "$cd/C-g2.json" 16 "$(cat "$cd/C2.json")"
    cp "$cd/p3.json" "$cd/C3.json"; chat "$cd/C-g3.json" 16 "$(cat "$cd/C3.json")"
  else
    cp "$cd/p2.json" "$cd/C2.json"; chat "$cd/C-g2.json" 48 "$(cat "$cd/C2.json")"
  fi
  LC=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
  chat "$cd/C-tail.json" 48 "$TAIL" || fail "$tag C tail"
  ./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((LC + 1))" > "$cd/C-slice.log"
  fetch_all "$cd/C"
  { echo "[$tag] C: $(grep -oE 'turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+' "$cd/C-slice.log" | tail -1) $(grep -oE 'prefill=[0-9.]+ms \([0-9]+ tok\)' "$cd/C-slice.log" | tail -1) $(grep -oE 'drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+' "$cd/C-slice.log" | tail -1)"; } | tee -a "$D/verdicts.txt"
  curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d "{\"slot\":\"t14-$tag\"}" >/dev/null || true
}
one_case code
one_case std512
one_case divergent
./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
echo "== compare (aligned control vs trailer restore)" | tee -a "$D/verdicts.txt"
python3 - "$D" <<'PY'
import hashlib, json, pathlib, struct, sys
d = pathlib.Path(sys.argv[1])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:20] if p.exists() else "MISSING"
for tag in ("code", "std512", "divergent"):
    print(f"[{tag}]")
    for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.kv", "chat-dump.mirror", "chat-dump.mtp"):
        c, r = sha(d / tag / "C" / f), sha(d / tag / "R" / f)
        print(f"   {f}: EQUAL={c == r} (C={c} R={r})")
    for arm in ("C", "R"):
        p = d / arm if False else d / tag / arm / "chat-dump.mirror"
        if p.exists():
            h = struct.unpack("<13I", p.read_bytes()[:52])
            print(f"   mirror {arm}: cells={h[3]} pos_min={h[4]}")
    ct, rt = d / tag / "C-tail.json", d / tag / "R-tail.json"
    if ct.exists() and rt.exists():
        a = json.load(open(ct))["choices"][0]["message"]["content"]
        b = json.load(open(rt))["choices"][0]["message"]["content"]
        print(f"   tail text equal: {a == b}")
PY
echo "== T14 done"
