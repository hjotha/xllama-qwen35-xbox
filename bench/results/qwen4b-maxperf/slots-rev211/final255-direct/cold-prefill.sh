#!/usr/bin/env bash
# T10-trajectory-control.sh — plan 009 G-L rigorous isolation on rev255 Final/LTCG.
# Same prompt trajectory and segmentation for the control, compared at the
# SAME point (post-prefill, pre-decode) of the tail turn:
#   P  producer: T1 -> T2 -> T3 (multi-turn assembly), save slot S
#   C  control:  fresh process, REPLAY T1 -> T2 -> T3 (same prompts, same
#                segmentation), then the SAME tail P4
#   R  restore:  new process, preload, restore S, then the SAME tail P4
#   F  fresh:    new process, ONE turn with the full continuation prompt
#                (P3+r3+P4) — the multi-ubatch single-prefill arm
# Arms are compared at the tail turn's pre-decode point with: target KV
# bytes, MTP mirror KV bytes, pending row, token ids/positions, and the
# per-step logits/token ids. No epsilon: byte comparison only.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/final255-direct/cold-prefill"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14"
API=http://192.168.1.26:11434
cd "$REPO"
set -a
source ~/.config/xllama/xbox-env
set +a
mkdir -p "$D"
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
[[ "$PFN" == "$PFN_EXPECT" ]] || { echo "PFN_MISMATCH got=$PFN" >&2; exit 1; }
# shellcheck disable=SC1091
source "$D/../T2-lib.sh"
: > "$D/verdicts.txt"
curl -fsS -m 30 "$API/v1/slots" > "$D/slots-pre.json"
python3 - "$D/slots-pre.json" <<'CHECK'
import json,sys
assert json.load(open(sys.argv[1]))['slots']==[]
CHECK
printf '1' > "$D/sw-on.txt"
./scripts/deploy.sh upload-file "$D/sw-on.txt" "$PFN" "" d3d12swiglu.txt >/dev/null

P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
ASK2="Describe how a sorted merge works, step by step."
ASK3="Now explain RoPE attention to a junior engineer."
ASK4="Write a short operational briefing on the northern depot cipher from the first message, then the sorted-merge summary, then RoPE in two lines."

printf '1' > "$D/chat-dump.flag"
./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null
echo "flag armed" | tee -a "$D/verdicts.txt"

# ---- P: the producer trajectory; replies become the fixed prompt files
restart_verified "P"
chat "$D/t1.json" 16 "$P1"
R1=$(content_of "$D/t1.json")
python3 - "$D" "$R1" "$ASK2" <<'PY'
import json, sys
d, r1, ask2 = sys.argv[1], sys.argv[2], sys.argv[3]
open(f"{d}/T2.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": ask2}]))
PY
chat "$D/t2.json" 48 "$(cat "$D/T2.json")"
R2=$(content_of "$D/t2.json")
python3 - "$D" "$R1" "$R2" "$ASK3" <<'PY'
import json, sys
d, r1, r2, ask3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
open(f"{d}/T3.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": r2},
    {"role": "user", "content": ask3}]))
PY
chat "$D/t3.json" 48 "$(cat "$D/T3.json")"
R3=$(content_of "$D/t3.json")
python3 - "$D" "$R1" "$R2" "$R3" "$ASK4" <<'PY'
import json, sys
d, r1, r2, r3, ask4 = sys.argv[1:6]
open(f"{d}/TAIL.json", "w").write(json.dumps([
    {"role": "user", "content": "The northern depot cipher is BRAVO-7. Acknowledge with: noted."},
    {"role": "assistant", "content": r1},
    {"role": "user", "content": "Describe how a sorted merge works, step by step."},
    {"role": "assistant", "content": r2},
    {"role": "user", "content": "Now explain RoPE attention to a junior engineer."},
    {"role": "assistant", "content": r3},
    {"role": "user", "content": ask4}]))
print("prompt files fixed")
PY
slot_save "$D/save.json" "r255cold"
echo "P done; slot tokens: $(python3 -c "import json;print(json.load(open('$D/save.json'))['tokens'])")" | tee -a "$D/verdicts.txt"

# ---- C: control — same trajectory, then the same tail
restart_verified "C1"
chat "$D/c1.json" 16 "$P1"
chat "$D/c2.json" 48 "$(cat "$D/T2.json")"
chat "$D/c3.json" 48 "$(cat "$D/T3.json")"
mkdir -p "$D/C"
mkdir -p "$D/C"; for f in chat-dump.logits chat-dump.tokens chat-dump.tokens.prefill chat-dump.kv chat-dump.mirror chat-dump.mtp; do ./scripts/deploy.sh fetch-file "$PFN" "$f" "$D/C/pre-$f" >/dev/null 2>&1 || echo "MISSING pre-$f" | tee -a "$D/verdicts.txt"; done
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/C/tail.json" 32 "$(cat "$D/TAIL.json")" || fail "C tail"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/C/tail-slice.log"
mkdir -p "$D/C"; for f in chat-dump.logits chat-dump.tokens chat-dump.tokens.prefill chat-dump.kv chat-dump.mirror chat-dump.mtp; do ./scripts/deploy.sh fetch-file "$PFN" "$f" "$D/C/$f" >/dev/null 2>&1 || echo "MISSING C/$f" | tee -a "$D/verdicts.txt"; done
{ echo "== C tail"; grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/C/tail-slice.log" | tail -1 || true
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/C/tail-slice.log" | tail -1 || true
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/C/tail-slice.log" | tail -1 || true; } | tee -a "$D/verdicts.txt"

# ---- R: restore the SAME slot, then the same tail
restart_verified "R"
mkdir -p "$D/R"
chat "$D/r-pre.json" 1 '[{"role":"user","content":"x"}]'
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
slot_restore "$D/r-restore.json" "r255cold" || fail "R restore"
chat "$D/r-tail.json" 32 "$(cat "$D/TAIL.json")" || fail "R tail"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/R/tail-slice.log"
mkdir -p "$D/R"; for f in chat-dump.logits chat-dump.tokens chat-dump.tokens.prefill chat-dump.kv chat-dump.mirror chat-dump.mtp; do ./scripts/deploy.sh fetch-file "$PFN" "$f" "$D/R/$f" >/dev/null 2>&1 || echo "MISSING R/$f" | tee -a "$D/verdicts.txt"; done
{ echo "== R tail"
  grep -m1 -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$D/R/tail-slice.log" || echo "no-restore"
  grep -m1 -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/R/tail-slice.log" || true
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/R/tail-slice.log" | tail -1 || true
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/R/tail-slice.log" | tail -1 || true; } | tee -a "$D/verdicts.txt"

# ---- F: fresh, one full turn of the whole continuation prompt
restart_verified "F"
mkdir -p "$D/F"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
chat "$D/F/tail.json" 32 "$(cat "$D/TAIL.json")" || fail "F tail"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/F/tail-slice.log"
mkdir -p "$D/F"; for f in chat-dump.logits chat-dump.tokens chat-dump.tokens.prefill chat-dump.kv chat-dump.mirror chat-dump.mtp; do ./scripts/deploy.sh fetch-file "$PFN" "$f" "$D/F/$f" >/dev/null 2>&1 || echo "MISSING F/$f" | tee -a "$D/verdicts.txt"; done
{ echo "== F tail"; grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$D/F/tail-slice.log" | tail -1 || true
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$D/F/tail-slice.log" | tail -1 || true
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$D/F/tail-slice.log" | tail -1 || true; } | tee -a "$D/verdicts.txt"

./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"r255cold"}' >/dev/null || true
echo "== compare (byte only)" | tee -a "$D/verdicts.txt"
python3 - "$D" <<'PY'
import hashlib, pathlib, sys
d = pathlib.Path(sys.argv[1])
def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.kv", "chat-dump.mirror", "chat-dump.mtp"):
    C, R, F = sha(d / "C" / f), sha(d / "R" / f), sha(d / "F" / f)
    print(f"{f}:")
    print(f"   C={C[:20]} R={R[:20]} F={F[:20]}")
    print(f"   C_vs_R EQUAL={C==R}  F_vs_C EQUAL={F==C}  F_vs_R EQUAL={F==R}")
# turn-boundary ledger
for arm in ("C", "R", "F"):
    sl = d / arm / "tail-slice.log"
    if sl.exists():
        tb = [l for l in sl.read_text(errors="replace").splitlines() if "chat turn:" in l or "KV state loaded" in l]
        print(f"{arm} turn-boundary lines: {len(tb)}")
        for l in tb[:3]:
            print("   ", l[:130])
PY
echo "== T10 done"

python3 - "$D" <<'CHECK'
import hashlib,json,pathlib,sys
d=pathlib.Path(sys.argv[1])
save=json.loads((d/"save.json").read_text())
restore=json.loads((d/"r-restore.json").read_text())
assert save["ok"] is True and restore["ok"] is True and save["tokens"]==restore["tokens"]
report={"revision":255,"mode":"Final/LTCG","snapshot_tokens":save["tokens"],"comparisons":{}}
for f in ("chat-dump.logits","chat-dump.tokens","chat-dump.kv","chat-dump.mirror","chat-dump.mtp"):
    data={a:(d/a/f).read_bytes() for a in ("C","R","F")}
    report["comparisons"][f]={"sha256":{a:hashlib.sha256(x).hexdigest() for a,x in data.items()},"bytes":{a:len(x) for a,x in data.items()},"aligned_C_R_equal":data["C"]==data["R"],"cold_F_R_equal":data["F"]==data["R"]}
(d/"strict-receipt.json").write_text(json.dumps(report,indent=2)+"\n")
assert all(x["aligned_C_R_equal"] for x in report["comparisons"].values()),"Matched trajectory state/continuation regression"
assert report["comparisons"]["chat-dump.tokens"]["cold_F_R_equal"],"G-T cold token byte equality FAILED"
assert report["comparisons"]["chat-dump.logits"]["cold_F_R_equal"],"G-L cold logits byte equality FAILED; no epsilon permitted"
assert all(x["cold_F_R_equal"] for x in report["comparisons"].values()), "Cold state mismatch"
print("PASS: strict G-T and G-L and all state cold full-prefill byte equality")
CHECK
