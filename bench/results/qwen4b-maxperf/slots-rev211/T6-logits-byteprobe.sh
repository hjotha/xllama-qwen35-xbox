#!/usr/bin/env bash
# T6-logits-byteprobe.sh — plan 009 S3 closure: byte-level comparison of
# per-step logits + sampled token ids, fresh prefill vs slot-restore in a
# NEW process, on the rev239 Iteration probe (chat-dump.flag armed, default
# OFF). Identical prompt/mode (model qwen35-4b-mtp, greedy, max_tokens=16,
# batch 256/ubatch 64 via the API defaults). Separated restore/prefill/
# decode times per arm; WS per arm; text sha per arm. Comparison is pure
# sha256 byte equality (no epsilon); a first-diff offset report is produced
# by compare-chatdump.py when they differ.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
D="$REPO/bench/results/qwen4b-maxperf/slots-rev211/T6b"
PFN_EXPECT="GianlucaMazza.xllama_1.6.0.240_x64__pj67f1fcj4n14"
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
P1='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]'
P2='[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."},{"role":"assistant","content":"noted"},{"role":"user","content":"Repeat the cipher back exactly."}]'

printf '1' > "$D/chat-dump.flag"
arm_flag() { ./scripts/deploy.sh upload-file "$D/chat-dump.flag" "$PFN" "" chat-dump.flag >/dev/null; }
arm_flag
echo "flag armed" | tee -a "$D/verdicts.txt"

capture() { # ad  — fetch the three dump files
  local ad="$1"
  for f in chat-dump.logits chat-dump.tokens chat-dump.tokens.prefill; do
    ./scripts/deploy.sh fetch-file "$PFN" "$f" "$ad/$f" >/dev/null 2>&1 || echo "MISSING $f" | tee -a "$D/verdicts.txt"
  done
}
times() { # slicefile
  local sl="$1"
  grep -m1 -oE "KV state loaded — [0-9]+ tokens, [0-9.]+ MB, [0-9.]+ ms" "$sl" || echo "no-restore (fresh arm: no restore ran)"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$sl" | tail -1 || true
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$sl" | tail -1 || true
  grep -oE "decode: [0-9.]+ tok/s" "$sl" | tail -1 || true
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$sl" | tail -1 || true
}

# FRESH: new process, no restore, full prompt
restart_verified "fresh"
L0=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
ws_sample "fresh-pre"
chat "$D/fresh-cont.json" 16 "$P2" || fail "fresh cont"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L0 + 1))" > "$D/fresh-slice.log"
{ echo "== FRESH"; times "$D/fresh-slice.log"
  python3 -c "import json;print('text_sha', __import__('hashlib').sha256(json.load(open('$D/fresh-cont.json'))['choices'][0]['message']['content'].encode()).hexdigest()[:16])"; } | tee -a "$D/verdicts.txt"
mkdir -p "$D/fresh"; capture "$D/fresh"; ws_sample "fresh-post"

# RESTORE: seed+save in one process, NEW process, restore, same prompt
restart_verified "restore-seed"
chat "$D/seed.json" 16 "$P1"
slot_save "$D/save.json" "t6"
restart_verified "restore-new"
chat "$D/pre.json" 1 '[{"role":"user","content":"x"}]'
L1=$(./scripts/deploy.sh get-log "$PFN" 2>/dev/null | wc -l)
ws_sample "restore-pre"
slot_restore "$D/restore.json" "t6" || fail "restore"
chat "$D/restore-cont.json" 16 "$P2" || fail "restore cont"
./scripts/deploy.sh get-log "$PFN" 2>/dev/null | tail -n +"$((L1 + 1))" > "$D/restore-slice.log"
{ echo "== RESTORE"; times "$D/restore-slice.log"
  python3 -c "import json;print('text_sha', __import__('hashlib').sha256(json.load(open('$D/restore-cont.json'))['choices'][0]['message']['content'].encode()).hexdigest()[:16])"; } | tee -a "$D/verdicts.txt"
mkdir -p "$D/restore"; capture "$D/restore"; ws_sample "restore-post"

./scripts/deploy.sh delete-file "$PFN" chat-dump.flag >/dev/null 2>&1 || true
curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d '{"slot":"t6"}' >/dev/null || true
echo "== compare" | tee -a "$D/verdicts.txt"
python3 - "$D" <<'PY'
import hashlib, sys, pathlib
d = pathlib.Path(sys.argv[1])
for f in ("chat-dump.logits", "chat-dump.tokens", "chat-dump.tokens.prefill"):
    a, b = d / "fresh" / f, d / "restore" / f
    if not a.exists() or not b.exists():
        print(f"{f}: MISSING fresh={a.exists()} restore={b.exists()}")
        continue
    ha = hashlib.sha256(a.read_bytes()).hexdigest()
    hb = hashlib.sha256(b.read_bytes()).hexdigest()
    print(f"{f}: fresh_sha={ha[:32]} restore_sha={hb[:32]} BYTE_EQUAL={ha == hb} bytes={a.stat().st_size}/{b.stat().st_size}")
PY
echo "== T6 done"
