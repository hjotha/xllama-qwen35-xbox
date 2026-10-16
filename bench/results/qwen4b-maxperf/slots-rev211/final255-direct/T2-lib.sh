#!/usr/bin/env bash
# T2-lib.sh — shared helpers for the T2 expanded reversible tests (plan 009).
# API-level only; no code changes; MTP stays on; same thresholds as the
# battery (exact tuples, exact texts, verified restarts). Sourced by
# T2a-slots.sh / T2b-slots.sh with $REPO $D $PFN $PFN_EXPECT $API set.
pass() { echo "PASS: $1" | tee -a "$D/verdicts.txt"; }
fail() { echo "FAIL: $1" | tee -a "$D/verdicts.txt"; return 1; }
chat() { # out max_tokens messages_json
  curl -s -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen35-4b-mtp\",\"temperature\":0,\"max_tokens\":$2,\"messages\":$3}" > "$1"
  grep -q '"choices"' "$1" || { fail "chat $1 no choices"; return 1; }
}
slot_save() { # out id
  curl -s -m 300 "$API/v1/slots/save" -H 'Content-Type: application/json' -d "{\"slot\":\"$2\"}" > "$1"
  grep -q '"ok":true' "$1" || { fail "slot save $2"; return 1; }
}
slot_restore() { # out id
  curl -s -m 300 "$API/v1/slots/restore" -H 'Content-Type: application/json' -d "{\"slot\":\"$2\"}" > "$1"
  grep -q '"ok":true' "$1"
}
slot_delete_q() { # id (quiet)
  curl -s -m 30 "$API/v1/slots/delete" -H 'Content-Type: application/json' -d "{\"slot\":\"$1\"}" >/dev/null || true
}
restart_verified() { # reason
  ./scripts/deploy.sh stop-app "$PFN" >/dev/null 2>&1 || true
  local down=0
  for _ in $(seq 1 20); do
    if ! curl -s -m 4 "$API/" >/dev/null 2>&1; then down=$((down + 1)); else down=0; fi
    [[ "$down" -ge 3 ]] && break
    sleep 3
  done
  [[ "$down" -ge 3 ]] || { fail "restart ($1): app never died"; return 1; }
  ./scripts/deploy.sh start-app "$PFN" >/dev/null 2>&1 || true
  local ok=""
  for _ in $(seq 1 60); do
    if curl -s -m 5 "$API/" >/dev/null 2>&1; then ok=1; break; fi
    sleep 3
  done
  [[ -n "$ok" ]] || { fail "restart ($1): app never came back"; return 1; }
  pass "restart verified ($1)"
}
set_flag() { # mode | none
  ./scripts/deploy.sh delete-file "$PFN" mtp-rebuild-test.flag >/dev/null 2>&1 || true
  if [[ "$1" != "none" ]]; then
    printf '%s' "$1" > "$D/flagbody.txt"
    ./scripts/deploy.sh upload-file "$D/flagbody.txt" "$PFN" "" mtp-rebuild-test.flag >/dev/null
  fi
}
ws_sample() { # label — process working set in MB (portal taskmanager)
  local ws raw
  raw=$(./scripts/deploy.sh status "$PFN" 2>/dev/null | grep -oE "ws=[0-9]+" | head -1)
  ws=${raw#ws=}
  if [[ -n "$ws" ]]; then
    echo "WS $1: $((ws / 1048576)) MB (raw=$ws)" | tee -a "$D/verdicts.txt"
  else
    echo "WS $1: UNAVAILABLE" | tee -a "$D/verdicts.txt"
  fi
}
content_of() { python3 -c "import json,sys;print(json.load(open('$1'))['choices'][0]['message']['content'])"; }
usage_of() { python3 -c "import json,sys;u=json.load(open('$1'))['usage'];print(f\"prompt_tok={u['prompt_tokens']} completion_tok={u['completion_tokens']}\")"; }
slice_turn() { # slicedir — last turn/prefill/drafted/rounds lines
  local ad="$1"
  grep -oE "turn full=[01] kv_before=[0-9]+ pf=[0-9]+ prompt=[0-9]+" "$ad/slice.log" | tail -1
  grep -oE "prefill=[0-9.]+ms \([0-9]+ tok\) reuse=[01]" "$ad/slice.log" | tail -1
  grep -oE "drafted=[0-9]+ \(mtp=[0-9]+ lookup=[0-9]+\) spec_accept=[0-9]+" "$ad/slice.log" | tail -1
  grep -oE "rounds=[0-9]+ decodes=[0-9]+ discarded=[0-9]+ catchup_tok=[0-9]+" "$ad/slice.log" | tail -1
}
