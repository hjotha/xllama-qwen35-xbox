# CORRECTION: multi-turn/session gate v1 (api-multiturn-rev209.sh)

Status: the v1 "NO reload between 3 turns / session reuse" wording is WRONG
and is SUPERSEDED by the temporal map below. What stands from v1: 3x HTTP
200 with the recorded replies (turn2 exact recall of QUARTZ-LUMEN-4821),
single bind `on (mtp_capable=1)`, MTP drafting active on every turn, real
counters at backend_free (383 calls / 524 matmuls / 48 GATED_DELTA_NET /
79 FFN SWIGLU). The recall JSONs prove API-level conversation continuity
(history carried in the request messages), NOT KV-cache reuse.

Temporal map of api/session-segment.log (538 lines, L0=424225):
- L8-76: startup load of qwen35-4b-mtp + bind `on (mtp_capable=1)`.
- L244: `session: turn full=1 kv_before=0 pf=50 prompt=50` = T1, full
  prefill (correct for a first turn on a fresh session).
- L256-259: T1 generates 1 token (`session generate: n=1 ... reuse=0
  drafted=2 ...`).
- L264-265: manifest merge + `ini mtp REJECTED: model 'lfm25-350m'...`
  (per-model MTP admission for the non-loaded catalogue default).
- L268-274: backend_free flushes (383 calls / 524 matmuls / 48 GDN /
  79 FFN SWIGLU) — the session's GPU backend is TORN DOWN between T1 and T2.
- L275-276: default-model preload REJECTED (non-fatal) under the leftover
  bench scope2 profile (`d3d12twocol.txt`/`cpurepackforcegemv.txt` from the
  termgate battery — still present during the v1 gate).
- L277/L281+: `session config: model=qwen35-4b-mtp ...` — OUR session is
  RE-CREATED (full load_tensors).
- L488: `session: turn full=1 kv_before=0 pf=77 prompt=77` = T2, full
  prefill with EMPTY resident KV (T1's KV was destroyed with the session).
- L505-511: T2 generates 11 tokens (the codeword reply).
- L517: `session: KV rewind unsupported (hybrid cache) — full re-prefill
  (#170): common=73 resident=88 prompt=120` = T3: the bridge DID attempt
  reuse (common prefix 73 of resident 88), but the hybrid cache refuses
  tail-rewind (src/bridge/session.cpp:699-700, #170a), so full re-prefill.
- L518/L529-533: T3 full prefill (120 tok) + 6 generated tokens.

Mechanism (src/bridge/session.cpp:640-790, session_hub.h:23-123): API turns
arrive as FULL prompts (full=1); the bridge rewinds resident KV to the
common token prefix when possible ("KV prefix reuse — kept N of M" when it
works). On this hybrid-cache model the rewind is refused, so every API turn
full-prefills from the request history. A background default-model preload
(which bumps hub generation and clears KV) plus the leftover bench scope2
knobs explain the T1->T2 session recreation in v1's environment — the v2
redo runs in the CLEAN state (knobs verified absent) to test whether the
interference persists, and adds the OFF equivalent + per-turn log slices.

Consequences: api/receipt.txt line "turn2_recall=PASS" stays TRUE (recall);
the v1 receipt line "model_load_lines_in_segment=6 (0 => no reload across
the 3 turns)" is FALSE (reload at L277 between T1 and T2) and is corrected
here; api/notes-final-cleanup.txt "NO load/reload between the 3 turns" is
corrected here; receipt.txt section 5 and plans/008 DELIVERED section are
corrected to "API-level continuity (request history), full prefill per turn
on this hybrid model" once the v2 redo confirms the clean-state behavior.
Full prefill is NEVER reclassified as reuse.
