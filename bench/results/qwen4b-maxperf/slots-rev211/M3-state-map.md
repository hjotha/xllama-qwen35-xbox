# M3 state map — novo processo, zerocarry (v2), MTP-on

Instrumentation note: the `turn-end drafter state` line reports
pending_pos/mirror_end/kv_tokens = the mirror's MAX POSITION, not its
valid-row set. Zerocarry and control print IDENTICAL end lines
(pending_pos=65 mirror_end=65 kv_tokens=66) while the underlying row
sets DIFFER — aggregate lines cannot serve as equivalence proof (this
is exactly why they were excluded from evidence).

## Restored from the v2 slot file (identical to a plain restore)

| state | mechanism | reaches fresh-equivalent? |
| ----- | --------- | ------------------------ |
| TARGET KV, seq 0, positions [0,n) | `llama_state_seq_set_data` (session.cpp:1539) — includes all cache tensors of the context, recurrent/hybrid cells included when the fork serialises them | YES — bit-identical bytes by construction of the file |
| token ids | `m_kv_tokens := file tokens` (1562) | YES |
| GDN/recurrent state | part of the TARGET KV blob; on restore the d3d12 GDN op is pure dispatch over cache cells (ggml_d3d12.cpp: d3d12_gdn_supported gate) — no separate state exists outside the cache | YES (same blob) |
| sampler | greedy: no cross-turn state (documented: session.cpp:972-977) | YES by construction |

## Drafter (MTP) state after zerocarry install

| state | zerocarry value | fresh-equivalent value (steady state) | equal? |
| ----- | --------------- | ------------------------------------- | ------ |
| draft-context KV rows [0, kv_keep) | ABSENT — hole. `mtp_catchup_batch` replays only the current batch (decode_loop.h:564-595), so a tail-only turn writes rows [kv_keep, kv_keep+pf) only. `align_to(0)` at load cleared everything below | full prefix replayed across prior turns (process() per batch) | **NO — the hole is the real difference** |
| draft-context KV rows [kv_keep, end) | replayed with TRUE target hidden rows (h_rows) | same | YES |
| pending row (carry into next replay) | ZERO vector at pos n-1 (M3 install) | TRUE hidden state of token n-1 (from the target's last decode) | **NO — decoy for exactly one row's input** |
| carry snapshot (m_carry_*) | none | last committed prefix end + true row | NO (advisory only; commit_carry path) |

## Where the decoys act (one row each)

1. `process(tail, h_rows, nullptr, pos0=kv_keep)`: first replay row
   consumes the pending (zero) instead of the true state of token
   kv_keep-1 (mtp_draft.cpp:524-547). One row of the mirror is computed
   from a false input: position kv_keep-1.
2. `draft()` at later rounds reads the true pending (process() rewrites
   it from h_rows at the end — mtp_draft.cpp:606-607), so the decoy does
   not propagate.
3. Rows [0, kv_keep-2) stay empty for the session: draft attention over
   the deep prefix is structurally missing; on the tested prompts
   acceptance stayed 81-88% (see T3-M3-report) and text stayed
   identical — but nothing in the mechanism guarantees this for
   deep-prefix-dependent continuations.

## Safety / fallback (structural, not tuned)

- Drafts are advisory: the target verifies every candidate against its
  own logits, so a wrong draft costs time, never output text. Empty
  draft => classic decode (existing F2.3 behavior).
- Construction is fail-closed: no drafter / n mismatch / bad mode =>
  no carry install; the load path keeps the existing `align_to(0)` +
  full-prefill behavior (load_state unchanged in rev235 baseline).
- If acceptance collapses on a new prompt family, the mechanism
  degrades toward classic-decode speed with IDENTICAL text — the
  measurement, not a threshold, decides M3's validity.

## Unproven bounds (drives item 2)

Deep-prefix-dependent continuations; code/library families; multi-turn
chains on one restored slot; minimal suffixes; long generations (256+).
These are the exact axes of the amplification matrix (T4).
