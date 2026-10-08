# SWIGLU extension gates (rev154, default OFF) — 256-token + Session

Package: GianlucaMazza.xllama_1.6.0.154_x64__pj67f1fcj4n14

## Full-ID parity: EXACT

- chat256: `f78ce837...` both arms, runs 2/3.
- code256: `c7d84255...` both arms, runs 2/3.
- resume-parity=OK (chat256/code256 sw0 vs sw1).

## Per-round proposal/acceptance traces (VROUND feed=, accepted=)

- code256: 87 rounds, sequences identical OFF vs ON.
- chat256: sequences diverge at round 49 (sw0 93 rounds, sw1 94; each arm
  self-identical across run2/run3), while the final full IDs remain exact.
  This is a per-round proposal divergence — the draft context's FFN SWIGLU
  also runs on D3D12, shifting draft proposals by ULPs — not a committed-token
  divergence. Per the strict per-round gate it is a FAIL for this cell; the
  canonical full-ID gate passes.

## Session scenarios (--mtp-session): PASS

- Both arms 4/4 ok=1 parity=1 first_diff=-1 (reset_prompt_swap, edited_prefix,
  delta_continuation, multi_chunk); scenario token dumps byte-identical
  (22/24 files; `bench-mtp-session.csv` and `config.txt` differ by metadata
  only).

## Timing (profile OFF; 2-4 rows/arm)

- chat256 clean interleaved pair: OFF 28.13/28.36/28.05/28.04 vs ON
  29.59/29.45/29.52/29.66 → **+5.22%** (all rows separated).
- NOTE: the first chat256 OFF cell ran while a 492 MB device log was being
  written (leftover `ggmlprof=sched2`) and was discarded as contaminated
  (it showed +23%; the clean re-run is +5.22%).
- code256 pair (post-cleanup): +4.30%.

## Verdict

Knob stays default OFF. The candidate passes full-ID parity, Session parity
and timing (+3.6-6.5% across workloads), but the 256-token chat per-round
proposal trace diverges with equal final IDs; the strict per-round gate is
therefore not fully met. Next: either restrict the whitelist so the draft
context's FFN SWIGLU stays on CPU (needs a context/name discriminator), or
accept the full-ID-only criterion; then termination/API gates and the final
optimized build.
