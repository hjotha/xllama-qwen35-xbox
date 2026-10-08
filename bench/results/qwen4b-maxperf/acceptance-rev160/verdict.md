# Final acceptance — rev160 (LTCG) — FFN SWIGLU sequential candidate

- Package: `GianlucaMazza.xllama_1.6.0.160_x64__pj67f1fcj4n14`
- MSIX sha256: `74C9DEDDA15493A6673353F08E29A77EE16D21A124783BA023DC8956A03EF1C7`
  (builder-log `sha256-fixed` == fetched local sha; LTCG final build, 1049.51 s)
- Model: `qwen35-4b-mtp` sha256
  `3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630`
- Knob `d3d12swiglu.txt` default OFF; process profile binds once at the first
  context and never changes in-process.

## Full-token parity (run1+run2, all cells)

OK against the recorded baseline digests for both knob OFF and ON: five
sequential pairs (chat64 `84e08196…`, code64 `ff42c5be…`, chat256
`f78ce837…`, code256 `c7d84255…`, std512 `44546453…`) plus three MTP pairs
(chat64-mtp, code64-mtp, chat256-mtp — same digests), runs 1-2.
**code256 and std512 were not run on the MTP arm in the final grid.**

## Timing (median of 2 rows/cell, profile OFF, knob OFF vs ON)

| cell        | OFF    | ON     | delta  |
| ----------- | ------ | ------ | ------ |
| chat64      | 23.950 | 26.300 | +9.81% |
| code64      | 24.020 | 25.370 | +5.62% |
| chat256     | 24.420 | 26.090 | +6.84% |
| code256     | 23.510 | 24.730 | +5.19% |
| std512      | 23.780 | 25.210 | +6.01% |
| chat64-mtp  | 28.940 | 28.780 | −0.55% |
| code64-mtp  | 34.440 | 34.480 | +0.12% |
| chat256-mtp | 28.090 | 28.260 | +0.61% |

MTP rows are the **FFN-forced-OFF** arm (knob=1 binds the MTP process to FFN
off): deltas are within run noise — no MTP regression, no dispatch.

Prior gate-run labels (kept as measured, **not swapped**): rev154 chat64-seq
**+6.53%**, code64-seq **+5.65%**. The rev154 MTP rows (chat64 +4.42%,
code64 +3.61%) were **FFN OFF/ON trials with C8 held fixed** gate-wide: they
are the rejected FFN candidate's gains — invalidated by the chat256 round-49
proposal divergence — and are **not C8 evidence**. C8 was measured separately
in its own A/B (rev131: +1.6 to +3.2% per arm, seq and MTP).

## Termination gate (termgate `cancel,stop,eog`, 6 arms)

All data rows `ok=1`; sw0 vs sw1 CSVs byte-identical for seq, ref, cand;
device log shows seq process `profile bound: on (mtp_capable=0)` with FFN
dispatches, MTP processes `bound: off (mtp_capable=1)` with zero dispatches.

## Session gate (`--mtp-session --mtp 4 --mtp-pmin 75`, knob ON)

4/4 scenarios OK (reset_prompt_swap, edited_prefix, delta_continuation,
multi_chunk), drafter active, ids matched the reference.

## API gate (OpenAI-compat `/v1/chat/completions`, knob ON, seq session)

`validate-api.sh chat` with `MODEL=qwen35-4b-mtp`: PASS (assistant replied,
concurrency 200/200). Note: the script's default model (`lfm25-350m`) is
rejected by the startup scope2 profile — unrelated to FFN; use
`MODEL=qwen35-4b-mtp`.

## Limitations (explicit)

- The knob remains **default OFF**; the sequential-only FFN profile is
  experimental until promoted by a separate decision.
- Enabling it is a **process-level choice, fixed for the process lifetime**
  (a restart re-evaluates it): the first context fixes the profile; a later
  conflicting MTP context is rejected before creation ("restart required");
  no capability change under a live context.
- **MTP FFN candidate is closed**: the target-only discriminator still
  diverges from OFF at chat256 round 49 (final IDs exact in all arms) — the
  target-FFN arithmetic drifts the drafter. MTP keeps FFN OFF; no FFN gain is
  claimed on the MTP arm. C8 (timestamp gating) is independent and was
  measured in its own A/B (rev131: +1.6 to +3.2% per arm).
- No additional bitwise arithmetic rewrites were attempted (per review).
