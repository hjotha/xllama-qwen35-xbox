# Plan 008 complete island — implemented, engages, ADD/RMS gates green, timing INCONCLUSIVE

## Engagement and structure (rev174, same build)

- Knob `d3d12island.txt=1` -> log `island D3D12 (SWIGLU+ADD+RMS_NORM+MUL)`;
  last-run dispatch counters: **2150 RMS_NORM + 2487 MUL + 1510 ADD**.
- Ordered split adjacency (decode graphs): transitions **6780 -> 5726
  (-15.5%)**; prefill **1845 -> 1535 (-16.8%)**. Boundary diff matches the
  plan-008 prediction: `MUL_MAT->ADD` -1166, `MUL->MUL_MAT` -814,
  `MUL_MAT->RMS_NORM` -214; residual `MUL->SCALE/ROPE/GET_ROWS` +853.

## Numerical gates (device)

- Selftest (`d3d12be`, rev174) — **scoped**: the NEW island ops pass —
  `add_f32` bit_mismatch=0 max_ulp=0 (36 864 elements incl. NaN/Inf/±0/
  subnormals) and `rms_norm_f32` exact=**36864/36864** max_rel=0 max_ulp=0
  (incl. all-zero/all-ones/NaN rows). The PRE-EXISTING C4 diagnostic kernels
  do **not** pass their adversarial corpus: `silu_avx2 ok=0` (ordered region
  bit-exact, `ord_ulp_max=0`; 80 edge cases with `all_ulp_max=53824988`,
  `all_rel=1`) and `swiglu_avx2 ok=0` (ordered region bit-exact; 180 edge
  cases, `all_ulp_max=62212381`, `all_abs=1.01e+31`). These are the
  documented C4 open items (polynomial branch edges |n|≈126/192, subnormals/
  extremes) now settled by the device corpus.
- Production-path scope of those failures: SILU is **selftest-only**
  (`g_silu_test`), never on a product path. SWIGLU reaches a product path only
  through the C4 knob (`d3d12swiglu.txt`, default absent = off) or the island
  knob, so the failing kernel is **not** on the tested production path
  (rev175 default: all off). It IS on the island-ON SEQ arm, whose digests
  were nevertheless exact on the tested prompts — the edge discrepancies did
  not flip a token there, but they are a real numerical debt for any seq FFN
  promotion.
- Full-token digests exact in all 8 extended cells: chat256 `f78ce8372ccc7281`
  and code256 `c7d84255e08e2077`, seq and MTP, island OFF and ON.
- Strict per-round VROUND feed/accepted traces: **IDENTICAL** OFF vs ON for
  chat256-MTP (93 rounds) and code256-MTP (87 rounds). (The seq path has no
  verify rounds; digests are its parity gate.) **This does NOT validate the
  previously divergent FFN/MTP path**: the product policy binds FFN OFF for
  MTP, so the island's MTP arm is ADD+RMS_NORM+MUL only and SWIGLU never ran
  there. The C4 FFN/MTP divergence question remains open and requires the
  equal-history diagnosis.
- API chat PASS; termgate 6/6 with the island ON, seq/ref/cand byte-identical
  to rev163; peak WS and gpu_mem unchanged (3670 MB / 2693 MB).

## Timing — INCONCLUSIVE across two same-build paired sessions

| session           | cell               | OFF   | ON    | delta                       |
| ----------------- | ------------------ | ----- | ----- | --------------------------- |
| rev172 (3 rounds) | std512 MTP prefill | 89.1  | 90.2  | +1.19%                      |
| rev172            | chat64 seq decode  | 24.55 | 26.26 | **+7.0%**                   |
| rev174 (3 rounds) | chat64 MTP decode  | 28.34 | 29.07 | +2.58% (k0 outlier 24.07)   |
| rev174 (7 rounds) | chat64 seq decode  | 26.13 | 26.33 | **+0.77%** (ranges overlap) |

The ON arms agree across builds (seq decode ~25.8-26.6); the OFF baselines
disagree (23.8-24.7 on rev172 vs 25.9-26.6 on rev174), so the rev172 +7% seq
delta is not reproduced. Two same-build interleaved measurements disagree on
the magnitude -> **no reproducible timing gain is demonstrated**; the rev172
numbers cannot be used as a claim.

## Decision

Keep the island **knob-gated OFF** (production-safe): structure and
correctness are proven, timing is not. Promotion requires a controlled
re-measurement (single session, more interleaved rounds, thermal control, both
arms) that separates the delta from baseline drift.

## Still open (owner)

- Equal-history FFN/MTP diagnosis (conditional histories, KV rollback, row
  alignment, RNG) with the strict trace gate retained.
- Same-boundary prefill instrument (`StepSplit("prefill")`) for the
  ubatch 128/256 regression, which is still unexplained at stage level.
