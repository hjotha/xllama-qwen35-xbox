# Plan 008 — complete elementwise island (SWIGLU + ADD + RMS_NORM + MUL)

## Adjacency evidence (ordered splits, rev165/rev168 censuses)

Decode graphs (95/run): 110 transitions/graph, 10 491 per run. The boundary
classes and what plan-007's partial move did:

| boundary class (k0, chain OFF)                 | per run | after plan-007 (k1)    |
| ---------------------------------------------- | ------- | ---------------------- |
| D3D12→CPU `MUL_MAT→ADD`                        | 1802    | 1802 (unchanged)       |
| CPU→D3D12 `SWIGLU→MUL_MAT`                     | 1525    | 1525 (unchanged)       |
| CPU→D3D12 `MUL→MUL_MAT`                        | 1455    | **93** (merged, −1262) |
| D3D12→CPU `MUL_MAT→RMS_NORM`                   | 1247    | 901 (−346)             |
| D3D12→CPU `MUL_MAT→SWIGLU`                     | 901     | 901                    |
| D3D12→CPU `MUL_MAT→CONCAT`                     | 624     | 624                    |
| CPU→D3D12 `SIGMOID→GATED_DELT`                 | 624     | 624                    |
| **new in k1** `ADD→RMS_NORM` (CPU→D3D12)       | –       | **+1802**              |
| **new in k1** `MUL→SCALE/ROPE/GET_ROWS/CONCAT` | –       | +1342                  |

Plan-007 merged its intended pair (−1608) but paid +3600 in new boundaries:
the now-GPU `RMS_NORM` is fed by the still-CPU residual `ADD` (+1802), and the
now-GPU norm `MUL` feeds still-CPU consumers (+1342). Net +1992 decode
transitions — the mechanism behind 124→148 splits/graph.

**Rule this establishes:** a node move removes boundaries only when the whole
elementwise run between two GPU matmuls moves together. That run is:

`MUL_MAT (D3D12) → SWIGLU → MUL_MAT (D3D12) → ADD → RMS_NORM → MUL → MUL_MAT (D3D12)`

Adjacency-proved island members and their boundary deltas if moved together:
`SWIGLU` (−1525 −901), `ADD` (−1802 and removes the k1 `ADD→RMS_NORM` +1802),
`RMS_NORM`/`MUL` (as plan 007, −1608, now without the +1802 penalty). Residual
unavoidable: `MUL→SCALE/ROPE/GET_ROWS/CONCAT` (~+1342, the norm weight MUL
feeding non-matmul consumers, mostly GDN layers) and `SIGMOID→GATED_DELT` and
`MUL_MAT→CONCAT` (out of scope). Expected net ≈ −4.5k of 10.5k decode
transitions if the prediction holds — to be measured, not claimed.

## Implementation (narrowly gated)

- One knob (`d3d12island.txt` = "1") enables the complete set; default off.
  Partial enabling is what plan 007 proved worthless.
- ADD kernel (new): f32 elementwise add, same-shape or broadcast, contiguous;
  bit-exact per IEEE (single rounded add per element, order-independent).
- SWIGLU: reuse the existing bit-exact C4 kernel + predicate (already in the
  tree, product path gated by `d3d12_set_swiglu_product_mode`); the island
  knob enables it for the FFN nodes instead of the rejected FFN-only knob.
- RMS_NORM + MUL: restore the plan-007 kernels (patch in `normmul-rev168/`),
  now enabled only as part of the island so the boundaries actually collapse.

## Numerical validation (before any timing claim)

1. ADD: bit-exact by construction; selftest rows vs the CPU kernel over
   same-shape and broadcast forms, adversarial values (NaN/Inf/±0, subnormals,
   extremes, odd sizes), plus row/length edges.
2. RMS_NORM: tolerance-based (double-lane reduction, precise sqrt/div).
   Report max relative error, ULP histogram and exact-match rate on a real
   corpus plus adversarial rows; the existing full-token digest gate decides.
3. SWIGLU: already bit-exact (C4 corpus, 4697 checks); re-run its selftest
   rows after wiring the island.
4. End-to-end: full-token digests (chat64/code64, seq+MTP), the strict
   per-round `VROUND feed=` trace gate (MTP), and with plan 007's chain the
   digests already held exactly.

## Measurement (same build, interleaved, no cross-build claims)

- OFF/ON paired, profile OFF: std512 prefill + chat64 decode, MTP and seq.
- Census (sched2) OFF vs ON: engagement counters (dispatches) and the actual
  per-graph transition count, decode and prefill.
- Accept only if the transition count materially drops AND the paired timing
  separates from noise with the gates green; otherwise revert to rev169.

## Still separate (owner 591/599)

- Equal-history FFN/MTP diagnosis (target/draft logits, row alignment, KV
  rollback, RNG) — the strict trace gate stays until that evidence exists.
- Prefill same-boundary instrument (`StepSplit("prefill")`) to explain why
  ubatch 128/256 regress while n_batch alone does not.

## Outcome (rev174): implemented, all gates green, timing not reproduced

- Engagement: 2150 RMS_NORM + 2487 MUL + 1510 ADD dispatches; decode
  transitions -15.5%, prefill -16.8% (ordered adjacency, same build).
- Gates: ADD bit-exact and RMS_NORM 36864/36864 exact (device selftest);
  all 8 extended digests exact; chat256/code256 MTP VROUND traces identical
  OFF/ON; API + termgate 6/6 identical; memory unchanged.
- Timing: rev172 showed +7% seq / ~flat MTP; rev174 (7 interleaved rounds)
  showed +0.77% seq with baselines shifted -> not reproducible; production
  stays OFF (knob-gated) pending a controlled re-measurement.

## Accepted config intervention (rev188): n_batch=256, n_ubatch=64

- Evidence: same-build paired A/B (3 interleaved reps) prefill +8.23% (std512)
  and +7.38% (code64) with decode flat; end-to-end LAN API total latency
  -204 ms (-4.8%) on a 275-token prompt and -75 ms (-3.3%) on a 96-token
  prompt; full-token digests identical; API PASS; termgate 6/6 identical;
  memory unchanged (peak 3673 MB, gpu 2692 MB).
- Mechanism (measured, not assumed): fewer prefill chunks at the same physical
  ubatch amortize the per-chunk wall-d3w; per-token CPU volume is lower at
  wider batches and the GPU time slightly better; the elembench kernels are
  width-flat and the MTP catch-up is not involved (seq control).
- Rollback: `bench/results/qwen4b-maxperf/nb256-rev187/llama.ini.backup`
  (n_batch=64) — one `deploy.sh upload-file` restores it.
- Installed: LTCG rev188 (sha 73CB3B61...), PFN ..._1.6.0.188_...

## Decode verify-round attribution round (rev188 LTCG, config 256/64)

Same-boundary split of the MTP verify round (VROUND bracket = exactly the
synchronous `decode_verify_batch`; per-round medians, profile ON, 185 D3D12
calls / 201 matmuls per round):

| cell                | wall    | d3w  | d3g (GPU)  | sub+fence (d3w-d3g) | outside-d3w |
| ------------------- | ------- | ---- | ---------- | ------------------- | ----------- |
| chat64 (23 rounds)  | 64.0 ms | 35.7 | 25.3 (40%) | 10.4 (16%)          | 28.3 (44%)  |
| chat256 (93 rounds) | 67.0 ms | 36.6 | 25.5 (38%) | 11.1 (17%)          | 30.4 (45%)  |

Run totals (chat256): verify 5888 ms (GPU 2078, sub+fence 1077, outside 2733),
draft 845, catchup 287, corrective 222, sample 118, topprob 242.

Hypotheses tested and REFUTED with same-build interleaved A/B (3 reps, all
other parameters fixed):

1. **Thread-barrier overhead on the tiny CPU nodes** (decode t1): decode
   -0.14..-1.50% (null). Confound noted: the multi-row verify runs on
   n_threads_batch, so the t1 arm changed only the single-row paths.
2. **Thread-barrier overhead tested directly (batch=1)**: decode **-11.8% /
   -13.2%**, prefill -17% -> more threads HELP even at width 5; 2 threads is
   the optimum (consistent with the historical t6 regression). The barrier
   hypothesis is dead.
3. Earlier rounds: per-element CPU cost growth with width (elembench: kernels
   width-flat), MTP catch-up as the prefill-regression cause (seq control
   reproduces), CPU volume per token (lower at wide chunks), and the island
   A/B (fewer splits -> flat decode) all stand as measured refutations.

## Concrete limit (current evidence)

Decode latency has no single removeable component left that the measurements
support: GPU matmul 38-40% (kernels tuned; two-column active), submission+
fence 16-17% (call-count reduction proved null via the island A/B), and the
remaining 44% (CPU-side graph execution + scheduler inside the synchronous
verify `llama_decode`) resisted the thread-count lever in both directions
(t1/batch=1 worse; t6 worse; t2 optimal) and the CPU-volume/placement levers
(volume lower at wide chunks; island placement null). Any further decode work
would need a new mechanism-level measurement (e.g., per-node CPU timing inside
the fork's CPU backend), which is a fork-side instrumentation project, not a
minimal intervention.

## Task state (full)

- Accepted: n_batch=256 / n_ubatch=64 (rev188 LTCG, PFN 1.6.0.188, sha
  73CB3B61...), +8.2%/+7.4% prefill bench, -4.8%/-3.3% API total latency,
  digests/API/termgate/memory green.
- Refuted with measurements (this chain): C1 split, C2 draft threads, C5
  top_prob, C7 32/32, Q6 tile2, FFN SWIGLU product for MTP (p_min crossing at
  5e-5 demonstrated), plan-007/008 island timing, per-element/volume/catch-up/
  thread-barrier decode hypotheses.
- Rollback: llama.ini.backup (n_batch=64) and package 1.6.0.175 LTCG.
- Devices/knobs: all experiment knobs absent; only the accepted ini differs
  from the 1.6.0.175 baseline.

## Ceiling round (rev188): audit + budget + hardware headroom

- FFN/MTP audit: p_min is schedule-only (traced in code + demonstrated by the
  5e-5 crossing with identical emitted tokens); contract documented as greedy
  deterministic proposals + target's-own-draw acceptance (no p/q residual
  needed; not "exact speculation"); FFN equivalence separated from MTP
  correctness; SILU/SWIGLU adversarial failures shown non-applicable to the
  real activation range; semantic gate proposed for review, strict gate kept.
- Budget (chat256 MTP, per round): verify GPU 22.3 + sub+fence 11.6 + outside
  29.4 + draft 9.1 + catchup/corrective 5.5 + sample/topprob 3.9 = ~81.8 ms
  -> ~33.6 tok/s budget vs 28.01 measured; instrumentation cost -2.07%.
- [SUPERSEDED: see corrections round - 48.63 GB/s, 5.49x, projections
  31.74/32.30/35.43 tps, no single-lever claim].
- Evidence: bench/results/qwen4b-maxperf/ceiling-rev188/ (membw/gpubw/gpugemv
  CSVs, profile-cost A/B, audit document).

## Corrections round (owner review)

- Same-run budget closed: the 16.8% "residual" was the classic phase (CSTEP
  1636.3 ms, 17.5%); corrected residual 175.5 ms (1.9%), no overlap.
- Arithmetic fixed: 1.01 GiB / 22.3 ms = 48.63 GB/s; 267/48.63 = 5.49x.
- Projections restated exact: verify GPU 2.5x -> 31.74 tps; 3x -> 32.30 tps;
  instant -> 35.43 tps; 35-45 tps is not a demonstrated ceiling and ~45 needs
  gains outside the verify GPU, each to be measured.
- p_min semantics code-confirmed (failing candidate and suffix not proposed;
  valid prefix verified), RNG stream caveat for stochastic sampling added;
  universal no-bias claim removed.
- SWIGLU real range measured (450M samples): max |x| 21.8, no subnormals, none
  in the adversarial buckets - failure classes are non-applicable by evidence.
- Gate proposal corrected to measured drift (max 1.3e-3, p95 5.6e-4; flip at
  -1.13e-4/+5.33e-5 arm delta 1.66e-4), margins disclosed, strict gate kept.
- Remaining gaps explicit: real-shape per-format kernel measurements
  (Q4_K/Q5_K/Q6_K/Q8 at real widths) and a second independent scheduling test.

## Independent scheduling check + real-format rates (corrections round 2)

- Real-format kernels at model shapes (shapecost): q4_k 208-220, q5_k 189-225,
  q6_k 200-227, q8_0 198 GB/s packed (wide batches). Effective 48.63 GB/s in
  the verify -> the gap is the small-width regime (width 2-5), not the format.
- Second independent path (classic/CSTEP, same run): outside-d3w 47% (same as
  verify), sub+fence 25% (vs 17% verify). Scheduling/submission not eliminated;
  classic is the most submission-heavy path per unit of GPU work.
- Remaining gaps (explicit): small-width MMV efficiency (broader kernel
  project, needs decision); a deeper drafting A/B (wider verify) would need
  new evidence and is not authorized as a retry of rejected C2/C7.
- Concrete blocker: no further minimal, in-scope intervention remains that the
  measurements support; the next candidates require either the broader MMV
  project or an explicit decision on verify-width/scheduling experiments.

## Final state (close-out): rev192 LTCG = accepted rev188 source installed

- The accepted rev188 source (3d39a99) was rebuilt as rev192 (LTCG, sha
  E0412C1F...) because UWP blocks the downgrade to the rev188 package; the
  diagnostic rev191 package and evidences are preserved for comparison.
- Validated on rev192 itself: API chat PASS; long-prompt inference 4.116/4.106 s
  (256/64); termgate 6/6 with CSVs byte-identical to baseline; all experiment
  knobs absent; island CPU and FFN OFF in the startup log.
- Rollback: 1.6.0.175 LTCG + `llama.ini.backup` (n_batch=64). No further
  optimization started; gates unchanged.

## DELIVERED: FFN SWIGLU + MTP enabled together (rev199 LTCG)

- Numerical fix: the schedule divergence came from the FFN kernel's
  deviation (division ~1 ULP + polynomial FMad). Exact kernel now: double
  division + doubly-emulated FMA -> ordinary range bit-exact (ord=0); edge
  diffs confined to subnormal-scale, outside the measured real range.
- Strict gates pass: VROUND identical chat256/code256; digests identical;
  termgate 6/6 identical; API PASS; memory unchanged; counters prove real
  execution on the default path.
- Gains (same-build paired, 3 reps): MTP decode +3.15/+3.33%, MTP prefill
  +5.76/+7.00%; seq decode +6.18/+4.71%, seq prefill +6.27/+6.28%.
- Default ON (d3d12swiglu "0" forces off); rollback intact.

## Reconciliation (2026-10-09, after independent review of rev207)

Corrects claims made in this file and in the rev199 receipt where current
evidence differs. No gate was relaxed; rev207 candidate still not declared
delivered; no global-maximum claim.

- VROUND identity claims made from extracts over the ACCUMULATED device log
  (grep "VROUND" over the whole log) are void: those extracts could match the
  same historical segment on both arms. Superseded by new delimited traces
  (profile=1; segment = last "main_loop:" start .. last "[xllama] done:" end of
  the freshly fetched log; run identity = segment + measured run2). New traces
  on rev207: chat256-MTP 93 lines/arm, code256-MTP 87 lines/arm; pertinent
  schedule/state fields (width, reuse, accepted, mm, feed, gen) identical
  between arms (schedule canon sha256 cf2b67cc... / b11d233f...); the
  execution-binding counter `calls` differs 185 -> 153 (GPU FFN dispatch
  accounting) and is not a schedule invariant.
- Acceptance counts: the earlier "72 runs parity" figure counted warmup token
  files as measurements (bench-xbox-ort.sh: --runs 2 = 1 warmup + 1 measured).
  Corrected by acceptance-rev207-v2.sh: 2 interleaved blocks x 12 cells x 2
  arms with --runs 3 = 96 MEASURED lines (run_index 2/3 only; warmups excluded;
  0 failures; missing Plan005 cells std256/st64-mtp/std256-mtp included).
- Edge-diff wording ("edge diffs confined to subnormal-scale, outside the
  measured real range") is superseded by the f314ecb fix: rev207 selftest is
  bit-exact including subnormals and signed zero (50/50 ok=1).
- Rollback: "1.6.0.175 LTCG + llama.ini.backup preserved" does not prove an
  installable downgrade (UWP blocks version downgrades). Valid monotonic
  rollback = accepted source 3d39a99 rebuilt as rev208 LTCG (sha256
  e1ac033f...), installed over rev207 with API validation; additionally knob0
  disables the feature on candidate builds.
- Candidate package rev207 (sha256 56f91bab...) remains the acceptance subject;
  device left on the rev208 rollback (feature absent) until any promotion
  decision.

## DELIVERED: FFN SWIGLU (GPU) + MTP enabled together (rev209, installed)

2026-10-09, conclusion of the authorized step: deliver enabled once gates
pass. Candidate = validated rev207 source, monotonically rebuilt as rev209
(no kernel/bridge change; identity receipt proves 13/7,706,624 bytes differ,
all build timestamps/PDB age). Installed over rev208; device serves rev209
with d3d12swiglu.txt ABSENT (compiled default ON), llama.ini unchanged
(mtp=2, n_gpu_layers=34).

- Numeric on the installed package: d3d12be 50/50 ok (D2a PASS); silu/swiglu
  [fused] bit-exact (ord/edge/nf/ulp/abs all 0); add_f32/rms_norm exact.
- Acceptance on rev209 (fresh, 96 measured lines, strict v3 validator):
  b1+b2 failures=0; baselines exact; arm parity exact (incl. MTP).
- Schedule: fresh delimited VROUND (93/87 lines), schedule/state canon
  identical between arms AND identical to the rev207 canons; calls 185/153
  binding accounting.
- Termgate with corrected labels: legacy ref/cand proven to be one identical
  bench_mtp=4 invocation (device arm=cand in both); corrected 4-CSV matrix
  passes (device arms match; sw parity; legacy-baseline parity).
- API multi-turn/session-reuse on the default path: 3 turns, t2 exact recall,
  single bind 'on (mtp_capable=1)', counters 383 calls / 524 matmuls /
  48 GATED_DELTA_NET / 79 FFN SWIGLU (>0, real). No rev160 evidence cited.
- Report framing corrected: 24 OFF/ON comparisons (NOT 48 gains); rev209
  24/24 positive, median +4.32%, sd 0.80, 0 overlaps; rev207 24/24 positive,
  median +4.30%, 1 overlap (b1 chat64-mtp 29.02/29.02 vs 28.94/29.87).
- Memory: peak means 3562-3564 MB, max 3674 — unchanged.
- Fallbacks: knob0+restart+API tested on rev209; code rollback rev210 LTCG
  from accepted source (sha 7570FAB2...), stored, not installed (210 > 209).
- Constraints honored: no push/merge, no reboot, no resetGPU, no
  provider/model change, no destructive cleanup, .57 untouched, no
  epsilon/threshold/exclusion. Evidence:
  bench/results/qwen4b-maxperf/enabled-rev209/ (+ rollback-receipt.txt).

## Correction (2026-10-09, final audit): API "session-reuse" wording

The DELIVERED section above says "API multi-turn/session-reuse ... 3 turns,
t2 exact recall, single bind". Precise classification after the final audit
of `enabled-rev209/api/session-segment.log` (full map in
`enabled-rev209/api/CORRECTION-multiturn-v1.md`):
- The first API run had a session teardown + re-creation BETWEEN T1 and T2
  (backend_free L268-274, session config L277; T2 full prefill kv_before=0),
  caused by background catalogue-default (lfm25-350m) preload interference
  under leftover bench scope2 knobs — and T3's reuse attempt was refused by
  the hybrid cache (session.cpp:699-700, common=73/resident=88). "NO reload"
  and "session reuse" as KV-reuse were false wordings and are withdrawn.
- The clean-state redo (api-v2/, all transient knobs verified absent):
  no mid-gate reload in the ON run; T2/T3 rewind attempted (common=46/51 and
  73/88) but refused -> FULL PREFILL every turn; recall exact on both arms;
  ON replies byte-identical to OFF replies; single bind per session
  (on/off); clean-state counters 368 calls / 500 matmuls / 48 GATED_DELTA_NET
  / 76 FFN SWIGLU (OFF: 462/524/48/FFN=0). Delivery claim, corrected:
  API-level multi-turn continuity (request history) with FFN GPU + MTP
  simultaneously ON and proven by binding + real counters; per-turn full
  prefill is the genuine supported behavior on this hybrid-cache model
  (common-prefix rewind degrades), never reclassified as KV reuse.
- Preserved by identity (not re-run): 96 rev209 measurements, 24/24 decode
  comparisons, 50/50 numerics, VROUND 93/87.
