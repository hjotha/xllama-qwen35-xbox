# Plan 006 — Qwen 4B Xbox max-perf: ranked candidate analysis

> **Historical ranking, superseded (2026-10-08).** The ranked list below was
> the working analysis during the optimization cycle. The final deliverable is
> rev160; see `bench/results/qwen4b-maxperf/FINAL-REPORT.md` for the final
> candidate dispositions and evidence. The bounded cycle is complete; nothing
> here is pending implementation.

The same-boundary DSPLIT/CSTEP/DSTEP instrumentation first
shipped in rev129 and was superseded by the rev130 measurement-integrity fix
before any device run; the corrected rev130 build was measured by the
independent reviewer (`rev130-reviewed-dsplit/`, parity OK): chat64 OFF seq
23.81 / MTP 28.22 tok/s; DSPLIT seq d3w ≈54% / d3g 29%, MTP d3w ≈53% /
d3g 34-35%. C8 (timestamp gating) measured +3.15% seq / +2.60% MTP decode
(chat64); its profile-bound default has since shipped (final: enabled).

Evidence classes used throughout:

- **Measured** — has a console artifact (CSV/log) behind it; pointers given.
- **Static** — derived from reading the pinned source; no device number.

All changes stay host-owned (`include/`, `src/bridge/`, `uwp/`) or maintained
`patches/*.patch`; the submodule is never committed to. Timing comparisons are
profile OFF; profile ON only for attribution, with the new instrumentation's
own overhead measured first. Every candidate's validation includes full-ID
sha256 parity per run and interleaved opposite-order repeated blocks.

## Measured baseline costs (inputs to the ranking)

| item                                                           | value                                                               | source                                                   |
| -------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------- |
| MTP decode split (chat64)                                      | verify 64.5%, classic 20.3%, draft 8.5%, catchup 3.1%, topprob 2.4% | `baseline-rev126/` MTP_STATS + `README.md`               |
| verify sub-split (nested: d3g ⊆ d3w ⊆ wall)                    | d3g 21.5%, d3w 36.5%, CPU+prep 28.0% of decode                      | same                                                     |
| seq decode                                                     | 42.5 ms/token (chat64)                                              | `baseline-rev126/csv`                                    |
| prefill                                                        | seq ~95 t/s; MTP ~87 t/s (−7-8%)                                    | same                                                     |
| prefill thread scaling (t2 vs t6, old profile, chat119 MTP)    | t6 prefill +6.5% (92.5 vs 86.8 t/s), decode −12% (24.9 vs 28.2)     | `bench/results/fast-mtp-20261007/r120-win-t{2,6}.csv`    |
| D3D12 submission cadence                                       | one synchronous command-list + fence span per scheduler split       | `src/bridge/ggml_d3d12.cpp:1179`                         |
| GPU timestamp queries on every graph, profile OFF included     | 2×EndQuery + ResolveQueryData + Map/Unmap per graph_compute         | `src/bridge/ggml_d3d12.cpp:1476`, `1613-1634`            |
| per-call whole-run D3D12 overhead (aggregate, not decode-only) | ~65 µs/graph_compute call, GPU busy 67-86% of D3D12 wall            | `baseline-rev126` logs; aggregate only, not attributable |
| cross-backend split-input copies                               | zero observed                                                       | plan004 rev102 (`GGML_BACKEND_COPY_PROFILE`: 0 lines)    |
| graph rebuild vs reuse                                         | 78.6% rebuild, marginal reuse cost ~+2% when disabled               | plan004 rev102                                           |

## Measurement corrections (applied to the ranking)

- **Process-CPU is not kernel time.** `run_now` waits with `spin=true,
spin_us=-1` (unbounded `YieldProcessor`) by default
  (`src/bridge/ggml_d3d12.cpp:1179-1186`,
  `src/bridge/d3d12_compute.cpp:158-183`), so process CPU includes polling.
  `cpu ≈ 2× wall` must not be read as "two cores of useful CPU work"; active
  kernels need a separate attribution. Bounded-spin/event-wait stays a
  measured hypothesis: rev101 measured it (decode neutral, TTFT worse,
  rejected) — re-testing requires new evidence, not the CPU ratio.
- **D3D12_Host is mapped write-back/shared memory**, not a discrete GPU
  upload ring (`src/bridge/llama_gpu.h:25-33`, `ggml_d3d12.cpp:984`). A
  backend boundary here means a fence/scheduler boundary, not a PCIe-style
  copy. The zero copy-profile result (rev102) is consistent with that; it is
  not proof that no memcpy exists anywhere. Candidates must distinguish real
  memcpy/weight readback from fence/scheduler boundaries.
- **`d3d12_wall_ms` includes the timestamp readback** (Map/Unmap happens
  before the wall clock is sampled, `ggml_d3d12.cpp:1634-1643`), so every
  d3w number to date carries that overhead — which is exactly what C8
  removes.

Measured by rev130 (independent reviewer run): whole-decode DSPLIT d3w/d3g/CPU
window deltas and per-step splits. Still open: per-step log overhead under the
new instrumentation, and the same numbers for code/longer-chat grids.

## Ranked candidates

| #   | scope | lever                                                       | class    | expected size (bounded by arithmetic)                                  | effort / risk      |
| --- | ----- | ----------------------------------------------------------- | -------- | ---------------------------------------------------------------------- | ------------------ |
| C8  | both  | disable GPU timestamp queries when profiling is OFF         | measured | **+1.6-3.2% decode (chat64/code/256); profile-bound default shipped**  | trivial / very low |
| C1  | both  | decouple `n_threads_batch` from `n_threads`                 | measured | **REJECTED: MTP −9.5%/−21.6% at tb4/tb6; retain 2**                    | low / low          |
| C2  | MTP   | draft CPU threads knob 1→2                                  | measured | **no robust gain (+0.7%, overlapping); retain 1**                      | low / low          |
| C3  | both  | deferred D3D12 wait (dependency-audited, not blanket async) | static   | up to the CPU/GPU serialization overlap; ceiling not establishable yet | high / high        |
| C4  | both  | cheap ops (RMS_NORM/ROPE) on D3D12 to merge splits          | static   | split-count × ~65 µs; count not yet per-decode-measured                | high / high        |
| C5  | MTP   | `top_prob` scan optimization                                | measured | ≤2.4% of decode                                                        | low / low          |
| C6  | MTP   | eliminate/batch catch-up replays (private nextn KV)         | static   | catchup 3.1% + part of draft decode count                              | high / high (fork) |
| C7  | both  | prefill ubatch 32 (negative control)                        | measured | **REJECTED (joint 32/32): prefill −6.6%; retain 64/64**                | trivial / none     |

\* C8's _overhead_ is statically certain (queries run regardless of
profile_phases); its _recoverable size_ is not yet measured.

### C8 — Disable GPU timestamp queries when profiling is OFF (likely first A/B)

- **Static basis, certain.** `backend_graph_compute` takes the timestamp path
  whenever the query heap exists (`const bool ts = g.ts && g.ts_rb;`,
  `src/bridge/ggml_d3d12.cpp:1476`), and the heap is created on any device
  that reports a timestamp frequency (`:1139-1144`). So every graph_compute —
  prefill, classic, verify, draft, catch-up, in product runs too — issues
  `EndQuery(0)`, `EndQuery(1)`, `ResolveQueryData`, then `Map`/`Unmap` on a
  READBACK buffer (`:1481`, `:1613-1634`). `d3d12_wall_ms` is sampled only
  after that Map (`:1634-1643`), so the release overhead is inside every d3w
  number measured so far.
- **Mechanism.** Removing the query + resolve + readback removes a GPU command
  pair, a driver readback path and a Map/Unmap per call, with no effect on
  compute results (queries are read-only). Counters stay honest by reporting
  GPU timing as disabled rather than 0 when the path is off.
- **Counter correctness (required in the same change).** `g.gpu_ms +=
g.last_gpu_ms` (`ggml_d3d12.cpp:1632`) accumulates unconditionally while
  `g.last_gpu_ms` is only assigned on a successful Map (`:1634-1643`). A
  disabled or failed sample would keep re-adding the previous graph's timing.
  The change must: reset `last_gpu_ms` per call before the query path;
  accumulate only a sample captured **for this call**; and count unavailable
  samples separately (`n_ts_unavailable`), so printers show GPU timing as
  `disabled`/`unavailable` — never a measured zero. `d3d12_gpu_ms()` keeps the
  "sum of valid samples" meaning and availability is exposed alongside it.
- **Implementation.** Host-only: `d3d12_set_gpu_timestamps(bool)` +
  `d3d12timestamps.txt` knob at the existing apply-knob sites; default
  unchanged (= today's always-on) until the A/B; attribution runs
  (profile ON) keep timestamps ON so d3g remains available.
- **Expected size.** 6k-48k calls per run (measured) × per-call cost. Even
  10-20 µs/call is 60-960 ms/run; the real cost must be measured, not assumed.
- **Measured result (rev131 `c8-timestamps-rev131/`).** Profile OFF, chat64,
  ON→OFF→ON plus reverse-order blocks, all 20 dumps full-ID exact:
  seq medians 23.80 → 24.55 tok/s (**+3.15%**), MTP 28.28 → 29.02 (**+2.60%**).
  Engagement verified (`gpu timestamps=on/off`); ON summaries `ts_valid=7423`,
  `ts_unavailable=0`; OFF summaries `timing unavailable: N calls` (never a
  measured zero). Knob retained; default-ON baseline preserved. **Proposed
  next default (needs review):** timestamps enabled iff profiling is ON, with
  `d3d12timestamps.txt` as an explicit override; attribution runbooks then
  upload `1` explicitly.
- **Validation.** Rev129 + knob: timestamps ON vs OFF, profile OFF both arms,
  chat64 seq+MTP and std-512 seq+MTP, interleaved opposite-order blocks;
  full-ID sha256 parity; decode/TTFT/whole-call; profile-ON pair with ts ON to
  keep attribution valid. Counter tests: (a) a host-testable pure helper for
  the per-call sample/invalid/disabled state, proven in doctest for valid,
  invalid, disabled and Map-failure sequences (no stale carry-over); (b) a
  device ON→OFF→ON sequence in one package showing each segment's counters are
  independent, re-enable resumes sane increments, and `n_ts_unavailable`
  counts exactly the OFF calls. Forcing a real Map failure on device is not
  safe/feasible; that case is covered by the host helper and stated as such.
  If the measured gain holds, default product OFF with profiling tied to
  profile ON.

### C1 — Decouple prefill/batch threads from decode threads (top candidate)

- **Measured basis.** Prefill is 36-45% of whole-call at 64 output tokens
  (`std512` TTFT 3.15-3.48 s). With all threads at 6, prefill was 6.5% faster
  and decode 12% slower (r120 CSVs above). So one thread count cannot be
  optimal for both phases.
- **Static mechanism.** In the pinned fork, `llama_context::graph_compute`
  chooses threads per graph: `batched ? n_threads_batch : n_threads`
  (`llama.cpp/src/llama-context.cpp:4338`), and `batched` is exactly
  `ubatch.n_tokens > 1` (`llama.cpp/src/llama-context.cpp:3069`). Our adapter
  pins `cparams.n_threads_batch = n_threads`
  (`src/bridge/inference.cpp:475-478`, `src/bridge/session.cpp:493-502`), so
  prefill chunks, verify batches (B=2-4) and catch-up batches all run on the
  slow decode thread count, while classic single-token steps run on
  `n_threads`. The persistent pool wrapper already creates a second pool when
  the two differ (`src/bridge/llama_gpu.h:84-110`,
  `src/bridge/inference.cpp:519-520`).
- **Expected size, honestly bounded.** Not a new speedup: it is harvesting the
  already-measured t6 prefill delta while keeping the t2 decode. If prefill
  gains ~6% on a 3.15 s TTFT that is ~190 ms; for chat64 whole-call (~3.7 s)
  ≈ +2-3% end-to-end. The open question — does a 6-thread verify batch also
  help or hurt? — is exactly what the A/B measures; t6/t6 was slower overall,
  but that conflates classic and batch parts.
- **Implementation.** Add `n_threads_batch` to `InferenceParams` /
  `SessionParams` (host), a bench knob (`bench_batch_threads.txt`) and the
  `llama.ini` key for the session path; default unchanged (equal to
  `n_threads`) until measured.
- **Validation.** Arms t2/2 (baseline), t2/4, t2/6, t6/6 (control) on chat64
  seq+MTP and std-512 seq+MTP; interleaved 2 blocks opposite order, profile
  OFF; TTFT, prefill t/s, decode t/s, whole-call; full-ID sha256 per run;
  verify DSPLIT d3w changes (profile ON, overhead paired separately).

### C2 — Draft CPU threads (MTP only)

- **Measured basis.** Draft = 8.5% of MTP decode; the draft context is capped
  to 1 thread by `XLLAMA_MTP_THREADS` (`src/bridge/mtp_draft.cpp:110-115`),
  including its catch-up batches (`n_threads_batch = n_threads`, same block).
- **Static mechanism.** Two draft decodes per round are sequential (depth 2);
  more threads can only speed each draft graph's CPU parts. Contention with
  the target's classic/verify work on the 4-core CPU is the risk.
- **Expected size.** Bounded by draft's share × partial speedup (≤ ~4%).
- **Validation.** Knob `mtp_threads.txt` = 1 vs 2 at fixed t2/t2; same
  interleaved block protocol; full-ID parity; also record DSPLIT draft DSTEPs.

### C3 — Deferred D3D12 wait, dependency-audited (highest ceiling, highest risk)

- **Measured basis.** Every D3D12 graph compute is synchronous: submit then
  fence-wait inside `run_now` (`src/bridge/ggml_d3d12.cpp:1179-1186`,
  `src/bridge/d3d12_compute.cpp:158-183`). Verify rounds show d3g 21.5% of
  decode vs d3w 36.5%; the GPU is idle while CPU splits run. rev101 bounded
  spin changed nothing (rejected), so the lever is not the wait policy.
- **Static hypothesis.** If D3D12 submissions did not block until completion
  (queue depth ≥ 2 with waits only before data consumers), CPU splits and GPU
  execution could overlap. Whether this is safe depends entirely on buffer
  ownership: the D3D12_Host shared buffer means the CPU could read/write
  tensors a pending GPU dispatch still owns.
- **Explicit limits.** This is NOT a blanket async/barrier removal; it needs a
  written per-tensor dependency audit and is the one candidate that could
  silently corrupt state. It stays behind the rev129 DSPLIT evidence and a
  design review before any code. Effort high; risk high.

### C4 — Merge CPU/GPU boundaries by supporting cheap ops on D3D12

- **Static basis.** The backend accepts only MUL_MAT, GATED_DELTA_NET and
  metadata ops (`src/bridge/ggml_d3d12.cpp:1452-1465`, `1739-1749`); every
  norm/rope/softmax splits the graph and adds a synchronous submission. A
  GDN kernel precedent exists (`shaders/ggml_d3d12_gated_delta_net.hlsl`).
- **Corrected placement census (rev135 `sched2-rev135/`, complete `## SPLIT`
  text boundaries; the earlier CONCAT-alone claim is retracted).** Per big
  graph (370 splits): 40× [ADD, RMS_NORM, MUL]; 32× singleton [SWIGLU];
  24× the long GDN chain [CONCAT, CPY×3, SCALE, GET_ROWS×2, CPY, SSM_CONV,
  SILU, RMS_NORM, SCALE, RMS_NORM, SCALE, small MUL_MAT, ADD, SOFTPLUS, MUL,
  MUL_MAT, SIGMOID]; 24× singleton [CPY]; 24× [RMS_NORM, MUL, SWIGLU];
  23× [ADD, RMS_NORM, MUL, SCALE, GET_ROWS×2, CPY]; 8× [RMS_NORM, MUL, ROPE];
  8× longer attention chains.
- **Singleton CPY contract audit: does NOT hold for a backend-only change.**
  Raw split bodies show the CPY node itself CPU with dst `cache_s_l*` CPU and
  src D3D12 (e.g. split326, and splits 16/28/50/62/74/96): the recurrent-state
  destination is CPU-resident, so a D3D12 copy would require a state-cache
  placement change (llama memory code, not our backend), and the graph
  comments document same-buffer read-before-write overlap hazards
  (`llama-graph.cpp:5086/5167` copies gather→view of the same cache).
  Tight whitelisting (same type, both D3D12, contiguous, non-overlapping)
  would simply refuse, i.e. no gain. Retracted as the first candidate.
- **Remaining viable singleton island: FFN SWIGLU (32/graph).** Its gate/up
  inputs and down-matmul output are D3D12; only the numerics block it.
- **SWIGLU device corpus result (rev147/rev150): probe not bit-exact; ordinary
  drift is 1-3 ULP and bounded; full-model gates still mandatory.**
  rev147 showed the implemented probe is not bit-exact (6999/36864 ordinary
  mismatches, 1-ULP first diffs, MXCSR 0x1fbb) — an implemented-probe failure
  only, not proof about all ports and not an FMad-vs-division diagnosis.
  rev150 extended the probe: ordinary-domain ULP histogram SILU
  **1/2/3+ = 6954/45/0**, ordinary max |diff| 9.5e-07; split-SWIGLU with
  randomized up **1/2/3+ = 5283/807/5**; **nf=0 on both** (no NaN/Inf drift);
  edge counts 80/180. Absolute SWIGLU differences scale with the adversarial
  3.4e38 operands; the ULP/relative error stays at 1-3 ULP. Evidence:
  `bench/results/qwen4b-maxperf/silu-selftest-rev150-d3d12be.csv` +
  `swiglu-probe-rev150-verdict.md`.
- **Gating decision (per review):** ordinary errors are small and bounded with
  no NaN/Inf drift, so the isolated contiguous GPU-resident FFN SWIGLU was
  wired behind a default-OFF product knob (`d3d12swiglu.txt`) with a narrow
  FFN lineage whitelist (`ffn_swiglu`/`ffn_gate`/`ffn_up` prefixes, `ne0=9216`,
  all operand dims matching the output, owned host/UAV buffers, PSO + gate).
- **rev154 product-trial gates: PASSED.**
  - Full-ID parity: exact (sw0 vs sw1, chat64/code64 × seq/MTP, runs 2/3).
  - Per-round proposal/acceptance parity: VROUND `feed=` sequences and
    `accepted=` counts identical OFF vs ON (23 chat64 rounds, 21 code64).
  - Engagement (last-segment slicing): OFF `knob=off dispatches=0`; ON
    `knob=on` with 1041-2240 dispatches; counter labeled by actual placement.
  - Timing (profile OFF, 2 rows/arm): chat64-MTP **+4.42%**, code64-MTP
    **+3.61%**, chat64-seq **+6.53%**, code64-seq **+5.65%**, all rows
    separated; the rev153 exploratory run (before the lineage whitelist)
    showed +4.5-5.7% on the same workloads, consistent across two packages.
  - Evidence: `bench/results/qwen4b-maxperf/swiglu-gates-rev154/`
    (`gates-rev154-analysis.txt`, `verdict.md`); the rev153 receipt's
    cumulative engagement lines are marked invalid (corrected slicing in
    `swiglu-gates-rev153/engagement-corrected.txt`).
- **Promotion path (not yet done):** knob stays default OFF until the expanded
  repetition grid + full acceptance (code/256/session/termination/API) + the
  final optimized build pass. This is now the strongest measured candidate
  alongside the shipped C8 profile-bound default.
- **Extension gates (rev154, `swiglu-ext-rev154/extension-verdict.md`).**
  - 256-token full-ID parity exact (chat256 `f78ce837`, code256 `c7d84255`,
    both arms, runs 2/3).
  - Per-round traces: code256 (87 rounds) identical OFF vs ON; **chat256
    diverges at round 49** (sw0 93 rounds vs sw1 94; stable per arm) while the
    final full IDs stay exact — the draft context's FFN SWIGLU also runs on
    D3D12, shifting proposals by ULPs. Strict per-round gate FAILS for this
    cell; canonical full-ID gate passes.
  - Session scenarios: 4/4 parity in both arms, scenario dumps byte-identical
    (metadata files excepted).
  - Clean interleaved timing: chat256 **+5.22%**, code256 **+4.30%**
    (a first chat256 OFF cell contaminated by a 492 MB leftover `sched2`
    device log showed +23% and was discarded).
  - Next: target-context-only restriction (draft FFN SWIGLU back to CPU) or
    reviewer's acceptance of the full-ID-only criterion; then
    termination/API gates and the final optimized build.
- **Target-only discriminator result (rev155,
  `swiglu-targetonly-rev155/targetonly-verdict.md`): the divergence is NOT the
  draft context.** With the verified suffix<n_layer discriminator (target
  layers 0..31 vs nextn layer 32; mode `2` = target-only, 4352 dispatches vs
  4743 for all-FFN, off = 0), the chat256 per-round feed sequence still
  diverges from OFF at round 49 exactly as all-FFN does, while full IDs stay
  exact (`f78ce837` in all three arms). Conclusion: target-FFN arithmetic
  drift feeds the drafter; excluding the nextn FFN changes nothing.
  **Decision per the review rule: record and keep the MTP SWIGLU candidate
  OFF** (no endless near-identical builds).
- **Sequential-only FFN SWIGLU remains independently viable** (no drafter, so
  no proposal-parity gate): full-ID parity already exact for chat64/code64/
  chat256 seq with the knob ON. It proceeds through its own API/cancel/
  termination gates and the final optimized build, default OFF until then.
- **Process-profile enforcement (rev159, `seqprofile-rev159/seqprofile-verdict.md`).**
  The backend binds the capability at the FIRST context and never changes it
  while the process lives (`SwigluModePolicy`: bind-once; a later conflicting
  MTP mode is rejected BEFORE context creation via
  `d3d12_swiglu_profile_accepts` -> "restart required"; the conflict check
  precedes any backend-state mutation; once bound ON a later knob=0 cannot
  open the gate). Device verification with knob=1: MTP-configured process
  logs `profile bound: off (mtp_capable=1)` with zero dispatches; seq process
  logs `bound: on (mtp_capable=0)` with 2112 dispatches; token parity
  `84e08196…` both. Host latch tests 16/16, full ctest green. **No MTP FFN
  candidate** (per the review: MTP keeps FFN OFF; C8 MTP gains retained);
  the sequential candidate stays experimental until the API/session/cancel/
  stop/EOS gates and the final optimized build pass.
- **Risk.** Numeric emulation drift (dxc contraction), buffer placement
  (D3D12_Host shared activations), split-planner interaction; the written
  buffer/dependency audit precedes any fence/async discussion.

### C5 — `top_prob` scan optimization (MTP only)

- **Measured basis.** 55.6 ms/run = 2.4% of decode; the function already has
  an exact early-exit on the sum (`src/bridge/mtp_draft.cpp:42-64`), but still
  scans the full 248k-vocab logits for the max and, for accepted candidates,
  sums until the limit.
- **Numerical constraint (from review).** `top_prob` drives the p_min gate; a
  different summation order is NOT automatically exact — FP reassociation can
  flip accept/reject decisions near the threshold even with identical logits.
  Therefore: only the max scan may be vectorized (max is order-independent for
  finite values, and NaN/±inf handling must match the current semantics
  exactly), while the exp accumulation keeps strict ascending-index scalar
  order (the existing early-exit already preserves it). Any variant that
  reassociates the sum is rejected unless it is provably order-preserving.
  Caching a per-round logsumexp is not trivially valid (rows differ by depth)
  and is not proposed.
- **Validation beyond final-token parity.** Adversarial near-threshold cases:
  logits engineered so the running sum is within a few ULPs of `1/p_min`
  (including exact ties) and non-finite inputs, asserting identical
  accept/reject decisions and identical proposal sets per round across arms —
  final full-ID sha256 parity remains required on top.
- **Expected size.** ≤2.4% ceiling, partial recovery; low risk only if the
  order-preservation constraint holds.

### C6 — Catch-up replays / private nextn KV (MTP only)

- **Measured basis.** catchup ≈ 3.1% of decode (70.7 ms/run), 61 replayed
  tokens/run; `decodes=86` per run includes these
  (`baseline-rev126` MTP_STATS).
- **Static basis.** For `LLAMA_CONTEXT_TYPE_MTP` the fork filters the KV to
  the nextn layers (`llama.cpp/src/llama-model.cpp:3399-3401`), i.e. the draft
  keeps a private nextn KV that the catch-up rebuilds; `ctx_other` is set
  (`src/bridge/mtp_draft.cpp:99-115`) but the shared-cache branch exists only
  for Gemma4Assistant (`llama.cpp/src/llama-model.cpp:3419-3436`).
- **Limits.** Eliminating replays needs fork-level shared/cached nextn rows —
  outside the maintained-patch budget without a clear upstream design. A
  bounded host-side variant (defer and batch replays before the next draft)
  is possible in principle but has no measured slack: classic tokens already
  replay one at a time because the next draft needs the row; batching would
  only merge consecutive replay opportunities that rarely accumulate.
- **Verdict.** Record as a bounded ceiling/blocker entry, not a first wave
  candidate.

### C7 — Prefill ubatch 32 (negative control)

- **Static.** ubatch 64 beat 128/256; smaller chunks add submission overhead
  and reduce matmul efficiency. Cheap knob-only check to close the curve.

## Measured-closed (do not re-litigate without new evidence)

- decode threads t6/t3 vs t2 (paired, r120 + rev100);
- bounded spin wait (rev101);
- graph reuse (rev102);
- `offload_kqv` re-test: attention operands are f32 and the D3D12 backend
  refuses f32 matmuls (`MUL_MAT refused: f32 …` in device logs; supported-type
  filter `src/bridge/ggml_d3d12.cpp:1749+`), so placement is structurally
  inert without a new kernel;
- Q6 tile 2 (rev127, −1.7%/−2.0%); ubatch 128/256 (−13%/−29%);
- batch threads tb4/tb6 at decode 2 (rev132: MTP −9.5%/−21.6%; retain 2);
- draft CPU threads 2 (rev133: +0.7% overlapping; retain 1);
- joint n_batch/n_ubatch 32/32 (rev133: prefill −6.6%; retain 64/64);
- lossy KV (parity diff), draft depth/pmin sweeps (flat within noise);
- cross-backend copy elimination (zero copies measured).

## Sequencing on console return

1. ~~rev129 runbook~~ done as rev130: profile-OFF timing pair + profile-ON
   attribution pair, parity OK; DSPLIT quantified.
2. C8 done: knob-gated timestamp disable with per-call sample invalidation,
   unavailable-sample counter, labelled summaries, selftest forcing/rejection;
   device A/B chat64 +3.15% seq / +2.60% MTP, extended to code/256 (+1.6-3.2%,
   all OFF rows above ON, parity exact). Profile-bound default implemented
   (absent knob = OFF unless profiling) and validated on device, including
   SessionHub identity (host regression 7/7) so a batch-thread change is a
   different load.
3. C1 rejected (retain decode/batch 2; MTP −9.5%/−21.6% at tb4/tb6); C2 no
   robust gain (retain draft 1); C7 joint 32/32 rejected (prefill −6.6%;
   retain 64/64).
4. Same-boundary attribution (rev133, code64/chat256 MTP): verify 63-79% of
   decode; d3w ≈50% (d3g ≈34%), wall−d3w ≈50%. Next meaningful bottleneck is
   the verify round's CPU-side vs submission/fence split — C3/C4 designs or a
   bounded C5 top_prob experiment, chosen with this data.
5. Re-rank C3/C4/C5/C6 with rev130/rev133 numbers; only then design the next
   build. Any structural change gets a written dependency/numerics plan first.
6. Final candidate only: full LTCG rebuild + complete acceptance grid
   (both arms, 3 prompts × 64/256, session, termination, API), host suite,
   formatters, evidence report. Fast iteration builds stay O2/no-LTCG.

## Owner review questions

- C8: acceptable to make the product default "timestamps OFF unless
  profiling is ON" once measured, or keep a static knob?
- Is decoupling batch/verify threads in scope for the product default, or
  bench-only until a full acceptance grid passes?
- Acceptable scope for D3D12 f32 attention work (C3/C4) or should we cap
  effort at C8/C1/C2/C5/C7 plus documentation of the remaining ceilings?

## Outcome (2026-10-08, rev160 LTCG)

- Final candidate build: MSIX sha256
  `74C9DEDDA15493A6673353F08E29A77EE16D21A124783BA023DC8956A03EF1C7`, PFN
  `GianlucaMazza.xllama_1.6.0.160_x64__pj67f1fcj4n14`; acceptance grid +
  termgate + session + API all green
  (`bench/results/qwen4b-maxperf/acceptance-rev160/verdict.md`,
  `FINAL-REPORT.md`).
- Attribution correction: the rev154 MTP +4.42%/+3.61% rows were FFN OFF/ON
  trials with C8 held fixed — rejected FFN candidate gains (chat256 round-49
  divergence), not C8 evidence. C8 stands on its own rev131 A/B (+1.6-3.2%
  per arm). The profile choice is fixed for the process lifetime (a restart
  re-evaluates it), not irreversible.
- Device restored to the documented production profile (`llama.ini` mtp=2,
  `d3d12swiglu.txt=0`, profiling off); optional sequential preset preserved
  in `bench/results/qwen4b-maxperf/presets/`.
- **C5 closed (rev161 trial, no practical gain).** The AVX max scan was
  bit-exact (4697 adversarial checks; SIMD confirmed in the MSVC object) and
  2.6x on the isolated loop, but on-console `top_prob` saved only 2.4-2.5
  ms/run (~4.3%) — the scalar double exp/sum dominates — and the end-to-end
  paired deltas (chat64 +0.43% mean, overlapping ranges; code64 +1.24%, ranges
  touch) are not causal. Candidate reverted (patch preserved in
  `bench/results/qwen4b-maxperf/c5-rev161/`); no LTCG spent. Device runs the
  verified source as the final accepted-source LTCG package 1.6.0.163
  (1.6.0.160 downgrade blocked and uninstall forbidden; 1.6.0.162 interim)
  with the production profile, API, key token parity and termination gates
  verified (`restore-rev163/receipt.txt`).
- **Residual bottlenecks (unchanged):** MTP decode ≈ verify 64.5%, classic
  20.3%, draft 8.5%, catchup 3.1%; verify split d3w ≈50% with ≈50% CPU-side
  vs submission/fence; `top_prob` is exp/sum-bound (order-preserving
  constraint), not scan-bound.
