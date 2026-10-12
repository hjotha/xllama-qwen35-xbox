# Plan 007 — bounded GPU chain: RMS_NORM + MUL on D3D12 (owner 599)

Goal: reduce CPU round-trips by giving the D3D12 backend a **complete, bounded
chain** of the graph's element ops — `RMS_NORM` followed by the norm-weight
`MUL`. Both are high-count CPU nodes that sit between D3D12 `MUL_MAT` nodes in
the same graphs, so the working hypothesis is that the scheduler then merges
those CPU splits into the adjacent D3D12 ones. **That merge is a hypothesis,
not a construction**: it depends on the actual graph edges, tensor placement
and split assignment, so the split count before/after is the first measurement
of this plan and decides acceptance.

## What was measured first (profile before optimizing)

Independent factorial on the production package (1.6.0.165), same model
(`qwen35-4b-mtp`), same workload, profile OFF, 2 interleaved rounds,
27/27 cells OK (`batchfactorial-rev165/`, 6 valid combos × 2 prompts × 2 rounds

- attribution cells):

| n_batch | n_ubatch | prefill chat t/s | prefill std512 t/s | decode t/s | WS MB |
| ------- | -------- | ---------------- | ------------------ | ---------- | ----- |
| 64      | 64       | 88.4             | 88.6               | 28.1–29.1  | 3666  |
| 128     | 64       | 90.8             | 91.7               | 28.1       | 3669  |
| 256     | 64       | 92.2             | 91.9               | 27.8       | 3672  |
| 128     | 128      | 78.1             | 77.9               | 28.2       | 3732  |
| 256     | 128      | 78.1             | 78.1               | 28.2       | 3735  |
| 256     | 256      | 80.1             | 67.7               | 28.3       | 3858  |

Established: the prefill regression tracks **n_ubatch alone** (64 fastest for
every n_batch; 128 ≈ −12%, 256 ≈ −24% on the long prompt); n_batch alone is
neutral-to-positive; decode is flat; GPU memory is constant (2692 MB) and only
WS grows. The earlier "n_batch 128/256" verdicts were confounded sweeps.

Census (`ggmlprof.txt=sched2`, per-cell logs under `batchfactorial-rev165/logs/`):
per graph the scheduler emits **~123 splits, 61 CPU + 61 D3D12**, alternating
almost every boundary, with **~326 CPU node executions per graph**:

| CPU op   | per graph |
| -------- | --------- |
| RMS_NORM | 52        |
| MUL      | 48        |
| CPY      | 44        |
| GET_ROWS | ~32       |
| SCALE    | ~30       |
| SWIGLU   | ~18       |
| CONCAT   | ~8        |
| SET_ROWS | ~6        |

`dev_supports_op` accepts only MUL_MAT, GATED_DELTA_NET, the selftest-gated
UNARY/GLU probes and view ops (`src/bridge/ggml_d3d12.cpp:1990`), so every
norm/mul/cpy lands on the CPU and interleaves with the GPU splits.

Same-boundary stage profile (profile ON, per-run slices) records stage wall
and the d3w/d3g brackets. The per-call CPU clock is NOT a bottleneck signal:
it sums across worker threads and includes the backend's busy-spin wait, so it
can exceed the stage wall (e.g. draft wall 316 ms / d3w 160 ms / cpu 391 ms
at 64/64). Any CPU-side bottleneck claim must come from stage wall minus GPU
time plus copy/synchronization attribution, which this plan adds rather than
infers. Prefill has
**no same-boundary instrument today** (DSTEP only wraps draft/catchup,
`decode_loop.h:547,822`) — one step of this plan adds a profile-only
`StepSplit("prefill")` so prefill gets the same wall/d3w/d3g/cpu bracket.

## Non-goals (explicit)

No residual ADD, no CPY elimination, no SCALE/GET_ROWS/CONCAT port, no
replacement of CPU orchestration or sampling, no promise of a speedup before it
is measured. If the pair does not measurably reduce split count or paired
prefill/decode time, it is reverted.

## Implementation shape

- `RMS_NORM` kernel: f32 in/out, contiguous, `ne[0] % 8 == 0`, one row per
  thread group reduction, `eps` as root constant; predicate rejects anything
  else so the CPU kernel still owns it.
- `MUL` kernel: f32, two supported forms — 1D weight broadcast along `ne[0]`
  (weight `ne[0] == src->ne[0]`, other dims 1, stride 0 on dim 1) and scalar
  (`ne` all 1, value in root constants).
- Both wired into `dev_supports_op` and the op collector in
  `backend_graph_compute` (which aborts on unknown ops).
- Numerics: kernels must match the CPU ggml results bit-for-bit on the shapes
  actually produced; adversarial selftest rows (all-zero row, ±0, NaN/Inf
  inputs, one huge element, eps-sensitive rows, tiny/huge rows, weight with
  zeros) before any device timing.

## Validation gates (in order)

1. Host: predicate tests, selftest-vs-CPU row comparison, formatters, full ctest.
2. Fast iteration build on .193 (no LTCG): census shows the split count drop
   (the chain's purpose) — the accepted evidence for the design, measured
   before any perf claim.
3. Device: full-token parity (chat64/code64 seq+MTP, digests vs baseline), MTP
   acceptance and draft-feed behavior (per-round `VROUND feed=`), API chat
   PASS, session/stop/cancel/EOS termgate, memory guard (`gpu_mem_mb`, peak WS).
4. Paired prefill **and** decode for seq and MTP, profile OFF, ≥2 interleaved
   rounds per arm, medians and spreads reported (no promised numbers).
5. Only if 2–4 pass: full LTCG final build; otherwise revert to rev165.

## Numerics reference (established, not assumed)

The f32 CPU kernel (`ggml_compute_forward_rms_norm_f32`) accumulates
`sum += (double)(x[i]*x[i])` in **strictly ascending** index order, then
`mean = (float)(sum/ne00)` and `scale = 1.0f/sqrtf(mean + eps)`. A parallel
double reduction reorders the additions, so the shader must use `double` lanes
plus a `double` tree and then round once to float exactly like the CPU path;
`1/sqrt` must stay a precise float divide/sqrt (no `rsqrt`). This does not make
the orders equal, so the plan measures, per adversarial row, the exact-match
rate and the relative error against the CPU kernel, and the end-to-end full-token
parity plus the MTP acceptance/logit gates decide acceptance. No bitwise
identity is promised without that evidence.

## Known unknowns (to close by measurement, not assertion)

- Whether the 64/128/256 ubatch prefill regression is CPU element work, copy
  staging, or L2/DRAM residency — the census separates op counts and bytes,
  not time; the prefill bracket (step above) and the paired sweep are the
  instruments.
- Whether the `SWIGLU`/`SCALE`/`GET_ROWS` CPU ops are worth a second bounded
  chain (not authorized yet).
- The drafter is greedy and acceptance is deterministic equality; the emitted
  token is always the target's own sample, so MTP cannot bias output — the
  equal-history numerical diagnosis continues under this plan's evidence.

## Outcome (measured, REJECTED — rev168)

- Engagement proven on rev168: knob `bench_normmul.txt=1` logs
  `RMS_NORM+MUL D3D12` and the census run reports **3515 RMS_NORM + 4060 MUL
  dispatches**; census shows RMS_NORM 26.8/g and MUL 31.1/g moved to D3D12
  (the non-1D-broadcast MULs and GDN norms stay on CPU).
- **Split count rose**: 124/g (62 CPU + 62 D3D12) -> 148/g (74 + 74): the
  merge hypothesis is falsified by measurement.
- **Paired same-build A/B (interleaved, profile OFF)**: std512 MTP prefill
  -1.03% / decode -3.55%; chat64 MTP -2.96% / -4.17%; chat64 seq +3.46% /
  -5.83%. All token digests exact (chat64 `84e08196…`, std512 `44546453…`).
- The rev167 A/B is void (the knob had no call site; both arms ran the CPU
  chain); the rev166 cross-build numbers are exploratory only.
- Reverted; production restored as 1.6.0.169 LTCG from the verified rev165
  source (API PASS, parity, termgate 6/6 byte-identical). Patch, shaders and
  DXIL preserved in `bench/results/qwen4b-maxperf/normmul-rev168/`.
- A future attempt needs the whole chain covered (all MUL forms + GDN norms)
  so the islands collapse, plus the FP64 reduction cost isolated first.
