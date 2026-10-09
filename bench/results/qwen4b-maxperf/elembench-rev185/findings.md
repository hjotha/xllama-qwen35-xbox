# Round: elementwise CPU cost vs width + prefill regression localization

## Hypotheses tested this round

H1: the per-element CPU kernel cost grows with the row width (cache/L2), which
would explain the ubatch 128/256 prefill regression.
H2: the regression lives in the MTP catch-up path.
(Kept from the previous round: per-call scheduler/allocation/copy overhead.)

## Measurements

### Elementwise CPU kernels on the console (elembench, rev187)

Single-op ggml graphs on the console CPU with the app's wiring (n_threads=2,
persistent threadpool), median of 40, deterministic inputs:

| op                  | slope (ns/elem) | intercept (ns/call) |
| ------------------- | --------------- | ------------------- |
| add                 | 0.40            | 3052                |
| mul (1-D broadcast) | 0.28            | 3246                |
| rms_norm            | 0.59            | 1590                |
| cpy (contiguous)    | 0.15            | 441                 |

Empty-graph floor: **731 ns** per call. The earlier 200 us/call reading was a
standalone-backend artifact (threadpool wake per call): the app-like wiring
gives microsecond-scale per-call overhead. Per-element cost at 64 vs 256 rows
(163840 vs 655360 elements): add 0.414 vs 0.406, rms_norm 0.524 vs 0.587
(+12%) -> **H1 REFUTED for the elementwise kernels**: they are fast and
essentially width-flat at the workload shapes.

### Prefill localization (PSTEP bracket, rev183+)

std512 (298 tokens), profile ON:

| cell        | wall | d3w  | d3g  | wall-d3w | ms/token |
| ----------- | ---- | ---- | ---- | -------- | -------- |
| MTP 64/64   | 3187 | 2073 | 1903 | 1114     | 10.69    |
| MTP 256/256 | 4296 | 1924 | 1874 | 2372     | 14.42    |
| seq 64/64   | 3151 | 2076 | 1943 | 1075     | 10.57    |
| seq 256/256 | 4100 | 1757 | 1705 | 2343     | 13.76    |

**H2 REFUTED**: the +30% regression and the wall-d3w doubling persist with MTP
off (no drafter, no catch-up) -> the cost is in the target path.

### Wide-prefill graph census (sched2, graphs selected by the wide norm-0)

CPU bytes per wide chunk: 816 MiB (64-row chunks) vs **2543 MiB** (256-row
chunks) ~3.1x, in the classes tied to the state path and the elementwise
chain: SWIGLU 1152, RMS_NORM 1086, MUL 896, GATED_DELT(CPU) 720, CONCAT 576,
SSM_CONV 576, SILU 576 MiB per 256-row chunk. (Method note: the first graphs in
the device log are 1-row decode graphs; only graphs carrying the wide `norm-0`
are the prefill chunks - the naive "first N graphs" slice compared the wrong
populations and was discarded.)

## Decision

- H1 and H2 are discarded with measurements; no kernel change follows from
  them, and no experimental code is proposed on their basis.
- The sustained bottleneck for prefill is the **CPU-side graph volume in wide
  chunks** (target path), now quantified per class; the CPU kernels themselves
  are fast, so the remaining lever is the volume/granularity: either keep
  ubatch at 64 (already the production config) or reduce the wide-chunk CPU
  volume (fusion/placement), which needs its own evidence before any change.
- Decode side unchanged: verify-GPU-bound; the CPU elementwise share is small
  per this bench.

Instrumentation added (diagnostic-only, default off): `elembench` flag +
`run_elembench` (CSV in LocalState); `DSTEP kind=prefill` bracket.
