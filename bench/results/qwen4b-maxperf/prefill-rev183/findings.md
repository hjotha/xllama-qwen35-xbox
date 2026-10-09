# Prefill same-boundary attribution (rev183, owner 599)

Instrument: `DSTEP kind=prefill` per chunk inside `prefill_chunked` (same
wall/d3w/d3g/cpu/calls/mm accounting as the decode steps), profile ON only.
Workload: std512 (298 prompt tokens), MTP, greedy, one run per cell.

| cell    | chunks | wall ms | d3w  | d3g  | wall-d3w         | ms/token       | GPU ms/token |
| ------- | ------ | ------- | ---- | ---- | ---------------- | -------------- | ------------ |
| 64/64   | 5      | 3187    | 2073 | 1903 | 1114             | 10.69          | 6.39         |
| 128/128 | 3      | 3619    | 1973 | 1896 | **1646 (+48%)**  | 12.14 (+13.6%) | 6.36         |
| 256/256 | 2      | 4296    | 1924 | 1874 | **2372 (+113%)** | 14.42 (+34.8%) | 6.29         |
| 256/64  | 2      | 3171    | 2057 | 1915 | 1114             | 10.64          | 6.42         |

Findings:

- **GPU compute is flat**: d3g total 1874-1915 ms across all ubatch settings
  (6.3-6.4 ms/token). The ubatch regression is not GPU matmul time.
- The regression tracks **wall minus d3w** (the scheduler plus the CPU-assigned
  node work outside the D3D12-per-call bracket): +48% at ubatch 128, +113% at
  ubatch 256 per run, matching the per-token prefill regression (+13.6%/+34.8%).
- `calls`/`mm` fall with bigger ubatch (925/1005 at 64 vs 346/402 at 256): the
  graph is traversed fewer times but each CPU-side node handles wider rows, and
  the CPU-side cost grows superlinearly with width (cache/L2 residency of the
  widest elementwise tensors). This matches the earlier census (per-graph CPU
  node count constant, CPU bytes per graph growing with ubatch).
- b256/u64 keeps the CPU chunk width at 64 and is as fast as 64/64 despite the
  larger n_batch: the controlling variable is the ubatch width, consistent with
  the independent factorial.

Caveat: the per-call `cpu` clock sums thread time and includes the backend's
busy-wait, so it is not used as a bottleneck signal here; the load-bearing
metric is wall minus d3w, a wall-clock bracket, together with the flat d3g.

## Class census (sched2, prefill graphs only)

Per chunk the graph structure is identical; only the row width changes:

| cell    | chunks | CPU nodes/chunk | CPU classes/chunk                                                                    |
| ------- | ------ | --------------- | ------------------------------------------------------------------------------------ |
| 64/64   | 5      | 996             | RMS_NORM 153, CPY 144, MUL 137, GET_ROWS 98, SCALE 96, ADD 88, SWIGLU 56, MUL_MAT 48 |
| 256/256 | 2      | 996             | same counts                                                                          |

- The per-chunk node counts are identical, so 256/256 executes FEWER node
  invocations in total (1992 vs 4980) yet is slower: the regression is
  per-node width behaviour, not node count and not graph structure.
- **Every prefill split reports 0 split inputs**: the scheduler introduces no
  explicit cross-backend copy at these boundaries (D3D12_Host activations), so
  copy volume is not the explanation. The cost is the CPU-side node execution
  (RMS_NORM/CPY/MUL/GET_ROWS/SCALE/ADD/SWIGLU) whose per-element cost grows
  with the row width.
- Cache/L2 residency of the wider tensors remains the leading hypothesis for
  the per-element growth but is **not yet measured** (the CPU kernels are not
  instrumented per node); a shape-controlled elementwise microbenchmark on the
  console CPU would be the next discriminator.
