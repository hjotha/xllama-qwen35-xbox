# GGUF GPU decode — design and D1 gates

> **SSOT for the GGUF GPU decode design** (H6 follow-up, #228): architecture,
> cost model, predeclared gates D1/D2 and their verdicts. Kernel evidence lives
> in [phase15-re-opt.md](phase15-re-opt.md) (WS-E / H6.3); performance numbers
> live in [benchmarks.md](benchmarks.md) and `bench/results/`. Platform limits
> are only in [uwp-constraints.md](uwp-constraints.md).

**Status (2026-10-02):** D1 = `D2-matmul-only`; **D2a = PASS** (backend
`d3d12` selftest, CI `1.6.0.1125`); **D2 = FAIL on peak RAM** (CI `1.6.0.1138`):
decode 1.59× / 1.58× and prefill pass on Coder-3B / LFM2.5-1.2B, peak RAM is
+539 / +265 MB over the CPU. The backend ships opt-in
(`gguf_gpu_layers.txt`), default off; GGUF decode ships on the CPU.

## Why

GGUF decode on Series S is bandwidth-bound on the CPU (~12–13 GB/s effective,
[phase15-re-opt.md](phase15-re-opt.md) baseline). H6.3 measured our own D3D12
Q4*K GEMV at **143.06 GB/s packed** (`bench/results/phase15-gpugemv-h63.csv`,
K3). The predeclared ladder sends K3 to a decode \_design*, not a backend: one
GEMV tile is not a decode step. This document fixes the architecture and the
gates before any backend code exists.

## Architecture

A **ggml backend `d3d12`**, owned by xllama (`src/bridge/`, not the
`llama.cpp/` submodule), statically linked and registered at runtime with
`ggml_backend_register()` before the model loads. The API is public
(`ggml-backend.h`); no dlopen ([uwp-constraints.md](uwp-constraints.md) §3).

- **Shape:** the `ggml-blas` template — `supports_op` claims `MUL_MAT` plus the
  view ops (`NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE`), and an opt-in Qwen
  `GATED_DELTA_NET` kernel. Other operations stay on the CPU backend.
- **Placement:** weights go to the backend's buffer type through
  `n_gpu_layers` (already `SessionParams::n_gpu_layers`, default 0). The
  scheduler runs an op where its weight lives; weights of a type the backend
  does not support stay on the CPU through `weight_buft_supported`.
  Supported matmul weights are Q4_0, Q4_K, Q5_K, Q6_K and Q8_0; Q8_0 can
  be disabled for a same-package CPU control. `token_embd` / `GET_ROWS` are always CPU.
- **Kernels:** the H6.3 `rows` shape (64 threads × 4 rows, X in registers) per
  weight type: Q4_K and Q4_0 (same density, 0.5625 B/weight) and Q6_K (tied
  `lm_head` of every target model, `attn_v`/`ffn_down` on Q4_K_M
  `use_more_bits` layers).
- **Device:** the system D3D12 runtime via `d3d12_dyn`, never the Agility
  factory ([uwp-constraints.md](uwp-constraints.md) §7). Shared plumbing:
  `src/bridge/d3d12_compute.{h,cpp}` (probes and backend).
- **Invariants kept:** logits come back to the CPU, so the sampler chain
  (`sampler_chain.h`), `fit_prompt` and `SessionHub` (one resident model) do
  not change. The CPU path stays the default; the backend is opt-in.

Rejected: the ggml Vulkan backend (no Vulkan in the AppContainer); DirectML
(no fused low-bit GEMM, [uwp-constraints.md](uwp-constraints.md) §12); a
hand-written forward pass outside ggml (duplicates llama.cpp per-architecture
code — LFM2 short-conv, Qwen biases — and splits the sampler chain).

## Cost model

```
t_token = Σ_splits (t_rt + bytes_split / BW) + t_cpu_rest
```

- **BW** = 143.06 GB/s (H6.3 `rows`, `kGpustepRowsBwGbs`).
- **Split rule (assumption, D2 verifies with `ggml_backend_sched_get_n_splits`):**
  adjacent matmuls sharing an input run in one split — q/k/v, o, gate+up,
  down — so ~4 splits per layer plus the `lm_head`.
- **t_rt** — one CPU↔GPU round trip per split — and **t_cpu_rest** (norms,
  RoPE, attention, conv, sampling on the CPU) are unknown. D1 measures t_rt and
  the whole step; t_cpu_rest is measured in D2.

Model tables (`gpustep_model()`, `include/xllama/gpustep.h`) come from the
GGUF headers (LFM2.5 QAD Q4_0) and the Qwen2.5-3B config plus the llama.cpp
Q4_K_M promotion rule (Coder-3B):

| Model                 | Splits | Bytes / token | GEMV at BW | + t_rt 20 µs |  + 50 µs | + 100 µs |
| --------------------- | -----: | ------------: | ---------: | -----------: | -------: | -------: |
| `qwen25-coder-3b`     |    145 |       1923 MB |   13.44 ms |     16.34 ms | 20.69 ms | 27.94 ms |
| `lfm25-1.2b-instruct` |     65 |        693 MB |    4.84 ms |      6.14 ms |  8.09 ms | 11.34 ms |
| `lfm25-350m`          |     65 |        217 MB |    1.51 ms |      2.81 ms |  4.76 ms |  8.01 ms |

**Projection, not a claim** — matmul+sync time only, without t_cpu_rest.
`xllama-cli --gpustep` prints this table. The smaller the model, the larger the
share of fixed per-split cost: GPU decode is a 1B–3B lever first.

## Risks and the gate that settles each

| Risk                                                                                                                                     | Settled by    |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Round trips dominate (~145 per Coder-3B token)                                                                                           | D1a, D1b      |
| A system D3D12 device fails inside the XAML process (chat and LAN API live there; every GPU probe so far was headless)                   | D1d           |
| Copies are avoidable on unified memory (weights/activations in a CPU-visible heap)                                                       | D1c (informs) |
| Prefill: weights in the GPU buffer also send prefill matmuls to the GPU — needs a batched kernel or CPU-side prefill                     | D2 prefill    |
| Numerics: fp32 accumulation order differs from the CPU, greedy text can diverge                                                          | D2 H9         |
| ORT DirectML (Agility factory) in the same process after our device exists (`routing` gate)                                              | D2 gates      |
| GPU budget 3801 MB ([uwp-constraints.md](uwp-constraints.md) §7) and contention with diffusion/DML (audit Proposal C, compute admission) | D2 peak / D3  |

## Phases and predeclared gates

All console gates run on the CI MSVC package (`xllama-appx`), never a
crossbuild.

### D1 — probe (`gpustep`, measure-only)

`gpustep.flag` (headless) and `gpustep-inproc.flag` (inside the XAML process,
after `Window.Activate()`), `scripts/bench-gpustep.sh`, CSV
`bench/results/phase15-gpustep-d1.csv`. Sections: `caps` (UMA,
CacheCoherentUMA), `heap` (N=K=8192 GEMV with W in DEFAULT / UPLOAD / CUSTOM
WRITE_BACK-L0), `rt` (N=256 K=2048: host X → GPU → host Y; event-copy,
spin-copy, spin-zerocopy), `step` (one simulated token per model; `sync` = CPU
round trip per split, `nosync` = all splits in one submit). Q6_K matmuls are
simulated as bytes-equivalent Q4_K rows.

| Gate  | Criterion                                                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------------------------------------- |
| D1-G1 | `rt` copy variants and the DEFAULT `heap` GEMV match the CPU reference (tolerance as gpugemv G1) in the headless process        |
| D1a   | best headless `rt` median ≤ **100 µs**                                                                                          |
| D1b   | headless `qwen25-coder-3b/sync` median ≤ **40 ms** (40 + ≤ 11 ms CPU work = 51 ms = 1.4× the CPU 14.0 tok/s, `phase14-console`) |
| D1c   | decision only: best CPU-visible heap ≥ 0.9× DEFAULT GEMV BW → host-visible buffers in D2, else DEFAULT + copies                 |
| D1d   | in-XAML run: D1-G1 holds and no device-removed / HRESULT error                                                                  |

Ladder (`gpustep_evaluate`, printed by `xllama-cli --gpustep-verdict`):

| Verdict          | Condition                                                    | Next                                           |
| ---------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| `D2-matmul-only` | D1-G1 + D1a + D1b + D1d                                      | D2 as designed above                           |
| `D2-fused`       | D1d, but D1a or D1b fails while the `nosync` step is ≤ 40 ms | D2 claims more ops per split (fewer syncs)     |
| `park-3b`        | D1d, and even the `nosync` step misses 40 ms                 | park #228 with the number                      |
| `park-gui`       | D1d fails                                                    | park for GUI/API (headless-only is no product) |
| `NotAVerdict`    | a required row is missing or D1-G1 fails headless            | fix the probe, rerun                           |

### D1 result (2026-10-01)

CI MSVC package `1.6.0.1117` (run 36927534262), Series S, both processes,
`bench/results/phase15-gpustep-d1.csv`. Every row `ok=1`, `d3d12_ran=1`;
`caps`: UMA = 1, CacheCoherentUMA = 1.

| Measure                                                 | Headless            | In-XAML             |
| ------------------------------------------------------- | ------------------- | ------------------- |
| `rt` spin-zerocopy / spin-copy / event-copy (median µs) | 49.7 / 53.1 / 56.9  | 61.0 / 60.8 / 71.0  |
| `rt` p90, best variant (µs)                             | 53.8                | 155.3               |
| `heap` DEFAULT / UPLOAD / CUSTOM (GB/s)                 | 144.4 / 57.7 / 57.8 | 118.7 / 52.1 / 53.9 |
| `qwen25-coder-3b` sync / nosync (ms)                    | **21.16** / 14.74   | 21.50 / 14.90       |
| `lfm25-1.2b-instruct` sync / nosync (ms)                | 8.50 / 5.68         | 8.66 / 5.69         |
| `lfm25-350m` sync / nosync (ms)                         | 4.89 / 2.26         | 4.97 / 2.27         |

| Gate  | Result                                                                                        |
| ----- | --------------------------------------------------------------------------------------------- |
| D1-G1 | PASS (headless and in-XAML)                                                                   |
| D1a   | PASS — 49.7 µs ≤ 100                                                                          |
| D1b   | PASS — 21.16 ms ≤ 40                                                                          |
| D1c   | **DEFAULT + copy** — CPU-visible heaps read at 0.40× DEFAULT (57.8 vs 144.4 GB/s) despite UMA |
| D1d   | PASS — the system D3D12 device works next to the compositor                                   |

**Ladder: `D2-matmul-only`.** What the numbers say:

- The cost model holds: Coder-3B `sync` 21.16 ms against 20.69 ms projected at
  t_rt = 50 µs; `nosync` 14.74 ms against 13.44 ms of pure GEMV.
- Sync costs ~6.4 ms per Coder-3B token (145 splits × ~44 µs): real, but
  inside the gate. D2-fused stays an optimisation, not a prerequisite.
- Weights go in DEFAULT heaps. UMA is real (`cc_uma=1`) but CPU-visible pages
  halve GPU read bandwidth; only small per-split activations go through upload
  and readback buffers (zero-copy X/Y saves ~3 µs per round trip).
- In-XAML: medians match headless, the p90 round trip triples (155 µs) and
  DEFAULT-heap GEMV drops 18% — the compositor shares the GPU. D2 must be
  measured in the UI process, not only headless.
- Still no tok/s: CPU-side ops (t_cpu_rest) are not in any of these numbers.

The `gpugemv` rerun on the same package (regression check for the
`d3d12_compute` refactor) gave `rows` 138.11 GB/s (−3.5% vs 143.06, inside
±5%), `wave32` 24.99, `dot4` 139.50 — K3 unchanged.

### D2 — implementation (decided against the pinned llama.cpp)

`include/xllama/ggml_d3d12.h` + `src/bridge/ggml_d3d12.cpp`, a GPU-type ggml
device registered with `ggml_backend_register()`:

- **Two buffer types.** llama.cpp lists a GPU device's _default_ buft before
  its extra bufts, and the scheduler allocates activations in the default one.
  So the default is **`D3D12_Host`** (CUSTOM heap, WRITE_BACK/L0,
  `is_host`): CPU→GPU copies become `memcpy`, the CPU reads results in place,
  one round trip per split (the D1 zero-copy path). Weights go to the extra
  buft **`D3D12_Weights`** (DEFAULT heap, D1c).
- **`supports_op`** accepts `MUL_MAT` only when the weight already lives in
  `D3D12_Weights` (Q4_0 / Q4_K / Q5_K / Q6_K, K multiple of 256, contiguous, 2-D;
  F32 activations). llama.cpp's per-weight buft probe therefore skips the host
  buft; norms, biases, short-conv, `token_embd` and other types fall back to the
  CPU list by themselves.
- **Kernels** `shaders/ggml_d3d12_mmv_{q4_0,q4_k,q5_k,q6_k}.hlsl`: the H6.3 `rows`
  layout plus a column index for prefill; Q4_0 (18 B) and Q6_K (210 B) blocks
  are read with 2-byte-aligned dword loads (Coder-3B Q6_K `ffn_down` rows are
  9030 B), while Q4_K (144 B) and Q5_K (176 B) blocks are 4-byte aligned and load
  whole dwords. Weight tensors get 16 B of padding: root descriptors have no bounds
  check. Two blobs per type, 64 or 128 threads per group,
  picked by K (`d3d12_mm_threads`: 128 from K = 4096): long-K `ffn_down`
  needs 8 chunks in flight, short K starves them (D2a runs 1 and 2).
- **`graph_compute`**: root constants + root SRV/UAVs per matmul, one submit,
  spin fence; the call returns with the work done.

**D2a gate (console selftest `d3d12be.flag`, `scripts/bench-d3d12-selftest.sh`,
predeclared):** every type × shape (`{n, k}` from Coder-3B and LFM2.5 tensors,
ncols 1 / 7 / 512) within `rel_err ≤ 1e-2` of ggml's dequantizers, and every
decode case (ncols = 1) at ≥ 100 GB/s packed (GPU timestamps). Host tests
emulate each kernel's lane mapping against the same dequantizers.

**Shape-cost bench (same flag, separate CSV, reported not gated):** after the
gate rows and their `.done` are written, `run_d3d12_shape_cost` costs the 11
real `(type, N, K)` triples of the `tttarget` histogram at batch widths
1/2/3/5 — smallest weights first, the Q6_K 248320×2560 lm_head last — into
`d3d12sc-result.csv` (+ `.done`), with wall time next to the GPU timestamp
plus the packed weight size and the process peak working set per row. Each
row is the median of 5 consecutive timed runs with the max-min range next to
it, so variance is measured rather than sampled once; the D2a gate rows keep
their exact single-sample methodology. Weights
are generated in bounded 16 MiB chunks from the same shape-only seed, so the
large shapes never need their full fp32 source; host tests prove the bytes
(and the rng continuation) identical to the one-shot path. The D2a gate above
keeps its exact inputs: nothing in this CSV can pass or fail it.

**Paired OLD/NEW experiment (q4_k, B ≥ 2):** each such case runs 4 OLD+NEW
pairs on the same resident tensors with alternating order (both variants
warmed), emitting one row per pair per variant (`variant`, `pair` columns)
plus synthetic edge rows (odd N = 1023, single-row N = 1, documented as
non-histogram). The bench script aggregates same-run pair deltas
(NEW−OLD); a speedup is claimed only from those pairs, never across runs.
Both variants meet the unchanged `rel_err ≤ 1e-2` threshold per row.

**Real-decode knob (variant 2, allowlisted):** `d3d12twocol.txt` with content
`auto` in LocalState engages the two-column tile in real decode (read by the
tttarget bench with scoped restoration; default OLD). Variant 2 dispatches NEW
only on the hardcoded allowlist — the rev57 same-run pair winners (five q4_k
shapes at B=2/3/5); B=1, B=4, prefill widths, small shapes and other types
stay OLD until measured and validated. Variant 1 (force, bench internals only)
never comes from the knob. Broadening the allowlist is a code change with
fresh evidence, never a knob edit.

**Padded-stride gate (same bench):** strided X/Y rows and weight rows with
sentinel padding (`pads` column, `x/y` floats and `w` bytes, odd values to
catch alignment assumptions) on winner shapes with column tails. Padding is
verified untouched against sentinels on device readback per variant, and
outputs must match the strided reference; a clobbered pad fails the row even
when the numerics match. supports_op routes strided inputs to CPU in
production (it requires contiguity), so this is defense in depth: the bench
bypasses it with direct graph_compute to prove the kernels honor strides.

### D2a result (2026-10-01/02, three console runs)

`scripts/bench-d3d12-selftest.sh`, GPU timestamps, rel_err against ggml's
dequantizers (all runs: 12/12 correct, rel_err ≤ 1.7e-7):

| Decode shape (n × k) | Run 1, 64 threads | Run 2, 128 threads | Run 3, width by K |
| -------------------- | ----------------: | -----------------: | ----------------: |
| q4_0 8192 × 2048     |             127.0 |              118.2 |         **116.9** |
| q4_0 2048 × 8192     |          **90.6** |              104.7 |         **104.0** |
| q4_k 11008 × 2048    |             144.9 |              132.5 |         **143.6** |
| q4_k 2048 × 11008    |             127.4 |              130.3 |         **129.6** |
| q6_k 2048 × 11008    |          **94.3** |              100.7 |         **102.3** |
| q6_k 65536 × 1024    |             110.7 |           **64.1** |         **114.4** |
| D2a                  |              FAIL |               FAIL |          **PASS** |

GB/s packed, CI packages `1.6.0.1123` / `1124` / `1125`; CSVs
`bench/results/d2a-d3d12-selftest{,-t128,-adaptive}.csv`. Run 1 starved
long-K matmuls (512 groups for N = 2048); run 2 starved short K (4 chunks for
8 slots); run 3 picks the width per K and passes every decode case. The
q4_0 8192 × 2048 row sits ~10 GB/s below run 1 at the same width — run-to-run
spread on a 9.4 MB matmul (identical blob), not a code difference. Prefill
columns run at ~3–5.5 µs per column at 1024 × 1024; the prefill gate is D2b's.

### D2 — product gate (written before D1, measured in D2b)

`qwen25-coder-3b` first, `lfm25-1.2b-instruct` second, both with
`n_gpu_layers` = all against the same package's CPU t6 run:

- decode ≥ **1.4×** CPU (median of 3 recorded runs, prefill and decode reported
  separately);
- prefill at P=1000 ≥ **0.9×** CPU;
- H9 score ≥ the CPU run on the same model;
- peak RAM ≤ CPU run; GPU memory ≤ budget;
- `validate-console.sh all` PASS with the backend enabled (includes `routing`,
  i.e. ORT DirectML after our device).

### D2b — integration (opt-in, default off)

- `src/bridge/llama_gpu.h` is the one place a GPU-layer request becomes llama
  params:
  - `devices` = `{D3D12}`, `n_gpu_layers`, `no_host`;
  - `offload_kqv = false`;
  - persistent CPU threadpools (`llama_attach_threadpool`): without them,
    ggml-cpu builds a pool per split.
  - Both paths use it: `run_inference_llama` (CLI, headless bench) and
    `LlamaSession` (GUI, LAN API).
- The device reports `D3D12_Host` as its host buft. llama.cpp then puts the
  CPU backend's compute buffer there, `supports_buft` accepts it, and splits
  share activations in place: one 305 MiB compute buffer instead of 322 + 81
  on Coder-3B, and no input copies.
- `SessionHub` reloads when the GPU-layer request changes (weights move between
  buffer types).
- **How to enable (experimental):**
  - `LocalState\gguf_gpu_layers.txt` with a layer count (`99` = all) for GUI and
    API sessions, or `LocalState\llama.ini` with `n_gpu_layers = 99` (the
    single-purpose file wins when both exist);
  - `bench_gpu_layers.txt` via `scripts/bench-xbox-ort.sh --gpu-layers N`
    (host tag `-gN`, CSV `backend` = `d3d12`);
  - `xllama-cli --gpu-layers N` (no D3D12 on Linux → CPU).
  - `--ignore-eog` (`bench_ignore_eog.txt`, tag `-noeog`) decodes exactly
    `n_predict` tokens so both arms time the same length.

### D2 result (2026-10-02, CI `1.6.0.1138`)

`bench/results/2026-10-02-d2-gguf-gpu-hostbuft.csv`: median of runs 2–4,
standard-512 prompt and 256 tokens for decode, long-1k (P ≈ 950–1000) for
prefill, CPU t6 against `--gpu-layers 99` on the same package.

| Criterion                 | Coder-3B (Q4_K_M)            | LFM2.5-1.2B (QAD Q4_0)         | LFM2.5-350M (informative) |
| ------------------------- | ---------------------------- | ------------------------------ | ------------------------- |
| decode ≥ 1.4×             | **PASS** 14.4 → 22.8 (1.59×) | **PASS** 39.6 → 62.5 (1.58×)   | 99.9 → 95.3 (0.95×)       |
| prefill P=1000 ≥ 0.9×     | **PASS** 45.6 → 60.1 (1.32×) | **PASS** 108.7 → 189.3 (1.74×) | 368 → 546 (1.48×)         |
| peak RAM ≤ CPU            | **FAIL** 2044 → 2583 MB      | **FAIL** 783 → 1048 MB         | 311 → 514 MB              |
| GPU memory ≤ 3801 MB      | PASS 1851                    | PASS 678                       | 224                       |
| H9 ≥ CPU                  | **FAIL** 6/8 → 5/8           | **PASS** 6/8 → 6/8             | —                         |
| `validate-console.sh all` | PASS (knob = 99)             | PASS (knob = 99)               | —                         |

**D2 = FAIL** — peak RAM on both gate models, H9 on Coder-3B; every speed
criterion passes.

- **H9** (`bench/results/2026-10-02-d2-h9-{cpu,gpu}.jsonl`, LAN API, XAML
  process, temperature 0, two identical runs per arm): on Coder-3B,
  `constrained_summary` becomes one sentence instead of the three-item list.
  The d3d12 matmuls take f32 activations where ggml-cpu quantizes them to q8,
  and greedy text diverges (#312).
- **UI process:** the same H9 requests end to end (load and prefill
  included) run 7.34 → 11.17 tok/s on Coder-3B (1.52×) and 14.13 → 21.93 on
  LFM2.5-1.2B (1.55×).
- **Peak RAM:** on UMA the DEFAULT-heap weights count in the working set like
  CPU weights, so the excess is what the GPU path adds. Coder-3B load log:
  - 244 MiB: llama.cpp duplicates the tied `token_embd` into `D3D12_Weights`
    as the lm_head; it also stays on the CPU for `GET_ROWS`;
  - ~300 MB: the D3D12 runtime and the larger compute reservation.
- **Tied lm_head on the CPU** (`-ocpu` rows in
  `2026-10-02-d2-gguf-gpu-final.csv`, CI `1.6.0.1137`): Coder-3B peak 2620 →
  2110 MB, still above the CPU's 2044, and LFM2.5-1.2B decode 1.37×, below the
  gate. Removed rather than kept as a per-model switch.

Follow-ups: peak RAM #309, CPU-side split cost #310, numerics #312,
multi-column prefill #313.

### D3 — product decision

Default on/off per model in `docs/model-matrix.md` after D2; the
measured-is-not-shipped ladder applies.

## Xbox MTP backend experiments

The 2026-10-07 campaign extends the experimental backend for Qwen3.5 MTP.
Controls below are LocalState files read before model loading; restart the
app after changing them. They do not replace the model validation ladder.

- `d3d12q8.txt`: absent or `1` enables GPU Q8_0 matmul; `0` selects the CPU
  control. The MTP hidden-state projection uses this weight type.
- `d3d12gdn.txt`: `1` enables supported gated-delta graphs on the GPU;
  `2` uses CPU for one token and GPU for larger supported batches; absent
  or `0` keeps the CPU control. The shader preserves the fork's recurrent
  snapshot layout and rejects unsupported shapes.
- `d3d12q6tile.txt`: `1` keeps the original Q6 kernel; `2` or `4` selects
  column reuse for the Qwen LM-head at batch widths 2, 3 and 5. Other shapes
  use the original kernel. Experimental `0` selects two columns at width 2
  and four columns at widths 3 and 5, using the same allowlist. This is independent of `d3d12twocol.txt`, which
  controls the existing Q4 tile allowlist.
- `mtp_threads.txt`: draft worker count, bounded by the existing target
  threadpool's capacity. Both frontends already attach that shared pool;
  there is no separate draft pool.
- `mtp_catchup_logits.txt`: absent or `0` avoids logits for committed-row
  catch-up, matching the native reference; `1` retains the legacy control.
  Empty output matmuls remain assigned to their owning GPU weights and are
  skipped before dispatch. Rejecting those empty nodes would make the
  scheduler copy the LM-head weights to CPU despite no output being needed.
- `flashattn.txt`: `1` forces the native fused CPU attention route; absent
  or `0` retains auto selection. This does not enable a GPU attention shader.
  Experimental `2` also disables native single-query split-KV reduction
  through patch 0006. The same per-query reduction then serves sequential
  decode and multi-query MTP verification; this addresses the observed F16
  split/unsplit argmax drift. Performance and full-token parity still require
  paired hardware confirmation. The R123 FA2 grid records identical outputs
  across the three prompts at both generation caps; final package receipts
  remain in the same evidence directory.

Hardware receipts, paired kernel results and complete token dumps are in
[`bench/results/fast-mtp-20261007/`](../bench/results/fast-mtp-20261007/).
The [Windows runbook](windows-dev-vm.md#incremental-xbox-iteration-on-the-windows-build-host)
describes separate incremental and final optimized builds.

## Decision log

| Date       | Decision                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-01 | Design: ggml backend `d3d12`, matmul-only first, opt-in. D1 gates and the D2 product gate predeclared before any run.                                                                                                                                                                                                                                                                                               |
| 2026-10-01 | D2a implementation: two buffer types (`D3D12_Host` default, `D3D12_Weights` extra) so weights skip the host buft via `supports_op`; Q4_0 / Q4_K / Q6_K kernels; D2a selftest gate predeclared.                                                                                                                                                                                                                      |
| 2026-10-02 | **D2a = PASS** (run 3, CI `1.6.0.1125`): 12/12 correct, every decode shape ≥ 102 GB/s with the width picked by K (64 threads below K = 4096, 128 from it). Runs 1–2 failed on speed and stay recorded.                                                                                                                                                                                                              |
| 2026-10-02 | **D2 = FAIL** (CI `1.6.0.1138`): decode 1.59× / 1.58×, prefill 1.32× / 1.74×, `validate-console.sh all` PASS on Coder-3B / LFM2.5-1.2B; peak RAM +539 / +265 MB over the CPU and Coder-3B H9 5/8 vs 6/8. The backend stays opt-in (`gguf_gpu_layers.txt`), default off. Tied lm_head on the CPU measured and dropped (1.2B decode 1.37×). Follow-ups #309 #310 #312 #313; 350M (0.95×) is never a candidate (#311). |
| 2026-10-01 | **D1 = `D2-matmul-only`** (CI `1.6.0.1117`): round trip 49.7 µs, simulated Coder-3B token 21.16 ms with sync, in-XAML PASS; weights in DEFAULT heaps (CPU-visible heaps 0.40×). D2 gates unchanged; measure D2 in the UI process too.                                                                                                                                                                               |
