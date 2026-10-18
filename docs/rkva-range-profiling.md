# RKVA range-attention profiling and kernel modes

This page covers the Xbox-side range-attention engine (`uwp/rkva/remote-attn-engine.*`).
It describes two opt-in features: per-stage D3D12 timestamp profiling, and selectable
optimized decode kernels (Q4_0, plus future F16, Q8_0 and native KVarN4). Both are off by default. While they stay off, the engine records
the same commands as before. The wire control (op 20, `RKVA_RANGE_PROFILE`) and the
transport counters belong to `uwp/rkva/remote-attn.cpp` and `remote-attn-protocol.h`.

Status: the profiling and Q4 kernels were compiled for the isolated 1.6.0.261/262
candidates; root qualified them on a console as synthetic one-layer timings, not model
tokens/s. The F16, Q8_0 and KVarN4 kernels (future 1.6.0.263 candidate) are checked
only by CPU models and DXIL validation. They have never run on a GPU. Do not read any
of this page as a performance claim.

## Engine interface

```cpp
bool set_range_profiling(bool enabled, bool reset, uint32_t q4_kernel_mode, std::string& error);
std::string get_range_profile_json();
```

- `set_range_profiling` rejects the call and changes nothing when any of these hold:
  - the engine is not ready;
  - a submission is still unconfirmed;
  - the session is in terminal failure;
  - the mode is invalid;
  - the mode is ≥1 and the optimized pipelines are unavailable;
  - the partials buffer for the optimized kernel cannot be allocated.

  The caller must invoke it while the engine is idle, under the existing serialized
  RPC dispatch.

- `reset` clears the sample ring and the counters.
- `enabled` starts or stops sampling.
- `q4_kernel_mode` takes effect from the next attention call. The name is legacy
  (see [Per-storage kernels](#per-storage-kernels-future-263)); the value is a
  storage-independent kernel mode.
- `get_range_profile_json` is read-only. It returns the schema
  `rkva-range-profile-v1`. The helper namespace is `rkva_rprof`, in
  `uwp/rkva/rkva-range-profile.h`. That header is WinRT-free, so the host can test it.

## Q4 kernel modes

| Mode | Name              | Rows per split | Max splits | Notes                    |
| ---- | ----------------- | -------------- | ---------- | ------------------------ |
| 0    | `reference`       | 256            | 32         | live259 shader (default) |
| 1    | `q4_gqa_balanced` | 128            | 128        | optimized                |
| 2    | `q4_gqa_fine`     | 32             | 256        | optimized                |
| 3    | `q4_gqa_coarse`   | 512            | 64         | optimized                |

The reference split count is `min(32, ceil(active_count/256))`. The reference dispatch is
`(n_head, n_tokens, split_k)`.

The optimized kernel is `shaders/rkva/rkva_range_attn_q4_opt.hlsl`, dispatched as
`(n_head_kv, n_tokens, split_k)` with 128-thread groups:

- **GQA sharing.** One group serves all 6 Q heads that read one KV head. Each Q4_0 K/V row
  is loaded and dequantized once per KV head, not once per Q head.
- **Wave-owned rows.** Each wave owns whole rows and reduces with `WaveActiveSum`. There
  are no group barriers inside the row loop. Waves merge their online-softmax states once,
  at the end.
- **Dword loads.** Q4_0 data is read with 4-byte loads, 4 elements per lane-quarter. The
  values are identical to the reference dequantization, `float(q - 8) * f16 scale`.
- **Splits.** When `split_k == 1`, the kernel writes the final O and LSE directly and skips
  the reducer. Otherwise `rkva_range_reduce_opt.hlsl` merges up to 256 splits.

The host routes a call to the optimized path only when all of these hold:

- the mode is ≥1;
- storage is Q4_0;
- `1 <= n_tokens <= 9` (nq 2..9 use the multi-query tiles below in modes 1 and 2);
- `head_dim == 256`;
- `n_head == 6 * n_head_kv`;
- the device reports wave ops with 32–128 lanes.

Every other call uses the reference kernel. Causal positions, precision, and the
OUT/LSE layout are unchanged. Results can differ from the
reference only in FP32 summation order. The numpy model
`tests/rkva_range_q4_opt_model.py` checks the lane mapping, the load equivalence and the
split math against a float64 oracle.

## Q4 multi-query tiles (266, MTP verify)

An MTP verify call attends `nq = N + 1` contiguous causal queries in one layer (N2 →
nq3, N4 → nq5). The per-query kernel above dispatches every query as its own group
(`Dispatch.y = n_tokens`), so it reloads and re-dequantizes every K/V row per query and
does a per-row `WaveActiveSum` for each query and head. The 266 source adds
`shaders/rkva/rkva_range_attn_q4_mq.hlsl` (entry `CSRangeAttnQ4Mq`), which is compiled
twice with `RKVA_MQ_QT` = 2 and 4 to `rkva_range_attn_q4_mq2_dxil.h` and
`rkva_range_attn_q4_mq4_dxil.h`.

- **Dispatch** `(n_head_kv, ceil(nq / QT), split_k)` with 64-thread groups. One group
  serves 6 GQA heads × QT queries of one KV head and split.
- **Score phase.** Rows are walked in 64-row tiles, and thread `i` owns row `tile + i`.
  Each thread loads and dequantizes its K row **once** and dots it with all 6·QT query
  vectors, with no cross-lane reduction per row. The per-block f16 scale is factored out
  of each 32-element block sum. The values are exact and only the summation order
  changes.
- **Online softmax per (query, head)** keeps one tile max (wave max plus a cross-wave
  merge in LDS) and one rescale per tile, not per row. Each query masks its own causal
  limit `min(active, position − origin + 1)`, so a tile shared by queries with different
  limits stays exact.
- **V phase.** Thread `i` owns 4 output dims of the 256. Each V row is loaded and
  dequantized once per group and accumulated into all 6·QT outputs from the LDS
  probabilities.
- **Output** is the same `[nq][n_head][256]` OUT and `[nq][n_head]` LSE. When
  `split_k == 1` the kernel writes them directly; otherwise it writes the per-query
  partials that the unchanged `rkva_range_reduce_opt.hlsl` merges. LSE is
  `max + log(denominator)`, or −inf for an empty prefix.
- **Splits.** `q4_mq_split` takes the mode's split policy and rounds the rows up to whole
  64-row tiles, so its `split_k` is never above the per-query one and fits the bounded
  partials. At 58128 live rows mode 1 gives 114 × 512 and mode 2 gives 228 × 256.

The selection keeps the wire field name `q4_kernel_mode`:

| Mode | nq1 (unchanged)             | nq 2..9                                  |
| ---- | --------------------------- | ---------------------------------------- |
| 0    | `reference`                 | `reference`                              |
| 1    | per-query `q4_gqa_balanced` | **`q4_mq4_balanced`** (4-query tile)     |
| 2    | per-query `q4_gqa_fine`     | **`q4_mq2_fine`** (2-query tile)         |
| 3    | per-query `q4_gqa_coarse`   | per-query `q4_gqa_coarse` (A/B baseline) |

A mode-1/2 call whose tile PSO failed to build takes the per-query kernel, which is also
correct. The per-query DXIL, the nq1 path, mode 0, and every F16/Q8_0/KVarN4 path are
unchanged. The report adds a `q4_multi_query` object with the mode map, availability,
status and call count of each tile PSO. Bucket names use the kernel that actually ran,
for example `attention/q4_0/optimized-q4_mq4_balanced/nq2-16/r32k+/warm`.

**Resources** (DXC `cs_6_0`, validation on, no allocas): LDS 3168 B (QT2) and 6336 B
(QT4); DXIL 39432 and 72184 bytes. Estimated VGPRs, not measured because no AMD ISA dump
is available, are about 90 (QT2) and about 150 (QT4). Most of that is the 6·QT float4
accumulators. QT4 may halve occupancy relative to QT2, which is why both tiles are built.
Two group barriers per 64-row tile.

**Checks.** `tests/rkva_range_q4_mq_model.py` checks the following against an
independent per-query float64 oracle:

- the load equivalence with signed f16 scales;
- row and dim ownership at wave 32/64/128;
- the partials layout;
- the split invariants;
- an FP32 tile emulation with nq 1/2/3/5/9, causal, masked, future, unsorted and
  partial-tile cases, up to 58368 live rows.

`--self-test` proves that each injected bug is detected. The gated live probe is
`tests/rkva_range_q4_mq_probe.py`; without `--execute` it prints its plan only.

These are one-layer kernel checks. The emulation is not GPU timing, and no performance
claim is made until root measures model tokens/s on the console.

## F16 / Q8_0 multi-query tiles (267, GPU qualification pending)

The 267 source ports the Q4 tile algorithm above to the F16 and Q8_0 storages.
`shaders/rkva/rkva_range_attn_fmt_mq.hlsl` (entry `CSRangeAttnFmtMq`) is compiled with
`RKVA_OPT_STORAGE` = 0 (F16, 512-byte rows) or 2 (Q8_0, 272-byte rows) and `RKVA_MQ_QT`
= 2 or 4, giving four headers:

- `rkva_range_attn_f16_mq2_dxil.h`
- `rkva_range_attn_f16_mq4_dxil.h`
- `rkva_range_attn_q8_mq2_dxil.h`
- `rkva_range_attn_q8_mq4_dxil.h`

Dispatch, 64-row tiles, row ownership in the score phase, the online softmax per
(query, head) with each query's own causal limit, the V phase, the OUT/LSE layout, the
partials for `rkva_range_reduce_opt.hlsl` and `q4_mq_split` are all the Q4 tile kernel's.
Only the decode differs:

- **F16 score.** Each thread reads its K row as 16-byte loads and converts each half
  with `f16tof32` once for all 6·QT dots.
- **Q8_0 score.** One 68-byte pair of blocks is read as five aligned loads, because
  blocks are 34 bytes and only 2-byte aligned. The int8 values are sign-extended. The
  signed f16 block scale is factored out of each 32-element block sum. `int8 × f16` is
  exact in FP32, so only the summation order changes.
- **V phase.** Each thread uses the per-query `load_unit` of `rkva_range_attn_gqa_opt.hlsl`
  for its 4 dims, once per row for all 6·QT outputs.

The engine creates these PSOs on the format-optimized root signature after the
per-query F16/Q8_0 PSOs. A failure is non-fatal and records
`<fmt> multi-query pipeline state: 0x…`. A tile PSO is available only when the per-query
kernel of its storage is also available.

| Mode | F16 / Q8_0 nq1 (unchanged) | F16 / Q8_0 nq 2..9                          |
| ---- | -------------------------- | ------------------------------------------- |
| 0    | `reference`                | `reference`                                 |
| 1    | `<fmt>_gqa_balanced`       | **`<fmt>_mq4_balanced`** (4-query tile)     |
| 2    | `<fmt>_gqa_fine`           | **`<fmt>_mq2_fine`** (2-query tile)         |
| 3    | `<fmt>_gqa_coarse`         | per-query `<fmt>_gqa_coarse` (in-build A/B) |

`<fmt>` is `f16` or `q8`. A missing tile PSO falls back to the per-query kernel. The
Q4 tile kernels, Q4 mode map, nq1 paths, mode 0, the per-query DXIL and every KVarN4
path, including descriptor 3, are unchanged. All existing DXIL headers regenerate
byte-identically.

The report adds `format_multi_query` with `nq_min`, `nq_max`, `tile_rows`, `mode_map`, and
for `f16` and `q8_0` the `mq2`/`mq4` `available`, `status` and `calls` fields. Bucket
names use the kernel that actually ran, for example
`attention/q8_0/optimized-q8_mq4_balanced/nq2-16/r32k+/warm`. The sample's `pass1_groups`
counts `n_head_kv × ceil(nq / QT) × split_k`.

**Resources** (DXC `cs_6_0`, validation on, no allocas): LDS 3168 B (QT2) and 6336 B
(QT4). DXIL sizes are F16 31216/56340 and Q8_0 39144/71888 bytes (QT2/QT4). VGPR use is
not measured because no AMD ISA dump is available. QT4 holds 24 float4 accumulators plus
a decoded 32-value block, so it may spill or lose occupancy; QT2 exists for that reason.

**Checks.** `tests/rkva_range_fmt_mq_model.py` replays the shader loads on the real F16
and Q8_0 bytes and checks them against the reference decode, including bounds and
unaligned Q8_0 scales. It also checks row and dim ownership at wave 32/64/128 and the
split and partials layout. Its FP32 tile emulation runs against a per-query float64
oracle with signed scales, outliers, nq 1/2/3/5/9, causal, masked, future, unsorted and
partial-tile cases, live 1 to 58368. `--self-test` injects load bugs (misaligned Q8_0
scale, swapped scales, swapped halves) and kernel bugs (union limit, off-by-one, missing
rescale) and requires every one to be detected. The gated live probe is
`tests/rkva_range_fmt_mq_probe.py --storage f16|q8`; without `--execute` it prints its
plan only.

**Host integration.** The generic Bee `GGML_RKVA_RANGE_KERNEL` opt-in already
selects storage-specific optimized kernels after positive availability validation.
The historical `GGML_RKVA_Q4_KERNEL` setting remains Q4-only. These new tiles
need independent GPU and model qualification; the FP32 emulation is not a timing
result. No protocol or client extension is needed.

**Native KVarN4 descriptor 3.** No tile kernel is added. Near a 128-row seal, a row
can be a sealed record for one query and a stage or precision-tail row for an earlier
query. A shared tile may therefore only reuse rows read as records by every query in
the tile, and must read every other row with each query's own serial-prefix rules.
`tests/rkva_kvarn4_mq_partition_model.py` checks that partition. The handoff
`HANDOFF-FORMATS-MQ267.md` holds the plan. The descriptor-3 optimized prefix route is
unchanged.

## Per-storage kernels (future 263)

The wire field and engine argument keep the legacy name `q4_kernel_mode` so op 20 and
its payload layout do not change. In the future 263 source, modes 1, 2 and 3 select the
optimized kernel of **every** storage that has one available. Each storage keeps the
same split policy:

| Storage  | Mode 1 / 2 / 3 kernel name                                      | Shader                            |
| -------- | --------------------------------------------------------------- | --------------------------------- |
| `q4_0`   | `q4_gqa_balanced` / `q4_gqa_fine` / `q4_gqa_coarse`             | `rkva_range_attn_q4_opt.hlsl`     |
| `f16`    | `f16_gqa_balanced` / `f16_gqa_fine` / `f16_gqa_coarse`          | `rkva_range_attn_gqa_opt.hlsl`    |
| `q8_0`   | `q8_gqa_balanced` / `q8_gqa_fine` / `q8_gqa_coarse`             | `rkva_range_attn_gqa_opt.hlsl`    |
| `kvarn4` | `kvarn4_gqa_balanced` / `kvarn4_gqa_fine` / `kvarn4_gqa_coarse` | `rkva_range_attn_kvarn4_opt.hlsl` |

Mode 0 is the reference kernel for every storage. The Q4_0 paths, names, DXIL and split
policy are byte-for-byte those of 261/262.

- **F16 / Q8_0** (`RKVA_OPT_STORAGE` 0 or 2). The structure is the Q4 kernel's: one
  128-thread group per (KV head, query, split) serves the 6 Q heads of that KV head,
  waves own whole rows, there are no barriers in the row loop, and online-softmax
  states merge once. Loads are 8-byte (F16, 4 halves) or 4-byte (Q8_0, 4 int8 values)
  per lane unit; Q8_0 blocks are 34 bytes, so the dword loads handle the 2-byte
  misalignment explicitly. The decoded values are identical to the reference
  (`f16tof32`, or `float(int8) * f16 scale`).
- **KVarN4** reads the genuine descriptor-v1 buffers directly: Bee records, the
  rotated F16 stage and the original-domain precision tail. Nothing is reinterpreted
  as Q4_0, materialized or renumbered (see [rkva-kvarn-range.md](rkva-kvarn-range.md)).
  Rows inside complete record groups run as aligned 8-token tiles; stage, tail and
  split-edge rows take a per-row path with the reference decode. The host passes
  `split_rows` as a multiple of 8 (`kvarn4_split`), and the gate also requires
  `origin % 128 == 0` and `precision_tail_tokens` 0 or 128.

Shared gate: mode ≥1, `1 <= n_tokens <= 9`, `head_dim == 256`, `n_head == 6 * n_head_kv`,
wave size 32–128, and the storage's PSO plus the optimized reducer and partials buffer
are present. Otherwise the call takes the correct reference kernel. A PSO that fails to
build marks only its own storage unavailable, with the HRESULT in `optimized_kernels.<storage>.status`.
`set_range_profiling` still rejects modes ≥1 when the Q4 optimized pipelines are
missing, as in 261.

**Host integration.** `GGML_RKVA_RANGE_KERNEL` selects these storage-specific
optimized paths after positive capability validation. The historical
`GGML_RKVA_Q4_KERNEL` alias is restricted to Q4. No additional protocol field
is needed. GPU kernel and model measurements remain separate qualification.

## Timing semantics

- **GPU stages.** D3D12 timestamp queries (`EndQuery` on the direct queue) are resolved
  into a readback ring with 4 sets of 8 slots. They are read after the fence wait the call
  already performs. Profiling adds no waits.

  | Call      | Stages                                                    |
  | --------- | --------------------------------------------------------- |
  | Upload    | `copy`                                                    |
  | Attention | `pass1`, `reduce` (0 on the direct path), `readback_copy` |

  Durations are tick deltas divided by `gpu_timestamp_frequency_hz`, the value
  `GetTimestampFrequency` reports. Timestamps are end-of-pipe: the cost of a barrier
  counts toward the stage before it. The engine discards a set that is zero or non-monotonic (so `gpu_samples` <
  `samples`); `timestamp_map_failures` counts only readback Map failures. If the query heap is unavailable,
  `gpu_timestamps=false` and only CPU spans are reported.

- **CPU spans.** These are `std::chrono::steady_clock` wall time and are not GPU time.
  The spans are disjoint and their sum is `engine_total`:

  | Span         | What it covers                                       |
  | ------------ | ---------------------------------------------------- |
  | `validate`   | argument and session checks                          |
  | `alloc`      | grow-only buffer creation; the call is marked `cold` |
  | `map_copy`   | Map, memcpy and Unmap of the inputs                  |
  | `record`     | allocator and list reset, then command recording     |
  | `submit`     | Close, ExecuteCommandLists and Signal                |
  | `fence_wait` | blocking wait: GPU execution plus queue latency      |
  | `readback`   | attention outputs only                               |

  `profile_overhead_us` is the bookkeeping after the work and is excluded from
  `engine_total`.

- **Transport metrics.** The existing transport `compute_us` is CPU wall time around the
  whole engine call. `prepare_us` includes `compute_us`, so do not add the two.

## JSON report

The header fields are:

- `profiling`, `q4_kernel_mode` (legacy name), `q4_kernel_mode_name` (the Q4 kernel name)
- `gpu_timestamps`, `gpu_timestamp_frequency_hz`, `gpu_timestamp_status`
- `q4_optimized_available`, `q4_optimized_status`, `wave_lane_min` / `wave_lane_max`
- `sample_capacity` (16384), `samples_recorded`, `samples_in_ring`, `samples_overwritten`
- `kernel_mode_scope` (text describing the legacy mode naming) and
  `optimized_kernels.{f16,q4_0,q8_0,kvarn4}` with `available`, `status`,
  `optimized_calls` and `reference_calls` (attention calls per storage, while profiling)
- `kvarn4_serial_prefix.{reference,optimized}` with `available` and `status` for the
  native KVarN4 descriptor revision 3 (serial-prefix) PSOs; see
  [rkva-kvarn-range.md](rkva-kvarn-range.md)

`counters` are engine-wide. They also count legacy and snapshot submits:

- calls: `upload_calls`, `attention_calls`, `optimized_q4_calls`, `reference_q4_calls`
  (Q4_0 only, as in 261), `failed_calls`
- queue: `submits`, `fence_blocking_waits` and `fence_already_complete`
- work: `dispatches`, `copy_regions`
- allocation: `buffer_creates` and `buffer_create_bytes`
- timestamps: `timestamp_resolves`, `timestamp_map_failures`

`buckets` keys follow `kind/storage/[kernel/]nq-band/rows-band/warm|cold`:

- `storage` is `kvarn4-prefix` for revision-3 native sessions, so serial-prefix
  calls never mix with legacy resident-image buckets.
- `kernel` appears on attention keys only: `reference` or `optimized-<kernel name>`,
  for example `optimized-q4_gqa_balanced` or `optimized-kvarn4_gqa_fine`.
- nq bands: `nq1`, `nq2-16`, `nq17-128`, `nq129+`.
- rows bands (upload rows, or attention `active_count`): `r0`, `r1-256`, `r257-1k`,
  `r1k-4k`, `r4k-16k`, `r16k-32k`, `r32k+`.
- `cold` means the call grew a buffer.

Each bucket holds `samples`, `gpu_samples`, `fence_blocked`, `alloc_grew` and
`direct_output`. For each of `n_tokens`, `rows`/`active_rows`, `bytes`, `split_k`,
`pass1_groups`, every `gpu_us` stage plus `total`, every `cpu_us` span plus
`engine_total`, and `profile_overhead_us`, it reports n, mean, min, p50, p90, p99, max
and sum. Percentiles are computed over the ring.

Warm decode at near-128k context falls in `attention/<storage>/<kernel>/nq1/r16k-32k/warm`.
Prefill and the first cold Q256 probe land in other buckets.

## Usage via op 20 (root wiring)

Payload `rkva_range_profile {flags, q4_kernel_mode, reserved}`:

| Flag        | Bit | Meaning                                       |
| ----------- | --- | --------------------------------------------- |
| enable      | 1   | start sampling                                |
| reset       | 2   | clear the ring and counters                   |
| trace       | 4   | transport debug only                          |
| report_only | 8   | return the report; must be alone, with mode 0 |

The response is `{"engine": <this report>, "transport": {...}}` and is capped at 1 MB.

A typical A/B run:

1. Send `enable|reset` with mode 0, run warm decode, then send `report_only`.
2. Repeat with modes 1, 2 and 3.
3. Compare `gpu_us.pass1` + `reduce` and `cpu_us.fence_wait` in the warm nq1 bucket.
4. Check OUT/LSE parity against mode 0 before you select a default.

The probe script `tests/rkva_range_q4_kernel_probe.py` stays offline unless you pass
`--execute`. Its `warm_nq1` field filters on `optimized-<mode>` and is therefore empty
for modes 1–3; the full `report` it stores is correct.

`tests/rkva_range_format_kernel_probe.py` (future 263) covers all four storages:

```bash
python3 tests/rkva_range_format_kernel_probe.py                  # offline plan
python3 tests/rkva_range_format_kernel_probe.py --self-test      # offline KVarN4 upload/oracle replay
python3 tests/rkva_range_format_kernel_probe.py --summarize RAW.json [--output NEW.json]
python3 tests/rkva_range_format_kernel_probe.py --execute --host HOST \
  --formats f16 q8_0 q4_0 kvarn4 --output NEW-RAW.json           # console lease holder only
```

- `--execute` checks parity against a float64 oracle and against mode 0, and checks the
  per-storage kernel selection for every mode. It then runs a warm decode per mode and
  restores mode 0 with profiling off.
- The KVarN4 data uses genuine Bee records (from `--bee-fixtures`) mixed with synthetic
  signed records, the rotated stage and the precision tail.
- `--summarize` reads raw reports from either probe with the correct
  `optimized-<kernel name>` filter. It never writes over its input or an existing
  `--output`.

CPU models (no GPU):

```bash
python3 tests/rkva_range_q4_opt_model.py
python3 tests/rkva_range_fmt_opt_model.py [--quick]   # default includes 28672 rows
python3 tests/rkva_range_kvarn4_opt_model.py [--quick]
```

## Root console qualification, 2026-10-18

Package 1.6.0.263 (SHA256 `d14f0732ab1bd034dd84dce220bd99849eddd50751e77caa6d75d2fa0438188e`) passed 108 independent OUT/LSE/selection cases per format (F16, Q8_0, KVarN4, Q4_0), capacity 28672, all four modes and precision tail 128. Native KVarN4 tail 0 also passed. Combined upload-attention probes passed for Q4/Q8/F16, including nq256 and rejected-frame no-mutation checks.

Raw artifacts and exact invocation/provenance are kept in `/home/hjotha/build/rkva-q4-optimization-20261018/console263/`; host models and shader compilation are separate evidence. These one-layer probes are not model TPS and do not qualify model quality or native snapshot replay. Default mode 0 remains reference; optimized modes require opt-in.
