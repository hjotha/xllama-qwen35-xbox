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
- `n_tokens == 1`;
- `head_dim == 256`;
- `n_head == 6 * n_head_kv`;
- the device reports wave ops with 32–128 lanes.

Every other call uses the reference kernel. Causal positions, precision, and the
OUT/LSE layout are unchanged. Results can differ from the
reference only in FP32 summation order. The numpy model
`tests/rkva_range_q4_opt_model.py` checks the lane mapping, the load equivalence and the
split math against a float64 oracle.

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

Shared gate: mode ≥1, `n_tokens == 1`, `head_dim == 256`, `n_head == 6 * n_head_kv`,
wave size 32–128, and the storage's PSO plus the optimized reducer and partials buffer
are present. Otherwise the call takes the correct reference kernel. A PSO that fails to
build marks only its own storage unavailable, with the HRESULT in `optimized_kernels.<storage>.status`.
`set_range_profiling` still rejects modes ≥1 when the Q4 optimized pipelines are
missing, as in 261.

**Scope: raw probes only.** The Bee host keeps its Q4-only opt-in gate, so model runs
never select the F16, Q8_0 or KVarN4 kernels. Only a raw RKVA client that sends op 20
with a nonzero mode (the probe below) reaches them. Exposing them to model runs needs
a per-storage opt-in field that root owns in the protocol and the Bee client.

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

`counters` are engine-wide. They also count legacy and snapshot submits:

- calls: `upload_calls`, `attention_calls`, `optimized_q4_calls`, `reference_q4_calls`
  (Q4_0 only, as in 261), `failed_calls`
- queue: `submits`, `fence_blocking_waits` and `fence_already_complete`
- work: `dispatches`, `copy_regions`
- allocation: `buffer_creates` and `buffer_create_bytes`
- timestamps: `timestamp_resolves`, `timestamp_map_failures`

`buckets` keys follow `kind/storage/[kernel/]nq-band/rows-band/warm|cold`:

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
