# RKVA range-attention profiling and Q4 kernel modes

This page covers the Xbox-side range-attention engine (`uwp/rkva/remote-attn-engine.*`).
It describes two opt-in features: per-stage D3D12 timestamp profiling, and selectable
Q4_0 decode kernels. Both are off by default. While they stay off, the engine records
the same commands as before. The wire control (op 20, `RKVA_RANGE_PROFILE`) and the
transport counters belong to `uwp/rkva/remote-attn.cpp` and `remote-attn-protocol.h`.

Status: compiled for the isolated 1.6.0.261 candidate only. Nothing on this page
has been measured on a console. Do not read any of it as a performance claim.

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
- `q4_kernel_mode` takes effect from the next attention call.
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

Every other call uses the reference kernel. Q8_0, F16 and KVarN4 behaviour, causal
positions, precision, and the OUT/LSE layout are unchanged. Results can differ from the
reference only in FP32 summation order. The numpy model
`tests/rkva_range_q4_opt_model.py` checks the lane mapping, the load equivalence and the
split math against a float64 oracle.

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

- `profiling`, `q4_kernel_mode`, `q4_kernel_mode_name`
- `gpu_timestamps`, `gpu_timestamp_frequency_hz`, `gpu_timestamp_status`
- `q4_optimized_available`, `q4_optimized_status`, `wave_lane_min` / `wave_lane_max`
- `sample_capacity` (16384), `samples_recorded`, `samples_in_ring`, `samples_overwritten`

`counters` are engine-wide. They also count legacy and snapshot submits:

- calls: `upload_calls`, `attention_calls`, `optimized_q4_calls`, `reference_q4_calls`,
  `failed_calls`
- queue: `submits`, `fence_blocking_waits` and `fence_already_complete`
- work: `dispatches`, `copy_regions`
- allocation: `buffer_creates` and `buffer_create_bytes`
- timestamps: `timestamp_resolves`, `timestamp_map_failures`

`buckets` keys follow `kind/storage/[kernel/]nq-band/rows-band/warm|cold`:

- `kernel` appears on attention keys only: `reference` or `optimized-<mode>`.
- nq bands: `nq1`, `nq2-16`, `nq17-128`, `nq129+`.
- rows bands (upload rows, or attention `active_count`): `r0`, `r1-256`, `r257-1k`,
  `r1k-4k`, `r4k-16k`, `r16k-32k`, `r32k+`.
- `cold` means the call grew a buffer.

Each bucket holds `samples`, `gpu_samples`, `fence_blocked`, `alloc_grew` and
`direct_output`. For each of `n_tokens`, `rows`/`active_rows`, `bytes`, `split_k`,
`pass1_groups`, every `gpu_us` stage plus `total`, every `cpu_us` span plus
`engine_total`, and `profile_overhead_us`, it reports n, mean, min, p50, p90, p99, max
and sum. Percentiles are computed over the ring.

Warm decode at near-128k context falls in `attention/q4_0/<kernel>/nq1/r16k-32k/warm`.
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
`--execute`.
