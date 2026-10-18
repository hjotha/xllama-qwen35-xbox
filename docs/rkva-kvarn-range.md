# Native KVarN4 positional overflow

This future RKVA extension consumes **native Bee KVarN4 records**, rotated F16 stage rows, and an original-domain F16 precision-tail overlay. It does not decode and persist the cache as Q4_0, Q8_0, or F16. The host sends Q and receives OUT in the original domain, with natural-log LSE for the existing stable CUDA/RPC merge.

The implementation is isolated on `feat/rkva-kvarn4-range`, based on the uncommitted Q4/Q8 range implementation at xllama base `aadce35c267dff658e2080061cc453e7da427979`. It is a candidate for package 1.6.0.260. The live 1.6.0.259 console must remain untouched during the separate Q4/Q8 model benchmarks. No GPU parity, model quality, or model TPS claim is made for this candidate.

## Descriptor and records

The authoritative producer is the current Bee tree, particularly `src/llama-kvarn.cpp`, `src/llama-kv-cache-kvarn.cpp`, and `ggml/src/ggml-cuda/kvarn.cu`. The current cache store explicitly sets eager-record mode (`op_params[9] = 1`); the legacy Xbox delayed-store policy is not reused.

Capability bit 2 in HELLO_ACK enables native KVarN4. Operation 15 (`RKVA_KVARN_RANGE_CONFIG`) accepts a 72-byte descriptor; operation 16 (`RKVA_KVARN_RANGE_UPLOAD`) accepts native bytes. Existing operation 14 computes attention without appending. Existing wire formats and the Q4/Q8 paths retain their numbers.

Descriptor revision 1 requires:

- Origin aligned to 128 tokens, capacity within signed global-position bounds.
- Head dimension 256, two physical 128-dimensional slices per KV head.
- 128-token groups, K4/V4, 8960 bytes per physical-head record.
- Eager committed records, three stage slots, two rotating live slots, and a 128-token **global** sink.
- WHT revision 1: normalized H128 per slice, followed by the normalized two-slice Hadamard mix. Both stored K and V are rotated.
- Precision-tail length 0 or 128. The current Bee default needs 128, even when its requested tail argument is zero.
- Single dense contiguous sequence, ordinary causal attention, no SWA, no holes or context shift. Explicit producer stage maps must be reordered into the canonical transport slots.

Unknown versions, layouts, policies, geometries, and domains fail closed before allocation. The descriptor uses storage flag bit 2; the engine's internal storage index is 3.

Each native record contains 8192 bytes of packed low-nibble-first unsigned 4-bit values, followed by 128 F16 scales, 128 F16 zero points, and 128 F16 factors. K's payload indices are `[dimension][token]`; V's are `[token][dimension]`. Reconstruction is `(quant * scale + zero) * factor`. The older CPU _combined tile_ layout orders V's three axes differently, so it must be repacked explicitly. GPU record layout and combined CPU tile layout must never be conflated.

Compact record group `i` represents **global group `origin/128 + i`**. Only global group 0 is pinned as a sink. For origin 102400, compact group 0 is global group 800 and must be quantized when complete; it is not an extra exact sink. A stage row for global group `g` uses slot `g == 0 ? 0 : 1 + (g - 1) % 2`.

## Snapshot, append, and rewind

The 32-byte upload header contains live count, a contiguous record-group interval, flags, append position/count, and two zero reserved words. Payload planes are K records, V records, K stage rows, V stage rows, K original-domain tail rows, and V original-domain tail rows.

A RESET_SNAPSHOT starts at record group 0, carries all completed records, all 384 physical stage rows, and the most recent `min(live_count, tail_tokens)` tail rows in chronological order. Append position/count must be zero for this mode.

An ordinary append advances from the previous live count and uploads only new stage/tail rows plus newly completed record groups. Stage rows use `[physical KV head][128]`; tail rows use original `[KV head][256]`. The server maps stage rows to global canonical slots and the precision tail to a 128-row ring. Batched appends may wrap either ring and may cross several record boundaries. Every completed group must be covered before attention; future record groups and gaps are rejected. A decode append without a new record boundary transfers 8192 bytes per layer for 4 KV heads, plus its header. No full-stage snapshot is required per token.

RESET, TRIM, and REWIND invalidate every native layer's stage/record/tail coverage. Attention and non-reset appends are rejected until the host supplies a complete new snapshot. Old future records cannot become visible after a rewind. Restoring a native checkpoint likewise requires an explicit RESET_SNAPSHOT. Every layer has independent coverage and readiness.

The exact precision tail replaces the latest original-domain rows in the attention domain. It is not the rounded rotated F16 stage. The shader computes tail scores using original Q, body scores using WHT(Q), accumulates the body in rotated V, inversely transforms its numerator, then adds the original-domain tail numerator. The ordinary split-K reducer returns the merged OUT and LSE.

## Bee client integration and remaining gates

Package263 has passed root GPU tests against independent oracles for both tail settings. The original list below describes the client integration contract. The experimental Bee native producer is being integrated separately; full model measurements and native snapshot replay are independent gates:

1. Capability/descriptor negotiation for the appended operations, guarded by capability bit 2.
2. A genuine KVarN4 producer for overflow positions, with original F16 exact-tail ownership. It must not begin from already quantized Q4_0/Q8_0 data and label the result equivalent to an unquantized KVarN producer.
3. Native record extraction, explicit-stage-map canonicalization, global origin semantics, and dirty-group/row upload tracking.
4. RESET_SNAPSHOT after reset, trim, rewind, checkpoint restore, or epoch invalidation, before querying remote attention.
5. The existing original-domain OUT/LSE CUDA merge, with zero upload/attention RPC calls below position 102400.

Uploading already generated record bytes preserves the producer's quantization. CPU and GPU producers can still differ in balancing/reduction rounding; identical format names alone do not prove equal logits. First qualify the shader against the same bytes, then measure model log-probability divergence and full-context speed.

## Optimized decode kernel (263, raw GPU qualified)

`shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl` is an optional nq=1 decode kernel over the
same native buffers. It is selected only by a raw op 20 mode 1–3 (legacy field name
`q4_kernel_mode`, see [rkva-range-profiling.md](rkva-range-profiling.md)). The Bee host's
Q4-only opt-in gate does not reach it, and mode 0 (the default) keeps the reference kernel.

- **Native sources only.** Every element comes from its descriptor-v1 source: a Bee
  record (`precise(q*scale+zero)*other`, K scale/zero per dim, V per token), the rotated
  F16 stage slot, or the original-domain F16 precision tail. There is no Q4_0
  reinterpretation, no materialization and no renumbering.
- **Same math, different summation order.** Q is rotated with the same butterfly
  WHT256 as the reference. Body rows score and accumulate in the rotated domain and
  tail rows in the original domain. The split numerator is `WHT(body) + tail` with a
  shared max and denominator. The partial layout, the reducer contract, OUT/LSE and
  the causal limit are unchanged.
- **Work shape.** One 128-thread group serves the 6 Q heads of a KV head. Waves own
  whole rows or 8-token tiles. Tiles cover body rows inside complete record groups:
  one K dword carries 8 tokens of a dim, and one V dword carries 8 dims of a token.
  Sink and live stage rows, tail rows and unaligned split edges take a per-row path.
- **Gate.** In addition to the shared shape gate it requires `origin % 128 == 0` and
  `precision_tail_tokens` 0 or 128, and the host passes `split_rows` as a multiple of 8.
  A missing PSO marks `optimized_kernels.kvarn4.available=false`, and the reference
  kernel is used.

`tests/rkva_range_kvarn4_opt_model.py` checks it on the CPU, without a GPU:

- An FP64 oracle uses an explicit Sylvester H256/16 matrix and its own record decoder.
- Float32 emulations of the reference shader and the optimized shader use bounds-checked
  dword loads over NaN-poisoned buffers.
- K/V decode is bit-exact against the reference expression.
- Every row is covered exactly once.
- Wave sizes 32/64/128, signed scales and `other`, both tail settings, stage and group
  boundaries, masked rows and the 28672-row case are covered.
- Genuine Bee records come from the fixture directory, checked against Bee's own
  dequantizer and WHT.

`tests/rkva_range_format_kernel_probe.py --self-test` replays the live probe's own op
15/16 payloads into an independent image of the engine's buffer placement and compares
its oracle with the model's. The live A/B run (`--execute`) is for the console lease
holder only.

## Verification and future console gate

The source includes a standalone CPU contract test and a fixture decoder/domain test. `rkva_kvarn_fixture_bee.cpp` must be compiled against the intended Bee checkout's `src/llama-kvarn.cpp` and ggml-base, then run with an output-directory argument. It emits native records, input/dequantized tiles, and WHT reference vectors. No model weights are needed.

The CPU gate checks descriptor rejection, eager record coverage, efficient append mappings, tail wrap, and rewind resynchronization. The fixture decoder compares every reconstructed K/V value with Bee's own CPU dequantizer and checks the exact WHT ordering. Mixed-domain causal attention is compared against independently materialized original-domain attention. These tests are not GPU execution.

For a reserved console running the future candidate, execute the separate gate explicitly:

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/rkva_kvarn_range_probe.py \
  --execute --host 192.168.1.26 --fixtures /path/to/bee-fixtures \
  --output /path/to/kvarn4-console-gate.json
```

Without `--execute`, the probe performs no network access. The future gate exercises positions before/at/after the overflow origin, eager commit boundaries, stage/tail ring wrap, batched appends, split-K OUT/LSE, and full resynchronization after trim/reset. Follow it with full model correctness and TPS measurements before using or deploying the feature. This staged work does not qualify speculative decoding, multiple sequences, noncausal attention, or arbitrary tail lengths.
