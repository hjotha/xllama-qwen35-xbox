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
- **Wave64 decode specialization (`CSRangeAttnKvarn4Decode`).** When compiled with
  `RKVA_DECODE_OPT=1`, the kernel specializes for Xbox Series X's 64-lane waves (2 waves per
  128-thread group), cutting LDS from 24 KiB to 12 KiB to double workgroup occupancy per CU,
  eliminating dead wave32 second-unit registers, ping-ponging WHT256 (9 barriers vs 17),
  hoisting record-group scale/zero/other metadata across 128-token groups, and using
  record-group-aligned split-K (`kvarn4_decode_split`).

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

## Bounded verification (264, raw GPU qualified)

The engine candidate also selects existing optimized kernels for query batches
of one through nine rows (eight MTP drafts plus their verification anchor).
Independent query workgroups already read their own causal position and write
separate OUT/LSE and split partials; no HLSL or generated DXIL change is required.
Larger batches retain reference attention. Op20 reports `fast_nq_max` for every
available optimized storage kernel; older reports without the field promise
only nq1. The shared bounded optimized partial workspace reserves about 54.4 MiB
for H24/D256 at nine queries and 256 splits. Package 1.6.0.264 passed raw GPU OUT/LSE and causal query-index validation.
Full-model MTP correctness and throughput are measured separately.

The raw format probe covers nq1/2/3/4/7/8/9 and nq10 fallback, with distinct
per-query positions, fully masked queries, independent OUT/LSE oracles and
mode0 comparisons. Native KVarN4 MTP remains unsupported in Bee until its live
native owner can roll back and restore correctly; raw multiquery kernel support
does not establish that state contract.

Package264 SHA256 `9902eb3ca173866242fbf2fd497262b0c8ce8d589fe8155ba0d869e858a051df` was built on Windows193 with SDK26100/MSVC14.44 and deployed in place. The float64 oracle probe passed 192 cases per format (F16, Q8_0, Q4_0 and native KVarN4) with precision-tail128, plus 192 native KVarN4 cases with tail0: 960 cases total, all four kernel modes, nq1/2/3/4/7/8/9 and nq10 reference fallback, checkpoints1/127/128/129/257/28672. These are kernel/protocol results, not model TPS or native-KVarN4-MTP qualification. Artifacts: `/home/hjotha/build/rkva-q4-optimization-20261018/console264`.

## Bounded resident rollback candidate (not console qualified)

Capability bit 6 (`RKVA_CAP_KVARN_REWIND`) and operation 21 (`RKVA_KVARN_RANGE_REWIND`) jointly negotiate native resident rollback. Both are required, together with client HELLO flag bit16 (`RKVA_PROTO_NATIVE_REWIND_V2`); native record support alone continues to mean append-only operation. Older clients do not receive the new capability or opcode advertisement, and keep their empty CONFIG/op16 ACKs. New clients retain descriptor-v1 append-only compatibility with older servers and reject active MTP there. This capability does not enable native FULL snapshots, disk persistence, arbitrary edits, or importing a Q4/Q8 overflow checkpoint.

The host retains the latest 256 original F32 rows per layer and permits removal of at most 128 rows. Bee limits draft-MTP depth to 127 because checkpoint fallback can restore before the verification anchor (N+1 rows); the 128-row direct-RS boundary is not promoted separately. It rebuilds the affected incomplete group with the canonical original-F32 native store and restores the original-domain precision tail from those retained inputs. Completed records below the retained prefix are unchanged. A removed completed group becomes an incomplete stage group; future rows remain masked until overwritten and resealed. Two global stage ping-pong slots retain the needed prefix at this rewind bound. No Q4 decoding or dequantize/requantize path participates.

The operation carries a 24-byte `expected_generation/pos0/n_layers/tail_tokens/reserved` header, followed by an 8-byte `expected_live/new_live` header and chronological K/V original-domain F16 tail rows for each layer. Only overwritten ring rows in `[max(0,new_live-tail_tokens), min(new_live,max(0,expected_live-tail_tokens)))` are carried; overlapping prefix rows stay resident. A depth-one rejection normally repairs one row per layer. Descriptor tail lengths are 0 or 128. Bee selects Xbox precision tail independently with `GGML_RKVA_NATIVE_TAIL_TOKENS=0|128` (default 128), recorded as `native_precision_tail_tokens` in host stats. Tail 0 omits both upload and rewind tail planes; original-F32 retention and native canonical resealing remain enabled. All layer counts, distances, readiness, mutation generation, payload sizes and reserved fields are validated before any writes. Descriptor version 2 retains the same native record layout and returns an 8-byte generation seed in CONFIG ACK; version 1 keeps its empty ACK. Process-lifetime atomic configuration epochs reserve nonoverlapping 32-bit mutation-counter ranges, including engine/HELLO/session recreation and reconfiguration. Exhaustion fails closed. Uploads, RESET, legacy TRIM/REWIND and operation 21 advance the session generation; overflow fails closed. Old operation 21 payloads are refused after same-count rewind/reappend (ABA). One command list repairs all tail rings, one fence precedes count publication, and one ACK publishes the transaction. A failed or malformed ACK poisons the host connection; GPU submission failure invalidates the server session.

Native PARTIAL checkpoints use the existing owner-bound hybrid attention envelope, cell identity and dirty-prefix lineage, plus a live native-prefix/retained-input coverage preflight before CPU/recurrent mutation or staged restore. A second rewind can be refused even within 128 tokens if its older originals have already fallen outside retention. A prefix anchor before a rejected suffix remains valid; a checkpoint containing rewritten rows does not become valid merely because token IDs and cell IDs match again. Remote epoch invalidation cannot be repaired by Q4 bookkeeping. Full clear uses RESET and releases the host native owners after ACK. Legacy TRIM/REWIND operations retain their full native invalidation semantics; only negotiated operation 21 provides the bounded resident contract.

Qualification requires the focused CPU native producer, wire parser, TCP ACK/failure and real hybrid-memory tests, followed by a Windows build, live native rollback OUT/LSE probes at 127/128/129 and tail0/128, and full-model MTP token parity and complete-cycle throughput. Source and CPU results alone do not establish Xbox or MTP performance qualification.

## Serial-prefix descriptor revision 3 (266 candidate, not GPU qualified)

Revisions 1 and 2 compute every query of a batch over the **resident image**. Their shaders select the staged partial group, the sealed-record bound and the 128-row precision-tail window from `active_count`, and the tail ring holds 128 rows. A batched MTP verification (nq 2..9) does not match serial decoding in two cases:

- A batch crosses a 128-token seal. An earlier query can then read the newly sealed record instead of its own staged partial group.
- The 128-row ring overwrote the first `nq-1` exact rows that an earlier query still needs.

Revisions 1 and 2 keep these exact semantics. Their PSOs, DXIL and codecs are unchanged. Descriptor `layout_version` 3 uses the same 72-byte structure and record format and implies **query-prefix attention**:

- Each query uses `q_count = min(active_count, position - origin + 1)`.
- Sealed records cover groups `< floor(q_count / 128)`.
- The current partial group `(q_count - 1) / 128` comes from its stage slot when `q_count % 128 != 0`.
- The exact tail is `[max(0, q_count - precision_tail_tokens), q_count)`.
- Split-K policy still follows `active_count`, so OUT can differ from nq1 by reduction-order rounding. The qualified OUT/LSE tolerances apply, not bitwise equality.

The shaders implement this with `RKVA_SERIAL_PREFIX=1` variants (`rkva_range_attn_kvarn4_prefix_dxil.h`, `rkva_range_attn_kvarn4_opt_prefix_dxil.h`). Both bound the record region by the query window before any sealed-record load. The legacy variants compile from the same source with the macro at 0, and their DXIL is byte-identical to 264.

With precision tail 128, revision 3 stores a **physical 256-row ORIGINAL-F16 ring** behind the logical 128-row window. That adds 8 MiB across 16 layers for HKV4/D256. With tail 0 it stores no ring. Op16 already carries the original F16 for every appended row, so the wire format is unchanged and no bytes are added per normal append. The tail is never reconstructed by dequantizing records or inverting the stage WHT.

### Coverage

The engine tracks per layer the oldest row whose original tail (`tail_begin`) and stage slot (`stage_begin`) are still resident:

- **RESET** provides only the newest `min(live, 128)` tail rows, even though the ring holds 256. It sets `stage_begin = (live_group - 1) * 128`, since the 384-row stage image holds the sink plus both newest ping-pong groups.
- **Append** extends the ring modulo 256 and advances both bounds.
- **Op21 rewind** (at most 128 rows) lowers `tail_begin` to `keep`. When its ORIGINAL-F16 repair rows are contiguous with the resident ring, it extends `tail_begin` down to the repair start. Coverage is published only after the fence succeeds, together with counts and generation. Rewind never rewrites stage slots.

Attention on a revision-3 session validates the batch before allocation or Map. Query positions must be contiguous, end at `origin + active_count - 1` and hold at most 128 queries. Leading queries below the origin are masked and skipped.

The engine refuses the batch when a needed row is not resident. For tail 128 the check is `tail_begin <= max(0, first_q_count - 128)`. For tail 0, every query with a partial group needs `stage_begin <= its group start`. A refusal returns an error and does not invalidate the session. For example, nq3 straight after a deep RESET is refused; its newest nq1 query is accepted.

Generation, stale-epoch, count, ACK and failure-poison guards are unchanged.

### PSO policy

A revision-3 session never uses a legacy PSO:

- **CONFIG** fails closed when the reference prefix PSO is missing.
- **Mode 0** uses the reference prefix PSO.
- **Modes 1..3, nq <= 9** use the optimized prefix PSO when it is available. Otherwise they use the reference prefix PSO; the result is still correct.

Op20 reports `kvarn4_serial_prefix.{reference,optimized}.{available,status}`. Its attention buckets use the storage key `kvarn4-prefix`.

### Disk snapshots

The revision-1 disk codec assumes a 128-row ring. SAVE and RESTORE on a revision-3 session fail with `snapshot: native descriptor revision 3 has no disk codec`, and nothing is written or mutated. Native FULL snapshots remain unsupported.

### Root wiring (proposed, root-owned)

- Advertise capability bit 7 `RKVA_CAP_KVARN_SERIAL_PREFIX` only to clients whose HELLO sets proto flag bit 17. Refuse op15 `layout_version == 3` without bit 17.
- Return the 8-byte generation seed for `layout_version >= 2`.
- Op21 already requires bit 16.
- Op14 currently maps every engine refusal to `RKVA_ERR_GPU`. A coverage refusal does not invalidate the engine session. Root may report it as `RKVA_ERR_BAD_PAYLOAD` when `session_invalid()` is false, as op21 already does, so that the client can fall back to nq1 instead of poisoning the connection.
- The engine API is unchanged.

### Verification

- `tests/rkva_kvarn_range_cpu.cpp` checks ring placement, coverage and query gates against a versioned-row residency simulator. That includes an exhaustive decode to 520 rows with rollbacks.
- `tests/rkva_kvarn_serial_prefix_model.py` encodes native records from a canonical ORIGINAL-F32 store and builds a separate FP64 serial image per query prefix. It checks the emulated revision-3 reference and optimized kernels, run over the engine-layout buffer, at lives 127/128/129/255/256/257, nq 1/2/3/5/9, tail 0/128, origin 72704 and capacity 58368. It also covers rollback and re-append across the seal and ring wrap, masked queries and refused batches.
- The legacy resident-image oracle (`rkva_range_kvarn4_opt_model.oracle_sources`) is a separate contract. It is shown to disagree with serial semantics.
- `tests/rkva_kvarn_serial_prefix_probe.py` is the future raw GPU gate. It is offline unless `--execute --host` is given, and `--self-test` replays its real payloads into an offline engine transliteration.

No GPU, console or model result exists for revision 3.
