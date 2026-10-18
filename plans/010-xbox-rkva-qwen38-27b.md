# Plan 010: Xbox Series X RKVA Remote Attention for Qwen3.8-27B Production Expansion (`.57` ↔ `.26`)

- **Date:** 2026-10-16
- **Status:** IMPLEMENTED / VALIDATED
- **Validation Run:** Revision 256 on Xbox (`192.168.1.26:50054`), `beellama.cpp` on `.57` (`feat/rkva-xbox-client`). 450 RPC requests completed at `status=0`, GPU decode compute latency 150–230 µs/layer.
- **Target Repositories & Hosts:**
  - **Xbox UWP Server (`192.168.1.26`):** `hjotha/xllama-qwen35-xbox` (local checkout `/home/hjotha/worktrees/xllama-qwen35-4b`, build host `.193` `C:\Users\hjotha\worktrees\xllama-final247`, reference prototype at `.193` `C:\Users\hjotha\worktrees\xllama-dflash-xbox`)
  - **Host Inference Engine (`192.168.1.57` — GOKAYA):** `/home/hjotha/beellama.cpp` + `/home/hjotha/router-production.ini`

---

## 1. Objective

Enable the Xbox Series X (`192.168.1.26`) running `xllama` UWP to operate as a remote **RKVA v1 (`ggml-remote-attn`, TCP `:50054`)** KV-cache and Flash-Attention co-processor for `beellama.cpp` on `.57` (`192.168.1.57`), expanding the usable context window and freeing RTX 4070 VRAM for the production model `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` (`qwen-3.8-27b`).

### Why `ggml-remote-attn` (RKVA v1) Instead of Generic `ggml-rpc`

- Generic `ggml-rpc` offloads full decoder layers (weights + MoE/FFN + Gated DeltaNet + activations) over 1 Gbps Ethernet, serializing dozens of tensor dispatches per layer and collapsing decode throughput to `< 1 tok/s`.
- **RKVA v1 (`ggml-remote-attn`)** keeps 100% of the model weights, 48 Gated DeltaNet recurrent states (fixed size), QKV/output projections, Q/K RMSNorm, RoPE, and attention output gate on the **RTX 4070 (`.57`)**.
- Only the **16 Full-Attention layers** (`il = 3, 7, 11, ..., 63`, GQA `24 Q / 4 KV`, `head_dim = 256`, `head_slices = 2`) send `Q`, `K_new`, `V_new`, and `positions` to the Xbox (`:50054`), which stores the KV cache in **KVarN4** (WHT + 2D Sinkhorn 4-bit quantization) in its 336–560 GB/s GDDR6 memory and returns the attention output `OUT`.

---

## 2. Current Production Baseline on `.57` (`router-production.ini`)

- **Active Command:**
  `/home/hjotha/beellama.cpp/build-optimized/bin/llama-server --models-preset /home/hjotha/router-production.ini --models-max 1 --host 0.0.0.0 --port 8090 --metrics`
- **Model:** `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` (`qwen35` hybrid architecture: 64 layers = 48 Gated DeltaNet + 16 Full-Attention).
- **Hardware Memory on `.57`:**
  - **RTX 4070 (12 GiB VRAM):** `11,503 MiB / 12,282 MiB` used (~779 MiB free).
  - **System RAM (14 GiB UMA shared with Radeon 780M `vulkan:0`):** ~8.6 GiB available (`MemAvailable - 512 MiB` hard safety cap in `common_fit_mixed_kv_context`).
- **Current Adaptive Context Tiers:**
  - `S = 24576` (`mtp = 6`)
  - `M = 38912` (`mtp = 4`)
  - `L = 56320` (`mtp = 2`)
  - `XL = 72704` (`mtp = 2`)
  - `XXL = 102400` (`mtp = 0`, 100% RTX 4070 `kvarn4`)
  - `XXXL = 204800` (`mtp = 0`, `remote-attn = vulkan:0`, `kv-position-split-p = 96256`, `remote-attn-layers = auto`, `remote-attn-cache-type-k/v = q4_0`, `remote-attn-min-context = 131072`)
- **Current Bottlenecks:**
  1. Above `96,256–102,400` tokens, the RTX 4070 runs out of VRAM and must disable MTP (`n = 0`) on `XXL`/`XXXL`.
  2. On `XXXL` (`204,800` tokens), overflowing `[96256, 204800)` to `vulkan:0` (iGPU 780M) consumes `.57` system UMA RAM and cannot scale safely to `256K–320K+` tokens.

---

## 3. Memory & Context Expansion Math (`KVarN4` on Xbox Series X)

For `Qwen3.8-27B` Full-Attention layers (`n_head_kv = 4`, `head_dim = 256`, `head_slices = 2`, `block_size = 32`):

- **Packed K + V data:** `4 * 256 = 1024 B/token` (0.5 B/elem × 2 for K and V)
- **Row scales (F16):** `4 * 2 slices * 2 B * 2 (K+V) = 32 B/token`
- **Column scales (F16 per 32-token group):** `57.5 B/token`
- **Total per token per Full-Attention layer:** **`1,113.5 bytes/token`** (`~1.062 MiB / 1024 tokens`) + `~1.5 MiB` fixed staging/tail per layer.

| Context Length (`max_seq_len`) | Per Full-Attn Layer (`kvarn4`) | 8 Layers on Xbox (50% offload) | 16 Layers on Xbox (100% offload) |
| :----------------------------- | :----------------------------: | :----------------------------: | :------------------------------: |
| **128K (`131,072`)**           |          `141.5 MiB`           |           `1.11 GiB`           |            `2.21 GiB`            |
| **200K (`204,800`)**           |          `220.25 MiB`          |           `1.72 GiB`           |            `3.44 GiB`            |
| **256K (`262,144`)**           |          `281.5 MiB`           |           `2.20 GiB`           |            `4.40 GiB`            |
| **320K (`327,680`)**           |          `351.5 MiB`           |           `2.75 GiB`           | _(exceeds ~4.5 GiB UWP budget)_  |

- With the local Xbox model (`qwen35-4b-mtp`) unloaded in RPC mode, the UWP Game Heap provides **~4.6–5.0 GiB** of usable D3D12 buffer budget:
  - **All 16 layers fit on Xbox up to `262,144` tokens (256K)** (`4.40 GiB`), freeing up to `2.2 GiB` of RTX 4070 VRAM (allowing MTP `n=2..4` to stay enabled at `100K+` context).
  - **In Position-Split / Hybrid mode** (RTX 4070 holds `[0, 96256)` + Xbox holds `[96256, C)`), the Xbox can hold an additional **160K to 256K overflow tokens** across all 16 layers, pushing total context on `.57` to **`262,144 – 327,680+` tokens** with zero pressure on `.57` system RAM.

---

## 4. Existing Code & HLSL Shader Inventory

### 4.1 What Already Exists in `beellama.cpp` (`.57`) and Submodule `llama.cpp`

- `ggml/src/ggml-remote-attn/remote-attn-protocol.h`: Binary wire protocol `RKVA v1` (magic `0x41564B52`, port `50054`, opcodes `HELLO`, `CREATE_SESSION`, `STEP`, `TRIM`, `RESET`, `PING`).
- `ggml/src/ggml-remote-attn/ggml-remote-attn.cpp`: Client backend implementing `GGML_OP_REMOTE_ATTN`, `F32`/`F16` wire conversion, and session lifecycle (`trim_layer`, `reset_layer`).
- `src/models/qwen35.cpp:404-422` & `src/llama-context.cpp:1665-1760`: Graph builder (`build_attn_remote` / `llama_Starting_remote_attn`) and `layer_filter_cb` (`src/llama-model.cpp:3216`) that skips local KV allocation for remote layers.

### 4.2 HLSL Shader Inventory (Do Missing Shaders Need to Be Rewritten in HLSL?)

In `C:\Users\hjotha\worktrees\xllama-dflash-xbox\shaders\rkva\` on `.193` (commits `5d6172a` and `cf2500a`), **5 of the 6 required shaders are ALREADY written in HLSL and compiled to DXIL** (`shaders/generated/rkva_*_dxil.h`):

| Shader File                                        | Target / Role                                                                                | Current HLSL Status on `.193`                                                    | Action Required                                                                                                                                                                                                                                                                                                                                   |
| :------------------------------------------------- | :------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `shaders/rkva/rkva_wht.hlsl`                       | Fast Walsh-Hadamard Transform (`D=128`, `[numthreads(128,1,1)]`) for `Q`, `K_new`, and `OUT` | **100% written in HLSL** & compiled (`rkva_wht_dxil.h`)                          | **None** — port as-is into `xllama`.                                                                                                                                                                                                                                                                                                              |
| `shaders/rkva/rkva_store_f16.hlsl`                 | KV Store for `F16` cache mode                                                                | **100% written in HLSL** & compiled (`rkva_store_f16_dxil.h`)                    | **None** — port as-is into `xllama`.                                                                                                                                                                                                                                                                                                              |
| `shaders/rkva/rkva_flash_attn.hlsl`                | Flash Attention for `F16` cache mode                                                         | **100% written in HLSL** & compiled (`rkva_flash_attn_dxil.h`)                   | **None** — port as-is into `xllama`.                                                                                                                                                                                                                                                                                                              |
| `shaders/rkva/rkva_store_kvarn4.hlsl`              | KVarN4 KV Store (WHT + staging + 2D Sinkhorn 32-token group quantization)                    | **Already written in HLSL** (279 lines, `rkva_store_kvarn4_dxil.h`)              | **Incremental update in HLSL** (not a full rewrite): parallelize Stage 1 over `(4, n_tokens, 1)` for prefill batches (`ubatch=256`) and add `k_tail/v_tail` F16 circular write when `tail_tokens > 0`.                                                                                                                                            |
| `shaders/rkva/rkva_flash_attn_kvarn4.hlsl`         | KVarN4 Flash Attention (4-bit unpack + Sinkhorn scale multiply + online softmax)             | **Already written in HLSL** (267 lines, `rkva_flash_attn_kvarn4_dxil.h`)         | **Incremental update in HLSL** (not a full rewrite):<br>1. Add `split_k` range `[split_start, split_end)` and write partial `(O_part, m_part, l_part)` when `split_k > 1`;<br>2. Replace the 7-barrier shared-memory tree dot-product with `WaveActiveSum` (Wave32) + 4-wave LDS sum (1 barrier);<br>3. Add `k_tail/v_tail` F16 exact tail merge. |
| `shaders/rkva/rkva_flash_attn_split_k_reduce.hlsl` | Online softmax reduction across `split_k` partial outputs `(O_part, m_part, l_part) -> OUT`  | **MISSING (only exists in GLSL as `flash_attn_split_k_reduce.comp`, 122 lines)** | **Write 1 new HLSL shader (~90 lines)** translating `flash_attn_split_k_reduce.comp` and compile with `dxc.exe -T cs_6_2 -O3 -Fh shaders/generated/rkva_flash_attn_split_k_reduce_dxil.h`.                                                                                                                                                        |

> **Summary on Shaders:** We do **not** need to rewrite the KVarN4 or WHT shaders from scratch in HLSL — 5 shaders (`rkva_wht.hlsl`, `rkva_store_f16.hlsl`, `rkva_flash_attn.hlsl`, `rkva_store_kvarn4.hlsl`, `rkva_flash_attn_kvarn4.hlsl`) are already in HLSL on `.193`. Only **1 small shader (`rkva_flash_attn_split_k_reduce.hlsl`, ~90 lines)** needs to be written from scratch in HLSL (ported from the 122-line `flash_attn_split_k_reduce.comp`), plus targeted edits to `rkva_flash_attn_kvarn4.hlsl` and `rkva_store_kvarn4.hlsl` for `split_k`, `WaveActiveSum`, and `tail_tokens`.

---

## 5. Phased Implementation Plan

### Phase 1 — Port RKVA Server (`:50054`) into Main `xllama` App + UWP Heap Management

**Target:** `hjotha/xllama-qwen35-xbox` (dedicated worktree)

1. Copy from `.193` (`C:\Users\hjotha\worktrees\xllama-dflash-xbox`):
   - `shaders/rkva/*.hlsl` and `shaders/generated/rkva_*_dxil.h`
   - `uwp/remote-attn-protocol.h`, `uwp/remote-attn.{h,cpp}`, `uwp/remote-attn-engine.{h,cpp}`
   - Register the source files in `uwp/xllama.vcxproj`.
2. **Single-Foreground-App Integration & Auto-Evict:**
   - Start `serve_remote_attn(50054)` on a background thread in `uwp/App.cpp` / `uwp/api-server.cpp` alongside the `:11434` HTTP server.
   - When `RKVA_CREATE_SESSION` arrives on `:50054` (or via `POST /v1/rpc/mode` on `:11434`), automatically unload any active local `LlamaSession` (`inference-bridge`), freeing the ~4.03 GiB used by `qwen35-4b-mtp` so the `RemoteAttnEngine` has the full ~4.6–5.0 GiB Game Heap.
   - Expose RKVA session metrics (`rkva_active`, `n_layers`, `max_seq_len`, `kv_type`, `allocated_mb`, `steps`) in `GET /v1/health`.

### Phase 2 — Per-Layer D3D12 Resource Allocation (`std::vector<Buffer> layer_kv`)

**Target:** `uwp/remote-attn-engine.{h,cpp}`

1. **Problem:** Currently `RemoteAttnEngine::create_session` (`remote-attn-engine.cpp:419`) allocates a single `ID3D12Resource` (`session->kv`) of size `layer_stride * n_layers`. At `128K–256K` tokens with 16 layers, this single buffer is **`2.21 GiB – 4.40 GiB`**, which fails D3D12 UWP single-resource allocation limits (~2 GiB) and overflows 32-bit `uint` byte offsets in HLSL `ByteAddressBuffer`.
2. **Solution:**
   - Replace `Buffer kv` with `std::vector<Buffer> layer_kv` (`n_layers` separate `D3D12_HEAP_TYPE_DEFAULT` committed resources of size `layer_stride` = `141.5 MiB` @ 128K or `281.5 MiB` @ 256K each).
   - In `dispatch_store_kvarn4`, `dispatch_flash_attn_kvarn4`, `trim`, and `reset`, bind `session->layer_kv[layer_idx].gpu_va() + local_offset` to the Root Descriptors (`SRV`/`UAV`).
   - No changes to HLSL buffer indexing are required because every root descriptor already points directly to a single layer's sub-buffer.

### Phase 3 — HLSL Shader Upgrades for 128K–256K Context (`split_k` + Wave32 + Parallel Store)

**Target:** `shaders/rkva/*.hlsl` + `uwp/remote-attn-engine.cpp`

1. **Write `shaders/rkva/rkva_flash_attn_split_k_reduce.hlsl` (~90 lines):**
   - Port `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp` to HLSL CS 6.2.
   - Takes intermediate buffer `split_accum` (`O_part [split_k, n_tokens, n_head_q, D]` + `(m_part, l_part) [split_k, n_tokens, n_head_q, 2]`) and writes final `OUT [n_tokens, n_head_q, D]`.
2. **Upgrade `shaders/rkva/rkva_flash_attn_kvarn4.hlsl`:**
   - Add `split_k` partitioning (`split_idx = group_id.z` or encoded in `group_id.y = slice_group + split_idx * 8`) so decode (`n_tokens = 1`) with `split_k = 16..32` launches **128 to 256 workgroups** across all 52 RDNA2 CUs instead of only 8 workgroups.
   - Replace the 7-barrier shared-memory reduction tree in the inner loop with `WaveActiveSum` (`cs_6_2` Wave32 intrinsic) + 4-wave LDS sum (1 barrier per head instead of 7).
   - Add `k_tail/v_tail` F16 circular buffer read for exact recent tokens when `tail_tokens > 0`.
3. **Upgrade `shaders/rkva/rkva_store_kvarn4.hlsl`:**
   - Parallelize Stage 1 (`WHT + staging + tail write`) across `(4, n_tokens, 1)` workgroups and Stage 2 (`Sinkhorn 32-token group flush`) across touched groups, eliminating the serial `for (uint token = 0; token < n_tokens; ++token)` bottleneck during `ubatch = 256` prefill.
4. **Compile all updated HLSL shaders on `.193`:**
   - Run `dxc.exe -T cs_6_2 -E main -O3 -Fh shaders/generated/rkva_*_dxil.h` and commit the updated headers.

### Phase 4 — `RKVA_WIRE_F16` Transport & Gigabit LAN Optimization

**Target:** `uwp/remote-attn.cpp` + `ggml-remote-attn.cpp`

1. Enable `RKVA_WIRE_F16` negotiation in `RKVA_HELLO_ACK` so `Q`, `K_new`, `V_new`, and `OUT` travel over TCP as `fp16` (`28 KB/layer/token` = `448 KB/token` across 16 layers in decode; `114 MB` per 256-token ubatch in prefill — 50% bandwidth reduction vs `F32`).
2. Use AVX2/F16C (`_cvtps_ph` / `_cvtph_ps`) on the Xbox Zen 2 CPU (or GPU FP16 upload/readback) and keep `TCP_NODELAY = 1` + `4 MiB` socket buffers on both `.57` and `.26`.

### Phase 5 — `beellama.cpp` Host Integration on `.57` (`router-production.ini`)

**Target:** `/home/hjotha/beellama.cpp` on `192.168.1.57` (dedicated branch from `main`)

1. **Allow TCP `HOST:PORT` in Adaptive Context Tiers:**
   - Update `common_fit_mixed_kv_context()` (`common/common.cpp:1780`), `adaptive_profile_is_mixed()` (`tools/server/server-context.cpp:6114`), and `server-mixed-kv-handoff.cpp` so `--remote-attn 192.168.1.26:50054` works with `--remote-attn-min-context` and `--adaptive-context-tiers`, using `vram_free_bytes` reported by the Xbox `RKVA_HELLO_ACK` instead of requiring local `vulkan:0`.
2. **Support Two Production Operating Modes on `.57`:**
   - **Mode A — Layer-Offload RKVA:** Offload `8` to `16` Full-Attention layers to `192.168.1.26:50054` (`kvarn4`) on high context tiers (`XL`/`XXL`/`XXXL`), freeing `1.1–2.2 GiB` of RTX 4070 VRAM so MTP can remain active at `100K+` context and `XXXL` can scale to `262,144` tokens.
   - **Mode B — Position-Split over RKVA (`[0, 96256)` on RTX 4070 CUDA + `[96256, 262144+)` on Xbox RKVA):** Extend `build_attn_position_split` (`src/llama-context.cpp:1765`) so the overflow band `[P, C)` can target the remote RKVA session on the Xbox. Below `96,256` tokens, 100% of attention runs locally on the RTX 4070 with zero network traffic; above `96,256` tokens, overflow KV is stored and attended on the Xbox without touching `.57` system RAM.
3. **Automatic Fallback to `vulkan:0`:**
   - If `192.168.1.26:50054` is unreachable at context initialization, automatically fall back to `remote-attn = vulkan:0` so `.57` production never fails when the Xbox is powered off.

---

## 6. Validation Gates

- **Gate 1 — Shader & Engine Self-Test on Xbox (`.193` → `.26`):**
  - Build MSIX with RKVA server integrated into `xllama`, deploy to `192.168.1.26`, verify `RKVA_HELLO` and `RKVA_CREATE_SESSION` with 16 layers × `262,144` tokens (`4.40 GiB` across 16 D3D12 buffers), and verify numerical parity of `rkva_wht` + `rkva_store_kvarn4` + `rkva_flash_attn_kvarn4` + `rkva_flash_attn_split_k_reduce` (`max_abs_err < 5e-3`, cosine similarity `> 0.999`).
- **Gate 2 — End-to-End Parity & MTP `TRIM` (`192.168.1.57` → `192.168.1.26:50054`):**
  - Run `Qwen3.8-27B` on `.57` with `--remote-attn 192.168.1.26:50054` at `4K` and `32K` context (`temp = 0`), confirming greedy token identity against local CUDA and verifying `RKVA_OP_TRIM` during MTP draft token rejection and `/slots` save/restore/erase.
- **Gate 3 — Long-Context Ladder (`128K` → `200K` → `256K+`):**
  - Validate prefill and decode at `131,072`, `204,800`, and `262,144` tokens, recording RTX 4070 VRAM usage, `.57` system RAM headroom, Xbox D3D12 buffer usage, per-step network vs GPU shader latency, and final prefill/decode TPS.

## RKVA v2 positional overflow (2026-10-18)

The v2 service adds a separate F16 range mode for host positional splits. It
accepts both v1 and v2 request headers, responds with the request version, and
advertises the new operations only to a v2 client negotiating F16 cache storage
and the original attention domain. Legacy KVarN attention is unchanged.

- `RANGE_CONFIG` (12), after `CREATE_SESSION`: configure the absolute origin and
  compact capacity; clear the row counts of every layer without reallocating KV.
- `RANGE_UPLOAD` (13): upload all K rows followed by all V rows in F16. Physical
  row `r` represents global position `origin + r`. An upload starting at row zero
  replaces that layer's history; other uploads must append contiguously.
- `RANGE_ATTN` (14): submit absolute query positions and original-domain F32 Q.
  Return all F32 OUT values followed by all F32 natural-log softmax normalizers
  (LSE). Attention does not append KV. Causal masking uses global positions;
  an empty or fully masked range returns OUT zero and LSE negative infinity.

For the proposed `[102400, 131072)` Xbox range, the 16 dense-attention layers
(24 Q heads, 4 KV heads, dimension 256) use 1.75 GiB of F16 K/V storage. Values
already quantized in the host overflow cache are dequantized before upload;
this mode applies no additional WHT, inverse WHT, or KVarN quantization. Partial
results use split-K GPU attention and reduction so the host can merge their
OUT/LSE with its CUDA prefix attention. RESET preserves the range origin;
TRIM/REWIND interpret their position as absolute and truncate every layer's row
count. Each layer maintains an independent row count.

Recompile the two embedded shaders with
`DXC=/path/to/dxc scripts/compile-rkva-range-shaders.sh`. Run the live console
correctness gate with `python3 tests/rkva_range_probe.py --host <console-ip>`.
The probe compares GPU OUT/LSE against CPU float64 attention at both sides of
the origin, exercises split-K and session mutation, and checks v1 compatibility.
This kernel gate is separate from the model-level throughput comparison.

### Native Q4_0 range storage

The following capability extension retains the v2 wire layouts and the F16
range path. `HELLO_ACK.reserved[0] & RKVA_CAP_RANGE_Q4_0` advertises native
standard Q4_0 storage. A range client may then set `RKVA_SESSION_RANGE` in
`CREATE_SESSION.flags` to defer allocation until `RANGE_CONFIG` selects storage.
This explicit session flag prevents an unused F16 allocation from limiting a
Q4_0 session's capacity. Legacy session creation still allocates immediately.

`RANGE_CONFIG.reserved[0] & RKVA_RANGE_STORAGE_Q4_0` selects Q4_0 storage;
`RANGE_UPLOAD.wire = RKVA_WIRE_Q4_0` sends the host's original packed rows
(18 bytes per 32 values: F16 scale and 16 packed bytes). K and V remain separate
contiguous planes. Other storage flags are rejected. The shader decodes
`(nibble - 8) * scale`, including negative scales and unaligned 18-byte blocks.
It introduces no additional compressor, rotation, or quantization step.
Reconfiguration discards the old history and releases the old resources before
allocating the requested storage. Total session KV remains bounded by the
advertised 4 GiB budget.

At 28,672 overflow rows and 16 dense layers, native Q4_0 uses 528,482,304 bytes
(0.4921875 GiB), compared with 1,879,048,192 bytes (1.75 GiB) for F16. These
figures describe console KV allocation; the host's cache mirror and model
resources must also be included when comparing system capacity.

Run `tests/rkva_range_probe.py --host <console-ip> --q4` for random packed-block
GPU parity, including both scale signs and extreme nibbles. Run
`tests/rkva_range_capacity.py --host <console-ip> --q4` to fill all 16 layers
with 28,672 rows and verify analytical OUT/LSE results. Omitting `--q4` retains
the F16 checks. Throughput from these kernel gates is not model inference TPS.

The analytical batched gate is `tests/rkva_range_prefill.py --host <console-ip>`.
It exercises 256 queries, 24 Q heads, 4 KV heads, dimension 256, and split-K 32
against a 28,672-row layer in both F16 and native Q4_0 storage. V varies by
head and dimension, so the check covers GQA and packed OUT/LSE layout as well
as empty and causal boundary queries. The oracle uses constant K/V across
positions, avoiding an expensive CPU attention reference for this larger shape.

### Native Q8_0 range storage

`RKVA_CAP_RANGE_Q8_0` is bit 1 of `HELLO_ACK.reserved[0]`; it advertises
native Q8_0 storage alongside the existing Q4_0 capability. Selecting
`RKVA_RANGE_STORAGE_Q8_0` (bit 1) in `RANGE_CONFIG.reserved[0]` requires
`RKVA_WIRE_Q8_0 = 3` uploads. Q4_0 and Q8_0 configuration flags are mutually
exclusive. F16 selection remains zero, and all previous wire values and layouts
remain unchanged. Explicit `RKVA_SESSION_RANGE` creation defers allocation for
both quantized formats.

Q8_0 transfers and stores the original standard GGML blocks: 34 bytes for 32
values, consisting of an F16 scale and 32 signed int8 values. The shader computes
`int8_value * scale`, including negative scales and int8 values from -128 to 127.
No additional server quantizer, WHT, or F16 cache expansion is applied. At
28,672 overflow rows and 16 dense layers, console Q8_0 K/V uses 998,244,352 bytes
(0.9296875 GiB), excluding scratch and the separate host mirror.

The three range storage formats use separate PSOs compiled from the same HLSL
source with `RKVA_RANGE_STORAGE=0`, `1`, or `2`. This specializes the cache loader
without adding another runtime format branch to the F16/Q4_0 kernels. They share
the existing root signature and F32 OUT/LSE response layout.

Use `tests/rkva_range_probe.py --host <console-ip> --q8` for random GPU parity,
including both scale signs, int8 extrema, unaligned 34-byte blocks, session
mutation, format switches, and rejection of conflicting flags. Use
`tests/rkva_range_capacity.py --host <console-ip> --q8` for the full 16-layer
28,672-row allocation/upload gate. `tests/rkva_range_prefill.py` exercises all
three storage PSOs at 256 queries with an analytical causal oracle. These tests
verify implementation parity; model quality requires separate model-level
measurements and is not assumed to be lossless.
