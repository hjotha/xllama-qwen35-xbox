# Plan 010: Xbox Series X RKVA Remote Attention for Qwen3.8-27B Production Expansion (`.57` ↔ `.26`)

- **Date:** 2026-10-16
- **Status:** PLANNED
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
| :--- | :---: | :---: | :---: |
| **128K (`131,072`)** | `141.5 MiB` | `1.11 GiB` | `2.21 GiB` |
| **200K (`204,800`)** | `220.25 MiB` | `1.72 GiB` | `3.44 GiB` |
| **256K (`262,144`)** | `281.5 MiB` | `2.20 GiB` | `4.40 GiB` |
| **320K (`327,680`)** | `351.5 MiB` | `2.75 GiB` | *(exceeds ~4.5 GiB UWP budget)* |

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

| Shader File | Target / Role | Current HLSL Status on `.193` | Action Required |
| :--- | :--- | :--- | :--- |
| `shaders/rkva/rkva_wht.hlsl` | Fast Walsh-Hadamard Transform (`D=128`, `[numthreads(128,1,1)]`) for `Q`, `K_new`, and `OUT` | **100% written in HLSL** & compiled (`rkva_wht_dxil.h`) | **None** — port as-is into `xllama`. |
| `shaders/rkva/rkva_store_f16.hlsl` | KV Store for `F16` cache mode | **100% written in HLSL** & compiled (`rkva_store_f16_dxil.h`) | **None** — port as-is into `xllama`. |
| `shaders/rkva/rkva_flash_attn.hlsl` | Flash Attention for `F16` cache mode | **100% written in HLSL** & compiled (`rkva_flash_attn_dxil.h`) | **None** — port as-is into `xllama`. |
| `shaders/rkva/rkva_store_kvarn4.hlsl` | KVarN4 KV Store (WHT + staging + 2D Sinkhorn 32-token group quantization) | **Already written in HLSL** (279 lines, `rkva_store_kvarn4_dxil.h`) | **Incremental update in HLSL** (not a full rewrite): parallelize Stage 1 over `(4, n_tokens, 1)` for prefill batches (`ubatch=256`) and add `k_tail/v_tail` F16 circular write when `tail_tokens > 0`. |
| `shaders/rkva/rkva_flash_attn_kvarn4.hlsl` | KVarN4 Flash Attention (4-bit unpack + Sinkhorn scale multiply + online softmax) | **Already written in HLSL** (267 lines, `rkva_flash_attn_kvarn4_dxil.h`) | **Incremental update in HLSL** (not a full rewrite):<br>1. Add `split_k` range `[split_start, split_end)` and write partial `(O_part, m_part, l_part)` when `split_k > 1`;<br>2. Replace the 7-barrier shared-memory tree dot-product with `WaveActiveSum` (Wave32) + 4-wave LDS sum (1 barrier);<br>3. Add `k_tail/v_tail` F16 exact tail merge. |
| `shaders/rkva/rkva_flash_attn_split_k_reduce.hlsl` | Online softmax reduction across `split_k` partial outputs `(O_part, m_part, l_part) -> OUT` | **MISSING (only exists in GLSL as `flash_attn_split_k_reduce.comp`, 122 lines)** | **Write 1 new HLSL shader (~90 lines)** translating `flash_attn_split_k_reduce.comp` and compile with `dxc.exe -T cs_6_2 -O3 -Fh shaders/generated/rkva_flash_attn_split_k_reduce_dxil.h`. |

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
