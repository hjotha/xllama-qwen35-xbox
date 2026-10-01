# F7 Kernel Optimization & Real Production Results (2026-09-28)

## Overview & Goal

Optimize the DFlash2 Xbox GPU GEMV kernels (`shaders/dflash/dflash_draft_quant_q8.hlsl` and `shaders/dflash/q3k_q8k_matmul.hlsl`) to achieve effective weight bandwidth $\ge 20\text{ GB/s}$ and DRAFT compute $\le 46.000\text{ ms/cycle}$ with 5 positions, while strictly preserving exact GGML integer arithmetic (bit-for-bit parity).

Baseline on App 1.0.0.42:

- DRAFT compute: $64.516\text{ ms}$ @ $14.2\text{ GB/s}$
- Dedicated bench cycle RTT: $74.483\text{ ms}$
- Real production TPS on `.57`: $21.26\text{ tok/s}$

---

## Architectural Improvements

### 1. Grouped Cooperative Memory Transactions (128-bit `Load4`)

- Replaced per-lane unaligned byte/word loads for weights `W` with grouped cooperative loads into `groupshared` LDS.
- Vectorized activations: replaced redundant per-thread scalar global memory loads of `Q8` with cooperative 128-bit `Load4` into `groupshared uint s_q8[512]` and `s_d8[8]`.
- All 128 `uint4`s across 8 batch elements are loaded in parallel by lanes 0..127 in a single coalesced memory transaction.

### 2. Tightened LDS Footprint

- Tightened `partial` reduction array in `dflash_draft_quant_q8.hlsl` from `[8 * 64 * 2]` ($4096\text{ bytes}$) to `[2 * 8 * 8]` ($512\text{ bytes}$).
- Tightened `wave_sum` in `q3k_q8k_matmul.hlsl` from `[8 * 64]` ($2048\text{ bytes}$) to `[8 * 8]` ($256\text{ bytes}$).
- Total LDS footprint per workgroup dropped to $<2.8\text{ KB}$, enabling high compute unit occupancy and register caching on Xbox Series X (AMD RDNA2).

### 3. Parallel Batch Reduction across Wave 0

- Eliminated serialized 8-batch inner loops on thread 0 while other 255 threads sat idle at synchronization barriers.
- Distributed batch reduction across Wave 0 (`lane < 8`), reducing VGPR register pressure from 8 vector accumulators per thread to 1 scalar float per thread.

---

## Iterative Hardware Validation & Progression

| Version                 | Description                                   | DRAFT Compute (mean)                            | Cycle RTT (mean)       | Effective Bandwidth     | Probe 17 Status             |
| ----------------------- | --------------------------------------------- | ----------------------------------------------- | ---------------------- | ----------------------- | --------------------------- |
| **1.0.0.42** (Baseline) | Per-lane unaligned loads                      | $64.516\text{ ms}$                              | $74.483\text{ ms}$     | $14.21\text{ GB/s}$     | Baseline reference          |
| **1.0.0.43**            | Grouped LDS weight loads                      | $61.950\text{ ms}$                              | $71.820\text{ ms}$     | $14.80\text{ GB/s}$     | Bit-identical               |
| **1.0.0.44**            | Cooperative Q8 LDS + Wave0 parallel reduction | $46.994\text{ ms}$                              | $56.941\text{ ms}$     | $19.51\text{ GB/s}$     | Bit-identical               |
| **1.0.0.45** (Final)    | 128-bit `Load4` for W/Q8 + tight LDS          | **$44.668\text{ ms}$** (min $44.617\text{ ms}$) | **$54.508\text{ ms}$** | **$20.52\text{ GB/s}$** | **Bit-identical (0 diffs)** |

Target achieved: **$44.668\text{ ms} \le 46.000\text{ ms}$** and **$20.52\text{ GB/s} \ge 20.00\text{ GB/s}$**.

---

## Exact GGML Numerical Parity Check

Probe 17 comparison between baseline and App 1.0.0.45 (`draft-1.0.0.45.json`):

- Total checkpoint differences: **0**
- Total test case differences: **0**
- Every checkpoint (`kqv_out`, `attn_conv_out`, `l_out`, etc.) across layers 0..4 matches bit-for-bit with exact identical RMS, maxAbs, cosine, and NMSE.

---

## Real Production Campaign on GOKAYA (.57)

Workload: Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp target on CUDA0, Xbox DFlash2 GPU RPC (`192.168.1.26:50053`), $n_{\text{predict}} = 512$, seed 1234, $n_{\max} = 4$.

| Round       | App Version  | Mode          | TPS (tok/s) | Wall Time (s) | Output SHA-256          | Restore Exit |
| ----------- | ------------ | ------------- | ----------- | ------------- | ----------------------- | ------------ |
| Baseline    | 1.0.0.42     | CPU RPC       | 12.92       | ~39.6 s       | `31e48d4676ed...e218dc` | 0            |
| Baseline    | 1.0.0.42     | D3D12 GPU     | 21.26       | ~24.1 s       | `31e48d4676ed...e218dc` | 0            |
| **Round 1** | **1.0.0.45** | **D3D12 GPU** | **25.746**  | **20.22 s**   | `31e48d4676ed...e218dc` | **0**        |
| **Round 2** | **1.0.0.45** | **D3D12 GPU** | **25.774**  | **20.18 s**   | `31e48d4676ed...e218dc` | **0**        |
| **Round 3** | **1.0.0.45** | **D3D12 GPU** | **25.739**  | **20.24 s**   | `31e48d4676ed...e218dc` | **0**        |

- **Mean Production TPS:** **$25.753\text{ tok/s}$** (+21.1% over GPU baseline, **1.99x speedup over CPU**).
- **Correctness:** All completions generated the identical SHA-256 hash `31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc`.
- **System Health:** Xbox backend restored to `cpu` default; production service `llama-server-root.service` active and healthy (`{"status":"ok"}`).
