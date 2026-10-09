# Accepted intervention: llama.ini n_batch=256 (n_ubatch stays 64)

## Hypothesis and mechanism

Per-token normalization of the census + PSTEP attribution showed that per
token the CPU volume is LOWER and the GPU time slightly BETTER at wider
chunks, while wall-d3w per token grows: the cost is per-chunk overhead, not
per-element work (elembench: kernels width-flat) and not the MTP catch-up
(seq control). Therefore: fewer prefill chunks (larger n_batch) at the SAME
physical ubatch (64) should amortize that overhead without the ubatch-256
penalty. This is a config-only change, no code.

## Same-build paired A/B (rev187 iteration, one variable: --batch 64 vs 256)

Interleaved 3 reps per cell, profile OFF, everything else fixed
(threads 2, ctx 2048, gpu layers 34, ubatch 64, MTP depth 2/pmin 50, greedy,
seed 1):

| cell                 | prefill b64 -> b256           | decode                  |
| -------------------- | ----------------------------- | ----------------------- |
| std512               | 87.7 -> 94.9 t/s (**+8.23%**) | 29.00 -> 28.67 (-1.14%) |
| code64               | 87.1 -> 93.5 t/s (**+7.38%**) | 34.44 -> 34.43 (-0.03%) |
| chat64 (tiny prompt) | 93.3 -> 85.7 (-8.23%)         | 28.87 -> 28.89 (+0.07%) |

## End-to-end acceptance (LAN API, total wall latency, max_tokens=32)

| prompt     | n_batch=64 | n_batch=256 | delta               |
| ---------- | ---------- | ----------- | ------------------- |
| 275 tokens | 4.237 s    | **4.033 s** | **-204 ms (-4.8%)** |
| 96 tokens  | 2.309 s    | **2.234 s** | -75 ms (-3.3%)      |

(rep 1 of each config is cold load and excluded; reps 2-3 identical within
ms.)

## Gates

- Full-token digests identical b64 vs b256: chat64 `84e081965076a0c9`,
  code64 `ff42c5beec71ff49`, std512 `84e081965076a0c9`.
- API chat PASS on the 256 config; termgate 6/6, seq/ref/cand byte-identical
  to rev163.
- Memory: peak WS and gpu_mem unchanged (bm rows below).
- The tiny-prompt bench prefill regression (-8%) does not reproduce
  end-to-end (the 96-token API prompt is FASTER); the practical workload
  (any prompt >= ~2 chunks at 64) wins.

## Decision

Keep `n_batch=256` with `n_ubatch=64` in the production llama.ini. Rollback:
`llama.ini.backup` (n_batch=64) in this directory; one upload restores it.
Final installed build after this round: LTCG rev188 (same source, optimized),
config 256/64.

## LTCG rev188 verification (final state)

- MSIX sha256 `73CB3B6156F0A734CFF2C0EDF7E46787DF0F6BDBD7A58839CE5EE354A2B11AE0`
  (builder == fetched), PFN `GianlucaMazza.xllama_1.6.0.188_x64__pj67f1fcj4n14`;
  llama.ini verified on device: n_batch=256 n_ubatch=64 mtp=2 n_predict=48
  n_seq_max=2; all experiment knobs absent.
- API chat PASS; long-prompt latency reps: 4.191 (cold) / 4.096 / 4.083 s,
  matching the iteration A/B (-4.8% vs the 64 baseline) on the optimized build.
