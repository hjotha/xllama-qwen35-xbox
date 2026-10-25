# F6 real production campaign - 2026-09-28

Workload (plan section 6): Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp target on
CUDA0 (.57), Xbox draft via DFLASH_XBOX_RPC=192.168.1.26:50053, prompt
"Write a short paragraph explaining why the sky appears blue.", n_predict 512,
seed 1234, temperature 1.0, top_p 0.95, top_k 20, cache_prompt false,
ignore_eos true, ctx 8192, spec-type draft-dflash, spec-draft-n-max 4,
draft-p-min 0. Production service llama-server-root.service stopped per run
and restored afterwards (RESTORE_EXIT=0 in all six runs).

Xbox app 1.0.0.42, backend selected by LocalState\dflash-backend.txt.

| Round | d3d12 tps | cpu tps | content sha256 (both modes)                                      |
| ----- | --------- | ------- | ---------------------------------------------------------------- |
| 1     | 21.318    | 12.916  | 31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc |
| 2     | 21.195    | 12.922  | 31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc |
| 3     | 21.273    | 12.921  | 31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc |

Means: d3d12 21.26 tok/s (spread 21.19-21.32), cpu 12.92 tok/s. Speedup 1.65x
with byte-identical generated content across all six runs (no correctness or
stability failures). Raw completions and summaries live on .57 at
/tmp/dflash-profile-<mode>-completion.json and -summary.json; server logs at
/tmp/dflash-<mode>-server.log.

F7 direction from the measured cycle: draft compute 64.5 ms per cycle (5
positions) corresponds to ~917 MB of K-quant weights per cycle at ~14 GB/s
effective; the project's gpugemv campaign measured 25+ GB/s with tuned
kernels, so replacing the q3k/q4k/q5k x q8k GEMV kernels with the gpugemv
variants is the next lever. n_max/p_min sweeps and mock/local modes remain
open items.
