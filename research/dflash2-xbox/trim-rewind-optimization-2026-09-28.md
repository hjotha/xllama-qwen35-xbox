# Xbox DFlash RPC: TRIM and rewind optimization

Measured on 2026-09-28. Xbox app `HJotha.XboxDFlashRpc` version `1.0.0.10`.

The original control round trips cost 74.046 ms/cycle. The optimized path folds both suffix removals into the existing SYNC request. Its standalone TRIM and rewind round trips are eliminated; KV work remains on the Xbox inside SYNC.

Real generation improved from 9.622 to 12.806 tok/s (33.09%). Mean cycle latency fell from 291.299 to 218.663 ms.

## Changes

- Enable TCP_NODELAY on the Linux client; keep the header with its payload using MSG_MORE where available.
- Add optional protocol-v2 opcode 7 (SYNC_TRIM). An empty request probes support without changing KV.
- Remove the union of pending draft and rewind suffixes once before decoding the same target features. A rewind to zero retains the original full reset behavior.
- Keep old SYNC/TRIM/RESET handlers and fallback when an older app rejects the capability probe. Keep the accept-time TRIM for drafts without a subsequent feature SYNC.
- Preserve DRAFT anchor replacement, models, selector, target verification, sampling, and KV types.
- DFLASH_XBOX_LEGACY_TRIM=1 reproduces separate control requests with the optimized TCP client.

## Fixed workload and method

`n_max=4`, `p_min=0`, `n_predict=512`, seed 1234, temperature 1.0, top_p 0.95, top_k 20, cache_prompt=false, ignore_eos=true. Prompt: `Write a short paragraph explaining why the sky appears blue.`

Target: `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, RTX 4070 on 192.168.1.57, target KVarN4 KV, context 8192. Drafter: `/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf`, Xbox CPU, Q4_0 KV.

The original and optimized host sources were rebuilt into separate snapshots using the same core and backend libraries. All full comparisons use the same Xbox 1.0.0.10 app; its original v2 handlers remain available. Production was stopped with explicit user authorization and restored by EXIT cleanup.

Each original/TCP-only/local condition has one run. The fused real and Xbox-mock conditions each have three runs. Per-cycle statistics exclude the first cycle (prefill accounting) and final cycle (no following draft timestamp). TPS uses the full 512-token request.

## Runs

| Run               |    TPS | Mean cycle ms | TRIM ms | Rewind ms | Cycles | Accepted / proposed |
| ----------------- | -----: | ------------: | ------: | --------: | -----: | ------------------: |
| trim-before-real  |  9.622 |       291.299 |  42.142 |    31.904 |    182 |           329 / 728 |
| trim-before-xbox  | 21.652 |       129.225 |  41.483 |    31.771 |    182 |           329 / 728 |
| trim-tcp-real     | 12.796 |       218.674 |   1.362 |     0.492 |    182 |           329 / 728 |
| trim-tcp-xbox     | 48.560 |        57.206 |   0.477 |     0.270 |    182 |           329 / 728 |
| trim-fused-real-1 | 12.782 |       219.111 |   0.000 |     0.000 |    182 |           329 / 728 |
| trim-fused-xbox-1 | 49.186 |        56.452 |   0.000 |     0.000 |    182 |           329 / 728 |
| trim-fused-real-2 | 12.800 |       218.757 |   0.000 |     0.000 |    182 |           329 / 728 |
| trim-fused-xbox-2 | 49.157 |        56.436 |   0.000 |     0.000 |    182 |           329 / 728 |
| trim-fused-real-3 | 12.837 |       218.120 |   0.000 |     0.000 |    182 |           329 / 728 |
| trim-fused-xbox-3 | 49.276 |        56.494 |   0.000 |     0.000 |    182 |           329 / 728 |
| trim-fused-local  | 58.161 |        47.816 |   0.000 |     0.000 |    182 |           329 / 728 |

All 11 requests have identical output SHA-256 and identical position/proposal/acceptance trajectories.

Output SHA-256: `31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc`.

## Real and mock limits

Fused real TPS across three runs: mean 12.806, range 12.782–12.837.

Fused Xbox mock: 56.461 ms/cycle, 49.206 tok/s (range 49.157–49.276). This mock removes DRAFT and SYNC compute while preserving the real RPC path and captured acceptance.

The prior standalone Swift baseline was 34.99 tok/s; it was not rerun here. At 2.813187 final request tokens/cycle, break-even is 80.400 ms/cycle. The remaining real pipeline is dominated by Xbox inference and target verification.

## Small-control transport experiment

Actual Xbox KV removal inside fused SYNC averaged 0.135381 ms/request across the three real runs. This work is included in SYNC preparation; zero standalone TRIM/rewind latency does not mean the KV operation was skipped.

- trim-before.jsonl: 99 steady calls, mean 40.666 ms, median 40.855, p95 41.753, min 39.943, max 42.025.
- trim-after-old-app.jsonl: 99 steady calls, mean 0.400 ms, median 0.395, p95 0.430, min 0.343, max 0.798.

Both microbenchmarks used the same original Xbox 1.0.0.9 app. The new client also successfully fell back after the unsupported capability probe. The result is consistent with the split-write Nagle/delayed-ACK stall; no packet capture was used to distinguish the individual TCP effects.

Most of the throughput gain comes from the TCP fix. TCP-only real generation measured 12.796 tok/s versus 12.806 for the full optimization; that small difference is within the observed real-run variation. Fusion removes the control round trips and their overhead, but is not responsible for the entire 33% speedup.

## Original real: stage statistics

RPC totals include their server compute; nested diagnostic rows overlap and must not be added together.

| Stage                              | Mean ms | Median ms |  p95 ms |  Min ms |  Max ms | % cycle |
| ---------------------------------- | ------: | --------: | ------: | ------: | ------: | ------: |
| full_cycle_us                      | 291.299 |   298.830 | 306.913 | 250.760 | 328.196 |  100.00 |
| draft_rpc_us                       | 142.488 |   142.622 | 148.151 | 136.061 | 149.793 |   48.91 |
| draft_compute_us                   | 139.627 |   139.736 | 145.268 | 133.333 | 146.553 |   47.93 |
| sync_rpc_us                        |  22.215 |    22.212 |  22.899 |  21.382 |  23.657 |    7.63 |
| sync_compute_us                    |  16.602 |    16.604 |  16.752 |  16.423 |  16.800 |    5.70 |
| sync_prepare_including_fused_kv_us |   0.037 |     0.037 |   0.041 |   0.034 |   0.057 |    0.01 |
| trim_rpc_us                        |  42.142 |    42.139 |  43.673 |  40.045 |  44.327 |   14.47 |
| rewind_rpc_us                      |  31.904 |    40.952 |  42.616 |   0.000 |  43.632 |   10.95 |
| control_rpc_us                     |  74.046 |    83.008 |  85.499 |  40.654 |  86.784 |   25.42 |
| target_verify_us                   |  43.334 |    42.995 |  44.793 |  41.558 |  62.699 |   14.88 |
| selector_us                        |   5.765 |     5.691 |   6.276 |   4.621 |   7.551 |    1.98 |
| sample_us                          |   1.735 |     1.421 |   3.063 |   0.420 |   3.607 |    0.60 |
| commit_kv_us                       |   0.178 |     0.159 |   0.239 |   0.085 |   0.275 |    0.06 |
| xbox_compute_total_us              | 156.229 |   156.322 | 161.890 | 149.928 | 163.130 |   53.63 |
| rpc_noncompute_us                  |  82.520 |    91.487 |  94.141 |  49.055 |  95.217 |   28.33 |

## TCP-only real: stage statistics

RPC totals include their server compute; nested diagnostic rows overlap and must not be added together.

| Stage                              | Mean ms | Median ms |  p95 ms |  Min ms |  Max ms | % cycle |
| ---------------------------------- | ------: | --------: | ------: | ------: | ------: | ------: |
| full_cycle_us                      | 218.674 |   218.460 | 225.594 | 209.225 | 244.130 |  100.00 |
| draft_rpc_us                       | 142.254 |   142.490 | 147.951 | 135.972 | 149.111 |   65.05 |
| draft_compute_us                   | 139.508 |   139.680 | 145.133 | 132.781 | 146.443 |   63.80 |
| sync_rpc_us                        |  22.113 |    22.127 |  22.718 |  21.378 |  23.137 |   10.11 |
| sync_compute_us                    |  16.600 |    16.585 |  16.750 |  16.447 |  17.195 |    7.59 |
| sync_prepare_including_fused_kv_us |   0.039 |     0.038 |   0.042 |   0.035 |   0.072 |    0.02 |
| trim_rpc_us                        |   1.362 |     1.788 |   1.867 |   0.499 |   2.313 |    0.62 |
| rewind_rpc_us                      |   0.492 |     0.600 |   0.777 |   0.000 |   1.829 |    0.22 |
| control_rpc_us                     |   1.853 |     1.825 |   2.599 |   0.506 |   3.637 |    0.85 |
| target_verify_us                   |  43.339 |    42.987 |  44.808 |  41.623 |  64.264 |   19.82 |
| selector_us                        |   5.766 |     5.686 |   6.378 |   4.806 |   8.411 |    2.64 |
| sample_us                          |   1.738 |     1.429 |   3.038 |   0.383 |   3.169 |    0.79 |
| commit_kv_us                       |   0.170 |     0.154 |   0.241 |   0.083 |   0.342 |    0.08 |
| xbox_compute_total_us              | 156.108 |   156.236 | 161.730 | 149.547 | 163.053 |   71.39 |
| rpc_noncompute_us                  |  10.113 |    10.155 |  11.280 |   7.954 |  12.047 |    4.62 |

## Fused real, pooled three runs: stage statistics

RPC totals include their server compute; nested diagnostic rows overlap and must not be added together.

| Stage                              | Mean ms | Median ms |  p95 ms |  Min ms |  Max ms | % cycle |
| ---------------------------------- | ------: | --------: | ------: | ------: | ------: | ------: |
| full_cycle_us                      | 218.663 |   218.401 | 225.548 | 208.810 | 245.894 |  100.00 |
| draft_rpc_us                       | 143.908 |   143.915 | 149.392 | 136.949 | 151.370 |   65.81 |
| draft_compute_us                   | 141.129 |   141.136 | 146.577 | 133.941 | 148.650 |   64.54 |
| sync_rpc_us                        |  22.313 |    22.336 |  22.902 |  21.531 |  26.328 |   10.20 |
| sync_compute_us                    |  16.641 |    16.636 |  16.789 |  16.473 |  17.259 |    7.61 |
| sync_prepare_including_fused_kv_us |   0.174 |     0.173 |   0.189 |   0.165 |   0.197 |    0.08 |
| trim_rpc_us                        |   0.000 |     0.000 |   0.000 |   0.000 |   0.000 |    0.00 |
| rewind_rpc_us                      |   0.000 |     0.000 |   0.000 |   0.000 |   0.000 |    0.00 |
| control_rpc_us                     |   0.000 |     0.000 |   0.000 |   0.000 |   0.000 |    0.00 |
| target_verify_us                   |  43.289 |    42.951 |  44.660 |  41.526 |  62.712 |   19.80 |
| selector_us                        |   5.794 |     5.699 |   6.535 |   4.613 |   9.574 |    2.65 |
| sample_us                          |   1.768 |     1.464 |   3.152 |   0.372 |   3.770 |    0.81 |
| commit_kv_us                       |   0.178 |     0.157 |   0.239 |   0.082 |   1.268 |    0.08 |
| xbox_compute_total_us              | 157.771 |   157.783 | 163.240 | 150.755 | 165.413 |   72.15 |
| rpc_noncompute_us                  |   8.450 |     8.446 |   9.117 |   7.589 |  12.329 |    3.86 |

## Fused Xbox mock, pooled three runs: stage statistics

RPC totals include their server compute; nested diagnostic rows overlap and must not be added together.

| Stage                              | Mean ms | Median ms | p95 ms | Min ms | Max ms | % cycle |
| ---------------------------------- | ------: | --------: | -----: | -----: | -----: | ------: |
| full_cycle_us                      |  56.461 |    55.930 | 58.824 | 52.430 | 78.941 |  100.00 |
| draft_rpc_us                       |   2.270 |     2.252 |  2.419 |  2.084 |  2.629 |    4.02 |
| draft_compute_us                   |   0.000 |     0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| sync_rpc_us                        |   4.900 |     4.886 |  5.056 |  4.687 |  9.504 |    8.68 |
| sync_compute_us                    |   0.000 |     0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| sync_prepare_including_fused_kv_us |   0.038 |     0.037 |  0.041 |  0.034 |  0.057 |    0.07 |
| trim_rpc_us                        |   0.000 |     0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| rewind_rpc_us                      |   0.000 |     0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| control_rpc_us                     |   0.000 |     0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| target_verify_us                   |  41.080 |    40.537 | 42.476 | 38.542 | 61.242 |   72.76 |
| selector_us                        |   5.438 |     5.308 |  5.878 |  4.615 |  8.336 |    9.63 |
| sample_us                          |   1.747 |     1.344 |  3.155 |  0.343 |  3.382 |    3.09 |
| commit_kv_us                       |   0.119 |     0.121 |  0.142 |  0.061 |  0.187 |    0.21 |
| xbox_compute_total_us              |   0.000 |     0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| rpc_noncompute_us                  |   7.170 |     7.157 |  7.375 |  6.846 | 11.740 |   12.70 |

## Validation and artifacts

- Linux research server and Xbox Release MSIX builds succeeded. Existing compiler warnings are retained in build logs.
- Two existing DFlash static regressions passed.
- Real Xbox protocol tests compare hidden output bit for bit after cuts before/equal/after the draft anchor, reset, a 64-row SYNC, five rejected malformed requests, and a nonmutating capability probe.
- All benchmark requests returned 512 tokens; exact trajectories and text are asserted by analyze_trim.py.
- The final real DRAFT capture is byte-for-byte identical to the previous profiling capture: 18,639,712 bytes, SHA-256 `8edb11cb003f3b973b442daa0750e18025ef05bfedc9ce5c55d65c8d4080d0ce`.
- Raw host traces, request summaries, completion JSON, Xbox logs, source snapshots, and cycles.csv are retained.
- Git diff --check and Python/shell syntax checks passed. No model-wide quality suite was claimed.

Worktree: `C:\Users\hjotha\worktrees\xllama-dflash-xbox`, branch `research/dflash-xbox-rpc`. Host checkout: `/home/hjotha/beellama.cpp`, branch `research/dflash2-xbox-rpc` (existing llama.cpp direct-checkout exception). No commit, merge or push: the broader research changes already present remain uncommitted.

Base xllama commit: `4bad1d58d3c38dbab3d484e1dab5a611cc20c58e`. Base llama.cpp commit on both hosts: `57e515b8c0801926d298395cc900a7e154248940`. The measured code includes pre-existing research modifications plus `optimization.patch`; `source-hashes.json` records the exact before/after sources.

## Final operational state

- Comparison script finished with `TRIM_RUN_EXIT=0`.
- Production `llama-server-root.service` restored to `active`; `/health` returned HTTP 200 with `{"status":"ok"}`.
- No listener remained on test port 59567.
- Xbox process `DFlashRpc.exe`, PID 3976, was running package `HJotha.XboxDFlashRpc_1.0.0.10_x64__3hvcqmcd1t14m`.
- Final Xbox control request explicitly disabled mock: op 6, nTokens 0, status 0, mock false.
