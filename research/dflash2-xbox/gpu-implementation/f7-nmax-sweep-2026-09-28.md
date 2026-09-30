# F7 n_max sweep (d3d12 Xbox), 2026-09-28

Same fixed workload as the F6 campaign, target on CUDA0 (.57), Xbox app
1.0.0.42 with backend=d3d12, spec-draft-n-max varied. 512 tokens, seed 1234.

| n_max | tps                                 | content sha256                                                           |
| ----- | ----------------------------------- | ------------------------------------------------------------------------ |
| 2     | 20.818                              | e9f989d14d83dee386ee2c014215e12825eab935be89269eca9dfec99cd955b3         |
| 4     | 21.318 / 21.195 / 21.273 (3 rounds) | 31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc         |
| 6     | 18.446                              | 31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc         |
| 7     | not feasible                        | target-side CUDA OOM: extra 981.45 MiB compute buffer, cudaMalloc failed |

The d3d12 Xbox sweet spot in this configuration is n_max=4; n_max=6 lowers TPS
(more positions per cycle without compensating acceptance) and n_max=7 does
not fit the 4070 target context in this harness. Production service restored
after every run (8090 /health 200). Remaining F7 levers: q3k/q4k/q5k x q8k
GEMV kernel bandwidth (~14 GB/s effective vs 25+ GB/s in the gpugemv
campaign) and RPC copy/wait trims (cycle RTT 74.2 ms vs 68 ms compute).
