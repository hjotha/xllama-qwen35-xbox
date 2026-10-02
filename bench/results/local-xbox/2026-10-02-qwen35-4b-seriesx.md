# qwen35-4b on Xbox Series X .26 (experimental, outside the catalogue)

Package: upstream release `xllama_1.6.0.1046_x64.msix` (unified, GGUF via
llama.cpp), sha256 `84f7366fb205db77635b84dc38737e479e2912881dcc3b11e546f53b46dc3898`.
Model: `Qwen3.5-4B-Q4_K_M.gguf` 2740937888 B, sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`, arch `qwen35`,
32 blocks, 2560 embedding, GQA 16/4. Side-loaded to
`LocalState\models\qwen35-4b\` via Device Portal; catalogue entry via
`LocalState\manifest.json` override (per-entry merge).

Bench knobs: `prompt.txt` 2309 B (469 prompt tok), `bench_npredict.txt` 128.

| prompt tok/s | decode tok/s | peak WS MB | load ms | n_ctx | threads | backend |
| ------------ | ------------- | ---------- | ------- | ----- | ------- | ------- |
| 23.86        | 9.66          | 2924       | 24084   | 2048  | 6       | cpu     |
| 24.21        | 9.64          | 2924       | 23858   | 2048  | 6       | cpu     |

Two repeats of the same CPU configuration; peak 2924 MB is under the Phi-4-mini
validated 2765 MB band by 159 MB and far under the heap ceiling. Both runs
generated the full 128 tokens.

GPU decode NOT measured: `bench_gpu_layers.txt` is D2b code added after this
release was built (`docs/gguf-gpu-decode.md` cites CI 1.6.0.1138; this package is
1.6.0.1046), so the installed binary ignores the knob — no `[llama] gguf gpu
layers:` log line, no `-gN` host tag, `backend` stayed `cpu`. D2 is also recorded
FAIL and off by default upstream.
