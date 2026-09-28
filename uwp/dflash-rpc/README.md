# Xbox DFlash RPC

This UWP app runs the DFlash2 drafter on Xbox Series X. The persistent DFL2 v2
service listens on TCP 50053. The target model, its KV cache, selector, and
sampling stay on the Linux host. The app also exposes the older ggml RPC service
on TCP 50052.

## Build

Clone the repository with its `llama.cpp` submodule, which points to the matching
beellama fork commit. On the HPBEATS build host, run
`research/dflash2-xbox/build-dflash-rpc.cmd` with Visual Studio 2022,
`C:\BuildTools` C++ tools, Windows SDK 26100, and Microsoft.AI.DirectML 1.15.4
in the local NuGet cache. The build produces an unsigned
MSIX under `uwp/dflash-rpc/AppPackages/`; sign it with a locally trusted test
certificate before installing through Xbox Device Portal. Put the DFlash GGUF
in the app's LocalState as `dflash-q3.gguf`. Model weights and certificates are
not stored in Git.

## Execution and diagnostics

Version 1.0.0.17 still runs SYNC and DRAFT through ggml CPU by default
(`n_gpu_layers=0`, context 49152, Q4_0 KV). GPU operations are opt-in
diagnostics and never change the DFL2 sequence:

| Op | Operation |
| -- | --------- |
| 8 | Synthetic Q4_K D3D12 probe and adapter capabilities |
| 9 | Real GGUF Q4_K FFN-down matrix, HLSL and ggml CPU |
| 10 | Same Q4_K matrix converted once to FP16, DirectML GEMM |
| 11 | Real GGUF Q3_K `fc.weight`, HLSL and ggml CPU, batch 1–64 |
| 12 | Real GGUF Q5_K attention V matrix, HLSL and ggml CPU, batch 1–64 |

The real-matrix probes use 50 warmups and three sets of 200 measurements.
They report GPU timestamps separately from host time including readback.
DirectML's FP16 conversion changes the compute representation and consumes
more GPU memory than packed GGUF weights. None of these results is DFlash
generation throughput.

`research/dflash2-xbox/test_trim_protocol.py` exercises fused SYNC_TRIM and
invalid requests. `research/dflash2-xbox/gpu-implementation/reference_capture_proxy.py`
captures and replays complete DFL2 frames. The capture and model are local
test inputs; do not commit them. The eight-stage GPU plan is in
`research/dflash2-xbox/dflash-gpu-implementation-plan-2026-09-28.md`.

## Current evidence

On Series X, version 1.0.0.17 completed 369 captured DFL2 operations with
bitwise-identical status and payloads, and all six TRIM protocol cases passed.
The real Q4_K FFN-down tensor is 50,135,040 bytes (`17408 x 5120`). Its HLSL
probe measured 0.894 ms with readback, versus 1.629–1.635 ms for DirectML FP16,
using averages of the three run medians. Those timings exclude the complete
model graph, KV, RPC transport, and target verification. See the staged
evidence under `research/dflash2-xbox/gpu-implementation/`.
