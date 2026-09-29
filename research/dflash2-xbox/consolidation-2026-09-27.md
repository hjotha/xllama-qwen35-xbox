# DFlash2 Xbox: consolidated working state

Date: 2026-09-27. Active source worktree: `C:\Users\hjotha\worktrees\xllama-dflash-xbox`, branch `research/dflash-xbox-rpc`. The nested `llama.cpp` checkout is on `research/dflash-xbox-rpc` at base commit `57e515b8c0801926d298395cc900a7e154248940`. No commit, merge, or push was requested.

## Provenance

| Prior worktree             | Material retained here                                                                              | Status                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `xbox-dflash-architecture` | `dflash2-remote-pipeline-audit-2026-09-27.md`                                                       | Read-only architecture audit.                                                   |
| `xbox-dflash-rpc-probe`    | LAN report, three raw JSON datasets, Python clients, `transport-probe` UWP source, `xbox_portal.py` | DFR2 transport probe, not model inference. Original worktree remains untouched. |
| `xllama-dflash2-xbox`      | Four UWP compatibility edits already byte-identical in the active nested checkout                   | No unique source to import. Original worktree remains untouched.                |
| `xllama-dflash-xbox`       | GGML RPC UWP app and DFL2 dedicated draft server sources                                            | Active implementation base.                                                     |

The copied `transport-probe/DFlashRpcProbe` directory contains generated `AppPackages`, `bin`, and `obj` subdirectories. A bounded `Remove-Item -LiteralPath ... -Recurse -Force` command was rejected by automatic policy, so they remain for now. They are copied artifacts, not new implementation.

## Live observations

- Xbox `192.168.1.26`: package `HJotha.XboxDFlashRpc_1.0.0.2_x64__3hvcqmcd1t14m` is installed and `DFlashRpc.exe` was running. Generic GGML RPC on TCP 50052 replied with protocol 6.0.1 and one CPU device. Twenty pings from HPBEATS had p50 611 us and p95 889 us. The dedicated TCP 50053 listener was closed.
- The live UWP telemetry file reported `appMemoryUsageLimitBytes=1073741824` (App classification). This is below the requested ~5 GiB Game budget. Package version 1.0.0.3 exists in source but had not been built/deployed at this checkpoint.
- The Q3 GGUF was copied from `/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf` to local staging. Source and local SHA-256 both equal `72809a53e89cb959a3149c9b3752ad181b640164b51a3b9f445712520beec5dc`; length is 916702560 bytes. Device Portal accepted the upload with HTTP 200 and listed `LocalState\dflash-q3.gguf` at the same length. A console-side SHA has not yet been measured.
- GOKAYA `.57`: target branch `research/dflash2-xbox-rpc` at `57e515b8c`, with only RPC-version and MTP sync-timing edits pending before this run. The active preset was ISTA Swift 1.5 + local MTP, target/KV on `.57`; there was no DFlash remote process. The current `llama-server` binary rejects `--rpc`, so baseline C needs a matching RPC-enabled build.
- On 2026-09-27, `nvidia-smi` on `.57` returned `No devices were found`; kernel logged a UVM global fatal error requiring reboot. The ISTA preset was unloaded/failed. Current A/B GPU baselines cannot be measured until GPU recovery. Prior DFlash measurements used a different target/KV and are not comparable baselines.

## Architecture under implementation

`target (.57)` captures five F32 hidden states per target token (25,600 floats / 102,400 bytes); the Xbox keeps only the DFlash2 model and its draft KV. The target's token embeddings, output head, main weights, and main KV stay on `.57`. A split DFlash graph on Xbox must accept host-computed anchor/MASK embeddings and return the final normalized hidden states. The `.57` must apply its target output head and DFlash selector, verify proposed tokens with ISTA, and send accepted-position/rollback state to the Xbox. The current dedicated UWP source only defines HELLO, RESET, SYNC, and DRAFT, and still assumes `ctx_other` on its draft path; it does not yet satisfy that flow.

## Validation gates and comparison table

| Baseline                          | Same Swift 1.5 / Q3 / kvarn4 workload                  | Decode TPS | Acceptance | Status       |
| --------------------------------- | ------------------------------------------------------ | ---------: | ---------: | ------------ |
| A target alone on `.57`           | Pending GPU recovery                                   |          — |          — | Not measured |
| B target + local DFlash2          | Pending GPU recovery                                   |          — |          — | Not measured |
| C target + generic Xbox RPC       | Requires RPC-enabled `.57` binary and graph validation |          — |          — | Not measured |
| D target + dedicated Xbox RPC     | Requires split graph, session protocol, parity         |          — |          — | Not measured |
| E host projection + Xbox backbone | Requires D and host selector placement                 |          — |          — | Not measured |

Do not classify remote DFlash2 as viable, marginal, or not viable from the measured transport alone. The next discriminating tests are: GPU recovery on `.57`; Game classification and model load on Xbox; deterministic split parity; then 512-token A/B/C/D/E workloads with identical prompts, seed, sampling, target/quant/KV, and n_max 2/4/6/8 where supported.
