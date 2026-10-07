# Plan 001: Make Qwen3.5 MTP correct, then demonstrate an Xbox speedup

Date: 2026-10-03. Priority: P1. Effort: M/L. Risk: HIGH for decode-state changes; LOW for instrumentation. Status: IN PROGRESS — OpenCode read the full plan at 20:29:30 UTC and began the drift check; implementation/performance gates remain pending.

## Objective and authorization

The user requests continued implementation toward fully functional Qwen3.5-4B MTP that is faster than no-MTP on the Xbox. Rev29 is neither a successful speedup nor proof that the objective is impossible. Continue the existing task without asking whether to abandon it after each build. Do not claim success for positive draft counters, CPU tests, or a historical baseline alone. If a concrete blocker survives the experiments below, document measured evidence and the remaining technical options; never substitute an invented speedup or an endless retry loop.

This plan was prepared by read-only source review, original-fork comparison, OpenCode transcript inspection, and a live Xbox log read. No implementation/build/benchmark was performed by this review.

## Exact workspace and source authority

- Executor: existing OpenCode session `ses_f0444f521ffeKHL4QP5XiShCsW`, optiplex/.157, tmux `0:0.0`.
- Reuse task worktree `/home/hjotha/worktrees/xllama-mtp`, branch `spike/beellama-pin`.
- Planned at xllama HEAD `1eb0c7d35a6ddf1d15a8abf3c55f46ad830ae1a7`.
- Local BeeLlama submodule pin `982eaadaa9dbe761f00205bc20b6c04b7329b58d`; it already has local UWP export patches. Preserve them. Persist necessary submodule changes as project patches, following `AGENTS.md`.
- The user's ORIGINAL fork is on `hjotha@192.168.1.57:/home/hjotha/beellama.cpp`, verified HEAD `638c565b860381a9be1aa74337c982b83db46468`, branch `feat/occupancy-position-split`. It has unrelated dirty CUDA and test-kvarn work. READ ONLY: do not checkout, reset, build, restart services, or deploy there for this task.
- Original .57 `common/speculative.cpp`, `common/common.h`, `common/common.cpp`, and `src/llama-memory-recurrent.cpp` match the local submodule byte-for-byte. `src/llama-context.cpp` differs only in the three local C API export edits. Therefore the local references below represent the original MTP and rollback behavior; updating the pin blindly is unnecessary.
- Windows builds stay on .193 using the existing `XllamaMtpBuild` scheduled task and its working incremental MSBuild configuration. Preserve `uwp/ggml-uwp/x64` objects and the certificate workflow. Do not replace this with a Linux cross-build or a wholesale toolchain migration.

First inspect `git status`, the active build task, and current package revision. Run this drift check from the worktree and inspect any delta before editing:

```bash
git diff --stat 1eb0c7d35a6ddf1d15a8abf3c55f46ad830ae1a7..HEAD -- src/bridge include/xllama shaders tests patches
git status --short --branch
```

The worktree is shared with the existing task owner. Preserve concurrent work. This advisor added only `plans/`; do not commit unrelated existing changes. Follow applicable repository validation/integration rules; no success merge while required checks fail.

## What the evidence actually says

- Rev29 transcript contains 3.96 / 4.28 / 4.42 tok/s at MTP depth 1, with 16/13, 14/13, 17/16 drafted/accepted. Correct aggregate: **47 proposed, 42 accepted, 89.36% conditional acceptance**. This does not measure the number of draft attempts declined by `p_min`, prefix correctness, or achieved acceleration.
- `/tmp/opencode/bench-r29.csv` retains only runs 2 and 3. `bench-xbox-ort.sh --runs 3` performs three runs but drops run 1 as warmup. Use `--runs 4` for three recorded samples. The 18.43 tok/s baseline is historical and must be remeasured with the same new package and effective settings.
- Recorded Rev29 commands did NOT include `--kv-q8`, and retained CSV host tags have no `-kvq8`. The copied claim that these runs used q8 KV is unsupported. Do not carry it into a new report without checking effective cache types.
- Live console log: target RS = 50.25 MiB; target compute reserve = 504.02 MiB; draft KV = 8.00 MiB, draft compute = 5.85 MiB at depth 4; target/draft graph splits = 322/12. The draft logs `shared=0`. Multi-token runs contain both refused target rewinds and draft position reuse errors.
- `peak_ws_mb` comes from `PeakWorkingSetSize` (`src/bridge/platform.cpp:75-79`); `gpu_budget_mb` comes from DXGI `QueryVideoMemoryInfo(...).Budget` (`:147-151`). Dividing 4029 by 4147 mixes accounting domains. It is not proof of 97% of the UWP app memory limit. Do not add process working set and GPU usage either without understanding overlap.
- GPU coverage is explicitly limited to weight matmuls: `src/bridge/llama_gpu.h:61-65` disables KQV offload; `src/bridge/ggml_d3d12.cpp:878-924` supports selected MUL_MAT plus views. Attention, recurrence, norms, etc. use the CPU. A count of offloaded layers does not mean the whole graph runs in HLSL.

## Prioritized findings

| Priority | Finding and evidence                                                                                                                                                                                                                                                                          | Confidence                       | Effort / risk |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------- |
| 1        | Partial-reject branch trims, decodes the correction, then the generic final trim deletes that correction: `decode_loop.h:433-470`, helper `:193-197`. Pre-existing verifier issue now used by MTP.                                                                                            | High                             | S / medium    |
| 1        | Target setup never assigns `n_rs_seq`: `inference.cpp:450-474`, `session.cpp:488-516`. Original common setup derives it from MTP draft depth.                                                                                                                                                 | High                             | S / medium    |
| 1        | MTP bridge lacks original accepted-target hidden-row catch-up and acceptance lifecycle. `mtp_draft.cpp:185-325` also contains no tail-removal call despite the Rev29 commit message.                                                                                                          | High                             | M / high      |
| 1        | Current verifier decodes the lead separately, then verifies drafts only: `decode_loop.h:268-362`. It misses the original combined target batch and requires a precise token/hidden-position audit.                                                                                            | High                             | M / high      |
| 2        | Persistent CPU pools are attached only to target contexts: `inference.cpp:488`, `session.cpp:542`; draft init has no attachment. CPU fallback creates/frees disposable pools: `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:3518-3523,3576-3577`. The resulting performance share must be measured. | High on omission, medium on cost | S / medium    |
| 2        | Multiple per-round/per-depth logs hit unbuffered disk output: `mtp_draft.cpp:248-303`, `platform.cpp:45-65`. `top_k(10)` before greedy is redundant for the selected token, and `top_prob` scans the vocabulary again.                                                                        | High, unmeasured cost            | S / low       |
| 3        | Current matmul kernel dispatches each token column independently: `ggml_d3d12.cpp:763-778`, `shaders/ggml_d3d12_mmv_q4_k.hlsl:31-61`. Batch support does not establish near-constant cost or explicit weight reuse across tokens.                                                             | High, size of gain unknown       | M/L / high    |

## Phase 1: Restore a correct state machine

1. Fix the double trim before increasing the rollback window. A rejected suffix must be removed once. A corrective token recorded in history must still exist in target memory afterward. Cover first-token rejection, partial rejection, full acceptance, stop/EOG, cancellation, and generation-limit truncation. Distinguish an already-emitted pending token from a token already decoded into memory.

2. Derive target `cparams.n_rs_seq` from `mtp_n_max` when MTP is enabled, in both `run_inference_llama` and persistent `LlamaSession`. Reuse existing params; do not add a new configuration system just for this. Original references:
   - `llama.cpp/common/common.h:471-480`: `need_n_rs_seq()` returns `draft.n_max` for MTP.
   - `llama.cpp/common/common.cpp:3432-3449`: effective `n_batch`/`n_ubatch` must permit at least `n_rs_seq + 2`; otherwise the original switches to checkpoints.
   - `llama.cpp/src/llama-memory-recurrent.cpp:181-186,226-236`: bounded, single-use pending recurrent rollback.
     Log resolved values on target and draft. Merely changing the draft or setting an arbitrary positive number is insufficient.

3. Restore original Qwen MTP token/hidden-row semantics. Read these small sections in order:
   - `llama.cpp/tools/server/server-context.cpp:8480-8489`: `id_last = slot.sampled`, `pos0 = prompt.tokens.pos_next()`.
   - `:1647-1657`: that sampled anchor is subsequently included in the target batch `[anchor, drafts...]`.
   - `llama.cpp/common/speculative.cpp:4093-4191`: private-draft catch-up with shifted **target** hidden rows, then `verify_h`/`pending_h` tracking.
   - `:4273-4276`: draft input token, next position, previous target carry.
   - `:4421-4435`: accepted target row selected into carry.
     In the combined path the anchor has been sampled but not yet target-decoded; the carry describes the preceding decoded position. Current xllama first decodes `token` in `classic_step`, then passes that same token with its own hidden row at the next position. Do not copy that ordering into the corrected MTP path. Prove token IDs, positions, and hidden-row indices against the original for the first three rounds.

4. Implement the smallest single-sequence equivalent of `process`/`accept`, with explicit draft prefix alignment and checked suffix removal. Reuse the original helper if it fits the UWP build cleanly; otherwise port the required invariants rather than the whole server. Chunk catch-up batches to the small draft context's real capacity. Handle prefix jumps after declined drafts and corrections. Do not assert target/draft positions are equal when the designed anchor/carry relationship requires a one-position offset; assert the documented relation.

5. Preserve Qwen-specific constraints. `ctx_other` is cleared for Qwen at `llama-context.cpp:717-750`; draft memory is private, as live `shared=0` confirms. One head is supported (`models/qwen35.cpp:683`); do not add Gemma shared-position logic or multi-head chain machinery. The original target-only bootstrap at `common/speculative.cpp:4196-4200` requires managed/split weights, which this D3D12 port deliberately disables. Do not bypass that guard. Use the normal compatible path first, then consider a separately validated optimization.

6. Consolidate target verification into `[anchor, draft0, ...]` with all required output rows. After target decode, row 0 predicts draft0, row 1 predicts draft1, and the final accepted row yields the next correction/bonus token. Keep the next anchor pending when appropriate; do not immediately decode it and recreate the extra target pass. Restore target/draft carry after partial accept. The old comment at `decode_loop.h:252-255` reports a batch-versus-sequential mismatch: this is a required numerical/parity investigation, not a comment to delete and assume solved. Avoid changing unrelated prompt-lookup behavior unless a shared verifier bug requires it.

Gate: deterministic greedy token IDs equal no-MTP for the same prompt/settings; both contexts remain aligned after forced rejects at positions 0, 1, and the last drafted position; correction remains in state; no refused rewind, duplicate position, invalid hidden row, or silent fallback. Counter positivity alone fails this gate.

## Phase 2: Remove avoidable cost and measure the real bottleneck

- Reuse `GgufCpuThreadpools` (`src/bridge/llama_gpu.h:68-109`) for draft as well as target, with correct lifetime. If sharing a pool between sequential contexts, prove there is no concurrent graph use and detach/free safely; avoid two spin-waiting pools competing for the console CPUs. Measure draft wall time before and after.
- Put verbose draft logs and vector scans behind explicit diagnostics; production runs should aggregate timings once per generation. Keep errors visible. Remove redundant top-k ahead of greedy if token/tie behavior is unchanged; retain a correct confidence calculation.
- Accumulate draft attempt count, declined attempts, depth histogram, proposed/accepted tokens, tokens per target call, draft decode time, sampling/confidence time, catch-up time, target verify time, rollback/replay time, CPU/GPU synchronization time, and peak memory. Include attempts that returned no proposals.
- A 64-token generation at 4.28 tok/s takes about 14.95 seconds; at 18.43 it takes about 3.47 seconds. The approximately 11.48-second difference is not explained by citing an unmeasured 20 ms draft cost. Do not derive per-stage times from the overall TPS.
- `backend_free()` uses shared global GPU counters and resets them (`ggml_d3d12.cpp:709-721`). Do not interpret a second zero-counter destructor as proof one context did no GPU work. Measure per-phase deltas.

Gate: timers and counters account for the observed generation wall time within documented instrumentation error; no throughput claim relies on verbose diagnostic runs alone.

## Phase 3: Fit rollback storage without weakening the comparison

Original recurrent allocation is `n_rows = mem_size * (1 + n_rs_seq)` (`llama-memory-recurrent.cpp:101-103`). Given measured 50.25 MiB at zero, depth 2 adds approximately 100.5 MiB and depth 4 approximately 201 MiB for that component, before graph/scratch effects. These are allocation estimates, not a measured OOM or complete memory prediction.

Start at depth 1/2 and measure actual UWP app usage/limit, process working set, and DXGI current usage/budget separately. Test smaller prefill micro-batches, e.g. `--ubatch 64` and 128, against the measured 504.02 MiB reserve. Keep enough batch capacity for the rollback window. Report prefill/TTFT tradeoffs. Try q8 KV only as an explicit, matched A/B configuration and verify effective types/fallback behavior. Keep `n_ctx` the same in both arms; a smaller-context experiment needs a new matching baseline and explicit labeling.

The original also offers checkpoint/replay in `llama.cpp/examples/speculative-simple/speculative-simple.cpp` and `common/common.cpp`. Consider that measured fallback if bounded RS storage cannot fit. It may cost too much time; it is not automatically preferable. Never suppress a failed rewind or continue with stale recurrence.

## Phase 4: Make the target batch pay, if profiling requires it

The original algorithm can produce accepted proposals plus one additional target token in a cycle; depth 1 is not inherently slower. Reference: [Leviathan et al., algorithm 1 and section 3.3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf). The ideal formula assumes enough parallelism, which must be measured on this D3D12 implementation.

Measure the target's batch sizes 1, 2, 3, and 5. The current HLSL uses `col = gid.y` and each group loads its own weight blocks, so do not promise that verifying four tokens costs the same as one. If the correct loop plus pool/log fixes still lose, use the profile to implement a small-width multi-column quantized matmul with explicit weight/dequantization reuse for the actual Q4_K/Q5_K/Q6_K matrices. Reuse the existing backend, dispatch fallback, and numerical self-tests. Change only kernels contributing materially to measured time. Avoid implementing a full D3D12 operator stack or backend sampler before evidence shows that is needed.

Decision equation using measured values: `speedup = mean_committed_tokens_per_round * baseline_ms_per_token / total_round_ms`. Include declined drafts, catch-up, rollback, sampling, and synchronization. Conditional acceptance of 89% is not a speed estimate.

## Verification commands and deployment

The existing Linux workflow is:

```bash
cmake --preset linux-test
cmake --build build/linux-test --parallel 4
ctest --test-dir build/linux-test --output-on-failure
python3 scripts/check-coherence.py
```

Expected: exit 0 for the relevant build/checks and new state-machine regressions. Extend `tests/test_decode_loop.cpp` and the existing doctest conventions (`tests/CMakeLists.txt`) with meaningful rejection/correction tests; do not count the opt-in test as exercised if `XLLAMA_TEST_MODEL` is unset. Earlier OpenCode notes report a SIGILL in `test_session.cpp`; reproduce/classify it against the unchanged base or use a compatible build host, rather than claiming the full suite passed. Use the repository's pinned clang-format 22.1.5 on changed C++ and relevant shader/compiler checks. Linux success is not Xbox validation.

For Windows: inspect/reuse `XllamaMtpBuild` on .193, increment its real `BuildRevision`, preserve incremental objects, wait for terminal task/build success and complete MSIX/certificate, then use the existing deploy script. Record package version, code SHA, model hash, and real console hardware. The CSV `xbox-series-s` label is hard-coded at `uwp/inference-bridge.cpp:338`, so it cannot resolve the S/X ambiguity by itself.

After correctness passes, representative matched smoke commands are below. Run from the task worktree, with existing credentials loaded without printing them. `set -a` is needed if xbox-env contains assignments without exports.

```bash
set -a
source /home/hjotha/.config/xllama/xbox-env
set +a
./scripts/bench-xbox-ort.sh qwen35-4b-mtp --runs 4 --n-predict 64 --gpu-layers 99 --ctx 2048 --ubatch 64 --ignore-eog --out /tmp/opencode/mtp-reviewed-baseline.csv
./scripts/bench-xbox-ort.sh qwen35-4b-mtp --runs 4 --n-predict 64 --gpu-layers 99 --ctx 2048 --ubatch 64 --ignore-eog --mtp 1 --out /tmp/opencode/mtp-reviewed-n1.csv
./scripts/bench-xbox-ort.sh qwen35-4b-mtp --runs 4 --n-predict 64 --gpu-layers 99 --ctx 2048 --ubatch 64 --ignore-eog --mtp 2 --out /tmp/opencode/mtp-reviewed-n2.csv
```

Use fresh output files per package/revision; the script appends. Proceed to depth 4 only after depth 2 is correct and fits memory. For performance acceptance, interleave baseline/MTP runs, use at least three measured repeats after warmup, and extend to at least 256 generated tokens on three fixed prompts (normal prose, code, and repetition). Match model weights, prompt, context, threads, cache type, sampling, token limit, package, and diagnostics. Save actual token IDs or output hashes and full logs. Add q8 to both arms only for a separately labeled matched experiment.

## Completion and stop criteria

All are required for claiming the user objective achieved:

- Tests cover full acceptance, rejection at multiple depths, correction preservation, stops, length limits, and continued session state.
- Greedy output equals baseline and target/draft state is coherent on real Qwen3.5 Xbox runs.
- No OOM, aborted generation, silent MTP disable, or repeated-position error.
- Three retained repetitions per prompt/config, actual MTP activity, same effective settings in baseline/candidate.
- Median MTP decode speed exceeds the freshly matched no-MTP median consistently; target at least 10% margin to distinguish gain from noise. Report historical 18.43 tok/s separately, and do not conceal any fresh baseline regression.
- Archive commands, logs, raw samples, model/package/source identity, memory metrics, and generated-token counts; report TTFT/prefill tradeoffs.

On a failed gate, preserve evidence, fix its cause, and continue the next bounded experiment. Do not request another strategic decision for ordinary fixes/builds already authorized. Stop only the unsafe/failing path if state correctness or memory safety fails; never weaken assertions to make a benchmark finish. If measured kernel/memory limits still preclude speedup after the supported alternatives, report that unresolved blocker and the exact required next change honestly. A failed experiment is not permission to announce MTP success or declare the entire architecture impossible.

Scope: MTP bridge/verifier, both context setup callers, existing CPU pool helper, focused tests, benchmark telemetry, and necessary D3D12 hot kernels/build packaging. Unrelated app UI, diffusion, training, router/bot services, .57 occupancy work, whole-fork rebases, and speculative frameworks are out of scope. Do not rewrite generated OpenWiki or benchmark documentation by hand.

Maintenance: the accepted-prefix/token/hidden-row relation and one-time rollback are the core invariants. New decoding modes, sampler state, multi-turn reuse, or batch-kernel changes must rerun these gates.
