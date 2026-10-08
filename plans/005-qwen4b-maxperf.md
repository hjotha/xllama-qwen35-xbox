# Plan 005 — Qwen 4B on Xbox: maximum practical performance (MTP and non-MTP)

Status: active. Owner: supervisor session on optiplex (.157), building on .193,
measuring on the Xbox console. No push/PR/merge/release in this task.

## Goal

Establish the strongest **measured, repeatable** end-to-end Qwen 4B decode
performance practically supportable today on the console, for both the classic
(no-MTP) and MTP paths: optimize the dominant measured bottleneck one at a time,
keep only statistically/repeatably beneficial changes, and deliver clean
reviewable commits, maintained patches for `llama.cpp`, and a full evidence
report. "Maximum practical" means evidence-bounded, not theoretical.

## Verified starting point

- Repo: `hjotha/xllama-qwen35-xbox`, `main` = `a773e581182c05cddb608ee6aa63d9434a2b382d`
  (squash merge of PR #1, "perf(mtp): validated Xbox MTP and incremental
  Windows builds"), equal to live `origin/main`. Includes the rev124 campaign
  work (Q8/Q4/Q6/GDN D3D12 kernels, two-column tiles, FA2 control, MTP carry,
  incremental Windows build scripts `scripts/build-xbox-local.ps1` +
  `scripts/run-xbox-build.ps1`, `build-uwp.ps1 -Iteration`).
- Work branch for this plan: `feat/qwen4b-maxperf`, new worktree
  `/home/hjotha/worktrees/xllama-qwen35-4b-maxperf`, based on `main`.
- Spike worktree `/home/hjotha/worktrees/xllama-mtp` (branch
  `spike/beellama-pin`, dirty `llama.cpp` submodule) is **preserved untouched**.
- Console identity verified 2026-10-07 via Device Portal `/api/os/info`:
  `Platform: Xbox Series X` (ComputerName XBOX, SystemOS
  26100.9438.amd64fre.xb_flt_2608ge.260902-1030). Historical "Series S" host
  tags in old CSVs are wrong for this console; new rows record actual identity.
- Installed baseline package: `GianlucaMazza.xllama_1.6.0.124_x64__pj67f1fcj4n14`,
  MSIX sha256 `04b5bae6186a501916190f453c675619a883ef63215b36c4bff13b0a9ab860ea`.
- Models (LocalState/models): `qwen35-4b` and `qwen35-4b-mtp`.
  MTP GGUF sha256 `3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630`
  (2,834,975,040 B), fnv `f3559ed1a2922585`; non-MTP GGUF sha256 to be recorded
  before the non-MTP arm is claimed. Model/quantization is frozen for the plan.
- Durable prior receipts: `bench/results/fast-mtp-20261007/`
  (`final124-acceptance.json`, `diagnosis.json`, `final-iteration-cache-proof.json`).
  rev124 delivered profile: gpu 34/34, ctx 2048, batch 64/ubatch 64, F16 KV,
  threads 2, mtp depth 2 / pmin 50, GDN D3D12, Q8 GPU, q6tile 0, FA2,
  twocol auto, scope2; seq ≈ 22.8–23.8 tok/s, MTP 27.3–34.0 tok/s
  (prose/code, 64/256), full-ID parity.

## Build path and prebuild invariant

- Iteration builds on .193 reuse the warm Release object cache of
  `C:\Users\hjotha\worktrees\xllama-mtp-fast` (worktree of the qbox repo,
  content = main modulo known per-file diffs). Target ≈41 s warm vs ≈712 s
  full (LTCG) final.
- Build driver: scheduled task `XllamaMtpBuild` ->
  `scripts/build-xbox-local.ps1 -BuildRevision N` (wrapper
  `scripts/run-xbox-build.ps1`), absolute cwd, marker `exit.txt` = 0 required.
- **Prebuild invariant, every build:** (1) commands run from an absolute cwd;
  (2) source transfer exit 0; (3) local and builder SHA-256 equal per transferred
  file; (4) the exact new symbol/string of the change is present in the builder
  tree before the build starts; (5) record the built MSIX sha256 + PFN.
- Backend (`llama.cpp/`) changes are maintained as `patches/0NNN-*.patch`
  applied by `scripts/apply-uwp-patches.sh`; never committed inside the
  submodule.
- Only the final candidate may use a full/optimized (LTCG) rebuild; iteration
  builds stay lightweight. Final candidate is re-verified on the console.

## Measurement protocol (both arms unless stated)

- Runner: `scripts/bench-xbox-ort.sh` (existing), device knobs hash-recorded.
- **Primary comparisons toggle MTP only on the exact same `qwen35-4b-mtp` GGUF**
  (sha256 `3874209241c9a397...`), same profile: that is the MTP-off vs MTP-on
  pair and the only sequential correctness reference for the MTP model. The
  separate `qwen35-4b` GGUF is an additional, separately labeled workload
  (never the sequential reference for the MTP model); its sha256 is recorded
  before any claim that uses it.
- Profile OFF (`--profile-phases 0`) for every timing claim; profile ON pairs
  only for attribution, with overhead quantified separately.
- Deterministic: `--greedy --seed 1 --tokens`, full-token sha256 parity per
  run index; fixed probe set: `spec-chat-open` (prose) and `spec-code-edit`
  (code) at 64 and 256 tokens, short/long prompts; multi-turn/session reuse via
  `--mtp-session`; termination via the durable termgate driver.
- Interleaved A/B repetitions (≥2 measured rows/arm per block, ≥2 blocks for a
  claim), medians and spreads reported; TTFT, prefill tok/s, decode tok/s,
  p50/p95 token latency where measurable, acceptance/catch-up cost, peak WS,
  warm/cold distinction.
- Acceptance gate for retaining a change: repeated paired product gain with
  full-ID parity under deterministic settings, or a clearly bounded optional
  knob documented with its measured effect. Failed experiments are documented
  and reverted without touching unrelated work.

## Phases

0. **Recon + fast-build proof** (this file, status file, branch/worktree,
   builder content equality, one iteration build + deploy + smoke).
1. **Baseline reproduction + profile**: seq vs MTP, prose/code, 64/256, profile
   OFF timings and one profile ON attribution pair on the built baseline;
   reconcile phases (prefill, verify, draft, catchup, sample, copies/submit).
2. **Bottleneck ranking**: from measured profiles only (CPU graph per ctx,
   GPU graph, d3w/submission+fences, copies, verification widths, draft cost).
3. **One-at-a-time experiments** targeting the top bottleneck, chosen from
   existing evidence and tools first: attention CPU fallback cost, small-batch
   verify cost, quantized matmul/GDN tiles, transfer/sync overhead, prefill
   batching, draft depth/pmin policy. Each experiment: prebuild invariant,
   interleaved paired measurement, parity, retain/revert decision.
4. **Final candidate**: complete optimization set, full LTCG rebuild, console
   acceptance grid (both arms, 3 prompts × 64/256, session, termination, API),
   host suite + formatters + coherence, evidence report, clean commits.

## Stop rules

- Stop and report when the dominant measured bottlenecks are addressed, the
  strongest supported configuration is established, or a concrete
  hardware/API/access blocker needs the user.
- Never trade correctness for speed, never hide fallback, never relax asserts
  or claim untested numbers.
- Console defaults/knobs are restored to the chosen candidate profile after
  runs; model data and app install are never wiped.

## Experiment log (chronological)

1. **Baseline rev126** (warm no-change build 40.93 s): full-ID-exact MTP/seq
   pairs; MTP gains +16-47% decode; prefill MTP −7-8%; attribution recorded in
   `bench/results/qwen4b-maxperf/README.md`.
2. **rev127 host-tag fix** (changed-source iteration 109.47 s): verified
   `xbox-series-x` tag, token parity unchanged.
3. **Q6 tile A/B** (rev127, knob-only): tile 2 rejected (−1.69% chat, −1.95%
   code); tile 0 retained.
4. **Prefill ubatch sweep** (rev127, knob-only): 128/256 regress prefill
   (−13%/−29%); 64/64 retained. u256 MTP cell truncated by the console going
   offline mid-cell (not a workload slowdown); runner wait helper fixed.
5. **Same-boundary decode instrumentation** (`DSPLIT` whole-decode d3w/d3g/cpu,
   `CSTEP` classic steps, `DSTEP` draft/catch-up steps — all profile-gated).
   rev129 built it first; the rev130 measurement-integrity fix (mandatory
   profile argument, prefill policy ordering, unavailable-sample semantics)
   superseded it before any device run. rev130 was deployed and measured by
   the independent reviewer (`rev130-reviewed-dsplit/`, 19:08:27Z, parity OK):
   chat64 OFF seq 23.81 / MTP 28.22 tok/s; DSPLIT seq d3w ≈54% / d3g 29%,
   MTP d3w ≈53% / d3g 34-35%; profile-ON overhead within noise in that sample.
   Timing comparisons stay profile OFF.
6. **Blocker (2026-10-07 ~17:31Z, resolved 18:13Z):** console .26 went offline
   mid-run and the owner restored it; no power/network changes and no repeated
   relaunches. All speed claims remain pending repeated console tests.
7. **Offline candidate analysis (2026-10-07):** `plans/006-qwen4b-maxperf-candidates.md`
   ranks the next levers with measured vs static evidence, mechanism, risk and
   A/B/parity validation. Top items: C8 (disable per-graph GPU timestamp
   queries unless profiling is ON — paid on every graph_compute today), C1
   (decouple `n_threads_batch` from decode threads, using the measured t6
   prefill win without the t6 decode loss), C2 (draft threads), then the
   structural options (deferred D3D12 wait, D3D12-side ops, catch-up
   elimination). No code built from it yet.
