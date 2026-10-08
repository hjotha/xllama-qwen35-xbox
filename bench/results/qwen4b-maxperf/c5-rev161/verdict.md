# C5 `top_prob` max-scan AVX — bounded experiment, closed (no practical gain)

- Package tried: `GianlucaMazza.xllama_1.6.0.161_x64__pj67f1fcj4n14`, MSIX sha256
  `823263EA5E57213769708AECCAA0F8D1FE669254DEC1271439AD3E20159ADC04` (fast
  iteration build, no LTCG). Reverted afterwards; device back on rev160.
- Complete candidate patch preserved: `c5-candidate.patch` (537 lines, 8 file
  entries — the five tracked edits plus the three new files
  `include/xllama/mtp_top_prob.h`, `src/bridge/mtp_top_prob.cpp`,
  `tests/test_mtp_top_prob.cpp`), sha256
  `eb74b69c5415f46b423da73a33457759183079bac146604c716cd5b32ca96358`
  (independently verified). Harness `verify-avx-harness.cpp` (sha256 prefix
  `d57930e8114461fc…`).

## Scope (as bounded by review)

Only the max scan changed; exp, float subtraction, scalar double accumulation,
the early exit and the limit/return are byte-for-byte the original code
(assembly of the built object shows `vcvtss2sd` → `call exp` → `vaddsd` →
`vcomisd`/`ja` in order).

## MSVC assembly (built object, `mtp_top_prob.msvc.disasm.txt`)

`mtp_logits_max_avx` in the rev161 object:

- `vbroadcastss ymm4, xmm2` (seed), loop body `vmovups ymm1,[rcx]` +
  `vcmpgt_oqps ymm0,ymm1,ymm4` + `vblendvps ymm4,ymm4,ymm1,ymm0` (8 elements
  per step), lane fold with `vextractf128`/`vshufps`/`vmaxss`, scalar tail.
- The scalar function stays the 4x-unrolled `vmaxss` chain (no packed max).

## Isolated harness (real `mtp_top_prob.cpp`, /arch:AVX host, run on .193)

- `c5_avx_checks=4697 fails=0` — bit-exact vs an independent scalar reference
  for first-NaN sticky, later-NaN ignored, ±inf, signed-zero lane-order
  crossings, tails 1..33, nextafter p_min boundaries around the returned
  probability with n > 8 (both sides of the gate crossed).
- Timing: scan 185.0 → 70.7 µs (**2.62×**), core 185.8 → 69.4 µs (**2.68×**).
  (AVX1 host; the console is a wider AVX2 core.)

## On-console A/B (rev161, MTP, chat64/code64)

- Knob confirmed applied in the bench path: log
  `main_loop: bench_topprob_avx.txt='1' -> max scan avx`.
- Per-round traces: own-segment VROUND `feed=`/`accepted=` sequences
  byte-identical, scalar vs AVX (chat64 23 rounds, code64 21 rounds).
- Final token IDs identical to the baseline digests in every run
  (chat64 `84e08196…`, code64 `ff42c5be…`).
- `top_prob` attribution (profile ON, single runs): chat64 55.5 → 53.1 ms
  (−2.4 ms), code64 64.3 → 61.8 ms (−2.5 ms) — i.e. the max scan is only a
  small fraction of `top_prob` on real logits; the exp/sum scan dominates.
- End-to-end paired timing (profile OFF, interleaved, 6 runs/arm/cell):
  - chat64: scalar mean **28.8666667** / median 28.895 vs AVX mean **28.99** /
    median 29.025 → mean **+0.4273%**, **ranges overlap**.
  - code64: scalar mean **34.3033333** / median 34.305 vs AVX mean **34.73** /
    median 34.805 → mean **+1.2438%**, ranges touch at 34.38 (one scalar run
    equals the AVX minimum).
  - The 2.4-2.5 ms `top_prob` saving is ≈0.1% of decode and cannot explain
    the larger end-to-end deltas; no causal separation from run noise.

## Verdict

**Closed: no demonstrated practical gain.** Bit-exact and real SIMD, but the
recoverable time is bounded by the max-scan share of `top_prob` (~4% on the
console logits), which is below end-to-end variability. Candidate edits
reverted in the worktree and the build mirror (C5 symbols absent; commit
history unchanged). Default stays OFF and no LTCG was spent.

**Restore (see `restore-receipt.txt`):** UWP refuses the 1.6.0.160 downgrade
against the installed 1.6.0.161 and uninstalling is forbidden (model data
lives in LocalState), so the verified rev160 source was rebuilt as
**1.6.0.162** (fast iteration, sha256
`249E4C3DBA078E00F3AAD90F666418647650D482C6AE4E33FFF17B46A5E3104C`, builder
sha == fetched). Production profile restored and verified: `llama.ini` mtp=2,
`d3d12swiglu.txt=0`, bench/termgate/profiling knobs absent, startup profile
twocol=auto repack=2; `validate-api.sh chat` with `MODEL=qwen35-4b-mtp`
returned PASS (`FFN SWIGLU profile bound: off (mtp_capable=1)`, no C5 knob
lines). Residual bottleneck: `top_prob` is dominated by the
scalar double `exp` accumulation with the order-preservation constraint, plus
the known verify-side ceilings (verify ≈64.5% of decode, classic ≈20.3%,
draft ≈8.5%, catchup ≈3.1%; d3w ≈50% with a ~50% CPU-side/submission split).
