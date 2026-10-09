# Final completion — Qwen 4B Xbox max-perf (rev160 LTCG)

## Delivered

- **C8 GPU-timestamp gating** (measured +1.6 to +3.2% per arm in its own A/B,
  rev131): profile-bound default — timestamps off unless profiling is on.
- **FFN SWIGLU on D3D12** behind the default-OFF knob `d3d12swiglu.txt`
  (`1` = all FFN, `2` = target-only), narrow whitelist (prefix, `ne0==9216`,
  dims, owned buffers, `count%256==0`), exact-group dispatch, UAV barrier.
- **Process-bound profile policy** (`SwigluModePolicy`): first context binds
  the capability for the process lifetime (a restart re-evaluates it); a later
  conflicting MTP context is rejected before creation ("restart required"); no
  live-context change and no dropped scheduled ops. Host latch tests 16/16.
- Diagnostics (profile-gated): `DSPLIT`/`CSTEP`/`DSTEP`/`VROUND feed=`,
  timestamps counters; host suites green (ctest 1/1, formatters clean).

## Final build and gates (package `…_1.6.0.160_…`, MSIX

`74C9DEDDA15493A6673353F08E29A77EE16D21A124783BA023DC8956A03EF1C7`, LTCG)

- Full-token parity vs baseline digests, knob OFF/ON, runs 1-2: **five
  sequential pairs** (chat64, code64, chat256, code256, std512) plus **three
  MTP pairs** (chat64, code64, chat256). code256 and std512 were not run on
  the MTP arm in the final grid.
- Seq FFN timing (median of 2, profile OFF, OFF→ON): chat64 +9.81%, code64
  +5.62%, chat256 +6.84%, code256 +5.19%, std512 +6.01%. Prior gate A/B
  (rev154, labels not swapped): chat64-seq **+6.53%**, code64-seq **+5.65%**.
- Termgate 6/6 arms `ok=1`, sw0/sw1 byte-identical; session gate 4/4; API
  PASS (seq preset and production MTP profile).
- MTP arm with the knob ON binds FFN off (no dispatch; deltas −0.55/+0.12/
  +0.61%, noise).

## Attribution correction

The rev154 MTP rows (chat64 +4.42%, code64 +3.61%) were **FFN OFF/ON trials
with C8 held fixed** gate-wide. They are the rejected FFN candidate's numbers
— invalidated by the chat256 round-49 proposal divergence — and are **not C8
evidence**. C8's gain stands on its own rev131 A/B.

## Candidate disposition (bounded cycle, complete)

Final rev160 dispositions — engineering confidence from the measured evidence
in this cycle, not a universal maximum proof:

- **C8 timestamp gating** — enabled (profile-bound default; own A/B rev131:
  +1.6 to +3.2% per arm).
- **C1 batch/verify thread split** — rejected (retained threads 2, batch 2).
- **C2 draft threads** — no robust gain (retained `mtp_threads=1`).
- **C7 joint n_batch/n_ubatch 32/32** — rejected (prefill −6.6%; retained
  64/64).
- **Q6 tile 2** — rejected (no gain / regression on the Q5_K kernel).
- **C4 seq FFN SWIGLU** — optional, validated (default OFF); MTP arm rejected
  (chat256 round-49 proposal divergence).
- **C3 async waits, C6 shared-nextn** — higher risk; deferred, not
  implemented.
- **C5 top_prob max-scan AVX** — tried and closed (rev161): bit-exact,
  isolated scan 2.6x, but only −2.4/−2.5 ms/run on console (exp/sum-bound,
  below noise); reverted, patch preserved in
  `bench/results/qwen4b-maxperf/c5-rev161/`.

Remaining research is explicit and bounded; the cycle is complete and no
further implementation is planned here.

## Source reproducibility

Parent commit `d960126` alone does not pin the final source: `git status`
reports `m llama.cpp` (modified submodule content). Recorded in
`source-rev160/`: submodule HEAD
`982eaadaa9dbe761f00205bc20b6c04b7329b58d`, porcelain status + diffstat, and
the binary-safe worktree diff `llama-cpp-worktree.patch` (sha256
`7cd11abc97cc2bb4fe196fb73a4465633f1b7d5aa5b7fa97b44184478b4eb144`; per-file
sha256 in `llama-cpp-sha256.txt`). Reproduce with `git checkout <HEAD>` +
`git apply --binary`.

Excluded raw evidence logs are preserved and gzipped on disk (uncommitted
because of size); the filesystem has ~20 GiB free.

## Limitations (remaining experimental)

- `d3d12swiglu.txt` stays **default OFF**; the sequential FFN profile is
  experimental and unpromoted.
- The profile is fixed for the **process lifetime** (until restart);
  enabling requires the seq preset, and an MTP context cannot be created
  while the seq profile is bound.
- **MTP FFN is closed**: the target-only discriminator still diverges from
  OFF at chat256 round 49 (final IDs exact in all arms).

## Device state (restored)

Production MTP profile: `llama.ini` mtp=2, `d3d12swiglu.txt=0`, profiling
knobs absent, startup scope2 profile (`d3d12twocol.txt=auto`,
`cpurepackforcegemv.txt=2`). API verified PASS on the installed package.
Optional seq preset preserved: `presets/llama-ini-mtp0-seq.txt` (see
`presets/README.md`).

After the C5 trial the device ran the same verified source as the final
accepted-source package 1.6.0.163 (full Release LTCG), then 1.6.0.169 after
the plan-007 RMS_NORM+MUL chain was measured and rejected, and is now
**1.6.0.175** (full Release LTCG) carrying the plan-008 island code with
`d3d12island` default OFF (measured structural win, timing not reproduced).
UWP blocks downgrades and uninstalling is forbidden — model data lives in
LocalState; interim fast-iteration rebuilds were 1.6.0.162, 1.6.0.164..168 and
1.6.0.170..174. API, key token parity and the termination gate passed on
1.6.0.163, 1.6.0.169 and 1.6.0.175: receipts in `restore-rev163/`,
`release-rev169/`, `release-rev175/` and `c5-rev161/restore-receipt.txt`.

Evidence: `bench/results/qwen4b-maxperf/` (`acceptance-rev160/verdict.md`,
`seqgates-rev159/`, `swiglu-*/`, README). No publish or merge.
