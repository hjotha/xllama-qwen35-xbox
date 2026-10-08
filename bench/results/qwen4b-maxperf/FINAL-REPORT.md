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

- Full-token parity vs baseline digests on every cell (chat64/code64/chat256/
  code256/std512 × seq/MTP × knob OFF/ON, runs 1-2).
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

Evidence: `bench/results/qwen4b-maxperf/` (`acceptance-rev160/verdict.md`,
`seqgates-rev159/`, `swiglu-*/`, README). No publish or merge.
