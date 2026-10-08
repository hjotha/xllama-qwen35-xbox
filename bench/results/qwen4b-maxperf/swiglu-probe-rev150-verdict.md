# Extended SILU/SWIGLU probe corpus (rev150, 2026-10-08T00:26:36Z)

Package: GianlucaMazza.xllama_1.6.0.150_x64__pj67f1fcj4n14
MSIX sha256: 31bc692a2d2b2ec89bd766a0e7d1fad170eef023bcfb9ca9af1047e0b8acc60e

silu_avx2,9216,4,4,0,0.0000,0.00,0,1,xbox-series-s,2026-10-08T00:26:36Z,mxcsr=0x1fbb ord=6999 ord_ulp_max=53824988 ord_abs=9.54e-07 ord_rel=1 hist0/1/2/3/4-7/8+=0/6954/45/0/0/0 edge=80 nf=0 ulp_max_all=53824988 [i=9 in=0xbf541280 got=0xbe80eeaa ref=0xbe80eeab] [i=13 in=0xc1413092 got=0xb8906eac ref=0xb8906ead] [i=22 in=0x40a8dcf0 got=0x40a80141 ref=0x40a80140] [i=29 in=0xc190957f got=0xb4895f2e ref=0xb4895f2d]

swiglu_avx2,9216,4,4,0,0.0000,0.00,0,1,xbox-series-s,2026-10-08T00:26:36Z,mxcsr=0x1fbb ord=6095 ord_ulp_max=62212381 ord_abs=1.01e+31 ord_rel=1 hist0/1/2/3/4-7/8+=0/5283/807/5/0/0 edge=180 nf=0 ulp_max_all=62212381 [i=6 in=0xc1799f6b got=0xb5c2f4f6 ref=0xb5c2f4f7] [i=11 in=0xc0a444cc got=0xbd2edbde ref=0xbd2edbdf] [i=20 in=0xc08825b6 got=0xbd3e2c4a ref=0xbd3e2c4b] [i=23 in=0xc13127c2 got=0xb9a2e023 ref=0xb9a2e022]

## Reading (ordinary domain, separated from edge/subnormal)

- SILU: ordinary mismatches 6999 with ULP histogram 1/2/3+ = 6954/45/0;
  ordinary max |diff| 9.5e-07; no NaN/Inf drift (nf=0); edge=80.
- SWIGLU (randomized up): ordinary histogram 1/2/3+ = 5283/807/5; nf=0;
  edge=180. Absolute differences scale with the huge adversarial operands
  (3.4e38 up) while the ULP/relative error stays at 1-3 ULP.
- MXCSR at the direct AVX2 reference call: 0x1fbb (FTZ/DAZ off).
- The large ord_ulp_max values come from near-zero/edge denominators; the
  per-class histogram is the authoritative ordinary-domain view.

## Gating decision per review

Ordinary errors are small and bounded (1-3 ULP, no NaN/Inf drift), so the
review authorizes wiring ONLY the isolated contiguous GPU-resident FFN
SWIGLU behind a default-OFF knob, followed by same-model deterministic
OFF/ON full tokens plus MTP proposal/acceptance comparison. Those
full-model gates are mandatory; on failure or uncontrolled finite error,
revert the wiring and move to C5.
