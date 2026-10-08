# SILU-on-D3D12 probe result (rev147, fresh run 2026-10-08T00:07:46Z)

Package: GianlucaMazza.xllama_1.6.0.147_x64__pj67f1cn14 (verified before deploy)
MSIX sha256: eb0c8cef3d4027e1f53dfa6f4632ab3b5fc0a8dee5fb4c8295600d33e9e512e1

Row:
`silu_avx2,9216,4,4,5.38e+07,0.0000,0.00,0,1,xbox-series-s,2026-10-08T00:07:46Z,finite=6999 edge=80 nonfinite=0 ulp_max=53824988 mxcsr=0x1fbb [...]`

## Verdict: THIS IMPLEMENTED PROBE is not bit-exact. Nothing broader is claimed.

- It is a failure of this specific implementation (explicit `mad()` + `-Gis`
  DXIL, direct AVX2 `ggml_vec_silu_f32` reference on the same thread, MXCSR
  0x1fbb with FTZ/DAZ off). It does NOT prove that all exact ports are
  impossible, and it does not isolate FMad vs division vs exp as the cause:
  the combined metric mixes ordinary and edge cases and the max ULP is
  edge-dominated.
- Canonical plan-005 acceptance is full-ID parity plus repeated product gain
  on the full model, not universal per-op bit equality. The probe's role is
  to bound the per-op drift before any wiring decision.

## Evidence kept

- `silu-selftest-rev147-d3d12be.csv` (row above; fresh process, unique date).
- finite mismatches 6999/36864, edge 80, non-finite 0; first diffs are 1 ULP
  in the ordinary range (e.g. i=9 in=0xbf541280 got=0xbe80eeaa
  ref=0xbe80eeab).

## Next bounded test (per review)

Extend the probe reporting: ordinary-domain max ULP / max abs / max relative
plus a ULP histogram, reported separately from extreme/subnormal cases; add
the actual SWIGLU multiply with randomized up values. Only if ordinary errors
are small and well bounded (no NaN/Inf drift) is the isolated contiguous
GPU-resident FFN SWIGLU wired behind a default-OFF knob and gated on
same-model deterministic OFF/ON full tokens plus MTP proposal/acceptance
comparison. If those fail or finite errors are not controlled, revert the
wiring and move to C5. No deeper exact-exp emulation.
