# SWIGLU exact-port feasibility (rev135 toolchain, dxc 1.8.2502.11)

- utc: 2026-10-07T22:10Z (initial, now corrected), 2026-10-07T22:2xZ (mad)
- **Verdict: NOT YET PROVEN — pending the bounded explicit-mad device corpus
  test.** The earlier "infeasible" verdict was premature (it grepped for `fma`
  text in ordinary mul/add DXIL).

## Compiler evidence (corrected)

- Plain `a*b+c` chains: dxc emits unfused `fmul`/`fadd` (default: all ops
  `fast`, `fdiv fast`; `-Gis`: no fast flags, still unfused). Zero fma text.
- **Explicit float `mad(a,b,c)` + `-Gis`**: dxc emits **10×
  `dx.op.tertiary.f32` (FMad) with `!dx.precise`** — the fused float
  multiply-add matching the CPU's `_mm256_fmadd_ps`/`_mm256_fnmadd_ps`
  chains — and precise `fdiv` for the silu division. Artifacts:
  `swiglu_mad.hlsl`, `swiglu_mad.dxil.txt` (see lines with `dx.op.tertiary`).
- The HLSL `fma()` intrinsic is double (do not use); `mad()` is the relevant
  intrinsic (Microsoft: mad targets hardware MAD, may be fused or unfused
  consistently on that hardware). `-ffp-contract=fast` is not in this dxc.

## Remaining unknowns that only the device corpus can settle

1. Whether the Xbox driver executes FMad fused for these operands (and the
   divide rounds like `_mm256_div_ps`).
2. MXCSR FTZ/DAZ on the CPU side vs GPU subnormal handling (no FTZ/DAZ
   manipulation found in ggml-cpu; Windows default is off).
3. Signed zero, finite extremes, subnormals, and the polynomial's branch
   boundaries (|n| ≈ 126/192) in the actual hardware.
4. Scalar tails: FFN dims are 8-divisible, so the vector path should cover
   the whole row; the corpus must confirm no tail divergence.

## Next step (bounded)

Device microkernel corpus: build a `ggml_silu` (and then `ggml_swiglu`) case
into the existing `d3d12be` selftest, CPU reference on the same device, GPU
via the FMad kernel, report max ULP, bit-mismatch count and non-finite
behavior on a real-range + adversarial corpus. Default OFF, no broad
production wiring until the corpus result; preserve existing barriers and
strict shape/alias checks.
