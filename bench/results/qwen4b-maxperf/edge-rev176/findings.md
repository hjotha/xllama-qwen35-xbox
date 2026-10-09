# SWIGLU/SILU adversarial-edge diagnosis (owner 591/599) — causal finding

## Formula comparison (shader vs exact CPU AVX2 source)

Compared `shaders/ggml_d3d12_silu.hlsl`/`_swiglu_probe.hlsl` against the pinned
`ggml_v_expf`/`ggml_v_silu` (vec.h:1215/1255): coefficients
(`0x1.715476p+0f`, `0x1.7f7d1cp-20f`, `0x1.62e4p-1f`, p0..p4), the nested
`fmadd` tree for `j`, the `e/k` bit construction, the `|n|>126` / `|n|>192`
branches and the `g/s1/s2` fallback are all faithful — **no transcription
bug**.

## Fusion semantics — measured (rev177 selftest, dual reference)

The selftest now compares the GPU output against BOTH a fused reference
(`ggml_vec_silu_f32`, hardware FMA) and an unfused emulation (identical
formula with `volatile` mul/add intermediates):

| reference          | ordinary mismatches | ord max ULP | ord abs  | ord rel  |
| ------------------ | ------------------- | ----------- | -------- | -------- |
| fused (CPU kernel) | 6999                | **2**       | 9.54e-07 | 1.97e-07 |
| unfused emulation  | 36483               | 2.2e9       | large    | large    |

The GPU matches the **fused** semantics by a wide margin: the Xbox driver
lowers `dx.op.tertiary` (`mad()`) fused. The residual is therefore NOT fusion.

## Proven bug found and fixed — the diagnostic metric, not the kernel

`probe_metrics` declared `ulp_max_ord`/`maxabs_ord`/`maxrel_ord` but never
assigned them, so every report read `ord_ulp_max=0` and the C4-era "ordered
region bit-exact" impression was an artifact. Fixed (rev177): the correct
picture is **silu 6954 mismatches at 1 ULP + 45 at 2 ULP** (relative error
~2e-7) on ordinary inputs, with the large ULP numbers confined to edge inputs
(subnormal-scale outputs / >1e30 magnitudes, `nf=0`: sign/NaN/Inf
classification still matches). The swiglu row produced by the same run is not
present in the CSV (row longer than the 512-byte row buffer — diagnostic
tooling item, not a kernel result; the same arithmetic is covered by the silu
row).

## Consequence

- SILU is selftest-only; SWIGLU reaches product only via the C4 knob (or the
  island knob), so **production (knobs OFF) is unaffected**.
- The island's own ADD/RMS_NORM rows remain bit-exact (`bit_mismatch=0`;
  `exact=36864/36864`).
- Any future seq-FFN promotion carries a documented ≤2 ULP ordinary deviation
  (never token-flipping in the tested gates, but the strict bit-exactness claim
  is dead).

## Next measured gate (open)

Equal-history FFN/MTP diagnosis: reproduce the C4 divergence with a
diagnostic knob that lets FFN SWIGLU run in MTP, and capture at the first
divergent round the target row top-k/margins, the draft proposal distribution,
the committed history ids, the KV rollback span and the RNG state — with the
strict per-round trace gate retained.
