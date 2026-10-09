# FFN/MTP audit + decode budget (CORRECTED, rev188/rev190)

Supersedes all earlier versions of this document. Every number below is from
device runs on this console; raw sources are named per section.

## 1. p_min semantics (code-confirmed)

Draft loop order: sample -> `if (p < p_min) { ++n_discarded; break; }` ->
push_back. On failure the failing candidate and the round's suffix are NOT
proposed; the valid prefix already proposed is verified normally. p_min touches
no target state. Committed tokens = accepted drafts (each equal to the target's
own draw) + the correction token on the first rejection (that same draw).
RNG: drafter is greedy (no RNG); the target sampler draws once per verify row
it samples - with temp>0 the RNG stream differs from sequential decoding even
though every emitted token is the target's own draw. Universal bias-freeness is
NOT claimed: the conditional distribution is the target's modulo
batched-vs-sequential logit ULPs, and the stream difference is disclosed.
Demonstrated: identical 256-token streams across the FFN-induced p_min
crossing (OFF p_full=0.4998874068, -1.126e-4 below; ON 0.5000532866, +5.33e-5
above). Source: `mtpdiag-rev179/logs20`, `logs18`.

## 2. MTP correctness vs FFN numerical equivalence (separated)

- FFN kernel vs CPU: <=3 ULP ordinary (rel 2.6e-7), fused-mad semantics proven
  by the dual reference; token streams identical on 64/256/512-token gates
  seq+MTP.
- MTP correctness: committed = target draws; zero-accepted/rejection/rollback
  sequences identical (23 each, in-order correction tokens equal) and the
  termgate cancel/stop/EOS 6/6 byte-identical to baseline with FFN active in
  MTP. Sources: `ceiling-rev188`, `termgate-ffnmtp-rev188`, `logs20`.

## 3. SWIGLU real input range (device, rev189/190; gate and up separate)

| run | operand | samples | max |x| | 30-87 | 87-126 | 126-192 | >192 | subn | nf | pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seq256 | gate | 110.6M | 21.8 | 0 | 0 | 0 | 0 | 0 | 0 | 8256/0 skip |
| seq256 | up | 110.6M | 16.3 | 0 | 0 | 0 | 0 | 0 | 0 | 8256/0 |
| mtp256 | gate | 125.0M | 21.8 | 0 | 0 | 0 | 0 | 0 | 0 | 4743/0 |
| mtp256 | up | 125.0M | 24.7 | 0 | 0 | 0 | 0 | 0 | 0 | 4743/0 |
| mtp512 | gate | 211.4M | 21.8 | 0 | 0 | 0 | 0 | 0 | 0 | 9153/0 |
| mtp512 | up | 211.4M | 24.7 | 0 | 0 | 0 | 0 | 0 | 0 | 9153/0 |

~894M samples (seq + MTP draft/verify, 256/512-token contexts), `pairs_skipped=0`
(full coverage of the product dispatches in these runs). The observed range is
(-24.7, 24.7); within it the <=3 ULP ordinary deviation is the applicable
class. The synthetic selftest corpus keeps the extreme branches covered as a
boundary test; their non-appearance in these runs is an observation of this
workload/context, not a universal exemption, and p_min distance is not used as
an equivalence argument anywhere.

## 4. Same-run decode budget (CORRECTED) - source `ceiling-rev188/logs-prof1-r1`

Run: chat256 MTP, rev188 LTCG, profile ON; decode wall = 256/27.44 = 9329 ms.

| phase                                  | ms     | share |
| -------------------------------------- | ------ | ----- |
| verify (VROUND wall; sum == verify_ms) | 5826.8 | 62.5% |
| classic (CSTEP, 37 steps)              | 1636.3 | 17.5% |
| draft                                  | 835.6  | 9.0%  |
| catchup                                | 281.4  | 3.0%  |
| topprob                                | 237.5  | 2.5%  |
| corrective                             | 218.8  | 2.3%  |
| sample                                 | 117.1  | 1.3%  |
| residual (unattributed; disclosed)     | 175.5  | 1.9%  |

Token accounting (same run): 93 rounds, 357 decodes, 76 discarded, 251
catchup tokens, 256 emitted (~2.75 emitted/round). Instrumentation cost
measured -2.07% decode (acceptance runs OFF).

## 5. Kernel rates at real shapes and widths (existing selftest, source

`edge-rev176/d3d12be-rev177.csv`)

| format | shape      | 1 col | 2 col | 3 col | 5 col | marginal us/col |
| ------ | ---------- | ----- | ----- | ----- | ----- | --------------- |
| q4_k   | 11008x2048 | 220.2 | 122.0 | 84.4  | 51.9  | 46.4-46.7       |
| q6_k   | 2048x11008 | 199.2 | 107.5 | 71.1  | 44.6  | 79.3-83.7       |
| q8_0   | 2560x5120  | 197.8 | 108.1 | 74.3  | 45.0  | 58.4-60.1       |
| q5_k   | 11008x2048 | 225.0 | -     | -     | -     | gap             |
| q4_k   | 2048x11008 | 208.3 | -     | -     | -     | gap             |
| q6_k   | 65536x1024 | 227.5 | -     | -     | -     | gap             |

GB/s packed; the marginal cost per added column is approximately a full weight
pass (no amortization across columns, in both the OLD and the two-column
paths). The workload's verify (widths 2-5) therefore runs at the low-width
rates, explaining the 48.63 GB/s effective (1.01 GiB weights / 22.3 ms d3g).
Missing width rows for q5_k and two shapes are listed as a measurement gap.

## 6. Candidate wall fractions (same run)

| candidate                                                              | wall fraction |
| ---------------------------------------------------------------------- | ------------- |
| verify GPU matmul (d3g 2104 ms, MMV-dominated, no column amortization) | 22.6%         |
| classic GPU (449 ms)                                                   | 4.8%          |
| verify sub+fence (1011 ms)                                             | 10.8%         |
| classic sub+fence (415 ms)                                             | 4.4%          |
| outside-d3w verify (2712 ms) + classic (773 ms)                        | 37.4%         |
| draft + catchup + corrective + sample + topprob                        | 18.1%         |
| residual                                                               | 1.9%          |

The 47% outside-d3w bucket replicates in the independent classic path; the
classic path's submission share is higher (25% vs 17% of its own wall). No
claim that scheduling is eliminated, and no single-lever claim.

## 7. Projections (corrected)

Verify GPU accelerated 2.5x -> 31.74 tok/s; 3x -> 32.30; verify GPU instant ->
35.43. These are verify-GPU-only projections at the current phase costs; 35-45
tok/s is NOT a demonstrated ceiling, and ~45 would require measured gains in
other phases as well.

## 8. Gate proposal (for explicit review; strict gate unchanged)

Token/state criteria, margins disclosed: (a) final token IDs identical;
(b) committed prefix identical up to a divergence; (c) any divergent decision
must be a p_min branch whose gate distance lies within the measured per-pair
inter-arm drift (225 pairs: max 1.3e-3, p99 8.0e-4, p95 5.6e-4, median 1.1e-4;
the one flip at -1.13e-4/+5.33e-5, arm delta 1.66e-4), reported explicitly per
case; (d) KV accounting/rollback consistent; (e) greedy runs deterministic,
stochastic runs disclose the RNG-stream difference. Not applied.

## 9. Width matrix completed (rev191) and the reuse mechanism

Complete d3d12be matrix (raw gpu_ms; all ok=1; source
`widthmatrix-rev190/d3d12be-rev191.csv`):

| format | shape      | 1 col | 2 col | 3 col | 5 col | marginal/col |
| ------ | ---------- | ----- | ----- | ----- | ----- | ------------ |
| q4_k   | 11008x2048 | 0.057 | 0.104 | 0.151 | 0.252 | ~0.047-0.049 |
| q4_k   | 2048x11008 | 0.062 | 0.110 | 0.157 | 0.258 | ~0.047-0.049 |
| q5_k   | 11008x2048 | 0.069 | 0.126 | 0.186 | 0.299 | ~0.057-0.059 |
| q5_k   | 2048x11008 | 0.082 | 0.142 | 0.200 | 0.317 | ~0.059-0.060 |
| q6_k   | 2048x11008 | 0.091 | 0.179 | 0.252 | 0.415 | ~0.080-0.088 |
| q6_k   | 65536x1024 | 0.247 | 0.478 | 0.722 | 1.215 | ~0.231-0.242 |
| q8_0   | 2560x5120  | 0.071 | 0.127 | 0.188 | 0.311 | ~0.056-0.060 |

Framing (as required): packed_gbs is unique weight bytes / time, so its fall
with width is definitional once time grows with columns - it is NOT, by
itself, proof of inefficiency. The measured mechanism facts are:

- raw time grows ~linearly with width, and the **marginal per column
  (~0.047 ms q4_k) is ~82-86% of the single-column time (~0.057 ms)**, i.e.
  each added column costs nearly a full weight pass in the variant measured
  by this instrument;
- the actual reuse mechanism in production is the **paired 2-column (q4_k)
  and 2/4-column (q6_k LM head) variants**, whose paired comparisons (same
  shapecost run, `d3d12sc-rev191.csv`) show raw-time reductions of **24-36%**
  at widths 2-5 in the real shapes (e.g. q4_k 9216x2560 w5 0.2682 -> 0.1726
  ms; LM head c4 w3 5.84 -> 3.79 ms; all pair-verified ok=1);
- production coverage (allowlists in code): the five real q4_k shapes at
  B=2/3/5 and the LM head q6_k, i.e. the decode's dominant weight traffic
  (census: ffn_up/gate ~13 MiB x832, attn_q ~11 MiB x208, ssm_out ~6 MiB,
  attn_output ~5 MiB, all q4_k-class) already runs the optimized variants.
  Remaining OLD-variant traffic: small attn_k/v (1-2 MiB) and any non-model
  formats; q5_k/q6_k non-LM tensors are negligible in this model.

## 10. Final audit result and next-intervention proposal (measured potential)

- No large measured kernel lever remains for this model: the shapes that
  dominate the weights are already on the optimized variants with measured
  24-36% raw-time wins; the OLD-path per-column cost is only relevant to
  shapes that are small here.
- Top remaining measured candidate: the **classic (CSTEP) path scheduling**,
  1636.3 ms = 17.5% of the decode wall with only 27% GPU and 25% sub+fence
  (415 ms). A minimal intervention on its per-step submission would target
  at most ~4.4% of the wall; the island A/B tempers expectations and any
  change needs its own same-build paired test. Proposed for explicit
  authorization as the next experiment; not started.
- Kernel-family gap (no measured need here): q5_k/q6_k non-LM column-sharing
  variants; a broad GEMM project is explicitly NOT proposed or started.
- Strict gates unchanged; no promotion.
