# Equal-history FFN/MTP divergence — root cause DEMONSTRATED (rev182)

Repro: chat256, MTP, greedy, seed 1, `bench_swiglu_mtp_diag.txt=1` (diagnostic
override) + `d3d12swiglu.txt=1` (4743 FFN SWIGLU dispatches) vs OFF (1).

## The first differing decision (exact capture)

The first differing event is the **draft proposal of token 369 at depth 0**
(pmin line 118; event 192 in the rev181 run). Both arms propose the same token
with the same argmax:

| field                | OFF                             | ON                          |
| -------------------- | ------------------------------- | --------------------------- |
| top1 (17g)           | 369:21.893901824951172          | 369:21.894189834594727      |
| top2 (17g)           | 8772:20.417074203491211         | 8772:20.416996002197266     |
| argmax margin m2     | 1.4768276214599609              | 1.4771938323974609          |
| **top_prob p (17g)** | **0** (early exit)              | **0.5000532865524292**      |
| p_min                | 0.5                             | 0.5                         |
| **p - p_min**        | -0.5 (early-exit verdict)       | **+5.3286552429199219e-05** |
| outcome              | proposal discarded (round ends) | proposal verified, accepted |

Target-logit change between arms: ~2.9e-04 absolute on a 21.9-magnitude logit
(~1.3e-05 relative), the network-amplified envelope of the FFN kernel.

## Why the earlier margin argument was wrong (corrected)

The ~0.0288/0.1449 margins cited before were the _target_ verify rows where
both arms made the SAME decision - not the divergence point. The actual
decision boundary is the **p_min (top_prob) gate**, and there the margin is
**5.3e-05** while the logit-induced change is ~3e-04 atomic-scale on the
candidate's probability: same order -> the change **crosses the decision
margin**, which is exactly what a trajectory explanation requires. The argmax
margins (1.42) are three orders of magnitude larger, so no argmax flip was
involved.

## Sequence

OFF discards 369 (p_min early exit) -> the round ends -> the next draft event
is a new depth-0 proposal (7309); ON passes 369 -> depth-1 continues (8744) ->
369 verified and accepted. From there the feed/round structure differs, the
run ends with one extra round, and the 256-token output ids remain identical
(`f78ce8372ccc7281`). KV accounting consistent in both arms (same kv_end
sequence up to the divergence, 23 rollback>0 events, same spans); RNG is not
consumed (greedy) - neither stale state nor RNG explains it.

## Status of the strict gate and open risk

Classification: **expected floating-point trajectory change crossing a
genuinely marginal p_min decision (5e-05)** - demonstrated, not assumed. The
strict per-round trace gate stays for any FFN/MTP promotion; an evidence-backed
relaxation proposal would have to name the p_min margin explicitly (e.g. accept
divergence iff the diverging decision is a p_min branch within a measured FP
envelope and committed prefixes/final ids/KV accounting match). Other defects
are not excluded by one identical 256-token output; further prompts and the
p_min-branch margin distribution would be the next characterisation.

## Closure with the real probabilities (rev184, aligned by (pos, drafted))

`top_prob_full` (full-vocab sum, no early exit) plus the proposal position make
the arms comparable without relying on raw record indices (branches differ in
depth). Aligning the 225 common proposals by `(pos, drafted)`:

- **Exactly one verdict flips in the whole run**: pos=249, drafted=369, depth 0
  - OFF: `p_full = 0.49988740682601929` (sum 2.0004504396745615 > limit 2),
    verdict 0, **delta = -1.1259e-04 below p_min**
  - ON: `p_full = 0.5000532865524292` (sum 1.9997868028002117), verdict 1,
    **delta = +5.3287e-05 above p_min**
  - the two real probabilities straddle the threshold by ~1e-4 and the
    inter-arm FP change is 1.66e-04 in probability space.
- The other 224 common proposals keep their verdicts while their `p_full`
  values drift up to ~8e-4 (the FFN FP envelope), i.e. only a proposal whose
  probability sits inside that envelope of the gate can flip.

This closes the numerical cause with the REAL pre-early-exit value: the `p=0`
of OFF is the early-exit marker; its actual probability was 0.4998874, below
the 0.5 gate, while ON's was 0.5000533. No gate was relaxed and no other
mechanism is needed to explain the round-49 divergence.
