# n_batch/n_ubatch 128/256 on the final 1.6.0.163 package (MTP arm, profile OFF)

Joint b/ub (as rev127), two prompts, interleaved 64 → 128 → 256 × 2 rounds,
`--runs 1` each; full-token digests `84e08196…` (chat64) and `44546453…`
(std-512) exact in all six cells — every cell answered, no OOM, no
allocation error in the logs (VRAM post-load 2692 MB / budget 4147 MB
unchanged).

| prompt       | b/ub  | prefill t/s (median of 2) | decode t/s | peak WS MB |
| ------------ | ----- | ------------------------- | ---------- | ---------- |
| chat64       | 64/64 | **90.3**                  | 28.66      | 3667       |
| chat64       | 128   | 79.6 (−12%)               | 28.62      | 3734       |
| chat64       | 256   | 78.8 (−13%)               | 28.59      | 3857       |
| standard-512 | 64/64 | **89.3**                  | 28.02      | 3670       |
| standard-512 | 128   | 78.6 (−12%)               | 28.16      | 3733       |
| standard-512 | 256   | 66.5 (−26%)               | 28.09      | 3862       |

**Verdict:** 128/256 responds (no OOM; peak WS +1.8% / +5.2%), but prefill
loses 12–26% while decode is flat within noise — 64/64 stays. Same direction
as the rev127 sweep, now complete on the final package.
