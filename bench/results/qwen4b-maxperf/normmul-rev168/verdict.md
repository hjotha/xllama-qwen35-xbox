# Plan 007 RMS_NORM+MUL D3D12 chain — measured, REJECTED (net negative)

## Engagement (proven, rev168)

- Knob `bench_normmul.txt=1` -> log `main_loop: bench_normmul.txt='1'
-> RMS_NORM+MUL D3D12`; dispatch counters in the census run:
  **3515 RMS_NORM + 4060 MUL dispatches**.
- Census (sched2, per graph): RMS_NORM CPU 52.3 -> 25.5, D3D12 0 -> 26.8;
  MUL CPU 48.0 -> 16.9, D3D12 0 -> 31.1 (the remaining CPU nodes are the
  non-1D-broadcast `gate`/`attn_gated` MULs and GDN norms outside the strict
  predicate).

## Paired A/B, same build (rev168), interleaved knob 0/1, profile OFF

| cell       | prefill k0 -> k1      | decode k0 -> k1             |
| ---------- | --------------------- | --------------------------- |
| std512 MTP | 88.2 -> 87.2 (-1.03%) | 28.16 -> 27.16 (**-3.55%**) |
| chat64 MTP | 92.0 -> 89.3 (-2.96%) | 28.81 -> 27.61 (**-4.17%**) |
| chat64 seq | 94.5 -> 97.7 (+3.46%) | 24.59 -> 23.16 (**-5.83%**) |

- **Split count rose**: 124/g (62 CPU + 62 D3D12) -> 148/g (74 + 74). The
  partial movement inserted new CPU<->D3D12 boundaries instead of collapsing
  the islands; the working hypothesis of plan 007 ("the scheduler merges the
  CPU splits into the adjacent D3D12 ones") is **falsified by measurement**.
- Numerics: all token digests exact (chat64 `84e08196…`, std512 `44546453…`);
  the double-lane reduction did not flip a token. FP64 is real in the DXIL
  (`fadd double`/`fdiv double`, precise `Sqrt`), but the per-op cost was not
  separated from the dispatch/boundary overhead — the net is negative either
  way.
- Earlier rev166 cross-build numbers and the first rev167 A/B are **not**
  evidence: rev167's knob had no call site, so both arms ran the CPU chain
  (control); that A/B measured noise, not the chain.

## Verdict

Both plan-007 acceptance criteria fail (no split reduction; paired timing
negative, decode -3.5..-5.8%). The implementation is reverted and the
production source restored; the patch and shaders are preserved in this
directory. A future attempt would need the _whole_ chain covered (all MUL
forms plus the GDN norms) so the islands actually collapse, and the FP64
reduction cost isolated first — otherwise the measured outcome is expected to
repeat.
