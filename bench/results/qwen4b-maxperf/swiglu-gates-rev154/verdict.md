# rev154 SWIGLU product-trial gates (narrow FFN lineage) — PASSED

Package: GianlucaMazza.xllama_1.6.0.154_x64__pj67f1fcj4n14
MSIX sha256: 5d589313c9640b1149ca42c8981f756f927107b48f1907bc2efcbaf3d769771d
Run: ,

1. Full-ID parity: OK, 0 failures (sw0 vs sw1, chat64/code64 x seq/mtp, runs 2/3).
2. Per-round proposal/acceptance parity: VROUND feed= sequences and accepted= counts
   identical OFF vs ON for chat64 (23 rounds) and code64 (21 rounds).
3. Engagement (last-segment slicing): OFF knob=off dispatches=0; ON knob=on
   dispatches 1041-2240. Narrow FFN lineage whitelist enforced
   (ffn_swiglu/ffn_gate/ffn_up prefixes, ne0=9216, dims match).
4. Timing (profile OFF, 2 rows/arm): chat64-mtp +4.42%, code64-mtp +3.61%,
   chat64-seq +6.53%, code64-seq +5.65% (all rows separated).

Caveat: 2 measured rows/arm in this gate run; rev153 exploratory run
(without the lineage whitelist) showed +4.5-5.7% on the same four
workloads, so the effect is consistent across two packages/runs.
Knob remains default OFF; promotion requires the expanded repetition
grid + full acceptance (code/256/session/termination/API) + final build.
