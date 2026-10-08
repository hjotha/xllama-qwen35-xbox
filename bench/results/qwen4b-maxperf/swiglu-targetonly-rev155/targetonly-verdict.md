# Target-only FFN SWIGLU discriminator (rev155, chat256, profile ON)

Package: GianlucaMazza.xllama_1.6.0.155_x64__pj67f1fcj4n14

Arms: off (0 dispatches), all-FFN (4743), target-only (4352; nextn
layer-32 FFN excluded via the verified suffix<n_layer discriminator).

Full IDs: all three arms f78ce837... EXACT.

Per-round VROUND feed sequences: BOTH all-FFN and target-only diverge from
OFF at round 49 (arm: width2 accepted1 feed=[1817,369]; off: width3.
accepted2 feed=[16421,2972,16759]; round counts 94 vs 93).

Conclusion: excluding the nextn/draft FFN from GPU does NOT remove the
per-round proposal divergence; the cause class is target-FFN arithmetic
drift feeding the drafter, not the draft context's own FFN.
Per the review rule: record and keep the MTP SWIGLU candidate OFF.
Sequential-only FFN optimization (no drafter) stays independently viable
and proceeds through its own API/cancel/termination gates.
