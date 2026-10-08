# qwen4b-maxperf evidence (plan 005)

Ranked offline candidate analysis for the next optimization wave (measured vs
static, mechanism, risk, A/B validation): `plans/006-qwen4b-maxperf-candidates.md`.

## Host-tag annotation (2026-10-07)

Baseline CSVs under `baseline-rev126/` carry the historical hardcoded host
prefix `xbox-series-s`. That label was wrong: the console was verified via
Device Portal `/api/os/info` as **Platform: Xbox Series X** (SystemOS
26100.9438.amd64fre.xb_flt_2608ge.260902-1030). Raw CSVs are preserved
byte-for-byte; read their `host` column family prefix as historical. Since
rev127 the bench path reads the operator-verified label from LocalState
`bench_host_tag.txt` (fallback `xbox-series`), so future runs are tagged
`xbox-series-x` on this console.

## Baseline (rev126, MSIX sha256 5bde8869ab587350fa42fb4a81ef1d47744697db672abb0d9bb6ed521682105f)

Package built from `feat/qwen4b-maxperf` == main a773e58 build inputs (all 170
changed source files SHA-equal on builder). Profile: gpu 34/34, ctx 2048,
batch/ubatch 64, F16 KV, threads 2, MTP depth 2 / pmin 50, GDN D3D12, Q8 GPU,
q6tile 0, FA2, twocol auto, scope2; same `qwen35-4b-mtp` GGUF (sha256
`3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630`); MTP
toggle only.

Medians of paired profile-OFF rows (2 rows/cell/block; two blocks for 64):

| workload           | seq tok/s | MTP tok/s | MTP gain | seq TTFT | MTP TTFT |
| ------------------ | --------- | --------- | -------- | -------- | -------- |
| chat 64 (119 tok)  | 23.87     | 28.1      | +17.8%   | 1251 ms  | 1435 ms  |
| code 64 (324 tok)  | 23.07     | 33.8      | +46.5%   | 3387 ms  | 3763 ms  |
| chat 256           | 23.65     | 27.5      | +16.3%   | 1267 ms  | 1443 ms  |
| code 256           | 22.65     | 32.56     | +43.8%   | 3382 ms  | 3790 ms  |
| std-512 64 (298 t) | 23.17     | 27.30     | +17.8%   | 3152 ms  | 3483 ms  |

Prefill: seq 94.6-95.8 tok/s, MTP 86.5-88.9 tok/s (MTP prefill ≈ −7-8%).
Full-ID parity: all dumps byte-identical MTP vs seq, both blocks, and equal to
the rev124 acceptance digests (chat64 `84e08196…`, code64 `ff42c5be…`).
Profile ON/OFF overhead within noise (<1% decode).

## Experiment 1 — Q6_K LM-head tile 0 vs 2 (rev127, `q6tile-rev127/`)

Interleaved paired blocks (2 rows/arm/block, 2 blocks, opposite order), decode
tok/s medians, profile OFF, full-ID parity OK on all 16 dumps:

| workload | tile 0 (rows)            | tile 2 (rows)        | Δ median |
| -------- | ------------------------ | -------------------- | -------- |
| chat 64  | **28.35** [28.20, 28.52] | 27.87 [27.69, 28.08] | −1.69%   |
| code 64  | **34.05** [33.81, 34.35] | 33.38 [33.31, 33.59] | −1.95%   |

Engagement verified per cell from the run log (`Q6_K LM-head columns=0/2`).
Verdict: tile 2 is a small consistent loss in product decode (8/8 rows), so
**tile 0 is retained** (device knob `d3d12q6tile.txt=0`). The synthetic
self-test component win (r119: q6_k_c2 1.12-1.44x on the matmul alone) does
not translate to real decode at these widths.

## Experiment 2 — prefill n_batch/n_ubatch sweep (rev127, `ubatch-rev127/`)

std-512 prompt (298 tokens), `--runs 2`, profile OFF; prefill ms/tok-s per cell:

| arm            | seq prefill            | MTP prefill                |
| -------------- | ---------------------- | -------------------------- |
| u64 (baseline) | **3108 ms / 95.9 t/s** | **3346 ms / 89.1 t/s**     |
| u128           | 3569 ms / 83.5 t/s     | 3730 ms / 77.5 t/s         |
| u256           | 4364 ms / 68.3 t/s     | 4566 ms / 65.3 t/s (run 1) |

(1) The u256 MTP cell's run 1 completed normally (CSV row 17:21:08Z). Its run 2
never produced a marker because the **console went offline mid-cell** (its
WDP endpoint and ICMP are unreachable as of 2026-10-07T17:35Z; the whole
network dropped, not just the app). The runner's old wait helper counted
sleeps only and had no per-request timeout, so it burned ~9 extra minutes
before giving up — the delay was the runner plus a dead console, **not** a
u256 workload slowdown. The helper is fixed (absolute deadline +
`--connect-timeout 5 --max-time 20`). All completed dumps are full-ID exact
vs the baseline digest `4454645…`. Verdict for the measured part: **64/64 is
kept**; u128/u256 are clearly slower prefill (−13% / −29%); the u256 MTP
cell is incomplete (1 row) and is not used for any claim.

## Experiment 4 — batch/decode thread split, C1 (rev132, `c1-threads-rev132/`)

Decode threads fixed at 2; batch threads 2/4/6, chat64 and std-512, timestamps
OFF, parity OK (all 30 dumps exact). Medians of 4 rows (2 per opposite-order
block):

| workload        | tb=2                     | tb=4                 | tb=6                   |
| --------------- | ------------------------ | -------------------- | ---------------------- |
| seq decode      | **24.50** [24.11, 24.55] | 24.46 [23.81, 24.59] | 24.28 [24.16, 24.67]   |
| MTP decode      | **28.90** [28.84, 29.05] | 26.16 [25.92, 26.28] | 22.65 [22.50, 23.17]   |
| std-512 prefill | 97.0 t/s (94.6-99.3)     | 98.2                 | 101.5 t/s (99.5-103.5) |

The +4.6% prefill at tb=6 is inside the two-sample spread and cannot pay for
the MTP verify regression (−9.5% at tb=4, −21.6% at tb=6). Note: `MtpDrafter`
sets its own `n_threads_batch` to its capped draft count (`mtp_draft.cpp:110`),
so target batch threads affect prefill/verify, not draft catch-up. Decision:
**retain decode=2, batch=2 (default)**; the `n_threads_batch` plumbing stays as
a bounded optional knob, not promoted.

Effective counts verified from the context, not just knobs: with the batch
knob absent the engine reports `decode=2 batch=2`; with `bench_batch_threads=4`
it reports `decode=2 batch=4` (per-run segments, `c2c7-rev133/`).

## Experiment 5 — draft CPU threads 1 vs 2, C2 (rev133, `c2c7-rev133/`)

chat64 MTP, t2/t2, timestamps OFF, parity OK; 4 rows per arm (opposite-order
blocks). draft=1 median **28.655** [27.78, 28.94]; draft=2 median 28.855
[28.76, 28.99] → +0.70%, overlapping ranges (one draft-1 outlier at 27.78).
Drafted 35 / accepted 30 per run on both arms. Decision: **no robust
improvement; retain `mtp_threads=1`**.

## Experiment 6 — joint prefill batch 32/32 vs 64/64, C7 (rev133, `c2c7-rev133/`)

std-512 seq, **both `n_batch` and `n_ubatch` varied together** (joint batch
configuration, not an isolated ubatch effect), timestamps OFF, parity OK:
64/64 prefill median **96.07 t/s** (TTFT 3104.2 ms) vs 32/32 89.71 t/s
(TTFT 3323.9 ms), all four 64 rows above all four 32 rows; decode unchanged.
Decision: **retain 64/64**.

## Experiment 8 — per-op/scheduler-split attribution for verify (rev134, `attrib-rev134/`)

New profile-only counters (default OFF): `VROUND ... calls= mm=`, `DSPLIT
calls= mm=`, `DSTEP ... calls= mm=` — D3D12 graph_compute calls and matmul
dispatches on the same boundaries as d3w/d3g/cpu. Parity OK on all four
measured cells; instrumentation overhead is bounded by the same-package
profile ON/OFF pair and a cross-package ON/ON comparison (caveats below).

| cell        | verify round | wall | d3w  | d3g  | calls | mm  | sub+fence | sub+fence/call |
| ----------- | ------------ | ---- | ---- | ---- | ----- | --- | --------- | -------------- |
| code64 MTP  | width 3 (21) | 70.5 | 36.6 | 25.9 | 185   | 201 | 10.7 ms   | 0.058 ms       |
| chat256 MTP | width 2 (35) | 54.9 | 27.3 | 17.3 | 185   | 201 | 10.0 ms   | 0.054 ms       |
| chat256 MTP | width 3 (58) | 70.4 | 36.7 | 25.8 | 185   | 201 | 10.9 ms   | 0.059 ms       |

Whole-decode DSPLIT: code64 4496 calls / 4976 mm, d3w 939 ms (49.5%), d3g
667 ms (35.1%); chat256 27374 calls / 30258 mm, d3w 4820 ms (50.5%), d3g
3193 ms (33.4%). Submission+fence is 14.3% (code64) / 17.0% (chat256) of
decode; the rest of d3w is GPU time. The verify graph is **~185 alternating
CPU/D3D12 splits per round, ~1.09 matmuls per backend call** — the scheduler
emits one D3D12 call per matmul because CPU ops (norms/rope/attention) sit
between them.

Overhead: same-package ON/OFF decode −1.49% (code64) / −2.65% (chat256),
consistent with the known timestamp-ON cost; cross-package ON/ON (rev133 vs
rev134, different time blocks, not clean) −0.59% / −1.90%. Counter overhead
alone is therefore bounded at ≤~2% and is not separable from timestamps with
the current data.

Next candidate sized by this data: **C4 (merge boundaries by running the
cheap CPU ops on D3D12)**. Two norms per layer would remove ≈68 of the 185
calls per round (≈3.9 ms/round ≈ 4-5% decode on code64); adding rope ≈100
fewer (≈8% decode). Requires new exact-numerics kernels (HLSL + DXIL + host
parity), so it needs the written design in plan 006 before any build.

## Experiment 7 — timestamp default binding extended to code and 256 tokens (rev133, `c8ext-rev133/`)

Profile OFF; arms are the new **profile-bound default** (knob absent → timestamps
OFF) vs explicit ON; opposite-order blocks; parity OK on all 18 measured dumps;
engagement verified per cell (`off (profile-bound default)` / `on (explicit
knob)`; profile-ON cells auto-report `on (profile-bound default)`).

| workload     | default OFF (tok/s)      | explicit ON (tok/s)  | OFF vs ON |
| ------------ | ------------------------ | -------------------- | --------- |
| code 64      | **23.79** [23.68, 23.89] | 23.04 [22.99, 23.09] | +3.23%    |
| code 64 MTP  | **34.44** [34.33, 34.55] | 33.89 [33.76, 34.03] | +1.61%    |
| chat 256     | **24.29** [24.19, 24.38] | 23.69 [23.59, 23.78] | +2.53%    |
| chat 256 MTP | **27.88** [27.88, 27.89] | 27.37 [27.35, 27.38] | +1.90%    |

All eight OFF rows sit above their ON counterparts; TTFT unchanged within
noise. The C8 win therefore extends beyond chat64 (code and 256-token
workloads, +1.6% to +3.2%), and the profile-bound default is validated on
device.

Same-boundary attribution from the two profile-ON cells (last segment, run 2):

| cell        | decode wall | verify share | d3w   | d3g   | wall−d3w |
| ----------- | ----------- | ------------ | ----- | ----- | -------- |
| code64 MTP  | 1886.9 ms   | 78.8%        | 49.7% | 35.4% | 50.3%    |
| chat256 MTP | 9370.6 ms   | 63.0%        | 51.1% | 34.2% | 48.9%    |

Verify-round medians: width 3 ≈ 70 ms/round (d3w 36.5, d3g 25.9), width 2 ≈
52.8 ms (d3w 27.2, d3g 17.3). The next bottleneck is the verify round's split:
≈half D3D12 submission+fence+GPU (submission/fence ≈ 15-17% of decode) and
≈half CPU-side work, on top of verify's 63-79% share of decode.

## Experiment 3 — GPU timestamp queries ON vs OFF (rev131, `c8-timestamps-rev131/`)

Design: profile OFF on every cell, chat64, parity required, ON→OFF→ON
sequences per workload plus a reverse-order block. All 20 dumps full-ID exact
(`84e08196…`).

| workload | ts ON rows (tok/s)                       | ts OFF rows (tok/s)        | OFF vs ON  |
| -------- | ---------------------------------------- | -------------------------- | ---------- |
| seq      | 23.78, 23.79, 23.77, 23.81, 23.99, 24.01 | 24.10, 24.37, 24.73, 24.90 | **+3.15%** |
| MTP      | 28.43, 28.11, 27.72, 28.45, 28.37, 28.19 | 29.00, 29.03, 28.38, 29.03 | **+2.60%** |

Medians: seq 23.80 → 24.55; MTP 28.28 → 29.02. All four OFF rows sit above
their ON median (one MTP ON row, 28.45, overlaps the lowest OFF row).
TTFT unchanged within noise. Engagement verified per cell (`gpu
timestamps=on/off` lines); ON summaries report `ts_valid=7423` (full run) with
`ts_unavailable=0`; OFF summaries report `timing unavailable: N calls` — never
a measured zero. Counter semantics unit-tested in `tests/test_ggml_d3d12.cpp`
(stale-sample regression). Verdict: repeatable gain; the knob stays, and the
default-ON baseline is preserved until the profile-bound default is reviewed.

## Attribution correction (whole-run vs decode-scope)

The whole-run D3D12 backend aggregate (`graph_compute calls/matmuls/wall/GPU`
printed at context free) spans **all phases and contexts in the process** —
prefill, decode, MTP draft context, catch-up — and the load/setup window.
It must not be divided by a decode-only duration: such a ratio asserts a
decode share the aggregate cannot support. Only bracket deltas are
attributable. Currently the only same-boundary instrumentation is the verify
bracket (`VROUND` in `decode_loop.h`, nesting d3g ⊆ d3w ⊆ verify wall). A
same-boundary whole-decode split (DSPLIT) and per-classic-step split (CSTEP)
were added in the instrumentation revision; rev129 was superseded by the
rev130 measurement-integrity fix (mandatory profile argument, prefill policy
ordering, unavailable-sample semantics) before any device run. The corrected
rev130 build was deployed and measured by the independent reviewer
(`rev130-reviewed-dsplit/`, 2026-10-07T18:57-19:08Z, exit 0, parity OK):
chat64 profile OFF seq 23.81 / MTP 28.22 tok/s; same-boundary DSPLIT seq d3w
≈54% / d3g 29%, MTP d3w ≈53% / d3g 34-35%; profile-ON overhead
indistinguishable from noise in that opposite-order sample. Profile-ON cells
are for attribution only; timing comparisons are profile OFF, and the ON
overhead is measured before any attribution is trusted.

## Final acceptance — rev160 (LTCG), sequential-only FFN candidate

Package `GianlucaMazza.xllama_1.6.0.160_x64__pj67f1fcj4n14`, MSIX sha256
`74C9DEDDA15493A6673353F08E29A77EE16D21A124783BA023DC8956A03EF1C7` (builder ==
fetched). Full evidence and tables: `acceptance-rev160/verdict.md`.

- Full-token parity OK for every cell (chat64/code64/chat256/code256/std512,
  seq and MTP, knob OFF and ON, runs 1-2) against the baseline digests.
- Seq timing (median of 2, knob OFF→ON): chat64 +9.81%, code64 +5.62%,
  chat256 +6.84%, code256 +5.19%, std512 +6.01%. Gate-run labels as measured:
  rev154 chat64-seq **+6.53%**, code64-seq **+5.65%**.
- MTP arm with the knob ON binds the process to FFN off: deltas −0.55%/+0.12%/
  +0.61% (no regression; no dispatch). No FFN gain is claimed on MTP; the
  rev154 MTP +4.42%/+3.61% rows were FFN OFF/ON trials with C8 held fixed and
  are rejected as an FFN candidate (chat256 round-49 divergence), not C8
  evidence. C8 was measured separately (rev131: +1.6 to +3.2% per arm).
- Termgate 6 arms all `ok=1`, sw0/sw1 byte-identical; session gate 4/4;
  API chat PASS (`MODEL=qwen35-4b-mtp`).
- Knob stays **default OFF**; the seq profile is bound once per process and a
  conflicting MTP context is rejected before creation (`restart required`).
  The MTP FFN candidate is closed (target-only diverges at chat256 round 49).
