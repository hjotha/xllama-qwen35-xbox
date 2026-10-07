# Qwen4B-MTP functional MVP — proven configuration & exact usage

Scope: benchmark-proven config + normal-session configuration status.
No GUI-ready claim: the normal GUI was not launched/verified in this
cycle (see "Normal-session status" below).

## 1. Package & model identity (validated read-only)

- Installed package: `GianlucaMazza.xllama_1.6.0.111_x64__pj67f1fcj4n14`
- Package sha256: `6b3e96d80d687df68b2bd0fe734968cbee6c9bc8b297682370615e63e52747f1`
  - builder artifact (durable on .193): `C:\Users\hjotha\build\xllama-q5k\xllama-q5k-fixed.msix`
  - local fetch (ephemeral): `/tmp/xllama-rev110/xllama-q5k-fixed-rev111.msix`
  - build-local.log `sha256-fixed` matches; runner `verify_expected_package` passed
- Model: `Qwen3.5-4B-MTP-Q4_K_M.gguf` (LocalState models/qwen35-4b-mtp),
  sha256 `3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630`, 2,834,975,040 B,
  runtime gguf fnv `f3559ed1a2922585`
- rev110 build `4D1E3419…` is INVALID/not-delivered (silent transfer failure); only rev111 is proven.

## 2. Proven benchmark configuration (all acceptance evidence at this config)

- Placement 34/34 layers on D3D12 (`offloaded 34/34 layers to GPU`), gpu_mem 2614-2678/4147 MB, peak_ws ≤ 3776 MB
- n_batch=64, n_ubatch=64 (effective line `prefill batch override: n_batch=64 n_ubatch=64`), ctx=2048 (engine default), MTP mtp=4, pmin=75, greedy seed1
- **Required scope2 knob (two parts — the run command has part 1; part 2 must be uploaded, the runner does not manage it):**

```
set -a; source ~/.config/xllama/xbox-env; set +a
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
printf '2' > /tmp/scope.txt && ./scripts/deploy.sh upload-file /tmp/scope.txt "$PFN" "" "cpurepackforcegemv.txt"
# part 1 is in every command below: --twocol auto
```

## 3. Reproduce the acceptance pair (runner on .157)

```
export XLLAMA_MSIX_SHA256=6b3e96d80d687df68b2bd0fe734968cbee6c9bc8b297682370615e63e52747f1
./scripts/bench-xbox-ort.sh qwen35-4b-mtp --prompt bench/prompts/spec-chat-open.txt \
  --n-predict 64 --runs 3 --greedy --seed 1 --tokens --mtp 0 \
  --gpu-layers 34 --batch 64 --ubatch 64 --twocol auto --profile-phases 0 \
  --out bench/results/<tag>-ref.csv
# then the identical command with --mtp 4 for the MTP arm
```

Post-run verification (all three must hold): get-log contains
`offloaded 34/34 layers to GPU` AND `prefill batch override: n_batch=64 n_ubatch=64`;
CSV host column contains `-u64-b64-` and `-g34`.

## 4. Termination-integrity gate (durable, no /tmp dependency)

Durable driver: `bench/results/mvp-20261007-evidence/termgate-run.sh`
(scratch dirs inside it are runtime-only; the script itself no longer
lives in /tmp). Run: `./bench/results/mvp-20261007-evidence/termgate-run.sh <package-sha>`
with the three device files pre-uploaded: `bench_gpu_layers.txt=34`,
`bench_n_batch.txt=64`, `bench_ubatch.txt=64`, plus the scope2 upload
from §2. Result archive: `termgate-result.csv` + `termgate-arms/`
(72 dumps, seq/ref/cand).

- Result: **8/8 ok=1** — cancel@6, cancel@13, stop@probe, stop@run
  (stop_branch=spec round=4), eog@0 cap, eog@1 cap, eog@2 cap,
  **eog@3 natural eos tok=248044 spec-reject** (exactly 1 natural EOG;
  the other three are capped completions — do not describe all four as
  natural eos). Cross-arm dumps 24/24 byte-equal incl. cancel*-resume
  (subsequent-prompt integrity).

## 5. Acceptance summary (benchmark path — PASS)

- Full IDs: MTP vs same-config sequential, run1/2/3 byte-exact (full sha256)
- Real drafting: drafted=27, spec_accept=25 every measured run (MTP_STATS rounds=15)
- Memory safe; profile-OFF speed reported separately:
  sequential 19.66 tok/s / TTFT 1347 ms; MTP 19.86 tok/s / TTFT 1739 ms
  (≈+1.0% decode, +~392 ms TTFT). np16 smoke TPS (g28 16.75, g34 19.24) = exploratory only.
- ≥10% speed work remains OPTIONAL (plan004), not an MVP blocker.

## 6. Normal-session status (configured via existing settings; NOT GUI-verified)

- `LocalState\llama.ini` was **backed up** (see evidence folder
  `llama.ini.backup-pre-mvp`) and set to the proven values:
  `n_gpu_layers=34 kv_q8=0 n_ctx=2048 n_batch=64 n_ubatch=64`
  (n_threads auto); unrelated keys/comments preserved. Precedence note:
  single-purpose files (`gguf_gpu_layers.txt`, `kv_q8.txt`) override the ini.
- Validated on the **API/session path** (existing settings only):
  `validate-api.sh chat` **PASS** (endpoint + chat + concurrency), with
  log evidence `offloaded 34/34`, `gguf gpu layers: 34 on D3D12`,
  session generate lines `mtp=0` (evidence: `mvp-20261007-evidence/api-session.log`).
  Effective `n_batch/n_ubatch` for the session path is applied by code
  (`apply_llama_ini_session`) but **there is no effective-print for
  session batches** (only the bench path prints it) — observability gap.
- **Exact minimal gaps for full normal-flow parity (report only, not built):**
  1. **MTP selection missing in normal flow**: `SessionParams.mtp`
     defaults false (`session.h:65`) and neither `MainPage.cpp` nor
     `api-server.cpp` sets it — normal/API sessions run `mtp=0`.
     Minimal change: a settings field/API param wired to `sp.mtp` +
     `mtp_pmin` at the session-construction site(s).
  2. **scope2 (twocol) missing in normal flow**: `apply_twocol_knob` is
     only called by bench/termgate/replay paths. Minimal change: call it
     (or read `d3d12twocol.txt`) at the normal session-construction site.
  3. **GUI ini application gap**: `MainPage.cpp` does not call
     `apply_llama_ini_session` (only `api-server.cpp` does) — GUI would
     not pick up ini `n_batch/n_ubatch/n_ctx`. Minimal change: one call
     beside the existing `gguf_gpu_layers_knob()` line.
- **Not claimed:** normal GUI readiness (GUI not launched/verified this
  cycle); MTP/scope2 via GUI; session effective batch observability.

## 6b. Normal-flow delivery package: rev114 (supersedes rev111 for the app)

- Package sha256: `9d522c88a7a27fdc8d9d00dfb9e3d5e49919063ca94a2ea777b6eb310b315ec5`
  (build-local.log `sha256-fixed` identical; installed
  `GianlucaMazza.xllama_1.6.0.114_x64__pj67f1fcj4n14`).
- rev112 = built but superseded (blocked parser design, never accepted);
  rev113 = compile-failed (no package). Only rev111 (benchmark pass) and
  **rev114 (benchmark + normal-flow pass)** are delivered artifacts.
- Normal flow (llama.ini keys above, scope2 files present at startup):
  startup line `twocol='auto' repack='2' (immutable...restart...)`;
  `session config: ... mtp=1 depth=4 pmin=0.75 n_ctx=2048 n_batch=64
n_ubatch=64 gpu=34 kv_q8=0`; API chat accepted>0
  (drafted=136/spec_accept=118 on a long reply; 4/1 and 1/1 on short
  ones); wrong model → HTTP 500 with the scope2-restriction reason;
  ini edits (mtp 4→0, n_batch 64→128) force hub recreate (new
  `session config:` line); profile-file hot change →
  `REJECTED, startup profile stays (restart required)`; client abort →
  next request HTTP 200; concurrent 200/200; termgate 8/8 ok=1.
- Bench same-config pair re-verified on rev114: run1/2/3 EXACT
  (`mvp114-20261007-*.csv/.tokens`), drafted=27/spec_accept=25.
- **Result classification:** (a) benchmark PASS = rev111/rev114
  `mvp-20261007` + `mvp114-20261007` cells; (b) the earlier
  `validate-api chat` PASS on rev111 was an **MTP-OFF API smoke —
  superseded**; (c) the rev114 `validate-api chat` + curl matrix is the
  **API MTP-ON acceptance** evidence.

## 7. Evidence inventory (durable, under bench/results/)

- `mvp-20261007-*.csv/.tokens` — acceptance cells
- `mvp-20261007-INSTRUCTIONS.md` — this file
- `mvp-20261007-evidence/` — device logs (ref/mtp/termgate/api),
  `termgate-result.csv`, `termgate-run.sh` (durable driver),
  `termgate-arms/` (72 dumps), `llama.ini.backup-pre-mvp`
- plan004 §completion records — full run history and gates
  No credentials/secrets are stored in these artifacts.
