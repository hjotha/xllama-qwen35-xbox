# Plan 009 — KV slot save/restore on disk (hybrid-cache reuse fix)

Authorized: Henrique, 2026-10-09 19:47 UTC, msg 841. New work; the rev209
FFN GPU+MTP delivery is DONE and published — preserve its baseline (rev209
installed + enabled, sources, evidence, rollback rev210). No new push/merge
without specific confirmation. No .57 touch. Xbox tests + .193 builds
allowed; monotonic versions for deploy/restore.

## 0. Cause (established, not assumed)

- In-RAM cross-turn reuse fails on qwen35-4b-mtp because the recurrent
  GatedDeltaNet state cannot tail-rewind: `llama_memory_recurrent::can_seq_rm`
  allows rollback of at most `n_rs_seq` tokens (ours = 2; log: `speculative
n_rs_seq=2`), and the hybrid wrapper requires BOTH caches to agree
  (`llama-memory-hybrid.cpp:188-210`). Our T2/T3 rewinds (15/47 tokens)
  always refuse → full clear + full prefill. This is model math, not a bug
  to patch around: the fix is snapshot save/restore (never rewind), per the
  beellama reference (read-only inspection, not copied blindly).
- The submodule pin HAS full hybrid serialization:
  `llama_state_seq_{save,load}_file` + `state_write/state_read` on both
  `llama-memory-recurrent.cpp` and `llama-memory-hybrid.cpp`. The drafter
  keeps a PRIVATE mirror context (`MtpDrafter::ctx()`), so draft state is
  never in the blob — it must be reset and rebuilt via catch-up (existing
  F3.5 `can_carry_from`/`align_to` machinery).
- Foundation already in tree: `KvStore` (file mgmt: valid_id/path_for/erase/
  prune, `LocalState\kv`, caps 3 files/192MB), `Session::save_state`/
  `load_state` (session.cpp:1257-1480: magic/version/text header, token
  array + native blob, 8MB chunked I/O for AppContainer, tmp+rename
  atomicity, size ceilings, fp string, clear+prefill fallback), GUI
  per-conversation snapshots (MainPage #170b), `#170a` common-prefix path.

## 1. Gaps to close (this plan)

1. Fingerprint too narrow: `v1 n_ctx kv_q8 lora params desc` misses rope
   scale/base, YaRN params, cache K/V types, n_seq_max. A re-quantized or
   re-roped build could load stale state. -> v2 fingerprint + version bump
   (old files refused).
2. `load_state` does not reset the MTP drafter mirror (stale pending row
   survives a restore in-process). -> explicit `align_to(0)` on load, with
   fallback handling; never silently keep a stale mirror.
3. No slot concept on the API path (GUI-only snapshots). -> named slots in
   `LocalState\slots\` via `KvStore{dir}` + additive API routes
   (save/restore/list/delete), serialized under the session lock, documented
   in `docs/api-endpoint.md`.
4. No reuse observability: restore timing not separated from prefill/decode;
   no kept/reprocessed counters. -> log `slot restore kept=N reprocessed=M`
   - restore_ms in the generate path; per-turn `turn full=` lines stay.
5. No validation battery for any of the above on device.

## 2. Stages and gates (each stage commits locally; no push/merge)

- S1 core (session.cpp/h, kv_store.h if needed, fingerprint v2, drafter
  reset, counters/timing logs). Gate: Linux build + ctest + shellcheck/
  clang-format on touched files + `check-win-syntax.sh` if a splat exists.
- S2 API (api-server.cpp routes + docs). Gate: rev211+ device build
  (monotonic; verify exe identity vs rev209 post-build — the tar-mtime
  incremental trap must not relabel a stale exe), install, API smoke:
  save/restore/list/delete round-trip + observed reuse line.
- S3 battery on the installed build, FFN GPU+MTP ON + OFF-equivalent
  control, fresh runs only: cross-request reuse; fresh-process reuse
  (restart); reused/reprocessed counters; logits + multi-token continuation
  vs fresh prefill (byte/token equality, no epsilon); prefix divergence;
  rollback to older slot; cancel/stop/EOS; slot concurrency/isolation;
  corrupt + incompatible file; fingerprint refusal; atomic write (tmp+
  rename, no partial); restore/prefill/decode timing split; memory/
  stability/API; no regression: numeric selftest + acceptance parity +
  termgate on the new build.
- S4 receipt + report (blockers and progress explicit). Publication needs a
  separate confirmation.

## 3. Standing rules

Model/formats/gates/extremes preserved; no epsilon, no threshold relaxation,
no reference swaps. UWP storage only (`LocalState`), no new permissions, no
credentials. Single resident session; slots never auto-load (explicit
restore only; fallback is always a normal prefill). `.57` untouched.

## Incident 2026-10-09 ~20:0x UTC: external *.log deletion in this worktree

An external actor deleted 111 tracked `*.log` files (plus all untracked raw
logs, ~2GB) under `bench/results/` in this worktree — pattern: every `.log`
regardless of size, including small committed evidence (segments,
driver.logs). CSVs/tokens/JSONs/TXTs/scripts, `.git`, `/tmp` artifacts and
the Linux build tree were untouched. Nothing of mine deletes logs; the
deletions appeared between two status checks. Committed `.log` content is
recoverable from git history; untracked raw logs are gone (their distilled
evidence — segments, canons, CSVs, receipts — was already committed).
Rule going forward: commit distilled evidence IMMEDIATELY after producing
it; do not rely on `.log` files persisting on disk; bulk restore is futile
until the deleter is identified (out of scope for this plan).

## Root cause of common=36 (2026-10-09, reuse-diag on rev212)

The `kv_reuse_diag.txt` seam dump on a save/restore/continue probe proved
the blocker is NOT BPE and NOT history text: it is TEMPLATE ASYMMETRY.
Resident after the seed turn: `[..., im_start, assistant, '\n', <think>,
'\n\n'(271), </think>, ..., 'ack']` — the generation prompt carries the
Qwen3 no-think `gen_suffix`. History rendering omitted it
(`chat_prompt.cpp`: "History assistant headers never carry gen_suffix"),
so continuation prompts have `[..., im_start, assistant, '\n', 'ack', ...]`
and the common prefix breaks at the suffix (36/37), forcing full prefill
on top of the hybrid rewind refusal.
Fix (rev214): replay `gen_suffix` in history assistant turns
(`header + suffix + content + close`), byte-exactly as generated. No-op
for empty-suffix formats; single-turn prompts render no history, so the
rev209 acceptance/termgate/VROUND gates are unaffected by construction
(still re-run as no-regression proof). Host chat-prompt tests 14/14 with
the updated qwen expectation. Turns with real (non-empty) thinking content
absent from history still diverge there and full-prefill (safe fallback).

## Build flow (msg907, 2026-10-09 21:47 UTC — acknowledged)

- `run-xbox-build.ps1 -BuildRevision N` without `-Final` forwards
  `-Iteration` to `build-uwp.ps1` (verified in `build-xbox-local.ps1`:
  `mode="iteration"` logged, same `WorkRoot`, MSBuild incremental cache
  shared). With `-Final` the full LTCG path runs.
- Rules from here on: functional rounds use iteration builds (fast link,
  same revision monotonicity, no `-Final`); never `-Clean` without need;
  every evidence records MODE (iteration/final) + msix sha256 + exe size;
  iteration TPS is NEVER reported as a final result. `-Final`/LTCG is
  reserved for closing: performance, stability and equivalence gates on
  the exact shipped binary (plus exe-identity check vs the prior final).
- Build 214 (Final, template fix) in flight — not interrupted. Model,
  formats, gates and the .57 task unchanged.

## Status: implemented + validated (2026-10-10 ~00:35 UTC, no push/merge)

Shipped in branch (rev215 iteration for functional, rev216 Final for
closing): fingerprint v2, drafter reset on load, slots API
(save/restore/list/delete + docs), template suffix replay, preload
no-evict, reuse-diag knob. Evidence: slots-rev211/ (ON+OFF batteries,
harvest probe), acceptance-rev216/ (closing gates on the final binary).
Headline results: 7 kept lines per arm; MTP-off harvest (40 kept, 20-tail
prefill, 209.9ms); ON target-match established but F3.5 clears (mirror
cannot carry — carry-at-boundary redesign flagged out-of-scope); closing
gates green (numeric 50/50, 96 lines 0 failures, 24/24 median +4.57%,
termgate 6/6 arm-excluded, VROUND canons = 207/209, api recall + 193/120,
slot smoke). Device: rev216 Final serving, FFN+MTP enabled.

## Carry prime design + status (2026-10-10 ~01:30 UTC)

Finding: with the template fix, target-level prefix match is established
(`kept 41 of 61`), but F3.5 still clears — `can_carry_from` is structurally
false at turn boundaries (pending rows are a within-turn mechanism; the
mirror rebuild needs target hidden rows = target prefill). Same work as
before, MTP stays active. Harvesting ON-savings needs the carry
re-established, not assumed.
Implementation (branch, rev218 iteration): inside the F3.5 would-clear
path, when pure extension (kv_keep == resident size) over a non-empty
mirror: re-decode the last resident token at its own position (idempotent
overwrite; yields its hidden row), then one depth-1 draft round sets the
pending row; proposals never committed. Guarded (divergent tails never
prime); any failure falls through to the existing clear (fail-closed).
Adds MtpDrafter::mirror_empty/n_embd. Clear line now reports prime
skip/fail reasons. First device datum: `prime failed: recompute decode` —
llama_decode rejects non-append batches (`X < Y` sparse-position rule),
so overwrite-recompute is impossible; the design already avoids it by
construction (prime runs only pre-clear, never overwriting committed
state... under investigation: the recompute DID fail, see below).
Wait — correction from the device error: the recompute decode of token[S-1]
at S-1 with cache end X=S fails the append-only check. The prime as coded
cannot work that way. Next: capture h(S-1) WITHOUT re-decoding (see below).

## Blocker 2026-10-10 ~01:10 UTC: device app wedged post rev218 install

After rev218 install, the app loads the model ("Session loaded") but the
API never serves (listener started, connections hang; 3 restarts fail;
no dumps). Not the package delta (F3.5-path-only logging change cannot run
at boot) and disk is writable. Suspected thermal/contention after ~12h
sustained load (prior post-vround wedge recovered by restart; this one
persists). No reboot/resetGPU per constraints. App stopped at 01:34:11 UTC
to idle/cool; leftovers cleaned (papa.kv, writetest). Next: retry start;
if still wedged, report device as blocker with reboot as the explicit
out-of-scope ask. Prime device validation (rev218) pending on serving.

## Status 2026-10-10 ~01:45 UTC: device wedged, work paused for serving

- After rev218 served the ROMEO probe, the API went dead (~01:07) and every
  restart since launches, listens, loads the model ("Session loaded") and
  freezes: no crash dumps, portal healthy, GET hangs (not refused). Same
  signature as the post-vround wedge, but persistent across 4+ restarts
  (that one recovered after one). No package change at the boundary (rev218
  served before dying) and the rev218 delta cannot run at boot (F3.5-path
  logging only) — device/environment degradation (thermal after ~13h load
  and/or background GPU contention; another actor's METRIC polling was seen
  on .193). No reboot/resetGPU per constraints. App left STOPPED (coolest
  idle) at 01:45; will retry start after a soak; if still wedged, device
  reboot is the explicit out-of-scope ask.
- Validation state of carry prime: Linux-compiles, format-clean, committed;
  device proof PENDING on serving (rev218 installed). Nothing else runs on
  the device until serving recovers (no blind retries billed as gates).

## Carry mechanism: fully mapped (2026-10-10 ~02:00 UTC) — SCOPE REQUEST

Ground truth (code + device probes):

- `process()` DOES stash pending on success (last replayed row). But turns
  end WITHOUT a trailing catch-up covering the final committed token
  (loop exits after sampling/commit; draft rounds in between leave
  future-pos pending), so pending is stale at every turn boundary and
  F3.5 clears every turn. Observed without exception (kept established,
  pf==full, on all builds).
- Recompute-overwrite is impossible (`X < Y` append-only rule, device
  error captured). Mirror rebuild needs target hidden rows = target
  prefill (chicken-egg). Plain-embedding mirror prefill would desync
  silently (catch-up feeds target states by design). All shortcuts dead.
- The only sound lever: establish the carry WHILE the rows are live,
  i.e. a trailing single-row catch-up of the final committed token at
  turn end (uses live nextn rows; sets pending=h(S-1)@S-1 via the existing
  process() stash). Must be placed exactly where the loop knows the final
  token came from a target decode (a blind flush could stash a stale row
  -> false-positive carry -> silent draft degradation caught only by
  verify; must not do that).

EXACT scope-expansion request (not implemented; needs confirmation):

1. decode_loop.h: at turn end, iff the final committed token's hidden row
   is live in the nextn buffer, run one single-row catch-up replay for it
   (reuses mtp_catchup_batch/process; fail-closed: any failure leaves
   pending stale -> status-quo clear path). Hot path touched once per
   turn (not per token); no output change possible (no sampling/commit).
2. session.cpp F3.5 prime path (already written, committed, device-pending)
   then harvests in-RAM continuation: kept + tail-only prefill + drafting.
3. Follow-up (separate): persist the mirror blob in slot files so
   post-restart mirrors are non-empty (prime guard) — same harvest across
   restarts. Slot format v3, old files refused.
   Validation for (1): in-RAM 2-turn (kept + pf==tail + drafted>0 + exact
   reply), then full slot battery + regression (numeric/acceptance/termgate)
   on a Final build. No push/merge. No model/format/gate changes.

## Slot format v3 PROPOSAL (pending — DO NOT implement without confirmation)

Needed only for post-restart reuse (in-RAM reuse needs no format change).
Exact layout delta over v2 (session.cpp save_state/load_state):

- Header line gains one field: `xllama-kv\t<n_state>\t<n_tokens>\t<fp>\t<mirror_n_state>\n`
  with fp starting `v3` (kStateVersion 3). Body appends the draft-mirror
  native seq blob (`llama_state_seq_get_data` on the draft ctx, seq 0)
  after the target blob iff mirror non-empty at save; `mirror_n_state==0`
  means "no mirror persisted" (MTP was off or drafter absent — valid).
- Read compatibility: magic + exact fp compare refuse v1/v2 files on v3
  code AND v3 files on v2 code (both directions fail-closed to normal
  prefill; snapshots are caches, never truth). No migration: stale files
  are re-created by use; dropping one costs a re-prefill only.
- Validation mirrors existing rules: both sizes ≤ ceilings, tokens ≤
  n_ctx, exact byte counts (truncation refused); target set_data then
  mirror set_data; any failure clears + prefills (existing fail-closed).
  Pending is NOT persisted (within-turn by design; stale pending is the
  bug being fixed — re-established at runtime by prime/carry).
- Impact: ~2x file size (~100MB for 40-80 tok); ms-scale extra RW;
  same tmp+rename atomicity; same API/JSON; GUI kv/ snapshots ride the
  same code path automatically. MTP-off saves (mirror 0) restore into
  MTP-on sessions as today (clear + rebuild); MTP-on saves restore the
  mirror for prime/carry. No model/config/permission changes.

## Device blocked 2026-10-10 ~02:15 UTC (reboot explicitly out of scope)

- Every launch since ~01:07: listener starts, lfm preload loads
  ("Session loaded"), then total freeze: API hangs (not 503/refused), log
  frozen, no dumps, portal healthy. Identical across 5+ stop/start cycles
  (including after 40-min soak with app stopped). rev218 served normally
  before dying mid-serve with no package change, and its delta cannot run
  at boot — this is device/environment degradation (prime suspects:
  thermal after ~13h sustained load, or sick D3D12 state from hard kills;
  background GPU contention unexcluded), NOT a code regression.
- Ruled out: disk full (writes OK), missing flags (cleaned), package
  identity (rev218 served ROMEO probe), credentials/network path
  (second-LAN-host test also hangs: device-side wedge, confirmed).
- Forbidden remedies (not attempted): reboot, GPU reset. App left STOPPED
  (coolest idle) for a future user-driven reboot + serve check.
- Validation debt queued on serving recovery (all code ready, committed):
  turn-end drafter-state datum (rev218 installed), carry prime harvest
  (rev217 proved path, rev218+ has reasons), full slot battery on the
  prime build, then Final + closing gates. No blind work billed as gates.

## Status 2026-10-10 (rev232 Final instalado, EDCC7C4E…)

- Termgate: comparação restaurada de TODOS os campos exceto `arm`
  (termgate-rev209.sh); rev228/229/230 CSVs = failures=0 vs legado rev192,
  contadores drafted/rounds idênticos (eog@0: 51/18). Colapso 51→4 de
  rev227 causado pelo replay corretivo (removido em dc394fb).
- Bateria v3 (slot-battery.sh): restarts verificados (morte 3× GET fail +
  renascimento), integridade exata 15 turnos (full,kv_before,pf,prompt) +
  IDs de resposta + marcadores início/fim; ON e OFF = 0 failures.
  fresh-em-processo-novo (#3/#14) distinto de continuação-após-seed
  (#2/#4, incremental kv=42 pf=20 provado).
- A/B controlado fontes216 (2e20b3e) vs correção, Final, mesma config,
  quiescência (bench/results/qwen4b-maxperf/
  ab-216-vs-corrected/ab-report.txt): termgate idêntico; TPS MTP-on
  corrigido MEDIDO ~0.8–1.3% abaixo do build 216src nas duas ordens de
  pacote desta rodada. Sem atribuição causal (ordem de células fixa
  dentro de cada pacote A/B; efeito de ordem não separado; térmico não
  excluído — sem telemetria; B' cobre só o lado B).
- PENDENTE (não implementado, aguarda aprovação de necessidade +
  compatibilidade): restore de disco em processo novo usa full prefill
  (mirror não reanexa à memória de decode; bateria documenta kv_before=0
  nos turnos #5/#7/#8/#9/#13). Proposta v3 (blob do mirror persistido)
  em docs/slots-formato-v3 neste plano — formato atual NÃO alterado.

## Restore em processo novo — bloqueio exato (2026-10-10, código + log)

Comportamento: `/v1/slots/restore` carrega o blob com sucesso ("KV state
loaded — N tokens") mas o turno seguinte faz full prefill (kv_before=0).
Cadeia causal no código (rev232 = HEAD):

1. `load_state` restaura a sequência 0 do TARGET via
   `llama_state_seq_set_data` (session.cpp:1539) e instala
   `m_kv_tokens` = tokens do arquivo (1562). O arquivo v2 JÁ contém os
   token ids (1527–1533), não só o blob.
2. Em processo novo o residente está vazio ⇒ `same_prefix=false` ⇒ o
   espelho MTP é limpo via `align_to(0)` (1577–1583) — correto por
   design: o blob restaura só o target, não o espelho do drafter.
3. No turno seguinte (full prompt), o prefix-match por token encontra
   `kv_keep>0` (683–687), mas o gate F3.5 exige
   `can_carry_from(kv_keep)` no espelho (774–786); espelho vazio ⇒
   clear do TARGET e de `m_kv_tokens` (781–783) ⇒ kv_before=0 e full
   prefill. Log prova: "draft mirror cannot carry the reused prefix"
   ×3 na janela de cada braço da bateria + linhas "KV state loaded".
4. Contra-prova: mesmo processo com `same_prefix=true` (restore seos do
   salvamento imediato) mantém o espelho e reusa (turno #12,
   kv_before=64). O restore do target em si está íntegro; o descarte é
   do gate F3.5 por espelho vazio.

Turnos afetados (bateria): #5/#7/#8/#9/#13 (kv_before=0, pf=prompt).

### Opções compatíveis com o formato ATUAL (v2: header+tokens+blob)

Nenhuma implementada; aguardam escolha explícita.

- Opção B (comportamento, sem nova primitive): após um restore que
  limpa o espelho, manter o reuse do target no F3.5 e rodar aquele
  turno em modo clássico (drafting off), reconstruindo o espelho via
  catch-up/commit_carry existente no fim do turno. Ganho: 1º turno
  pós-restore com prefill só da cauda, a custo clássico; MTP volta no
  turno seguinte. Risco: toca o hot path F3.5 — exige os mesmos gates de
  dc394fb (termgate full-field + bateria ON/OFF) antes de adoção.
- Opção A (rebuild do espelho no load): reconstruir o espelho do
  drafter a partir dos token ids já presentes no arquivo v2, de modo que
  `can_carry_from(n)` seja válido após o load. RESTRIÇÃO já provada
  neste plano: `llama_decode` é append-only — não se re-decodifica por
  cima de linhas ocupadas do target; o rebuild precisa de hidden states
  por posição (decode em sequência scratch ou primitive draft-only).
  VIABILIDADE NÃO PROVADA — exige spike de design antes de prometer.
- Opção D (v3: persistir blob do espelho): fora do escopo atual; não
  implementar sem aprovação explícita de necessidade+compatibilidade.

Registro: formatov2 intacto; sem push/merge; baseline209/210/216 e
config 256/64 preservadas.

## Opção A — veredito de viabilidade (2026-10-10, só leitura de código; B rejeitada)

Opção B descartada por decisão (desligar drafting viola o requisito
MTP-on). Investigação da A (rebuild do espelho no load, formato v2,
MTP-on) — conclusão: **INVIÁVEL em v2; a única pendência real exige v3.
Nada implementado; aguarda autorização.**

Cadeia de prova (código, sem execução nova):

1. O espelho = KV própria do draft-context (`MtpDrafter::m_ctx`) +
   hidden row pendente (`m_pending_h`, `m_pending_pos`) + snapshot
   (`m_carry_*`) — `include/xllama/mtp_draft.h:188-208`. O único caminho
   de rebuild é `process(tokens, h_rows, h_pending, n, pos0)` (:154),
   que exige por posição: token ids (HÁ no v2) + **hidden rows do
   TARGET** (`h_rows`).
2. `h_rows` só existe como efeito colateral de um `llama_decode` vivo no
   target: o chamador as coleta via `llama_get_embeddings_nextn_ith`
   (`src/bridge/decode_loop.h:581-591`). Após `llama_state_seq_set_data`
   (restore, `session.cpp:1539`) nenhum decode rodou — não há hidden
   rows, e elas **não são recuperáveis** da KV restaurada (o blob de
   seq-state carrega chaves/valores por layer, não a hidden final por
   posição; o buffer de embeddings é indefinido pós-`set_data`).
3. Regenerar `h_rows` = prefill completo do target: na seq 0 é proibido
   (append-only por cima das linhas restauradas); em scratch sequence é
   o custo integral do prefill + segunda cópia da KV do prefixo no
   orçamento apertado do console. Em ambos os casos custo >= o caminho
   atual (full prefill, que JÁ reconstrói o espelho via catch-up —
   comentário em `session.cpp:1563-1569`). Sem ganho de latência;
   trabalho estritamente maior.
4. Persistir só a carry row (~10KB) é **incorreto**: a KV-história do
   draft continuaria vazia e o draft head atenderia a histórico errado —
   drafts lixo, rejeitados pela verificação do target (fail-safe, mas
   mais lento que clássico), violando o contrato documentado do
   chamador ("must have proven that prefix holds the same tokens",
   `mtp_draft.h:124-127`).
5. A única variante íntegra da A = persistir a KV do draft-context (o
   mesmo mecanismo `llama_state_seq_{get,set}_data` sobre `m_ctx`) +
   carry row + prova de igualdade de prefixo verificável no load (em
   processo novo não há como "provar" implicitamente a pré-condição do
   `can_carry_from`). Isso **É** a proposta v3 (blob do espelho) —
   mudança de formato.

Necessidade exata de v3 (para autorização futura, NÃO implementar
agora): por arquivo de slot, adicionalmente (a) blob KV do
draft-context no mesmo formato seq-state, com header de capacidade
próprio; (b) carry row + pos; (c) hash dos tokens do prefixo do draft
para a checagem de igualdade no load. Compatibilidade exigida: arquivos
v2 seguem carregáveis (restore só-target = comportamento atual);
loader v3 com version-gate, recusando/ignorando a seção do espelho em
mismatch no mesmo estilo fail-closed existente. Sem v3, sem push/merge,
sem relaxamento de gate.

## T1 — rebuild v2 medido no device (2026-10-10, rev233 Iteration; hook REVERTIDO)

Artefato de teste: rev233 Iteration (sem LTCG, SDK 26100 override,
test-sign manual com o mesmo cert; MSIX sha
23260736b2cef84efab3c5a7b3bf0c4499a2756d00cc3aa12d12621c22e58a9d).
Fontes = rev232 + hook T1 (flag-gated, fail-closed; patch preservado em
`bench/results/qwen4b-maxperf/slots-rev211/T1-rebuild-hook.patch`, src
revertido após a medição). Sem flag o hook é inerte (braço control
idêntico à rev232). TPS de Iteration NUNCA comparado com Final.

Protocolo (slots-rev211/T1-rebuild-test.sh + T1-longgen.sh): seed(40) →
save → restart verificado → preload mínimo (restore exige sessão
residente) → restore → continue (prompt idêntico, greedy). 3 braços
curtos + 2 longos (64 tok, resposta referencia o contexto inicial).

| braço                                   | restore                      | turno        | prefill cauda/total | drafted | accept   | texto     |
| --------------------------------------- | ---------------------------- | ------------ | ------------------- | ------- | -------- | --------- |
| control                                 | 7,9ms                        | (1,0,62,62)  | 677,0 (62)          | 4       | 3        | BRAVO-7   |
| full (harvest 418,6 + process 20,6)     | 474,1ms                      | (1,42,20,62) | 251,1 (20)          | 4       | 3        | = control |
| carryonly (harvest 457,0 + process 0,0) | 495,4ms                      | (1,42,20,62) | 232,8 (20)          | 2       | 2        | = control |
| full64                                  | harvest 418,8 + process 20,3 | (1,42,42,84) | 465,5 (42)          | 34      | 29 (85%) | 318ch     |
| carryonly64                             | harvest 440,1 + process 0,0  | (1,42,42,84) | 448,2 (42)          | 32      | 28 (88%) | = full64  |

Conclusões medidas (corrigem a análise anterior onde indicado):

1. Rebuild full a partir de v2 FUNCIONA: kv_before=42, pf=cauda,
   drafted/acceptance IDÊNTICOS ao control, texto idêntico —
   equivalência de estado por contadores + texto, não só texto.
2. SEM ganho de latência: harvest roda a taxa de prefill (418,6ms/42tok
   ≈ 10,0ms/tok vs 677ms/62tok ≈ 10,9ms/tok). Totais: full 725ms vs
   control 685ms. O forward pass sobre tokens mantidos é irredutível
   para hidden states exatos — nada em v2 o evita.
3. CORREÇÃO: carry bare é funcionalmente ADEQUADO (88% ≈ 85% do full,
   texto idêntico) — a pending row carrega o sumário do prefixo
   profundo; o catch-up da cauda supre o histórico recente. A cláusula
   "mirror holds the prefix" NÃO se isolou como falha nestes prompts.
   Piso de custo idêntico (harvest da última linha exigido do mesmo
   jeito).
4. Quantificado mas NÃO adotado (MTP deve ficar ligado): primeiro turno
   clássico pós-restore (só cauda, sem harvest) ≈ 258ms vs 685ms (2,6x)
   — só para o cálculo do prêmio da v3.
5. Prêmio v3 em base medida: restore ~10ms (blob+espelho) + prefill só
   da cauda COM drafting desde o turno 1 ≈ 260ms vs 685ms — ~425ms por
   continuação pós-restore neste tamanho. Proposta mínima inalterada
   (aguarda autorização; nada implementado).
6. Memória: pico rev232 = 3675MB (blocos b1+b2, estável vs 3674 rev227);
   T1 sem alocação persistente nova; h_rows transitório <0,5MB aqui
   (21MB pior caso em n_ctx cheio — chunked/liberado).

Impedimento estrutural (reproduzível, evidência em slots-rev211/T1 e
T1-longgen: JSONs + slice.log por braço): n_seq_max=1 (default, parte do
fingerprint) proíbe scratch sequence — harvest reusa a seq 0
(clear → harvest → set_data); re-decode sobre linhas ocupadas não
tentado (append-only). Relatório integral: slots-rev211/T1-report.txt.

Estado do device ao fim: rev233 Iteration instalada (hook dormente sem
flag — flag ausente verificado 404; slots vazios; servindo). Artefato
validado segue rev232 Final (MSIX preservado); REINSTALAR rev232 antes
de quaisquer gates finais. UWP monotônica preservada (233 > 232; sem
nova Final cunhada para teste).

## M3 zero-carry — via v2 eficiente medida (2026-10-10, rev234 Iteration)

Hook estendido com modo `zerocarry` (carry zero, SEM harvest; patch em
slots-rev211/T1-rebuild-hook.patch, src revertido). Todos os braços no
MESMO build (rev234 Iteration). Evidência: slots-rev211/T3-M3,
slots-rev211/T3b-M3x, relatório T3-M3-report.txt.

- zerocarry curto: restore 8,1ms (blob puro), (1,42,20,62), drafted=2,
  accept=2, texto = control. Total ≈ 244ms vs 685ms.
- zerocarry64: (1,42,42,84), 32/28 (88%) vs full 34/29 (85%);
  rounds/decodes/discarded 20/88/23 vs 21/89/21; texto idêntico.
- DIV-ZERO (cauda divergente 78tok): (1,42,36,78), 7/5 (71%), texto =
  braços DIV T2a. Reuso parcial sobrevive sem nenhum conhecimento real
  do prefixo.
- LONG-ZERO (n=151): (1,151,22,173), 21/17 (81%) = LONG-FULL; total
  ≈ 308ms vs 1885ms (full-harvest) vs 1866,6ms (fresh) — ~6x com
  drafting ativo. Texto idêntico ao LONG-FULL rev233 (cross-build).
- WS plano 3856–3858MB em todos os braços (delta de memória ≈ 0).

Leitura honesta: carry row E histórico do draft dispensáveis NESTES
prompts; validade geral ABERTA (famílias de prompts, gerações longas,
discriminação de prefixo profundo, código). MTP ligado o tempo todo
(drafting engaja no mesmo turno) — não é o modo clássico B.

REVISÃO da necessidade v3: v3 (blob do espelho) deixa de ser a única
via — passa a FALLBACK se M3 não generalizar. M3-first. Proposta mínima
(prod irrevisado: load path sem flag + acceptance-watch fallback +
gates Final completos) APRESENTADA, NÃO implementada — aguarda
autorização, como v3. Relatório T1 corrigido nos limites (equivalência,
carry-only, projeções) em slots-rev211/T1-report.txt.

## Baseline rev235 Final (2026-10-10)

Fontes verificadas idênticas às rev232 (hashes no receipt-rev235.txt);
Final LTCG 561s, SDK 26100 override (22621 ausente), mesmo cert,
assinatura automática. Instalada por upgrade 234→235 (sem uninstall,
LocalState/modelos preservados). Revalidação direcionada: vround canon
PASS (93/87 idênticos 209/216/232/235), slotsmoke PASS, api-recall PASS
(184/120). Evidência pesada rev232 transportada por identidade de
fontes + amarração (rationale no receipt, não oculto). Baselines
preservadas: 188/192/209/210/216/232 + 235. Sem push/merge, workflows,
reboot/reset, novas permissões. Iteração futura: rev236+ Iteration;
fechamento: Final.

FALTA (não declarar completo): validação ampla da via M3 (outras
famílias de prompts, caudas divergentes longas, gerações 256+, code) +
decisão de produtização (M3) ou fallback (v3) + gates Final completos
sobre o binário que levar a via adotada.

## Caminho selado M3 zero-carry + candidato Final rev238 (2026-10-10)

Mapa de estado (M3-state-map.md): blob restaura KV target (incl.
recorrente/GDN) + token ids; drafter após restore = espelho VAZIO abaixo
de kv_keep (buraco) + pending ZERO em n-1 (decoy) + carry snapshot
ausente. Igualdade de estado final NÃO alegada: `turn-end drafter state`
imprime o mesmo fim em ambos os braços enquanto os conjuntos de linhas
diferem — agregados não provam equivalência.

Amplificação T4 (mesmo build rev236, 5 famílias, braços fresh/control/
zerocarry/full conforme família):

- A cipher 42tok: zero (1,42,20,62) 271ms vs control 673ms, texto =
- B code 309tok: zero (1,309,34,343) 418ms vs 3722ms, texto = (81% acc)
- C std512 284tok: zero (1,284,25,309) 315ms vs 3333ms, texto = (72%)
- E deep-prefix 187tok ("qual cipher exatamente?"): zero (1,187,38,225)
  442ms vs control 2424ms, texto =, full-harvest 2028ms de restore
- F sufixo mínimo: zero (1,42,16,58) 188ms vs 622ms, texto =
- divergente (T3b): (1,42,36,78) 391ms, texto =
- multi-turn T4c (3 turnos pós-restore): mzero t1 (1,42,20,62) 259ms com
  drafting (control t1 = full 657ms!), t2/t3 idênticos ao control
  (66/22, 94/27), textos iguais nos 3 turnos.
  WS plano 3855–3858MB em todos os braços; restore ~8–14ms; rebuild
  zerocarry 0ms (full 1884+104ms em n=187) — custos separados, nunca
  embutidos no restore.

Implementação selada (rev237 Iteration → rev238 Final, patch
M3-zerocarry-sealed.patch, 49 linhas): `prime_zero_carry(pos)` pós
`align_to(0)` no `load_state`, fail-closed, invariantes no header
(mtp_draft.h). Sem flag, sem modo clássico, MTP sempre ligado.

Gates no Final rev238: bateria ON/OFF failures=0 (tabela repinada por
prova de protocolo: #5/#7/#9 viram (1,42,20,62); #8/#13 continuam full
por F3.5), slotsmoke PASS, api-recall PASS (184/120), vround canon PASS
(93/87 idênticos, 8/8 baselines). Receipt: acceptance-rev238/.

BLOQUEIOS/LACUNAS concretas restantes:

1. Evidência cobre 5 famílias + divergente + multi-turn 3 turnos; faltam
   gerações longas (256+) e discriminação deep-prefix com resposta longa.
2. Custo de acceptance por família (50–81%) não tem política de fallback
   automático — degrada para velocidade clássica sem corruptar texto.
3. Revisão do decoy row (pending zero) como contrato de design
   permanente — hoje medido como seguro via verificação do target.
4. Termgate/numerics/shapes/blocos 24comp do rev238 transportados por
   identidade de fonte + gates de amarração (decode loop intacto);
   reruns pesados só se o alvo mudar.

Fechamento das lacunas (2026-10-10, receipt acceptance-rev238):

1. FECHADA por medida: T5 long/deep (188tok pós-restore, aceitação 78%
   vs 85%, texto idêntico, WS plano).
2. FECHADA por medida + argumento estrutural: 12 cenários, aceitação
   50–100%, zero divergência de texto; fallback = decode clássico quando
   draft vazio; nenhum threshold inventado.
3. FECHADA como revisão: contrato decoy mantido (header mtp_draft.h) com
   a base medida above; nenhuma relaxação de gate.
4. FECHADA por medição direta: termgate failures=0, D2a 50/50,
   shapecost 282/282 no rev238. 24-comparisons transportado (racional
   registrado).

Objetivo 009: restore em processo novo com MTP + reuso real entre
requests VALIDADOS no rev238 com bounds nomeados; persistência
COMPLETA do estado MTP segue sendo v3 (proposta documentada, NÃO
autorizada). Plano não declarado completo quanto à via v3.

## S3 — igualdade de logits/continuação: registro preciso (2026-10-10)

Escopo: apenas a cláusula S3 "logits + multi-token continuation vs fresh
prefill (byte/token equality, no epsilon)". NADA de v3; formato, gates,
provedor e baselines intactos; sem push/merge; sem repetir testes
provados.

### Medido no rev238 Final (caminhos + hashes)

- Igualdade byte do texto da continuação restore-vs-fresh:
  `slots-rev211/T5/verdicts.txt` — fresh `CTOK=243/188 sha=b66495492af4`
  vs restore `CTOK=56/188 sha=b66495492af4` (geração longa 188tok,
  deep-prefix). Os ARQUIVOS json diferem por design (usage.prompt_tokens
  243 vs 56 = reuso vs prefill integral; sha256 dos arquivos
  a33ebd75de9d8683… vs 9fef95bf50bfa859…) — o que iguala são os BYTES do
  conteúdo retornado.
- Demais famílias: `slots-rev211/T4a/verdicts.txt` (A/B/C/F: shas iguais
  fresh=control=zero), `T4b` (E: 4 braços iguais), `T4c` (multi-turn
  t1/t2/t3 iguais). completion_tokens igual por braço.
- Determinismo greedy independente do restore: sidecars de tokens
  byte-idênticos entre 209/216/232/235/238
  (`acceptance-rev238/vround/`, `vround-canon-rev235.txt`).

### NÃO medido (gates exatos, ainda abertos)

- G-L: igualdade BYTE dos arrays de logits por passo de decode
  (restore vs fresh). Impossível na superfície atual: a API emite
  `logprobs: null` (`uwp/api-server.cpp:548-549`) e NUNCA seta
  `dump_logits_path` no caminho de chat; o único dumper
  (`uwp/inference-bridge.cpp:1395-1410` → `inference.cpp:653-672`) é um
  probe STANDALONE (contexto fresco, `n_predict=1`, prompt de arquivo),
  sem restore e sem multi-token.
- G-T: igualdade BYTE dos token ids amostrados da continuação. A API
  retorna só texto; a lane de bench que escreve sidecars `.tokens` não
  tem restore (`src/bridge/bench.cpp` sem `load_state`; `uwp/App.cpp`
  sem slots no caminho bench).
- Por que texto igual NÃO substitui: argmax idêntico é consistente com
  acúmulos f32 de baixa ordem diferentes; "no epsilon" exige comparação
  byte dos arrays/ids — que o caminho de restore hoje não emite.

### Separação do estado interno MTP (conforme instrução)

G-L/G-T são gates do caminho do TARGET. O espelho do drafter v2 é
PARCIAL após restore por design (buraco [0,kv_keep) + pending zero-decoy
— `slots-rev211/M3-state-map.md`, sha256 8b21a142332c4290…). Igualdade
de estado MTP interno NÃO é alegada e NÃO faz parte da S3; o impacto
prático do draft é limitado pelos 50–100% de aceitação medidos com
verificação pelo target.

### Menor teste necessário (NÃO executado — depende de build de probe)

Uma build Iteration (rev239), UMA mudança flag-gated no caminho de
generate da API: com `LocalState\chat-dump.flag` presente, gravar por
passo os token ids amostrados + os logits (reutilizando `write_logit_dump`
de `inference.cpp`) em `LocalState\chat-dump.bin` (+ `.done`); fetch via
WDP; sha256 fresh-vs-restore com prompts idênticos. Custo: build
incremental ~2–3min + 2 runs device ~4min. Fecha G-T integral e G-L por
passo; diagnóstico flag-gated/fail-closed; formato v2 intacto.
Registrado em vez de executado porque a mudança toca o caminho de chat
(não o de load selado no rev238) e merece escopo próprio de
build+validação.

## S3 — FECHADO com probe de bytes (2026-10-10, receipt slots-rev211/S3-receipt.txt)

Probe mínimo implementado e executado (flag `LocalState\chat-dump.flag`,
default OFF verificado; patch `S3-chatdump-probe.patch`; Iteration 239/240):
captura por passo, no momento DA AMOSTRAGEM (não no emit — o anchor é
emitido após um decode e a linha -1 já é outra; bug do probe 239
corrigida no 240), dos bytes de logits + ids amostrados.

- G-T (ids da continuação multítoken): BYTE-IGUAL fresh vs restore em
  processo novo — 4/4 passos (n=42) e 32/32 (n=187). Texto byte-igual.
- G-L (bytes dos logits): BYTE-IGUAL no caso single-ubatch (3.973.136 B);
  no caso multi-ubatch diferem na classe f32 (max_abs 4,5e-02) — e o
  CONTROLE prova que a diferença é do caminho aritmético de reuso, não do
  restore: restore-vs-restore também não é byte-igual (1,9e-02) enquanto
  fresh-vs-fresh é byte-igual; argmax idêntico em 32/32 passos.
- Conclusão S3: igualdade de token/texto PASSADA byte a byte; igualdade de
  logits PASSADA byte a byte só com prefill fresh em ubatch único — acima
  disso o gate byte de logits não é atingível no caminho de reuso (e não é
  propriedade do restore). Sem epsilon aplicado; limites registrados no
  receipt (itens 1–4).
- Preservado: rev238 Final intocado (rollback exige Final 241 por
  monotonicidade — documentado, não executado).

## G-L multi-ubatch: instrumentação pronta, build BLOQUEADA (2026-10-10)

Instrumento adicional (flag-gated, fail-closed, reversível): dump do blob
KV do target pós-prefill + posição do espelho MTP + pending row
(`chat-dump.kv`, `chat-dump.mtp`) no mesmo `chat-dump.flag` do probe S3;
acessores test-only em mtp_draft.h. Host build+ctest OK (só o caso D3D12
pré-existente). Experimento T7 desenhado: F1/F2 fresh e R1/R2 restore com o
MESMO arquivo de slot (isolar não-determinismo de procedência do slot —
T6d reconstruiu o slot por outro caminho de prefill, o que invalida a
comparação restore/restore como prova de não-determinismo).

BLOQUEIO (permissão, não de código): build rev241 Iteration parou no
pré-step de certificado do build-uwp.ps1 — `Get-PfxCertificate` no PFX
`uwp/xllama-test.pfx` (cert 89F82DB7, a confiada no device) falha com
"Access denied" (NTE_PERM 0x80090010) e `signtool /f` idem. Sessão
SSH/pwsh não-interativa perdeu acesso ao container de chaves daquele PFX;
criar cert novo é negado (NTE_PERM també). A store do usuário tem 3 certs
com chave privada (916D7760/21E98358 CN=xllama-dev, CD6D10EC
CN=Hjotha Xbox Probe) — NENHUMA é a 89F82DB7 confiada no device.

Menor mudança possível (APRESENTADA, NÃO executada — parada antes):
A) assinar com 916D7760 (existente, com chave) + `deploy.sh install-cert`
do .cer dela no device (1 passo, mesmo procedimento já usado na rev233);
impacto: segunda identidade dev confiável no device de teste; baseline
89F82DB7 intacta.
B) dono reabilitar o acesso à chave do PFX (login interativo na .193) e
rerun do build — zero mudança de identidade.
Sem push/merge; rev238 Final intacto (25ed611a…); fontes T7 preservadas
neste commit.

## G-L ISOLADO (2026-10-11, rev241 Iteration validado por hash; receipts T7/T7b)

Matriz T7 com o MESMO arquivo de slot restaurado duas vezes:

- F1==F2 e **R1==R2** byte a byte nos logits por passo → nenhum
  não-determinismo de execução; a diferença restore/restore do T6d era
  procedência do slot (reconstruído por outro caminho de prefill).
- Roundtrip restore→save: slot **byte-idêntico** ao fonte (58.824.767 B,
  0 diferenças) → o restauro é fiel ao blob; não há perda de estado.
- F1 vs R1: logits diferem deterministicamente (1º passo, max_abs 9,9e-03;
  argmax idêntico 32/32; ids e texto byte-iguais). Estado: slots pós-turno
  diferem em 27.389.118/61.284.467 bytes (offset inicial 546.171);
  MTP mirror_pos_max=229 igual, pending row difere (proveniência).
- Causa: prefill fresh em 4 ubatches (64/64/64/38) vs blob multi-turno +
  cauda de 43 — ordens de acumulação f32 diferentes. Limite estrutural do
  REUSO, não do restore; critério byte de logits inatingível no multi-ubatch
  (não relaxado; tokens seguem byte-iguais em todos os cenários).

## Pacote rev241 + G-L fechado (2026-10-11, receipt rev241-evidence-receipt.txt)

Gates pendentes com evidência própria na rev241: T2b 18/18 (isolamento,
concorrência 4/4, corrupt 4/4, cancel com save pós-kill, sessão íntegra);
bateria ON 24/0 e OFF 24/0 (atomicidade tmp+rename, stop/EOS, rollback,
divergente, identidade, chain, prefixo). Tempos separados: restore
8,3–8,4ms; prefill 480ms/43tok reuse vs 2358ms/230tok fresh; WS plano
3854–3858MB; estabilidade sem wedge em ~40 restarts verificados.

G-L (CORRIGIDO 2026-10-11 — ver T10 abaixo): T8 correlaciona geometria
mas NÃO isola acumulação (comparou slots de 187 vs 140 tokens com
fronteiras de turno diferentes); T9 prova somente re-prefill integral
(kv_before=0/reuse=0). Restauro fiel ao blob (roundtrip byte-idêntico,
T7b) e sem não-determinismo de execução (T7 F1==F2, R1==R2) permanecem.
A alegação "mecanismo comprovado" foi retirada.

## T10 — controle de trajetória + ON/OFF real (2026-10-11, receipt T10-corrections-receipt.txt)

Efeitos separados no MESMO ponto pós-prefill/pré-decode (rev242):

1. Serialização/restore: efeito ZERO — C (réplica fresca da trajetória
   T1→T2→T3 + cauda) == R (restore do slot + mesma cauda) byte a byte em
   KV do target, pending row, posições, ids, logits e tokens.
2. Pending row verdadeira vs zero-decoy: efeito ZERO no target (C==R),
   apesar de os espelhos MTP diferirem (C cheio [0,230); R com buraco
   [0,187) — M3-state-map). Espelho diferente + logits idênticos prova
   que o drafter não alimenta as decisões do target neste cenário.
3. Segmentação: única fonte da divergência fresh/restore — F (prompt
   integral em 4 ubatches) vs C (trajetória multi-turno + cauda), mesmo
   conteúdo: logits/KV/pending diferem, tokens iguais. A explicação
   f32/nível de kernel segue interpretação do efeito de segmentação.
   G-L: byte-igual quando a geometria das linhas compartilhadas coincide
   (incluindo mesma trajetória C vs R); no reuso multi-ubatch a diferença
   vem da segmentação, isolada por controle. Gate não relaxado.

ON/OFF real: KNOB do slot-battery.sh só rotulava diretórios (defeito do
harness — ambos os braços rodavam FFN ON por default desde 2026-10-09).
Corrigido: carrega d3d12swiglu.txt, reinicia (modo vinculado no startup)
e verifica a linha de modo do device. ON: mode=all, bind=3,
failures=0. OFF: mode=off, bind=0, ffn_disp=0 (api-recall),
failures=0. Config efetiva qwen35-4b-mtp/mtp=1/256/64/gpu=34 registrada
nos dois braços. Evidências: slots-rev211/slots-on|slots-off (rev242),
acceptance-rev242/api/receipt.txt.

## Espelho MTP investigado (T11/T11b, 2026-10-11; receipt T11-mirror-receipt.txt)

- Regiões válidas provadas pelo header do blob: C = cells @0 (completo);
  R = cells **@187** em todos os turnos — o buraco [0,187) NUNCA é
  reconstruído pelo restore. Presumir "parcial = benigno" foi substituído
  por medição.
- Bytes alinhados por posição: divergência contígua a partir da costura
  (187): t4=[187,209], t5=[187,241], t6=[187,278]; ponta recente
  convergence a byte-idêntico. Estado do drafter DIFERE do fresco — de
  forma limitada, medida e crescente.
- Caminho do target: logits, KV, pending, ids e posições byte-idênticos
  em 3 turnos consecutivos, com espelhos diferentes → o drafter não
  alimenta as decisões do target. Aceitação 77–95% (R) vs 82–90% (C),
  com rejeição/rollback ocorrendo nos dois braços (discarded≥0,
  zero recusas de seq_rm).
- Ownership/lifetime: restore dentro de processo com estado prévio
  produz KV byte-idêntico ao controle — set_data substitui integralmente.
- Fronteiras de turno IMPORTAM (T11b): H2 (2 turnos) ≠ C (4 turnos) ≠ F
  (1 turno), tokens iguais; com n_ubatch=64 fixo, a geometria do prefill
  (fronteiras + larguras) determina os bytes das linhas compartilhadas.
- ON/OFF real na rev242 (errata E1 no receipt rev241): ON mode=all
  bind=3 / OFF mode=off bind=0 ffn_disp=0, ambos failures=0, config
  efetiva registrada. Falso FFN_MODE_LINE_NOT_FOUND: causa raiz
  `grep -m1`→SIGPIPE(141)+pipefail (reproduzido; errata E2).
  AssertionError anterior identificado como âncoras de patch de docs
  (errata E3).
- Identidade rev242: sha256 89c9b5c9…, 22.047.094 B, Version 1.6.0.242,
  modo Iteration corroborado por 3 artefatos (binário Release-iteration,
  linha exclusiva de build, assinatura manual dev). Sem rebuild.

LIMITES ABERTOS: (a) crescimento do cone divergente do drafter — efeito
no target zero nos turnos medidos, arrasto em conversas muito longas não
medido; (b) uma família de prompts e 3 turnos pós-cauda; (c) a
explicação f32/kernel da diferença por geometria segue interpretação do
efeito medido.

## #170c — trailer MTP implementado e validado (rev243, 2026-10-11)

Correção mínima local e reversível da persistência (patch
`slots-rev211/plan009-mtp-trailer.patch`): o slot carrega, ALÉM do blob do
target, um trailer versionado (magic `XLMTPST1`, version 1) com o blob KV
do draft-context, o carry/pending REAL (não decoy), a extensão do espelho
e uma prova de prefixo (FNV-1a dos tokens [pos_min,n)). Escrito no mesmo
tmp antes do rename (atômico). Leitura v2 preservada: arquivo sem trailer
(corrente ou de leitor antigo) cai no caminho seguro documentado
(clear + prime) e NÃO é tratado como igualdade. Fail-closed comprovado em
5 variantes (magic/version/hash/truncado/sem-trailer): cada uma recusada
com motivo no log e sessão íntegra. Rollback: reverter patch + build
monotônica, ou reinstalar MSIX anterior.

Evidência (C = trajetória fresca idêntica vs R = restore com trailer,
mesmo ponto pré-decode, bytes estritos):

- logits, tokens, **KV/GDN (57.348.240 B)**, espelho e pending
  byte-iguais em T13a (t4/t5), T13b (curto/médio/longo) e T13c.
- posições: espelho completo (pos_min=0, cells=N) em todos os restores.
- aceitação/rejeição/rollback idênticos (drafted/accept/rounds/decodes/
  discarded iguais; fases de decode dentro do ruído de contenção).
- tempos separados: restore 73–78ms; prefill da cauda ~322ms/27tok;
  decode por MTP_STATS.
- gates na rev243: bateria ON/OFF reais (modos verificados) failures=0;
  T2b 18/18; vround 8/8; termgate 0; api-recall PASS (184/120).
- LIMITE: igualdade byte demonstrada para o controle alinhado (mesma
  trajetória); prefill integral vs restore segue diferindo em bytes com
  decisões idênticas (segmentação, T10/T11b) — não é equivalência global.
- Transportado (não re-medido): numéricos/shapes/24-comp — mudança só em
  session.cpp; decode/kernels intactos desde rev232; justificativa no
  receipt.

Identidade testada: PFN 1.6.0.243, MSIX sha256
2cc87201ae67b8dc913be49a2dedefa134bd83696697be3332e2c61accca5900,
22.049.003 B, Iteration. Receipt: slots-rev211/rev243-mtp-trailer-receipt.txt.

## Fechamento #170c (2026-10-12): famílias ampliadas + Final/LTCG

Pendências do receipt rev243 fechadas:

1. **T14 — famílias/trajetórias ampliadas com controle alinhado (rev243):**
   code (spec-code-edit), std512 (standard-512) e trajetória com turno
   DIVERGENTE — cinco classes byte-iguais em todos (logits, tokens,
   **KV/GDN**, espelho, pending), espelho completo (pos_min=0), texto da
   cauda idêntico. Receipt: slots-rev211/rev244-closure-receipt.txt.
2. **Final/LTCG (rev244)**, mesmas fontes (verificadas), upgrade monotônico:
   - #170c alinhado (T15): 5/5 byte-iguais com os MESMOS shas da
     rev243 Iteration (logits f5e9300f, KV ede8a04e, espelho bd703bef,
     pending 807588c8) — os dois modos de build produzem bytes idênticos
     de estado/decisão (comparação de bytes, não de TPS).
   - bateria ON/OFF reais (modos verificados): failures=0.
   - vround 8/8 (93/87), termgate 0, api-recall PASS (184/120).
   - MSIX sha256 638ee54d1115f13e6b65b68ec7a1425563952c0c7f4a45ecc6dd5701c7b0f56a,
     Version 1.6.0.244, Final/LTCG (link 8m37s).

Contagem corrigida (auditoria 2026-10-12 — a afirmação de "9 cenários,
cinco classes" era superdeclaração):

- **5 cenários com as CINCO classes byte-iguais** (logits, tokens,
  **KV/GDN**, espelho, pending): T13c, T14 code, T14 std512, T14
  divergent, T15 (Final).
- **5 cenários (T13a t4/t5 + T13b curto/médio/longo) com QUATRO classes
  medidas** — KV/GDN **não coletado** nesses; a igualdade do target neles
  repousa em logits/tokens byte-idênticos + os 5 cenários com KV.
  Distinção preservada: igualdade byte vale no controle alinhado; prefill
  integral vs restore segue diferindo por segmentação com decisões
  idênticas (T10/T11b) — sem equivalência global.

Estado: Xbox em rev244 Final servindo, flags ausentes, slots vazios.
Rollback: rev243/rev238 preservados. Sem push/merge.

## Auditoria final #170c (2026-10-12, receipt rev244-audit-receipt.txt)

Correções: contagem vigente 5+5 (cinco classes medidas só em T13c, T14×3 e
T15; quatro classes em T13a t4/t5 + T13b curto/médio/longo, com KV/GDN não
coletado nesses). Fingerprint: `slot-battery.sh:198` edita `hdr[3]` (campo
fingerprint) e `load_state` compara com `state_fingerprint()`; o rótulo
"wrong-version" não é teste de versão do trailer (esse é o T13a `version`).

Atomicidade — subgate ABERTO (não declarado fechado): dois saves
rejeitados antes da escrita (T16) e o `.tmp` stale inerte (B11) provam o
caminho, mas uma falha de I/O DURANTE a escrita não foi demonstrável com
segurança: kills por tempo caem após o rename; kill adaptativo com listing
do portal (parser corrigido) nunca viu o `.tmp` parcial em 3 tentativas;
medição do piso: 60 polls durante save de 55,5 MB, tmp observado 0 vezes
(janela tmp→rename abaixo do listing). Faltaria instrumentação inside do
app ou um meio controlado de falha de escrita — fora do escopo autorizado.

Distinção alinhado vs segmentação preservada; sem equivalência global.
Estado: rev244 Final servindo, slots vazios, flags ausentes. Sem push/merge.

## Atomicidade — demonstração determinística + explicação dos kills (T18, 2026-10-12)

- **F1 (falha no open)**: `t18.kv.tmp` como diretório → save recusado
  ("cannot open") e o **snapshot anterior preservado byte a byte**
  (sha a7518436…, restore+chat ok).
- **F2 (falha no rename)**: `t19.kv` como diretório → escrita completa,
  rename falha → **nenhum `.tmp` residual e nenhuma entrada parcial na
  loja** (tratamento do temporário comprovado deterministicamente).
- **Falhas pós-kill do T17 explicadas e reproduzidas**: save sem sessão
  residente (400 "chat first"); restore com o modelo default em vez do
  modelo do slot (recusa de fingerprint). A falha final do próprio T18
  era o diretório residual do F1 — confirmado por chat+save 200 após
  removê-lo. Nenhum defeito de produto.
- **Lacuna restante (exata)**: morte do processo no meio do blob deixa
  `.tmp` parcial — não demonstrável com segurança pelos meios atuais
  (executado 3x sem sucesso; 60 polls sem ver o tmp). Proposta: knob de
  injeção de falha em `save_state` (probe Iteration ~20 linhas, fora do
  escopo atual) — subgate de atomicidade fica **parcialmente demonstrado**
  (snapshot + temporário + rejeições + `.tmp` stale inerte), sem
  fechamento da morte mid-write.

Receipt: slots-rev211/T18-atomicity-receipt.txt. Estado: rev244 Final
servindo, slots e diretório vazios, flags ausentes. Sem push/merge.

## Matriz critério/evidência/status/lacuna (2026-10-12; leitura de 005, 008, 009 + recibos)

Leitura integral de `plans/005-qwen4b-maxperf.md`, `plans/008-complete-island.md`,
`plans/009-kv-slot-persist.md` e dos recibos `T18-atomicity-receipt.txt`,
`T18-host-midwrite-receipt.txt`, `rev244-closure-receipt.txt`,
`rev244-audit-receipt.txt` (mais `S3-receipt.txt`, `T10/T11`, `rev241-evidence`
citados por eles). Evidência = saída/comando real registrado; lacuna = o que
falta, com meios já esgotados.

### Objetivo atual no Xbox — plan 009 / #170c (gate de atomicidade)

| Critério                                                   | Evidência                                                                                                                                                                                                                                                                           | Status                                                | Lacuna                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Save/restore + reuso entre requests                        | bateria ON/OFF failures=0 (rev244, modos verificados); T2b 18/18 com concorrência 4/4 e corrupt 4/4 (rev241)                                                                                                                                                                        | **FECHADO**                                           | —                                                                                                                                                                                                                                                                                                       |
| Restore em processo novo com MTP (prime zero-carry selado) | rev238 selado; rev244 Final: kv_before>0, drafted>0, textos byte-iguais                                                                                                                                                                                                             | **FECHADO**                                           | v3 (espelho completo persistido) = aprimoramento NÃO autorizado, fora do gate                                                                                                                                                                                                                           |
| Tokens/texto byte-iguais fresh vs restore (G-T)            | S3-receipt: 4/4 passos (n=42) e 32/32 (n=187) byte-idênticos                                                                                                                                                                                                                        | **FECHADO**                                           | —                                                                                                                                                                                                                                                                                                       |
| Logits byte-iguais (G-L)                                   | T7 (R1==R2, roundtrip slot byte-idêntico) + T10 (C==R no mesmo ponto; segmentação é a divergência medida)                                                                                                                                                                           | **FECHADO COM LIMITE REGISTRADO** (gate não relaxado) | limite estrutural do reuso multi-ubatch, não do restore                                                                                                                                                                                                                                                 |
| Famílias ampliadas                                         | rev244-closure + audit: 5 cenários com cinco classes (T13c, T14×3, T15), 5 com quatro (T13a t4/t5, T13b×3)                                                                                                                                                                          | **FECHADO COM DISTINÇÃO**                             | KV/GDN não coletado nos 5 de quatro classes (registrado; não é superdeclaração)                                                                                                                                                                                                                         |
| Artefato Final/LTCG com a correção                         | rev244 PFN 1.6.0.244, MSIX 638ee54d…, Final/LTCG; vround 8/8, termgate 0, api-recall PASS, bateria ON/OFF 0                                                                                                                                                                         | **FECHADO**                                           | nenhum gate Final pendente na rev244                                                                                                                                                                                                                                                                    |
| Memória/estabilidade na rev244                             | T16: WS 87→3861 MB, plano 3857–3861, 3 restarts + restore + 3 turnos                                                                                                                                                                                                                | **FECHADO**                                           | um ciclo; sem telemetria térmica (limites registrados)                                                                                                                                                                                                                                                  |
| Recusa de fingerprint incompatível                         | T16 + `slot-battery.sh:198` (hdr[3]) + `state_fingerprint()`                                                                                                                                                                                                                        | **FECHADO**                                           | —                                                                                                                                                                                                                                                                                                       |
| Atomicidade: snapshot anterior preservado em save falho    | T18 F1 (device: sha a7518436 idêntico, restore+chat ok) + host (ref.kv byte-idêntico)                                                                                                                                                                                               | **FECHADO**                                           | —                                                                                                                                                                                                                                                                                                       |
| Atomicidade: temporário tratado em falha de rename         | T18 F2 (device: nenhum `.tmp` residual, loja sem entrada parcial)                                                                                                                                                                                                                   | **FECHADO**                                           | —                                                                                                                                                                                                                                                                                                       |
| Atomicidade: falha de I/O mid-blob (host)                  | ctest final 1/1 **exit=0, 69,78 s** no commit 62ff76a (após a instrumentação final); caso focal 49/49; 5/5 runs observando `max=181254` nos dois `.tmp`; três distinções explícitas no recibo (pré-criado/inode do observador; size≠fsync; I/O simulado≠morte/concorrência/handler) | **FECHADO NO HOST, COM AS TRÊS DISTINÇÕES**           | não substitui a demonstração no device                                                                                                                                                                                                                                                                  |
| Atomicidade: **morte do processo mid-write**               | T16/T17/T17b esgotaram os meios externos (kills caem após o rename; 0/60 polls viram o `.tmp`; encher disco proibido); strace/`ptrace_scope` recusados (2 recusas preservadas no recibo)                                                                                            | **ABERTO — lacuna exata**                             | exige probe in-app (~40 linhas em `session.cpp`, mudança de produção): plano concreto em `T18-midwrite-probe-proposal.md` (claim atômico por rename; binding por identidade de request pendente de variante a/b em `uwp/`+`session.h`) — **apresentado, aguarda aprovação antes de qualquer alteração** |
| Handler Xbox para falha de I/O mid-write                   | no device só F1/F2/T16 (open, rename, rejeições pré-escrita)                                                                                                                                                                                                                        | **ABERTO (depende do probe no device)**               | mesmo probe numa rev Iteration (rev245) + gates de binding (bateria ON/OFF, vround, termgate, api-recall) antes de qualquer Final                                                                                                                                                                       |

### Plan 008 (ilha elementwise + entregas derivadas)

| Critério                       | Evidência                                                                                              | Status                                                  | Lacuna                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Ilha SWIGLU+ADD+RMS_NORM+MUL   | rev174: transições −15,5%, gates numéricos verdes; timing não reproduzido (rev174 7 rounds)            | **ENTREGUE COMO OFF** (knob `d3d12island`, default off) | re-medição controlada opcional; exige decisão explícita (não autorizada agora)                                    |
| Config n_batch=256/n_ubatch=64 | rev188: prefill +8,2%/+7,4%, API −4,8%/−3,3%, digests/API/termgate verdes; rollback `llama.ini.backup` | **ACEITO/INSTALADO**                                    | —                                                                                                                 |
| FFN SWIGLU (GPU) + MTP juntos  | rev209 installed + default ON; acceptance 96 linhas 0 falhas; 24/24 mediana +4,32%                     | **ENTREGUE**                                            | rollback rev210 armazenado                                                                                        |
| Teto de decode (ceiling)       | orçamento fechado (1,9% residual); refutações por A/B (barreira de threads, volume, ilha)              | **BLOCKER DOCUMENTADO**                                 | sem alavanca mínima restante medida; próximos passos = projeto MMV maior ou decisão de-width — decisão do usuário |

### Plan 005 (Qwen 4B maxperf)

| Critério                                         | Evidência                                                                                                    | Status                                                               | Lacuna                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Configuração mais forte medida (ambos os braços) | rev126 baseline → 24/24 positivos rev209/244, mediana ~+4,4%; seq ~22,8–23,8 / MTP 27,3–34,0 tok/s no rev124 | **MAIOR PARTE FECHADA** (stop rule: gargomos dominantes endereçados) | candidatos do plans/006 (C8 etc.) não autorizados; nenhuma ação ativa |
| Regras de medição/paridade                       | protocolo A/B interleaved, perfil OFF, sha de tokens                                                         | **MANTIDO**                                                          | nenhuma                                                               |
| Trabalho 009 como subfluxo                       | entradas 8–11 no log apontam para 009                                                                        | **SEM MUDANÇA no 005**                                               | segue o status do 009                                                 |

### O que falta para fechar o objetivo atual no Xbox + próximo passo seguro

Falta **um item**: demonstrar, no device, falha de I/O **durante** a
gravação do blob (e, num escopo estrito, a variante morte-mid-write, que
segue coberta apenas estruturalmente). Os meios externos estão esgotados e
as rotas de privilégio foram recusadas — nenhum teste host adicional fecha
isso, porque o handler Xbox não é exercitado pelo processo host.

Próximo passo seguro (a ser aprovado, não executado — exige mudança de
produção, portanto plano apresentado primeiro): aplicar o diff já
desenhado de `T18-midwrite-probe-proposal.md` (knob `slot-fault.target.txt`
one-shot com claim atômico por rename, ~40 linhas em `session.cpp`,
default off), build **Iteration** rev245 em .193, rodar os 5 passos do §4
do probe (id `t20`, isolamento, snapshot preservado, `.claimed`, contenção
2 requests ⇒ exatamente 1 disparo), limpeza, e gates de binding
(bateria ON/OFF, vround, termgate, api-recall) antes de qualquer Final.
A vinculação por identidade de request (variantes a/b em `uwp/` +
`include/xllama/session.h`) é decisão separada. Até a aprovação: subgate
de atomicidade permanece **parcialmente demonstrado** (rotina + snapshot +
temporário + rejeições + `.tmp` stale + host mid-write com distinções),
sem fechamento da morte mid-write. Nenhum item dos planos 005/008 exige
ação para fechar o 009; nenhum gate Final novo é declarado aqui.
