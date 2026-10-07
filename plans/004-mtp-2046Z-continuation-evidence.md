# MTP: evidência da continuação de 2026-10-04T20:46Z

Rodada conduzida na worktree `/home/hjotha/worktrees/xllama-mtp`, branch
`spike/beellama-pin`. Objetivo: reconciliar o contexto da sessão de
implementação, acompanhar o teste `batch8` já em execução e registrar o
resultado real. Nenhum commit, merge, push, deploy ou acesso mutável a
`.193`/`.57`/Xbox foi feito.

## Estado reconciliado (antes do batch8)

- `HEAD` = `3bccef8344fa4c32db739992f342fa92eacab2f3` — _feat(mtp): plan 003 stages 1-4_
- Alterações locais preservadas, 5 arquivos, 97 inserções / 41 remoções:
  `src/bridge/decode_loop.h`, `src/bridge/mtp_draft.cpp`,
  `tests/test_decode_loop.cpp`, `tests/test_sampling.cpp` e o ponteiro do
  submódulo (dirty).
- `sha256` do diff worktree: `2e9465385c405110866d335adec95cf87836f1ddf062c7561eb3bc5a2990f8af`
- Submódulo `llama.cpp` = `982eaadaa9dbe761f00205bc20b6c04b7329b58d`;
  `sha256` do diff do submódulo: `42f640361cc758fdfd52607c106fcfbbf730da81604c60c39b0c06e709199b6a`
  — idêntico ao registrado no plano 003.
- Fixture: `/tmp/opencode/stage-mtp.gguf`, 2 834 975 040 bytes,
  `sha256` começando por `3874209241c9a397e2f62cd3f70f80fd`.
- Binário: `build/linux-local/tests/xllama-tests`, mtime 2026-10-04 22:27:50
  (+0200), construído a partir exatamente do diff acima.

### O que o diff implementa

1. `decode_loop.h` — o catch-up F2 foi movido para depois do verify/trim e
   passou a replays `n_committed = llama_memory_seq_pos_max(mem, 0) - pos_before`,
   lido do KV (verdade de campo), em vez de `n_feed`. Cobre EOG/stop no meio
   do percurso e não reencena tokens rejeitados.
2. `decode_loop.h` — `classic_step` agora também repass o token clássico
   committing (`mtp_catchup_batch` de 1 linha na posição corrente), porque o
   token clássico não passa por batch de verificação e deixava o mirror do
   drafter uma posição atrás.
3. `mtp_draft.cpp` — `draft()` passa a marcar `m_pending_valid = true`, para
   que a linha pendente sobreviva ao próximo round.
4. `mtp_draft.cpp` — `process()` só aceita o carry quando
   `m_pending_valid && m_pending_pos == pos0 - 1`; caso contrário registra
   `catch-up has no carry for pos0=...` e falha fechado (F2.3).
5. `mtp_draft.cpp` — `process()` faz `seq_rm` da cauda especulativa antes do
   replay, porque `draft()` havia escrito rows que o target nunca decodificou
   naquele contexto.
6. Testes — `mtp_run` ganha `expect_draft` (o caso de abort não tem o que
   draftar) e o teste de paridade KV-reuse passa a exigir `r2.n_drafted > 0`
   e comparar a referência fria sobre o texto cumulativo exato
   (`CHECK(r3.output_text == r2.output_text)`).

## Teste acompanhado: batch8

Não houve kill, interrupção, rebuild nem teste duplicado. O processo
`858211` (pai `858210`, `timeout 3600`) foi acompanhado até o término
observado.

Comandos (do executor anterior, preservados):

```bash
cmake --build build/linux-local --target xllama-tests -j4
XLLAMA_TEST_MODEL=/tmp/opencode/stage-mtp.gguf timeout 3600 \
  ./build/linux-local/tests/xllama-tests --test-case='mtp: session KV-reuse*' \
  > /tmp/opencode/mtp-fix-batch8.log 2>&1
```

Início 2026-10-04T20:27:46Z; término observado em 2026-10-04T21:01:03Z
(`tail --pid=858211` retornou; log com mtime 23:01:03 +0200, 153 283 bytes,
2784 linhas).

Resultado real do log:

```
[doctest] test cases: 1 | 1 passed | 0 failed | 296 skipped
[doctest] assertions: 8 | 8 passed | 0 failed |
[doctest] Status: SUCCESS!
```

Marcadores de falha **ausentes** no log (contagem 0):
`no carry`, `catch-up failed`, `drafting disabled`, `seq_rm refused`,
`FAILURE`. As linhas `round dbg` (instrumentação temporária) também não
aparecem mais — foram removidas conforme o plano.

Gerações observadas:

| etapa           | reuse | drafted | spec_accept | prefill               |
| --------------- | ----- | ------- | ----------- | --------------------- |
| turno 1         | 0     | 6       | 6           | 99 362,8 ms (5 tok)   |
| turno delta     | 1     | 12      | 8           | 91 479,2 ms (5 tok)   |
| referência fria | 0     | 12      | 8           | 497 336,8 ms (22 tok) |

`MTP_STATS` final: `rounds=0 decodes=30 discarded=1 catchup_tok=12
draft_ms=57098.8 sample_ms=182.7 topprob_ms=154.1 catchup_ms=49806.0
verify_ms=525525.0 corrective_ms=0.0`.

Contraste com o batch7 (que falhou, `run exit=1` registrado na sessão):
`catch-up has no carry for pos0=15 (pending_pos=13, pending_valid=1)` →
`catch-up failed — falling back to classic decoding` →
`catch-up has no carry for pos0=17` → `session prefill catch-up failed;
drafting disabled` → `Status: FAILURE!`. No batch7 aparecia ainda
`pos_before=24 n_feed=5 n_committed=1 tail_trimmed=1`, ou seja, round em que
apenas 1 dos 5 tokens alimentados foi commitado — exatamente o caso que o
`n_committed` lido do KV resolve. No batch8 esse caso aparece como
`n_committed=1 n_keep=1 tail_trimmed=1` **sem** falha de carry.

### Limite do que foi observado

O **exit code** do processo 858211 não foi observado diretamente: ele não é
filho desta sessão, então `wait()` não se aplica, e a linha `run exit=` do
runner ainda não foi gravada na sessão de implementação. O que está
observado é o veredito do próprio doctest no log (`Status: SUCCESS!`,
1/1 passed, 8/8 assertions, 0 failed). Nenhuma suíte mais ampla foi executada:
o filtro `mtp: session KV-reuse*` deixou 296 casos skipped.

## Checklist

Aprovado:

- [x] Caso específico `mtp: session KV-reuse*` executado com fixture MTP real
- [x] Paridade cold versus reused no caso (`r3.output_text == r2.output_text`)
- [x] Drafting sobrevive ao delta com prefixo reutilizado (`reuse=1 drafted=12`)
- [x] Carry contínuo entre round com draft e round clássico
- [x] Catch-up limitado ao prefixo efetivamente commitado (`n_committed`)
- [x] Logs de diagnóstico removidos
- [x] Nenhum assert afrouxado; nenhuma falha escondida por fallback silencioso

Falhou / não resolvido:

- [ ] Gate amplo `mtp:*` (rollback, reset, delta, chunks, EOG, stop, abort,
      limites) — não executado; exige a suíte completa
- [ ] Reconciliação de contadores com `n_generated`/`n_drafted`/`n_accepted`
      (`rounds=0` com `decodes=30` ainda não explicado)
- [ ] Gate Windows/Xbox e sweep de >=10% — dependência externa, não alcançável
      nesta máquina

Não executado (bloqueio registrado):

- [ ] Qualquer gate local amplo, porque o outro executor está vivo nesta
      worktree (ver bloqueio)

## Correção factual: o PID 886167 sou eu

Uma revisão anterior deste documento registrou um "bloqueio: outro executor
ativo" no PID `886167`. **Esse diagnóstico estava errado e foi removido.** O
argumento inicial `--session ses_ef9321855ffe46W43e7wRgitjt` não prova que outro
agente executa aquela sessão: a mesma UI navegou para
`ses_efa149467ffeO53KlU7FhEg3I0` e recebeu o steer `msg_108ac47ed001BnaYV3MBnO5ywf`.
A linha "running" no banco é estado persistido, não processo vivo.

Verificado por ancestralidade, com `ps` de PID/PPID/comm:

```
886167 (opencode) <- 880803 (bash) <- 880802 (bash) <- 1 (tmux) <- 0 (systemd)
```

`pgrep -c -x opencode` = 1 e `pgrep -c -x xllama-tests` = 0 no momento da
checagem. Não houve dois escritores: todas as edições subsequentes (o contador
`rounds`, `align_to`, `can_carry_from`, o fallback de re-prefill e os casos de
teste) foram feitas nesta mesma sessão, com build e formatação conferidos.
Nenhum arquivo alheio foi alterado e nenhum processo foi morto.

## Próximos passos propostos, nesta ordem

1. O usuário decide o executor duplicado: encerrar ou pausar o PID `886167`,
   ou deixar que ele prossiga e esta sessão apenas registrar. Sem essa
   decisão, dois escritores na mesma worktree.
2. Com exclusividade estabelecida, rodar o gate amplo local em
   `build/linux-local` com a mesma fixture, cobrindo `mtp:*` sem skips,
   greedy/seed, rollback, reset, delta, chunks, EOG, stop, abort e limites.
   Não usar o preset `linux-test`: ele força F16C e já produziu SIGILL
   neste N5105.
3. Explicar `rounds=0` contra `decodes=30` e reconciliar os contadores com
   `n_generated`/`n_drafted`/`n_accepted`, como exige o plano 003 §2.
4. Só então preparar os gates Windows/Xbox e reportar a dependência ao
   alcançá-la. Shaders só se o perfil de custo for favorável.

---

# Continuação: correções, gates e validação no Xbox

## Correções aplicadas nesta continuação

1. **Contador `rounds` (bug de instrumentação).** `n_mtp_rounds` era declarado em
   `decode_loop.h:177` e impresso em `session.cpp:898`/`inference.cpp:734`, mas
   **nunca era incrementado**. Incrementado no round com lote de draft não
   vazio, junto de `n_drafted`. Sem mudança de semântica.
2. **Vida útil do drafter no reset/troca de prompt (F3.5).**
   `MtpDrafter::align_to(pos)` alinha o espelho privado ao rewind do target, e
   `MtpDrafter::can_carry_from(pos)` prova se o carry serve à posição pedida.
   `session.cpp` chama os dois: espelho limpo quando `kv_keep == 0`; re-prefill
   completo quando `kv_keep > 0` e o espelho não pode carregar o prefixo. O
   drafter **não** é mais destruído por fallback silencioso.
3. **Critério do teste chunked.** A comparação era `substr(0,1)` — o primeiro
   caractere, nem o primeiro token. Substituída por igualdade integral de texto
   com o índice da primeira divergência em `INFO`. `n_batch` reduzido para 16
   para manter o caminho multi-chunk sem horas de prefill; a configuração
   exercitada é registrada no próprio teste.

## Defeito encontrado e documentado (gate-B2, antes das correções)

`mtp: reset and prompt swap keep the drafter honest` **falhou de verdade**
(exit 1, 9/10): `CHECK(r2.n_drafted > 0)` recebeu 0. Mecanismo no log:

```
session: KV prefix reuse none — resident 13 tok, prompt 3 tok (first mismatch at 0)
mtp: catch-up decode failed at off=0 rc=-1
mtp: session prefill catch-up failed; drafting disabled
```

O espelho privado conservava a posição 12, o novo prefill começava em 0,
`llama_decode` recusou, e o drafting foi desativado **permanentemente** sem
reconstrução. Esse é exatamente o item F3.5 do plano 003.

## Gates no host (N5105, `build/linux-local`, fixture `stage-mtp.gguf`)

| gate | caso                                        | exit | resultado                                                                                                      |
| ---- | ------------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------- |
| A    | `mtp: greedy parity`                        | —    | 6/6 asserções, 2162,6 s (dentro de uma invocação de grupo cujo exit foi 124 por timeout meu em caso posterior) |
| A2   | `mtp: fixed-seed sampling parity`           | 0    | **aprovado**, 6/6                                                                                              |
| A3   | `mtp: p_min=100 rejects every proposal`     | 0    | **aprovado**, 5/5                                                                                              |
| A4   | `mtp: stop sequence and EOG` + `mtp: abort` | 0    | **aprovado**, 2/2, 20/20                                                                                       |
| A5   | `mtp: n_predict limit`                      | 0    | **aprovado**, 7/7                                                                                              |
| B    | `mtp: session KV-reuse delta parity`        | 0    | **aprovado**, `rounds=4 decodes=30`                                                                            |
| B2   | `mtp: reset and prompt swap`                | 1    | **falhou** (defeito acima); inconclusivo após a correção                                                       |
| C    | suíte de sessão sem MTP                     | —    | **interrompido** pelo usuário (PID 1979200, órfão encerrado)                                                   |
| D    | `mtp: reset` pós-correção                   | —    | **interrompido** (SIGTERM, `CRASHED`); inconclusivo                                                            |

Contadores após a correção, no host: `rounds=9` (fixed-seed), `rounds=4`
(KV-reuse, com `decodes=30` igual ao batch8), `rounds=0 discarded=23` em
`p_min=100` — zero rounds correto ali, porque nenhuma proposta passou.

## Validação no hardware (Xbox, Rev 38)

Checkout `.193` trazido para `3bccef8` por `fetch` + fast-forward (sem push) e
os 6 arquivos corrigidos transferidos por scp com SHA-256 conferindo byte a byte
nos dois lados. Build MSVC/UWP: `Build succeeded`, `0 Error(s)`,
`00:20:02.90`, wrapper `exit=0`.

- msix do build: `xllama_1.6.0.38_x64.msix`, SHA-256
  `FCB14FF78AAF9682C38D93B78279AB51682C9C3706DE2C1ED8D37238B7C1CF6F`
- msix republicado/assinado: `xllama-q5k-fixed.msix`, SHA-256
  `892A9C3FA011E14F285D5746FFAF28E3708124D4F26EF7849EE59700E9D58D9E`
  (o próprio `build-local.log` registra o mesmo hash)
- console `192.168.1.26`: deploy aceito, `code=0 Success`, PFN instalado
  `GianlucaMazza.xllama_1.6.0.38_x64__pj67f1fcj4n14`
- backend **d3d12 real**, tag de host `xbox-series-s-u64-*-g99-noeog`

**Paridade por IDs inteiros (5 runs pareados, `--greedy --tokens`):**

| run | IDs baseline | IDs MTP | paridade | primeira divergência |
| --- | ------------ | ------- | -------- | -------------------- |
| 2   | 65           | 65      | idêntica | —                    |
| 3   | 65           | 65      | idêntica | —                    |
| 4   | 65           | 65      | idêntica | —                    |
| 5   | 65           | 65      | idêntica | —                    |
| 6   | 65           | 65      | idêntica | —                    |

**MTP comprovadamente ativo** no console: `MTP_ACTIVE n_drafted=35
n_spec_accepted=29` e `MTP_STATS rounds=16 decodes=128 discarded=30
catchup_tok=45 draft_ms=300.5 verify_ms=1505.7 corrective_ms=169.6`. O braço
baseline saiu `drafted=0 spec_accept=0`, então a paridade não é trivial.

| braço             | n   | decode mediana | min   | max   | prompt mediana | peak WS |
| ----------------- | --- | -------------- | ----- | ----- | -------------- | ------- |
| baseline greedy   | 5   | 18,52          | 18,43 | 18,56 | 83,38          | 3949 MB |
| MTP d4 p75 greedy | 5   | 18,95          | 18,83 | 19,00 | 64,63          | 4288 MB |

Ganho mediano pareado: **+2,3%**. O gate de ≥10% **não** foi atingido; o
resultado negativo está medido e documentado, não atribuído a D3D12 em geral.

## O que falta, com precisão

- **Inconclusivo:** `mtp: reset and prompt swap` e `mtp: edited full prompt with
a reused prefix` pós-correção — o código está no host e no console, mas os
  dois foram interrompidos antes de produzirem veredito.
- **Inconclusivo:** suíte de sessão sem MTP (Gate C), interrompida.
- **Dependente de harness no console:** os cenários de Session
  reset/delta/rewind com comparação de IDs não são expostos pelo
  `bench-xbox-ort.sh`, que faz uma geração por run. Expor isso exige uma rota de
  múltiplos turnos com `reset_kv`/`reuse_kv` e dump de IDs por turno no
  `uwp/inference-bridge.cpp`, ou seja um ciclo novo de build+deploy. O plano
  proíbe substituir reset de sessão por benchmark frio, então isso **não** foi
  contorno com bench de throughput.
- **Não reconciliado por completo:** os tempos de fase ainda não explicam o total
  (prefill e carga do modelo não entram em `draft_ms`/`catchup_ms`/`verify_ms`).
  `rounds` deixar de ser zero não implica reconciliação.

---

# Continuação: overhead da instrumentação, perfil por fase e histogramas

## Overhead ON/OFF — medido por segmento delimitado, não por log acumulado

O primeiro par ON/OFF foi **inválido** e está descartado: os logs do dispositivo
acumulam entre invocações, e a leitura misturou 39 linhas `PHASE` (ON) com 65
(OFF) de rodadas anteriores. Recalculado só sobre o segmento final — do **último**
`MTP_SESSION_CONFIG` até `bench-mtp-session.csv written`, 13 linhas por braço:

| braço | decode mediana | residual máximo                |
| ----- | -------------- | ------------------------------ |
| ON    | 592,6 ms       | 1,0 ms                         |
| OFF   | 591,4 ms       | (timers zerados por definição) |

**Não há conclusão de overhead.** Um único par heterogêneo não demonstra nada
abaixo da resolução do timer: a diferença observada (1,2 ms em ~592 ms) é
comparável ao spread entre execuções do mesmo braço. Para medir é preciso blocos
ON/OFF alternados e aleatorizados, pelo mesmo cenário, com repetições e
incerteza. Registrado como **pendente**, não como resultado.

O OFF é real e verificável no artefato: `profile=0`, todas as fases em 0,0 e
`accounted_ms=0.0`, com os 11 sidecars de IDs **byte a byte idênticos** ao ON. Os
contadores e o denominador (`t_decode_ms`, `t_first_token_ms`) não são gated — sem
eles o OFF não seria comparável.

## Correção que habilitou o OFF

`profile_phases` era transportado e impresso, mas **nenhum timer consultava a
flag**: o `--profile 0` media instrumentação ligada. Gateado em 24 pontos do
`decode_loop.h` e nos 4 acumuladores de tempo do `mtp_draft.cpp`, mantendo os
contadores. Também corrigido: `account_maintain` repetia `apply_stop_sequences`,
duplicando trabalho e mutando `output_text` duas vezes por token — a função agora
faz só `accept_token`, e o scan de stop fica só em `emit_token`.

## Residual: agora medido, não residual

Com `sample_target_ms`, `emit_ms`, `maintain_ms` e `classic_catchup_ms` medidos,
o residual caiu de **9,5–12 ms** (13 chamadas) para **0,0–1,0 ms**. As fases
somam 97,7–99,8% do decode. Medianas no ON: `verify_ms` 274,1 · `classic_decode_ms`
370,7 · `draft_ms` 40,3 · `catchup_ms` 37,2 · `topprob_ms` 27,2 · `sample_ms` 9,2 ·
`sample_target_ms` 10,4 · `classic_catchup_ms` 26,9.

**Regime por contexto, sem rotular errado:** em `delta_continuation` e
`multi_chunk` o `verify_ms` domina (309–424 ms com classic em 0); em
`reset_prompt_swap` domina o `classic_decode_ms` (596–644 ms com verify em 0).
O turno `reset` é uma **referência sequencial** em que o drafting não avançou —
não é um cenário de reset, e não será rotulado como tal.

## Perfis por tipo/shape: instrumento pronto, dados ainda não coletados

Os contadores de lifetime eram zerados em `backend_free`, então não atribuíam
tempo a fase nem a shape. Adicionado:

- `d3d12_shape` por `graph_compute`: type, N, K, B, threads, calls, gpu_ms e
  wall_ms **daquela chamada** (buffer dimensionado pelo número de nós, sem
  truncamento silencioso);
- `PHASE_PREFILL`: delta de calls/matmuls/gpu em volta do prefill real da sessão,
  sem decode extra e sem mudança de estado;
- acessores somente-leitura `d3d12_graph_calls/matmul_count/gpu_ms`, com stubs
  zero no host CPU, para o delta por fase.

O selftest B=1/2/3/5 nos shapes do Qwen permanece rotulado como **matmul
sintético**, não `T_target(B)` de decode: o seed depende de B e não cobre
KV/DeltaNet. `T_target(B)` verdadeiro exige prefixo, tokens, KV e estado
recorrente equivalentes restaurados entre medidas.

## Ainda pendente, em ordem

1. Blocos ON/OFF alternados e aleatorizados, com repetições e incerteza — a
   medição de overhead honesta.
2. Coleta do histograma `d3d12_shape` por fase/contexto para eleger o tipo
   dominante com dado, não com hipótese.
3. `T_target(B)` de decode completo com estado equivalente restaurado.
4. Só então decidir o tile de duas colunas, mantendo o caminho B1, a ordem FP32
   por coluna e o tratamento da coluna ímpar.
5. Protocolo de produto pareado: 3 prompts, 64 e 256 tokens, profundidades 1/2/4,
   IDs completos e incerteza do ganho. A rodada exploratória anterior (+2,3%) **não**
   é o protocolo final.
6. Evidência dos gates de borda ainda não executados nesta revisão — rollback de
   logits, EOG/stop/cancel/limite. Não são verdes por inferência a partir do reset.

## Correção do deadlock e do histograma (antes de qualquer deploy)

A primeira versão do `d3d12_shape` **não foi implantada**: tinha defeitos que
seriam fatais no console.

- **Deadlock.** `backend_graph_compute` já segura `g.mu` com um `lock_guard` no
  topo da função; o bloco novo tentava `lock_guard` no **mesmo** `std::mutex`
  não-recursivo. Removido: o histograma apenas acrescenta. Verificado por
  varredura estática — 1 lock em `backend_graph_compute`, 0 dentro do callback
  `run_now`, 1 em cada acessor e em `backend_free`.
- **Chave e campos errados.** O tipo agora vem de `w->type` (o destino `node->type`
  é F32), `N = w->ne[1]`, `K = w->ne[0]`, `B = node->ne[1]`, coerente com
  `d3d12_mm_dispatch(node->ne[0], node->ne[1])`; `thr = d3d12_mm_threads(w->ne[0])`.
- **`off` nunca avançava.** Cada `snprintf` sobrescrevia o mesmo ponto, e
  gpu/wall sobrescreviam todos os shapes. Substituído por agregação em mapa por
  chave com descarte por grafo, teto explícito de 24 shapes e marcação
  `(truncated)` quando corta. Lógica extraída e testada isoladamente: 42 nós
  agregam em 4 chaves, a chave repetida soma, e o grafo seguinte começa limpo.
- **Tempo não atribuível.** Um par de timestamps envolve o **grafo** inteiro, então
  `graph_gpu_ms`/`graph_wall_ms` são rotulados como tal e **não** atribuídos a cada
  shape; só `mm=` (chamadas por shape) é por shape. Um registro por grafo, não por
  nó: logar por chamada altera desempenho e volume.
- **Hot path.** O histograma tem switch próprio (`d3d12_set_shape_log`), ligado ao
  mesmo knob do bench, separado dos timers de decode porque construir a string é
  trabalho real dentro de `graph_compute`.

## O histograma estava truncado — perfis invalidados

A conclusão anterior de "todas as chaves `complete=1`" era **falsa**: o
`evid-147` contém 26 drenagens e **quatro truncadas** (`session_decode keys=79
complete=0` e `keys=50 complete=0`). O teto de 48 cortou estados reais. Os seis
blocos ON/OFF (`onoff-1..6`) usam esse mesmo coletor com teto e estão
**invalidados como perfil** — 8 a 16 drenagens truncadas por log. Eles continuam
válidos apenas como evidência dos IDs e dos gates (6/6 `gate=1`), não como
medição de overhead nem de completude de shapes.

Correções aplicadas:

- **Sem teto.** A drenagem é **paginada**: `part=i/n keys_total=N`, todos os
  registros por fase/contexto, cada linha terminada em newline (havia linhas
  coladas `1[xllama]` por falta de `\n`).
- **Parser falha fechado.** O analisador recusa a medição se **qualquer** drenagem
  do segmento tiver página faltando, ou se alguma linha `PHASE` tiver o flag
  `profile` errado para o braço. Nunca filtra para as completas. Verificado:
  ele rejeita o formato antigo exatamente como deve.
- **Contagem de matmuls não é custo.** O perfil mostra Q4_K com mais chamadas
  (9072 em classic, 5240 em prefill, 3200 em verify), mas contagem **não** prova
  custo dominante — o vocabulário Q6 (`t=14 N=248320`) pesa por bytes, não por
  ocorrências. Q4_K é **candidato**, não domínio de tempo estabelecido. A escolha
  de tipo exige custo GPU/wall medido e `T_target(B)` verdadeiro no mesmo estado,
  não a atribuição do tempo do grafo a cada tipo.

## Medição ON/OFF válida (rev48, coletor paginado)

Três execuções por braço, alternadas (`ON OFF OFF ON ON OFF`), mesmo
pacote/config/GGUF, cada uma com segmento isolado e drenagens completas
(`session_decode` 4/4 partes, 43 chaves; `session_prefill` 1/1, 12 chaves; 26
drenagens por execução). Mediana de `decode_ms` por execução:

| execução | braço | mediana decode | drenagens    |
| -------- | ----- | -------------- | ------------ |
| 1        | ON    | 591,5 ms       | 26 completas |
| 2        | OFF   | 593,0 ms       | 0 (OFF real) |
| 3        | OFF   | 591,3 ms       | 0            |
| 4        | ON    | 611,3 ms       | 26 completas |
| 5        | ON    | 597,5 ms       | 26 completas |
| 6        | OFF   | 588,5 ms       | 0            |

Diferença das medianas: **+6,0 ms (ON − OFF)**. Bootstrap sobre as medianas
por execução (10 000 reamostragens): **IC 95% = [−1,5; +22,8] ms** — o intervalo
**contém zero**. Com três execuções por braço a perturbação da instrumentação é
**indistinguível de zero**, com limite superior de ~23 ms (~2–4% do decode). Não
é "abaixo da resolução do timer" nem um número pontual; é um intervalo que inclui
zero.

Cobertura: `accounted/decode` = **99,92%** no ON, residual máximo **1,00 ms** em
443,9–661,5 ms de decode — ou seja **99,85–100,0%** atribuído, não 97,7–99,8%.

## Correção do coletor (teto arbitrário removido)

O teto de 48 chaves cortava estados reais: `evid-147` tinha 4 drenagens truncadas
(`keys=79` e `keys=50`), e os blocos `onoff-1..6` tinham 8–16 cada. Substituído
por emissão **paginada sem teto** (`part=i/n keys_total=N`, cada linha com
newline — havia linhas coladas `1[xllama]` por falta de `\n`). O parser falha
fechado se qualquer drenagem do segmento tiver página faltando, e não filtra para
as completas. Os perfis anteriores ficam **invalidados como perfil**; seguem
válidos como evidência de IDs e gates.

## Q4_K é candidato, não domínio de tempo

O histograma do rev48 mede **contagem** de matmuls por contexto/fase, não custo:
`target/classic` 12 663, `prefill` 7 448, `verify` 4 501, `draft/draft` 912,
`draft/catchup` 476 no segmento. Q4_K (`t=12`) tem mais ocorrências, mas o
vocabulário Q6 (`t=14 N=248320`) pesa por bytes. Elegibilidade do tile de duas
colunas exige custo GPU/wall medido e `T_target(B)` verdadeiro no mesmo estado —
o ensaio por profundidade de draft deu spread de 127,9–466,6 ms no mesmo braço e
**não** estabelece B1/2/3/5.

## 2026-10-05 rev55/rev56: cost x frequency ranking + two-column variant design

Drain semantics (source): `d3d12_shape_drain` swaps the buffer, so `mm=` is
per-drain; summing the 54 `label=tttarget` lines is valid. Two scope
populations: `phase=tttarget` (inside the measured interval, `tttarget.cpp:502`)
is the authoritative decode work; `phase=decode` is reference/parity work
(B=1/5 only). Both agree on the top set.

Ranking (Rev55 single samples, confirmed with Rev56 medians of 5; in-run
ranges 1-5% of median):

- #1 at every B >= 2 is `q4_k 9216x2560` (31-37% of matmul GPU cost).
  lm_head is #1 only at B=1 (27.6%), #2 at B=2/3, minor at B=5.
- #3 is `q5_k 8192x2560` (~13-15%), #4 `q6_k 2560x9216` (~10-15%).
  Repeated Q4 matmuls dominate in aggregate, not the largest single shape.

Variance finding: in-run ranges are tight, but the lm_head median moved
12-13% between the rev55 and rev56 runs (small shapes <1%). OLD vs NEW must
be compared back-to-back in the SAME package/run, never across runs.

Narrowest justified variant (q4_k only, design, not yet implemented):

- New dxc flavor `-D TWO_COL=1` of `shaders/ggml_d3d12_mmv_q4_k.hlsl`
  (dxc.exe confirmed on .193: `Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe`),
  emitting `ggml_d3d12_mmv_q4_k_t{64,128}_2col_dxil.h` via
  `scripts/compile-gpugemv-shader.sh` (regenerate only the new targets).
- One group computes columns `2*gid.y, 2*gid.y+1`: weight loads shared,
  per-column x loads/accumulators duplicated with the EXACT FP32 expression
  order; `reduce_store` reused sequentially per column (groupshared unchanged).
  `ncols` rides in root-constant `pad0` (no signature change); tail guarded by
  uniform `col1 < ncols` (no wave divergence); B=1 always dispatches OLD.
- Backend: new PSO + `d3d12_set_kernel_variant(0|1)` runtime switch; bench runs
  OLD then NEW back-to-back per q4_k B>=2 case with a `variant` column
  (gate CSV untouched, thresholds unchanged). Host tests: dispatch planner
  groups_y=ceil(ncols/2); on-device parity via rel_err per row.
- If the measured delta is too small, document and pivot to measured
  non-kernel costs; no broad shader work.

## 2026-10-05 rev57: q4_k two-column tile measured (paired OLD/NEW, same tensors)

dxc 1.8.2502.11 on .193, `-D TWO_COL=1`, cs_6_0: t64 11440B (OLD 8912),
t128 11876B (OLD 9112). Review fix applied before build: dispatch uses the
`d3d12_kernel_variant()` getter (the static sat below its first use), and rows
carry `twocol` = recorded NEW dispatches (counter delta), not the request.

Rev57 Xbox run (`d2a-d3d12-selftest-20261005-2col*.csv`): D2a 24/24 PASS,
188/188 shape-cost rows ok=1, max rel_err 2.61e-7 (tol 1e-2). tc evidence:
80/80 NEW pair rows tc=1, 80/80 OLD pair rows tc=0, all summary rows tc=0,
zero TCMISMATCH. Device free-log: 408 matmuls (120 2col) = 20 paired cases x
(4 timed + warmup + verify) NEW computes, exact. B1 stayed OLD throughout.

Same-run pair deltas (NEW-OLD, 4 pairs each, ranges << |median|):

- Winners (all real triples >= 2560x4096): 1.18x-1.59x. Top contributor
  q4_k 9216x2560 B=5: d_med=-0.0946ms, range 0.0073 (13x separation).
- Regressions (documented, default OLD retained): 1024x2560 B=2 0.77x,
  B=3/5 ~0.96x; synthetic 1023x2560 B=2 0.75x, 1x256 B=2 0.74x.
  Correctness holds everywhere (odd N, N=1, tails, pads via host emulation).
- Deployment rule for the future real-decode flag (not yet wired): engage NEW
  only where measured wins with margin (interim: n >= 2048, conservative;
  1025-2559 unmeasured). No global enablement without that gate.

Next: real-decode flag path + engagement rule, then target logits/IDs,
Session gates, rollback/edge gates, paired product protocol.

## 2026-10-05 rev58-61: allowlist knob, padded gate, two bench bugs, paired matrix

Variant 3-state (0 OLD / 1 force-bench / 2 auto-allowlist) + hardcoded
allowlist (5 q4_k shapes x B=2/3/5) + `d3d12twocol.txt` knob in run_tttarget
with scoped restore (`--twocol off|auto` in bench-xbox-ttarget.sh, knob
deleted on exit). Host suite 306/11547.

Padded gate (rev58): strided X/Y/W + sentinels + tails, both variants. Two
bench bugs found by device evidence, both fixed + host-pinned:

1. reference pitch conflated packed size with nb[1] (rel_err=1 both variants
   while kernels agreed) -> pass w_nb1, param renamed row_pitch.
2. ggml_nbytes has no trailing last-row/col storage -> padding checks
   over-read heap garbage (false "clobbered") -> skip final row/col.
   Rev61: D2a 24/24 PASS, 204/204 rows ok, max rel_err 2.61e-7, 16/16 padded rows
   ok with tc=1 on NEW. X/Y/W stride handling proven on Xbox for OLD and NEW.

Paired target matrix rev61 (same package, identical prefixes/seeds/model):
OLD vs auto arms, 24 rows each, argmax_match 24/24 both. NEW>0 proven in log
(240/120 2col matmuls). Wall: B1 -0.19% (OLD path, negative control), B2
-1.70%, B3 -3.47%, B5 -2.74%. GPU: -2.2%/-10.3%/-6.4%/-9.6%. End-to-end drift
(NEW vs sequential reference): maxabs 0.0033, maxrel 0.0002, min margin
0.1185 (36x headroom, greedy safe; stochastic needs the protocol).
Verdict: kernel wins do NOT reach the >=10% product gate (wall +2-3%).
Default OLD retained. Negative-on-gate result stands unless the 3-prompt
protocol shows otherwise.

## 2026-10-05 rev64: session gates, EOG gates, 3x2 product protocol

Fixture record: matrix fixture qwen35-4b fbfa70529de15cc8; session fixture
qwen35-4b-mtp f3559ed1a2922585 (different files, documented in
tttarget-20261005-provenance.json; msix 1.6.0.61/8ea26ed3 for matrix,
1.6.0.64/069c5243 for session work). MTP model cannot load with full D3D12
offload (fork: MTP host backing needs CPU/CUDA buffers); partial
gpu-layers=28 keeps the head CPU-native and loads, with base matmuls on D3D12.

Session gates (partial-28, pmin75, rev64): OLD 4/4 PASS, AUTO 4/4 PASS, all
token dumps byte-identical across arms (8/8 files), NEW>0 in auto (452+ 2col
matmuls in log). Knob: TwocolKnobScope in main_loop + run_tttarget, --twocol
in both bench scripts, knob deleted on exit; stale pmin leak fixed.
Verdict expect_pmin default 0->-1 (was demanding -pmin0 on default runs).
CSV reason now RFC-4180 quoted; verdict rejects malformed rows (strict).

EOG gates: EOG-on OLD vs AUTO ids identical (64 tok); EOG-on vs --ignore-eog
AUTO identical. No EOS observed on fixtures (documented limitation; EOG
machinery is variant-independent given identical logits). Cancel: no runner
exists (UI/API infra, non-numerical) — not covered.
Rollback: exercised implicitly via MTP mismatch paths in session scenarios
(parity held); no isolated rollback-under-mismatch datapoint.

Product protocol (rev64, qwen35-4b-mtp, gpu28, greedy+seed, --runs 3 = 1
warmup + 2 recorded, rotated arm order, 18 runs, dumps preserved):
ids old==auto in ALL 12 cells (kernel parity at 64/256 tokens). MTP vs
sequential diverges on 2/3 prompts (token 33/39, pre-existing MTP exactness
issue, kernel-independent — OLD and AUTO agree byte-for-byte).
decode tok/s: MTP benefit 1.02x-1.52x (prompt-dependent); kernel benefit
+0.3% to +2.3% (6/6 positive, at run-to-run noise edge in some cells).
Product gate >=10%: NOT MET. Default OLD retained.

## 2026-10-06 rev68/69: controlled replay verdict (actual block, row 2)

Harness repaired from evidence (not constants): prefix = 119 prompt +
30 committed ids (outputs 0..29), each part asserted separately; block =
observed feed [1902,1353,506]@[149,150,151] (feed log capture); all rows real,
no dummy; row 2 (bonus, predicts divergent pos 152) is the verdict after rows
0..1 prove formation. Flags match failing runs (rs_seq=4, nextn on, gpu28,
load_mtp, OLD kernels). compare_replay_rows fixed (raw argmax, was max-abs)
with negative-logit + non-finite controls (host suite 307/11563... recount
after test adds).

Rev69 replay (qwen35-4b-mtp f3559ed1, rev69 238a11e9): branch A reproduces
sequential (final 279, margin 0.002611); branch B row 2 = 279 identical;
maxabs=0.0 maxrel=0.0 argmax_match, both reps bit-identical, finite. Batching
arithmetic (all ops, incl. SSM/recurrent under these flags) is BIT-EXACT on
identical prefix/KV/positions.
Verdict scope: the live MTP-vs-sequential divergence is NOT batching
numerics. Remaining cause class: accumulated state at batch time (KV /
recurrent / carry / positions), since prefix tokens, flags, kernels and
batching are all controlled identical. Kernel exonerated again (OLD==AUTO
everywhere + this). Defaults OFF/OLD retained; no production consequence.
An unexplained rev67-tagged replay-failure line pair exists in the cumulative
log (message text matches no committed source; failed safe by refusal).
Noted, not attributed; the rev69 verdict stands on its own complete evidence.

## 2026-10-06 rev72: live-path replay verdict (tail+trim shifts logits)

A/B/C replay of the ACTUAL idx=23 round (feed [8772,279,84810,11]@[142..145],
keep 3, trim 145, corrective 2261; prefix prompt119+outputs0..22; rs4/nextn/
gpu28; all asserts passed, deterministic both reps):

- A-vs-C (accepted 3-batch vs singles): maxabs=0 (batching exact, as rev69).
- B-vs-C (4-batch + trim-to-144 vs clean): maxabs=0.0202, maxrel=8.9e-4,
  argmax preserved here (47973, margins 0.164 vs 0.159).
  The batch+trim path shifts logits ~0.02 systematically vs the clean path.
  Same argmax here only because the margin is healthy; at the step-33 boundary
  (margin 0.0026) a shift of this size flips 279->1204. Consistent with the
  live divergence fingerprint exactly.
  Scope, honestly bounded: proves the live reject+trim path (not batching
  arithmetic, not kernels, not flags) moves logits at the observed scale.
  Does NOT separate batch-width numerics (4-wide vs 3-wide) from trim residue
  (rows 0..2 of B were asserted, not recorded); neither fix is in scope (both
  live in fork batch/trim/recurrent machinery). Drafter/shared-KV side effects
  remain unexcluded as additional contributors. Defaults OFF/OLD retained;
  no production consequence. Next decision (user): fork-level trim/recurrent
  work, or accept MTP-exactness as bounded by margins.

---

## 2026-10-06 RECONCILE: superseded interpretations (raw data retained)

The three interpretations below are explicitly SUPERSEDED by sourced later
evidence. Raw observations, logs, and CSVs stand unchanged; only the
conclusions drawn from them are corrected. Nothing here changes the exact
parity gate (still RED) or the defaults (OFF/OLD retained).

### 1. Rev69 did NOT rule out all batching numerics

Superseded text: "Batching arithmetic (all ops, incl. SSM/recurrent under
these flags) is BIT-EXACT" and "the live MTP-vs-sequential divergence is NOT
batching numerics" (§ rev68/69 above).
What rev69 actually compared: singles vs ONE accepted 3-wide batch (same
width class), maxabs=0. Rev73 then compared 4-wide vs 3-wide batch rows from
the identical prefix and found maxabs=0.0506/0.0556/0.0505 pre-trim (bit-
identical across reps; A-vs-C still 0). So width-4-vs-3 batch arithmetic
differs; rev69's accepted-batch-vs-singles result cannot generalize to it.
Retraction scope: the generalization only. Rev69's own rows (A==B, 279,
margin 0.002611) remain valid measurements.

### 2. Rev72 did NOT isolate trim rather than batching

Superseded text: "proves the live reject+trim path (not batching arithmetic,
not kernels, not flags)" and the B-vs-C framing (§ rev72 above).
Rev73 showed the 4-wide rows already differ BEFORE any trim call, so B-vs-C
(0.0202) conflates width4 + trim. "Not batching arithmetic" is retracted.
Raw rev72 numbers (A-vs-C 0; B-vs-C 0.0202318, argmax preserved) stand.

### 3. Bounded fork diagnosis/fix is NOT outside scope

Superseded text: "neither fix is in scope (both live in fork batch/trim/
recurrent machinery)" and the "fork-level trim/recurrent work, or accept"
either/or (§ rev72 above, "O que falta" § "Dependente de harness").
Authorized and executed instead: read-only fork audit (TRACE-RESTORATION:
seq_rm/rs_idx contract, snapshot slot indexing, D3D12 op placement),
narrow cb_eval instrumentation, and reversible opt-in patch mandate with
patches/ maintenance route for fork code. No whole-fork rewrite was done or
proposed; plan001's scope line (no whole-fork rebases) still holds.

### Sourced summary of what replaced them

- Rev73 pre-trim control (actual idx=23 block, rev73 `F3731205`): B 4-wide
  rows 0..2 vs C 3-wide rows differ 0.0506/0.0556/0.0505 pre-trim, match=1,
  bit-identical reps; A-vs-C maxabs=0. Batching is NOT bit-exact across
  widths. Trim alone is not isolated by this (or any) B-vs-C.
- Rev74 captured-path reproduction (rev74 `789D2F44`): B through the
  identical remainder (B2 pair, single 567, B3 triple) gives output33=1204
  (25.7421, margin 0.010162) = the live feedcap TRACE row to 4 decimals;
  A/C give 279. Sufficient captured path to reproduce, stated without any
  low-level root-cause claim.
- Rev76 D/E (rev76 `60DEFFB1`): D (width4, no trim) gives drem5=279 with
  margin collapsed to 0.000484 (near-flip, no reproduction). E is INVALID
  as a trim-only control: E's single-11 decode rewrites only snapshot slot
  0 ("slot s = s tokens back"; single-token decodes leave older slots
  caller-owned), so post-trim restore reads stale slot 1 (through-143,
  missing token 144). E tested trim + stale restore; its "safe" outcome
  must not be read as trim-alone evidence. What stands: B flips, D
  near-flips; trim's solo effect is unknown from E. (B-vs-D is NOT a pure
  trim contrast either: the 4th token differs in-batch (11 vs 2261) and the
  correction batching differs (trim+single-corrective vs in-batch 4th row).)
- Rev83 first known forward mismatch (rev83 `6B95AE56`,
  `bench/results/diverge-20261006/rev82-diverge-result.csv` predecessor run):
  z-0 MUL_MAT graph idx62 `[4096,3,1,1]` maxabs=4.76837e-06; every earlier
  captured compute tensor matches, including QKV MUL_MAT idx13
  (`[8192,3,1,1]`, exact 0). z-0's inputs are verified equal (all captured
  predecessors match; uncaptured nodes are static by the token-axis rule;
  weights are the same resident tensors). Backend attribution (D3D12 N-tiled
  vs CPU threaded matmul paths) is PRELIMINARY, not a root-cause claim:
  only one refused shape ever logged (f32 2560x32; log capped at 8) and no
  per-op backend tags exist in the capture.
- Snapshot VIEW59 (`' (view)'` `[524288,1,3,1]`, graph idx59) compares
  DIFFERENT logical positions in all pre-tier exports (B3 slots 0..2 = pos
  144/143/142 vs B4 slots 0..2 = pos 145/144/143) and is NOT a causal first
  mismatch. Same-position (B3[0..2] vs B4[1..3]) comparison belongs to the
  tiered ranking, not to the old extent-only exports.

### Instrumentation failures and export corrections (harness, not product)

Recorded separately from the numerical findings above; none of these touch
production paths, gates, defaults, or the model:

- rev77 `9CC68F12`: diverge run failed `decode: n_tokens == 0` — harness bug
  (single-decode batch missing `b.n_tokens = 1`), one-line fix, rebuilt.
- rev78 `C0FAFD77`: silent app exit mid-capture, no dump — OOM: per-tensor
  16 MB cap with full raw retention peaked ~1 GB over the resident model.
  Fixed by slice-and-free inline capture + 512 MB global seatbelt +
  per-run progress boundaries. Hypothesis confirmed by rev80 progress
  accounting (518 MB retained at B3-on), not by assumption.
- rev80/81: full capture valid (paired=1354, first_diff=30, both reps) but
  lost at UWP pairing (split-block assumption vs interleaved device
  layout). Fixed by extracting pairing/export into the shared pure helper
  with host fixtures (sparse indices, asymmetric skips, odd counts).
  Pairing/export repairs are harness root causes, never model root causes.
- rev82 `2546CF58`: UWP "shape-drift" break at attention-score-shaped
  `[H,T,T]` tensors (two token axes vs the exactly-one rule). Fixed by the
  multi-axis mask rule (causal 3x3 block), RFC-4180 shape quoting, and
  drift-identity logging. rev82 CSV (156 pairs) is INVALID FOR RANKING
  (snapshot positional mismatches) and preserved as diagnostic-only.
- Min-rule verdict (match/shifted-clear by value) retired before use as
  evidence: alignment now comes from provenance tiers (Tier-1 per-token
  compute ranked unshifted-only; Tier-2 state/uncertain ops diagnostic-only
  with both alignments), with host anti-min-rule regressions.

### Artifact links (all hash-verified at fetch time)

- `bench/results/diverge-20261006/`: rev75/rev76 replay CSVs,
  rev80-diverge-result.csv, rev82-diverge-result.csv +
  rev82-diverge-log-slice.txt, xllama-rev80/81/82-full.log,
  NOTE-diverge-20261006.md (E retraction, harness-vs-model distinction).
- MSIX (fixed): rev72 `F3731205…`, rev73 `8C43E69D…`, rev74 `789D2F44…`,
  rev75 `EECD0B5A…`, rev76 `60DEFFB1…`, rev77 `9CC68F12…`, rev78 `C0FAFD77…`,
  rev79 `5D9C5F70…`, rev80 `C472BADD…`, rev81 `A7AA567B…`, rev82 `2546CF58…`,
  rev83 `6B95AE56…`. Rev72–83 backups retained on the build host.
- Host gates: 322/322 cases, 11608→11793 assertions across the period;
  clang-format 22.1.5, shellcheck, check-coherence.py clean at each step.

No unsupported all-backend or complete root-cause claim is made anywhere in
this entry. Next: bounded operation/dispatch diagnosis from the tiered
ranking, then an evidence-backed fix proposal if one meets the minimal,
reversible, reference-preserving bar.

## 2026-10-06 rev85 NARROW-TO-Z0: concrete operation path (preliminary)

First focused (non-broad) capture, rev85 `6B6A52EB`:
`bench/results/diverge-20261006/rev85-znarrow-result.csv` (10 rows) +
`rev85-znarrow-log-slice.txt`. Only z-0 + norm-0/attn_norm-0 captured
(name-filtered, src identities from tensor metadata only); contamination
control (callback off-vs-on) passed bit-exact; both reps bit-identical.

- z-0 src identities (decisive, not inferred):
  `z=z-0[f32 4096x3x1x1]` / `[4096x4x1x1]`,
  `src0=blk.0.attn_gate.weight[q4_K 2560x4096x1x1]` (static resident),
  `src1=attn_norm-0[f32 2560xNx1x1]` (bit-identical both widths, maxabs 0,
  reconfirmed in-run). Matches `build_qkvz` in
  `llama.cpp/src/models/qwen35.cpp:306`: z shares its input tensor with the
  exactly-matching QKV matmul — the ONLY remaining variable is the z matmul
  kernel itself. Output differs 4.76837e-06 deterministically, both reps.
- Placement: all target layers assigned D3D12 in these runs (33/33
  offloaded); Q4_K is a supported D3D12 weight type; the now dedup-complete
  refusal log contains no q4_K 4096x2560 (only f32 2560x32, q8_0 5120x2560,
  q6_K 2560x248320 shapes) → z-0 ran the D3D12 Q4_K mmv (t64, 64 threads),
  same as QKV. Not inferred from layer assignment alone.
- Eliminated as loci (read, not assumed): CPU vec_dot parity branch is a
  no-op on x86 (`nrows=1` for Q4_K except ARM mmla); CPU repack traits need
  N%8==0 (3,4 fail); IQ panel GEMM needs N>=8 (fails);
  TWO_COL variant at 0 dispatches in all runs.
- QKV `[8192,3,1,1]` exact + z `[4096,3,1,1]` varied, same backend/shader
  family/K: variation is (M,K,N)-path-specific, data- and dispatch-
  dependent — recorded as observed, without a generic FP-variance claim.
- PRECISE BLOCKER (honest): the OLD-path HLSL + dispatch read as
  per-column N-independent, and the CPU fallbacks read N-invariant, yet
  values differ — so per-op actual-execution facts (which device really ran
  z-0 B3/B4) or an unread shader interaction is still missing. Next minimal
  instrument: placement-forcing control (e.g. env-gated D3D12 Q4_K refusal)
  or per-node backend tags; then, if placed, the shader-level N-interaction.
  No control built yet; no parity/default change.

## 2026-10-06 rev91-94: repack-path control — replay boundary and product parity

Cause chain established with positive executed-path evidence (not buffer-name
inference):

- **Executed path (rev90)**: `zsrc ... buf=D3D12_Host/CPU_REPACK/D3D12_Host` via
  public ggml getters — z-0's weight sits in **CPU_REPACK** (the D3D12_Host
  activations are shared-buffer placement, not GPU-execution proof). The
  repack specialization for Q4_K + AVX2 is `q4_K_8x8_q8_K` (weight
  `ne[1]=4096 % 8 == 0`), and `ggml-cpu/repack.cpp` dispatches `gemm` when
  `nrows > 3`, else the scalar `gemv` tail, with four-row interleaved
  activation quantization (`ggml_quantize_mat_t`) feeding the blocked path and
  scalar `from_float` feeding the tail — the source-grounded width-dependent
  mechanism (B3 = all-scalar sequence, B4 = blocked quant + blocked GEMM).
- **Fix candidate (patches/0005, maintained as an xllama patch, applied by
  `scripts/apply-uwp-patches.sh`, no submodule commit)**: one shared predicate
  gates BOTH activation quantization and compute for the forced path —
  scalar-row quantization + scalar gemv tail for ALL rows, which at small
  widths is exactly the already-running B3 sequence (never blocked data under
  a scalar reader). Strict env select: absent/empty/0/other = off (baseline),
  `GGML_REPACK_FORCE_GEMV=1` = exact z-0 gate weight only, `=2` = every repack
  matmul at nrows ≤ 8 (measured probe). Race-free atomic-capped engagement
  log; small-batch-only so prefill/chunks and staged prefixes are untouched;
  knobs wired through `cpurepackforcegemv.txt` ("1"/"2") in `run_replay`,
  `run_znarrow`, and `main_loop` (per-process env, fork reads once at first
  use; delete the file to return to baseline on next launch).
- **rev91 (scope 1, znarrow)**: 8 engagement lines, each
  `repack z0 M=4096 K=2560 nrows=3/4 quant=scalar-all compute=gemv-all forced=1`;
  z-0 B3-vs-B4 **bit-exact** (was 4.76837e-06), downstream layer-0
  `norm-0` restored (`match=1`), 10-row CSV both reps. Local difference at
  z-0 removed by the combined path.
- **rev92 (scope 1, A/B/C remainder replay)**: 92-row CSV, `fail=-`,
  bit-identical reps. With control ON: `tailc5 = 279` (baseline **1204**) —
  the captured reject-path output33 flip is restored, decision margins
  healthy (seq 0.002611 / tail 0.015343); `drem5` margin recovered
  0.000484 → 0.008408. OFF reference on the same build still yields
  `tailc5=1204` (margin 0.010162, exact baseline): the control is strictly
  opt-in and the default path is unchanged.
- **rev93 (scope 1, product prompts)**: both originally failing prompts,
  full greedy IDs, fresh 3-arm runs (`repackctl-20261006-*`):
  spec-chat-open 64 EXACT vs sequential; standard-512 64 and 256 EXACT
  (baseline flips at token 39 confirmed on old-off); spec-chat-open 256
  restored through the old token-33 flip but diverged at a **new boundary
  (157: 5979 vs 12391)** — same-build determinism r2==r3 both arms.
- **rev94 (scope 2, product prompts)**: all four cells
  (`scope2-20261006-*`) **EXACT** vs fresh sequential baselines at 64/256,
  deterministic; 192 engagement lines across diverse weights
  (M=1024/2560/4096/8192/9216, K=2560/4096), scope field `2`; baseline
  cells logged `control off (baseline)`. The 157 divergence was the same
  `nrows>3` repack mechanism at other weights, measured by scope escalation,
  not assumed.
- **Scope of the claim**: the combined scalar-quant + gemv control (both
  scopes) restores exact MTP-vs-sequential IDs on the replay boundary and
  both formerly-failing product prompts at 64/256 on this fixture set. It
  does NOT separate quantization from accumulation causality (they are
  forced together by design), does NOT speak for widths > 8 (prefill keeps
  the blocked path), and is NOT a default or a performance claim. Defaults
  remain OFF/OLD; the exact-parity gate on defaults remains RED; no
  promotion/push/merge. Sequential (width-1) decode is provably unaffected
  by the control (N≤3 already degenerates to the scalar sequence), so
  sequential references remain valid.
- **Artifacts**: `patches/0005-cpu-repack-force-gemv-control.patch` +
  `patches/README.md` row (host apply/build/327-test verified each rev);
  `bench/results/repackctl-20261006-*`, `bench/results/scope2-20261006-*`,
  `bench/results/diverge-20261006/` (rev91 CSV, rev92 ON/OFF replay CSVs,
  rev93/94 dumps, `xllama-rev94-scope2-full.log`); MSIX rev91-94 hashes
  recorded in build logs with rev72-93 backups on the build host.

## 2026-10-06 CHECKPOINT (Xbox paused by owner — daughter playing, console out of Dev Mode)

Local bookkeeping only; zero Xbox network calls after the pause notice.

**Completed and saved (verified):**

- rev91-94 repack-control chain: z-0 executed path CPU_REPACK (rev90 getters);
  patches/0005 scope 1/2; replay output33 restored (tailc5 1204→279, OFF ref
  still 1204 on same build); product full-ID parity scope1 3/4 cells,
  scope2 **4/4 cells EXACT** (spec-chat-open, standard-512 × 64/256,
  both measured reps, deterministic).
- Artifacts: `bench/results/repackctl-20261006-*`, `bench/results/scope2-20261006-*`,
  `bench/results/diverge-20261006/` (64 files), rev92-94 MSIX hashes in build
  logs; rev72-93 MSIX backups on build host; device LocalState back to
  baseline (knob deleted, only api.flag) as of the last completed run.
- Host gates green at every rev: 327/327 tests, clang-format, coherence,
  patch re-check clean against submodule pin 982eaadaa (no commit).

**Interrupted / never started:**

- Stage 1 `spec-code-edit` 64/256 off-vs-scope2: driver staged at
  `/tmp/xllama-rev94/stage1-codeedit.sh`, **not executed** (blocked at the
  pre-run reachability probe: 192.168.1.26 unreachable, ping 100% loss,
  ARP INCOMPLETE → consistent with console off/out of Dev Mode; owner pause
  then canceled the turn).

**Pending (in order):**

1. Stage 1: spec-code-edit 64/256 IDs (script ready).
2. Session gate: four scenarios, scope2 + `--mtp-pmin 75`, proposals>0,
   lookup0, full IDs (script options verified: `--mtp-session`,
   validator requires `-session -mtp4 -pmin75 -g28` host tags).
3. OLD-vs-AUTO scope2: exact IDs + real 2col dispatch counters.
4. Cost pilot: randomized/alternating paired 3-arm, order/seed/package/GGUF/
   knobs recorded, profile ON/OFF genuinely applied, TTFT/decode/whole-call/
   memory separated, enough pairs (not 2-rep); ≥10% still not claimed.

- EOG/stop/cancel/resume gates remain explicitly INCOMPLETE (existing EOG-on
  runs never observed EOS).

**Exact next step on owner's explicit resume confirmation:** re-verify
reachability (`ping 192.168.1.26`, `deploy.sh pfn`), confirm PFN still
1.6.0.94, then run `/tmp/xllama-rev94/stage1-codeedit.sh`. No .193 build is
running (rev94 finished 17:30, exit=0). Defaults OFF/OLD, parity gate RED on
defaults; AGY offline-validator output remains ignored (not approved).

## 2026-10-06 resume: Stage 1-4 gates executed (rev94 reuse, no rebuild)

All runs on the installed rev94 package (`e1c7d81d…`), greedy, seed 1,
GPU28, GGUF `f3559ed1…`; device returned to baseline after each stage
(control knob deleted; LocalState ends with only `api.flag`).

**Stage 1 — third prompt (`scope2-20261006-spec-code-edit-*`):** 64 and 256
EXACT vs sequential, r2==r3 deterministic. With Stage "two-prompt" results
this is **6/6 prompt×length cells EXACT full-ID parity** for MTP4+scope2
across all three protocol prompts.

**Stage 2 — Session continuity/reset gate (`--mtp-session --mtp 4
--mtp-pmin 75`, run slice `session-scope2.log.slice.log`):** 4/4 scenarios
OK — reset_prompt_swap (mtp_draft 4, rounds 1), edited_prefix (7/3),
delta_continuation (12/4), multi_chunk (9/3), all lookup=0, parity 12/12
each, host-tag checks `-session -mtp4 -pmin75 -g28` enforced by the
validator; `MTP_SESSION_CONFIG … p_min=0.75 twocol=0`; 16
`repack ctrl scope=2 … forced=1` engagement lines inside the slice (B=5
verify batches).

**Stage 3 — OLD vs AUTO under scope2 (`…-auto-on2` runs):** full IDs EXACT
(old↔auto↔off, 64 and 256, deterministic). Real NEW dispatch: last auto-run
backend free-log `29376 matmuls (5650 2col)`; off-arm runs `(0 2col)`; knob
line `d3d12twocol.txt='auto' -> kernel variant 2 (prev 0)`.

**Stage 4 — bounded paired 3-arm cost pilot**
(`costpilot-20261006-b{1..6}-{a,b,c}.csv`, driver order printed and kept:
`b1:c b1:a b1:b b2:a b2:b b2:c b3:c b3:a b3:b b4:b b4:a b4:c b5:b b5:a
b5:c b6:b b6:c b6:a`; order seed **20261006**; arms a=MTP OLD control-off,
b=MTP OLD scope2, c=MTP AUTO scope2; `--profile-phases 0`, host tag
`xbox-series-s-mtp4-greedy-s1-g28-prof0` on all 18 runs; `--runs 2` =
warmup + 1 recorded; run_index 2 rows used):

| comparison (decode tok/s, n=6 paired blocks) | median    | range                      |
| -------------------------------------------- | --------- | -------------------------- |
| b−a = control (scope2) cost                  | **−3.1%** | −3.6 … −0.8 (6/6 negative) |
| c−b = two-col AUTO under control             | **+0.8%** | −0.1 … +2.4                |
| c−a = candidate vs baseline MTP              | **−1.9%** | −3.3 … −0.2                |

- TTFT paired delta (b−a) median **+14 ms** on ≈2.17 s (noise-scale);
  peak_ws 4320→4295 MB (−25 MB); gpu_mem constant 2273 MB; whole-call
  (ttft + n_gen/decode) mirrors decode (~+1.6% for b vs a).
- **Profile ON/OFF genuinely applied:** ON run host tag `prof1` + 263
  `PHASE … profile=1` lines; OFF run host tag `prof0` + zero new PHASE
  lines (380 == 380 across fetches) — instrumentation truly toggled
  (`profile-verify-20261006-{on,off}.csv`).
- **≥10% product gate: NOT met, not claimed** (no gate/default change).
  The −3.1% is the INCREMENTAL cost of the correctness workaround (scope2
  vs baseline MTP), NOT the dominant total cost — total decode time is
  dominated by target verify (reconciled attribution below: verify ≈ 54% of
  decode). One evidence-backed minimal offset already measured: AUTO
  two-col tile recovers ≈ +0.8% of the incremental penalty, which is why it
  belongs in any future candidate config. A width-invariant blocked path
  remains an option — NOT built, NOT claimed.
- **Explicitly still incomplete:** EOG/stop/cancel/resume gates — existing
  EOG-on runs never observed EOS; cancel has no runner; none exercised here.
- AGY offline validator: untouched, unapproved, ignored.

## 2026-10-06 rev97/98: termination gates — cancel, stop (later boundary), natural EOG

Headless `termgate.flag` gate (3 arms per run: seq=MTP0; ref=MTP4/OLD;
cand=MTP4+AUTO+scope2; knobs via bench_mtp.txt/d3d12twocol.txt/
cpurepackforcegemv.txt; GGUF `f3559ed1…`; greedy). Rev97 established the
corrected stop state-proof; rev98 added termination metadata (DecodeLoopResult

- InferenceResult: `eog_token/eog_branch/stop_branch/stop_round`, set at all
  5 EOG sites + all 5 stop sites + a loop round counter) and the later-boundary
  stop (`pick_boundary_spanning_stop(pieces, prefer_last=true)`).

**rev98 result — all 24 rows ok=1 across all three arms**
(`bench/results/termgate-20261006/{seq,ref,cand}.csv`):

| scenario            | verdict (all arms)                                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cancel@6, cancel@13 | ok=1, parity=1, first_diff=-1; ref/cand active=1, drafted 8/14, rounds 2/4, lookup=0                                                                       |
| stop@probe          | ok=1, reason=**eos**, n_eval=10, **eog_tok=248044** (branch: seq=classic, ref/cand=spec-reject — from result metadata)                                     |
| stop@run            | ok=1, parity=1, n_ids=10 (later boundary: span `start=9 stop_len=2`), drafted=8, rounds=2, **stop_branch=spec stop_round=4** (ref+cand) / classic/10 (seq) |
| eog@0..2            | ok=1, reason=cap, eog_tok=-1 — bounded non-EOG observations preserved                                                                                      |
| eog@3               | ok=1, reason=**eos**, n_eval=10, **eog_tok=248044**, branch classic (seq) / spec-reject (ref+cand)                                                         |

- **Stop during speculative verification (coverage demand met):** MTP arms
  fire the stop in the **spec branch at round 4 with drafted=8, rounds=2**
  preceding it; the seq arm records its classic path (round 10). No row
  claims coverage it does not have (`stop fired before any MTP round` guard).
- **Natural EOG (bounded fixture):** the count prompt emits EOG token
  **248044**; recorded from DecodeLoopResult→InferenceResult metadata at the
  sampling site — never inferred from length or the cap. `cap` probes remain
  distinct non-EOG observations.
- **Cancel landing:** ref/cand each show **2× `abort: mid-round trim`
  (i=3/5 @n=6, i=1/4 @n=13)** — cancellation during a speculative round
  with tail trim, both boundaries; seq shows 2× round-boundary (classic by
  construction). The cancelled turns' MTP_STATS show **corrective_ms=61.6
  (ref) / 61.0 (cand)** > 0 with rounds=2 — a verify-reject + corrective ran
  inside the cancelled turn (profile ON only for these diagnostic runs).
- **State proofs (not loosened):** resume==cold full IDs, resume-prefill ==
  cold-prefill (19 tokens), stop-run ids are an exact prefix of the probe
  ids, and cross-arm dumps (t1/resume/seqref/stop/eog) are byte-identical
  seq==ref==cand for every file. Documented contract for rev96→97: emit_token
  records the crossing token but accept_token is skipped for it, accepted
  text stripped out of the output is not prompt text, and the product
  resumes via a FULL-prompt (#170) text-derived turn — so the rev96
  `prompt+accepted` formula could not hold by construction; the exact
  expected prefix is `tokenize(prompt+stripped)` asserted byte-for-byte
  against the cold reference.
- Continuation after the later stop lands in a prompt whose stripped suffix
  ends mid-pattern: resume AND cold both EOG after 2 tokens (`tok=248044`)
  — equality holds from equal state, which is the point of the proof.
- Repairs since rev96 (harness, not product): concurrent-Session OOM fixed
  (one model load at a time), stop-resume switched to the product-realistic
  full-prompt turn, later stop boundary, metadata plumbing. Build rev98
  `3E4C02DE`; host 335/335 tests; defaults/gates unchanged; AGY untouched.
- Artifacts: `bench/results/termgate-20261006/` (3 CSVs, run segments,
  dumps-cand, full log).

## 2026-10-06 cost attribution (reconciled from existing delimited logs)

Sources (all already on disk; profile ON only where the ON arm was recorded;
timing claims come from profile-OFF runs — see §Stage 4):

- Per-generation phase table: rev98 termgate run segments
  (`bench/results/termgate-20261006/{ref,cand}-run-segment.log`), PHASE +
  `session generate` + MTP_STATS lines paired by (draft_ms, verify_ms),
  MTP-active turns only (drafted>0), n=7 turns × 2 arms, 63 tokens each.
- Product-scale timing: `costpilot-20261006-*` (18 runs, profile-OFF,
  chat64, paired blocks, seed 20261006).
- Width mix: `d3d12_shape … phase=verify … B=k mm=n` drains inside the same
  segments (D3D12 matmul-count composition; derived, NOT per-width timing).

**MTP-active turn medians (ref = MTP4/OLD/scope0; cand = MTP4/AUTO/scope2):**

| phase (ms/turn)                   | ref         | cand        | share of ref decode (517.2) |
| --------------------------------- | ----------- | ----------- | --------------------------- |
| target verify                     | **278.8**   | 294.6       | **54%**                     |
| draft (NextN compute)             | 39.4        | 38.2        | 7.6%                        |
| confidence softmax (topprob)      | 31.3        | 31.2        | 6.1%                        |
| draft catch-up                    | 21.4        | 20.9        | 4.1%                        |
| classic decode (incl. corrective) | 61.6        | 62.6        | 11.9%                       |
| corrective (subset of classic)    | 0 (sum 123) | 0 (sum 122) | —                           |
| sample target + sampling          | 17.5        | 16.8        | 3.4%                        |
| residual                          | 0.3         | 0.3         | 0.06%                       |

- accounted/decode = **0.997** (no unattributed lump).
- Useful-token economics (both arms identical): 63 tokens, 58 drafted,
  37 accepted (64%), 16 rounds → 3.94 tok/round; draft stack
  (draft+topprob+catchup ≈ 91 ms/turn ≈ **17%** of decode); verify
  35.5 ms/accepted-token, draft compute 7.5-8.0 ms/accepted token.
- Verify width mix (D3D12 matmul counts over the segment): **B2=2460,
  B3=1476, B4=1804, B5=5084** — the deepest batch (anchor+4 drafts) carries
  ≈47% of verify matmuls; B4+B5 = 64%.
- Product-scale whole-call (costpilot, profile OFF, chat64 medians):
  prefill/TTFT ≈ 2.17 s (≈36% of ≈5.95 s whole-call), decode ≈ 3.78 s;
  the 3-arm paired decode deltas remain b−a −3.1%, c−b +0.8%, c−a −1.9%
  (§Stage 4) — incremental penalties around a decode dominated by verify.
- Conclusion: the dominant TOTAL cost is **target verify (54%) with its
  width mix concentrated at B5/B4 (64% of matmuls at depth 4)**; the draft
  stack (17%) is second. The scope2 workaround penalty (−3.1%) is a small
  delta on top of that, not a component of it.

## 2026-10-06 draft-depth policy pilot (the one measured-bottleneck experiment)

Chosen from the attribution above: verify = 54% of decode with B4+B5 = 64%
of its matmuls, draft stack = 17% — depth drives both, and widths ≤ 3 are
already on the scalar repack path (no new width-variance risk). One
experiment only; package rev98 `3E4C02DE` (no rebuild).

- Design: spec-chat-open 64 tok, greedy seed 1, `--tokens`, `--runs 3`
  (warmup + 1 recorded), **`--profile-phases 0` (host tag `prof0` on every
  arm — timing genuinely unprofiled)**; candidate knobs held FIXED across
  all arms (`d3d12twocol=auto`, `cpurepackforcegemv=2`, recorded); arms =
  `--mtp 1|2|4`; order seed **20261006**, 6 randomized blocks; one fresh
  sequential reference (mtp0) first, same package/knobs.
- Recorded order (file mtimes): b1 d4,d1,d2 · b2 d1,d2,d4 · b3 d4,d1,d2 ·
  b4 d2,d1,d4 · b5 d2,d1,d4 · b6 d2,d4,d1.
- **ID parity: 18/18 runs (all depths × all blocks) == the fresh sequential
  reference (64/64 ids exact).**
- Decode (paired vs current d4, n=6 blocks):
  | depth | median tok/s | Δ vs d4 | range | rounds | drafted/accepted (per run) |
  | d1 | 16.80 | **+1.1%** | +0.5..+2.0 (6/6 positive) | 14 | 17/17 |
  | d2 | 16.66 | +0.2% | −0.9..+3.0 | 15 | 20/20 |
  | d4 | 16.62 | (base) | — | 17 | 22/21 |
- TTFT flat (2161-2173 ms, depth cannot touch prefill); whole-call moves
  with decode only (~+0.5% at d1).
- **Verdict: the ≥10% product gate is NOT met by depth reduction** — best
  case +1.1% (d1), consistently signed but an order of magnitude short. The
  result is a measured limit, not a proposal: shallower depth trades draft
  stack (~17% of decode) against lost acceptances/extra rounds and lands
  within noise of current depth 4 on this fixture.
- Next concrete bottleneck (from the attribution, now depth-eliminated):
  the per-round **target-verify fixed cost** (graph compute + recurrent/KV
  state ops for each verify round) — the 54% term that barely moved when
  both the width mix and draft depth were reduced. The next probe would
  split verify's round cost with the existing gpu_ms/wall_ms + shape timers
  (GPU graph vs CPU-side vs sync), a measurement task, not a rewrite.
  Bounded work stops here with the limit recorded.
- Evidence: `bench/results/depthpilot-20261006-*.csv` (19 CSVs + .r2 id
  dumps), `bench/results/depthpilot-20261006.log`. Device baseline knobs
  deleted after the run; host tags prove profile OFF; no defaults changed.

### Record correction (independent re-analysis of the same artifacts)

Two counting errors in the text above, corrected from the raw files (no
re-run):

- `--runs 3` produced **warmup + TWO measured rows (run2 and run3)**, not
  "warmup + 1 recorded". The full analysis pool is therefore **36/36
  measured dumps (2 runs × 3 depths × 6 blocks) vs the fresh sequential
  ref — all exact**.
- The table's Δ column used **run2 only** (labeled): d1 +1.1%, d2 +0.2%.
  The primary number is the **block-mean over both measured runs**: d1 vs d4
  median **+1.24%** [0.54, 2.26], d2 **+0.94%** [−0.06, 2.56]; r2-only
  stays recorded as the labeled subset (+1.14% / +0.18%). Both agree in
  sign and magnitude.

The verdict does not change (≥10% gate not met), with one precision added:
the pilot does **not** prove that a fixed per-round cost dominates —
acceptance/round counts and width work both changed with depth, so the
near-flat result is a bound on the net depth lever, not on any single
internal term.

## 2026-10-06 verify sub-cost split (rev99, existing-instrumentation gate)

Task `MEASURE-VERIFY-SUBCOSTS-157`: split verify's 54% share of decode
into GPU / submission-wait / CPU-sched, prove negligible instrumentation
overhead, then choose one minimal optimization or record a precise
ceiling/blocker.

- Instrumentation (rev99, host-only files, profile-gated, zero work when
  OFF): `d3d12_wall_ms()` accessor (header decl + non-Win32 stub + locked
  `g.wall_ms` read); per-round snapshots around
  `detail::decode_verify_batch` in `decode_loop.h`
  (`vr_width/vr_wall/vr_d3w/vr_d3g/vr_acc0`); `VROUND` log lines emitted at
  decode-loop end; `verify_d3w/verify_d3g/rounds_split` fields appended to
  the session-path PHASE printer. Nesting is enforced by construction:
  `gpu <= d3w <= wall <= verify_ms` — no value is counted twice.
  Host: clang-format clean, 335/335 tests, coherence green.
- Run: rev99 `449ED56C`, spec-chat-open 64 tok, greedy seed 1, candidate
  knobs auto+scope2, 5 cells: fresh sequential ref99, d4-OFF, d4-ON,
  d2-ON, d1-ON (profile ON/OFF pair on the same package).
- **Instrumentation exactness: sum(VROUND wall) == MTP_STATS verify_ms
  within 0.04 ms over all 6 profile-ON calls.**
- **Overhead: ON vs OFF (d4) decode +2.3 ms (+0.06%), TTFT −0.1 ms.**
- **IDs: 4 arms × 2 runs all byte-exact vs fresh ref99; d4-ON == d4-OFF
  both runs.** Profile does not perturb outputs.
- Split (92 verify rounds, 6 calls, pooled): **GPU 1458.8 ms = 17.4% of
  verify; submission-wait (d3w − d3g) 986.3 ms = 11.8%; CPU+sched (wall −
  d3w) 5927.0 ms = 70.8%.** Per-width medians (real decode rounds, not the
  synthetic selftest — the selftest does not exercise target decode/KV/
  DeltaNet and is invalid for B1/2/3/5 cost):

  | width | rounds |   wall | GPU (d3g) | sub+wait | cpu+sched | accepted (med) |
  | ----: | -----: | -----: | --------: | -------: | --------: | -------------: |
  |     2 |     72 |  81.42 |     13.83 |     8.56 |     59.03 |              1 |
  |     3 |     14 | 105.00 |     20.87 |     8.88 |     75.25 |              2 |
  |     4 |      6 | 131.26 |     28.48 |     9.23 |     93.55 |              3 |

  Widths present in real bench rounds: 2/3/4 (accepted = width − 1, i.e.
  every draft accepted); B5 did not occur on this fixture, and B1 =
  classic decode (outside the verify bracket). Cost is ~linear in width
  at +~25 ms per token of width; the per-round intercept (fixed
  recurrent/KV/state part) is large and mostly cpu+sched.

- Translation to shares of decode (verify = 54% of decode, plan004 cost
  attribution): **GPU ≈ 9.4% of decode; sub+wait ≈ 6.4%; cpu+sched ≈
  38.2%.**
- **Ceiling/blocker (corrected below; denominators made explicit):**
  - Definitions. A **time reduction** is `r = (T_before − T_after) /
T_before` on a stated denominator (decode time, or whole-call wall
    time). A **throughput gain** for the same token count is `g =
T_before/T_after − 1 = 1/(1 − r) − 1`. Throughput gain is always
    larger than the matching time reduction (`r = 10% ⇒ g ≈ 11.1%`), and it
    only applies to the time that carries the throughput metric: decode
    tok/s is `tokens / decode_time`, so decode-time reductions map to
    decode-throughput gains; whole-call wall time includes prefill.
  - GPU-only, decode-throughput denominator: eliminating ALL GPU verify
    time is an upper bound of `r = 9.4%` of decode ⇒ `g = 1/(1 − 0.094) − 1
    = **10.38% ≥ 10% — GPU-only is NOT mathematically excluded** by a ≥10%
    throughput goal. It is practically marginal: real optimizations recover
    only a fraction of that time, work moved off the GPU still costs
    somewhere, and the upper bound assumes zero GPU verify time.
  - Same elimination on **whole-call wall time** (prefill ≈ 36%, decode ≈
    64%, cost-attribution section): `r_call ≈ 0.094 × 0.64 ≈ 6.0%` ⇒
    `g_call ≈ 6.4% < 10%`. Sub+wait alone on decode: `r = 6.4%` ⇒ `g ≈
6.8%`.
  - cpu+sched (≈ 38.2% of decode; upper bound `g ≈ 62%` if fully removed)
    has by far the largest headroom — but it is an unsplit lump: ggml-CPU
    graph time + llama scheduler + KV/prep + cross-backend sync. Existing
    instrumentation separates none of it (no CPU-backend timer exists;
    adding one would edit the pinned llama.cpp submodule, which stays at
    exactly its 3 pre-existing foreign edits). The named lifting
    measurement is the host-side **process-CPU delta inside the verify
    bracket** (rev100, authorized next step): it re-expresses how much of
    `wall − d3w` is CPU burning vs time spent blocked. Its limit is part of
    the record: process CPU aggregates ALL threads, may legitimately exceed
    wall time under parallelism, and a CPU delta alone cannot separate
    scheduler work from CPU kernels nor directly attribute waiting (blocked
    time is simply not counted) — it narrows the lump, it does not fully
    resolve it.
- Evidence: `bench/results/vsplit-20261006-{ref99,d4off,d4on,d2on,d1on}.csv`
  plus `.run1/.run2.tokens` dumps,
  `/tmp/xllama-rev99/xllama-rev99-vsplit.log` (VROUND + MTP_STATS +
  backend-free lines), rev99 build log `exit=0`. Device baseline knobs
  deleted after the run; no defaults changed.

## 2026-10-07 rev100: process-CPU split inside the verify bracket (task MEASURE-CPU-VERIFY-157)

Builds on the rev99 split; package rev100 `DA307A06`. Host-side only: a
`process_cpu_ms()` helper in `platform.h`/`platform.cpp` (Windows
`GetProcessTimes` kernel+user for all threads; Linux
`CLOCK_PROCESS_CPUTIME_ID`) snapshotted around
`detail::decode_verify_batch` into `vr_cpu`, emitted per round as
`VROUND ... cpu=` and summed as PHASE `verify_cpu`. The pinned llama.cpp
submodule is untouched (still exactly its 3 pre-existing foreign edits).
Host gate: 337/337 tests (2 new `test_platform.cpp` cases; docs map updated
to 337/11875), clang-format clean, coherence green, shellcheck clean.
`check-win-syntax.sh` not runnable on this host (no clang-cl/splat) — the
.193 MSVC build is the Windows compile gate.

### Metric contract (limits are part of the record)

Process CPU aggregates **all** threads of the process; under parallelism a
delta may legitimately exceed the wall time of the enclosing phase (it did,
in 92/92 rounds); blocked/waiting time is simply not counted; and a CPU
delta alone **cannot separate scheduler work from CPU kernels, nor
directly attribute waiting**. Therefore every process-CPU number below is
an aggregate, not an attribution.

### Results (same 5 cells as rev99: fresh sequential ref, d4-OFF/ON, d2-ON,

d1-ON; spec-chat-open 64 tok, greedy seed 1, knobs auto+scope2)

- Instrumentation exactness: `sum(VROUND wall)` == `MTP_STATS verify_ms`
  within **0.03 ms** over all 6 profile-ON calls (92 rounds).
- Overhead ON vs OFF (d4): decode **+13.1 ms (+0.34%)**, negligible vs the
  ≥10% target.
- IDs: **all full-sha256 dumps exact** — every r100 arm (ON/OFF × d1/d2/d4)
  vs the fresh same-package sequential reference, ON == OFF, and (cross-
  revision) r100 ref == rev99 ref byte-identical: 48 dumps in one bucket
  per run index (14/24/10 at run1/2/3), zero distinct digests.
- Per-round split, pooled over 92 rounds (widths **observed** on real
  decode rounds: 2 (n=72), 3 (n=14), 4 (n=6); B1 = classic decode outside
  the verify bracket; **B5 never occurred on this fixture and is not
  inferred; the synthetic selftest remains invalid for these costs**):

  | width | rounds | wall med | GPU (d3g) | sub+wait | wall−d3w | process-CPU med |
  | ----: | -----: | -------: | --------: | -------: | -------: | --------------: |
  |     2 |     72 |    81.38 |     13.84 |     8.59 |    58.96 |          468.75 |
  |     3 |     14 |   105.37 |     20.91 |     8.88 |    75.58 |          656.25 |
  |     4 |      6 |   131.50 |     28.56 |     9.02 |    93.92 |          843.75 |

  Pooled: wall 8342.2, d3w 2401.6, d3g 1459.6, process-CPU **50187.5 ms =
  601.6% of verify wall** (≈6 cores of CPU burned per verify round on
  average; 92/92 rounds process-CPU > wall, as the contract predicts).
  Shares of verify wall unchanged from rev99: GPU 17.5%, sub+wait 11.3%,
  wall−d3w 71.2% (≈ 38% of decode at verify = 54%).

### Attribution precision (what is and is not claimed)

- **Confirmed spin, by code inspection, is exactly one loop:**
  `QueueFence::signal_and_wait(..., spin=true)` in
  `src/bridge/d3d12_compute.cpp` polls `GetCompletedValue()` with
  `YieldProcessor()` in an unbounded loop — used by `run_now` for **every**
  backend graph compute and buffer upload (`ggml_d3d12.cpp:718`), and its
  own header comment says it "burns one core". That is our code; the wait
  is inside `d3w`. Its quantitative share of the measured process-CPU is
  bounded (≤ 1 thread × d3w per round) but **not separately measured**.
- The remaining ≈5 cores of process-CPU are **unattributed until measured**:
  candidates include the ggml spin-wait threadpool (documented in
  `platform.h`, inside the pinned submodule), CPU graph splits, driver
  threads, and serial CPU kernels. Flat or non-flat wall curves cannot
  label them spin; no such label is claimed.
- **Process-CPU reduction is not a ≥10% product-throughput gain.** CPU
  seconds saved (power/thermal headroom) convert to throughput only if
  contention or clocks were limiting, which is unproven here.

### Thread-count paired repeats (profile OFF, before any thread choice)

Single-row sweep cells (n=1 measured row each) had suggested decode wall
flat between t3 and t6; that reading did not survive repetition. Paired
interleaved repeats (p1/p2 passes, order t6→t3 each, `--runs 3` → 2
measured rows per cell, same fixture/knobs, profile OFF):

| cell                  | decode rows (ms)       | decode med | TTFT rows (ms)         | TTFT med |
| --------------------- | ---------------------- | ---------- | ---------------------- | -------- |
| t6 (4 rows, 2 passes) | 3919, 3853, 3789, 3842 | 3847.3     | 2152, 2155, 2142, 2149 | 2150.2   |
| t3 (4 rows, 2 passes) | 3934, 3946, 3938, 3926 | 3936.0     | 2332, 2338, 2340, 2334 | 2336.1   |

- **t3 vs t6: decode +2.31% (t3 slower; both passes same sign: +1.38%,
  +3.07%), TTFT +8.64% (+8.44%, +8.95%), whole-call proxy (TTFT + decode)
  +4.58%.** IDs exact on all repeat dumps vs the same-package reference.
- Verdict: **no thread-count reduction is supported**; t6 dominates t3 on
  both phases. Defaults unchanged (`detect_threads_llama` cap stays 6 on
  UWP). The single-row "flat 3→6" observation is recorded as what it was —
  an n=1 artifact — not as evidence of spin waste.

### Next action in this task

Wait policy (the confirmed spin loop) is the remaining narrow reversible
host-side experiment: make the bounded-spin behavior **knob-gated with the
default exactly today's behavior** (`d3d12spinwait.txt` absent → infinite
spin as now), compare paired profile-OFF repeats (decode + TTFT +
whole-call + full IDs vs a fresh same-package reference), and only with
evidence consider anything further. Defaults stay unchanged throughout;
promotion requires the standing gate: repeated paired product gain ≥10%
with parity.

## 2026-10-07 rev101: bounded-spin wait policy (knob-gated, default unchanged)

Code (all host-side ours, submodule untouched at its 3 foreign edits):
`QueueFence::signal_and_wait(queue, spin, spin_us = -1)` — `-1` (default,
`d3d12spinwait.txt` absent) is the historical unbounded spin, bit-for-bit;
`N >= 0` caps the `GetCompletedValue` poll at N µs then takes the same
event wait. Knob applied via `apply_spinwait_knob` at the four
`apply_repack_ctrl_knob` sites (main_loop/replay/znarrow/termgate) with a
logged state line. Also fixed the plan003 zdispatch format bug found by
rev100's MSVC warnings (`wbuf=weights` literal was missing `%s`, shifting
every later arg — log-only, no numerics): warnings gone in the rev101
build. Host: 338/338 tests (new spin-knob round-trip case), format/coherence
green; `.193` build `exit=0`, sha `DB2E3F5C`.

**Knob engagement (log-verified):** 6 `spin_wait_us=500 (bounded spin)`
lines for the two bound cells × runs, 9 `...-1 (default unbounded spin)`
lines for ref + the two default cells — exact cell accounting.

**Paired profile-OFF repeats** (interleaved p1: def→bound, p2: bound→def,
`--runs 3` → 2 measured rows/cell, fresh same-package `ref101` first):

| metric                | default (def) | bounded 500 µs (b500) |                                                                 delta |
| --------------------- | ------------: | --------------------: | --------------------------------------------------------------------: |
| decode med (ms)       |        3864.8 |                3865.9 | **+0.03%** (neutral; pass deltas −0.45% / +0.77%, sign flips = noise) |
| TTFT med (ms)         |        2160.1 |                2194.0 |                   **+1.57%** (both passes same sign: +1.01% / +2.62%) |
| whole-call proxy (ms) |        6024.9 |                6059.9 |                                                            **+0.58%** |

**Profile-ON pair (CPU effect, same package/fixture):** process-CPU per
verify round 583.1 → 570.3 ms (−2.2%, −12.8 ms/round) with verify wall
−0.01%. **Label: unresolved, not real.** That is ONE pair, the delta
(12.8 ms) is at the Windows clock-tick scale (15.6 ms), and there are no
repeated confidence intervals — a −2.2% aggregate CPU move from a single
tick-scale pair cannot be called a real effect either way. The expected
larger saving (main thread spinning inside `d3w ≈ 28 ms`/call) does not
appear at full size; consistent with the contract that process-CPU
aggregates all threads and the ≈5.8-core background burn dominates the
total, but equally consistent with tick noise.

**IDs:** all 55 vsplit fixture dumps (rev99 + rev100 + rev101, every arm:
ON/OFF, d1/d2/d4, threads 1/2/3/6, spin default/bounded) are one full
sha256 per run index (23/23/9 at run1/2/3) — zero distinct digests,
cross-revision included.

**Verdict: bounded spin is not promoted.** Decode neutral, TTFT and
whole-call slightly worse (event-wake latency on prefill's many short
backend ops), process-CPU saving marginal. This is not a ≥10% product
gain — it is a CPU-seconds trade at best. The default (unbounded spin)
stays exactly as it was; the knob remains as a reversible experiment
device (file absent = today's behavior). Thread-count reduction remains
rejected (paired repeats above).

### Blocker and lowest-risk next measurement (end of this task)

- Both narrow our-code levers now measured: thread cap 6→3 costs
  +4.58% whole-call; bounded spin costs +0.58% whole-call. Neither yields a
  throughput gain, and process-CPU reduction (unresolved: one tick-scale
  pair, no repeats) is explicitly **not** the ≥10% product-throughput
  goal — a real CPU saving, if any, would still need to be shown to move
  product throughput at all.
- The dominant unexplained term stands: wall−d3w = 71% of verify ≈ 38% of
  decode, plus ≈5.8 cores of unattributed process CPU (confirmed spin is
  only the inspected `signal_and_wait` loop; everything else stays
  unattributed). Splitting it further requires either a CPU-kernel/scheduler
  timer inside the pinned llama.cpp submodule (excluded by policy: exactly
  3 foreign edits preserved) or an Xbox-side profiler (not available in
  this workflow).
- **Lowest-risk remaining measurement** (if attribution is pursued):
  snapshot the decode thread's own CPU time (`GetThreadTimes(GetCurrentThread())`)
  beside `process_cpu_ms()` inside the same profile-gated verify bracket —
  host-side, one vector, no behavior change. It separates "main thread
  burning" from "other threads burning" and bounds the inspected spin loop
  against the residual. Its limit: it still cannot split scheduler work
  from CPU kernels _inside_ the main-thread window — the submodule-internal
  split would remain blocked. Any optimization past that needs the standing
  gate: repeated paired product gain ≥10% with parity, defaults unchanged
  until then.

## 2026-10-07 read-only survey of host-owned paths and the pinned submodule (no writes)

Goal: attribute the wall−d3w lump (71% of verify ≈ 38% of decode) using
**existing supported instrumentation first**, deciding before any build
what each measurement distinguishes.

### What the code says (read-only)

- Decode prep is `llama_context::process_ubatch`
  (`llama.cpp/src/llama-context.cpp:3010`): reuse branch
  (`!graph_reuse_disable && res->can_reuse(gparams)` → only
  `set_inputs`) vs rebuild branch (`res->reset(); sched_reset;
model.build_graph; ggml_backend_sched_alloc_graph; set_inputs`).
  Two **commented-out** timing hooks already exist at lines 3035/3059
  (`graph build time`, `graph set inputs time`) — disabled, submodule.
- `can_reuse` compares per-input shapes: `n_tokens`
  (`llm_graph_input_embd/pos/attn_kv`), `n_outputs`
  (`out_ids`), and KV-dependent tails (`tail_identity.matches`,
  `self_kvarn_mat_idxs->ne[0] == mctx->get_n_kv()` in
  `llama.cpp/src/llama-graph.cpp:822`). **Observed verify widths vary
  (2/3/4) and n_kv grows every accepted round → reuse is busted on width
  transitions and possibly every round.** Rebuild-every-round is now the
  prime lump hypothesis, testable via the public `n_reused` counter
  (`llama_perf_context`, zero while `cparams.no_perf` — currently
  disabled by our defaults per `inference.cpp:714`).
- Sched split-execution (`llama.cpp/ggml/src/ggml-backend.cpp:1925`):
  without `callback_eval` each split computes as **one async
  `graph_compute`**; with it, per-node ask/view-graph walk plus a
  **full backend synchronize for every observed node** (line 1955) —
  i.e. `cb_eval` timing would measure a serialized, perturbed path
  (diverge.cpp needed explicit on/off contamination checks for IDs).
  Structural enumeration is available cheaper via
  `GGML_SCHED_DEBUG` → `ggml_backend_sched_print_assignments`
  (node→backend map, `sched->debug` read at sched creation).
- Existing env-gated facilities in the pin (all read-only to us, all
  `getenv`, all default-off):
  | env                                                                 | isolates                                                                      | where                                                |
  | ------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------- |
  | `GGML_BACKEND_COPY_PROFILE`                                         | inter-backend **state copies** + src/dst syncs, per tensor, with ms           | ggml-backend.cpp:1896 (static-once at first compute) |
  | `GGML_SCHED_DEBUG` (atoi level)                                     | **CPU/GPU split structure**: node→backend assignments; >1 adds per-node lines | ggml-backend.cpp:1990 (at sched new)                 |
  | `GGML_SCHED_DEBUG_REALLOC`                                          | graph **realloc/rebuild events**                                              | ggml-backend.cpp:1997                                |
  | `LLAMA_GRAPH_REUSE_DISABLE`                                         | A/B of **graph rebuild+alloc** (delta = build cost)                           | llama-context.cpp:886 (at ctx construction)          |
  | `SPEC_OPT_SINGLE_SYNC`                                              | A/B of redundant output-getter `synchronize()` barriers                       | llama-context.cpp:2227 (static-once)                 |
  | `llama_perf_context().n_reused`                                     | **per-call graph reuse count** (public API; needs `no_perf=false`)            | llama.h:1994                                         |
  | Env vars require in-process `_putenv` before first use (same proven |
  | pattern as the repack knob), i.e. a **host-owned knob build** — the |
  | instrumentation itself is supported pin functionality, not new      |
  | timers we invent.                                                   |

### Decisions taken before building (stated, then executed)

- **GetThreadTimes main-thread timer: SKIPPED, no build.** Hypotheses it
  would separate: H1 main thread burns wall−d3w (serial prep/sched/copies)
  vs H2 main blocked while other threads burn. Actionable change it would
  distinguish: H1 → reduce prep/copy/sync (exactly what the env A/Bs
  measure component-wise); H2 → wait-site tracing inside ggml/llama
  (submodule). Either outcome routes to the same next fork — run the env
  A/Bs first; if they do not explain the lump, the split needed is
  build-vs-set_inputs-vs-CPU-graph **inside** `process_ubatch`, where a
  coarse main/other CPU split still cannot resolve it. The timer cannot
  change the next decision, so per the standing rule it is not built.
- **`cb_eval` timing: not used** (perturbs scheduling — per-node sync +
  view-graph walk; enumeration duplicated by `GGML_SCHED_DEBUG`).
  `diverge.cpp`'s existing use stays untouched.
- **Host knob build (rev102) proceeds**, carrying exactly two
  decision-critical, numerics-neutral additions, hypotheses stated now:
  1. `ggmlprof.txt` device knob → strict single token → `_putenv` of the
     matching env above at the existing apply-knob sites (absent →
     nothing set → today's behavior byte-for-byte).
  2. `cparams.no_perf = false` + `n_reused` delta logged per verify call
     (`VROUND ... reuse=`): **if reuse deltas are 0 for nearly all
     rounds, rebuild-every-round is confirmed → prep is the prime lump
     component; if ≈1 per call, reuse works → lump lives in CPU-node
     eval/copies and the env A/Bs split those.** This single counter
     changes the next decision either way.
     Product A/Bs after deploy (`noreuse`, `singlesync` = decode/TTFT
     deltas; `copies`, `sched` = attribution logs), profile-OFF for timing
     cells, profile-ON for VROUND reuse/copies, fresh same-package
     sequential ref, full-ID parity on every cell whose switch is
     numerics-neutral (all of them: logging/params only).
- **Permission boundary (pre-drafted, not yet needed):** only if reuse is
  confirmed working AND the env A/Bs leave the lump unexplained, the
  necessary next step becomes the two commented timers inside
  `llama_context::process_ubatch` (`llama.cpp/src/llama-context.cpp`,
  functions are the graph-build and set-inputs scopes at lines ~3035 and
  ~3059): a temporary diagnostic uncomment/log, expected to appear in the
  device log as `graph build time` / `graph set inputs time` lines whose
  sum, with CPU-graph time, must reconcile against VROUND wall−d3w.
  Because `llama-context.cpp` is one of the **three pre-existing foreign
  edits**, rollback follows the snapshot/hash + diagnostic-only-delta
  procedure specified in full below (no stash, reset, checkout, or any
  reversal of the baseline foreign diff — reversing a pre-edit diff
  would delete the foreign edits). Why host timers/callbacks cannot
  answer: they only see `llama_decode` as a black box (wall−d3w) or
  perturb the path (`cb_eval`); the build/set-inputs split exists only
  between those commented lines. Work stops for explicit permission.
  (Permission was later granted under parent direction for exactly the
  default-OFF gated diagnostic described below.)

## 2026-10-07 rev102: supported-instrumentation campaign and the residual boundary

Host knob `ggmlprof.txt` (strict single token → `_putenv` of the matching
**pinned, default-off** env at the four apply-knob sites; absent → nothing
set) plus two numerics-neutral host additions (`cparams.no_perf=false`,
`VROUND ... reuse=` from `llama_perf_context().n_reused`). Hypotheses were
stated in the survey section before the build. Build rev102 `F0B6929A`,
host 338/338, hash-verified transfers, submodule still exactly its 3
foreign edits. Knob engagement verified from device log lines
(`ggmlprof.txt='...' -> ...`): absent×12, noreuse×6, singlesync×6,
copies×3, sched×3 — the run driver's echo labels read a stale local file
in two cells; the log is authoritative and correct.

### Results per facility

- **Reuse counter (`reuse=`): 78.6% of verify rounds rebuild the graph**
  (66 reuse=0 vs 18 reuse=1 of 84 rounds; reuse occurs on consecutive
  same-width rounds, rebuild on width transitions — consistent with
  `can_reuse` comparing `n_tokens`). Per review: **this proves missing
  reuse, not that rebuild/prep dominates wall — the dominance hypothesis
  stays RETAINED/UNPROVEN pending timing evidence below.**
- **`LLAMA_GRAPH_REUSE_DISABLE` A/B (profile OFF, interleaved p1/p2):**
  decode **+2.02% / +1.76%** (both passes same sign), TTFT +0.46%/+0.23%.
  **Correction: this measures only the _marginal_ cost of disabling the
  remaining (≈21%) reuse — the rounds/prefill batches that would have
  reused. It does not measure the total rebuild cost already paid by the
  78.6% of rounds that rebuild in the baseline, and no total-build-cost
  figure may be extrapolated from it** (the earlier ≈80 ms
  extrapolation is withdrawn). Total rebuild cost: UNMEASURED; the
  build-dominance hypothesis stays UNRESOLVED alongside
  set_inputs/apply/graph.
- **`SPEC_OPT_SINGLE_SYNC` (semantics-changing, not logging):** decode
  +0.88% / −1.56% (sign flips = noise), TTFT ≈0. **No gain → no
  promotion.** Full-ID parity was measured anyway: all r102 cells
  (including every singlesync dump) sit in the one full-sha256 bucket per
  run index (35/35/19 files, zero distinct). Termination/session guards
  are required before any future promotion and were **not** run (nothing
  to promote); numerics-neutrality was verified by IDs, never assumed
  from the name.
- **`GGML_BACKEND_COPY_PROFILE`: 0 lines.** The profile site is reachable
  (d3d12 `cpy_tensor_async=nullptr` forces the sync path) but no
  cross-backend split-input copies occur — activations are shared in
  place (`llama_gpu.h`). **Sched input copies: absent, not unmeasured.**
- **`GGML_SCHED_DEBUG`: structure captured.** 12,474 `## SPLIT` headers
  (all D3D12-titled survive our `gguf_gpu_log` filter; CPU-titled headers
  are dropped by the substring filter): decode calls show ~6 D3D12
  splits, prefill-style calls up to 131 — CPU split count not visible
  without widening our own log filter (host change, structure only, no
  timing → did not justify a build on its own).
- **no_perf always-on cost (as directed):** rev102 def OFF vs rev101 def
  OFF (same cells/knobs, cross-package, different time block): decode
  **+0.12%**, TTFT −0.44% — noise-level; a same-package ON/OON pair could
  not see it, so this cross-package proxy is the quantification, with its
  time-block caveat.
- **No-code lump experiments (profile ON, VROUND):**
  - **E1 thread sensitivity:** lump (wall−d3w) medians t6 **61.98** (avg
    of 59.73/64.24), t3 **67.01**, t2 **77.56** ms/round; d3w flat
    (22.45/22.91/21.88 — clean control). A 3-point fit
    lump(t)=S+P6·(6/t)^α estimates S ≈ 45.9 ms (α=1, 74%) to S ≈ 60.9 ms
    (α=2.5, 94-98%; α≥1.5 fits t3 with ≤1.7 ms residual). **Correction:
    under Amdahl-style assumptions this estimates a NON-SCALING
    FRACTION of the lump (74-98%), NOT proof of serial main-thread
    execution — scheduling, synchronization, bandwidth, and other
    effects that fail to scale with thread count are equally inside S,
    and the ≤26% complement is only "the scaling part under the same
    assumptions", not a measured pool-CPU-eval time.** Component
    identity: unresolved. IDs exact for all t2/t3/t6 dumps vs the
    same-package ref.
  - **E2 kvq8+flash (first attempt INVALID: bench-xbox-ort.sh owns
    `bench_kvq8.txt` and rewrote the manual upload with 0 — no #171
    engagement line; discarded). Re-run with the script's `--kv-q8`
    flag: engagement verified (2× `[xllama] KV cache: q8_0 + flash
attention (#171)`), lump 61.27 vs t6 61.98 → **−1.1% (no
    change)** — attention/KV via q8+flash is not a lump lever. IDs DIFF
    vs ref **as expected** (Q8 KV changes numerics; attribution cell,
    not a parity gate).
- **getenv sweep of the pin:** `GGML_BACKEND_COPY_PROFILE`,
  `GGML_SCHED_DEBUG(_REALLOC)`, `LLAMA_GRAPH_REUSE_DISABLE`,
  `SPEC_OPT_SINGLE_SYNC`, `LLAMA_TRACE` (model-loader only),
  `LLAMA_*_DEBUG` (structural), `LLAMA_KV_TAIL_PLANNER_TIMING`
  (planner-only) — **no facility times CPU graph eval, set_inputs,
  mctx->apply, or split orchestration. Supported-instrumentation options
  are exhausted.**
- **GetThreadTimes: remains skipped.** E1 bounds a non-scaling
  fraction (Amdahl estimate), not a main-thread/pool split; a main/other
  CPU split would still not name which phase or which non-scaling
  effect runs, and neither outcome routes anywhere except the phase
  diagnostic below — it cannot change the next decision.
- **Full-ID parity across the whole fixture family: 89 vsplit dumps, one
  full sha256 per run index, zero distinct digests** (rev99→rev102, all
  arms incl. singlesync/noreuse/t-cells; kvq8b excluded by design).

### Residual, predeclared decisions, and the authorized diagnostic

After every supported env and no-code experiment, the residual is:
**≈74-98% of the lump is a NON-SCALING FRACTION under Amdahl-style
assumptions (3-point thread fit — an estimate, NOT proof of serial
main-thread execution: scheduling, synchronization, bandwidth and other
non-scaling effects are equally inside it), within which the suspects
remain unsplit among {`mctx->apply` state/KV apply,
`model.build_graph`, `set_inputs`, split orchestration
(assignments/optimize/alloc-deps), the main thread's share of CPU-graph
execution, and non-scaling scheduler/sync effects}.** Host timers see
only the black box (wall−d3w); `cb_eval` perturbs the path (per-node
backend sync + view-graph walk); the env A/Bs bounded what they could
bound. **build / set_inputs / apply / graph hypotheses are retained
UNRESOLVED.**

Parent direction authorized exactly one **default-OFF gated**
`process_ubatch` phase diagnostic under the rules below; no other
submodule edit, no new timers elsewhere, no default promotion.

**Predeclared decisions — what each measured phase will be used for:**

| phase                         | actionable decision the measurement supports                                                                                                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mctx->apply`                 | apply dominant → propose a state/KV-update optimization as an exact change for review before editing, or a host param affecting it (kvq8 cross-evidence −1.1% is negative only for the attention path; apply itself untested) |
| `build_graph` (+ reset/alloc) | build dominant → propose graph-build/reuse optimization for review, or accept the cost; host knobs already bounded (noreuse marginal +2% only)                                                                                |
| `set_inputs`                  | set_inputs dominant → propose an input-staging/upload optimization for review before editing                                                                                                                                  |
| `graph_compute` minus d3w     | sched/CPU-graph remainder dominant → further split needs CPU-graph vs orchestration sites: report and STOP before any additional submodule edit                                                                               |
| residual bucket               | residual dominant → this four-phase cut is wrong: stop and re-scope; no further timers                                                                                                                                        |

- **Exact change (temporary, default-OFF gated, diagnostic only):** file
  `llama.cpp/src/llama-context.cpp`, function
  `llama_context::process_ubatch`: (a) time `mctx->apply()`; (b) time the
  rebuild branch (`res->reset()` → `ggml_backend_sched_alloc_graph`,
  covering `model.build_graph`); (c) time `res->set_inputs(&ubatch)`;
  (d) time the `graph_compute(...)` call. Gate: a function-local
  `static const bool` from `getenv("LLAMA_UBATCH_DIAG")` — absent/0 →
  **no timers and no output at all** (default behavior byte-for-byte);
  set only by the host knob `ggmlprof.txt=ubdiag` at the existing
  apply-knob sites, read once before first decode. Emission when on:
  `[diag] ubatch apply=%.3f build=%.3f set_inputs=%.3f graph=%.3f
reused=%d` (ms), and one `gguf_gpu_log` substring addition (host file)
  so the INFO line survives the existing filter. Nothing else in the
  submodule changes.
- **Accounting and explicit limits:** the four phases are sequential on
  the decode thread (no mutual overlap); `d3w ⊆ graph` by construction
  (never counted twice); per call report `sum(apply+build+set_inputs+
graph)`, VROUND wall, and **explicit `residual = wall − sum`** = the
  `llama_decode` work outside `process_ubatch` (batch init/split,
  output_reserve, sched_reserve, rewind) plus timer granularity — with
  ms and %; a residual >5% or negative blocks conclusions from that
  cell.
- **Validation:** host tests (338); profile OFF/ON pair with the diag
  present (profile overhead on top of diag); diag-OFF cells vs rev102
  same cells (always-gated-code overhead); full-ID sha256 parity of every
  dump against a fresh same-package sequential reference (plus
  cross-revision bucket).
- **Rollback — the only permitted reversal:**
  1. **Before any edit:** byte-identical snapshots + SHA-256 of every
     touched file (`llama.cpp/src/llama-context.cpp`,
     `uwp/inference-bridge.cpp`, `src/bridge/llama_gpu.h`) and the
     complete baseline submodule diff `git -C llama.cpp diff >
baseline-submodule.diff` (+ SHA-256). The baseline diff contains
     the three foreign edits and is a verification reference — **never
     reversed.**
  2. **After edits:** derive the diagnostic-only delta by `diff -u`
     snapshot vs edited file per file → `diag-only.patch` (contains only
     the diagnostic lines).
  3. **Remove only the diagnostic delta:** restore the snapshot bytes
     (equivalently reverse `diag-only.patch`) so each touched file is
     byte-identical to its baseline. **No stash, reset, force-checkout,
     or any reversal of the baseline diff; no unrelated code touched.**
  4. **Verify:** every restored file's SHA-256 == its recorded baseline
     hash; `git -C llama.cpp diff` byte-for-byte == the saved
     baseline-submodule.diff; `git -C llama.cpp status --short` shows
     the same 3 files with the foreign diff intact. Preserve snapshots,
     hashes, baseline diff, `diag-only.patch`, logs, and CSVs as review
     artifacts.
- **Why existing callbacks/host timers cannot answer it:** black-box
  bracket only (host), perturbing callback (`cb_eval`), and the exhausted
  env sweep — the apply/build/set_inputs/graph split exists only between
  those lines.

Defaults remain unchanged; nothing promoted (no ≥10% gain anywhere; the
closest candidates were rejected: bounded spin −0.58% whole-call, thread
cap +4.58%, singlesync noise). No commits.

## 2026-10-07 rev103: gated process_ubatch phase diagnostic — checkpoint

Executed exactly under parent direction: default-OFF gated four-phase
diagnostic in `llama_context::process_ubatch` only; snapshots/hashes and
the complete baseline submodule diff saved **before** any edit; isolated
`diag-only.patch` derived after; rollback removed only the diagnostic
delta; baseline verified byte-for-byte. Artifacts preserved in
`bench/results/diag-ubatch-20261007/` (`snapshot/`, `baseline-hashes.txt`,
`baseline-submodule.diff` + sha `42f64036…`, `diag-only.patch` + sha
`874b430c…` (99 lines: 37 +/1 real −, i.e. the extended filter line),
`diag-raw-sample.log`, `analysis.txt`). Host gate: 338/338, format and
coherence green; `.193` pre-transfer baseline hash matched exactly before
overwrite; build rev103 `0D3952B3`, exit=0. Device knobs cleaned after.

### Validation

- **Default-OFF overhead:** rev103 diag-OFF vs rev102 def-OFF (cross
  package, time-block caveat): decode **−0.77%**, TTFT +0.66% — noise
  level, no regression. Diag-ON emission cost (same package):
  **+1.92% decode** (171 log lines + per-call clocks) — measurement-mode
  only; profile atop diag −0.39% (noise).
- **IDs:** run1/run2 buckets: 46/46 fixture files one full sha256
  (every r103 cell incl. ref/dgoff/dgon against the fresh same-package
  sequential ref) with only `r102-kvq8b` distinct (by design, Q8
  numerics); run3: 24 files identical. **All r103 dumps exact.**
- **Engagement:** `ggmlprof.txt='ubdiag' -> LLAMA_UBATCH_DIAG=1` ×6
  (dgoff+dgon runs); `[diag] ubatch` lines ×171 per run, absent in all
  default-OFF runs.

### Findings vs the predeclared decisions (3 dgon runs, 513 phase lines)

| predeclared phase         | measured                                                                                                                                                                                                                                                                                                                                                                           | decision taken                                                                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mctx->apply`             | **0.00%** of batched sum, 0.01% of all                                                                                                                                                                                                                                                                                                                                             | hypothesis REJECTED — apply is not a lever                                                                                                                              |
| `set_inputs`              | **0.02%** (all: 0.07%)                                                                                                                                                                                                                                                                                                                                                             | REJECTED — no staging lever                                                                                                                                             |
| `build_graph`+reset/alloc | **1.34%** of batched sum; rebuild median **1.90 ms**, reused 0.00                                                                                                                                                                                                                                                                                                                  | REJECTED as dominance; consistent with the (now correctly scoped) noreuse marginal +2%                                                                                  |
| `graph_compute` − d3w     | **graph = 98.64%** of batched-call sum (99.05% of all); verify walls med 89.0 ms with d3w med 24.4 → **wall−d3w med 61.7 ms sits inside graph**                                                                                                                                                                                                                                    | **CONFIRMED locus** → predeclared gate: _further split needs CPU-graph vs orchestration sites: report and STOP before any additional submodule edit_ → **stopped here** |
| residual bucket           | **RETRACTED (checkpoint review):** nearest-duration matching is non-unique (verify/catchup share the magnitude class; the −3 ms negative residual proves circularity), so it does NOT validate per-call conservation — withdrawn: per-call conservation and "four-phase cut proven". Preserved: the batched graph-locus 98.64%, which is per-call self-contained and needs no join | superseded by the ID-tagged rev104 design below                                                                                                                         |

Per-run classes (consistent ×3): 1 prefill (graph 1820 ms), 26-28
draftish (sum med 15.04 ms, graph 99.4%), 99-101 classic (sum med
3.60 ms, graph 98.6%), 43 batched, 14 VROUND verify rounds.

### Rollback record (completed)

Snapshots restored over the three touched files; `sha256sum -c` vs
`baseline-hashes.txt` → all OK; `git -C llama.cpp diff` `cmp` vs saved
`baseline-submodule.diff` → **byte-identical**; `status --short` shows the
same 3 files with foreign edits intact; zero `ub_diag`/`LLAMA_UBATCH_DIAG`
traces remain; `.193` re-transferred to baseline (hash-OK ×3); host tests
338/338 after restore. No stash/reset/checkout used anywhere; the
baseline diff was never reversed. Note: the installed device package is
still rev103 (diag compiled in, default OFF, inert); the worktree and
`.193` sources are baseline — any future build reverts to baseline
automatically.

### Genuinely unresolved blocker

**`graph_compute` − d3w (≈61.7 ms/verify round ≈ 74% of verify wall ≈
~40% of decode)** is unsplit between {ggml-scheduler split/orchestration
(assignments, optimize, alloc-deps, events), CPU-backend graph
evaluation, and in-graph/scheduler synchronization}. Existing facilities
cannot attribute it (env sweep exhausted; `cb_eval` perturbs; host timers
see only the black box). Per the predeclaration, the next step — a
CPU-graph or split-orchestration timer inside `llama.cpp/ggml/src/` — is
a **further submodule edit and requires explicit approval of the exact
change first**; no such edit was made. No optimization is proposed at
this checkpoint: nothing measured yet clears the ≥10% paired product bar,
and no candidate exists whose mechanism targets this residual with
evidence. Defaults unchanged; no promotion; no commits.

## 2026-10-07 rev104 proposal (predeclared before build): ID/sequence-tagged phase split

Authorized as ONE coordinated default-OFF diagnostic. Design follows the
checkpoint review: **exact identity for every call, never nearest-time
matching.**

### Tagging (deterministic join)

- Host (`decode_loop.h`, profile-gated): `[xllama] VRCALL seq=%d width=%d
ctx=%llx` immediately before each `decode_verify_batch`, and
  `[xllama] VREND seq=%d` immediately after; `seq` = verify-round index.
- Submodule (`process_ubatch`, env-gated `LLAMA_UBATCH_DIAG`): each
  `[diag] ubatch` line gains **`id=` (monotonic per process), `ctx=`
  (this pointer), `nt=`, `gt=`**. Classification is exact:
  - **verify** = the single `[diag]` inside `VRCALL(k)..VREND(k)`
    (invariant: exactly one, `nt == width`, `ctx ==` host target ctx;
    violation → cell INVALID);
  - **prefill** = target ctx, `nt > 8` (fixture rule: prompt 119, widths
    ≤ 4 — stated, not inferred);
  - **classic** = target ctx, `nt == 1`; **catchup** = target ctx,
    `2 ≤ nt ≤ 8` outside windows; **draft** = `ctx !=` target ctx.
- `[xllama] sync ctx=… ms=…` lines from `llama_context::synchronize()`
  (env-gated): position-joined inside `VRCALL..VREND` (belongs to that
  verify call's wall) or outside (excluded).

### Narrow split inside graph_compute (one new submodule file: ggml-backend.cpp)

Stage timers at the single call site in
`ggml_backend_sched_graph_compute_async` and inside
`ggml_backend_sched_compute_splits` (static inspection: async wrapper =
reset → `ggml_backend_sched_alloc_graph` (split_graph +
alloc_splits) → compute_splits; per-split
`ggml_backend_graph_compute_async` on the non-callback path — the
`cb_eval` callback path is unused here because it perturbs timing; any
future callback use must state and quantify that perturbation first):

- **orchestration** = wall of `ggml_backend_sched_alloc_graph`;
- **compute_splits total**, with call-site backend sums **cpu_eval**
  (name CPU) and **d3_eval** (D3D12);
- **glue (in-graph sync)** = compute_splits − (cpu_eval + d3_eval):
  split-boundary `event_synchronize`, `event_record`, copy calls
  (measured zero on this fixture), scheduler bookkeeping.
- Accumulators: file-scope statics in ggml-backend.cpp, clock reads only
  when `LLAMA_UBATCH_DIAG=1`; four extern `ggml_diag_*_ms()` getters
  declared locally in llama-context.cpp — no header changes.

### Async/overlap accounting (no double counting)

Static facts: d3d12 `.graph_compute` is synchronous (fence spin completes
inside `run_now`; `.synchronize = nullptr`, "graph_compute completes
before returning") with no `graph_compute_async` iface → the sched's
async wrapper runs synchronously on the decode thread; CPU
`ggml_backend_cpu_graph_compute` joins its threads before returning.
Every bracket is therefore a sequential wall interval on one thread —
no temporal overlap between phases; completion waits count once, inside
the bracket that performs them (fence spin in d3_eval, pool join in
cpu_eval, split-boundary event waits in glue, llama `synchronize()` as
its own line). **Nesting rule:** top-level non-overlapping sum =
`apply + build + set_inputs + orchestration + compute_splits + finalize
(graph_compute_complete) + sync-inside-wall`; `cpu_eval/d3_eval/glue`
are CHILDREN of compute_splits and never added to the top-level sum.
`d3_eval` cross-checked against VROUND `d3w` (agreement within 10% or
the cell is INVALID). Stated limits: the first `[diag]` line (prefill)
may include ctx-init orchestration accumulated before the first window →
prefill stage values excluded from analysis; draft-ctx sync lines outside
verify windows excluded from verify accounting; the process-CPU and
Amdahl caveats above stand.

### Predeclared bounded decision table (before build)

Per verify call (exact join), each top-level component as % of that
call's VROUND wall:

| condition                       | action                                                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `cpu_eval ≥ 20%` of verify wall | propose the EXACT CPU-backend change (site + mechanism + expected gain) for review — no edit without approval |
| `orchestration ≥ 20%`           | propose exact scheduler/split-count change for review                                                         |
| `glue ≥ 20%`                    | propose exact in-graph sync/copy change for review                                                            |
| `sync-inside-wall ≥ 20%`        | propose exact llama synchronize change for review                                                             |
| any component 10–20%            | ceiling table with mechanism; no edit this cycle                                                              |
| any component < 10%             | ceiling entry only                                                                                            |
| residual > 5% or negative       | accounting incomplete: **stop, no component conclusions this cycle** (no histogram fallback)                  |
| no action row reached           | record precise ceiling for the whole residual; stop                                                           |

Threshold rationale (**corrected after checkpoint review**): 20% of
verify wall at a 32% verify share is `f = 0.064` of decode. Even **full**
recovery yields `1/(1−0.064) − 1 = 6.84%` throughput, and 50% recovery
`3.31%` — neither reaches the ≥10% bar. **The 20% threshold is
prioritization only, never evidence of reaching the product bar.** At this
share, a verify-phase component can reach 10% only with ≥ `f = 0.0909`
i.e. ≥ 28.4% of verify wall _and full recovery_ — larger shares need
proportionally less recovery, which must still be argued from mechanism.

### Discipline (unchanged)

Pre-edit byte snapshots + SHA-256 of all five touched files
(`llama.cpp/src/llama-context.cpp`, `llama.cpp/ggml/src/ggml-backend.cpp`
— pristine at baseline — `uwp/inference-bridge.cpp`,
`src/bridge/llama_gpu.h`, `src/bridge/decode_loop.h`) plus the complete
baseline submodule diff; isolated `diag-only.patch` derived after edits;
default OFF (no clock read, no output when the env is absent); host
gates 338 + format + coherence; OFF/ON overhead and full-ID parity vs a
fresh same-package sequential reference; rollback removes only the
diagnostic delta; baseline hashes and the complete foreign diff verified
byte-for-byte on **.157 (local worktree) and .193**; no optimization, no
default promotion, no model change until measurement review. This design
adds no evaluation callback — added work is clock reads, log lines, and
accumulator stores, quantified by the OFF/ON overhead pair.

## 2026-10-07 rev105 tagged result (rev104 attempt quarantined) and decisions

- **Identity/quarantine (corrected after independent review).** The
  first pass labeled `r105` actually ran against the still-installed
  **rev104** package (deploy skipped by a wrong-working-directory
  shell). **Correction: the CSVs were NOT overwritten — bench appends
  one row per run, so every r105 CSV holds four measured rows (2 from
  the mislabeled pass + 2 from the genuine pass); only the token
  sidecars overwrite (334 B, single-pass, mtimes inside the genuine
  windows, sha identical).** Those rows are **QUARANTINED by
  provenance filter, not deleted**: originals preserved byte-for-byte;
  admitted rows = `date >= 2026-10-07T01:35:00Z` (gap: last mislabeled
  row 01:33:59Z, first genuine 01:35:09Z; both passes follow the fixed
  driver order ref→p1off→p2off→tagoff→tagon with monotonic dates).
  Provenance-filtered tables written as
  `bench/results/vsplit-20261006-r105-*.provenance.csv` (2 admitted
  rows each); rule, anchors, and append-vs-overwrite evidence recorded
  beside them. Log joins are filtered independently by `VRCALL +
[diag]` presence (rev104 gate never fired — file-scope static
  initialized at load before `_putenv`; fixed to a lazy function-local
  static). Identity anchors for the admitted set: installed PFN
  `GianlucaMazza.xllama_1.6.0.105`, build `exit=0` sha `82A06F64…`,
  deploy `success=true`, 2052 `[diag] ubatch id=` lines (rev105
  format), dump mtimes inside the genuine window.
- **Join method (exact, no time matching):** windows `VRCALL(k)..VREND(k)`
  carry `seq`/`width`/`ctx`; each window must contain exactly one
  `[diag]` with `nt == width` and `ctx ==` host target; the VROUND row
  for `seq k` is the k-th row of the post-loop burst (code order — the
  emission loop and the window loop share the same `vr_*` index), with
  `width == width` as a second invariant. An earlier analysis keyed VROUND
  rows by width and admitted 9 coincidental collisions — **discarded**;
  the corrected join yields **42/42 windows, invalid = 0, width invariant
  true** across 3 genuine tagon runs. Classes observed (exact, by
  ctx/nt/window): **verify 42, draft 376 (ctx ≠ target), classic 87
  (nt = 1), prefill 3 (nt = 119), catchup 0 (none on this fixture)**,
  sync-outside 2868.
- **Exact tagged conservation (top-level non-overlapping). Primary
  claims use MEASURED runs only (run_index 2/3, n=28 joins; the warmup
  run, n=14, is reported separately and never used for claims).**
  Measured (n=28): `apply 0.01% + build 1.53% + set_inputs 0.01% +
orch 0.00% + comp 97.30% + finalize 0.00% + sync-inside-wall 0.00% +
residual 1.15% = 100.00%`. Per-call residual percent: **median
  **GATE verdict (predeclared per-call criterion, applied without
  pooling): FAIL.** The criterion is "residual >5% or negative for a
  verify call → accounting incomplete, stop, no component conclusions
  this cycle". Measured max = **5.56% (measured run3, seq0)**; measured
  run2 seq0 = 5.30% also fails; the warmup seq0 = 6.24% fails
  separately (warmup kept out of claims anyway). **Pooling does not
  retroactively change acceptance — the earlier "pooled PASSES,
  accounting complete" reading is RETRACTED.** Accounting is therefore
  INCOMPLETE this cycle; every component share below (cpu, d3, build,
  glue, and the rest) is retained as a **PROVISIONAL observation
  only**, and any conclusion gated on this table is withdrawn (see
  Outlier identification and Decision-table corrections below).
  Warmup (n=14, separate): pooled 1.27%, max 6.24% — consistent,
  unused for claims.
- **Nested accounting (children of `comp`, never added to the top):**
  Measured runs: `cpu_eval 67.85%`, `d3_eval 29.42%`, `glue 0.03%`
  (warmup separately: 68.33% / 28.80% / 0.03% — consistent).
  Cross-check `d3_eval` vs VROUND
  `d3w`: median abs error **0.9%**, max 1.0% → **VALID (≤10%)**.
  Known flaw disclosed: the `orch` timer wraps
  `graph_compute_async`'s alloc branch, which llama's direct
  `ggml_backend_sched_alloc_graph` call in the rebuild path never enters —
  `orch ≡ 0` is inert; llama-side split/alloc cost is instead inside
  `build` (1.53%, covering the 33/42 rebuild rounds), so orchestration is
  bounded ≤ 1.53% → ceiling entry, not an action row.
- **OFF/ON overhead (recomputed from admitted provenance rows only,
  n=2 measured rows per cell):** diag-OFF vs rev103 OFF pairs
  (cross-package): **+42.5 ms (+1.11%)**, noise band;
  diag-ON/profile-OFF same package: **+9.0 ms (+0.23%)**; profile atop
  diag: **−29.5 ms (−0.76%)** (noise). The earlier 4-row contaminated
  medians (OFF 3871.8, tagoff 3887.1, tagon 3882.3 →
  +0.94%/+0.67%/−0.12%) are **superseded**.
- **Full-ID parity: ONE fixture (spec-chat-open @ 64 tok, greedy seed
  1. across repeated cells — not a set of different prompts. At each
     run index all dumps are one full sha256: run1 56 files, run2 56,
     run3 34; sole outlier `r102-kvq8b` (by design, Q8 numerics) — ALL
     EXACT.** The r105 sidecars are genuine-pass-only (overwrite + mtime
     inside the genuine window + identical sha), so parity needs no row
     filter.
- **Decision-table outcomes (CORRECTED — gate FAILED this cycle):**
  no action row is reachable from a failed accounting gate. The
  measured shares (`cpu_eval 67.85%/68.0%`, `d3_eval 29.42%/29.2%`,
  `build 1.53%`, `glue 0.03%`, apply/set_inputs/finalize/sync ≈ 0) are
  **PROVISIONAL observations**, not gated conclusions; in particular
  the CPU ACTION-row conclusion ("the only component that can plausibly
  clear the ≥10% bar", ≈ 21.7% of decode with 45% recovery ≈ 10.6%) is
  **RETRACTED as gated** — the arithmetic stays on record but cannot
  drive a decision until a cycle passes the residual gate. Ceiling
  entries derived from these shares are provisional for the same reason.
- **Concrete next decision (HELD — justification provisional):** the
  knob-gated `offload_kqv` A/B (host-owned `apply_gguf_gpu_context`,
  default unchanged, reversible) remains the proposed next experiment,
  but its justification rests on the gated cpu_eval share and is
  **held pending a cycle that passes the predeclared residual gate**.
  No optimization or build proceeds in the meantime.
- **Outlier identification (existing code/logs only, no build):** the
  gate failures are exactly **3 of 42 joined calls, all `seq=0`** (the
  first verify call of each run): warmup 6.24% (6.06 ms / 97.04 ms),
  measured run2 **5.30%** (4.82 / 91.03), measured run3 **5.56%**
  (4.79 / 86.29). A second reproducible cluster: **seq5 in all three
  runs at 3.75-3.79%** — absolute gap **5.01-5.05 ms**, the same ~5 ms
  as seq0, diluted only because seq5 is the first width-4 call (wall ≈
  133 ms). All other positions: 0.43-0.65% (absolute **0.44-0.45 ms**
  baseline). The gap is therefore NOT spread: it is **~4.8-6.1 ms at
  the first call of the loop and ~5.0 ms at the first width-4 call**,
  reproducible at identical sequence positions in all three runs.
- **Precise unresolved residual (named from code; location not yet
  split):** the gap lies inside the VROUND `llama_decode` wall but
  outside every timed phase (`sync` lines + apply/build/set_inputs/
  orch/comp/finalize) — i.e. in the `llama_context::decode()` regions
  around `process_ubatch` (entry region ~lines 3363-3570: `balloc`
  init/split, `output_reserve`, `sched_reserve()` — which can split/alloc
  a graph OUTSIDE every current timer — kv/nextn bookkeeping, the
  output-id/`n_outputs` handling immediately before the call — plus the
  epilogue after `process_ubatch`), plus host bracket overhead (VRCALL/
  ScopeTag/clock reads, ≪ 1 ms and unable to explain ~5 ms). Code-located
  candidates for the two ~5 ms events (NOT proven from existing logs):
  first-use reserve/alloc paths at seq0; first width-4 capacity paths
  (ubatch split / output-array growth) at seq5. Splitting them further
  requires the bounded correction proposal below.
- **Inclusion manifest preserved:** the admission set is unchanged by
  this verdict — original CSVs byte-identical (SHA-256 in
  `bench/results/vsplit-20261006-r105-PROVENANCE.txt`), provenance
  tables and the `date >= 01:35:00Z` rule untouched, warmup kept
  separate; the gate failure is recorded as a verdict on the admitted
  set, never by altering inclusion.
- **Package/deployment anchor:** full MSIX sha256
  `82A06F64BBBFC795FBA216F6AD9DBACFABCECEEB1D4B127403EB5B65C665EB1A`
  (build log `exit=0`, matching local file); deploy anchor = Device
  Portal `Installation succeeded`, PFN
  `GianlucaMazza.xllama_1.6.0.105_x64__pj67f1fcj4n14`, time-bounded by
  the CSV gap (01:33:59Z last mislabeled row → 01:35:09Z first genuine
  row); an exact install wall-clock was not recorded.
- **Bounded correction proposal (approval required; NO criteria change,
  NO build started):** extend the same default-OFF isolated diagnostic
  with exactly one added timed pair in `llama_context::decode()` — `prep`
  (decode entry → `process_ubatch` entry, with the separately
  line-logged `sync` excluded) and `tail` (`process_ubatch` exit → decode
  exit) — so the residual decomposes into named `prep` + `tail` +
  bracket overhead and the predeclared 5% gate is re-evaluated on the
  SAME unchanged criterion (per call, no pooling). Same five-file
  snapshot/delta/rollback discipline, dual-host hash + foreign-diff
  verification, OFF/ON overhead and full-ID checks. Until an approved
  cycle passes the gate: component shares stay provisional, the
  offload_kqv A/B stays held, no optimization.
- Note: no evaluation callback was used; added work is clocks/logs/
  accumulators, quantified above. Process-CPU and Amdahl caveats from
  earlier sections still stand for any thread-scaling inference.

## 2026-10-07 rev106: approved prep/tail correction — GATE PASSES

Approved scope: smallest isolated default-OFF prep/tail instrumentation
only; no remainder-phases; explicit multi-ubatch handling; sync
single-counted; unchanged per-call ≤5% residual gate; warmup separate.

### Discipline executed

- **Pre-collection baselines (both hosts):** local 5-file SHA-256 +
  full submodule foreign diff re-verified == saved baseline; `.193`
  5-file hashes captured and equal to local; `.193` foreign diff
  (`git diff HEAD` of the three foreign files) == baseline, saved as
  `foreign-diff-193-pre106.txt`. Reconstructing the diagnostic state
  first required a repair: applying the multi-file `diag-only-r104.patch`
  per-file corrupted state — **fully restored from snapshots (hash-OK),
  stray `.orig/.rej` removed**, patch split into 5 per-file sections,
  dry-run ALL OK, then applied (counts match the original edit:
  k_ub_diag 14, ggml_diag_on 7, VRCALL/ubdiag/filter present).
- **New delta only:** `diag-only-r106.patch` (321 lines = preserved r104
  sections 269 + prep/tail hunks 52), sha `58cec6d2…`; host gates 338/338,
  format, coherence green before transfer; transfers hash-verified.
- **Before collecting:** build `exit=0` sha
  `203966786459BB6E3584F29D7F7A092A6A6D4C643C5C61AA2D2C137D20BAF33E`
  == local msix; install `success=true`; PFN asserted `_1.6.0.106_`;
  **engagement probe before the rest of the collection**: dspan 513,
  ubatch-id 1539, knob 27 (≥3 each) → continued only after OK.
- **New unambiguous run IDs + immutable originals:**
  `bench/results/pt106-20261007-*.csv` (prefix never used); manifest
  `pt106-20261007-MANIFEST.txt` records per-cell rows=2, run_index 2/3,
  unique monotonic dates (**SINGLE-PASS ALL CELLS: YES**), SHA-256 of
  every original, package identity and engagement — nothing rewritten.

### Instrumentation (direct intervals, no remainders)

RAII span in `llama_context::decode()`: **prep** = decode entry → first
`process_ubatch` entry; **inter** = summed direct gaps between consecutive
`process_ubatch` calls (multi-ubatch explicit; emitted as `n_ub=` on the
`[diag] dspan` line, integrity-checked against the ubatch-line count);
**tail** = last `process_ubatch` exit → decode exit (emitted at every
exit iff ≥1 ubatch completed). Accounting: `top = prep + inter + Σ
ubatch-top + tail` — **sync lines are inside-span subset evidence and are
NEVER added again** (measured: 0.003 ms = 0.0001% of wall in measured
runs). No phase is a remainder; residual stays wall − top.

### Results (admission: VRCALL + dspan presence — rev106-only segments)

- **MEASURED runs (run_index 2/3, n=28) — GATE: PASS.** Per-call
  residual: med **0.20%**, max **0.72%**, min 0.13%, all positive —
  every call ≤5% on the unchanged criterion. Warmup (n=14, separate):
  PASS too (max 0.72%). The three old gate failures are resolved by
  direct measurement: **seq0 first-call `prep` = 4.14/4.14/3.94 ms**
  (the previously missing ~4.8-6.1 ms), seq0 residual now 0.67-0.72%;
  the seq5 cluster now 0.13-0.15% (captured in that call's own prep).
- Top-level shares (measured): **prep 0.78% + inter 0.00% + ub_top
  98.76% + tail 0.24% + residual 0.22% = 100.00%**; children of ub_top:
  **cpu 68.37%, d3 28.78%, glue 0.03%, build 1.55%**; d3 vs VROUND d3w
  med 0.9% (max 1.1%) VALID. Multi-ubatch calls: 0 on this fixture
  (integrity checks all clean: dspan count, n_ub match, order, ctx/nt —
  zero failures; `dspan_outside` 471 = exactly the probe cell's out-of-
  window dspan lines: 513 − 42).
- **Same-package OFF/ON overhead (single-pass):** OFF pair 3878.3 ms
  (p1 3895.4 / p2 3861.2), ref 3853.1; diag-ON/profile-OFF **+49.7 ms
  (+1.28%)**; profile atop diag **+2.6 ms (+0.07%)**.
- **Fresh full-token parity: ALL EXACT** — 5 pt106 dumps per run index,
  one full sha256 each (3/3 buckets single); cross-package: the pt106
  family digest equals the legacy family digest.
- Verdict: accounting is now complete and gated-valid for this cycle;
  per approval, **no offload A/B, kernel change, optimization, or
  default promotion until this measurement is reviewed.**

### Rollback (completed, independently verified)

Snapshots restored: local 5/5 hashes OK, foreign diff byte-identical,
status = exactly the 3 foreign files, zero diag traces, 338/338 +
format + coherence. `.193`: 5/5 hashes OK; foreign diff == baseline AND
== the independent pre-collection capture (`foreign-diff-193-pre106.txt`
vs `post106`, byte-equal); `.193` status = 3 foreign files + the
standard `apply-uwp-patches.sh` targets (repack, local-split,
remote-attn, llama-mmap, kv-mixed-state-stream — build artifacts of the
rev106 build, never touched by hand); no ggml-backend/diag files.
Device knobs deleted. Evidence: `analysis-r106.txt`,
`pt106-20261007-MANIFEST.txt`, `diag-only-r106.patch` (+sha),
per-file split patches, `engage.log`, `full.log`.

## 2026-10-07 okqv A/B — precheck + predeclaration (before collection)

Approved: ONE host-only, default-OFF `offload_kqv` A/B using the existing
llama setting; no submodule kernel changes; no default promotion.

### Code precheck (read-only)

- llama's own default is `offload_kqv = true`
  (`llama-context.cpp:6244`); our host forces it `false` in
  `apply_gguf_gpu_context` (`llama_gpu.h`) for GGUF+GPU runs. Setting ON
  removes the CPU pin of the KV→attention-output chain
  (`llama-graph.cpp:3521`), letting the scheduler place those nodes.
- **Backend support:** our d3d12 op set is {MUL_MAT, NONE/RESHAPE/VIEW/
  PERMUTE/TRANSPOSE} only — attention softmax/fusion nodes are NOT
  supported on-device and will fall back to CPU via the scheduler. What
  CAN move: QK/PV-style MUL_MATs (and any resulting cross-backend input
  copies). **Therefore engagement must be measured, never assumed.**
- **Failure modes are soft in our config:** no native KV-tail attention
  on the device → WARN + `kv_tail_tokens` disabled (we do not set
  `LLAMA_KV_TAIL_ALLOW_GENERIC`); remote/local-attn throws require
  remote_attn_host/local_attn_dev, which we never set. Context creation
  failure is still treated as fail-fast (no CSV → abort).
- **Bounded memory/OOM fail-fast:** first cell is a 16-token probe with
  `offloadkqv.txt=1`; pass criteria = CSV produced (app alive),
  `gpu_mem_mb < gpu_budget_mb`, `peak_ws_mb` not beyond prior peaks
  (~4.3 GB observed). Any failure aborts the suite (rule R4).
- **Catchup fixture (and classification correction):** spec-chat-open64
  at d4 IS a real catchup fixture — existing MTP_STATS shows
  **`catchup_tok=63` in all three admitted pt106 tagon runs**. Prior
  wording said catchup class = 0 because out-of-window multi-token
  target-ctx calls numbered zero; that only shows no MULTI-TOKEN catchup
  batches outside windows. Scalar catchup replays a single token
  (`mtp_catchup_batch(..., 1 token, ...)`), so by ctx/nt alone it is
  indistinguishable from other nt=1 scalar calls (the "classic" /
  "draft" buckets) — **the old classifier could not detect it, and
  "catchup absent" is retracted**. Catchup existence is established by
  `catchup_tok=63`, not by the window classifier.

### Host-only change (3 files, default byte-identical when knob absent)

`inference_params.h` (+`bool offload_kqv=false`), `inference.cpp` (override
after `apply_gguf_gpu_context` only when the param is set, with a host log
line), `inference-bridge.cpp` (`offloadkqv.txt` via `read_local_int`).
Baseline snapshots `snapshot-r107/` + `baseline-hashes-r107.txt`; delta
`offloadkqv-host-only.patch` sha `6a9e741e…`; submodule foreign diff
byte-identical before/after (no submodule file touched this cycle).
Engagement/overhead use EXISTING counters only (d3d12 backend summary
`graph_compute/matmuls/wall/GPU`, `ggmlprof=copies` copy-profile lines,
`ggmlprof=sched` split headers, knob/host log lines) — no new diagnostic
delta introduced; any future new delta requires asking first.

### Pre-collection baseline gate + raw evidence (added by review)

- Before ANY timing row is admitted: verify installed rev107 identity
  (PFN `_1.6.0.107_`, msix sha == build log) **and confirm the
  default-OFF diagnostic baseline is clean**: a knob-absent cell must
  show ZERO `[diag]` lines and ZERO `offload_kqv=1` lines in the device
  log (rev106 diag code was fully rolled back; rev107 adds only the
  host knob) — recorded before the paired timing cells run.
- **Raw evidence preserved immutably:** full device log copy, per-cell
  backend-summary and `## SPLIT`/copy-profile extracts, all raw CSVs
  (originals never rewritten), and a collection manifest with hashes and
  single-pass checks.

### Predeclared decision rules (stated before collection)

- **R4 OOM/fail-fast:** probe criteria above fail → abort suite, report.
- **R1 engagement (REFINED before admitting rows):** timing deltas and
  generic `backend copy profile` line presence are NOT sufficient. Proof
  requires **scheduler/backend evidence of changed KQV/attention
  placement**: under `ggmlprof=sched` (existing `GGML_SCHED_DEBUG`),
  attention/KQV-relevant input tensor names (KV-cache/attention tensors)
  must appear inside D3D12-titled `## SPLIT` headers with ON but not OFF
  (or an equivalent backend-dispatch change of those operations to
  D3D12), plus the host `offload_kqv=1` knob line. Backend-summary
  matmul/wall/GPU counters are retained as supporting evidence only. If
  placement evidence cannot be produced from these existing traces,
  engagement is reported **INCONCLUSIVE (not proved)** — not "no-op" and
  not "proved" — and no benefit claim may stand on it.
- **R2 token parity:** full-sha256 equality of ALL of {ref-off, ref-on,
  d4-off, d4-on} across both passes at each run index (catchup fixture
  included) → any mismatch **REJECT as token drift**.
- **R3 material benefit:** paired profile-OFF cells (2 passes × 2
  measured rows per arm; whole-call = ttft + decode): ON gain must be
  positive in BOTH passes AND exceed the combined within-arm range
  (uncertainty with this n) → else **REJECT as no material benefit**.
- **No promotion regardless** of outcome; ≥10% repeated-gain + parity
  bar unchanged and separate. **CPU backend share stays an attribution
  hypothesis** (where verify time is spent), not proof of any specific
  op — engagement/ID/timing results decide, not the share number.

## 2026-10-07 okqv A/B result — engagement INCONCLUSIVE, benefit REJECTED

Admission gates (all enforced before rows were treated as evidence):

- **SHA hard equality PASS (4/4):** build-local.log `sha256-fixed` ==
  builder-side msix certutil == local fetched msix == driver argument ==
  `7f81636d…669fb` (the driver's advisory note branch was re-checked
  independently; record in `okqv-20261007-evidence/msix-sha-gate.txt`).
- **Default-OFF baseline clean PASS:** device log reset before the clean
  cell (full history preserved in earlier local copies); prefix before
  the first `offload_kqv=1` line: **[diag] = 0, offload lines = 0**.
- **Fail-fast probe PASS with actuals reported:** probe (ON, np16)
  `peak_ws_mb = 4273`, `gpu_mem_mb = 2273 < budget 4147`; suite-wide max
  `peak_ws = 4295 MB`. **Reconciliation:** the predeclared bound ("not
  beyond prior peaks ~4300 MB") governs and **passes** (4273 ≤ 4300;
  4295 ≤ 4300); the script's 4600 MB threshold never bound — the gate
  was not relaxed retroactively.
- **Fresh-log provenance:** raw evidence is the post-reset log
  (`okqv-20261007-evidence/xllama-rev107-fresh.log`, sha recorded in the
  manifest); cumulative historical `[diag]` contamination lives only in
  the pre-reset local copies and cannot mix with this run.

### Rule outcomes

- **R1 engagement: INCONCLUSIVE (not proved).** A matched OFF
  scheduler-trace cell (`eng-sched-off`) was collected alongside the ON
  cell (`eng-sched`), same prompt/np/runs/knobs. Evidence: **4158 D3D12
  `## SPLIT` headers each side with IDENTICAL (index, input-count)
  sequences (all input counts 0), backend counters identical (6526
  calls / 8232 matmuls / 1251 2col; wall 2113 vs 2155 ms within noise),
  zero `backend copy profile` lines.** Attention/KQV input NAMES are not
  observable — the `gguf_gpu_log` substring filter drops the input lines —
  so per the refined rule the placement change is **not proved**;
  timing deltas and copy-line absence are explicitly insufficient. The
  consistent structure/counter identity is recorded as supporting
  evidence of _no observable change_, not as proof either way.
- **R2 full-ID parity: PASS (fixture-separated).** np64 parity fixture:
  run1 10/10, run2 7/7, run3 6/6 dumps — one full sha256 per run index,
  OFF == ON == sequential refs == legacy family; np16 probe fixture:
  2/2 single digest (clean == probe). An initial blanket FAIL was a
  classifier artifact (np16 probe co-bucketed with np64) — disclosed and
  corrected. **Real catchup in the parity fixture:** fresh log
  `catchup_tok = 63` in 17 np64 d4 runs (=16 in the 2 np16 runs); the
  earlier "catchup = 0" classification remains retracted (scalar catchup
  is nt=1 on the draft ctx and was absorbed by the classic/draft buckets).
- **R3 paired profile-OFF benefit: REJECT — no material benefit.**
  Whole-call (decode+ttft): pass1 OFF 6024 → ON 6047 (**+23.1 ms**),
  pass2 OFF 6046 → ON 6009 (**−37.6 ms**) — **signs flip**; pooled
  medians OFF 6046.2 vs ON 6038.7 (−0.12%) with within-arm spreads 68 /
  77 ms — the rule (both passes negative AND beyond spread) fails.
  decode −0.45%, ttft −0.31% (all inside noise).
- **R4 fail-fast: PASS** (values above). **No promotion** regardless.

### Bounded result and next decision (for review)

The single approved host-only `offload_kqv` A/B yields **no material
benefit (rejected)** with **engagement unproved (inconclusive)** — the two
are independent: even a positive timing would not have stood without R1.
Host-only scope kept (3 files, delta `offloadkqv-host-only.patch` sha
`6a9e741e…`); **no submodule instrumentation was introduced** (split input
names remain hidden by the existing log filter — capturing them would
require a host log-filter widening, i.e. a new diagnostic delta, which
requires asking first). CPU backend share remains an attribution
hypothesis only. Decision options: (a) **close `offload_kqv` for this
fixture as rejected** (recommended: R3 alone forbids promotion), or
(b) request the one-line host log-filter widening to pursue placement
proof knowing R3 still forbids promotion on timing. All knobs restored,
defaults unchanged, no promotion.

## 2026-10-07 read-only next-step analysis (no new instrumentation) — rev107 closed, target-CPU cost model

- **rev107 `offload_kqv` A/B: CLOSED per review** — no material benefit
  (R3), engagement inconclusive (R1); no log-filter rebuild; no promotion.
- **Method:** re-analyzed the existing rev106 `[diag]` records
  (`/tmp/xllama-rev106/full.log`, gated-valid cycle) by (ctx, nt) class —
  no new timers, no source edits. Baseline and foreign edits untouched
  this turn.

### Exact per-run cost model (validated arithmetic)

Per tagon run (last-run ctx bucket; rounds=14 + classic=29 verified
against MTP_STATS):

- **target classic (nt=1): comp 59.08 ms, cpu 41.57 ms, d3 17.43 ms**
  (n=29) — the earlier "3.5 ms classic" was the DRAFT context (cpu 0.96
  ms med, n=1983 incl. catchup replays); draft-ctx work is negligible.
- verify nt2/3/4: cpu 57.06 / 73.00 / 88.25 ms (n=9/2/3) — perfectly
  linear: **cpu(target call) ≈ 26 ms fixed + 15.6 ms × nt**
  (differences 15.5/15.9/15.25 agree).
- Validation: Σnt over the43 target calls = 65 ≈ 64 generated tokens;
  model total = 26×43 + 15.6×65 ≈ 2132 ms ≈ measured 2130 ms — **the
  model reproduces the log**.
- **Share of decode (3858 ms):** target-CPU graph ≈ **55%** per run
  (classic 29×41.57 ≈ 1206 ms, i.e. the classic path dominates verify's
  14 calls ≈ 924 ms); split: **fixed ≈ 26×43 ≈ 1118 ms ≈ 29% of decode;
  per-token ≈ 15.6×65 ≈ 1014 ms ≈ 26%.** Prefill excluded
  (nt>4 cpu 1222 ms lives in TTFT). This supersedes the verify-window-only
  21.7% figure: cpu cost sits on BOTH classic and verify calls.

### ONE concrete likely expensive operation

**The ~26 ms fixed per-call portion of the target CPU-backend graph
(plan/dispatch/barrier over the CPU-resident node set), paid on every
target decode call (~43×/run ≈ 29% of decode).** The nodes are
CPU-resident by construction: the KV→attention-output chain is pinned
(`llama-graph.cpp:3521`, offload_kqv=false) and the backend implements
only MUL_MAT/view ops — placement changes are structurally inert, which
is exactly what rev107's identical ON/OFF split structure and counters
showed.

### Gain bound

Fixed ≈ 29% of decode: total elimination ⇒ `1/(1−0.29)−1 ≈ 41%`
throughput (impossible in practice); clearing the ≥10% bar needs only
`f ≥ 0.0909` ⇒ recover **≥31% of the fixed part** (or the per-token part,
26% of decode, needs ≥35% recovery). Both ceilings are large enough that
the bound does NOT kill the path — but no supported host lever remains
(see exclusions), so the bound is conditional on the diagnosis below.

### Exclusions — no repeat of rejected paths (verified this turn)

- threads (paired reject, +4.58% whole at t3), bounded spin (rev101),
  graph reuse (+2% marginal; reuse already active), offload_kqv/rs
  (rev107 closed; placement inert), depth/pmin draft-efficiency family
  (depth pilot ≤ +1.1%; pmin is the same trade — NOT re-proposed),
  **flash-attn already default-ON** (AUTO ⇒ `flash_attn=true`,
  llama-context.cpp:819; the only override is the training path),
  **persistent CPU pools already attached** (inference.cpp:519/541 —
  the per-graph pool-churn suspect is pre-fixed), `LLAMA_ATTN_ROT_DISABLE`
  is a no-op on F16 KV (only gates quantized-K rotation), lossy KV schemes
  (KVarN/snapkv/q8 variants) cannot pass the exact-ID gate by
  construction.

### Smallest decisive experiment (PROPOSAL ONLY — needs permission)

No supported host knob can split the fixed26 ms (all knobs above are
rejected/inert/pre-enabled), so the decisive measurement needs a
**temporary default-OFF micro-diagnosis inside
`ggml_backend_cpu_graph_compute`** (pinned submodule, ONE site): three
`ggml_time_us` scopes — {graph plan / threadpool dispatch+barrier,
node-loop execution} — emitted as one gated line, wired through the same
established pattern (env gate, snapshot+hash baseline of all touched
files, complete foreign-diff capture, isolated delta patch, dual-host
byte verification, host gates, no default change). **Parity risk: none**
(timing-only; numerics untouched). **Stop rules (stated up front):**

- plan+dispatch ≥ 70% of the fixed part ⇒ the cost is scheduling
  overhead → any fix lives in ggml-cpu (kernel/submodule) ⇒ **STOP,
  report as a parent-decision kernel proposal** (not self-authorized);
- plan+dispatch < 30% ⇒ fixed cost is node-loop/embedded ⇒ **no supported
  lever exists ⇒ record ceiling (fixed≈29% of decode unrecoverable
  host-side), stop the path**;
- 30–70% ⇒ inconclusive ⇒ one refinement only (split node-loop by op
  class) or stop.
  If a parent prefers zero submodule contact, the honest alternative is to
  record the ceiling now: every supported host-level change for this
  operation is already rejected or inert, and the 29% fixed bound is not
  addressable without kernel/scheduler territory.

## 2026-10-07 corrections + CPU op-class census + ONE narrow diagnostic proposal (docs-only, awaiting review)

### Corrections (superseding the previous read-only analysis)

- **26 ms intercept relabeled.** It is an _empirical fit constant_ of
  `cpu(call) ≈ 26 + 15.6×nt` over observed width classes (nt1 n=29,
  nt2-4 n=14) — **not a measured scheduler operation and not a proven
  recoverable budget.** Graph-width changes may alter op composition,
  cache behavior and thread behavior, so the derived "≈29% of decode =
  fixed part" is a **fit-based upper estimate, unverified as
  recoverable**; all downstream recovery arithmetic inherits that label.
- **rev107 placement remains INCONCLUSIVE.** Identical filtered
  `(index, input-count)` split headers are **not proof of inertness**
  (input names were dropped by the log filter; all counts were 0, a weak
  signal). The "structurally inert, exactly as rev107 showed" wording is
  **retracted**.
- **"Parity risk: none" for timers retracted.** A timing-only change is
  _expected_ numerics-neutral but must be validated empirically (full-ID
  OFF vs ON) like any change; timing-sensitive races make non-zero risk
  possible. Risk: to-be-measured, not zero.
- **"Lossy KV cannot pass exact IDs by construction" retracted** as
  overbroad. The empirical record stands on its own: kvq8 produced an
  **observed full-ID DIFF** (rejected on data); future lossy candidates
  must be tested empirically, not pre-judged.
- Empirical rejects retained unchanged: kvq8 (observed DIFF), depth ≤
  +1.1%, thread cap t3 (+4.58% whole), bounded spin (rev101), reuse
  (+2% marginal, already active), offload_kqv (R3 no benefit, R1
  inconclusive), flash already default-on, persistent pools already
  attached, `LLAMA_ATTN_ROT_DISABLE` no-op on F16.

### CPU op-class census (existing sources only)

- **OBSERVED** — `bench/results/diverge-20261006/rev82-diverge-result.csv`
  (cb_eval capture of target B3/B4 block decodes on the MTP ctx; name/op
  per node): 156 captured nodes (index range 0-238; **token-axis filtered
  → partial census**, plumbing/weight-only nodes excluded by capture
  rule). Histogram: VIEW 34, **MUL_MAT 26**, RESHAPE 24, RMS_NORM 17,
  MUL 14, UNARY 9, ADD 9, SCALE 6, **GLU 6**, TRANSPOSE 3, **SSM_CONV 3**,
  CPY 3, GET_ROWS 1, ROPE 1. Names show the MTP/recurrent block
  (`conv_states_*`, `linear_attn_qkv_mixed`, `q/k/v_conv_predelta`,
  `a_softplus`, `alpha`, attn-keyword rows 22, norm 19, ffn 12).
- **INFERRED (source)** — KV→attention-output chain pinned to CPU
  (`llama.cpp/src/llama-graph.cpp:3521`); d3d12 implements only
  MUL_MAT/view ops (`src/bridge/ggml_d3d12.cpp` op switches ~977/1202)
  → everything else executes on the CPU backend; MTP/nextn additions
  (`llama-graph.cpp` `t_h_nextn`/`n_layer_nextn` ~1888/2027;
  `llama.cpp/src/llama-memory-recurrent.cpp` state ops).
- Labels used below: class shares derived from the observed census are
  **count-based inference**, not timing measurements.

### ONE narrow corrected proposal (new instrumentation — PROPOSE ONLY, await review)

Two-tier, default-OFF (env gate; absent ⇒ zero clocks, zero output),
temporary, isolated delta, same snapshot/hash/foreign-diff/dual-host
rollback discipline. Files: `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp`
and `ggml-cpu.c` (both pristine today → status returns to exactly the
3 foreign files after rollback; both hosts verified byte-for-byte).

**Exact boundaries**

- Tier 1 — `ggml_backend_cpu_graph_compute` (ggml-cpu.cpp:170-190),
  three scopes, sequential and mutually exclusive on the decode thread
  (no overlap by construction):
  - **P (plan):** around `ggml_graph_plan(...)`;
  - **A (alloc):** around the `work_data` realloc block;
  - **C (compute):** around `ggml_graph_compute(cgraph, &cplan)` — the
    whole compute, parent of all Tier-2 values.
    Metric: **exclusive wall time** (caller thread). P+A+C partition the
    function exactly; residual = measurement granularity only.
- Tier 2 (inside C, only if Tier 1 shows C dominant):
  - **K (kickoff):** around `ggml_graph_compute_kickoff(...)` at
    ggml-cpu.c ~3565 — exclusive wall on the caller thread;
  - **W (worker node-loop):** in `ggml_graph_compute_thread`
    (ggml-cpu.c:3216-3290), per-thread **span** (compute-start → final
    barrier/return) plus per-thread **thread-CPU** over the same span;
    report `Σ spans` (wall, overlapping across threads) and `Σ CPU`
    (summed-thread CPU) separately.
  - **Class buckets (minimal op-class coverage):** one gated
    accumulation around the `ggml_compute_forward` call (~3246) with
    exactly four buckets chosen from the census: `MUL_MAT`,
    `STATE/SSM (SSM_CONV+SCALE+GLU+CONV-state)`, `NORM/ROPE
(RMS_NORM+ROPE)`, `OTHER elementwise/plumbing` — summed thread-CPU
    per bucket, one log line per graph compute. No per-node logging.

**Exclusive wall vs summed-thread CPU; overlap/barrier semantics**

- Tier-1 wall is authoritative and additive (P+A+C, single thread).
- Inside C, threads overlap arbitrarily: `Σ CPU` may exceed C (same
  contract label as process-CPU: aggregates threads, may exceed wall).
- Each worker span **includes** its in-loop `ggml_barrier` waits
  (c:3260 between nodes, final barrier ~3277): a spin barrier shows as
  CPU≈span for that thread; a blocking barrier as span≫CPU. Therefore
  `span−CPU` per thread ≈ blocked/sleep time, and `CPU` = node execution
  - any barrier spin. Node-exec vs in-loop-barrier is **not separable at
    these sites** (per-node timing deliberately excluded — overhead and
    scope creep); the decision table below treats that ambiguity
    explicitly.
- Double-count rule: K and W and class buckets are **children of C** and
  are never added to Tier-1 sums.

**Overhead / coverage / rollback cost**

- Overhead: Tier 1 = 3 clock pairs per CPU graph compute (~150 ns);
  Tier 2 = 2 thread-clocks per thread per compute + 2 per node
  (`CLOCK_THREAD_CPUTIME_ID`/steady, vDSO ~25 ns × ~250 nodes ×
  threads) ≈ tens of µs per call vs 41-88 ms cpu → <0.05%; one gated log
  line per call.
- Coverage: 100% of CPU-backend graph computes while the env is on
  (target classic + verify + draft ctx; analysis buckets by ctx as
  before); default OFF = zero clocks/zero lines (byte-identical
  behavior).
- Rollback cost: 2 files, snapshots+hashes captured pre-edit, full
  foreign-diff (3 files) recorded before/after, isolated
  `diag-only` patch, dual-host hash+diff verification, host 338-gate and
  formatters; IDs must be validated empirically OFF vs ON (parity risk
  to-be-measured).

**Decision table (what each measurement enables)**

- **P ≥30% of cpu-eval wall** → plan/replan is the target → propose
  ggml plan-caching/reuse change as a **parent-decision** submodule
  proposal. **P <10%** → plan ruled out (ceiling entry, no action).
- **A ≥10%** → per-call work-buffer growth → propose buffer-retention
  change (parent). Expected small.
- **C dominates; K ≥10%** → kickoff/dispatch cost → threadpool
  dispatch proposal (parent).
- **C dominates; K small; ΣCPU ≈ n_threads×C** → all threads CPU-saturated
  wall-to-wall ⇒ barriers are spin (included in CPU) and node-exec vs
  barrier **cannot be split at these sites** → **STOP** with the
  ceiling noted (no deeper instrumentation without a new review).
- **C dominates; ΣCPU clearly < n_threads×C** → class buckets decide the
  next proposal: `MUL_MAT` bucket dominant → QK/PV-family kernel proposal
  (parent); `STATE/SSM` dominant → recurrent-state kernel/placement
  proposal (parent; offload stays inconclusive); `NORM/ROPE` or `OTHER`
  dominant → node-count/fusion proposal on the existing fuse path
  (parent).
- Universal stop rules: any component <10% of cpu-eval → ceiling entry,
  no action; every actionable fix inside ggml-cpu/ggml is **parent
  territory — stop and report, never self-authorized**; ≥10% product bar
  arithmetic unchanged and still applies to any future proposal.
- The fit-intercept caveat applies: decisions key off **measured P/A/C/K/
  ΣCPU shares**, not off the26 ms fit.

No build, no instrumentation, no benchmark, no promotion in this step;
plan004 is the only artifact changed.

## 2026-10-07 rev108/109 pre-evidence review fixes (recorded before any evidence run)

The completed rev108 build (exit=0, sha `16C3776C…`) is left untouched
(healthy link not interrupted) but is **not evidence-bearing**: the
review blockers were fixed in the source before any run, requiring a new
build. Fixes applied:

1. **Provenance:** both `[diag] cpu` (cpp, 0-based id) and
   `[diag] cpuexec` (C, 1-based id) now emit **`g=%p` (cgraph pointer) +
   `sig=%llu` + `ov=%d`**; pairing rule = adjacency with **equal `g` and
   no interleaving** (explicit one-to-one nesting validation; ids are
   diagnostic only and never used for duration/order matching —
   nearest-duration joins remain forbidden). Mismatch ⇒ cell invalid.
2. **Concurrency:** global `d_*` counters are protected by a
   serialized-entry guard with **fail-fast sticky overlap flag `ov`** on
   both lines (normal scheduling unchanged; any concurrent CPU
   graph compute sets `ov=1` on all lines ⇒ evidence invalid). Worker
   buckets/barriers are caller-thread-only (`ith==0`), single-writer per
   call.
3. **Gate init:** lazy `static int` replaced by `ggml_diag_ensure_init()`
   called only from caller threads **before any kickoff** (workers read
   the gate afterwards, ordered by the kickoff atomics); exported via C
   linkage at file scope (block-scope `extern "C"` is ill-formed — found
   by the host compiler and fixed).
4. **Census identity:** `cpu-census` lines carry `g=%p sig=%llu
nodes/leafs trunc=%d`; the `cpu` line carries the same `sig` —
   repeated/interleaved topologies join by explicit signature, not
   order; `trunc` flags any ops-buffer truncation.
5. **Preservation confirmed:** all five touched files snapshotted with
   SHA-256 (`baseline-hashes-r108.txt`), complete 3-file foreign diff
   captured (`baseline-submodule-r108.diff`, `42f64036…`, pre-captured on
   **both** .157 and .193 with per-file hash equality), and `.193`'s
   stray okqv edit in `uwp/inference-bridge.cpp` was restored to baseline
   (hash `56e85ced…`) before this cycle. No kernel optimization.

## 2026-10-07 rev108 revised minimal CPU diagnostic — design accepted pre-build (constraints applied)

Supersedes the timing/API/overhead details of the previous proposal (its
invented `~150 ns / tens of µs / <0.05%` overhead figures are **deleted —
overhead will be measured, ON vs OFF**). Feasibility check: narrow
truthful coverage IS implementable with the constraints below (gaps
named, not hidden) — **no blocker**; proceeding to implementation only
after this section.

### Instrumentation rules (approved constraints, binding)

- **Clocks: `ggml_time_us()` only** (QPC on Xbox, already used
  throughout the pin). **No `CLOCK_THREAD_CPUTIME_ID`, no new per-thread
  CPU API, no summed-thread-CPU metrics anywhere.** Consequently no
  saturation-based inference: _all-thread saturation would not prove
  spinning_, and none is claimed.
- **Caller-thread between-node barriers are a separate reported scope**
  and labeled exactly as: _observed wall wait of the caller thread_ —
  never total CPU, never spin attribution.
- **No summing of nested/overlapping times**; parents and children
  reported separately; **P+A+C is NOT claimed to partition the function
  exactly** — an explicit measured residual closes each accounting.
- Default OFF (env `LLAMA_UBATCH_DIAG` via the existing `ggmlprof`
  `ubdiag` value): absent ⇒ zero clock reads, zero output, byte-identical
  behavior.

### Exact boundaries (measured wall, exclusive unless noted)

Tier 1 — `ggml_backend_cpu_graph_compute` (ggml-cpu.cpp:170-190):

- **T** = whole function; **P** = around `ggml_graph_plan`; **A** = around
  the `work_data` realloc block; **C** = around `ggml_graph_compute`.
  All four are caller-thread wall; `resid1 = T − (P+A+C)` reported every
  call (covers cplan field wiring/setup), gate G1 below.
  Tier 2 — inside **C** (never added to T/P/A/C sums):
- **K** = around `ggml_graph_compute_kickoff` (ggml-cpu.c ~3565);
- **F (4 class buckets)** = caller-thread (`ith == 0`) wall around
  `ggml_cpu_try_fuse_ops` (a fused span is ONE timed unit covering its
  consumed nodes) and around `ggml_compute_forward` otherwise — classes
  from the census: `MUL_MAT`, `STATE/SSM` (SSM_CONV/SCALE/GLU/conv-state),
  `NORM/ROPE`, `OTHER`;
- **B** = caller-thread (`ith == 0`) cumulative wall inside the in-loop
  `ggml_barrier` (c:3260) plus the final barrier — reported as
  **caller-thread between-node barrier wall (observed wait)**;
- `resid2 = C − (K + ΣF + B)` reported every call (loop/skip/return
  overhead), gate G2.
- Gap marked explicitly: other threads' node-slice execution is NOT
  instrumented (only the caller's); their waits appear only through the
  caller's B.

### Runtime CPU-placement census (exact package, not the rev82 proxy)

At the Tier-1 site (gated), emit a census line **once per distinct graph
topology signature** (`n_nodes`, plus a cheap hash of the op histogram):
full `cgraph->nodes` op histogram + executed-unit accounting:
`nodes_total = individually_timed + fused_consumed + skipped_nop +
skipped_nocompute` (integer-exact, gate G3), with fused spans counted.
Covers target classic/verify/draft/prefill topologies as they actually
occur on this package. Signature collisions (e.g., classic vs verify
sharing one topology) are declared as a labeling gap, not papered over.

### Provenance (exact ctx/call)

Host `VRCALL/VREND` markers (decode_loop.h, profile-gated) reopen exact
window joins: lines inside `VRCALL(k)..VREND(k)` = verify calls with the
window's ctx+seq; out-of-window lines are grouped by topology signature
(draft graphs are topologically distinct; classic-vs-verify share a
signature — declared). Per-line: monotonic call id + `n_nodes`.

### Predeclared gates (per call, measured — no invented budgets)

- **G1** `|resid1| ≤ 5% of T` — else coverage incomplete for that call.
- **G2** `|resid2| ≤ 5% of C`.
- **G3** executed-unit integer accounting exact (census equation).
- **G4** provenance: every verify window has ≥1 `cpu` line with matching
  call sequence; unmatched ⇒ cell invalid.
- Failure of G1/G2/G3/G4 ⇒ no conclusions from the affected cells
  (same acceptance discipline as the residual gate; no pooling).

### Validation plan (bounded, one package)

Identity (sha 4/4 + PFN `_1.6.0.108_`), engagement (census/cpu/exec lines
present ON, absent OFF), **measured** ON/OFF overhead (interleaved
profile-OFF pair, medians + spread), **same-package full-ID parity ON/OFF**
(fresh sequential ref + d4 arms, full sha256), conservation per G1-G4 on
one profile-ON provenance cell, then rollback: snapshots + hashes +
complete 3-file foreign diff captured pre-edit; reverse ONLY this
delta (5 files: ggml-cpu.cpp, ggml-cpu.c, llama_gpu.h narrow
`[diag] cpu` filter, inference-bridge `ubdiag` value, decode_loop
markers); verify baseline bytes and foreign diff **independently on
.157 and .193**; host 338-gate + formatters; knobs restored. No
optimization, no default promotion; every actionable fix found inside
ggml remains parent-decision territory.

## 2026-10-07 rev109 cpu109 evidence — measured operation classes (diagnostic-only; reverted)

Identity: SHA gate PASS (build-log == local msix == builder msix =
`cec17cca…cb14c`); PFN `_1.6.0.109_`. Fresh device log (history preserved
in earlier evidence dirs). Collection: `cpu109-20261007-*` cells (ref,
p1off/p1on, p2off/p2on, probe, provenance).

### Gates (as predeclared; failures reported, not pooled)

- **OFF-clean: PASS** — 10 knob-absent runs, 0 `[diag]` lines.
- **Pairing/provenance: PASS** — 78,312/78,312 pairs matched by
  adjacency `exec → [census] → cpu` with equal `g`, `exec.id = cpu.id+1`,
  equal `nodes`, census `sig` consistent; zero violations (no duration
  matching anywhere). **ov = 0 on every line (serialized entry held).**
- **G3 executed-unit coverage: PASS** (0 fails —
  `x + fused_nodes + skip_nop + skip_nocompute == n_nodes` exactly).
- **G4 windows: PASS** (42/42 VRCALL windows contain CPU lines).
- **G1: FAIL on 3 of 78,312 calls** (ids 2292/2312/6146, nodes 3-4,
  t≈40-60 µs — wiring residual is large only at µs-graph scale); median
  resid1 = 0.000%. Those calls excluded.
- **G2: FAIL for 77,145/77,784 small splits** (median resid2 44% —
  mechanism candidates, NOT proof: worker wake/join latency outside the
  caller-timed regions, failed-fuse attempt time (intended in resid2),
  entry affinity); **big graphs (nodes ≥200): 528/528 PASS, resid2 med
  0.98%.** Per the no-pooling rule, conclusions are restricted to
  gate-passing calls: 528 big + 639 small.

### Measured operation classes (caller-observed wall, % of C, gate-passing)

- **Big CPU graphs (n=528, the target/prefill workhorses; t med 18.9 ms,
  c med 18.9 ms): MUL_MAT forward regions 68.42% (mean 70.89),
  OTHER elementwise/plumbing 24.68%, caller barrier wall 3.21%,
  STATE/SSM 1.77%, NORM/ROPE 0.48%, resid2 0.98%.**
- Small passing (n=639): MUL_MAT 43.43%, OTHER 45.41%, barrier 3.23%.
- Tier-1 across all graphs: t med 0.056 ms, p med 0.001, a med 0.000,
  c med 0.054 — plan/alloc negligible on the decode path (measured, not
  assumed).
- Semantics: F/B are **caller-thread wall** (forward regions include
  waiting on other threads where the op synchronizes); no thread-CPU or
  spin claims; `b` is observed caller wait only.
- **Runtime census (exact package):** 78,312 census emissions, **10
  distinct topology signatures, all `trunc=0`** — e.g. the 437-node big
  signature: `MUL_MAT:49, VIEW:111, RESHAPE:59, RMS_NORM:30, CPY:41,
MUL:26, SCALE:21, GET_ROWS:23, ADD:17, SSM_CONV:5, GATED_DELTA_NET:5,
GLU:11, SOFT_MAX:1, ROPE:2, …` (observed runtime, not the rev82 proxy).
- **Measured overhead (not invented):** diag ON vs OFF, profile OFF,
  same package: decode **+472.3 ms (+12.24%)** (OFF med 3857.9, ON med
  4330.2; within-arm spreads ≤128 ms), ttft +0.91%. Overhead is large ⇒
  the diagnostic is attribution-only, never for product timing.
- **Full-ID parity: ALL EXACT** — 7 dumps × 3 run indices, one full
  sha256 each (ref, OFF, ON, probe, provenance).

### What next decision the evidence supports

CPU-backend **MUL_MAT forward regions ≈ 68% of big-graph C** is now the
measured expensive class (caller-observed wall; the exact op role —
attention QK/PV vs pinned chain vs refused q8_0 weights vs nextn
linears — is NOT yet attributed). Supported host knobs for CPU matmul
are exhausted (threads rejected on wall, placement inconclusive, kvq8
rejected on IDs). Decision for review: (a) fund one more narrow
attribution step — split CPU MUL_MAT by operand role (needs new
instrumentation, ask first), or (b) record the ceiling for this class
and stop. No optimization, no default promotion.

### Rollback (completed, independently verified)

All five files restored from `snapshot-r108/`: **.157** — 5/5 hashes
OK, foreign diff byte-identical, status = exactly the 3 foreign files,
zero diag traces, 338/338 + format + coherence; **.193** — 5/5 hashes
OK, `git diff HEAD` of the foreign files == baseline, zero diag traces,
status = 3 foreign + standard `apply-uwp-patches.sh` targets. Device
knobs (twocol/scope/ggmlprof) deleted; `cpu-diag-r108.patch` +
`cpu-diag-r108b.patch` (sha `082916ec…`), fresh log, `analysis.txt`,
`sha-gate.txt` preserved under `cpu109-20261007-evidence/`. The
completed-but-superseded rev108 build (`16C3776C…`) was never used for
evidence.

## 2026-10-07 Completion plan (parent-delivered, persisted) — functional MVP vs optional speed

**Scope: finish Qwen4B-MTP (Qwen3.5-4B-MTP-Q4_K_M.gguf, ctx2048) on
Xbox. No new general profiling, no kernel port, no extra build until this
plan is reviewed; the already-approved full-layer smoke runs first.**

### Milestone M0 — approved full-layer smoke (this session, supported flags only)

g28 control vs full-layer request **`--gpu-layers 34`** (34 = the
observed total from the installed log's `offloaded 28/34`; the raw
`-1`/≤0 path is REJECTED: our host `apply_gguf_gpu_layers` maps
`requested <= 0` to CPU-only, llama.cpp — llama-model.cpp:2330 — but our
guard runs first) on the **same installed package**
(rev109 `cec17cca…`), np16, after the recorded memory-budget precheck
(2,834,975,040 B model, sha `38742092…8630`; g28 gpu_mem 2273/4147,
headroom ≈1874 MB vs ≈500 MB estimated for the 6 unoffloaded layers).
Records: effective offload line (`offloaded X/34 layers to GPU`),
CPU/GPU buffers and fallbacks (refusal inventory), full token IDs A vs B,
observed memory with OOM/invalid-load fail-fast. No speed claims from the
smoke; restore `bench_gpu_layers.txt=28` and all knobs afterwards. **Max
proven safe placement = whatever the smoke actually proves (not assumed
28, not assumed full).**

### Milestone M1 — functional MVP (the acceptance gate; profile-OFF reporting only)

Configuration: fixed Qwen4B-MTP Q4_K_M, ctx2048, **actual
n_batch=64 / n_ubatch=64 verified from effective-config logs — never
inferred from a knob name.**

- **64/64 path:** supported session config first — the session/INI path
  already accepts `n_batch` (`apply_llama_ini_session`,
  `inference-bridge.cpp` llama.ini `n_batch`) and `open_session(n_batch,…)`
  is used by the parity/termgate harnesses; the bench runner's `--ubatch`
  sets only `n_ubatch`. Verification = the path's own effective-config
  print (bench path: `[xllama] prefill batch override: n_batch=… n_ubatch=…`;
  session/scenario prints like `MTP_SCENARIO … n_batch=…`). **Exact gap
  if bench-path 64/64 is required:** the runner has no `--batch` flag and
  `run_inference` reads no `bench_n_batch.txt` → a narrow 2-line adapter
  (runner writes `bench_n_batch.txt`, bridge reads it into
  `params.n_batch`, mirroring `bench_ubatch.txt`) — **describe for review
  before any build**; prefer the session path meanwhile.
- **Native reuse audit (honest):** the native llama MTP graph + graph
  reuse is already the executed path (MTP ctx, `can_reuse` evidence);
  `mtp_draft.cpp` is a self-contained bridge adapter (only xllama
  headers) mirroring `common_speculative` orchestration. `common/
speculative.cpp` (6945 lines) is **not compiled into UWP** (no vcxproj
  entries) and pulls `common.h/log/sampling/ngram/dflash-*` — NOT a
  drop-in; rule: audit minimal common-API reuse only where the UWP build
  stays straightforward, otherwise **keep the parity-tested adapter — no
  new MTP implementation, no wholesale rewrite**.
- **Acceptance checks:** model/package identity (gguf sha + package
  sha); real proposals AND accepted tokens > 0 (MTP_STATS drafted/
  spec_accept); full IDs vs same-config sequential reference;
  stop/EOG/cancel followed by a subsequent prompt (integrity — termgate
  suite pattern); memory safe (budget respected, peaks recorded); saved
  usable configuration + instructions (knobs/flags/INI/hashes).
- **Reporting:** profile-OFF TPS/TTFT reported separately (honest
  numbers, separate from any instrumented cells).
- **≥10% gain: OPTIONAL, not an MVP blocker** — tracked only after MVP
  acceptance, via the standing repeated paired + parity gate.

### Rev109 attribution correction (persisted with this plan)

Gate-passing small calls are **636, not 639** (639 was G2-only; 3 of
them fail G1 and are excluded). Conclusions therefore cover **528 big +
636 small passing calls only; ≈40% of CPU wall (the excluded failing
calls) carries no attribution**, and the measured classes stand **under
+12.24% instrumentation perturbation**, selected-big-graph only — never a
global or uninstrumented “68% CPU” claim. The rev109 diagnostic remains
rolled back.

## 2026-10-07 rev111 functional MVP — acceptance result (profile-OFF)

### Prebuild transfer invariant (persisted — the rev105-class bug, never again)

Before ANY expensive build, all in one fail-fast shell step with
**explicit absolute cwd**: (1) transfer commands must exit 0; (2) re-hash
every adapter-changed source file **locally and on the builder** and
require equality; (3) `findstr` the exact new fields on the builder
(e.g. `bench_n_batch`, `params.n_batch = bench_n_batch`, `-b%d`);
**a build is forbidden after any failed transfer or hash mismatch.**
Runner/host-only files (not built on .193) are excluded from the builder
hash set but must be shellcheck/format-gated locally.

### Rev110 invalid / not delivered

The rev110 build (`4D1E3419…`) is **INVALID and was never deployed**: its
`uwp/inference-bridge.cpp` transfer failed silently (missing workdir —
same class as rev105) and .193 built the previous baseline sources
instead of the adapter. Recorded here so no rev110 artifact is ever
cited as delivered. The subsequent correct transfer passed
(`TRANSFER-OK`, local == builder `ac2c8d87…`, 7 adapter field lines on
the builder) and produced **rev111 `6B3E96D8…`** — the only delivered
MVP package.

### Retrospective rev111 checks (all PASS, run with explicit cwd + fail-fast)

local==builder sha `ac2c8d87…` for the adapter file; 7/7 adapter fields
present on builder == local; installed PFN `…_1.6.0.111_`; effective log
**`prefill batch override: n_batch=64 n_ubatch=64`** (×6 across ref+MTP)
and **`offloaded 34/34 layers to GPU`**; runner host tags
`-u64-b64-g34` (+`-mtp4`) in CSVs; package sha local==build-log
`6B3E96D8…`; runner `verify_expected_package` passed under
`XLLAMA_MSIX_SHA256`.

### Acceptance results (config: g34, n_batch64/ubatch64, ctx2048, mtp4,

pmin75, twocol auto scope2, greedy seed1, package6B3E96D8, model sha
38742092… — full instructions in `bench/results/mvp-20261007-INSTRUCTIONS.md`)

- **Fresh SAME-config sequential reference** (`mvp-20261007-ref`, MTP off,
  runs3) vs **MTP-ON** (`mvp-20261007-mtp`): **full-ID parity PASS —
  run1/2/3 byte-exact (full sha256)**. No gate weakening: single
  comparison, same config, same package.
- **Real proposals AND accepted > 0:** `drafted=27 spec_accept=25` on
  every measured run (warmup rows drafted=0 are the seq arm); MTP_STATS
  rounds=15 decodes=103.
- **Termination integrity:** termgate at this config — **8/8 ok=1**
  (cancel@6, cancel@13, stop@probe, stop@run with stop_branch=spec
  round=4, eog@0-3); cross-arm dumps **24/24 byte-equal** across
  seq/ref/cand including `termgate-cancel*-resume.*` (subsequent-prompt
  integrity).
- **Memory:** ref gpu 2614/4147 peak 3461; MTP gpu 2678/4147 peak 3776 —
  within budget and the ~4.3 GB precedent.
- **M0/placement (independently verified):** `offloaded 34/34`; cross-
  placement IDs DIFFER (g28 vs g34) — recorded, NOT a pass against g28
  and NOT MTP evidence; g34 is now the single pinned config for all MVP
  IDs. np16 smoke TPS (g28 16.75, g34 19.24) = **exploratory only**.
- **Baseline hash-set exception (expected, not drift):** of the
  `baseline-hashes-r104` set only `uwp/inference-bridge.cpp` no longer
  matches — that is the authorized, builder-verified MVP adapter
  (now `ac2c8d87…` locally == builder, 7 `bench_n_batch` field lines),
  delivered in rev111 and intentionally kept. Every other baseline file
  is byte-identical, and the submodule foreign diff remains exactly the
  3 files (`cmp` vs baseline `42f64036…`).
- **Guard record:** `split_mtp_weights=false` → `info.managed=false` →
  the historical CPU/CUDA host-backing guard is **not active** on this
  path; no guard fix required. Foreign 3-file diff/hash independently
  intact throughout.
- **Profile-OFF speed (reported separately; no optimization claim):**
  sequential decode **19.66 tok/s** med / TTFT **1347 ms**; MTP decode
  **19.86 tok/s** / TTFT **1739 ms** (≈+1.0% decode, +~392 ms TTFT from
  the draft model on the prompt path). ≥10% remains OPTIONAL, not an MVP
  blocker.

## 2026-10-07 delivery cleanup — durable artifacts + normal-session status (final)

- **Durable artifacts** (no /tmp dependence): `bench/results/`
  `mvp-20261007-*.csv/.tokens`, `mvp-20261007-INSTRUCTIONS.md` (rewritten:
  self-contained, durable termgate driver path, exact two-part scope2
  command, precise EOG wording — **exactly 1 natural EOG (tok=248044,
  spec-reject) + 3 capped completions; never all four as natural eos**),
  and `mvp-20261007-evidence/` (device logs ref/mtp/termgate/api,
  `termgate-result.csv`, durable `termgate-run.sh`, `termgate-arms/` 72
  dumps, `llama.ini.backup-pre-mvp`, `api-session.log`). Secret scan
  over artifacts: clean (technical content only).
- **Package/release validation (read-only):** installed PFN
  `…_1.6.0.111_`; builder msix `C:\Users\hjotha\build\xllama-q5k\
xllama-q5k-fixed.msix` sha == local == build-log `6B3E96D8…`; local↔
  builder `uwp/inference-bridge.cpp` sha `ac2c8d87…` equal with 7 adapter
  field lines; submodule foreign diff == baseline (`cmp`); rev110 remains
  invalid/not-delivered.
- **Normal-session configuration (existing settings only):** backed up
  device `llama.ini` then set proven values `n_gpu_layers=34 kv_q8=0
n_ctx=2048 n_batch=64 n_ubatch=64` (unrelated keys preserved; upload
  verified by re-fetch diff). Validated via the sanctioned API path:
  `validate-api.sh chat` **PASS** with log evidence `offloaded 34/34`,
  `gguf gpu layers: 34 on D3D12`, session `mtp=0` lines. Effective
  session batch has **no print** (only bench prints it) — observability
  gap recorded.
- **Exact minimal gaps (report only — nothing built):** (1) MTP not
  selectable in normal flow (`sp.mtp` never set by `MainPage.cpp` or
  `api-server.cpp`; default false); (2) scope2/twocol not read in the
  normal session-construction path; (3) `MainPage.cpp` does not call
  `apply_llama_ini_session` (API path does). **No GUI-ready claim: the
  GUI was not launched/verified this cycle.**
- **Transient cleanup (safe):** deleted `d3d12twocol.txt`,
  `cpurepackforcegemv.txt` (recreated by documented commands), and the
  termgate trigger files; diagnostics (`ggmlprof/spinwait/offloadkqv`)
  confirmed absent; `llama.ini` (proven) and runner-managed
  `bench_gpu_layers/n_batch/ubatch` (34/64/64) left in place as the
  safe current state; `api.flag` (user-facing feature) left untouched.
- Optional ≥10% optimization: **stopped per instruction**; MVP delivery
  is the endpoint of this plan's required scope.

## 2026-10-07 rev114: normal-session MTP delivery — ACCEPTANCE PASS

Prebuild (persisted invariant, applied again): absolute cwd, transfer
exit 0, local==builder hashes for all 7 changed files, field/symbol
presence greps — ALL PASS before launching the build. Local preflight:
shared-file compile + 338/338 + format + coherence + foreign diff
byte-identical. rev112 = built but superseded (blocked parser design,
never accepted); rev113 = compile-failed (`apply_startup_profile`
namespace), no package, nothing killed.

### Design as delivered (minimal, no broad architecture)

- Parser `apply_llama_ini_session` is **populate-only**: exact canonical
  basename (`== "qwen35-4b-mtp"`, no substring), `mtp=0` explicit OFF,
  depth 1-16 / `mtp_pmin` 1-100 validated with distinct reject logs,
  unrelated models never receive MTP keys.
- **Immutable startup profile** (`apply_startup_profile` at
  `App::OnLaunched`, single-threaded, before any load): scope2
  files → applied once, restart-to-change; when active, **only
  qwen35-4b-mtp may load** — other models fail `Session::create` with
  `sp.config_reject` (populate-only mechanism; no global mutation in the
  parser). Changed-file check is **log-only** ("REJECTED, startup profile
  stays") and never claims new knobs applied; `session config:` logs the
  actual effective sp.
- **SessionHub reuse identity** = model + gpu_layers + mtp + depth +
  pmin + **n_ctx + n_batch + n_ubatch + n_threads + kv_q8** → forced
  recreate on any change; profile omitted by design (startup-immutable).

### Test matrix (device, rev114 `9D522C88`, all PASS)

- **Startup:** `twocol='auto' repack='2' (immutable...restart...)`.
- **Normal-session MTP:** `session config: model=qwen35-4b-mtp mtp=1
depth=4 pmin=0.75 n_ctx=2048 n_batch=64 n_ubatch=64 gpu=34 kv_q8=0`;
  API accepted>0 — **drafted=136/spec_accept=118** (long reply) and
  4/1, 1/1 on short chats; chat PASS; concurrent 200/200.
- **Model-off:** `mtp 4→0` in ini → hub recreate → `session config`
  shows `mtp=0` (explicit); restore → `mtp=1`.
- **Wrong model:** `model=qwen35-4b` under active profile → **HTTP 500**
  with the exact scope2 restriction reason (also exact-basename reject
  path in parser logs); unrelated model in DEFAULT mode loads with
  `mtp=0` (no hidden global impact).
- **Config reload/identity:** `n_batch 64→128` → recreate → config line
  `n_batch=128`; restored to proven 64.
- **Profile hot-change:** file edit mid-process →
  `ini: profile files changed after startup - REJECTED, startup profile
stays (restart required)`; chat still 200; file restored to `auto`.
- **Stop/cancel/next-request:** client abort (HTTP 000) → next request
  **HTTP 200** with content; termgate **8/8 ok=1** on rev114 (cancel@6/
  @13, stop@probe, stop@run, eog 1 natural eos + 3 cap).
- **Same-config sequential full IDs on rev114:** `mvp114-20261007`
  ref vs mtp run1/2/3 **EXACT**; drafted=27/spec_accept=25 each
  measured run; effective `offloaded 34/34` + `n_batch=64 n_ubatch=64`
  verified in-log; host tags `-u64-b64[-mtp4]-g34`.

### Evidence classification (artifact corrected)

- **Benchmark PASS:** `mvp-20261007` + `mvp114-20261007` cells
  (rev111/rev114).
- The earlier `validate-api chat` PASS on rev111 was an **MTP-OFF API
  smoke — superseded**.
- **API MTP-ON acceptance:** rev114 validate-api chat + curl matrix
  (logs `r114-a/b/bench.log`, `api-wrong-model-reject-rev114.json`,
  `termgate-result-rev114.csv` in the evidence folder).
- Foreign 3-file diff intact throughout; no kernel/profiling/rewrite;
  no promotion (≥10% remains optional).

- Evidence: `bench/results/vsplit-20261006-r102-*.csv` + `.run1..3.tokens`,
  `/tmp/xllama-rev102/xllama-rev102{,b,c,d}.log` (knob lines, VROUND
  reuse, split headers, #171 engagement), rev102 build `exit=0` sha
  `F0B6929A`.

- Evidence: `bench/results/vsplit-20261006-r101-{ref101,p1def,p1b500,p2def,p2b500,cpudef,cpub500}.csv` +
  `.run1..run3.tokens`, `/tmp/xllama-rev101/xllama-rev101-{spin,cpu}.log`
  (knob state lines + VROUND cpu), rev101 build `exit=0` sha `DB2E3F5C`.
  Device baseline knobs (twocol/scope/spinwait) deleted after the runs; no
  defaults changed; no commits.
