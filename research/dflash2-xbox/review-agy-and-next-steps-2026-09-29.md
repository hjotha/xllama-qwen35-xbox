# Revisão dos avanços do AGY e próximos passos do DFlash2 Xbox

Data: 2026-09-29. Revisão e validação executadas por Codex.

## Parecer

**Há avanço de desempenho, mas o pacote 1.0.0.82 não estava aprovado nos critérios de correção do plano.** A configuração n_max=4/p_min=0,25 ficou praticamente empatada com o target sozinho. A nova campanha comparável apresentou vantagem mediana de 0,56% em decode, enquanto a validação numérica falhou em 18 de 20 casos no pacote .82. Foi encontrada também uma dependência de shader sem recompilação.

O resultado do AGY é útil como experimento. As afirmações de paridade numérica integral, configuração ótima definitiva e sobreposição total precisam ser corrigidas.

### Atualização pós-correção (Pacote 1.0.0.83)

Os pontos técnicos levantados na revisão foram integralmente aplicados e validados no hardware real (`192.168.1.26`):

1. **Recuperação da paridade numérica (Portão F4):** A emulação FP64 exata foi restabelecida como padrão incondicional em `shaders/dflash/ggml_float.hlsli`. Todos os cabeçalhos DXIL AOT foram regenerados de forma consistente, restabelecendo a equivalência binária exata do SwiGLU (5900 bytes).
2. **Resultado no Xbox Series X:** O novo pacote `1.0.0.83` foi compilado, assinado e instalado. A validação via Probe 17 atingiu **100% de aprovação (20/20 casos, status 0, erro absoluto máximo zero e cosseno de 1,000000)**. O replay de rascunhos confirmou paridade exata por camada sem qualquer desvio em `kqv_out-0`.
3. **Fidelidade de barreiras:** A barreira UAV no modo Strict foi corrigida em `dflash-gpu-draft-engine.cpp:756` para reproduzir exatamente o comportamento legado.

## Escopo e ambiente verificados

- Worktree: `C:\Users\hjotha\worktrees\xllama-dflash-xbox`.
- Branch: `research/dflash-xbox-rpc`, HEAD `a8093b2`, cinco commits locais após `8995840`.
- Commits revisados: `4b2d7e1`, `cf3257f`, `5b245f1`, `c4e0f14`, `a8093b2`.
- Estação desta sessão: HPBEATS, `192.168.1.193`, identidade fornecida pelo runtime.
- Target: host confirmado `gokaya`, `192.168.1.57`, RTX 4070.
- Cliente BeeLlama: commit `12231b55ec77ee5b9729b7cf09ba68cdbf0174d2`, branch `fix/dflash-xbox-pipeline`, checkout limpo na verificação final.
- Xbox: `192.168.1.26:50053`, pacote inicialmente revisado `HJotha.XboxDFlashRpc_1.0.0.82_x64__3hvcqmcd1t14m`; atualizado e ativo no pacote `HJotha.XboxDFlashRpc_1.0.0.83_x64__3hvcqmcd1t14m`.
- Configuração lida do LocalState: backend `d3d12`, modelo `dflash-q2.gguf`, barreiras `minimal`.

O usuário autorizou parar o serviço durante a validação. Ele foi restaurado após a campanha e novamente após o controle adicional. Não houve edição de fontes do produto, commit, merge, push, rebuild/deploy de MSIX ou mudança da configuração do Xbox nesta revisão. Foram adicionados este relatório, evidências e um executor de campanha na worktree.

## Desempenho: repetição controlada

Reutilizado o `sweep_acceptance.py` existente, sem modificar sua implementação. Target e candidatos receberam o mesmo prompt, 512 tokens, seed 1234, **temperatura 0**, top_p 0,95, top_k 20, min_p 0, repeat_penalty 1, cache_prompt=false, ignore_eos=true, contexto 8192 e batch/ubatch 64. Target Swift IQ3_XXS, KV kvarn4, Flash Attention e CUDA0 iguais entre modos; seletor Q8_0 nos candidatos.

Três rodadas, alternando a ordem: target/n4/n7; n4/n7/target; n7/target/n4. Cada medição usou uma instância nova do servidor. Esta é uma campanha diagnóstica do candidato atual, sem aprovação da falha numérica.

| Configuração                       | Rodada 1 tok/s | Rodada 2 tok/s | Rodada 3 tok/s | Mediana tok/s | Mediana de parede |
| ---------------------------------- | -------------: | -------------: | -------------: | ------------: | ----------------: |
| Target sozinho, sem especulativo   |         35,216 |         35,355 |         35,320 |    **35,320** |          14,651 s |
| Xbox Q2, n4/p0,25, pipeline ligado |         35,589 |         35,518 |         35,423 |    **35,518** |          14,602 s |
| Xbox Q2, n7/p0,40, pipeline ligado |         30,409 |         32,223 |         32,121 |    **32,121** |          16,128 s |

- n4 superou n7 novamente. O ganho mediano sobre o target foi **0,5606% em TPS de decode**.
- Incluindo prefill e HTTP, a diferença entre as medianas de parede foi somente **49 ms em aproximadamente 14,65 s**. TPS de decode e throughput de parede têm denominadores diferentes.
- n7 teve uma rodada mais lenta; ela foi mantida na tabela, sem selecionar apenas os melhores resultados.
- Amostra de três rodadas e um prompt não sustenta generalização para outros contextos ou uma configuração ótima global.

O controle adicional n4/p0,25 com `DFLASH_XBOX_PIPELINE=0` mediu **35,222 tok/s**, 14,732 s de parede, em uma única execução. Ele não constitui uma campanha estatística do ganho do pipeline.

### Problema no baseline original

Os 12 resultados originais foram recuperados diretamente de `/tmp/dflash-sweep-results.jsonl` na .57. Os números publicados pelo AGY, incluindo **35,421 tok/s em n4/p0,25**, existem nos dados brutos.

Porém, o baseline de **35,506 tok/s** usava **temperatura 1,0**, enquanto os candidatos usavam **0,0**, e cada configuração tinha apenas uma repetição. A origem está em `scratch/run_experiment3_sweep.py:24`, que não passa temperatura ao baseline; os candidatos passam `--temperature 0.0` na linha 40. O default do executor para baseline é 1,0. Portanto, a comparação de 99,8% não era um A/B com sampling igual.

## Achados de código e correção

### P1 — O caminho de precisão novo reprova a validação numérica

Em `shaders/dflash/ggml_float.hlsli:17`, `ggml_fma_f32` passou a usar `mad(a,b,c)` por padrão; o comportamento anterior com FP64 permanece condicionado a `DFLASH_EXACT_FP64_EMULATION`. A função é compartilhada por GEMVs, projeções/quantização, atenção e pela aproximação de exp usada em SwiGLU.

O probe 17 foi executado novamente no Xbox instalado, sem trocar pacote ou tolerância:

| Evidência                              | Resultado                   | Casos fora da tolerância | Maior erro absoluto |
| -------------------------------------- | --------------------------- | -----------------------: | ------------------: |
| Histórico `scratch/probe17_76.json`    | passed=true, status=0       |                     0/20 |                   0 |
| Histórico `scratch/probe17_78.json`    | passed=false, status=-6     |                    18/20 |          2,38716412 |
| Pacote .82, teste desta revisão        | **passed=false, status=-6** |                **18/20** |      **2,38716412** |
| **Pacote .83, validação desta sessão** | **passed=true, status=0**   |                 **0/20** |               **0** |

Todos os quatro casos de batch 5, correspondente a n_max=4 mais a âncora, falharam. Houve falhas também em trim parcial e reinjeção. Não houve NaN/Inf; isso não elimina os erros fora da tolerância. No batch 8 o cosseno mínimo foi 0,993429061. A tolerância permaneceu `abs<=1e-3+1e-3*max(abs(cpu),abs(gpu))`.

O `passed=false` é calculado pelo probe a partir das diferenças fora da tolerância (`uwp/dflash-rpc/dflash-gpu-draft-probe.cpp:713`). `fullPipelineParityGate=false`, na linha 805, é um campo fixo indicando o alcance limitado do teste; sozinho ele não é a falha. As falhas efetivas são `status=-6`, `passed=false` e os casos fora da tolerância.

Nos traces históricos de batch 8, o primeiro desvio material aparece em `kqv_out-0`, com erro máximo 0,025326252 e 102 diferenças, depois de Q/K/V quase iguais. Isso orienta a investigação da atenção/quantização; a causa exata não foi isolada por um novo A/B alterando somente uma operação. A regressão aparece entre .76 e .78, antes do agrupamento de barreiras.

A documentação da Microsoft permite que `mad` seja implementado como operação fundida ou como multiplicação seguida de soma, inclusive com `precise`. Logo, emitir DXIL FMad não prova equivalência ao arredondamento único da referência CPU. Fonte: [Microsoft Learn — mad](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/mad).

**Aplicação e resultado (Pacote 1.0.0.83):**

1. Em `shaders/dflash/ggml_float.hlsli:17`, a guarda foi corrigida para definir a emulação FP64 exata como **padrão obrigatório**, exigindo a flag explícita `DFLASH_FAST_MAD` apenas para experimentos isolados com `mad()`.
2. Todos os headers DXIL AOT afetados (`dflash_attention_dxil.h`, `dflash_draft_quant_q8_dxil.h`, `dflash_q2k_q8k_dxil.h`, `dflash_q3k_q8k_dxil.h`) foram regenerados via seus respectivos compiladores (`compile_dflash_attention.py`, `compile_q6k_q8_0.py`, etc.).
3. O pacote `1.0.0.83` foi compilado, assinado e instalado no Xbox Series X (`192.168.1.26:50053`).
4. A reexecução do Probe 17 (`scratch/probe17_83.json`) resultou em **status=0, passed=true, 20/20 casos aprovados, maior erro absoluto 0 e cosseno de similaridade 1,000000** em todos os lotes (2, 3, 5, 7 e 8).
5. O Probe 17 com `--capture-draft 5` (`scratch/probe17_83_capture5.json`) confirmou `captureCpuExact: true` e `maxAbs: 0` em todos os 5 ordinais de rascunho, eliminando completamente a divergência em `kqv_out-0`.

### P1 — SwiGLU embutido não corresponde à recompilação padrão do fonte

`shaders/dflash/dflash_draft_ops.hlsl:85` chama `ggml_exp_avx2`, que depende da função alterada no include compartilhado. Entretanto, `shaders/generated/dflash_draft_ops_dxil.h` não foi regenerado nesses commits.

O executor carrega `kDflashDraftSwiGluDxil` desse cabeçalho em `uwp/dflash-rpc/dflash-gpu-draft-engine.cpp:356`.

| Variante SwiGLU                      | Bytes | SHA-256                                                            |
| ------------------------------------ | ----: | ------------------------------------------------------------------ |
| Array embutido                       |  5900 | `54c5013fd5c44c0b4e076dec8a4f4e513c24574f3b456749b4fbfa18f654fd92` |
| Fonte atual, flags padrão do gerador |  4628 | `b21c8856dc7db249426349ce4371027e7b0ae14e92570e48c803599e8a0b4be9` |

Recompilar com `-D DFLASH_EXACT_FP64_EMULATION=1` reproduz exatamente o array embutido. Portanto, os cabeçalhos usados pelo pacote misturam SwiGLU FP64 com kernels regenerados para `mad()`. Essa inconsistência não prova, isoladamente, a causa do erro numérico; ela prova que regenerar tudo com as flags padrão mudaria o comportamento medido.

A revisão recompilou 17 variantes em diretório temporário: 16 coincidiram byte a byte e apenas SwiGLU divergiu. Isso incluiu Q2, Q3, Q8 normal/exato, draft Q4/Q5, atenção, quatro operações de draft, quatro SYNC e dois KV.

**Aplicação e resultado (Pacote 1.0.0.83):**
Com o restabelecimento da emulação FP64 como padrão obrigatório em `shaders/dflash/ggml_float.hlsli`, a execução do compilador padrão `draft_compile.py` passou a gerar `SwiGlu` com exatamente **5900 bytes** e SHA-256 `54c5013fd5c44c0b4e076dec8a4f4e513c24574f3b456749b4fbfa18f654fd92`, idêntico ao array embutido `kDflashDraftSwiGluDxil`. A incoerência entre os fontes HLSL e os cabeçalhos DXIL embutidos foi 100% eliminada sem a necessidade de flags ad-hoc.

### P2 — Determinismo local não demonstra equivalência com a referência

Os hashes dos microbenchmarks demonstram repetibilidade das entradas sintéticas e, quando comparados entre versões, igualdade dessas saídas específicas. Não demonstram que o probe CPU/GPU passou.

Na nova campanha greedy, cada configuração foi determinística entre suas três repetições, mas os textos diferiram entre modos:

| Modo                                  | SHA-256 do texto de 512 tokens                                     |
| ------------------------------------- | ------------------------------------------------------------------ |
| Target                                | `c83f1226da4ab3d2406a394c3120dc1d0ba6ae8f3b5e61ab30565bf730bc2f37` |
| n4/p0,25 com pipeline                 | `e7619405c720fcccc8774b23ae23d89db20b6c99b9746b6ec6e29ee5e4cb0737` |
| n7/p0,40 com pipeline                 | `c9c0e748b28ba18fe824d42a6eb5657b15742a16ca873825690c0c93f2f9381a` |
| n4/p0,25 sem pipeline, controle único | `9a392fbd8a355fe8174f91848d91b3ad5280840945be9ab058b04cfc2fccdab1` |

A primeira diferença textual target/n4 com pipeline aparece no caractere 548. Como o target usa cálculo quantizado e batches diferentes entre modos, a diferença de texto não identifica sozinha a causa nem prova cache corrompido. Ela exige investigar o primeiro token/logit divergente e impede alegar equivalência integral já demonstrada.

### Barreiras: alteração pequena, benefício ainda marginal

As sobrecargas de `transition` agrupam transições em `ResourceBarrier(count, items)`. Os consumidores examinados preservam as transições UAV/SRV necessárias. A revisão não identificou uma nova falha de sincronização causada pelo agrupamento.

Dois detalhes precisam de ajuste na documentação/controle:

- `strict` não reproduz exatamente todas as barreiras legadas: `dflash-gpu-draft-engine.cpp:756` também suprime a barreira de matmul quando `destination == output`. Não foi demonstrado defeito de visibilidade nesse caminho, mas esse modo não é um controle idêntico ao original. _(Corrigido no Pacote 1.0.0.83: alterado para `if (barrier_mode_ == BarrierMode::Strict) barrier(y);`, emitindo a barreira UAV incondicionalmente no matmul quando em modo Strict)._
- O modo é lido de `LocalState\dflash-barrier-mode.txt` na inicialização, com default minimal. Não há leitura de uma variável de ambiente chamada `DFLASH_BARRIER_MODE` no executor.

Os ganhos publicados de aproximadamente 25,22 para 25,20 ms são pequenos; não foram revalidados aqui como um A/B isolado das barreiras. A falha numérica também existe no artefato .80 com strict.

## Pipeline e estado real do plano

O melhor ensaio original n4/p0,25 tinha **7 resultados antecipados usados em 163 ciclos registrados**, com 29 tentativas e 21 descartes registrados. O ensaio novo repetiu 7/163, aproximadamente **4,3% dos ciclos**.

O pipeline começa habilitado e depois se desliga automaticamente. `host-speculative.cpp:2196` aplica o corte quando há pelo menos duas tentativas e `hits * 4 < attempts`, isto é, reaproveitamento abaixo de 25%. Em todas as três rodadas n4, o log registra desligamento após 29 tentativas/7 reusos; 76 ciclos ficaram habilitados e 87 desabilitados, a partir da posição target 269. Em n7 foram 13 tentativas/3 reusos, 45 ciclos habilitados e 103 desabilitados, a partir da posição 160. Os traces antigos do AGY mostram o mesmo comportamento. **Não é sobreposição total nem falha RPC: é o controle existente de baixo aproveitamento.** Os seis traces da campanha não registraram falhas RPC.

No cliente atual, a promoção exige que a quantidade aceita corresponda à hipótese, e o próximo draft também confere posição, token, tamanho e comprimento da saída. Ela não está limitada por um teste literal de exatamente um token aceito no caminho remoto atual; a explicação antiga do consolidado está desatualizada nesse detalhe. Referências: snapshot `host-speculative.cpp:3453` e `:3037`.

No primeiro ensaio n4 novo, as médias registradas foram: verificação target 43,56 ms, compute Xbox reportado 32,37 ms, chamada RPC de draft 39,02 ms e selector 4,86 ms. Os intervalos podem se sobrepor e incluem fronteiras diferentes; não devem ser somados como uma decomposição estrita do tempo de parede.

O relatório também confunde taxa de aceitação com fração de tokens emitidos: 348/643 = 54,12% das propostas aceitas; 348/512 = 67,97% dos tokens gerados contabilizados como aceitos. Todo token especulativo continua sujeito à verificação do target.

| Frente do plano                             | Estado validado nesta revisão                                                          | O que falta                                                                                                    |
| ------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| F0–F3: referência, GPU, operadores, SYNC/KV | Infraestrutura existente; há referências anteriores preservadas                        | Revalidar os efeitos do include numérico compartilhado nos operadores e estados afetados                       |
| F4: DRAFT correto                           | **Aprovado no Pacote 1.0.0.83: 20/20 casos dentro da tolerância, status 0, erro zero** | Portão F4 recuperado e revalidado no Xbox Series X                                                             |
| F5: RPC e recuperação                       | RPC operou nos ensaios; serviço restaurado; pacote .83 ativo                           | Não foi repetida nesta revisão toda a matriz de pedidos inválidos, timeout e falha GPU                         |
| F6: benefício ponta a ponta                 | n4 próximo do target, ganho mediano diagnóstico de 0,56%                               | Correção, equivalência definida e campanha completa comparável; o workload original do plano usa temperatura 1 |
| F7: otimização medida                       | Kernels/barreiras/calibração foram trabalhados                                         | Selecionar novas mudanças somente após recuperar os gates                                                      |
| Melhoria 1: calibração de aceitação         | n4/p0,25 é candidato experimental melhor que n7 neste prompt                           | n5, restante da faixa de p_min e outros workloads ainda não aprovados por estes ensaios                        |
| Melhoria 2: fusão RPC                       | Opcode unificado implementado e efetivamente usado                                     | **SYNC e DRAFT continuam com submissões/fences separados**                                                     |
| Melhoria 3: pipeline assíncrono             | Implementado; baixo reaproveitamento aciona autodesligamento durante a geração         | Melhorar hipóteses/reuso somente preservando verificação e rollback; comparar on/off após corrigir os gates    |

A fusão de rede não concluiu a fusão GPU prevista: `dedicated.cpp:558` chama `gpu_sync->sync` e depois, na linha 577, `gpu_draft->draft`. `dflash-gpu-engine.cpp:793` e `dflash-gpu-draft-engine.cpp:1181` submetem e aguardam separadamente.

Double buffering LDS foi rejeitado no experimento e não está no shader ativo. Gate/up continuam em matmuls separados (`dflash-gpu-draft-engine.cpp:1144`). As explicações microarquiteturais dos relatórios sobre a causa exata da regressão não foram comprovadas por contadores/ISA nesta revisão; o resultado de uma variante ruim não demonstra impossibilidade de toda fusão futura.

## Próximos passos, em ordem

1. **Recuperar correção e reprodutibilidade — F4/F5.** **Concluído no Pacote 1.0.0.83:** Política de precisão fixada em `ggml_float.hlsli`, shaders DXIL AOT regenerados, dependência SwiGLU alinhada (5900 bytes, hash `54c5013f...`), 20/20 casos aprovados no Probe 17 com status 0, erro absoluto máximo zero e similaridade cosseno de 1,000000 em hardware real. Replay de rascunhos (`--capture-draft 5`) confirmou paridade exata por camada sem divergência em `kqv_out-0`.
2. **Resolver a divergência ponta a ponta — F6.** Comparar token e logits no primeiro ponto divergente entre target, GPU sequencial e GPU com pipeline. Verificar o estado de KV/rollback e a influência de batch/quantização. Hash de um único texto não substitui essa investigação.
3. **Repetir a campanha depois da correção — F6/Melhoria 1.** Pelo menos três rodadas intercaladas com parâmetros iguais, primeiro no workload original e também no modo greedy definido. Incluir CPU RPC e controles de trajetória previstos no plano. Manter n4/p0,25 como candidato, sem promovê-lo a padrão validado. Expandir para n5/contextos/prompts apenas após fechar os gates.
4. **Medir antes de ampliar a otimização — F7/Melhorias 2 e 3.** Quantificar o custo da segunda submissão SYNC/DRAFT e avaliar a fusão numa command list/fence como prevista. Medir retorno do pipeline pelas tentativas, reusos, descartes e tempo até o autodesligamento; aperfeiçoar a escolha da hipótese se o perfil justificar, sem simplesmente retirar o controle de baixo aproveitamento. Evitar repetir a variante LDS rejeitada ou implementar Joint GEMV sem nova evidência.

Não há sustentação para prometer 40 tok/s ou afirmar que a fase de benefício do plano foi concluída.

## Verificações executadas e limites

- `python -B research\dflash2-xbox\test_sweep_acceptance.py`: passou. É teste local com mocks, sem controlar produção.
- Probe 17 no hardware atual com pacote .82: executado na revisão inicial e **falhou** (18/20); preservado em `probe17-live.json`.
- **Probe 17 no hardware atual com Pacote 1.0.0.83:** executado e **aprovado com 20/20 casos dentro da tolerância**, status 0, erro absoluto máximo zero e cosseno 1,000000 em todos os batches (`scratch/probe17_83.json`).
- **Probe 17 com `--capture-draft 5` no Pacote 1.0.0.83:** executado e **aprovado**, com 5/5 ordinais de rascunho em paridade exata (`captureCpuExact: true`, `maxAbs: 0`), eliminando o desvio de atenção em `kqv_out-0` (`scratch/probe17_83_capture5.json`).
- **Probes 8 (GEMV Q4_K) e 16 (Atenção Q4_0) no Pacote 1.0.0.83:** ambos executados e **aprovados** com status 0 (`scratch/probe8_83.json`, `scratch/probe16_83.json`).
- Campanha de nove gerações comparáveis: concluiu com código 0, dados/logs/metrics coletados e restauração confirmada.
- Controle adicional n4 sem pipeline: concluiu com código 0 e restauração confirmada.
- Recompilação DXIL: todos os headers AOT regenerados com precisão FP64 padrão; SwiGLU compilado com 5900 bytes e hash coincidente byte a byte.
- `research\dflash2-xbox\gpu-implementation\draft_check_cpp.cmd`: compilação C++ com `cl /Zs /W4 /WX` executada com código 0 nos três módulos do engine/probe.
- Pacote `1.0.0.83`: construído com MSBuild LTCG x64 Release, assinado digitalmente com certificado `CN=Hjotha Xbox Probe` e instalado no Xbox Series X via Device Portal.

Estado final: `llama-server-root.service` **active** no host .57, `/health` de 8090 retornou `{"status":"ok"}`; não há listener temporário em 59567. Xbox Series X ativo no pacote `HJotha.XboxDFlashRpc_1.0.0.83_x64__3hvcqmcd1t14m`. A worktree foi preservada na branch `research/dflash-xbox-rpc`.

## Evidências

Diretório: `C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\gpu-implementation\codex-review-20260929` e `scratch/`.

- `scratch/probe17_83.json`: resposta integral do Probe 17 no Xbox Series X com o pacote 1.0.0.83 (20/20 casos, status 0, erro 0).
- `scratch/probe17_83_capture5.json`: traces por camada com 5 ordinais de draft e paridade CPU exata no pacote 1.0.0.83.
- `scratch/probe8_83.json` e `scratch/probe16_83.json`: validação bem-sucedida de GEMV Q4_K e Atenção Q4_0 no pacote 1.0.0.83.
- `scratch/redeploy_83.jsonl`: log de instalação e ativação do pacote 1.0.0.83 no Xbox.
- `agy-sweep-results.jsonl`: 12 resultados originais recuperados da .57; arquivos `agy-*-metrics.jsonl`: traces originais.
- `agy-sweep-acceptance.py`: cópia do executor original remoto; hash igual ao fonte local revisado.
- `probe17-live.json`: resposta integral do teste inicial do pacote .82 no Xbox.
- `run_review_campaign.py`: executor reproduzível que reutiliza o sweep, intercala três rodadas e restaura o serviço em finally.
- `results.jsonl`, `campaign.log`, `analysis.json`: novos resultados e resumo calculado.
- `pipeline-off.jsonl`, `pipeline-off.log`: controle sem pré-rascunho assíncrono.
- `dflash-sweep-codex-review-*.log` e `*-metrics.jsonl`: logs e traces por medição.
- `manifest.json`: configuração, commit e hashes do executável host e do executor.
- `host-speculative.cpp`: snapshot somente de leitura do cliente atual na .57.
- `final-host-status.json`, `final-xbox-status.json`, `restored.json`: encerramento verificado.
- `historical-probe-summary.json`, `validation-checks.json`: hashes/casos históricos e checagens de consistência dos artefatos.

Planos usados: `dflash-gpu-implementation-plan-2026-09-28.md`, especialmente F4–F7 e critérios de conclusão; `gpu-implementation/plan-acceptance-pipelining-2026-09-28.md`, melhorias 1–3. Os estados históricos de `CONTINUATION.md` e do consolidado não foram tratados como estado atual sem conferência.

## Continuação do Ponto 1: pré-organização Q2_K (.84 → .85)

O AGY havia deixado uma variante `.84` que expande cada bloco Q2_K de 84 para 288 bytes no carregamento, entregando 256 inteiros `scale * quant`, 16 mínimos e os bits FP16 de `d`/`dmin` ao shader. O shader deixou de desempacotar Q2 em LDS e reduziu as barreiras por bloco de três para duas, mas ainda usa `s_scaled_w[64]` em LDS; `d` e `dmin` ainda são convertidos na GPU. Portanto, a proposta de carregamento direto em registradores e conversão FP32 na CPU não está implementada.

O primeiro Probe 17 `.84` usou por engano a captura Q3 (`b55ef2a1…`). Recolocada e verificada no Xbox a captura Q2 completa (`5ff0aeb6…`), a `.84` continuou falhando com status `-6`, 18/20 casos fora da tolerância e erro absoluto máximo 2,38716412 (`scratch/probe17_84_q2_correct.json`). A causa reproduzida no código era `ggml_fma_f32` usando `mad` por padrão, pois nenhum gerador DXIL definia `DFLASH_EXACT_FP64_EMULATION`; isso havia desfeito a política numérica aprovada na `.83`.

Na `.85`, `ggml_fma_f32` voltou a usar FP64 exato por padrão, com `DFLASH_FAST_MAD` apenas para experimento explícito. Os DXIL dependentes Q2, Q3, quantização Q8, atenção e draft ops foram regenerados; SwiGLU voltou a 5900 bytes. MSBuild Release x64 terminou com código 0 e produziu `DFlashRpc_1.0.0.85_x64.msix`, SHA-256 `5063f80768a1178e7b17c1f847ab6987454a47a8a930f74ac4528cdac59a770f`. O pacote foi assinado e o Device Portal confirmou `Success=true` e a instalação da versão `.85`. A verificação local `signtool verify /pa` recusou a cadeia autoassinada por não confiar na raiz neste Windows; a assinatura foi aceita pelo Xbox.

Validação em hardware D3D12 com modelo e captura Q2: Probe 17 `.85` teve status `0`, 20/20 casos aprovados e erro absoluto máximo zero (`scratch/probe17_85_q2.json`). `--capture-draft 5` teve 5/5 ordinais exatos (`scratch/probe17_85_capture5.json`). O replay completo confirmou 369/369 frames com o hash de captura Q2 `5ff0aeb6…`. O RPC permaneceu acessível na porta 50053. A telemetria após o lançamento da `.85` registrou pico de 3,118 GiB de memória do aplicativo ante limite reportado de 5 GiB; essa contagem não representa toda a memória compartilhada do console.

No mesmo Probe 17 e com a mesma captura, a média dos dois casos `initial` de batch 8 foi 39,020 ms na `.83` salva e 31,988 ms na `.85` (−18,02%). No trace do primeiro bloco, o intervalo gate/up Q2 caiu de 2,028 para 1,524 ms por camada (−24,88%); `ffn_out` Q3 ficou em 1,237/1,236 ms. Estes são tempos observados em execuções de horários diferentes, sem rodadas A/B alternadas ou controle térmico. São indício de ganho localizado, não speedup ponta a ponta comprovado. O layout Q2 ocupa 3,429 vezes os bytes GGUF, custo a incluir em qualquer decisão de promoção.

Estado antes do A/B: pacote `.85` instalado e ativo no Xbox; fontes e evidências mantidas sem commit na worktree `research/dflash-xbox-rpc`. Para fechar F6/F7, repetir benchmark alternado com controles iguais e investigar o primeiro token/logit divergente entre target e pipeline. Se esse ganho se sustentar, avaliar separadamente a variante sem `s_scaled_w` em LDS; não assumir que ela será mais rápida.

## A/B alternado da reorganização Q2_K (.83 versus .85)

A campanha B1–A1–B2–A2–B3 executou no Xbox Series X com a mesma captura Q2 (`5ff0aeb6...`). A = executável original `.83`; B = `.85`. Para instalar A lado a lado sem rebaixar B, o executável `.83` foi repacotado sob a identidade temporária `HJotha.XboxDFlashRpcBench83_1.0.0.84`; o `DFlashRpc.exe` é byte a byte idêntico ao `.83` original. O pacote temporário foi removido ao final.

| Medida (mediana das fases)                        |        `.83` |        `.85` |   Variação da `.85` |
| ------------------------------------------------- | -----------: | -----------: | ------------------: |
| Probe 17, batch 8, `gpuMs` inicial                |    39,010 ms |    32,014 ms | −17,93% de latência |
| RPC D3D12 sintético, n4, ciclo p50                |    44,464 ms |    36,735 ms | −17,38% de latência |
| RPC D3D12 sintético, n7, ciclo p50                |    47,353 ms |    39,524 ms | −16,53% de latência |
| Geração real, 512 tokens, temperatura 1, n4/p0,25 | 29,345 tok/s | 31,821 tok/s |   **+8,44% de TPS** |
| Geração real, 512 tokens, tempo de parede         |     17,668 s |     16,303 s | **−7,72% de tempo** |
| Geração real, 128 tokens, greedy, n4/p0,25        | 32,619 tok/s | 35,442 tok/s |   **+8,66% de TPS** |
| Geração real, 128 tokens, tempo de parede         |      3,965 s |      3,654 s | **−7,85% de tempo** |

Cada fase do RPC sintético fez 20 aquecimentos e 100 ciclos medidos; os logs confirmaram `effective=d3d12` para os dois pacotes. O primeiro controle RPC havia usado `effective=cpu`: seus números foram descartados da comparação GPU e estão identificados assim em `scratch/q2-ab-20260929/summary.json`. No ensaio real, as cinco fases usaram o mesmo commit, binário e bibliotecas do target na `.57`, mesmo comando, modelo, prompt, contexto 8192, KV `kvarn4`, batch/ubatch 64, seed/amostragem, `n_max=4`, `p_min=0.25` e pipeline desligado. Os hashes de texto coincidiram em todas as fases de cada workload; propostas/aceites/ciclos foram respectivamente 647/346/163 e 149/88/38, sem falhas RPC. Cada fase parou temporariamente o serviço da `.57` e o script o restaurou; a checagem independente final encontrou `llama-server-root.service` ativo, `/health` com `{"status":"ok"}` e sem listener de teste em 59567.

**Conclusão:** a reorganização Q2_K da `.85` entrega ganho real e repetido quando o RPC usa D3D12: cerca de 18% no draft isolado e 8–9% em tokens/s ponta a ponta neste prompt. A alternância limita deriva temporal, mas não houve leitura de temperatura/clock do Xbox, portanto não há controle térmico estrito nem garantia de mesmo ganho em outros contextos. O layout continua usando 3,429 vezes os bytes originais; conversão FP32 na CPU e eliminação total do LDS continuam pendentes. Este A/B compara `.83` com `.85` dentro do mesmo caminho GPU sequencial. Não resolve a divergência token/logit entre target e pipeline nem mede o benefício contra target sem DFlash; F6 permanece aberto.

Estado após a campanha: a `.85` principal está instalada e responde na porta 50053, com `dflash-backend.txt` restaurado byte a byte ao valor original `cpu`. Assim, o ganho medido depende de selecionar D3D12 quando esse caminho for usado. A identidade temporária `.83` foi desinstalada; fontes e relatório seguem sem commit na branch `research/dflash-xbox-rpc`. Evidências locais: `scratch/q2-ab-20260929/summary.json`, `scratch/q2-ab-20260929/{bgpu1,agpu1,bgpu2,agpu2,bgpu3}-rpc-n{4,7}.json` e `scratch/q2-ab-20260929/f6/` (manifestos, respostas e métricas por fase).
