# MTP no Xbox: revisão da Rev 37 e próximos passos

Data: 2026-10-04. Revisão feita no optiplex. Status: revisão concluída; implementação deste plano não iniciada.

## Conclusão

A Rev 37 não demonstrou o ganho de 10% exigido para integrar o MTP. Manter a branch experimental separada de `main` continua correto. Entretanto, os resultados não demonstram que o Xbox inviabiliza MTP, nem que a implementação está completa e correta. Existem custos concretos no port e lacunas de validação que ainda precisam ser resolvidos.

O principal limite identificado no backend é que o lote de verificação executa multiplicações independentes por coluna/token. Ele não implementa compartilhamento explícito da leitura e desquantização dos pesos entre os tokens. Além disso, cada tentativa de draft executa uma cabeça MTP sequencial, com operações na CPU, sincronizações com D3D12, sampling e, quando `p_min > 0`, um cálculo de confiança sobre todo o vocabulário. Os tokens descartados também custam processamento.

Não há promessa de speedup. A próxima etapa deve produzir uma implementação verificável e uma decomposição dos custos; só então vale escolher uma otimização de backend.

## Estado exato e escopo

- Worktree existente: `/home/hjotha/worktrees/xllama-mtp`.
- Branch: `spike/beellama-pin`.
- HEAD: `051b05ecccb8d5677f561b04972731edb3fe66cd`.
- Submódulo BeeLlama: `982eaadaa9dbe761f00205bc20b6c04b7329b58d`.
- O submódulo tem três arquivos modificados. O diff corresponde byte a byte ao patch versionado `patches/0004-export-mtp-embedding-apis.patch`; SHA-256 do diff: `42f640361cc758fdfd52607c106fcfbbf730da81604c60c39b0c06e709199b6a`.
- `plans/` já existia como diretório não rastreado. Portanto, o checkout não estava literalmente limpo, embora os arquivos de implementação do repositório principal não tivessem modificações locais.
- O log histórico registra a instalação de `GianlucaMazza.xllama_1.6.0.37_x64__pj67f1fcj4n14`. Esta revisão não verificou o pacote atualmente instalado no console.
- Foram revisados o drafter, o loop de geração, seus chamadores CLI/Session, o backend D3D12, o shader Q4_K, os testes relevantes e os artefatos Rev 37. Não foi feita auditoria geral de ORT, treinamento, UI, segurança ou outros modelos.
- Nenhum código, serviço, pacote ou configuração do Xbox foi alterado nesta revisão. Não foram executados build, nova inferência ou novo benchmark. Foram validados os seis CSVs com o validador do projeto e recalculadas suas estatísticas.

Os números abaixo são do benchmark existente, informado como Xbox Series X. A coluna `host` usa o rótulo fixo `xbox-series-s`, montado em `uwp/inference-bridge.cpp`; ela não identifica o hardware físico. Registrar a identidade real no próximo benchmark.

## O que os dados realmente mostram

Fontes: `/tmp/opencode/r37/{base1,m4p75,m4p50,m4p25,m4p0,base2}.csv`, os `.log` correspondentes e `pipeline.log`. Cada braço fez seis execuções: uma de aquecimento e cinco medidas. Todos os 30 registros medidos têm 298 tokens de prompt e 64 tokens gerados, contexto 2048, seis threads, backend D3D12, `ubatch=64`, `gpu-layers=99` e `ignore-eog`. Os tags não indicam KV q8; não atribuir essa configuração à Rev 37.

| Braço | Mediana tok/s | Faixa das cinco medidas | Drafts propostos / aproveitados, soma | Aproveitamento |
| ----- | ------------: | ----------------------: | ------------------------------------: | -------------: |
| base1 |         17,71 |             17,43–17,97 |                                 0 / 0 |              — |
| m4p75 |         17,69 |             16,78–18,70 |                             130 / 112 |         86,15% |
| m4p50 |         17,08 |             16,26–18,21 |                             195 / 124 |         63,59% |
| m4p25 |         15,17 |             14,34–16,28 |                             281 / 112 |         39,86% |
| m4p0  |         14,19 |             13,63–16,13 |                             432 / 116 |         26,85% |
| base2 |         17,82 |             17,44–17,99 |                                 0 / 0 |              — |

Baseline combinado: **17,78 tok/s**, mediana de dez medidas. O melhor MTP ficou aproximadamente 0,5% abaixo, dentro da faixa de variação observada; o teste não estabelece uma diferença pequena com significância estatística. O requisito de +10% não foi atingido.

Reduzir `p_min` de 0,75 para zero aumentou as propostas de 130 para 432, mas os drafts aproveitados passaram apenas de 112 para 116. Foram 316 propostas não aproveitadas no p0. Essa diferença inclui caudas não verificadas após a primeira rejeição e cortes no fim da geração; não são necessariamente 316 rejeições independentes pelo target. A piora pode ser explicada por muito mais trabalho especulativo com quase o mesmo benefício. O sweep também altera profundidade efetiva, conteúdo gerado e custo do filtro de confiança; ele não isola banda de memória.

Correções na leitura dos logs:

- O p75 teve 77,42%–92,59% de aproveitamento por execução medida, não 85%–92% em todas elas.
- Os pares medidos de p75 são `27/25, 31/24, 21/18, 21/19, 30/26`. Os de p0 são `92/20, 92/22, 80/27, 88/20, 80/27`.
- Os `.log` contêm linhas de execuções anteriores ao braço e do aquecimento. Por exemplo, `m4p75.log:1-4` precede as cinco medidas. `summary.txt` mistura esses hashes e mostra apenas os últimos três pares de contadores.
- Para esta revisão foram usados os últimos cinco pares de cada log, conferidos contra a ordem, carga e throughput das execuções 2–6 em `pipeline.log`. Um novo coletor deve associar cada registro por identificador de execução, sem depender de posição no arquivo.
- Os dez hashes do baseline medido são distintos. Isso é compatível com os defaults de sampling e seed aleatória, mas não comprova a corretude do MTP.
- Os logs salvos de cada braço são extratos `done/output`. A conclusão verificável é que as 30 medidas completaram 64 tokens; esses extratos não bastam para certificar ausência de todo erro ou fallback interno.

## Por que MTP pode ser mais lento

No baseline, um token custa aproximadamente `1000 / 17,78 = 56,24 ms` nessa medição. No MTP, o tempo por token útil é:

```text
(tempo de draft + verificação target + sampling + manutenção de estado + correções)
---------------------------------------------------------------------------------
                       quantidade de tokens úteis emitidos
```

Com quatro drafts aceitos e um anchor, o orçamento para empatar cinco tokens seria cerca de 281,2 ms. Esse é um exemplo aritmético, não uma medição de um round. Todo o trabalho do MTP precisa caber abaixo desse orçamento para ganhar. A aceitação, sozinha, não informa se isso acontece.

1. **Depth 4 não produz quatro tokens gratuitamente.** O Qwen deste fork tem uma cabeça MTP (`llama.cpp/src/models/qwen35.cpp:682-683`). `src/bridge/mtp_draft.cpp:258-311` chama `llama_decode` sequencialmente para cada profundidade. A cabeça inclui atenção, FFN e projeção de saída sobre o vocabulário (`qwen35.cpp:744-848`); não é apenas escolher quatro IDs.
2. **Um lote lógico não garante compartilhamento dos pesos no shader.** `ggml_d3d12.cpp:26-35` mapeia colunas para grupos distintos. Em `shaders/ggml_d3d12_mmv_q4_k.hlsl:31-61`, cada grupo usa `col = gid.y` e executa seus próprios `W.Load4/Load2`. O hardware pode aproveitar cache, mas o kernel não reutiliza explicitamente o mesmo bloco desquantizado entre colunas. Portanto, não se pode considerar o custo de cinco tokens igual ao de um, nem concluir que os bytes lidos da DRAM quintuplicam sem medir.
3. **O caminho é híbrido CPU/GPU.** `src/bridge/llama_gpu.h:61-65` mantém atenção/KV na CPU; `ggml_d3d12.cpp:878-925` só aceita os tipos suportados de `MUL_MAT` e operações de view. Cada trecho D3D12 submetido espera a conclusão (`:483-491`), com espera ativa em `d3d12_compute.cpp:157-164`. A influência exata dessas esperas ainda não foi medida.
4. **Até o draft recusado pelo filtro custa.** `mtp_draft.cpp:259` executa a cabeça antes do teste de confiança em `:287-290`. A tentativa descartada nem entra em `n_drafted`. Para `p_min > 0`, `top_prob` percorre o vocabulário e soma exponenciais em double (`:33-46`). O p0 já evita esse cálculo, mas aumenta muito o desperdício de propostas.
5. **Uma rejeição faz outro decode do target.** Depois de verificar o lote e cortar a cauda, `decode_loop.h:561-569` decodifica imediatamente o token corretivo. Há oportunidade de carregá-lo como anchor pendente do próximo lote, respeitando os contratos de sampling, emissão e cache.

O mecanismo geral de ganho depende de amortizar a verificação de vários tokens, como descrito na [publicação original de speculative decoding, Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html). Uma execução limitada por memória pode justamente se beneficiar da reutilização de pesos; a situação atual precisa ser medida na implementação concreta.

## Achados priorizados

S = horas; M = cerca de um dia; L = vários dias. Esforços são estimativas de implementação e validação, não prazos garantidos.

| ID  | Prioridade e achado                                                | Evidência principal                                                                                       | Impacto                                                                                                                                                       | Esforço / risco | Confiança                                           |
| --- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------- |
| F1  | P0: estabelecer paridade e evidência por execução                  | `include/xllama/sampling.h:23-39`; `uwp/inference-bridge.cpp:318-339`; `tests/test_decode_loop.cpp:20-40` | O benchmark usa sampling estocástico com seed automática; o teste existente não ativa MTP. Aceitação e ausência de abort não certificam saída/cache corretos. | M / baixo       | Alta                                                |
| F2  | P1: completar sincronização do estado privado do draft             | `mtp_draft.cpp:185-317`; `llama.cpp/common/speculative.cpp:4093-4190,4421-4434`                           | O port recebe um hidden state, mas não reprocessa prompt e prefixo aceito como a referência. Pode perder qualidade de propostas; magnitude não medida.        | M–L / alto      | Alta para a omissão; efeito quantitativo não medido |
| F3  | P1: corrigir carry em sessões com prefixo reutilizado              | `decode_loop.h:70-78,320`; `session.cpp:635-647,766-777`                                                  | O índice inicial é inferido da posição absoluta, embora o buffer descreva o último batch. Não é válido para prefill de um delta após cache existente.         | M / médio       | Alta; não explica o benchmark frio Rev 37           |
| F4  | P1: instrumentar fases e reduzir trabalho especulativo dispensável | `mtp_draft.cpp:150-158,258-290`; `decode_loop.h:561-569`; `ggml_d3d12.cpp:709-721,791-805`                | Permite atribuir o custo e eliminar processamento redundante sem reescrever todo o backend.                                                                   | S–M / médio     | Alta para os custos; ganho ainda não medido         |
| F5  | P2: testar kernel com várias colunas e reutilização de pesos       | `ggml_d3d12.cpp:763-778`; shader Q4_K `:31-61`                                                            | Pode reduzir o custo da verificação, hoje feita como GEMVs independentes.                                                                                     | L / alto        | Alta para a estrutura; ganho ainda não medido       |

### Detalhes de estado que o executor precisa preservar

O comentário de `mtp_draft.cpp:86-90` sobre compartilhar KV não se aplica ao Qwen. `llama.cpp/src/llama-context.cpp:717-750` limpa `ctx_other` e só o restaura para outras arquiteturas. `qwen35.cpp:744,803` usa atenção com cache na cabeça MTP.

O drafter expõe apenas `init`, `ready`, `ctx`, `draft` e `params`. Não tem o equivalente de `process`/`accept` da referência para reconstruir as linhas do cache com os hidden states verificados pelo target. `m_pending_valid` nunca recebe `true`; a checagem de continuidade em `mtp_draft.cpp:194` não se torna ativa. O retorno de `llama_memory_seq_rm` em `:205` é ignorado.

Isso não prova por si só que a saída final esteja errada: um draft ruim pode ser rejeitado por um target correto. Demonstra que a qualidade das propostas não foi comparada sob o mesmo contrato de estado da referência e que faltam invariantes/testes.

Exemplo do F3: com 100 posições já preenchidas e um prefill de três tokens, o último batch tem linhas 0–2. Com `n_batch=2048`, `last_nextn_rows` usa `pos_max=102` e devolve 103, levando o primeiro draft a ler a linha 102. Essa linha não é a última linha desse decode; pode ser antiga ou inválida. A Rev 37 usa contexto novo por execução e não cobre esse caso.

Trechos atuais para conferir antes de editar:

```cpp
// src/bridge/mtp_draft.cpp:205
llama_memory_seq_rm(llama_get_memory(m_ctx), 0, pos, -1);

// src/bridge/decode_loop.h:75-78
const int n_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
const int pos = static_cast<int>(pos_max);
// The final chunk starts at the largest multiple of n_batch at or below pos.
return pos % n_batch + 1;

// src/bridge/decode_loop.h:561-564
// cand is not yet in the KV (only accepted drafts are).
if (!detail::decode_one(p.ctx, cand)) {
    log_output("[xllama] decode failed after speculative reject\n");
    stop_all = true;
```

## Roteiro de execução

Este é um plano para uma implementação posterior. Nenhum comando de build/deploy abaixo foi executado nesta revisão. Reusar os mecanismos existentes; não criar novo framework de benchmark, scheduler ou backend genérico.

### 1. Congelar evidência e adicionar a validação que falta

1. Na worktree da tarefa, conferir drift e preservar o patch aplicado no submódulo. Copiar os artefatos Rev 37 para a pasta de evidência da nova rodada antes que `/tmp` seja limpo. Registrar SHA do código, pin e patches, hash do MSIX, hash do GGUF, prompt e configuração efetiva.
2. Em `scripts/bench-xbox-ort.sh` e `uwp/inference-bridge.cpp`, adicionar transporte explícito de `greedy` e `seed` e registrar os valores realmente usados. Esses knobs ainda não existem nesse driver. Reusar `SamplingConfig` e `add_sampler_stages`, sem outro sampler. Um pedido ignorado ou um drafter desativado deve falhar a validação.
3. Salvar token IDs e associação por `run_index`/identificador único. Comparar sequências e informar a primeira divergência; o hash do texto pode continuar como resumo. Reusar os scripts de sidecar existentes quando aplicáveis, sem inventar temperatura ambiente ou consumo que não foram medidos.
4. Criar casos `mtp:*` no padrão doctest de `tests/test_decode_loop.cpp` e `tests/test_sampling.cpp`. Para o gate MTP, ausência do modelo ou dos tensores MTP é falha de pré-condição, não um teste aprovado por skip.
5. Testar baseline e MTP com greedy idêntico; depois sampling com seeds fixas. A validação por igualdade de amostras usada aqui não precisa ser substituída automaticamente por um novo algoritmo de rejection sampling: conferir primeiro que cada amostra target e seu estado são consumidos uma única vez. `llama_sampler_sample` já chama `llama_sampler_accept` (`llama-sampler.cpp:960`).

Cobrir aceitação total, rejeição na primeira posição e no meio do lote, `p_min` recusando toda a proposta, EOG, stop sequence, cancelamento e fim do limite de geração. Conferir logits após rollback contra uma execução sequencial do mesmo prefixo, além dos IDs finais. Caso diferenças numéricas alterem o argmax, registrar a primeira divergência e a margem entre candidatos; não dispensar o gate.

**Verificação:** build Linux e casos `mtp:*` com um GGUF MTP real; todos os casos devem executar, sem skips, e reproduzir baseline. Repetir o gate no Xbox para cobrir D3D12/recorrência; Linux sozinho não valida essa rota.

### 2. Medir o custo por fase antes de alterar o algoritmo

Usar `std::chrono` e os timestamps/contadores que já existem em `ggml_d3d12.cpp:791-805`. Os contadores atuais são globais e são zerados ao liberar um backend (`:709-721`); seu total não identifica target/draft nem separa prefill de geração. Expor snapshots para calcular deltas por chamada, identificando contexto e fase, sem zerar contadores usados pelo outro contexto.

Registrar, por geração: tentativas de draft, propostas entregues, descartes por confiança, profundidades, rounds, tokens aceitos, correções e quantos tokens úteis foram emitidos. Cronometrar separadamente draft decode, sampling draft, `top_prob`, verify target por tamanho de lote, decode corretivo e manutenção de estado. Separar tempo de GPU, tempo de submissão/espera e tempo total; não somar medições que se sobrepõem. Os buffers de ativação são visíveis à CPU (`ggml_d3d12.cpp:522-530`): não chamar toda espera de transferência de logits.

Medir `T_target(B)` para B=1,2,3,5 com prefixo, tokens, cache e estado recorrente equivalentes. Restaurar/recriar esse estado entre medidas para não comparar contextos diferentes. Registrar também os tipos e shapes das matmuls mais caras. Medir o overhead da instrumentação ligada/desligada.

**Verificação:** os contadores devem reconciliar com `n_generated`, `n_drafted` e `n_accepted`; os tempos de fases sem sobreposição devem explicar o tempo total dentro da resolução de medição. Não prosseguir para uma reescrita de shaders se o custo dominante estiver em outra etapa.

### 3. Corrigir o estado do draft e o carry da sessão

Arquivos: `src/bridge/mtp_draft.cpp`, `include/xllama/mtp_draft.h`, `src/bridge/decode_loop.h`, `src/bridge/inference.cpp`, `src/bridge/session.cpp` e os testes correspondentes.

1. Portar o contrato mínimo de `common/speculative.cpp:4093-4190,4421-4434` para uma sequência e uma cabeça. Alimentar o contexto privado com tokens e hidden states target corretamente deslocados. Decodificar o catch-up em blocos que caibam no batch efetivo do draft; não enviar o prompt inteiro ao contexto limitado a `n_max+2`.
2. Após verificação, reconstruir/atualizar o prefixo aceito com estados do target, remover a cauda descartada e carregar exatamente a linha que precede o anchor pendente. Preservar o estado recorrente do target ao recuar.
3. Validar retornos de remoção, posições e continuidade. Em falha, reinicializar o draft e usar fallback explicitamente contabilizado ou reportar erro; não continuar com estado desconhecido. Não tornar o teste verde desativando silenciosamente MTP.
4. Passar a quantidade real de linhas do último prefill/decode ao loop. Não reconstruí-la com `pos_max % n_batch`. Cobrir prefill com vários chunks, delta após prefixo reutilizado e segundo turno.
5. Integrar reset, troca de prompt, reaproveitamento, restauração de estado e context shift à vida útil do draft. Se não houver reconstrução validada para alguma operação, invalidar o drafter explicitamente e reconstruí-lo antes de voltar a especular.

**Verificação:** testes de paridade da etapa 1, mais sessão com prefixo 100 + delta 3, cold prefill maior que `n_batch`, sequência de rejeições, reset e dois turnos. Conferir posições e prefixo do draft com a referência; registrar separadamente o custo adicional de catch-up e a mudança de aceitação. Catch-up correto pode aumentar o custo bruto, portanto sua contribuição líquida precisa ser medida.

### 4. Reduzir trabalho dispensável, uma mudança por comparação

1. **Correção pendente:** substituir o decode imediato de `cand` em `decode_loop.h:562` pelo estado explícito de anchor pendente do próximo round MTP. O token já foi amostrado; não amostrar de novo nem chamar accept duas vezes. Guardar o hidden state da última linha aceita antes de qualquer decode que substitua o buffer. Separar emissão do token e inserção no KV; `on_accepted` mantém `m_kv_tokens` sincronizado com o cache. No final/cancelamento, satisfazer o contrato de estado da Session. Manter o caminho de prompt-lookup fora dessa primeira mudança.
2. **Limites:** reduzir a profundidade de draft ao espaço de contexto, batch e orçamento de saída restantes. Evitar produzir/verificar tokens que já não podem ser emitidos. Cobrir final de geração com um token restante.
3. **Sampling do draft:** medir e remover o `top_k(10)` antes de greedy se os testes confirmarem a seleção/tie-break desejados. Ele não precisa ordenar dez candidatos para escolher o maior logit. Não modificar o sampler target.
4. **Confiança:** preservar o bypass já existente para `p_min <= 0`. Se `top_prob` for relevante, um primeiro experimento exato é encerrar a soma quando ela exceder `1/p_min`: com máximo do logit já conhecido e termos positivos, o candidato já está abaixo do limite. Testar fronteiras e valores inválidos. Não confundir eliminar o cálculo com baixar o limiar e produzir mais drafts.

**Verificação:** todos os gates de estado/saída passam; o contador de decode corretivo diminui sem duplicar tokens; propostas/tokens de saída mantêm os limites. Medir cada alteração isolada no mesmo conjunto de prompts/seeds. A aceitação total já leva o próximo token target para o round seguinte; não adicionar um segundo “bonus token” por engano.

### Gate explícito: separar limite do Xbox de limitação da implementação

Antes de concluir que o Xbox não oferece ganho para MTP, medir explicitamente a escalabilidade do **verify target** com o mesmo modelo, pesos, prefixo, estado recorrente, backend D3D12 e configuração efetiva. O limite de memória de processo (por exemplo, a cota disponível ao app) e a largura de banda/latência do backend são restrições diferentes: não atribuir ausência de speedup à cota de memória sem evidência de paginação, eviction, pressão de working set ou transferências adicionais.

Medir `T_target(B)` para `B = 1, 2, 3, 5`, restaurando ou reconstruindo estado equivalente entre medidas. Registrar pelo menos tempo total, tempo GPU, submit/wait, sincronizações CPU/GPU, bytes/working set quando observáveis e shapes/tipos das matmuls dominantes. A métrica principal é o custo amortizado por coluna/token verificado:

```text
C_verify(B) = T_target(B) / B
R_scale(B)  = T_target(B) / (B * T_target(1))
```

`R_scale(B) ~= 1` significa crescimento quase linear, ruim para speculative decoding; quanto menor que 1, maior a amortização. Os limiares abaixo são **heurísticas de decisão**, não critérios universais:

- `T_target(5) >= 4 * T_target(1)`: pouco espaço para MTP no backend atual. Não concluir ainda "limite físico do Xbox"; primeiro verificar se o custo dominante é releitura/desquantização de pesos, fences ou sync.
- `T_target(5) ~= 2–3 * T_target(1)`: há amortização suficiente para MTP ser plausível, dependendo do custo do drafter, aceitação e correções.
- `T_target(5) <= 2 * T_target(1)`: forte headroom para MTP; se o end-to-end continuar sem ganho, procurar overhead fora do verify.
- Para o gate de produto de `>=10%`, o custo médio por token útil do caminho MTP precisa ficar em `<= 0,909 * T_baseline_token` sob a mesma carga. Esse cálculo deve incluir draft, verify, sampling, `top_prob`, manutenção/catch-up, correções e sincronizações; não usar apenas `T_target(B)` para declarar speedup.

Exemplo apenas ilustrativo: se o baseline estiver em ~56 ms/token, o caminho MTP precisa ficar em ~51 ms/token útil ou menos para atingir +10%. Um `T_target(5)` próximo de 5x `T_target(1)` torna isso improvável; um `T_target(5)` de 2–3x pode deixar margem suficiente, desde que o drafter e as sincronizações não consumam a economia.

**Ordem de decisão obrigatória:**

1. Corrigir paridade/estado/carry/reset e provar que MTP permanece ativo sem fallback silencioso.
2. Medir `T_target(1/2/3/5)` e decompor o custo por fase.
3. Se verify for dominante e escalar quase linearmente, testar a variante multi-coluna que reutiliza a leitura/desquantização dos pesos para B=2–5.
4. Repetir exatamente o mesmo microbenchmark após a mudança e comparar `T_target(B)`, `C_verify(B)` e `R_scale(B)`.
5. Só depois repetir o gate end-to-end no Xbox, com paridade preservada, e decidir se existe ganho real.
6. Se mesmo com estado correto, sync reduzido e verify amortizado o ganho continuar ausente, registrar isso como evidência de teto prático do hardware/backend para esta configuração; não extrapolar automaticamente para todo D3D12 ou todo MTP.

Esse gate existe para distinguir três resultados diferentes: **bug/overhead de implementação**, **backend D3D12 sem amortização suficiente**, ou **limite prático do hardware/configuração do Xbox**. Não pular diretamente do benchmark Rev 37 para a terceira conclusão.

### 5. Só com perfil favorável, experimentar reutilização de pesos

Arquivos permitidos nesta etapa: `src/bridge/ggml_d3d12.cpp`, `include/xllama/ggml_d3d12.h`, shaders `ggml_d3d12_mmv_*` relevantes, respectivos DXIL gerados pelo build existente e `tests/test_ggml_d3d12.cpp`.

Se o custo de `T_target(B)` for dominante e crescer desfavoravelmente, implementar uma variante estreita para B=2–5 que carregue/desquantize cada bloco de pesos e o aplique a mais de uma coluna. Começar pelo tipo/shape comprovadamente dominante; um arquivo Q4_K_M pode usar diferentes tipos de tensor, portanto não assumir que só Q4_K importa. Preservar a rota de uma coluna como controle.

Reusar `d3d12_mmv_emulate`, a referência `ggml` e `run_d3d12_selftest`. Incluir B=1,2,3,5, strides com padding, número de linhas não múltiplo do tile e shapes reais do Qwen. Validar erro numérico no Xbox e repetir paridade ponta a ponta; passar a tolerância do selftest isolado não garante os mesmos tokens.

**Verificação:** `scripts/check-win-syntax.sh`, build MSVC/UWP, testes de emulação, `scripts/bench-d3d12-selftest.sh` no Xbox e a comparação pareada. O selftest deve indicar `ok=1` e `d3d12_ran=1` em todos os casos exigidos. Não remover fences necessárias a consumidores CPU; qualquer redução de sincronização exige análise de dependências própria.

### 6. Repetir o gate de produto e decidir

- Mesmo MSIX/GGUF/prompt/configuração para cada comparação, com identidade e hashes registrados. Alternar ou randomizar a ordem dos braços por bloco; guardar seeds para reproduzir o mesmo texto no teste controlado.
- Começar com profundidades 1,2,4 e o limiar útil apontado pelo perfil, sem repetir um sweep amplo por hábito. Depois testar pelo menos três prompts representativos (prosa, código e conteúdo repetitivo), geração curta de 64 e sustentada de 256 tokens, respeitando o contexto.
- O driver descarta a primeira execução: `--runs 6` produz cinco medidas; `--runs 11` produz dez. Preservar resultados individuais e estimar a incerteza do ganho pareado.
- Exigir paridade greedy, testes de cache/rollback aprovados, MTP efetivamente ativo e ganho mediano de pelo menos 10% sobre baseline contemporâneo, com intervalo de confiança favorecendo MTP. Registrar TTFT, decode, memória e falhas separadamente. Não dividir working set pela cota DXGI: são métricas distintas.
- Se o ganho não aparecer, registrar o resultado por configuração e manter MTP desligado por padrão. Uma conclusão negativa após medir/corrigir esses pontos é válida; não substituí-la por uma afirmação universal sobre D3D12.

## Comandos e critérios de conclusão

Executar na worktree da implementação; estes são comandos do projeto para a rodada futura, não resultados desta revisão.

```bash
cd /home/hjotha/worktrees/xllama-mtp
git status --short --branch
git diff --stat 051b05ecccb8d5677f561b04972731edb3fe66cd..HEAD -- src/bridge include/xllama shaders scripts/bench-xbox-ort.sh uwp/inference-bridge.cpp tests patches
git -C llama.cpp diff --stat
cmake --preset linux-test
cmake --build build/linux-test -j"$(nproc)"
ctest --test-dir build/linux-test --output-on-failure
```

Depois que os casos `mtp:*` forem implementados, com a fixture local existente conferida como GGUF MTP:

```bash
XLLAMA_TEST_MODEL=/tmp/opencode/stage-mtp.gguf ./build/linux-test/tests/xllama-tests --test-case='mtp:*'
./build/linux-test/tests/xllama-tests --test-case='ggml_d3d12:*'
./scripts/check-win-syntax.sh
```

Esperado: exit 0, testes relevantes realmente executados e nenhum skip usado para declarar paridade. O check Windows depende do SDK/xwin já documentado; se indisponível, registrar a limitação e usar o build MSVC como gate, sem declarar a sintaxe aprovada.

Build UWP documentado, no checkout Windows da tarefa:

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
```

No fluxo de build existente em `.193`, conferir SHA e tarefa antes de disparar: não parar uma compilação alheia, não ignorar `MultipleInstances: IgnoreNew` e não reutilizar marcadores de conclusão de outra revisão. Só medir depois de associar pacote instalado ao SHA e aos patches testados. Uma nova revisão de pacote deve ser escolhida no momento da execução.

Comando atual do driver que reproduz as configurações de throughput do p75 (não certifica paridade enquanto os knobs da etapa 1 não existirem):

```bash
./scripts/bench-xbox-ort.sh qwen35-4b-mtp --runs 6 --n-predict 64 --gpu-layers 99 --ctx 2048 --ubatch 64 --ignore-eog --mtp 4 --mtp-pmin 75 --out /tmp/opencode/mtp-next-p75.csv
python3 scripts/validate-benchmark.py /tmp/opencode/mtp-next-p75.csv
```

Usar as credenciais locais configuradas segundo `AGENTS.md`; nunca escrevê-las no plano ou nas evidências. Estender o driver existente para os controles novos em vez de usar flags ainda inexistentes.

Critérios finais, todos obrigatórios antes de recomendar integração:

- [ ] Build e suíte do projeto aprovados; testes MTP efetivamente executados.
- [ ] IDs greedy iguais ao baseline nos cenários de aceitação, rejeição e continuidade.
- [ ] Estado após rollback comparado ao mesmo prefixo sequencial; resets/carry verificados.
- [ ] Tempos e contadores por fase reconciliados, com custo da instrumentação conhecido.
- [ ] Artefatos sem mistura de aquecimento, execuções anteriores ou configurações diferentes.
- [ ] Selftest no Xbox aprovado quando houver mudança de kernel; paridade ponta a ponta preservada.
- [ ] Ganho >=10% comprovado no protocolo pareado definido, com incerteza favorecendo MTP.

Se o último critério falhar, a rodada experimental pode ser encerrada com resultado negativo documentado, mas a integração continua reprovada.

## Limites e condições para interromper a implementação

Reusar a worktree da tarefa, preservar alterações de outros agentes e manter as mudanças do submódulo representadas em `patches/`. Não editar `main`, atualizar o pin inteiro, trocar o modelo, portar múltiplas cabeças ou reescrever ORT/UI para resolver este problema. Seguir C++17, RAII, o padrão doctest existente, clang-format 22.1.5 e o build DXIL do projeto.

Se houver drift nos trechos citados, outro executor estiver modificando os mesmos arquivos, faltar fixture MTP ou paridade falhar, resolver a causa/evidência antes de avançar ao teste de desempenho. Uma falha não deve ser mascarada desligando o MTP. Código correto sem ganho de desempenho é um resultado possível, e não autoriza afirmar speedup. Esta revisão/plano não executa commit, merge, push ou deploy.

O plano 001 segue como referência histórica de intenção. O plano 002 contém a orientação antiga de inferir o carry pela posição absoluta; usar o contrato de linhas do último batch aqui descrito. As correções já presentes — batch combinado, proteção contra double trim, `n_rs_seq`, threadpools persistentes e bypass de confiança no p0 — não devem ser refeitas.

## Hipóteses consideradas e descartadas

- “Todo MTP é mais lento com profundidade 1”: não decorre do algoritmo; depende do custo e da verificação conjunta.
- “O teste monotônico provou banda de memória”: descartado como conclusão causal; aceitação e quantidade de trabalho variam junto.
- “Não crashar e aceitar 90% prova corretude”: descartado; faltam comparação de tokens/logits e invariantes de estado.
- “É preciso copiar todos os logits por PCIe a cada token”: não demonstrado neste backend, que usa memória de ativações visível à CPU.
- “Falta emitir um bonus separado após aceitação total”: o próximo round já amostra os últimos logits; adicionar um token sem reconciliar a máquina de estados causaria erro.
- “O submódulo sujo significa alteração perdida”: nesta revisão o diff é exatamente o patch 0004 versionado.
- “Reescrever tudo para GPU ou integrar dot4 imediatamente”: adiado; primeiro medir a etapa dominante e tentar a mudança menor que preserve numericamente o caminho atual.

## Validações efetivamente realizadas nesta revisão

- Leitura dos arquivos citados e comparação com `common/speculative.cpp` no pin local.
- `python3 scripts/validate-benchmark.py` em cada um dos seis CSVs Rev 37: todos retornaram `OK schema=v2` e exit 0.
- Recálculo independente de medianas, faixas, tokens, contadores e hashes das execuções medidas.
- Conferência byte a byte do diff do submódulo com `patches/0004-export-mtp-embedding-apis.patch`.
- Nenhum novo build, teste numérico de MTP ou benchmark no Xbox foi executado. Os ganhos das propostas permanecem não medidos.

SHA-256 das fontes CSV e do log do pipeline:

```text
base1.csv    a6fc972c7356f2be629f9e1e2f5ecae30505488052ae00d964de54ebee3bf071
m4p75.csv    e00437cd30ceaba817ba68e0f249a0f02d44c280f3ed6924257aa63eb590a855
m4p50.csv    8d236e49c4a4c1cb0ac64750391e2382e7ddb994a6512c5d40d0ad68f841bb2d
m4p25.csv    cea6fa2c61f5e6b96f39704f69d14ddfa42e3867f66cce94498ebfce81412148
m4p0.csv     1f546f4317bdd97f218b3ad2a2d81b1e2ebf421e5324a5d97ec4bb605026c472
base2.csv    99b5e5a82cb915eb42b0cb9ea63ebbe2c115f5a98b8c07b2612d6b940e91be7e
pipeline.log 142a14264c682107d376cdc16ca13ee0625740f4144d687deffd0aa04b759a26
```
