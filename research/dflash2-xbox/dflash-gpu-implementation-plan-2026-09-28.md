# Plano de implementação: GPU no Xbox DFlash RPC

Data: 2026-09-28. Status: proposto; implementação GPU e novos benchmarks ainda não executados.

## 1. Objetivo e decisão

Implementar execução GPU do DFlash2 no Xbox Series X dentro do app separado `HJotha.XboxDFlashRpc`, preservando o protocolo RPC otimizado e a execução do target na RTX 4070 da `.57`. O resultado esperado é melhorar o TPS real de geração, com correção numérica e sem regressão de cache, aceitação ou estabilidade.

A direção principal é um executor D3D12 com kernels HLSL especializados no grafo DFlash. DirectML direto será comparado nas operações dominantes e poderá integrar o mesmo executor se vencer em latência, memória e correção. A escolha definitiva dos kernels depende de medições no Series X.

ONNX Runtime + DirectML fica fora da primeira implementação: não há um export DFlash validado neste projeto, e a conversão precisaria representar convolução dinâmica, KV mutável, cortes e entradas de hidden states. A CPU permanece como referência e caminho de recuperação.

Este documento é um plano solicitado para a pesquisa local. As medições originais continuam nos relatórios e arquivos de evidência referenciados; este arquivo não substitui as fontes de resultados do projeto.

## 2. Ambiente e regras de continuidade

| Item                                 | Localização ou condição                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| Host de desenvolvimento              | HPBEATS, Windows, `192.168.1.193`                                                          |
| Worktree existente                   | `C:\Users\hjotha\worktrees\xllama-dflash-xbox`                                             |
| Branch local                         | `research/dflash-xbox-rpc`                                                                 |
| Console                              | Xbox Series X, `192.168.1.26`                                                              |
| Serviço dedicado                     | TCP `50053`, protocolo DFL2 v2                                                             |
| Host do target                       | GOKAYA, Linux, `192.168.1.57`                                                              |
| Checkout do cliente no último ensaio | `/home/hjotha/beellama.cpp`, branch `research/dflash2-xbox-rpc`; revalidar antes de editar |
| Referência de app                    | `1.0.0.10`, versão usada no último relatório de otimização                                 |

- Reusar a worktree da tarefa; inspecionar branch, alterações e atividade antes de editar ou compilar. O estado atual contém pesquisa anterior não commitada.
- Não compilar enquanto outro agente altera os mesmos fontes. Alterações futuras na `.57` seguem a exceção do projeto beellama: checkout principal em branch da tarefa, sem criar outra worktree.
- Preservar pesos do target, KV principal, embeddings, output head, selector, verificação e sampling na `.57`.
- Manter o app RPC separado do xllama comum. Reaproveitar utilitários e shaders do xllama sem transformar seus testes de GPU em backend de produção por declaração.
- Conservar TCP_NODELAY no cliente e SYNC_TRIM. Não reintroduzir TRIM/rewind como round trips obrigatórios.
- Não agregar ou publicar alterações anteriores junto deste trabalho. Este pedido de plano não inclui commit, merge, push, build ou deploy.

## 3. Ponto de partida confirmado

O [relatório de TRIM/rewind](./trim-rewind-optimization-2026-09-28.md) contém os ensaios de 2026-09-28: três execuções reais otimizadas e três execuções com Xbox em mock, além dos controles. O compute Xbox ocupa cerca de 72% do ciclo real. A referência Swift sozinho é histórica e precisa ser refeita na campanha de comparação.

Referências para dimensionar a meta, retiradas desse relatório:

- Ciclo real: 218,663 ms; DRAFT + SYNC compute: 157,771 ms; TPS real médio: 12,806.
- Mock Xbox: 56,461 ms/ciclo e 49,206 tok/s, com a trajetória de aceitação capturada.
- Baseline Swift anterior: 34,99 tok/s. Esse valor não foi repetido no ensaio de TRIM.

A conta abaixo é um orçamento de latência, não uma previsão de desempenho GPU:

```text
Tokens por ciclo da referência = 512 / 182 = 2,813187
Ciclo para empatar com 34,99 tok/s = 80,399737 ms
Restante do ciclo real = 218,663 - 157,771 = 60,892 ms
Orçamento restante para o executor Xbox = 19,507737 ms
```

Sob a mesma aceitação e os mesmos demais custos, o executor GPU precisaria ficar abaixo de aproximadamente 19,5 ms por ciclo, incluindo seus novos custos de cópia, submissão e sincronização. Isso representa cerca de 8,1 vezes menos tempo que o compute CPU registrado. Recalcular o orçamento após repetir o baseline e a aceitação; ele não é um limite universal. O mock também não é uma promessa de TPS nem um teto para arquiteturas futuras.

### 3.1 Código e modelo

- `dedicated.cpp` fixa `n_gpu_layers=0`. A biblioteca `ggml-uwp` compila somente CPU; mudar esse parâmetro não cria um backend GPU.
- O contexto dedicado usa `n_ctx=49152`, batch/ubatch 64, quatro threads e KV Q4_0. Esses parâmetros são a referência inicial de semântica.
- O log de carga do modelo registra cinco blocos, hidden size 5120, FFN 17408, 32 heads Q, oito heads KV e head dimension 128. O drafter tem aproximadamente 1,92 bilhão de parâmetros; o target continua sendo o modelo de 27B.
- O GGUF Q3_K_M contém 32 tensores F32, 34 Q3_K, 13 Q4_K e dois Q5_K. Não implementar somente Q3_K ou somente o shader Q4_K existente.
- SYNC injeta, por posição, 25.600 floats oriundos das cinco camadas do target. DRAFT recebe embeddings e devolve hidden states F32 de largura 5120.
- DRAFT aceita `n_max` de 1 a 7; processa `n_max + 1` posições. Com `n_max=4`, são cinco posições. O HELLO atual não anuncia um campo de máximo de drafts.
- SYNC aceita de uma a 64 posições. O grafo tem convolução dinâmica, RMSNorm, RoPE, atenção, SwiGLU, residuais e escrita/leitura do KV.

Geometria e tipos vêm do [log de carga preservado](./evidence/dflash-core-after-swift-p70-1.0.0.7.log), produzido com o app 1.0.0.7. Usá-lo como inventário histórico e reconfirmar o hash e os metadados do GGUF na execução; ele não é uma medição de memória GPU da versão 1.0.0.10.

### 3.2 O que os testes do xllama permitem concluir

Os testes HLSL existentes demonstram execução D3D12 e fornecem infraestrutura aproveitável. O [CSV de GEMV Q4_K](../../bench/results/phase15-gpugemv-h62.csv) registra mediana de 25,4 GB/s no Series S, abaixo da meta de 40 GB/s daquela campanha. Isso não mede DFlash nem Series X.

O shader chamado `wave32` usa grupos de 32 threads e redução em memória compartilhada. O campo `wave_ops=0` desse teste informa que ele não utilizou intrínsecos de wave; não prova ausência de suporte do hardware. Consultar capacidades do dispositivo antes de selecionar variantes.

A documentação local generaliza que DirectML não teria fusão INT4 até 1.15.x. O [histórico oficial](https://learn.microsoft.com/en-us/windows/ai/directml/dml-feature-level-history) documenta INT4 Dequantize + GEMM fusion desde 1.15.0. O plano deve medir se essa fusão se aplica no Xbox e no grafo escolhido; uma sequência Dequantize + GEMM no código ORT, isoladamente, não prova materialização intermediária após compilação do grafo.

## 4. Arquitetura proposta

```text
.57: target + KV principal + embeddings + head + selector + aceitação
  |
  | DFL2 v2 persistente: SYNC_TRIM e DRAFT
  v
Xbox DFlash RPC
  validação de mensagens e estado da conexão
  executor CPU de referência OU executor GPU selecionado no início
    pesos GGUF residentes
    KV do drafter persistente
    buffers intermediários reutilizados
    D3D12/HLSL + operadores DirectML escolhidos por medição
  |
  | hidden states F32; mesmos formatos e ordem da referência
  v
.57: projeção/selector, verificação e confirmação dos tokens
```

### 4.1 Limite do executor

Criar um executor específico para o DFlash usado pelo RPC, com uma interface interna pequena para carregar o modelo, resetar, cortar o sufixo, executar SYNC e executar DRAFT. Reusar o leitor GGUF/ggml disponível; não criar outro parser ou um framework genérico de inferência.

Conservar o caminho CPU original como referência. A seleção de executor será feita antes de iniciar a sequência e permanecerá fixa durante a conexão. Uma configuração local proposta é `LocalState\dflash-backend.txt`, com `cpu` como padrão e `d3d12` como opção explícita. Esse arquivo e seus valores ainda precisam ser implementados; não são uma interface existente.

DirectML é uma opção interna do executor D3D12, inicialmente habilitada apenas no comparativo. A escolha de kernel por operação deve ser pequena, explícita e registrada no log. Não implementar um autotuner permanente antes de haver evidência de benefício.

### 4.2 Memória e execução

- Carregar e, se necessário, reorganizar os pesos uma vez. Preservar os valores representados pelo GGUF; conversão de layout não deve virar requantização silenciosa.
- Reutilizar buffers e descritores entre ciclos. Não recriar device, pipeline state, listas auxiliares ou grafos DirectML por chamada.
- Manter o KV e as ativações intermediárias na GPU. Fazer upload das entradas RPC e readback apenas das saídas necessárias ao protocolo.
- Agrupar dispatches por SYNC ou DRAFT, com barreiras corretas e sem espera CPU entre operadores. Medir submissão, fences e cópias; timestamps de kernels não substituem a duração observada pelo cliente.
- Iniciar com acumulação F32 e política de precisão explícita. Testar FP16 ou fusões adicionais como variantes separadas após obter a referência correta.
- Contabilizar pesos, KV, temporários, upload, readback, alocações DirectML e cópias CPU. Pesos FP16 e KV FP16 podem expandir significativamente a memória; não assumir que o budget nominal garante carga ou inferência.
- Não manter dois modelos e dois caches completos residentes apenas para ter fallback. A recuperação pode recarregar o executor CPU antes de uma nova sequência.
- Preservar inicialmente o contexto e a semântica Q4_0 do KV. Um cache F16 temporário é experimento identificado, com medição de memória e qualidade própria.
- Avaliar cache circular usando a janela SWA somente depois da paridade do cache inicial. Manter posições absolutas para RoPE; não truncar o contexto ou mudar a janela implicitamente.

### 4.3 Compatibilidade e falhas

Preservar os tamanhos dos headers e os payloads DFL2 v2. Manter HELLO, RESET, SYNC, DRAFT, TRIM, MOCK e SYNC_TRIM, incluindo a sondagem vazia de SYNC_TRIM sem mutação. Métricas novas devem ir primeiro para JSONL local, correlacionadas por conexão e `cycle_id`; não aumentar silenciosamente structs da resposta.

Continuar retornando erro para entradas inválidas antes de alterar o estado. Confirmar a conclusão da GPU antes de responder sucesso de uma mutação. Na primeira versão, não introduzir pipelining entre ciclos.

Fallback automático para CPU é permitido quando a GPU falha na inicialização, antes de criar estado de sequência, e deve aparecer no log. Se ocorrer falha ou remoção do dispositivo após uma mutação GPU, invalidar a sequência e fechar a conexão com erro: não continuar na CPU com um KV incompleto. Recuperação exige reinicialização e replay consistente das entradas confirmadas.

## 5. Mapa de arquivos

Os nomes de novos arquivos abaixo são propostos. Criar somente os necessários à etapa em execução.

| Arquivo ou diretório                               | Papel                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------- |
| `uwp/dflash-rpc/dedicated.cpp`                     | Separar o cálculo do tratamento RPC e preservar a ordem das mutações            |
| `uwp/dflash-rpc/dedicated_protocol.h`              | Referência de ABI; manter compatibilidade v2                                    |
| `uwp/dflash-rpc/DFlashRpc.vcxproj`                 | Incluir executor, shaders compilados e dependência DirectML apenas quando usada |
| `uwp/ggml-uwp.vcxproj`                             | Manter biblioteca CPU e acesso à infraestrutura GGUF/ggml                       |
| `uwp/dflash-rpc/dflash-engine.h`                   | Nova interface interna mínima do executor                                       |
| `uwp/dflash-rpc/dflash-gpu.cpp`                    | Novo executor D3D12 e gerenciamento de recursos                                 |
| `uwp/dflash-rpc/dflash-dml.cpp`                    | Comparativo e operadores DirectML, se justificados                              |
| `shaders/dflash/`                                  | Novos kernels e variantes de precisão explicitadas                              |
| `shaders/generated/`                               | Bytecode AOT gerado pelo fluxo de compilação, sem edição manual                 |
| `include/xllama/d3d12_dyn.h`                       | Resolver o runtime D3D12 do sistema                                             |
| `src/bridge/gpugemv.cpp` e `gpubw.cpp`             | Referência de criação de recursos, timers e verificação                         |
| `llama.cpp/src/models/dflash.cpp`                  | Especificação executável do grafo e das operações; evitar mudanças semânticas   |
| `research/dflash2-xbox/test_trim_protocol.py`      | Estender os testes existentes de estado e entradas inválidas                    |
| `research/dflash2-xbox/dflash_persistent_smoke.py` | Smoke por conexão, reset e repetição                                            |
| `research/dflash2-xbox/dflash_dedicated_bench.py`  | Base do benchmark de RPC; dados sintéticos não substituem replay real           |
| `research/dflash2-xbox/gpu-implementation/`        | Novos manifests, fixtures, traces, CSVs, hashes e relatório da campanha         |

## 6. Etapas e critérios de avanço

### F0 — Congelar referência e preparar entradas reais

1. Registrar commits base, diffs, hashes dos fontes, pacote, modelo e versão dos compiladores. Identificar alterações alheias e preservar os artefatos antigos.
2. Conferir o inventário exato dos tensores: nome, dimensões, tipo, strides, bytes e participação em SYNC/DRAFT.
3. Reusar a captura existente de saídas e verificar se há entradas completas correspondentes. Quando faltarem, instrumentar a captura de SYNC, embeddings DRAFT, posições e operações de estado. Não tratar captura de saída como replay completo.
4. Criar fixtures determinísticas pequenas e fixtures de chamadas reais, com hashes e referência CPU. Incluir pré-preenchimento, rejeição parcial, rejeição total, reset e reposicionamento.
5. Medir por grupo de operadores na CPU: projeção SYNC, QKV, atenção, FFN, convolução dinâmica e manutenção de KV. Registrar o custo da instrumentação.
6. Preservar um build/pacote CPU funcional e os scripts de instalação/retorno. A repetição do baseline target sozinho entra na F6, antes de qualquer conclusão de performance.

**Entrega:** manifest da referência, inventário de tensores e replay de sequência reproduzível.

**Critério de avanço:** CPU reproduz a referência com os mesmos dados e operações; entradas e saídas estão correlacionadas; nenhuma medição de GPU é declarada nesta etapa.

### F1 — Confirmar o caminho D3D12 no Series X

1. Adaptar a infraestrutura existente para um teste dentro do app RPC, acionado explicitamente e separado da geração normal.
2. Criar o dispositivo pelo runtime do sistema e excluir adapter de software. Não depender de Agility SDK para iniciar o app.
3. Registrar adapter, feature level, shader model, WaveOps e limites relevantes. Não fixar intrínsecos ou tamanho físico de wave pelo nome RDNA2.
4. Compilar shaders offline com alvo compatível com as capacidades verificadas. Preservar a variante sem intrínsecos de wave.
5. Medir upload, dispatch, readback e fence, com checksum de saída e timestamps GPU quando disponíveis. Se timestamps não forem suportados, registrar medição de parede sincronizada e sua limitação.
6. Registrar classificação Game/App, foreground, memória da aplicação e orçamento/alocação GPU observados. Não inferir a classificação pelo manifesto nem a capacidade pelo total físico do console.

**Entrega:** probe no Series X com dados íntegros, capacidades e tempos reais.

**Critério de avanço:** execução em GPU confirmada, resultado correto e recursos liberados ao encerrar. Build ou HTTP 202 isolados não aprovam esta etapa.

### F2 — Comparar os kernels com pesos e shapes reais

Implementar primeiro as multiplicações responsáveis pela maior parcela medida na F0. O comparativo deve incluir Q3_K, Q4_K, Q5_K e F32, priorizando os tipos efetivamente dominantes.

| Caso             | Dimensões e variantes                                                           |
| ---------------- | ------------------------------------------------------------------------------- |
| Projeção de SYNC | Entrada 25.600, saída 5120; lotes 1, 2, 4, 8, 16, 32 e 64                       |
| FFN              | Projeções 5120/17408, nos dois sentidos; confirmar orientação e strides no GGUF |
| Atenção          | Matrizes Q/K/V/O reais do inventário; não presumir que todas sejam quadradas    |
| DRAFT            | `n_max=1,2,4,6,7`, portanto lotes 2, 3, 5, 7 e 8                                |

Comparar:

1. HLSL quantizado, com dequantização dos blocos em registradores e reutilização dos pesos entre posições do lote.
2. DirectML direto com pesos convertidos uma vez para FP16/F32, se a alocação couber. Medir memória e custo inicial separadamente do custo por ciclo.
3. Fusão INT4 DirectML somente quando houver representação equivalente dos valores e suporte confirmado. Os layouts GGUF K não são tensores INT4 DirectML prontos; não converter Q3_K_M para outra quantização sem identificar a alteração de modelo.

Usar o mesmo peso e a mesma entrada em cada comparação. Registrar latência da operação e custo completo da fronteira CPU/GPU. Fazer pelo menos 50 aquecimentos e 200 iterações medidas por configuração selecionada, com três repetições independentes e ordem alternada dos candidatos. Medir a carga fria separadamente.

**Entrega:** CSV por tipo/shape, erro numérico, memória, mediana, p95 e tempo de transferência/submissão.

**Critério de avanço:** escolher o candidato correto mais rápido nas operações dominantes. Se nenhum superar a CPU no trabalho representativo incluindo custos inevitáveis, limitar o trabalho a um ajuste orientado pelo profiler e reavaliar antes do porte integral. Microbenchmark vencedor é motivo para prosseguir, não aprovação de TPS final.

### F3 — Implementar KV e SYNC na GPU

1. Implementar a alocação persistente e o mapeamento lógico de posições do KV do drafter, com a semântica da referência.
2. Portar projeção 25.600 → 5120, normalização e projeções K/V das cinco camadas, incluindo normalização de K e RoPE.
3. Implementar escrita e leitura Q4_0 ou usar uma variante temporária de precisão explicitamente identificada. Validar quantização e layout separadamente do cálculo da atenção.
4. Implementar RESET, TRIM e SYNC_TRIM na ordem já usada: validar pedido, remover sufixo ou resetar, injetar features, concluir execução e só então responder sucesso.
5. Evitar limpar ou copiar o cache inteiro em cortes de sufixo quando a semântica permitir invalidar posições. Demonstrar por teste que bytes antigos nunca voltam a participar da atenção.
6. Confirmar consistência entre SYNC de 64 posições e divisão em lotes menores, seguindo a referência CPU.

**Entrega:** replay de SYNC e operações de cache executado na GPU, com comparação de estado observável e saídas posteriores.

**Critério de avanço:** equivalência nas sequências de reset/corte/reinjeção e ausência de leitura de posições inválidas, incluindo limites de contexto e da janela SWA.

### F4 — Implementar DRAFT completo na GPU

1. Reproduzir escala e layout dos embeddings fornecidos pela `.57`, mantendo a âncora e os MASKs.
2. Portar RMSNorm, convolução dinâmica, QKV, RoPE, atenção com KV, projeção O, FFN/SwiGLU, residuais e normalização final.
3. Tratar a convolução dinâmica como a operação definida pelo código DFlash, incluindo geração dos coeficientes por token/grupo. Não substituí-la por Conv1D estática.
4. Reproduzir a máscara de atenção e a interação entre posições históricas e o bloco de drafts; não aplicar automaticamente uma máscara causal de decoder convencional.
5. Preservar a substituição do sufixo a partir da posição da âncora antes do DRAFT. Validar o DRAFT seguinte após cada possível padrão de aceitação.
6. Retornar os mesmos hidden states F32 e a mesma ordem de posições. Embeddings do target, logits, output head e selector continuam na `.57`.
7. Manter todo o bloco na GPU. Execução híbrida CPU/GPU por operador é aceitável como instrumento temporário de diagnóstico, claramente identificada no log.

**Entrega:** DRAFT e SYNC completos, incluindo cache, no executor GPU.

**Critério de avanço:** replay de sequência completo correto e nenhum fallback CPU oculto nos resultados apresentados como execução GPU.

### F5 — Integrar ao RPC com recuperação verificável

1. Conectar o executor à interface interna e à seleção local, preservando a compatibilidade v2 e o comportamento dos clientes antigos.
2. Registrar backend solicitado/efetivo, motivo de fallback, precisão, kernels e tempos por operação, conexão e ciclo.
3. Preservar a semântica temporal de `xbox_compute_us` como duração de parede da execução no servidor; documentar a fronteira exata e adicionar tempos GPU em logs separados. Não tratar esse campo como timestamp GPU.
4. Testar conexão nova, desconexão no meio de uma requisição, timeout, reset após erro e pedidos malformados. Validar limites e tamanhos antes de qualquer mutação.
5. Simular falha de inicialização e de execução do backend em testes controlados. Confirmar fallback antes da sequência e invalidação/fechamento após falha de estado.
6. Verificar liberação de recursos após ciclos repetidos de conexão e reset. Distinguir crescimento de caches estabilizado de vazamento.

**Entrega:** pacote experimental com CPU padrão, GPU opt-in e logs suficientes para atribuir falhas.

**Critério de avanço:** suíte de protocolo aprovada nos dois executores e recuperação sem continuação com cache divergente.

### F6 — Medir geração real e estabelecer viabilidade

Repetir no mesmo período os modos abaixo, usando o mesmo modelo target, GGUF do drafter, prompt, seed, sampling, contexto e configuração de produção aplicável:

| Modo                                             | Finalidade                                                                 |
| ------------------------------------------------ | -------------------------------------------------------------------------- |
| Swift sozinho, configuração do baseline anterior | Baseline de benefício para o usuário; registrar MTP e demais opções exatas |
| DFlash RPC CPU otimizado                         | Controle direto contra o app 1.0.0.10                                      |
| DFlash RPC GPU                                   | Candidato de implementação                                                 |
| Mock Xbox com trajetória CPU                     | Controle de transporte/host e comparação com o relatório anterior          |
| Mock Xbox com trajetória GPU, se ela diferir     | Separar efeito do compute de mudanças de aceitação                         |

Começar pelo workload original: prompt `Write a short paragraph explaining why the sky appears blue.`, `n_max=4`, `p_min=0`, 512 tokens, seed 1234, temperature 1.0, top_p 0.95, top_k 20, `cache_prompt=false`, `ignore_eos=true`, contexto target 8192 e tipos KV da referência.

Executar primeiro smoke curto e depois pelo menos três rodadas intercaladas dos modos completos. Não selecionar o melhor run. Expandir somente após essa aprovação para prompts de texto e código, `n_max=2/4/6/7` e contextos 8k/24k/40k quando o modelo e a memória permitirem. `n_max=8` excede o protocolo atual; não incluí-lo como configuração suportada.

Registrar por requisição e por ciclo:

- TTFT, duração total, tokens gerados e TPS de decode, com denominadores explícitos.
- DRAFT/SYNC completos, compute GPU, upload/readback, submissão/fence e preparação CPU.
- Verificação target, selector, sampling, commit, bytes transportados e chamadas RPC.
- Propostas, aceites, distribuição de tokens aceitos por ciclo, rejections e resets.
- Média, mediana, p95, mínimo/máximo e resultados de cada repetição.
- Memória da aplicação e GPU, estabilidade das alocações, backend efetivo e erros.

Os timers do servidor e do cliente podem se sobrepor; não somar linhas aninhadas ou inferir latência física de ida/volta com relógios não sincronizados. Distinguir TPS de decode do throughput de parede que inclui prefill. Aplicar o mesmo critério de inclusão de ciclos a todos os modos.

**Entrega:** relatório gerado a partir dos traces e tabela comparativa com os novos baselines.

**Critério de benefício demonstrado:** GPU supera o RPC CPU e o Swift sozinho nas três rodadas do workload de referência, sem falhas de correção ou estabilidade. Se o resultado ficar dentro da variação entre runs, ampliar a amostra de forma documentada antes de declarar ganho. Outros workloads recebem seus próprios veredictos; não generalizar uma vitória isolada.

### F7 — Otimizar o gargalo que restar

Depois de medir o pipeline completo, priorizar pela parcela de tempo real:

1. Fusões dentro do DFlash, reutilização de pesos entre posições e redução de buffers/barreiras.
2. Atenção/KV e armazenamento pela janela SWA, com testes de borda e posições absolutas.
3. Variantes de precisão ou DirectML por operador, se melhorarem o custo completo e preservarem a validação.
4. Ajuste de `n_max` e `p_min` em campanha separada, depois do A/B com parâmetros fixos.
5. Cópias e espera RPC remanescentes, apenas quando relevantes no novo perfil.

Se o target passar a dominar, registrar o limite e abrir trabalho separado na `.57`. Não atribuir ao backend Xbox um ganho obtido alterando target, modelo, KV ou sampling no mesmo ensaio.

## 7. Validação numérica e de estado

### 7.1 Referências e limites

Usar três níveis de comparação:

1. **Formato:** índices, escalas, dequantização e layout dos blocos GGUF e KV contra rotinas de referência. Verificar limites, padding, strides e valores não finitos.
2. **Operadores:** mesmas entradas e pesos contra referência CPU de matemática equivalente, com acúmulo F32 ou F64 de diagnóstico. A CPU GGML pode quantizar ativações internamente; não confundir essa diferença com erro de decodificação do peso na GPU.
3. **Sequência:** hidden states, decisões do selector, aceitação, cortes e saídas do pipeline completo, comparados com a execução CPU original.

Registrar erro absoluto máximo, erro relativo com denominador protegido, NMSE, similaridade de cosseno e NaN/Inf. NaN/Inf inesperado, acesso fora de limites ou formato incorreto são falhas imediatas.

Na F0, fixar em um manifest as tolerâncias por operador e precisão, derivadas de uma referência controlada, antes de escolher o candidato mais rápido. Justificar cada limite e mantê-lo igual entre candidatos da mesma precisão. Não relaxar tolerâncias automaticamente para aprovar um kernel. A aprovação numérica não substitui a validação do pipeline e da aceitação.

Bit a bit continua sendo a comparação do caminho CPU inalterado e dos replays capturados que prometem exatamente o mesmo cálculo. Não impor identidade binária CPU/GPU a reduções com ordem diferente. Alterações de texto em sampling estocástico, isoladamente, não provam erro nem correção: identificar a primeira divergência e testar também um modo determinístico cujo comportamento de referência esteja definido.

### 7.2 Casos obrigatórios de estado

- RESET e repetição da mesma sequência.
- TRIM antes, na posição e depois da âncora; corte para zero e reset completo.
- SYNC_TRIM com remoção e com reset; sondagem vazia sem alterar estado.
- Aceitação nula, parcial e total, seguida de novo DRAFT.
- Escrita sobre posições previamente descartadas, sem reutilizar valores obsoletos.
- SYNC com 1 e 64 posições, fracionamento em lotes e bordas da janela/contexto.
- DRAFT com todos os limites suportados, incluindo sete drafts/oito posições.
- Tamanhos incorretos, posições negativas e limites excedidos, rejeitados sem mutação parcial.
- Conexão nova após falha, desconexão e device removal simulado.

## 8. Build, instalação e operação

Reusar o build MSVC existente, com `C:\BuildTools` e SDK 26100 conforme o script. Compilar HLSL offline e incluir o bytecode no pacote. Se DirectML for incorporado, fixar a versão utilizada e confirmar a inclusão da DLL no MSIX; não atualizar simultaneamente toda a pilha ORT/GenAI.

Os comandos abaixo existem no projeto e foram conferidos durante a preparação do plano. São comandos para a execução futura, não resultados obtidos neste pedido. Executar a partir da worktree e criar o diretório de evidências antes de gravar nele.

```powershell
Set-Location 'C:\Users\hjotha\worktrees\xllama-dflash-xbox'
git status --short --branch
git worktree list --porcelain
New-Item -ItemType Directory -Force -Path 'research\dflash2-xbox\gpu-implementation'
& '.\research\dflash2-xbox\build-dflash-rpc.cmd'
```

Após instalar o pacote selecionado e confirmar o servidor no Xbox, executar os testes existentes. A aprovação bit a bit de TRIM compara sequências equivalentes no mesmo backend; uma falha com o novo executor deve ser investigada e não contornada com uma tolerância global.

```powershell
python '.\research\dflash2-xbox\test_trim_protocol.py' --host 192.168.1.26
python '.\research\dflash2-xbox\dflash_persistent_smoke.py' --host 192.168.1.26 --port 50053 --cycles 4 --drafts 4 --output '.\research\dflash2-xbox\gpu-implementation\persistent-smoke.json'
python '.\research\dflash2-xbox\dflash_dedicated_bench.py' --host 192.168.1.26 --port 50053 --warmup 50 --iterations 100 --sync-tokens 1 --drafts 4 --output '.\research\dflash2-xbox\gpu-implementation\rpc-smoke-bench.json' --cycles-output '.\research\dflash2-xbox\gpu-implementation\rpc-smoke-cycles.jsonl'
```

Esse benchmark usa entradas sintéticas e serve como smoke/probe. O replay real e o comparativo por kernel ainda precisam de implementação; não há comandos fictícios para eles neste plano. A suíte inicial pode ser ampliada para 1.000 ciclos depois de aprovar correção e estabilidade.

Para instalar e medir:

1. Inspecionar estado atual do console, pacote, classificação e serviços antes de agir. Reusar os helpers de Device Portal existentes, conferindo seus argumentos e efeitos na versão atual.
2. Usar versão de pacote crescente; preservar identidade, modelo em LocalState e pacote CPU de referência. Não desinstalar o app para contornar uma atualização.
3. Após deploy, verificar versão instalada, processo, listener e requisição real. HTTP 202 ou build aprovado não bastam.
4. Antes de usar scripts antigos da `.57`, ler seu fluxo de parada e restauração de produção. Não executá-los cegamente. Capturar a configuração ativa e respeitar trabalho em andamento.
5. Executar a campanha em processo finito, com timeout e limpeza garantida. Evitar polling de status em loop. Se a campanha precisar continuar fora da CLI, usar um job independente com resultado e notificação reais.
6. Ao terminar, desligar mock, encerrar o servidor de teste e verificar o estado restaurado do serviço original e `/health`. Se a execução tiver suspendido outro app do Xbox, restaurá-lo e confirmar seu estado conforme o que estava ativo antes.
7. Recuperação preferencial: voltar ao executor CPU por configuração e reiniciar uma sequência limpa. Caso seja necessário reinstalar um pacote de recuperação, respeitar as regras de versão do Device Portal sem perder LocalState.

## 9. Artefatos e atribuição de evidência

Cada campanha deve gravar uma pasta própria sob `gpu-implementation`, sem sobrescrever os ensaios de setembro já existentes. O manifest relaciona os arquivos gerados e seus hashes.

Entregáveis mínimos:

- Manifest com modelos, fontes, pacote, backends, precisão, configuração, capacidades e timestamps.
- Inventário de tensores e fixtures com entrada/saída, sequência de operações e hashes.
- Resultados por kernel e por operador, com tempos de GPU e tempos completos separados.
- Resultados de protocolo, cache, falha/recuperação e validação numérica.
- Traces por ciclo, respostas completas, estatísticas por run e relatório comparativo.
- Registro de instalação efetiva e do estado operacional ao finalizar.

Usar estados explícitos no relatório: implementado, compilado, instalado, executado na GPU, numericamente validado, protocolo validado e benefício medido. Não promover automaticamente um estado ao seguinte.

## 10. Riscos e decisões condicionais

| Risco                                                          | Resposta planejada                                                                            |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Matmul GPU lento para lote pequeno                             | Comparar shapes reais na F2; evitar investir no grafo completo sem ganho representativo       |
| DirectML não seleciona fusão ou converte pesos a cada execução | Medir grafo compilado e custo completo; comparar com HLSL residente                           |
| GGUF K incompatível com INT4 DirectML direto                   | Preservar valores e layout explícito; usar HLSL ou conversão única validada                   |
| FP16 aumenta demais o uso de memória                           | Medir alocação e pico; manter pesos quantizados e evitar duplicação                           |
| Erro em RMSNorm no contexto de execução                        | Validar operador e grafo integrado; usar decomposição/HLSL se necessário                      |
| Cache parece correto até ocorrer rejeição                      | Replay com todos os pontos de corte e novo DRAFT após cada mutação                            |
| Kernels rápidos, mas sincronização elimina o ganho             | Medir upload/readback/fence e agrupar o grafo, mantendo intermediários na GPU                 |
| Aceitação muda e mascara a comparação                          | Registrar trajetórias; usar replay fixo para isolamento e geração real para o benefício final |
| Target passa a dominar                                         | Publicar o novo perfil e tratar a otimização do target em trabalho separado                   |

O histórico de RMSNorm no xllama localizou uma falha no contexto ORT-GenAI/DirectML, enquanto probes plain-ORT passaram. Não classificar todo DirectML como numericamente incorreto com base nesse episódio; usar o [relatório específico](../../docs/dml-rmsnorm-fix-runbook.md).

## 11. Critérios de conclusão

- [ ] Executor GPU executa SYNC e DRAFT completos no Series X com backend efetivo comprovado.
- [ ] Protocolo v2 e SYNC_TRIM continuam compatíveis, com rejeição de pedidos inválidos antes de mutar estado.
- [ ] Tipos de pesos e precisão real estão documentados; não há mudança silenciosa de modelo ou KV.
- [ ] Operadores, hidden states, aceitação, cortes e recuperação foram validados.
- [ ] Pesos e KV permanecem residentes sem vazamento ou cópias completas recorrentes.
- [ ] Builds, pacotes e dados da campanha são reproduzíveis pelos manifests e hashes.
- [ ] CPU fallback funciona em sequência nova e erros GPU não deixam continuação com estado divergente.
- [ ] Baseline Swift e RPC CPU foram repetidos junto ao candidato; ganho real é apresentado por workload, com variação entre runs.
- [ ] Produção e console terminam no estado operacional definido para o encerramento, com mock desligado e evidência de saúde.

Se a GPU funcionar corretamente mas não vencer o target sozinho, registrar **implementação funcional sem benefício final demonstrado**. Não declarar o objetivo de performance concluído nesse caso.

## 12. Ordem de execução recomendada

```text
F0 Referência e replay
  -> F1 Capacidades e execução D3D12 no Series X
  -> F2 Comparativo de kernels reais e decisão HLSL/DirectML
  -> F3 SYNC + KV
  -> F4 DRAFT completo
  -> F5 RPC + recuperação
  -> F6 Baselines e geração real
  -> F7 Otimização guiada pelo novo perfil
```

A primeira entrega de implementação deve ser F0–F2: dados reproduzíveis, execução GPU confirmada e comparação das multiplicações reais. Ela decide onde investir no porte completo. Não estabelecer prazo ou TPS prometido antes dessa medição.

## 13. Fontes

- [Relatório TRIM/rewind, versão 1.0.0.10](./trim-rewind-optimization-2026-09-28.md).
- [Hashes dos fontes medidos](./trim-optimization-2026-09-28/source-hashes.json).
- [Servidor dedicado](../../uwp/dflash-rpc/dedicated.cpp), [protocolo](../../uwp/dflash-rpc/dedicated_protocol.h) e [grafo DFlash](../../llama.cpp/src/models/dflash.cpp).
- [Infraestrutura GEMV D3D12](../../src/bridge/gpugemv.cpp) e [shader Q4_K](../../shaders/gpugemv_q4k_wave32.hlsl).
- [Pesquisa GPU do xllama](../../docs/phase15-re-opt.md) e [restrições UWP](../../docs/uwp-constraints.md), lidas à luz do código e das ressalvas registradas acima.
- [DirectML: recursos, command lists e integração D3D12](https://learn.microsoft.com/en-us/windows/ai/directml/dml).
- [DirectML: histórico de feature levels e fusão INT4](https://learn.microsoft.com/en-us/windows/ai/directml/dml-feature-level-history).
- [ONNX Runtime DirectML: limitações e otimização por shapes](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).
- [Recursos UWP no Xbox](https://learn.microsoft.com/en-us/previous-versions/windows/uwp/xbox-apps/system-resource-allocation); confirmar o orçamento e as capacidades no console, não extrapolar a partir da ficha física.
