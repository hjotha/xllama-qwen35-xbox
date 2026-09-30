# Profiling Swift/ISTA + Xbox DFlash RPC (2026-09-27)

## Configuração e validade

- Target: Swift 1.5 / ISTA Qwen3.8-27B GSQ-RCO IQ3_XXS-mtp na `.57`, RTX 4070, contexto 8192, KV target `kvarn4`, paralelismo 1.
- Drafter: DFlash2 Q3_K_M no `HJotha.XboxDFlashRpc` 1.0.0.9, CPU, KV Q4_0; output head e seletor na `.57`.
- Prompt: `Write a short paragraph explaining why the sky appears blue.`; seed 1234; temperatura 1.0, top-p 0.95, top-k 20, `cache_prompt=false`, `ignore_eos=true`, `n_max=4`, `p_min=0`, `n_predict=512`.
- A: DFlash real; B: Xbox reproduz os hidden states capturados em A, preserva RPC e elimina `llama_decode` em DRAFT e SYNC; C: a `.57` reproduz os mesmos hidden states sem RPC por ciclo. Posição, token âncora e tamanho do bloco são validados no replay.
- Os três modos completaram 512 tokens e os mesmos 182 ciclos, 728 propostas, 329 aceites (45.19%), sem falha RPC; hash SHA-256 do texto final: `31e48d4676ed512e3ec25ad9caea71cc6ee26c15de6f7b7bceb7b87981e218dc`. A trajetória de posição/propostas/aceites também é idêntica.
- Tabelas por etapa usam 180 ciclos completos estáveis: excluem o primeiro, que inclui prefill, e o último, que não tem início de ciclo seguinte. TPS vem do tempo de geração integral dos 512 tokens. Um ensaio por modo, em ordem A/B/C; variação de clock e temperatura entre ensaios não foi controlada por repetições.

## Resultado direto

| Modo                     | Geração 512 tokens | TPS final | Ciclo completo médio |    Mediana |        p95 |
| ------------------------ | -----------------: | --------: | -------------------: | ---------: | ---------: |
| A: DFlash real no Xbox   |           53.094 s |      9.62 |           291.158 ms | 298.040 ms | 306.899 ms |
| B: RPC Xbox + MOCK_DRAFT |           23.646 s |     21.61 |           129.401 ms | 137.565 ms | 141.021 ms |
| C: replay local na .57   |            8.754 s |     58.38 |            47.630 ms |  47.316 ms |  49.153 ms |

O rendimento real é **2.813 tokens finais por ciclo**. O baseline Swift sozinho anterior foi 34.99 tok/s, então o ciclo completo deve cair abaixo de **80.400 ms** com a aceitação atual.

## Onde estão os ~149 ms fora do DRAFT RPC

Em A, ciclo completo 291.158 ms menos DRAFT RPC 142.167 ms = **148.991 ms**. A divisão direta é:

| Etapa fora do DRAFT RPC             | Média ms | % do ciclo A |
| ----------------------------------- | -------: | -----------: |
| Verificação target na RTX 4070      |   43.199 |       14.84% |
| TRIM RPC                            |   42.251 |       14.51% |
| Rewind RPC (139 de 180 ciclos)      |   32.000 |       10.99% |
| SYNC RPC, inclusive compute Xbox    |   22.273 |        7.65% |
| Seletor host                        |    5.799 |        1.99% |
| Sampling                            |    1.761 |        0.60% |
| Demais preparações, commit e espera |    1.707 |            — |

## Custos isolados e decisão

- **DFlash no Xbox:** DRAFT `llama_decode` 139.260 ms/ciclo (p95 144.784); SYNC `llama_decode` 16.655 ms. Total direto **155.915 ms/ciclo**. A−B mediu 161.802 ms/ciclo, incluindo pequenas variações nas demais etapas.
- **Rede/RPC:** soma direta dos componentes sem compute em A **81.439 ms/ciclo** (DRAFT 1.606, SYNC 5.582, TRIM 42.251, rewind 32.000). B−C mediu 81.824 ms/ciclo.
- **Target verification:** `llama_decode` + sincronização RTX 4070 **43.199 ms/ciclo**, p95 44.617. Submit 2.774; espera da GPU 40.425 ms.
- **Commit/KV/SYNC:** commit + remoção de KV target 0.180 ms. Captura + serialização SYNC 0.242 ms. SYNC RPC 22.273 ms (compute Xbox 16.655); TRIM + rewind 74.251 ms/ciclo.
- **Piso medido com DFlash=0 e RPC preservado:** B, 129.401 ms/ciclo e 21.61 tok/s. O RPC atual sozinho ainda fica abaixo do baseline 34.99 tok/s.
- **Piso medido sem RPC por ciclo:** C, 47.630 ms/ciclo e 58.38 tok/s. É o teto observado com a mesma aceitação se compute remoto e RPC por ciclo fossem nulos; não é limite absoluto do hardware.
- Para superar o baseline, A precisa reduzir pelo menos 210.758 ms/ciclo. Mesmo B precisa reduzir 49.001 ms/ciclo. Eliminar apenas o compute Xbox não basta.

## Três maiores gargalos

1. Compute DFlash Xbox, DRAFT + SYNC: **155.915 ms/ciclo** (53.55%).
2. TRIM + rewind do KV por RPC: **74.251 ms/ciclo** (25.50%). Em B, sem compute, TRIM ainda recebe o payload de 4 bytes no Xbox em média 40.760 ms. O cliente faz dois `send()` sem TCP_NODELAY; atraso Nagle/ACK é uma inferência forte, não uma captura de pacotes.
3. Verificação target RTX 4070: **43.199 ms/ciclo** (14.84%). É o principal piso de compute no host após remover o drafter remoto.

A soma dos estágios exclusivos mais `other_host_wait` fecha 100% do ciclo. `other_host_wait` em A foi 1.178 ms (0.40%), portanto não há uma barreira grande sem atribuição.

## Limites da separação de rede

Os relógios monotônicos do host e Xbox não são sincronizados. Os tempos de cada lado e o round-trip permitem atribuir o conjunto rede/socket, mas não dividir fisicamente cada direção com precisão sub-ms. `header_wait` no host inclui inferência Xbox; não é uma medida só da volta. DRAFT host `send()` e leitura do payload aparecem abaixo apenas como diagnósticos, pois já pertencem ao transporte agregado.

| Diagnóstico A                            | Média ms | Mediana ms |  p95 ms |  Mín ms |  Máx ms | % ciclo |
| ---------------------------------------- | -------: | ---------: | ------: | ------: | ------: | ------: |
| Host, duas chamadas send() do DRAFT      |    0.136 |      0.132 |   0.175 |   0.074 |   0.249 |    0.05 |
| Host, leitura do payload de resposta     |    0.469 |      0.378 |   0.908 |   0.035 |   1.863 |    0.16 |
| Host, espera cabeçalho inclusive compute |  141.561 |    141.820 | 147.229 | 134.710 | 148.075 |   48.62 |

## Etapas exclusivas: A: DFlash real no Xbox

Valores por 180 ciclos estáveis. Percentual calculado sobre a média do ciclo completo do respectivo modo; as linhas somam 100% com arredondamento.

| Etapa                                           | Média ms | Mediana ms |  p95 ms |  Mín ms |  Máx ms | % ciclo |
| ----------------------------------------------- | -------: | ---------: | ------: | ------: | ------: | ------: |
| 1. Preparar request DRAFT na .57                |    0.064 |      0.064 |   0.071 |   0.038 |   0.187 |    0.02 |
| 2. Serializar DRAFT                             |    0.009 |      0.008 |   0.015 |   0.005 |   0.028 |    0.00 |
| 3/9/10. Transporte e sockets do DRAFT, agregado |    1.606 |      1.599 |   1.947 |   1.272 |   2.908 |    0.55 |
| 4. Receber payload DRAFT no Xbox                |    0.987 |      0.995 |   1.019 |   0.806 |   1.188 |    0.34 |
| 5. Preparar DFlash no Xbox                      |    0.137 |      0.139 |   0.154 |   0.122 |   0.167 |    0.05 |
| 6. Inferencia DFlash DRAFT no Xbox              |  139.260 |    139.506 | 144.784 | 132.897 | 145.622 |   47.83 |
| 8. Copiar/serializar resposta Xbox              |    0.056 |      0.060 |   0.104 |   0.022 |   0.136 |    0.02 |
| 9. Enviar resposta pelo socket Xbox             |    0.120 |      0.109 |   0.153 |   0.099 |   0.162 |    0.04 |
| 10. Copiar resposta para o seletor na .57       |    0.010 |      0.009 |   0.011 |   0.005 |   0.072 |    0.00 |
| 7. Seletor na .57                               |    5.799 |      5.711 |   6.411 |   5.034 |   9.787 |    1.99 |
| 11. Processar propostas                         |    0.011 |      0.010 |   0.014 |   0.006 |   0.018 |    0.00 |
| 18. Preparar batch de verificacao               |    0.003 |      0.002 |   0.004 |   0.001 |   0.004 |    0.00 |
| 12a. Submeter verificacao target                |    2.774 |      1.201 |  23.063 |   0.839 |  35.601 |    0.95 |
| 12b. Aguardar RTX 4070                          |   40.425 |     41.671 |  41.726 |  20.161 |  52.586 |   13.88 |
| 16a. Capturar features para SYNC                |    0.147 |      0.155 |   0.197 |   0.078 |   0.288 |    0.05 |
| 16b. Serializar SYNC                            |    0.094 |      0.095 |   0.140 |   0.045 |   0.281 |    0.03 |
| 16c. Transporte e sockets do SYNC               |    5.582 |      5.592 |   6.246 |   4.898 |   6.502 |    1.92 |
| 17a. Preparar SYNC no Xbox                      |    0.037 |      0.036 |   0.039 |   0.033 |   0.059 |    0.01 |
| 17b. Inferencia SYNC no Xbox                    |   16.655 |     16.611 |  16.787 |  16.469 |  22.118 |    5.72 |
| 15a. TRIM RPC do KV Xbox                        |   42.251 |     42.284 |  43.754 |  40.189 |  44.214 |   14.51 |
| 15b. Rewind RPC do KV Xbox                      |   32.000 |     40.924 |  43.010 |   0.000 |  44.082 |   10.99 |
| 13. Sampling/aceitacao                          |    1.761 |      1.457 |   3.096 |   0.613 |   3.694 |    0.60 |
| 14. Commit dos tokens aceitos                   |    0.002 |      0.002 |   0.002 |   0.001 |   0.003 |    0.00 |
| 15c. Remocao KV target na .57                   |    0.179 |      0.157 |   0.238 |   0.081 |   0.705 |    0.06 |
| 14b. Processar e emitir tokens finais           |    0.010 |      0.010 |   0.014 |   0.006 |   0.022 |    0.00 |
| 18. Espera, barreira e overhead restante        |    1.178 |      1.250 |   1.461 |   0.716 |   1.662 |    0.40 |

## Etapas exclusivas: B: RPC Xbox + MOCK_DRAFT

Valores por 180 ciclos estáveis. Percentual calculado sobre a média do ciclo completo do respectivo modo; as linhas somam 100% com arredondamento.

| Etapa                                           | Média ms | Mediana ms | p95 ms | Mín ms | Máx ms | % ciclo |
| ----------------------------------------------- | -------: | ---------: | -----: | -----: | -----: | ------: |
| 1. Preparar request DRAFT na .57                |    0.063 |      0.063 |  0.070 |  0.037 |  0.098 |    0.05 |
| 2. Serializar DRAFT                             |    0.010 |      0.008 |  0.016 |  0.005 |  0.067 |    0.01 |
| 3/9/10. Transporte e sockets do DRAFT, agregado |    1.269 |      1.273 |  1.327 |  1.133 |  1.393 |    0.98 |
| 4. Receber payload DRAFT no Xbox                |    0.998 |      1.002 |  1.041 |  0.825 |  1.125 |    0.77 |
| 5. Preparar DFlash no Xbox                      |    0.023 |      0.019 |  0.048 |  0.018 |  0.052 |    0.02 |
| 6. Inferencia DFlash DRAFT no Xbox              |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 8. Copiar/serializar resposta Xbox              |    0.000 |      0.000 |  0.001 |  0.000 |  0.001 |    0.00 |
| 9. Enviar resposta pelo socket Xbox             |    0.064 |      0.057 |  0.112 |  0.052 |  0.137 |    0.05 |
| 10. Copiar resposta para o seletor na .57       |    0.007 |      0.007 |  0.008 |  0.004 |  0.018 |    0.01 |
| 7. Seletor na .57                               |    5.488 |      5.383 |  5.986 |  4.684 |  7.550 |    4.24 |
| 11. Processar propostas                         |    0.007 |      0.007 |  0.009 |  0.004 |  0.024 |    0.01 |
| 18. Preparar batch de verificacao               |    0.002 |      0.002 |  0.002 |  0.001 |  0.005 |    0.00 |
| 12a. Submeter verificacao target                |    2.628 |      1.157 | 21.863 |  0.783 | 33.154 |    2.03 |
| 12b. Aguardar RTX 4070                          |   37.469 |     38.498 | 38.918 | 19.362 | 48.528 |   28.96 |
| 16a. Capturar features para SYNC                |    0.154 |      0.160 |  0.205 |  0.057 |  0.356 |    0.12 |
| 16b. Serializar SYNC                            |    0.108 |      0.107 |  0.161 |  0.048 |  0.268 |    0.08 |
| 16c. Transporte e sockets do SYNC               |    4.897 |      4.889 |  5.043 |  4.705 |  5.140 |    3.78 |
| 17a. Preparar SYNC no Xbox                      |    0.037 |      0.037 |  0.040 |  0.034 |  0.042 |    0.03 |
| 17b. Inferencia SYNC no Xbox                    |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 15a. TRIM RPC do KV Xbox                        |   41.645 |     41.631 | 42.847 | 40.093 | 43.387 |   32.18 |
| 15b. Rewind RPC do KV Xbox                      |   31.778 |     40.864 | 42.445 |  0.000 | 42.942 |   24.56 |
| 13. Sampling/aceitacao                          |    1.761 |      1.395 |  3.172 |  0.359 |  3.335 |    1.36 |
| 14. Commit dos tokens aceitos                   |    0.001 |      0.001 |  0.002 |  0.000 |  0.002 |    0.00 |
| 15c. Remocao KV target na .57                   |    0.139 |      0.145 |  0.155 |  0.063 |  0.221 |    0.11 |
| 14b. Processar e emitir tokens finais           |    0.009 |      0.008 |  0.010 |  0.005 |  0.083 |    0.01 |
| 18. Espera, barreira e overhead restante        |    0.844 |      0.795 |  1.239 |  0.602 |  1.840 |    0.65 |

## Etapas exclusivas: C: replay local na .57

Valores por 180 ciclos estáveis. Percentual calculado sobre a média do ciclo completo do respectivo modo; as linhas somam 100% com arredondamento.

| Etapa                                           | Média ms | Mediana ms | p95 ms | Mín ms | Máx ms | % ciclo |
| ----------------------------------------------- | -------: | ---------: | -----: | -----: | -----: | ------: |
| 1. Preparar request DRAFT na .57                |    0.026 |      0.025 |  0.029 |  0.016 |  0.039 |    0.05 |
| 2. Serializar DRAFT                             |    0.003 |      0.003 |  0.004 |  0.003 |  0.013 |    0.01 |
| 3/9/10. Transporte e sockets do DRAFT, agregado |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 4. Receber payload DRAFT no Xbox                |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 5. Preparar DFlash no Xbox                      |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 6. Inferencia DFlash DRAFT no Xbox              |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 8. Copiar/serializar resposta Xbox              |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 9. Enviar resposta pelo socket Xbox             |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 10. Copiar resposta para o seletor na .57       |    0.008 |      0.007 |  0.013 |  0.003 |  0.020 |    0.02 |
| 7. Seletor na .57                               |    4.854 |      4.824 |  5.146 |  4.504 |  6.194 |   10.19 |
| 11. Processar propostas                         |    0.003 |      0.003 |  0.004 |  0.002 |  0.009 |    0.01 |
| 18. Preparar batch de verificacao               |    0.001 |      0.001 |  0.002 |  0.000 |  0.009 |    0.00 |
| 12a. Submeter verificacao target                |    2.143 |      0.662 | 21.249 |  0.567 | 31.381 |    4.50 |
| 12b. Aguardar RTX 4070                          |   39.261 |     40.450 | 40.782 | 19.149 | 50.279 |   82.43 |
| 16a. Capturar features para SYNC                |    0.045 |      0.043 |  0.055 |  0.036 |  0.065 |    0.09 |
| 16b. Serializar SYNC                            |    0.026 |      0.022 |  0.045 |  0.018 |  0.151 |    0.06 |
| 16c. Transporte e sockets do SYNC               |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 17a. Preparar SYNC no Xbox                      |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 17b. Inferencia SYNC no Xbox                    |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 15a. TRIM RPC do KV Xbox                        |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 15b. Rewind RPC do KV Xbox                      |    0.000 |      0.000 |  0.000 |  0.000 |  0.000 |    0.00 |
| 13. Sampling/aceitacao                          |    0.724 |      0.625 |  1.287 |  0.245 |  1.970 |    1.52 |
| 14. Commit dos tokens aceitos                   |    0.000 |      0.000 |  0.001 |  0.000 |  0.001 |    0.00 |
| 15c. Remocao KV target na .57                   |    0.068 |      0.069 |  0.085 |  0.036 |  0.101 |    0.14 |
| 14b. Processar e emitir tokens finais           |    0.004 |      0.003 |  0.006 |  0.001 |  0.009 |    0.01 |
| 18. Espera, barreira e overhead restante        |    0.466 |      0.441 |  0.581 |  0.354 |  1.212 |    0.98 |

## Artefatos e estado final

- `profiling-2026-09-27/cycles.csv`: 540 linhas de dados (180 ciclos estáveis × 3 modos), tempos por etapa e ciclo.
- `profiling-2026-09-27/analysis.json`: estatísticas completas e diagnósticos.
- `profiling-2026-09-27/dflash-profile-{real,xbox,local}.jsonl`: 182 ciclos brutos por modo; logs do servidor e Xbox no mesmo diretório.
- Build `.57` instrumentado em `/home/hjotha/beellama.cpp/build-dflash-xbox-dl`, sem substituir `/home/hjotha/beellama.cpp/build/bin/llama-server` da produção. O teste em 59567 terminou; o roteador de produção voltou a `active` e `/health` respondeu `200 {"status":"ok"}`. Na última checagem o worker carregado era Bonsai 2, selecionado pelo roteador de dois modelos; Swift não estava carregado naquele instante.
- O app Xbox 1.0.0.9 permanece instalado, com modo mock desativado pelo controle RPC no início de C. Instrumentação e scripts ficaram nas branches de pesquisa; nenhum commit, merge ou push foi feito.
