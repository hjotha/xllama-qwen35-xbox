# Swift 1.5 + DFlash2 Q3_K_M: parametros medidos em 2026-09-27

## Configuracao fixa

- Target: `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` no host `.57`, KV target `kvarn4`, RTX 4070, contexto 8192, `parallel=1`.
- Drafter: `/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf` no Xbox, pacote `HJotha.XboxDFlashRpc_1.0.0.8_x64__3hvcqmcd1t14m`, KV Q4_0, CPU. Output head, embeddings e seletor ficaram na `.57`.
- Prompt: `Write a short paragraph explaining why the sky appears blue.` (11 tokens no target). Seed 1234, temperatura 1.0, top-p 0.95, top-k 20, `cache_prompt=false`. Mesmo JSON para a varredura, `n_predict=64`.
- Cada linha da varredura e uma requisicao, depois da inicializacao do servidor, sem replicatas. TPS e o `predicted_per_second` reportado pelo `llama-server`. Taxas de aceitacao referem-se a tokens propostos pelo RPC em ciclos registrados no JSONL; o contador final do servidor pode diferir no ultimo ciclo.

## Varredura curta no Swift

| n_max | p_min | TPS final | ciclos | propostos | aceitos | aceitos/propostos | ciclos vazios | DRAFT RPC medio (ms) | ciclo medio (ms) | falhas RPC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.0 | 9.78 | 29 | 58 | 34 | 58.6% | 0 | 99.2 | 221.8 | 0 |
| 4 | 0.0 | **10.65** | 21 | 84 | 42 | 50.0% | 0 | 137.2 | 281.4 | 0 |
| 6 | 0.0 | 6.81 | 25 | 150 | 38 | 25.3% | 0 | 196.9 | 369.5 | 0 |
| 7 | 0.0 | 7.17 | 22 | 154 | 40 | 26.0% | 0 | 210.2 | 394.5 | 0 |
| 4 | 0.3 | 10.22 | 22 | 73 | 41 | 56.2% | 1 | 139.6 | 275.5 | 0 |
| 4 | 0.7 | 9.86 | 24 | 59 | 39 | 66.1% | 4 | 137.6 | 256.6 | 0 |
| 7 | 0.3 | 8.78 | 20 | 76 | 43 | 56.6% | 2 | 209.7 | 351.4 | 0 |
| 7 | 0.7 | 8.33 | 22 | 63 | 41 | 65.1% | 4 | 207.2 | 331.6 | 0 |

`p_min=0.70` e uma configuracao do MTP em producao, nao um valor recomendado pelo card do DFlash2. Aqui ele aumentou a fracao de aceitos entre os tokens selecionados, mas reduziu o TPS: o Xbox ainda computa o bloco inteiro antes do seletor cortar as propostas. O melhor ponto **desta varredura curta** foi `n_max=4, p_min=0`.

## Teste longo e baseline

Mesmo prompt, seed e sampling, agora `n_predict=512` e `ignore_eos=true` para garantir 512 tokens:

| Configuracao | Tokens | TPS final | Propostos | Aceitos | Ciclos | Falhas RPC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISTA Swift sozinho | 512 | **34.99** | - | - | - | - |
| ISTA Swift + DFlash2 Q3_K_M Xbox CPU, `n_max=4`, `p_min=0` | 512 | **9.53** | 728 | 329 | 182 | 0 |

O remoto ficou 3.67 vezes mais lento que o target sozinho nesta carga. No teste longo remoto, a aceitacao foi 45.2%; o DRAFT RPC medio foi 144.9 ms e o ciclo medio 294.4 ms. Um teste anterior de 512 sem `ignore_eos` encerrou por EOS apos 265 tokens (9.51 TPS) e nao atende ao workload de 512.

O baseline DFlash2 local em CPU foi iniciado com os mesmos parametros, mas excedeu o timeout de 180 s antes de entregar 512 tokens. O log mostrava 235 tokens e taxa acumulada de aproximadamente 1.91 tok/s naquele instante; isso **nao e um benchmark completo** e nao deve entrar na tabela comparativa de 512.

## Integridade, limites e estado final

- DFlash2 Q3_K_M publicado para Qwen3.8-27B base funcionou com o Swift 1.5: 512 tokens, 329 drafts aceitos e zero falhas RPC. Portanto **nao e necessario baixar outro GGUF para fazer esta arquitetura funcionar**. O impacto de um drafter treinado especificamente para Swift nao foi medido.
- O GGUF declara bloco 8, com ancora mais ate sete propostas. `n_max=8` nao e valido neste caminho; nao foi forcado.
- O pacote Xbox 1.0.0.8 remove KV a partir da ancora antes de cada DRAFT; o host remove a cauda antes do SYNC seguinte e trata rewind de posicao. Essas correcoes permitiram completar o teste longo sem fallback.
- A implementacao Xbox executa DFlash2 em CPU. Nao existe numero de `xbox_gpu_us` real. A conclusao sobre esta implementacao CPU e **NAO VIAVEL para ganho de TPS** neste target/hardware; a arquitetura com GPU Xbox ainda nao foi medida.
- Faltam paridade token a token com DFlash2 local, workloads 8k/24k/40k, medicao de RPC generico e breakdown completo de rede/GPU/commit. A varredura acima e de um unico prompt curto e nao define parametro otimo universal.
- Ao encerrar, `llama-server-root.service` estava `active` e `/health` em `127.0.0.1:8090` respondeu HTTP 200 `{"status":"ok"}`; porta de teste 59567 fechada; watchdog inativo. Nenhum commit, merge ou push foi feito.

## Fontes do parametro inicial

- Card oficial do GGUF DFlash2: <https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF> (exemplo `n_max=7`; temperatura 1, top-p 0.95, top-k 20 na avaliacao do modelo base).
- Documentacao do llama.cpp: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md> (bloco DFlash e limite `n_max`).
