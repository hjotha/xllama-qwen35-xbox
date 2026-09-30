# DFlash2 remoto no Xbox: mapeamento e benchmark de transporte

Data: 2026-09-27. Host target: GOKAYA `192.168.1.57`. Xbox Series X Developer Mode: `192.168.1.26`. Worktree: `C:\Users\hjotha\worktrees\xbox-dflash-rpc-probe`, branch `research/dflash-rpc-xbox`.

## Estado do objetivo

Foi implementado e medido um **probe binário de transporte TCP persistente** no Xbox. Ele não carrega DFlash2 nem executa compute GPU. Portanto ainda não há ms/bloco DFlash2, drafts/s, acceptance ou decode TPS do ISTA remoto. Nenhuma alegação de ganho final é válida nesta etapa.

O ISTA de produção permaneceu na `.57` (processo `llama-server` PID 2260245, porta local 33569, `/health` HTTP 200 após o teste) com pesos e KV locais. A produção usa `--spec-type draft-mtp`; DFlash2 não está ativo nela. Não executamos o `smoke_test_dflash2.py`, que pararia/reiniciaria a produção. O xllama no Xbox foi reaberto após o probe e respondeu `/health` HTTP 200; o processo do probe ficou suspenso.

## Pipeline DFlash2 encontrado no código da `.57`

BeeLlama `main` commit `57e515b8` estava limpo e foi examinado em leitura. Os GGUF DFlash2 Q2_K, Q3_K_M e Q4_K_M em `/home/hjotha/models/` têm respectivamente 705.430.880, 916.702.560 e 1.143.006.816 bytes. O Q3 usa cinco hidden states do target nas camadas `[6,20,34,48,62]`, hidden size 5120, bloco 8 e até sete drafts.

- `common/speculative.cpp:1919-2018`: `process()` extrai as cinco camadas F32 após verificação/prefill, concatena 25.600 floats por token (**102.400 bytes/token**) e injeta o batch no KV próprio do drafter. `src/llama-context.cpp:3130-3153,5725-5728` usa `ggml_backend_tensor_get_async` e sincroniza ao ler. O target já sincroniza antes do speculator em `server-context.cpp:9100-9109`.
- `common/speculative.cpp:2020-2105`: `draft()` faz um `llama_decode` de `[id_last,MASK...]` por bloco e percorre a lattice do seletor. O KV do drafter é distinto do KV principal. Rejeições são removidas dos dois contextos pelo fluxo de `server-context.cpp:9482-9509,9590` e `common/common.cpp:2388-2393`.
- `src/models/dflash.cpp:978-1007`: o noise block usa `token_embd.weight` do target para anchor e MASK. `dflash.cpp:1143-1200`: `output.weight` do target projeta o hidden final normalizado em logits. O seletor em `613-715` usa top-k, gate e as tabelas predecessor/successor. Os GGUF DFlash2 **não contêm** `token_embd.weight` ou `output.weight`; `dflash.cpp:978-985,1148-1156` os pega de `ctx_other`.

No GGUF ISTA ativo `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, embedding e output head quantizados ocupam **407.244.800 + 675.430.400 = 1.082.675.200 bytes**. Copiá-los ao Xbox preservaria o código atual, mas contrariaria o requisito de manter os pesos do target na `.57`.

### Split fiel a investigar

Um único RTT por bloco é algebricamente possível se a `.57` fizer o embedding lookup do anchor e mantiver output head **e seletor** locais. O Xbox receberia as features F32 e o embedding do anchor, injetaria o KV próprio, executaria o backbone DFlash2 e devolveria hidden final pós-normalização por posição, com gate se necessário. A `.57` calcularia logits, top-k, lattice e tokens. O MASK embedding pode ser enviado uma vez por sessão. As tabelas predecessor/successor Q3 do seletor somam **54.630.400 bytes** na `.57`.

Essa divisão exige uma nova entrada explícita de embeddings para o **noise block** em `dflash.cpp`. Usar apenas `llama_batch.embd` selecionaria o grafo de injeção KV (`dflash.cpp:898`) e alteraria o algoritmo. O host deve preservar IDs, transformações Hadamard/`embedding_scale`, posições e os pesos do seletor. Equivalência algébrica é plausível; paridade numérica entre backends e mesma sequência de drafts precisam de teste.

Para até três drafts, a estimativa F32 é **430.080 bytes de ida + 86.016 de volta** por ciclo. Para até sete, **839.680 + 172.032 bytes**. Headers não incluídos. O `process()` atual injeta também todo o prefill: 40.960 tokens implicariam **4.194.304.000 bytes** de features pela LAN na implementação fiel. Limitar ao SWA 2048 seria uma mudança semântica a provar separadamente.

O RPC genérico BeeLlama registra `--rpc` (`common/arg.cpp:3866-3871`), mas o xllama UWP atual não fornece daemon RPC/ggml GPU para o grafo DFlash2. Não foi possível fazer um benchmark DFlash2 pelo RPC genérico sem antes portar backend e loader; o protocolo dedicado de transporte foi usado para quantificar o custo de LAN. O xllama UWP tem GGUF CPU e ORT/DirectML para ONNX, não um backend GGUF DFlash2 D3D12.

## Probe implementado

`vs-debug/DFlashRpcProbe/` é um MSIX UWP de identidade separada, versão `1.0.3.0`, escuta em TCP `43125` quando está em primeiro plano. O protocolo de benchmark `DFR2` usa cabeçalho binário de 40 bytes, resposta de 72 bytes, `PING`, payload com SHA-256 e um modo de payload nos dois sentidos; não usa JSON, reconexão por iteração ou reload de pesos. O cliente está em `vs-debug/tools/dflash_transport_bench.py`. **DFR2 ainda não implementa `CREATE_SESSION`, `DRAFT`, `ACCEPT`, `ROLLBACK` ou execução do modelo.**

O cliente Linux da `.57` usa `TCP_NODELAY`. No UWP, `StreamSocketListener.Control.NoDelay` é `true` por padrão; [a documentação Microsoft](https://learn.microsoft.com/en-us/uwp/api/windows.networking.sockets.streamsocketlistenercontrol.nodelay) diz que o controle do listener deve ser definido antes de escutar. Uma tentativa de definir `socket.Control.NoDelay` depois de aceitar a conexão falhou no Xbox com `0x8000000E` e foi removida.

Build `Debug|x64` com MSBuild do VS2019 terminou com código 0 e produziu `DFlashRpcProbe_1.0.3.0_x64_Debug.msix`. O Device Portal aceitou o upload (`202`), a versão `1.0.3.0` apareceu na lista de pacotes, o launch retornou `200`, o processo e a porta foram vistos e o smoke validou hashes. O empacotamento inicial com VS2022 falhou no alvo NuGet de framework .NET; o VS2019 concluiu. O cliente passou em `python -m py_compile`.

## Rede direta `.57` → Xbox

`ping` direto: 20/20 respostas, mínimo/média/máximo **0,450 / 0,648 / 2,508 ms**. Cada linha TCP abaixo teve 100 warm-ups + **1.000 iterações aquecidas** numa conexão persistente. O probe faz SHA-256 do payload recebido e o cliente confere o hash a cada resposta. `full_cycle` inclui serialização Python, envio, leitura/hash no Xbox e resposta. O tempo `xbox_payload_read` inclui trânsito da LAN; não é uma cópia CPU isolada.

| Entrada → saída                           | p50 `full_cycle` |           p95 |           p99 |
| ----------------------------------------- | ---------------: | ------------: | ------------: |
| 0 → 72 B (PING)                           |         0,199 ms |      0,397 ms |      0,520 ms |
| 4 KiB → 72 B                              |         0,391 ms |      0,467 ms |      0,602 ms |
| 16 KiB → 72 B                             |         0,601 ms |      0,677 ms |      0,770 ms |
| 64 KiB → 72 B                             |         1,005 ms |      1,196 ms |      1,304 ms |
| 128 KiB → 72 B                            |         1,685 ms |      1,850 ms |      1,917 ms |
| 256 KiB → 72 B                            |         2,967 ms |      3,115 ms |      3,160 ms |
| 512 KiB → 72 B                            |         5,418 ms |      5,571 ms |      5,824 ms |
| 1 MiB → 72 B                              |        10,363 ms |     10,529 ms |     32,322 ms |
| 102.400 B → 72 B (1 token de features)    |         1,433 ms |      1,587 ms |      1,670 ms |
| 819.200 B → 72 B (8 tokens de features)   |         8,213 ms |      8,400 ms |     20,259 ms |
| **430.080 → 86.016 B (split, 3 drafts)**  |     **5,499 ms** |  **5,639 ms** |  **8,338 ms** |
| **839.680 → 172.032 B (split, 7 drafts)** |    **10,207 ms** | **10,377 ms** | **29,883 ms** |

O último par teve p50 de **0,064 ms** para serialização no host, **6,902 ms** no intervalo de leitura do payload no Xbox (inclui rede), e **0,629 ms** de SHA no Xbox. O p99 de 29,883 ms mostra cauda de latência que precisa ser considerada por ciclo. O SHA é custo do probe, não parte obrigatória do protocolo final; upload/compute/readback GPU e head/seletor na `.57` ainda acrescentariam tempo.

JSONs brutos: `research/dflash-transport-57-xbox-2026-09-27.json`, `research/dflash-transport-hidden-57-xbox-2026-09-27.json`, `research/dflash-transport-duplex-57-xbox-2026-09-27.json`. O último tem SHA-256 local `592AA96CC330E1C100B87E5EE12F8CC3FD6B70B1403D44322F6C0D97498C0335`.

## Respostas ainda pendentes e próximo trabalho

1. **DFlash2 roda no Xbox?** Ainda não. É necessário portar o grafo DFlash2 e seu GGUF para um backend Xbox (D3D12/DirectML ou referência CPU), carregar pesos Q2/Q3/Q4 uma vez e verificar paridade contra a `.57`. A documentação BeeLlama atual qualifica CUDA draft-KVarN; Xbox não está qualificado.
2. **ms/bloco e drafts/s?** Não medidos. O benchmark de 1.000 iterações GPU deve separar preparação CPU, upload, GPU, sync, readback e produção do bloco. O probe de rede não responde essas perguntas.
3. **Sessão e rejeição parcial:** implementar `CREATE_SESSION`, atualização de posição/aceitação + injeção + `DRAFT` em uma chamada, rollback exato do KV próprio, `RESET` e `DESTROY`. Testar A B C aceitos de A B C D E F, comparando o próximo bloco ao drafter local.
4. **TPS final:** não medido. Antes do A/B, obter baselines com o mesmo GGUF ISTA, prompt, sampling, clocks e contexto: target só, MTP local atual, DFlash2 local (se couber), DFlash2 em outra GPU se houver, e Xbox remoto. A RTX 4070 tinha apenas **18 MiB livres** na checagem, então não foi iniciado outro DFlash CUDA que afetaria a produção. Resultados históricos de DFlash2 usam outro target e não servem como baseline atual.
5. **Gargalo:** na parte medida, transferência do payload domina o ciclo grande; o gargalo do sistema completo permanece indeterminado. Primeiro avaliar redução de features somente após paridade. Mover a projeção para a `.57` exige seus pesos/compute no host e medição de custo, pois a GPU está quase cheia. Sobreposição entre verificação e próximo bloco depende da aceitação e das features novas; não presumir independência.

Esta worktree fica sem commit/merge porque a arquitetura fim a fim e a validação DFlash2 no Xbox ainda estão incompletas. O checkout principal e o serviço ISTA não foram alterados.
