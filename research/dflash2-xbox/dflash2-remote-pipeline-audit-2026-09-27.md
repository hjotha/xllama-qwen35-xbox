# DFlash2 remoto: pipeline atual e limites da primeira integração

Data: 27/09/2026. Leitura somente do fork `/home/hjotha/beellama.cpp` na
`.57` (branch `main`, commit `57e515b8`) e do código UWP disponível no
HPBEATS. Este documento registra o estado anterior a qualquer edição do
algoritmo DFlash2. Ainda não há inferência DFlash2 no Xbox nem benchmark
remoto de tokens/s.

## Estado observado

- O target Bonsai 2 PTQ1 sem MTP ocupa 5.946.648.928 bytes. O GGUF DFlash2
  Q4_K_M ocupa 1.143.007.104 bytes; Q6_K, 1.590.413.184 bytes; Q8_0,
  2.056.415.104 bytes. O Q4 declara arquitetura `dflash`, cinco blocos,
  hidden size 5.120, `block_size=8`, selector `top_k=16` e `rank=256`.
- O DFlash2 lê as entradas das camadas `[6,20,34,48,62]` do target. Cada
  posição fornece cinco vetores F32 de 5.120 elementos: **102.400 bytes por
  token** antes do framing. Sessenta e quatro posições exigiriam 6,25 MiB.
- O fork possui três `llama-server` ativos com Qwen/MTP; **nenhum DFlash2
  estava ativo** durante a leitura. Um log de 19/09 registrou 203/257 drafts
  aceitos e 57,70 tokens/s em um ensaio local PTQ1+DFlash2 Q8_0, `n_max=5`.
  Esse dado é histórico e não é baseline atual comparável ao Xbox.

## Caminho de um ciclo no fork

1. `common/speculative.cpp:1971-2004` lê `llama_get_embeddings_layer_inp`
   para as cinco camadas. Essa leitura sincroniza a saída do backend com CPU.
   O código concatena as features por posição e as injeta com
   `llama_decode(ctx_dft, batch_inject)` no cache próprio do drafter.
2. `common/speculative.cpp:2015-2135` forma `[id_last, mask, ...]`, decodifica
   um bloco com o draft, lê a lattice do selector e escolhe os candidatos.
   `src/models/dflash.cpp:613-715` produz 272 floats úteis por linha da
   lattice (`16` IDs e `16×16` scores), embora a saída atual seja preenchida
   até 5.120 floats por linha.
3. `tools/server/server-context.cpp:7999` captura checkpoints de cache.
   `server-context.cpp:9434` faz a verificação no target, computa
   `n_rollback`, restaura estado e reproduz o trecho aceito quando necessário.
   Uma implementação remota precisa reproduzir a mesma transição após
   aceitação parcial; `DRAFT` sozinho não basta.

## Dependência que impede mover o draft literalmente

O GGUF DFlash2 não inclui `token_embd.weight` nem `output.weight` próprios.
`src/models/dflash.cpp:978` busca os embeddings do token no `ctx_other`
(target), com transformações Hadamard quando aplicáveis. O selector também
usa a projeção de saída do target para formar logits e candidatos
(`src/models/dflash.cpp:613-715`). A criação atual define
`cparams.ctx_other = ctx_tgt` em `common/speculative.cpp:3805`.

Assim, iniciar o GGUF DFlash2 **isolado no Xbox** não reproduz o algoritmo
atual. Copiar essas matrizes do ISTA/Qwen para o console contrariaria o
requisito de manter os pesos principais na `.57`. Uma divisão a testar é
calcular na `.57` os embeddings de entrada necessários, executar o núcleo do
drafter no Xbox e retornar suas saídas para a projeção/seleção dependente do
target na `.57`. Esse desenho ainda precisa de um corte exato no grafo e de
paridade numérica antes de ser chamado equivalente ao DFlash2 local.

## RPC e rede

- O fork aceita `--rpc` e `--spec-draft-device`. Isso permite investigar um
  dispositivo GGML remoto só para o draft, mantendo o dispositivo do target
  local. O RPC GGML genérico expõe dispositivos e operações de grafo; não
  implementa por si só `CREATE_SESSION`, `DRAFT` e commit/rollback do ciclo
  especulativo. Também é preciso verificar se o grafo transfere matrizes do
  target ao dispositivo remoto. O [README oficial do RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
  descreve o split padrão de pesos e KV quando não há seleção explícita.
- O xllama UWP local tem servidor LAN, mas seu projeto usa ORT por padrão e
  compila o llama.cpp apenas no caminho CPU. Não há backend GGML Direct3D
  completo nesse projeto. O Xbox UWP suporta listener TCP com
  `privateNetworkClientServer`, conforme a [documentação de sockets da
  Microsoft](https://learn.microsoft.com/en-us/windows/uwp/networking/sockets).
- Na `.57`, `ping -c 3` para `192.168.1.26` retornou 3/3 respostas:
  0,591/1,176/2,287 ms mínimo/média/máximo. `GET /health` em
  `192.168.1.26:11434` expirou em 5 s; no mesmo período o Device Portal
  mostrava `xllama.exe` sem execução, após a instalação de um probe DFlash
  separado. ICMP não mede o custo de transferir 100 KiB nem um ciclo RPC.

## Gates para implementação e medição

1. Congelar um baseline local atual com o mesmo target, draft, quantizações,
   seed, prompt e thresholds. Os processos ativos usam MTP; o log de 19/09
   não substitui esse baseline.
2. Provar paridade do corte de grafo com features idênticas, incluindo
   embeddings/Hadamard, selector e aceitação parcial. Registrar divergências
   de FP16/FP32 ou de quantização.
3. Medir o transporte persistente `.57`→Xbox para 4 KiB até 1 MiB. O probe
   `DFlashRpcProbe` em desenvolvimento paralelo mede **somente transporte**;
   nenhum resultado dele pode ser rotulado como tempo de DFlash2.
4. Portar um runtime que carregue os pesos do drafter no Xbox e medir, no
   console real, execução por bloco e contabilidade Game/DXGI. Só então
   integrar o loop especulativo e comparar TPS com os baselines.

**Estado de decisão:** DFlash2 remoto no Xbox permanece **não medido**. Não
há base para classificá-lo como viável, marginal ou inviável neste estágio.
