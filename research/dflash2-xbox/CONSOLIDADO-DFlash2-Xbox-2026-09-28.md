# DFlash2 no Xbox Series X — Consolidação Completa (atualizada 29/09/2026)

Documento único consolidando toda a pesquisa, implementação e resultados medidos do DFlash2 rodando no Xbox. Gerado a partir dos relatórios e evidências em `research/dflash2-xbox/` da worktree ativa. Atualizado com: sessões AGY de 28/09 noite (otimização FFN), deep review e 3 experimentos de 29/09 (pacotes .78–.82), revisão Codex com recuperação de paridade (.83) e pré-organização Q2_K com A/B controlado (.84/.85).

- **Worktree:** `C:\Users\hjotha\worktrees\xllama-dflash-xbox`
- **Branch:** `research/dflash-xbox-rpc` (base de implementação `8b3faa7`)
- **Repositório privado:** `git@github.com:hjotha/xllama-dflash-xbox.git` (remote `private`)
- **Console:** Xbox Series X, `192.168.1.26`
- **Host do target:** GOKAYA, `192.168.1.57`

---

## 1. Objetivo

Rodar o **DFlash2** (drafter de decodificação especulativa de 5 camadas, ~1,92B parâmetros, derivado do Qwen3.8-27B) na **APU AMD RDNA2 do Xbox Series X via D3D12/HLSL**, enquanto o modelo alvo (Swift-1.5-Qwen3.8-27B, 27B) continua na RTX 4070 da `.57`. Objetivo final: superar os **34,99 tok/s** do Swift rodando sozinho.

O resultado hoje: **implementação funcional com ganho sobre CPU (1,99x)** e **empate prático com o target isolado** (35,518 vs 35,320 tok/s em campanha controlada; 35,421 tok/s no recorde da varredura AGY), mas a **paridade numérica integral e o ganho ponta a ponta sustentado ainda não estão fechados** (F6 aberto).

## 2. Arquitetura e topologia

```
.57 (GOKAYA, RTX 4070 CUDA0)                     Xbox Series X (D3D12, APU RDNA2)
  Swift-1.5-Qwen3.8-27B (target)                 DFlash2 27B (drafter, 5 blocos)
  - embeddings, output head, KV principal        - pesos GGUF residentes
  - selector head (Q8_0), verificação, sampling  - KV do drafter (Q4_0, 49.152 posições)
        |                                          - engines GPU D3D12 (SYNC + DRAFT)
        |  DFL2 v2, TCP :50053, TCP_NODELAY
        +----------------------------------------->  SYNC (atualiza KV, 25.600 floats/posição)
        <------------------------------------------+  DRAFT (devolve hidden states F32 [5, 5120])
        |  verificação em lote na 4070 (~55 ms)
```

- **Protocolo:** RPC binário `DFL2` v2 (HELLO, RESET, SYNC, DRAFT, TRIM, MOCK, SYNC_TRIM; opcode 18 = SYNC+DRAFT fundido; opcodes 8–12 = diagnósticos GPU isolados).
- **Ciclo sequencial real:** ~109 ms com n7 → ~2,81 tokens/ciclo (n7) e ~127,9 ms em fused n7 pós-verificação; com os pacotes .78+ o compute Xbox caiu de ~44,7 para ~25,2 ms, mas o ciclo real é dominado pela verificação do target (~43–58 ms). Componentes: verificação target ~43–58 ms, Xbox SYNC+DRAFT ~25–52 ms, RPC ~5–11 ms, selector ~4,9–6,7 ms.
- O app RPC **não inclui** `token_embd.weight`/`output.weight` do target; o corte respeita a regra de manter pesos principais na `.57`.

## 3. Inventário de infraestrutura

| Item                     | Valor                                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Xbox IP / Device Portal  | `192.168.1.26` / `https://192.168.1.26:11443` (HTTP Basic)                                                                                  |
| Porta RPC dedicada       | `50053` TCP (genérica GGML RPC: `50052`)                                                                                                    |
| Pacote UWP               | `HJotha.XboxDFlashRpc` (versões testadas 1.0.0.2 → 1.0.0.85)                                                                                |
| AUMID                    | `HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App`                                                                                                    |
| Classificação            | Game por padrão (`defaultuwpcontenttypetogame=True`); `AppMemoryUsageLimit` = 5.368.709.120 bytes (5 GiB)                                   |
| Toolchain                | MSVC v143, Windows SDK 10.0.26100.0, DXC em `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe`                           |
| Certificado              | SHA1 `CD6D10EC8332955BC5C1074BC20A297D0625F1F4` (store `My`); assinatura .85: `CN=Hjotha Xbox Probe`                                        |
| Cliente .57              | `/home/hjotha/beellama.cpp/build-dflash-xbox-dl/bin/llama-server` (build 12304, commit `12231b55e`; lib `b46f159f...`)                      |
| Serviço de produção .57  | `llama-server-root.service` (127.0.0.1:8090) — **sempre restaurado após testes**                                                            |
| GGUF Q3_K_M              | 916.702.560 bytes, SHA-256 `72809a53e89cb959a3149c9b3752ad181b640164b51a3b9f445712520beec5dc`; inventário: 32 F32, 34 Q3_K, 13 Q4_K, 2 Q5_K |
| Captura Q2 de referência | SHA `5ff0aeb6…` (replay 369/369 bit a bit); Q3: `b55ef2a1…`                                                                                 |

## 4. Estado por etapa (plano F0–F7)

| Etapa                 | Estado | Resumo                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F0** Referência     | ✅     | Captura real de 369 frames (Q3 SHA `b55ef2a1...`, Q2 SHA `5ff0aeb6...`), replay CPU determinístico **bit a bit**, fixtures correlacionadas (182 ciclos, 728 propostas, 329 aceitos), inventário de 81 tensores GGUF                                                                                                                                                                                  |
| **F1** Dispositivo    | ✅     | D3D12 hardware (adapter `SraKmd_arden`), Shader Model 6, WaveOps (64 lanes / Wave32 real), timestamps GPU; shaders compilados offline com DXC                                                                                                                                                                                                                                                        |
| **F2** Kernels        | ✅     | **Decisão HLSL**: Q3_K/Q4_K/Q5_K reais validados e 2,1–5,6x mais rápidos que CPU ggml com readback. DirectML Q4_K FP16 correto porém mais lento e 3,56x mais memória; Q3_K DirectML falhou (0x887A0005 / crash)                                                                                                                                                                                      |
| **F3** KV + SYNC      | ✅     | Cache físico Q4_0 GPU = CPU em **70.399/70.400 bytes**; gate Q8_K de escala fechado com F32 rounding explícito (packer)                                                                                                                                                                                                                                                                              |
| **F4** DRAFT          | ✅     | Root cause F4 (flip de bin Q8_K por ~1e-6 F32, amplificado ~667x pelo FFN). **Recuperado no pacote 1.0.0.83: Probe 17 = 20/20 casos, status 0, erro absoluto máximo 0, cosseno 1,000000**; replay `--capture-draft 5` com `captureCpuExact: true, maxAbs: 0`; probes 8 (GEMV Q4_K) e 16 (Atenção Q4_0) status 0. Também válido no `.85` com captura Q2                                               |
| **F5** RPC + fallback | ✅     | Executor GPU integrado: `LocalState\dflash-backend.txt` (`cpu` padrão / `d3d12`); fallback CPU antes da sequência, invalidação + erro após falha de estado; smokes CPU/GPU passam. `dflash-barrier-mode.txt` (`strict`/`minimal`, default minimal). Fusão RPC de rede feita (opcode 18), mas SYNC e DRAFT continuam em submissões/fences separados (`dedicated.cpp:558/577`, engines `:793`/`:1181`) |
| **F6** TPS real       | 🔶     | Campanhas: d3d12 21,26 tok/s vs CPU 12,92 tok/s (1,65x) e otimizada 25,75 (1,99x). Varredura AGY: **35,421 tok/s (n4/p0,25) — recorde**, gap 0,24% do target. Campanha controlada Codex (temp 0): **35,518 vs 35,320 (+0,56%)**. **Ainda aberto:** divergência token/logit entre target e pipeline não resolvida; baseline AGY usava temp 1,0 vs 0,0 dos candidatos                                  |
| **F7** Otimização     | ✅     | Kernels Load4/LDS → 20,52 GB/s, DRAFT 44,67 ms (pacote .45). **Série .78–.85: mad nativo + barreiras minimal → DRAFT 25,16–25,22 ms, recorde; pré-organização Q2_K (.85) → +8,4–8,7% TPS ponta a ponta vs .83**. Ver §5.8–5.11 e §6                                                                                                                                                                  |

**Conclusão de viabilidade (F7):** com aceitação de ~3,1–3,6 tokens/ciclo e compute Xbox a 25,2 ms (18 ms GEMV stream), o sistema chegou ao **empate prático com o target isolado**; superar de forma sustentada exige fechar F6 (divergência de trajetória) e/ou subir aceitação (alvo ≥ 4 tokens/ciclo).

## 5. Resultados de performance medidos

### 5.1 Campanha F6 (app 1.0.0.42, workload fixo: prompt "sky blue", 512 tokens, seed 1234, temp 1.0, top_p 0.95, top_k 20, n_max=4)

| Round | d3d12 tok/s | cpu tok/s | conteúdo (SHA-256)  |
| ----- | ----------- | --------- | ------------------- |
| 1     | 21.318      | 12.916    | `31e48d46...e218dc` |
| 2     | 21.195      | 12.922    | `31e48d46...e218dc` |
| 3     | 21.273      | 12.921    | `31e48d46...e218dc` |

### 5.2 Campanha otimizada (app 1.0.0.45, kernels Load4)

| Round | TPS    | Wall (s) | SHA-256             |
| ----- | ------ | -------- | ------------------- |
| 1     | 25.746 | 20.22    | `31e48d46...e218dc` |
| 2     | 25.774 | 20.18    | `31e48d46...e218dc` |
| 3     | 25.739 | 20.24    | `31e48d46...e218dc` |

Média **25.753 tok/s** (+21,1% sobre baseline GPU, **1,99x sobre CPU**).

### 5.3 Comparativo de modelos no Xbox (app 1.0.0.47)

| Quantização   | GGUF    | DRAFT GPU Xbox | CPU fallback  | TPS produção | Observação                                      |
| ------------- | ------- | -------------- | ------------- | ------------ | ----------------------------------------------- |
| **Q3_K_M**    | 916 MB  | 44,67 ms       | 162 ms (3,6x) | **25,75**    | baseline otimizado                              |
| **Q4_K_M**    | 1,14 GB | 47,68 ms       | 263 ms (5,5x) | **23,70**    | sweet spot qualidade/velocidade                 |
| **Q8_0**      | 2,05 GB | 185,74 ms      | 240 ms (1,3x) | **10,66**    | probe 17 com 2 cópias pode dar OOM (0x8007000E) |
| Swift isolado | —       | —              | —             | **34,99**    | meta a superar                                  |

### 5.4 Varredura n_max (app 1.0.0.42, d3d12)

| n_max | TPS                      | Nota                                              |
| ----- | ------------------------ | ------------------------------------------------- |
| 2     | 20.818                   |                                                   |
| 4     | 21.318 / 21.195 / 21.273 | ótimo operacional                                 |
| 6     | 18.446                   | mais posições sem aceitação compensatória         |
| 7     | inviável                 | OOM na verificação da 4070 (buffer CUDA +981 MiB) |

### 5.5 Experimento Q2 n7 (app 1.0.0.75, pacote final das correções numéricas)

| Configuração                                    | tok/s                                                           |
| ----------------------------------------------- | --------------------------------------------------------------- |
| Target sozinho, batch 64                        | 35.488                                                          |
| Q2 n7, p_min 0.30, SYNC/DRAFT separados         | 27.726                                                          |
| Q2 n7, p_min 0.30, RPC fundido (opcode 18)      | 27.975                                                          |
| Q2 n7, p_min 0.30, fundido + pipeline corrigido | 28.468 (ganho 1,76%, 4 drafts reusados/run)                     |
| Q2 n7, p_min 0.65                               | 26.822 (33 ciclos vazios: aceitação 65,8% mas throughput menor) |
| Q2 n4, p_min 0.30                               | 29.731                                                          |

Fusão SYNC+DRAFT economizou **1,234 ms/ciclo** (49,2+14,7 separados vs 62,7 fundido). Pipeline corrigido: pre-draft só no top-1 greedy com rank-2 como token alternativo; gate de score ≥ 0,25; off por padrão (`DFLASH_XBOX_PIPELINE=1`).

### 5.6 Anatomia do ciclo (fused n7, pacote .75, média)

| Componente               | ms/ciclo      |
| ------------------------ | ------------- |
| Verificação target       | 58,212        |
| Xbox SYNC + DRAFT        | 51,518        |
| RPC fora do compute Xbox | 11,164        |
| Selector host            | 6,733         |
| Preparação/proposta host | ~0,23         |
| **Total**                | **~127,9 ms** |

Aceitação real: 39,1% (n7), ~3,6 tokens/ciclo; seriam necessários ~4,5 tokens/ciclo para empatar com target sozinho.

### 5.7 Sessões AGY noite de 28/09 — otimização FFN (app 1.0.0.76/1.0.0.77)

Clarificação do histórico AGY vs campanha F6: o resultado histórico de **30.109 tok/s** do AGY usou Q3 no Xbox, **seletor host Q8_0, `n_max=4`, `p_min=0.3`, temperatura 0**, 512 tokens e 156 ciclos; o 25.753 tok/s usava temperatura 1. São workloads diferentes, não regressão. (Evidência preservada: `agy-replay-gpu-66-20260928/agy-baseline.json`.)

Comparativo de pacotes (mesmos parâmetros de geração, mesmo binário host, SHA de saída casando entre sequential/pipeline):

| Medição                      | AGY histórico                               | Pkg 1.0.0.66 | Pkg 1.0.0.70 | Pkg 1.0.0.75 |
| ---------------------------- | ------------------------------------------- | ------------ | ------------ | ------------ |
| Decode sequencial, tok/s     | 30.109                                      | 18.75881     | 25.264465    | 25.645881    |
| Decode pipeline, tok/s       | —                                           | 18.68848     | 25.227748    | 25.524224    |
| Target sozinho, tok/s        | ~34.99                                      | 35.41694     | 35.513456    | 35.429874    |
| SYNC+DRAFT compute/ciclo, ms | 44.668 DRAFT + 3.11 SYNC (timers separados) | 110.89468    | 69.414923    | 69.177506    |
| Ciclo ativo completo, ms     | 108.8 reportado                             | 171.28746    | 127.161115   | 126.737090   |

**Otimizações FFN aplicadas no pacote 1.0.0.77** (projeções `ffn_gate`/`ffn_up` Q2_K [5120, 17408] e `ffn_down` Q3_K [17408, 5120] concentram ~72% do tempo de draft — 27,2 ms de 38,2 ms):

1. **Q2_K (`q2k_q8k_matmul.hlsl`):** load coalescing — 21 leituras escalares 32-bit de pesos trocadas por 5 × `Load4()` de 128 bits + 1 escalar; hoisting das escalas `s_q2_d`/`s_q2_dmin` para LDS pela thread 0.
2. **Q3_K (`q3k_q8k_matmul.hlsl`):** índice de escala simplificado `half*8u + quarter*2u + (lane32>>4u)` → `element >> 4u` (provado bit a bit idêntico); hoisting de `s_q3_d` em LDS.
3. **Engine (`dflash-gpu-draft-engine.cpp`):** eliminação de barreira UAV desnecessária.

**Resultado medido: sem ganho — leve regressão (+0,60 ms).** Diagnóstico do gargalo (sessão AGY 20h50):

1. **Teto físico de banda:** o ciclo de draft lê **483,9 MB de pesos** nas 5 camadas (Gate+Up 292,4 MB + Down 191,5 MB); com ~20,52 GB/s efetivos na sandbox UWP, o limite físico é **483,9 MB / 20,52 GB/s = 23,58 ms** — irredutível para dispatches sequenciais.
2. **Penalidade FP64 no RDNA2:** operações de dupla precisão rodam a **1/32 da taxa FP32**; cada acumulação de bloco usa `ggml_fma_f32` (recorre a `double` para reproduzir fielmente a ordem/mantissa de arredondamento AVX2 do CPU).
3. **Ativações cabem em cache:** vetor de entrada para 8 tokens ≈ 46 KB, dentro de L1/L2 — o gargalo é 100% streaming de pesos VRAM + latência de acumulação FP64.

### 5.8 Deep review 29/09 — perfil completo por kernel (pacote 1.0.0.80, batch 8, 112 checkpoints)

Soma dos intervalos GPU medidos: **26,61 ms** (DRAFT puro do executor: 25,18–25,32 ms).

| Operação / Kernel                       | Dispatches |     Tempo GPU | % do DRAFT |   Bytes VRAM |     Throughput |
| --------------------------------------- | ---------: | ------------: | ---------: | -----------: | -------------: |
| FFN Gate + Up (`q2k_q8k_matmul.hlsl`)   |         10 | **11,356 ms** | **42,67%** |    232,24 MB |     20,45 GB/s |
| FFN Down (`q3k_q8k_matmul.hlsl`)        |          5 |  **7,618 ms** | **28,63%** |    152,06 MB |     19,96 GB/s |
| Atenção Q,K,V (`q2k_q8k_matmul.hlsl`)   |         15 |      3,330 ms |     12,51% |     51,61 MB |     15,50 GB/s |
| Atenção Out (`q2k_q8k_matmul.hlsl`)     |          5 |      1,829 ms |      6,87% |     34,41 MB |     18,81 GB/s |
| Conv projections                        |         10 |      1,072 ms |      4,03% |     21,50 MB |     20,06 GB/s |
| SWA Attention (`dflash_attention.hlsl`) |          5 |      0,854 ms |      3,21% |     ~12,5 MB |     14,64 GB/s |
| RMSNorm                                 |         11 |      0,300 ms |      1,12% |      ~1,2 MB |  compute-bound |
| SwiGLU                                  |          5 |      0,125 ms |      0,47% |      2,21 MB |             L2 |
| Hadamard (128/64)                       |         20 |      0,046 ms |      0,17% |      ~1,0 MB |  registers/LDS |
| Dynamic Conv + Scale-Add + Upload       |         25 |      0,083 ms |      0,32% |     < 1,0 MB |       in-cache |
| **TOTAL**                               |    **111** | **26,613 ms** |   **100%** | **492,0 MB** | **~19,8 GB/s** |

**Conclusão estrutural: GEMV = 24,88 ms de 26,61 ms (93,5%) do tempo GPU.** Qualquer ganho que não mexa no streaming de pesos do GEMV é marginal.

**Causa raiz física do GEMV (latency-bound, não bandwidth-bound):** cada bloco processa apenas **84 bytes de pesos Q2_K** com **3 barreiras LDS por bloco** (60 barreiras consecutivas em nb=20), sem prefetch assíncrono — as threads ficam ociosas esperando a latência de round-trip da GDDR6 (~300–400 ciclos) a cada bloco.

### 5.9 Três experimentos de otimização (29/09, pacotes .79–.82)

**Experimento 1 — Double-buffering / Ping-Pong LDS (pacote 1.0.0.81): REJEITADO.**
Prefetch do bloco k+1 em LDS enquanto processa o bloco k. Paridade bit a bit preservada (`differing_outputs: 0`), porém **regressão severa: DRAFT 25,22 → 36,97 ms (+46,6%)**; GEMV block 21,3 → 32,8 ms (+54%). Causa raiz RDNA2: contenção de portas de banco LDS (1 leitura + 1 escrita por ciclo) e perda dos deslocamentos imediatos `ds_read_b32 offset:imm` (viram endereçamento VGPR dinâmico, mais pressão de registradores). Reversão imediata.

**Experimento 2 — Fusão em lote de `ResourceBarrier(count, items)` (pacote 1.0.0.82): APROVADO.**
Sobrecargas de transição 2/3 recursos; agrupadas todas as transições do pipeline (init, matmul, norm, rope, hadamard, conv, scale_add, swiglu, pack_kv, attention). **Recorde histórico: DRAFT 25,16 ms** (mediana 25,20 vs 25,22 no .80), Fused compute 27,50 ms (−0,17), Cycle RTT fused 31,36 ms. Paridade bit a bit preservada. Mantido ativo com `dflash-barrier-mode: minimal`.

**Experimento 3 — Calibração do seletor host `draft-p-min` (pacote .82, n_max 7 e 4, temp 0):**

| Configuração        | n_max |    p_min |      tok/s |  Wall (s) | Propostas | Aceitos |  Aceitação | Tok/ciclo |
| ------------------- | ----: | -------: | ---------: | --------: | --------: | ------: | ---------: | --------: |
| Target baseline     |     — |        — |     35,506 |     14,58 |         0 |       0 |          — |      1,00 |
| Q2 n7               |     7 |     0,25 |     30,521 |     16,95 |      1069 |     356 |     33,30% |      3,28 |
| Q2 n7               |     7 |     0,30 |     31,628 |     16,38 |       942 |     367 |     38,96% |      3,53 |
| Q2 n7               |     7 |     0,35 |     31,666 |     16,37 |       903 |     366 |     40,53% |      3,51 |
| Q2 n7 [ótimo n7]    |     7 | **0,40** | **32,184** |     16,06 |       781 |     363 |     46,48% |      3,44 |
| Q2 n7               |     7 |     0,45 |     32,061 |     16,16 |       778 |     358 |     46,02% |      3,32 |
| Q2 n4               |     4 |     0,20 |     34,889 |     14,89 |       680 |     340 |     50,00% |      2,98 |
| **Q2 n4 [RECORDE]** | **4** | **0,25** | **35,421** | **14,69** |       643 |     348 | **54,12%** |      3,12 |
| Q2 n4               |     4 |     0,30 |     34,321 |     15,10 |       658 |     337 |     51,22% |      2,93 |
| Q2 n4               |     4 |     0,35 |     34,763 |     14,94 |       632 |     340 |     53,80% |      2,98 |
| Q2 n4               |     4 |     0,40 |     34,146 |     15,20 |       577 |     337 |     58,41% |      2,93 |
| Q2 n4               |     4 |     0,45 |     33,374 |     15,54 |       555 |     332 |     59,82% |      2,84 |

Causa raiz da superioridade de n4: menor sobrecarga do target ao validar lotes de até 4 tokens (aceitação física ~3/ciclo; n7 rejeita 3–4 tokens/ciclo, saturando a verificação). **Ressalva da revisão Codex: o baseline de 35,506 usava temperatura 1,0 enquanto os candidatos usavam 0,0 — a comparação de "paridade 99,8%" não era um A/B com sampling igual** (origem em `scratch/run_experiment3_sweep.py:24`, que não passa temperatura ao baseline).

### 5.10 Revisão Codex 29/09 — recuperação de paridade (pacotes .82 → .83)

- **Pacote .82 (barreiras em lote) reprovava a validação numérica:** Probe 17 com `passed=false, status=-6`, **18/20 casos fora da tolerância, erro absoluto máximo 2,38716412** (todos os 4 casos de batch 5 e casos de trim parcial/reinjeção; cosseno mínimo 0,993429061 no batch 8). Causa: `ggml_fma_f32` com `mad()` nativo por padrão (commit `4b2d7e1`, pacote .78). A doc da Microsoft permite mad como operação fundida OU multiplicação+soma — emitir DXIL `FMad` não prova equivalência ao arredondamento único da referência CPU.
- **Pacote .83 — correção aplicada e validada no hardware real:**
  1. `ggml_float.hlsli:17`: emulação FP64 exata restabelecida como **padrão obrigatório**; `DFLASH_FAST_MAD` só para experimentos isolados.
  2. Todos os headers DXIL AOT regenerados (`dflash_attention`, `dflash_draft_quant_q8`, `dflash_q2k_q8k`, `dflash_q3k_q8k`); **SwiGLU voltou a 5900 bytes**, SHA-256 `54c5013fd5c44c0b4e076dec8a4f4e513c24574f3b456749b4fbfa18f654fd92` (antes o array embutido `kDflashDraftSwiGluDxil` divergia do fonte — 5900 vs 4628 bytes; a incoerência entre fonte HLSL e cabeçalho DXIL embutido foi eliminada).
  3. Barreira UAV do matmul em modo Strict corrigida em `dflash-gpu-draft-engine.cpp:756` (emitida incondicionalmente; antes o strict também suprimia com `destination == output`).
  4. **Probe 17: 20/20 casos aprovados, status 0, erro absoluto máximo 0, cosseno 1,000000** em todos os batches (2, 3, 5, 7, 8); `--capture-draft 5`: `captureCpuExact: true, maxAbs: 0` (eliminou a divergência em `kqv_out-0`); probes 8 (GEMV Q4_K) e 16 (Atenção Q4_0) status 0.
- **Campanha controlada (3 rodadas intercaladas, temp 0, mesmo prompt/seed):**

| Configuração                |     R1 |     R2 |     R3 |    Mediana | Wall mediana |
| --------------------------- | -----: | -----: | -----: | ---------: | -----------: |
| Target sozinho              | 35,216 | 35,355 | 35,320 | **35,320** |     14,651 s |
| Xbox Q2 n4/p0,25 + pipeline | 35,589 | 35,518 | 35,423 | **35,518** |     14,602 s |
| Xbox Q2 n7/p0,40 + pipeline | 30,409 | 32,223 | 32,121 | **32,121** |     16,128 s |

Ganho mediano do n4 sobre o target: **+0,5606%** (49 ms de parede em ~14,65 s). Controle n4 sem pipeline: 35,222 tok/s. n4/p0,25 permanece **candidato experimental, não padrão validado**.

- **Divergência ponta a ponta (P2):** textos de 512 tokens diferem entre modos (target `c83f1226...`, n4 pipeline `e7619405...`, n7 `c9c0e748...`, n4 sem pipeline `9a392fbd...`); primeira diferença no caractere 548. Determinismo local não demonstra equivalência com a referência — F6 permanece aberto (rastrear primeiro token/logit divergente).
- **Pipeline assíncrono:** não é "sobreposição total"; o controle de baixo aproveitamento desliga o pre-draft quando `hits * 4 < attempts` (`host-speculative.cpp:2196`). n4: 29 tentativas/7 reusos, off a partir da posição 269 (76 ciclos on, 87 off); n7: 13/3, off na 160. ~4,3% dos ciclos com reuse. Sem falhas RPC nos 6 traces.
- **Correção estatística:** 348/643 = 54,12% das propostas aceitas; 348/512 = 67,97% dos tokens gerados contabilizados como aceitos (todo token especulativo continua verificado pelo target).

### 5.11 Pré-organização Q2_K e A/B controlado (pacotes .84 → .85)

- **Variante .84:** expande cada bloco Q2_K de 84 para **288 bytes** no carregamento (256 inteiros `scale * quant`, 16 mínimos e bits FP16 de `d`/`dmin`); shader parou de desempacotar Q2 em LDS e reduziu barreiras por bloco de 3 para 2; ainda usa `s_scaled_w[64]` em LDS e `d`/`dmin` convertidos na GPU (conversão FP32 na CPU **não** implementada). **Falhou Probe 17 (18/20, status -6)** — os geradores DXIL não definiam `DFLASH_EXACT_FP64_EMULATION` e `mad()` voltou por padrão (política numérica do .83 desfeita).
- **Pacote .85 (correção):** `ggml_fma_f32` FP64 exato por padrão de novo; DXIL regenerados; SwiGLU 5900 bytes; MSIX SHA-256 `5063f80768a1178e7b17c1f847ab6987454a47a8a930f74ac4528cdac59a770f`. **Probe 17: status 0, 20/20, erro 0**; capture-draft 5/5 exato; replay 369/369 com captura Q2 `5ff0aeb6…`. Pico de memória pós-lançamento: 3,118 GiB (limite reportado 5 GiB).
- **A/B alternado `.83` vs `.85` (B1–A1–B2–A2–B3, mesma captura Q2; `.83` repacotado como `HJotha.XboxDFlashRpcBench83_1.0.0.84` para instalação lado a lado, exe byte a byte idêntico; identidade temporária removida ao final):**

| Medida (mediana das fases)        |        `.83` |        `.85` |   Variação |
| --------------------------------- | -----------: | -----------: | ---------: |
| Probe 17 batch 8 `gpuMs` inicial  |    39,010 ms |    32,014 ms |    −17,93% |
| RPC sintético n4, ciclo p50       |    44,464 ms |    36,735 ms |    −17,38% |
| RPC sintético n7, ciclo p50       |    47,353 ms |    39,524 ms |    −16,53% |
| Geração 512 tok, temp 1, n4/p0,25 | 29,345 tok/s | 31,821 tok/s | **+8,44%** |
| Geração 512 tok, wall             |     17,668 s |     16,303 s | **−7,72%** |
| Geração 128 tok, greedy, n4/p0,25 | 32,619 tok/s | 35,442 tok/s | **+8,66%** |
| Geração 128 tok, wall             |      3,965 s |      3,654 s | **−7,85%** |

Gate/up Q2 por camada: 2,028 → 1,524 ms (−24,88%); `ffn_out` Q3 1,237/1,236 ms. Hashes de texto coincidiram entre fases; propostas/aceites/ciclos 647/346/163 e 149/88/38; sem falhas RPC. O primeiro controle RPC usou `effective=cpu` e foi descartado da comparação GPU.

**Conclusão:** ganho real e repetido do layout Q2_K .85 (~18% no draft isolado, +8–9% TPS ponta a ponta neste prompt), sem controle térmico estrito. O layout usa **3,429× os bytes GGUF**; conversão FP32 na CPU e eliminação total do LDS seguem pendentes. A `.85` está instalada, com `dflash-backend.txt` restaurado byte a byte a `cpu`. F6 segue aberto (A/B não resolve divergência target/pipeline nem mede contra target sem DFlash).

## 6. Otimização de kernels (F7) — o que mudou

1. **Cargas cooperativas 128-bit (`Load4`)** em vez de loads por lane para pesos W e ativações Q8 (LDS `uint s_q8[512]`, `s_d8[8]`).
2. **LDS apertado:** reduções `[2*8*8]` e `[8*8]` → < 2,8 KB/workgroup, alta ocupação de CUs.
3. **Redução paralela de batch na Wave 0** (lane < 8), eliminando loops serializados e reduzindo pressão de VGPR.

| Versão                             | DRAFT compute               | Cycle RTT          | Banda efetiva  | Probe 17                                                     |
| ---------------------------------- | --------------------------- | ------------------ | -------------- | ------------------------------------------------------------ |
| 1.0.0.42 (base)                    | 64,516 ms                   | 74,483 ms          | 14,21 GB/s     | referência                                                   |
| 1.0.0.43                           | 61,950 ms                   | 71,820 ms          | 14,80 GB/s     | bit-idêntico                                                 |
| 1.0.0.44                           | 46,994 ms                   | 56,941 ms          | 19,51 GB/s     | bit-idêntico                                                 |
| 1.0.0.45                           | **44,668 ms**               | **54,508 ms**      | **20,52 GB/s** | **bit-idêntico (0 diffs)**                                   |
| 1.0.0.77 (FP64 legado + FFN AGY)   | 37,187 ms                   | ~45,0 ms           | —              | bit-idêntico                                                 |
| 1.0.0.78 (mad nativo)              | 25,281 ms                   | 31,838 ms          | —              | **cosseno 0,997223 (regressão numérica)**                    |
| 1.0.0.80 minimal (barreiras)       | **25,182 ms**               | **31,765 ms**      | —              | cosseno 0,997223*                                            |
| 1.0.0.82 (ResourceBarrier em lote) | **25,162 ms**               | **31,364 ms**      | —              | bit a bit (parity_passed) mas **probe 17 status -6 (18/20)** |
| 1.0.0.83 (FP64 exato restaurado)   | ~31,99–39,02 ms†            | —                  | —              | **20/20, status 0, erro 0, cosseno 1,000000**                |
| 1.0.0.85 (pré-org Q2_K)            | 31,99 ms (batch 8 initial)† | 36,735 ms (n4 p50) | —              | **20/20, status 0, erro 0**                                  |

_\* No .80, cosseno 0,997223 refletia a tolerância provisória do modelo Q2 com AVX2 CPU; o hash de saída do pipeline de produção permaneceu idêntico entre .78 e .80._
_† O .83/.85 volta ao FP64 exato, que é mais lento que mad() — o ganho do .85 vem do layout Q2_K, não do mad (descartado por quebrar a paridade)._

Meta F7 original (≤ 46 ms, ≥ 20 GB/s) superada já no .45; a série .78–.85 baixou o compute para ~25,2 ms e depois ~32 ms com paridade exata recuperada (o mad que dava 25,2 ms foi rejeitado por reprovar a validação).

**Experimentos descartados ao longo da série:**

- **Redução via `WaveReadLaneAt` (pacote 1.0.0.79):** cosseno −0,008 no batch 8. RDNA2 usa **Wave32**: grupo de 64 threads = 2 waves; lane > 31 acessava fora da wave (valores indefinidos). Rejeitada; redução LDS mantida (NMSE ~1e-14, cosseno 1,000000).
- **Double-buffering LDS (1.0.0.81):** +46,6% (contenção de portas de banco LDS no RDNA2). Ver §5.9.
- **Fusão Joint GEMV Gate+Up:** rejeitada duas vezes (28/09 e deep review 29/09): volume de pesos irredutível (46,45 MB), economia de ativações < 1,9% do tráfego, e pressão de VGPR > 64 derrubaria occupancy de 8→4 waves/SIMD, degradando a ocultação de latência.

## 7. Paridade numérica e correções aplicadas

- **Referência:** grafo llama.cpp CPU real com MSVC AVX2/FMA/repack no Xbox (replay de 369 frames Q3 e Q2 bit a bit — captura Q2 `5ff0aeb6…` reconfirmada 369/369 no .85). Não é validação independente HF/PyTorch.
- **Correções no caminho GPU:** acumulação CPU-compatível Q2/Q3/Q4/Q5, RMSNorm, constantes RoPE, arredondamento de convolução dinâmica, SwiGLU AVX2, atenção online, `expf` UCRT (fingerprint 1.048.576 amostras igual em HPBEATS e Xbox).
- **Packer KV Q4_0:** F32 rounding explícito de `x + 8.5` antes de truncar (o DXIL usava fast add + convert); corrigiu o último byte de escala.
- **Root cause F4:** flip de bin Q8_K por delta ~1e-6 F32 em `ffn_norm-1`; correlação 0,999991, resíduo rms 1,07e-7. Kernels/quantizer/dots são fiéis; **bit-identidade entre reduções diferentes é impossível** e não é exigida (nota F0 do plano).
- **Política de precisão (29/09):** `ggml_float.hlsli` define **FP64 exato como padrão obrigatório**; `DFLASH_FAST_MAD` só por flag explícita. O `mad()` nativo do .78 gerava DXIL `FMad` (opcode 46) e o driver RDNA2 mapeia para `V_FMAC_F32`/`V_FMA_F32`, mas o `FMad` não garante arredondamento único (a spec D3DHLSL permite fundido ou mul+add) — reprovou 18/20 casos do Probe 17 (erro máx 2,38716412). Regenerar DXIL sem a flag certa desfaz a política (caso do .84).
- **Estado validado no pacote 1.0.0.85:** replay completo Q2 (369/369, `5ff0aeb6…`) + Q3; Probe 17 20/20 status 0 erro 0 cosseno 1,000000; capture-draft 5/5 exato; probes 8 e 16 status 0; 40 casos curtos maxAbs=0; 6 casos de TRIM; 5 testes Python; build OK.
- **Divergência remanescente (F6):** saída de 512 tokens GPU ≠ target sozinho (sequencial = pipeline entre si); primeira diferença no caractere 548 (target `c83f1226…` vs n4 `e7619405…`). Rastreamento token/logit de verificação/rollback ainda pendente para equivalência total.

## 8. Próximas melhorias planejadas (para bater os 35 tok/s de forma sustentada)

**Melhoria 1 — Aceitação (`draft-p-min`):** varredura concluída (29/09): **n4/p0,25 é o melhor candidato** (35,421 AGY; 35,518 controlado), n4 domina n7 em toda a faixa. Faltam: n5, demais p_min com A/B válido (temp igual ao baseline), outros workloads. Alvo ≥ 4 tokens/ciclo.

**Melhoria 2 — Fusão RPC:** opcode `SYNC_AND_DRAFT` implementado e medido (1,234 ms/ciclo), mas **SYNC e DRAFT continuam com submissões/fences separados** (`dedicated.cpp:558/577`; engines `:793`/`:1181`). Quantificar o custo da segunda submissão e fundir numa command list/fence.

**Melhoria 3 — Pipeline assíncrono:** implementado e corrigido, mas o controle de baixo aproveitamento (desliga quando reuso < 25%) atua na prática (~4,3% dos ciclos com reuse). Comparar on/off após fechar os gates; aperfeiçoar a hipótese sem retirar o controle.

**Melhoria 4 (avaliada e descartada) — Fusão Joint GEMV Gate+Up:** rejeitada em 28/09 e reconfirmada no deep review 29/09 (pesos irredutíveis + VGPR pressure + occupancy).

**Melhoria 5 (nova, validada) — Pré-organização Q2_K (.85):** layout 288 bytes/bloco com `scale * quant` inteiros; ganho real de ~18% no draft e +8–9% TPS. Pendências: conversão FP32 na CPU, eliminação total do LDS (`s_scaled_w`), custo de 3,429× memória.

**Próximos passos concretos (ordem da revisão Codex):**

1. **F6 — divergência ponta a ponta:** comparar token e logits no primeiro ponto divergente entre target, GPU sequencial e GPU com pipeline; verificar estado KV/rollback e influência de batch/quantização. Hash de texto único não substitui a investigação.
2. **Repetir campanha pós-correção:** ≥ 3 rodadas intercaladas com parâmetros iguais (workload original temp 1 e greedy temp 0), incluindo CPU RPC e controles de trajetória; n4/p0,25 como candidato, não padrão.
3. **Medir antes de ampliar otimização:** custo da 2ª submissão SYNC/DRAFT; retorno do pipeline por tentativas/reusos/descartes/tempo até autodesligamento; variante sem `s_scaled_w` em LDS só após A/B sustentado do .85.
4. **Sem prometer 40 tok/s nem declarar a fase de benefício concluída** enquanto F6 não fechar.

## 9. Operação passo a passo (deploy e teste)

```powershell
cd C:\Users\hjotha\worktrees\xllama-dflash-xbox

# 1. Recompilar shaders alterados (ex.: Q3_K / quant online / ggml_float)
python research\dflash2-xbox\gpu-implementation\q3k_q8k_compile.py
python research\dflash2-xbox\gpu-implementation\draft_compile_quant_q8.py
# Após mudar ggml_float.hlsli, regenerar TODOS os DXIL dependentes (Q2/Q3/Q8/atenção/draft ops)

# 2. Incrementar versão em uwp/dflash-rpc/Package.appxmanifest (ex.: 1.0.0.86)
# 3. Build do MSIX
research\dflash2-xbox\build-dflash-rpc.cmd

# 4. Assinar
& "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe" sign /fd SHA256 /sha1 CD6D10EC8332955BC5C1074BC20A297D0625F1F4 /s My uwp\dflash-rpc\AppPackages\DFlashRpc\DFlashRpc_1.0.0.86_x64_Test\DFlashRpc_1.0.0.86_x64.msix

# 5. Instalar no Xbox (Device Portal)
python research\dflash2-xbox\xbox_portal.py "/api/app/packagemanager/package?package=DFlashRpc_1.0.0.86_x64.msix" --method POST --upload <msix>

# 6. Configurar backend/modelo em LocalState (dflash-backend.txt: cpu|d3d12; dflash-model.txt: dflash-q2.gguf; dflash-barrier-mode.txt: strict|minimal)

# 7. Iniciar app e medir
python research\dflash2-xbox\dflash_dedicated_bench.py --host 192.168.1.26 --port 50053 --warmup 10 --iterations 20 --drafts 4
# Esperado no .85 (FP64 exato): draft_xbox_compute_us ~31.000-32.000, sync ~2.400, RTT ~36.000-40.000

# 8. Paridade (probe 17) e campanha real na .57
python research\dflash2-xbox\gpu-implementation\run_gpu_probe.py --host 192.168.1.26 --port 50053 --opcode 17 --output ...probe17-run.json
ssh hjotha@192.168.1.57   # bash /tmp/dflash-gpu-test.sh q3_gpu 4 /home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf 0
```

O `dflash-gpu-test.sh` para/restaura `llama-server-root.service` via trap EXIT, sobe instância temporária na porta 59567 e roda 512 tokens. Para campanhas comparáveis usar `sweep_acceptance.py`/`run_review_campaign.py` (intercala rodadas, restaura o serviço em `finally`).

## 10. Regras de segurança operacional

1. **Sempre restaurar** `dflash-backend.txt` para `cpu` ao fim dos testes (evita contexto D3D12 persistente/watchdog no Xbox).
2. **Preservar** `llama-server-root.service` na `.57` (vital, produção); conferir `systemctl is-active` após campanhas.
3. **Heap Xbox (Game 5 GiB):** Q8_0 + contexto CPU de trace + engine GPU simultâneos → `0x8007000E`; Q8_0 funciona sozinho. Pico medido do .85: 3,118 GiB.
4. **Git:** nunca commit direto em `main`; trabalhar em `research/dflash-xbox-rpc`; push para remote `private` só com validação 100%.
5. **Protocolo:** nunca aumentar structs de resposta silenciosamente; rejeitar entradas inválidas antes de mutar estado.
6. **Política de precisão:** mudanças em `ggml_float.hlsli` exigem regenerar todos os DXIL dependentes e revalidar Probe 17 no Xbox; `DFLASH_FAST_MAD`/`mad()` quebram a paridade (18/20 reprovaram).
7. **A/B de pacotes:** para comparar versões lado a lado no Xbox, repacotar o executável antigo sob identidade temporária (ex.: `XboxDFlashRpcBench83`); nunca rebaixar a versão instalada sem necessidade; remover identidades temporárias ao final.

## 11. Estado atual do repositório (29/09 — fim da revisão Codex + A/B .85)

- Últimos commits (todos locais na worktree, **sem merge no main**):
  - `a8093b2` — docs: n_max=4 sweep evaluation e sweet spot calibration
  - `c4e0f14` — feat: batch D3D12 resource barriers + 3 experimentos
  - `5b245f1` — docs: deep review de gargalos pós-FP64
  - `cf3257f` — perf: DFLASH_BARRIER_MODE (strict/minimal) + eliminação de barreiras UAV redundantes (pacote 1.0.0.80)
  - `4b2d7e1` — perf: substituição da emulação FP64 por mad nativo (pacote 1.0.0.78)
  - Anteriores: `8995840`, `3560da1`, `0e23450`, `8b3faa7`, `e579e2a`, `1266126`, `73a55c8`, `c73cb31`
- **Alterações não commitadas na worktree** (variante .85 + evidências da revisão, preservadas sem commit): `ggml_float.hlsli`, `q2k_q8k_matmul.hlsl`, DXIL gerados (`dflash_attention_dxil.h`, `dflash_draft_quant_q8_dxil.h`, `dflash_q2k_q8k_dxil.h`, `dflash_q3k_q8k_dxil.h`), `Package.appxmanifest`, `dedicated.cpp`, `dflash-gpu-draft-engine.cpp`, `dflash-gpu-engine.cpp`, plano mestre, relatório `review-agy-and-next-steps-2026-09-29.md` (untracked) + diretórios `scratch/` e `gpu-implementation/codex-review-20260929/` (untracked).
- Cliente host `.57`: BeeLlama `12231b55e` na branch `fix/dflash-xbox-pipeline` (checkout limpo na verificação final); lib `libllama-common.so.0.4.7` SHA256 `b46f159f...`.
- Xbox final: **app 1.0.0.85** ativo (identity temporária `.83` desinstalada), porta RPC 50053 OK; `dflash-backend.txt` restaurado a `cpu`; produção `.57` (`llama-server-root.service`, 8090) ativa e `/health` `{"status":"ok"}`; sem listener temporário em 59567.

## 12. Referências (fontes primárias na worktree)

| Arquivo                                                                                 | Conteúdo                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `research/dflash2-xbox/dflash-gpu-implementation-plan-2026-09-28.md`                    | Plano mestre F0–F7, arquitetura, gates, riscos                                                                                                                                                                                                                                                                                                                                                                                                      |
| `research/dflash2-xbox/gpu-implementation/f0-f2-results-2026-09-28.md`                  | F0–F2: captura, device, kernels HLSL vs DirectML                                                                                                                                                                                                                                                                                                                                                                                                    |
| `research/dflash2-xbox/gpu-implementation/f3-f4-results-2026-09-28.md`                  | F3–F4: KV/SYNC, divergência camada 1, root cause, F5                                                                                                                                                                                                                                                                                                                                                                                                |
| `research/dflash2-xbox/gpu-implementation/f6-real-production-campaign-2026-09-28.md`    | Campanha real 21,26 tok/s                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `research/dflash2-xbox/gpu-implementation/f7-kernel-optimization-results-2026-09-28.md` | Kernels Load4 → 25,75 tok/s                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `research/dflash2-xbox/gpu-implementation/f7-nmax-sweep-2026-09-28.md`                  | Varredura n_max                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `research/dflash2-xbox/gpu-implementation/f7-verdict-2026-09-28.md`                     | Veredito de viabilidade (banda × aceitação)                                                                                                                                                                                                                                                                                                                                                                                                         |
| `research/dflash2-xbox/gpu-implementation/status-2026-09-28.md`                         | Paridade numérica + Q2 n7 + comparativo pacotes .66/.70/.75                                                                                                                                                                                                                                                                                                                                                                                         |
| `research/dflash2-xbox/gpu-implementation/plan-acceptance-pipelining-2026-09-28.md`     | Melhorias 1–3 (aceitação, fusão, pipelining)                                                                                                                                                                                                                                                                                                                                                                                                        |
| `research/dflash2-xbox/dflash2-gpu-deep-review-and-bottlenecks-2026-09-29.md`           | **Deep review: perfil de 112 checkpoints, causa raiz GEMV latency-bound, barreiras UAV, Wave32, rejeição de fusão**                                                                                                                                                                                                                                                                                                                                 |
| `research/dflash2-xbox/experimentos-melhorias-gpu-2026-09-29.md`                        | **3 experimentos: double-buffering (rejeitado), ResourceBarrier em lote (aprovado), calibração p_min (recorde 35,421)**                                                                                                                                                                                                                                                                                                                             |
| `research/dflash2-xbox/review-agy-and-next-steps-2026-09-29.md`                         | **Revisão Codex: mad() reprova paridade, .83 recupera F4, campanha controlada, pré-org Q2_K .84/.85 + A/B**                                                                                                                                                                                                                                                                                                                                         |
| `research/dflash2-xbox/gpu-implementation/codex-review-20260929/` + `scratch/`          | Evidências: `probe17_83.json`, `probe17_83_capture5.json`, `probe17_85_q2.json`, `probe17_85_capture5.json`, `probe8_83.json`, `probe16_83.json`, `probe17_84_q2_correct.json`, `probe17-live.json` (.82 falha), `redeploy_83.jsonl`, `q2-ab-20260929/summary.json` e `q2-ab-20260929/f6/`, `agy-sweep-results.jsonl`, `results.jsonl`, `analysis.json`, `campaign.log`, `manifest.json`, `final-host-status.json`, `historical-probe-summary.json` |
| `research/dflash2-xbox/sweep_acceptance.py` + `test_sweep_acceptance.py`                | Executor de varredura de aceitação (usado no Experimento 3 e na campanha controlada)                                                                                                                                                                                                                                                                                                                                                                |
| `research/dflash2-xbox/redeploy-1.0.0.76/77/78.jsonl`                                   | Evidência de deploy dos pacotes da série                                                                                                                                                                                                                                                                                                                                                                                                            |
| Sessão AGY `4c44ecef-58b1-417f-ae20-bc2b95be9565` (transcript local)                    | Otimização FFN 1.0.0.77, diagnóstico de gargalo, relatório final 20h49                                                                                                                                                                                                                                                                                                                                                                              |
| Sessão AGY `ff7b6fe1-4c97-4943-8520-b4bf6dddd1d2` (transcript local)                    | Reavaliação Joint GEMV (decisão de não implementar), verificação saúde .57/Xbox 21h18                                                                                                                                                                                                                                                                                                                                                               |
| `research/dflash2-xbox/CONTINUATION.md`                                                 | Guia mestre de continuação (operacional, passo a passo)                                                                                                                                                                                                                                                                                                                                                                                             |
| `research/dflash2-xbox/current-status-2026-09-27.md`                                    | Histórico 1.0.0.2–1.0.0.7 (Game classification, memória)                                                                                                                                                                                                                                                                                                                                                                                            |
| `research/dflash2-xbox/consolidation-2026-09-27.md`                                     | Consolidação 27/09 (transporte, arquitetura)                                                                                                                                                                                                                                                                                                                                                                                                        |
| `uwp/dflash-rpc/`                                                                       | Servidor dedicado (`dedicated.cpp`), engines GPU (`dflash-gpu-engine.cpp`, `dflash-gpu-draft-engine.cpp`)                                                                                                                                                                                                                                                                                                                                           |
| `shaders/dflash/`                                                                       | HLSL: `ggml_float.hlsli` (política de precisão), `q2k_q8k_matmul`, `q3k_q8k_matmul`, `q6k_q8k_matmul`, `q8_0_q8k_matmul`, `dflash_draft_quant_q8`, `dflash_attention`, `dynamic_conv`, `hadamard_transform`                                                                                                                                                                                                                                         |

**Fontes externas:** [DFlash2 publicado (incoai)](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF), [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), [grafo DFlash2 llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp), [upstream PR DFlash2](https://github.com/ggml-org/llama.cpp/pull/27342), [Microsoft Learn — mad](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/mad).
