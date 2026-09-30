# DFlash2 no Xbox Series X — Consolidação Completa (até 28/09/2026, 21h20)

Documento único consolidando toda a pesquisa, implementação e resultados medidos do DFlash2 rodando no Xbox. Gerado a partir dos relatórios e evidências em `research/dflash2-xbox/` da worktree ativa. Atualizado com os resultados das sessões AGY de 28/09 noite (otimização FFN com pacote 1.0.0.77 e reavaliação da fusão Joint GEMV).

- **Worktree:** `C:\Users\hjotha\worktrees\xllama-dflash-xbox`
- **Branch:** `research/dflash-xbox-rpc` (base de implementação `8b3faa7`)
- **Repositório privado:** `git@github.com:hjotha/xllama-dflash-xbox.git` (remote `private`)
- **Console:** Xbox Series X, `192.168.1.26`
- **Host do target:** GOKAYA, `192.168.1.57`

---

## 1. Objetivo

Rodar o **DFlash2** (drafter de decodificação especulativa de 5 camadas, ~1,92B parâmetros, derivado do Qwen3.8-27B) na **APU AMD RDNA2 do Xbox Series X via D3D12/HLSL**, enquanto o modelo alvo (Swift-1.5-Qwen3.8-27B, 27B) continua na RTX 4070 da `.57`. Objetivo final: superar os **34,99 tok/s** do Swift rodando sozinho.

O resultado hoje: **implementação funcional com ganho sobre CPU (1,99x)**, mas ainda **sem benefício final sobre o target isolado** (25,75 tok/s vs 34,99 tok/s).

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
- **Ciclo sequencial real (n_max=4):** ~109 ms → ~2,81 tokens/ciclo. Componentes: verificação target ~55–58 ms, Xbox SYNC+DRAFT ~44,7+3,1 ms, RPC ~5–11 ms, selector ~5–7 ms.
- O app RPC **não inclui** `token_embd.weight`/`output.weight` do target; o corte respeita a regra de manter pesos principais na `.57`.

## 3. Inventário de infraestrutura

| Item                    | Valor                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Xbox IP / Device Portal | `192.168.1.26` / `https://192.168.1.26:11443` (HTTP Basic)                                                                                  |
| Porta RPC dedicada      | `50053` TCP (genérica GGML RPC: `50052`)                                                                                                    |
| Pacote UWP              | `HJotha.XboxDFlashRpc` (versões testadas 1.0.0.2 → 1.0.0.77)                                                                                |
| AUMID                   | `HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App`                                                                                                    |
| Classificação           | Game por padrão (`defaultuwpcontenttypetogame=True`); `AppMemoryUsageLimit` = 5.368.709.120 bytes (5 GiB)                                   |
| Toolchain               | MSVC v143, Windows SDK 10.0.26100.0, DXC em `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe`                           |
| Certificado             | SHA1 `CD6D10EC8332955BC5C1074BC20A297D0625F1F4` (store `My`)                                                                                |
| Cliente .57             | `/home/hjotha/beellama.cpp/build-dflash-xbox-dl/bin/llama-server` (build 12304, commit `12231b55e`; lib `b46f159f...`)                      |
| Serviço de produção .57 | `llama-server-root.service` (127.0.0.1:8090) — **sempre restaurado após testes**                                                            |
| GGUF Q3_K_M             | 916.702.560 bytes, SHA-256 `72809a53e89cb959a3149c9b3752ad181b640164b51a3b9f445712520beec5dc`; inventário: 32 F32, 34 Q3_K, 13 Q4_K, 2 Q5_K |

## 4. Estado por etapa (plano F0–F7)

| Etapa                 | Estado | Resumo                                                                                                                                                                                                                                                      |
| --------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F0** Referência     | ✅     | Captura real de 369 frames (SHA `b55ef2a1...`), replay CPU determinístico **bit a bit**, fixtures correlacionadas (182 ciclos, 728 propostas, 329 aceitos), inventário de 81 tensores GGUF                                                                  |
| **F1** Dispositivo    | ✅     | D3D12 hardware (adapter `SraKmd_arden`), Shader Model 6, WaveOps (64 lanes), timestamps GPU; shaders compilados offline com DXC                                                                                                                             |
| **F2** Kernels        | ✅     | **Decisão HLSL**: Q3_K/Q4_K/Q5_K reais validados e 2,1–5,6x mais rápidos que CPU ggml com readback. DirectML Q4_K FP16 correto porém mais lento e 3,56x mais memória; Q3_K DirectML falhou (0x887A0005 / crash)                                             |
| **F3** KV + SYNC      | ✅     | Cache físico Q4_0 GPU = CPU em **70.399/70.400 bytes** (1 byte de escala V pendente no meio da investigação; resolvido em seguida com fix de packer). Gate Q8_K de escala fechado com F32 rounding explícito                                                |
| **F4** DRAFT          | ✅     | Root cause encontrado: **sensibilidade a fronteira de quantização** (Q8_K flip de 1 bin por ~1e-6 F32, amplificado ~667x pelo FFN). Paridade exata atingida com kernels fiéis; bit-identidade entre ordens de redução diferentes é impossível por definição |
| **F5** RPC + fallback | ✅     | Executor GPU integrado: `LocalState\dflash-backend.txt` (`cpu` padrão / `d3d12`); fallback CPU antes da sequência, invalidação + erro após falha de estado; smokes CPU/GPU passam (bench: GPU DRAFT 64,5 ms vs CPU 131,9 ms)                                |
| **F6** TPS real       | ✅     | Campanha de produção ponta a ponta: **d3d12 21,26 tok/s vs CPU 12,92 tok/s (1,65x)**, saída SHA-256 idêntica nas 6 execuções                                                                                                                                |
| **F7** Otimização     | ✅     | Kernels cooperativos Load4/LDS → **20,52 GB/s** efetivos, DRAFT 44,67 ms, **TPS 25,75** (1,99x sobre CPU). Ver §6                                                                                                                                           |

**Conclusão de viabilidade (F7):** com aceitação atual de 2,81 tokens/ciclo, empatar Swift exige ~70 GB/s (fora do alcance medido de 25–40 GB/s). Com aceitação ≥ 4 tokens/ciclo, o orçamento cai para ~20 GB/s — **dentro do alcance**. Caminho: kernels classe gpugemv + calibração de aceitação (n_max já está no ótimo medido: 4).

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

### 5.6 Anatomia do ciclo (fused n7, média)

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

**Resultado medido: sem ganho — leve regressão (+0,60 ms).** Diagnóstico do gargalo (sessão AGY 20h50, report final):

1. **Teto físico de banda:** o ciclo de draft lê **483,9 MB de pesos** nas 5 camadas (Gate+Up 292,4 MB + Down 191,5 MB); com ~20,52 GB/s efetivos na sandbox UWP, o limite físico é **483,9 MB / 20,52 GB/s = 23,58 ms** — irredutível para dispatches sequenciais.
2. **Penalidade FP64 no RDNA2:** operações de dupla precisão rodam a **1/32 da taxa FP32**; cada acumulação de bloco usa `ggml_fma_f32` (recorre a `double` para reproduzir fielmente a ordem/mantissa de arredondamento AVX2 do CPU).
3. **Ativações cabem em cache:** vetor de entrada para 8 tokens ≈ 46 KB, dentro de L1/L2 — o gargalo é 100% streaming de pesos VRAM + latência de acumulação FP64.

**Decisão (sessão AGY 21h18): a fusão Joint GEMV (Gate+Up num único shader) NÃO deve ser implementada** — análise dos dados técnicos concluiu que deixaria a inferência mais lenta ou estagnada. Alternativa mapeada: overhead de RPC como principal candidato de gargalo fora do compute Xbox. Próximo passo recomendado: campanha end-to-end com `run_pipeline_campaign.py` na `.57` (Q2, pipeline assíncrono).

## 6. Otimização de kernels (F7) — o que mudou

1. **Cargas cooperativas 128-bit (`Load4`)** em vez de loads por lane para pesos W e ativações Q8 (LDS `uint s_q8[512]`, `s_d8[8]`).
2. **LDS apertado:** reduções `[2*8*8]` e `[8*8]` → < 2,8 KB/workgroup, alta ocupação de CUs.
3. **Redução paralela de batch na Wave 0** (lane < 8), eliminando loops serializados e reduzindo pressão de VGPR.

| Versão          | DRAFT compute | Cycle RTT     | Banda efetiva  | Probe 17                   |
| --------------- | ------------- | ------------- | -------------- | -------------------------- |
| 1.0.0.42 (base) | 64,516 ms     | 74,483 ms     | 14,21 GB/s     | referência                 |
| 1.0.0.43        | 61,950 ms     | 71,820 ms     | 14,80 GB/s     | bit-idêntico               |
| 1.0.0.44        | 46,994 ms     | 56,941 ms     | 19,51 GB/s     | bit-idêntico               |
| **1.0.0.45**    | **44,668 ms** | **54,508 ms** | **20,52 GB/s** | **bit-idêntico (0 diffs)** |

Meta F7 atingida: ≤ 46,000 ms e ≥ 20 GB/s. Q2/Q3 compartilham ganhos: grupo reduzido de 256→64 threads no Q2/Q3, 128 no Q4 down (Q5 mantém 256).

## 7. Paridade numérica e correções aplicadas

- **Referência:** grafo llama.cpp CPU real com MSVC AVX2/FMA/repack no Xbox (replay de 369 frames Q3 e 131 MB de captura Q2 bit a bit). Não é validação independente HF/PyTorch.
- **Correções no caminho GPU:** acumulação CPU-compatível Q2/Q3/Q4/Q5, RMSNorm, constantes RoPE, arredondamento de convolução dinâmica, SwiGLU AVX2, atenção online, `expf` UCRT (fingerprint 1.048.576 amostras igual em HPBEATS e Xbox).
- **Packer KV Q4_0:** F32 rounding explícito de `x + 8.5` antes de truncar (o DXIL usava fast add + convert); corrigiu o último byte de escala (CPU 22 vs GPU 23).
- **Root cause F4:** flip de bin Q8_K por delta ~1e-6 F32 em `ffn_norm-1` (0 de 5.120 valores na linha 0, 1 na linha 1); correlação 0,999991 entre delta replicado e observado, resíduo rms 1,07e-7. Kernels/quantizer/dots são fiéis; **bit-identidade entre reduções diferentes é impossível** e não é exigida (nota F0 do plano).
- **Estado final do pacote 1.0.0.75:** replay completo Q2 e Q3 (byte-idêntico em status/header/output, excluindo timing) + 40 casos curtos maxAbs=0; probe de KV com `nextafter(2.5f,0)+8.5f` → 11; 6 casos de TRIM; 5 testes Python; build OK.
- Diferença remanescente: saída de 512 tokens GPU ≠ target sozinho (mas sequencial = pipeline); rastreamento token/logit de verificação/rollback ainda pendente para equivalência total.

## 8. Próximas melhorias planejadas (para bater os 35 tok/s)

**Melhoria 1 — Aceitação (`draft-p-min`):** varredura 0,05–0,85 com n_max 4/5 e seletor Q8_0 no host; alvo ≥ 4,0 tokens/ciclo → 36,7 tok/s sem mudar C++. Evidência parcial: p_min 0,65 elevou aceitação a 65,8%, mas gerou ciclos vazios e caiu o TPS — precisa calibrar com custo do DRAFT pago.

**Melhoria 2 — Fusão RPC:** opcode `SYNC_AND_DRAFT` já implementado e medido (1,234 ms/ciclo; estimativa antiga 5–7 ms não confirmada). Próximo passo: fundir SYNC+DRAFT na mesma command list/fence (ainda há duas submissões).

**Melhoria 3 — Pipelining assíncrono:** pre-draft no Xbox enquanto a 4070 verifica; implementado e corrigido (hipótese greedy impossível identificada e resolvida), mas ganho medido pequeno (1,76% n7, 4 reusos/run); descartes ocorrem quando 4 ou 2 tokens são aceitos (só promove pre-draft com exatamente 1 aceito). Sem benefício de desempenho relevante até agora.

**Melhoria 4 (avaliada e descartada) — Fusão Joint GEMV Gate+Up:** análise técnica da sessão AGY de 28/09 noite concluiu que fundir `ffn_gate` e `ffn_up` num único shader não traria ganho (gargalo é banda de VRAM — 483,9 MB/ciclo → piso de 23,58 ms — e taxa FP64 1/32 no RDNA2, não emissão de instruções). Decisão: não implementar.

**Próximo passo concreto:** rodar `run_pipeline_campaign.py` na `.57` com Q2 no Xbox para medir ganho real do pipeline assíncrono ponta a ponta; investigar overhead de RPC fora do compute Xbox (11,164 ms/ciclo medidos) como alvo alternativo de redução.

## 9. Operação passo a passo (deploy e teste)

```powershell
cd C:\Users\hjotha\worktrees\xllama-dflash-xbox

# 1. Recompilar shaders alterados (ex.: Q3_K / quant online)
python research\dflash2-xbox\gpu-implementation\q3k_q8k_compile.py
python research\dflash2-xbox\gpu-implementation\draft_compile_quant_q8.py

# 2. Incrementar versão em uwp/dflash-rpc/Package.appxmanifest (ex.: 1.0.0.48)
# 3. Build do MSIX
research\dflash2-xbox\build-dflash-rpc.cmd

# 4. Assinar
& "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe" sign /fd SHA256 /sha1 CD6D10EC8332955BC5C1074BC20A297D0625F1F4 /s My uwp\dflash-rpc\AppPackages\DFlashRpc\DFlashRpc_1.0.0.48_x64_Test\DFlashRpc_1.0.0.48_x64.msix

# 5. Instalar no Xbox (Device Portal)
python research\dflash2-xbox\xbox_portal.py "/api/app/packagemanager/package?package=DFlashRpc_1.0.0.48_x64.msix" --method POST --upload <msix>

# 6. Configurar backend/modelo em LocalState (dflash-backend.txt: cpu|d3d12; dflash-model.txt: dflash-q3.gguf etc.)

# 7. Iniciar app e medir
python research\dflash2-xbox\dflash_dedicated_bench.py --host 192.168.1.26 --port 50053 --warmup 10 --iterations 20 --drafts 4
# Esperado: draft_xbox_compute_us ~44.600-47.700, sync ~3.100, RTT ~54.000-57.000

# 8. Paridade (probe 17) e campanha real na .57
python research\dflash2-xbox\gpu-implementation\run_gpu_probe.py --host 192.168.1.26 --port 50053 --opcode 17 --output ...probe17-run.json
ssh hjotha@192.168.1.57   # bash /tmp/dflash-gpu-test.sh q3_gpu 4 /home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf 0
```

O `dflash-gpu-test.sh` para/restaura `llama-server-root.service` via trap EXIT, sobe instância temporária na porta 59567 e roda 512 tokens.

## 10. Regras de segurança operacional

1. **Sempre restaurar** `dflash-backend.txt` para `cpu` ao fim dos testes (evita contexto D3D12 persistente/watchdog no Xbox).
2. **Preservar** `llama-server-root.service` na `.57` (vital, produção); conferir `systemctl is-active` após campanhas.
3. **Heap Xbox (Game 5 GiB):** Q8_0 + contexto CPU de trace + engine GPU simultâneos → `0x8007000E`; Q8_0 funciona sozinho (uma cópia do engine).
4. **Git:** nunca commit direto em `main`; trabalhar em `research/dflash-xbox-rpc`; push para remote `private` só com validação 100%.
5. **Protocolo:** nunca aumentar structs de resposta silenciosamente; rejeitar entradas inválidas antes de mutar estado.

## 11. Estado atual do repositório (28/09, 21h20 — fim das sessões AGY noite)

- Últimos commits: `8995840` (docs/logs de validação), `3560da1` (Q2 n7 + pipeline corrigido), `0e23450` (fix numérico Xbox + profile), `8b3faa7` (Q2 GPU + RPC fundido), `e579e2a` (shaders Q6_K/Q8_0), `1266126` (20,5 GB/s / 25,75 TPS).
- **Alterações não commitadas** na worktree (experimento FFN da sessão AGY, preservado sem commit por política — investigação sem ganho de velocidade): `q2k_q8k_matmul.hlsl`, `q3k_q8k_matmul.hlsl`, DXIL gerados, `Package.appxmanifest` (versão), `dflash-gpu-draft-engine.cpp`, `dflash-gpu-draft-probe.cpp` + diretório `scratch/` untracked.
- Cliente host `.57`: BeeLlama commit `7720d9f34` (baseline) e correção de pipeline `12231b55e` na branch `fix/dflash-xbox-pipeline` (push, sem merge em main); lib `libllama-common.so.0.4.7` SHA256 `b46f159f...` no build final.
- Xbox final: **app 1.0.0.77**, backend `d3d12`, modelo `dflash-q2.gguf`, porta RPC 50053 OK; produção `.57` (`llama-server-root.service`, 8090) restaurada e saudável (`{"status":"ok"}` verificado às 20h49).

## 12. Referências (fontes primárias na worktree)

| Arquivo                                                                                 | Conteúdo                                                                                                                                                         |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `research/dflash2-xbox/dflash-gpu-implementation-plan-2026-09-28.md`                    | Plano mestre F0–F7, arquitetura, gates, riscos                                                                                                                   |
| `research/dflash2-xbox/gpu-implementation/f0-f2-results-2026-09-28.md`                  | F0–F2: captura, device, kernels HLSL vs DirectML                                                                                                                 |
| `research/dflash2-xbox/gpu-implementation/f3-f4-results-2026-09-28.md`                  | F3–F4: KV/SYNC, divergência camada 1, root cause, F5                                                                                                             |
| `research/dflash2-xbox/gpu-implementation/f6-real-production-campaign-2026-09-28.md`    | Campanha real 21,26 tok/s                                                                                                                                        |
| `research/dflash2-xbox/gpu-implementation/f7-kernel-optimization-results-2026-09-28.md` | Kernels Load4 → 25,75 tok/s                                                                                                                                      |
| `research/dflash2-xbox/gpu-implementation/f7-nmax-sweep-2026-09-28.md`                  | Varredura n_max                                                                                                                                                  |
| `research/dflash2-xbox/gpu-implementation/f7-verdict-2026-09-28.md`                     | Veredito de viabilidade (banda × aceitação)                                                                                                                      |
| `research/dflash2-xbox/gpu-implementation/status-2026-09-28.md`                         | Paridade numérica + Q2 n7 + comparativo pacotes .66/.70/.75 (estado mais detalhado)                                                                              |
| `research/dflash2-xbox/gpu-implementation/plan-acceptance-pipelining-2026-09-28.md`     | Melhorias 1–3 (aceitação, fusão, pipelining)                                                                                                                     |
| `research/dflash2-xbox/redeploy-1.0.0.76.jsonl` / `redeploy-1.0.0.77.jsonl`             | Evidência de deploy dos pacotes da sessão AGY FFN                                                                                                                |
| Sessão AGY `4c44ecef-58b1-417f-ae20-bc2b95be9565` (transcript local)                    | Otimização FFN 1.0.0.77, diagnóstico de gargalo, relatório final 20h49                                                                                           |
| Sessão AGY `ff7b6fe1-4c97-4943-8520-b4bf6dddd1d2` (transcript local)                    | Reavaliação Joint GEMV (decisão de não implementar), verificação saúde .57/Xbox 21h18                                                                            |
| `research/dflash2-xbox/CONTINUATION.md`                                                 | Guia mestre de continuação (operacional, passo a passo)                                                                                                          |
| `research/dflash2-xbox/current-status-2026-09-27.md`                                    | Histórico 1.0.0.2–1.0.0.7 (Game classification, memória)                                                                                                         |
| `research/dflash2-xbox/consolidation-2026-09-27.md`                                     | Consolidação 27/09 (transporte, arquitetura)                                                                                                                     |
| `uwp/dflash-rpc/`                                                                       | Servidor dedicado (`dedicated.cpp`), engines GPU (`dflash-gpu-engine.cpp`, `dflash-gpu-draft-engine.cpp`)                                                        |
| `shaders/dflash/`                                                                       | HLSL: `q3k_q8k_matmul`, `q2k_q8k_matmul`, `q6k_q8k_matmul`, `q8_0_q8k_matmul`, `dflash_draft_quant_q8`, `dflash_attention`, `dynamic_conv`, `hadamard_transform` |

**Fontes externas:** [DFlash2 publicado (incoai)](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF), [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), [grafo DFlash2 llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp), [upstream PR DFlash2](https://github.com/ggml-org/llama.cpp/pull/27342).
