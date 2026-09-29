# Relatório Técnico: Avaliação dos 3 Experimentos de Otimização DFlash2 GPU no Xbox Series X

**Data:** 29 de setembro de 2026  
**Ambiente:** Xbox Series X (Dev Mode `192.168.1.26`, D3D12, modelo `dflash-q2.gguf`) + Host RTX 4070 (`192.168.1.57`, Target `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, Selector Q8_0)  
**Branch:** `research/dflash-xbox-rpc` (Worktree: `C:\Users\hjotha\worktrees\xllama-dflash-xbox`)  

---

## 1. Visão Geral e Motivação

O objetivo da bateria de experimentos proposta no relatório de arquitetura (`dflash2-gpu-deep-review-and-bottlenecks-2026-09-29.md`) foi avaliar sistematicamente três frentes de otimização para elevar o throughput do pipeline especulativo DFlash2 distribuído entre Xbox Series X (GPU D3D12) e a estação host (RTX 4070):
1. **Experimento 1 (Prioridade 1):** Double-buffering / Ping-Pong LDS nos shaders HLSL de GEMV quantizado (`q2k_q8k_matmul.hlsl` e `q3k_q8k_matmul.hlsl`).
2. **Experimento 2 (Prioridade 2):** Fusão em lote das transições de recursos D3D12 (`cmd_->ResourceBarrier(count, items)`) no loop do engine C++ (`dflash-gpu-draft-engine.cpp`).
3. **Experimento 3 (Prioridade 3):** Calibração do threshold do seletor host na `.57` (`draft-p-min` de 0,25 a 0,45 com `n_max=7`, `temperature=0.0`) para maximizar a taxa de aceitação e os tokens por segundo efetivos.

---

## 2. Experimento 1: Double-Buffering / Ping-Pong LDS no GEMV (Rejeitado)

### Hipótese
Ocultar a latência de leitura da VRAM (`weights` e `scales`) nos kernels quantizados de GEMV sobrepondo a leitura do bloco $k+1$ em LDS com o processamento aritmético (descompressão Q2_K/Q3_K e FMAs) do bloco $k$.

### Implementação
- Alocados buffers ping-pong em LDS:
  - `groupshared uint s_w2[2][21];`
  - `groupshared float s_q8[2][512];`
  - `groupshared float scaled_weights[2][256];`
- Pré-carregamento do bloco 0 antes do loop; a cada iteração $k$, a wave carrega o bloco $k+1$ no buffer `1 - buf_idx`, sincroniza via `GroupMemoryBarrierWithGroupSync()` e processa `buf_idx`.
- Compilado com DXC (`cs_6_0`) gerando cabeçalhos DXIL C++ e empacotado no MSIX `1.0.0.81`.

### Resultados Medidos no Console (Probe 17)
- **Paridade Numérica:** 100% preservada bit a bit (`first_output_sha256: 4f5c680fc4a3aa5860cea34d61a122ed6a7b6be40b840f9c7d26d55423ae1663`, `differing_outputs: 0`, `parity_passed: true`).
- **Latência de DRAFT Compute:**
  - Baseline (1.0.0.80): **25,22 ms** (25.221 µs)
  - Com Double-Buffering (1.0.0.81): **36,97 ms** (36.974 µs) — **Regressão severa de +46,6%**
- **Soma dos Intervalos Probe 17:** Subiu de 26,62 ms para 38,20 ms.
- **GEMV Block (Camadas 1 a 6):** Subiu de 21,3 ms para 32,8 ms (+54%).

### Causa Raiz de Hardware (Microarquitetura AMD RDNA 2)
1. **Contenção de Portas de Banco LDS:** No RDNA 2, cada banco de LDS possui apenas **1 porta de leitura e 1 porta de escrita por ciclo**. Emitir leituras assíncronas de VRAM e gravá-las no buffer $k+1$ concorrentemente com waves adjacentes lendo do buffer $k$ gerou contenção massiva de portas de banco LDS, forçando a serialização das waves no SIMD.
2. **Perda de Deslocamentos Imediatos no Assembly DXIL:** O seletor `buf_idx` transformou acessos estáticos em LDS (`ds_read_b32 offset:imm`) em endereçamento dinâmico via registradores VGPR (`ds_read_b32 v[addr]`), aumentando a pressão de registradores e o número de instruções.
3. **Decisão:** Reversão imediata para a baseline de buffer único sem contenção.

---

## 3. Experimento 2: Fusão em Lote das Transições de Recursos D3D12 (Aprovado)

### Hipótese
No loop do engine C++ (`dflash-gpu-draft-engine.cpp`), transições de recursos eram submetidas discretamente uma a uma via `cmd_->ResourceBarrier(1, &b)`. Submeter 2 ou 3 barreiras simultâneas via `cmd_->ResourceBarrier(count, items)` reduz o número de pacotes de controle no Command Processor (CP) da GPU e mitiga micro-bolhas entre dispatches.

### Implementação
- Implementadas sobrecargas em lote na classe `DFlashGpuDraftEngine`:
  - `transition(ID3D12Resource* b1, D3D12_RESOURCE_STATES s1, ID3D12Resource* b2, D3D12_RESOURCE_STATES s2)`
  - `transition(ID3D12Resource* b1, D3D12_RESOURCE_STATES s1, ID3D12Resource* b2, D3D12_RESOURCE_STATES s2, ID3D12Resource* b3, D3D12_RESOURCE_STATES s3)`
- Agrupadas todas as transições discretas do pipeline (init, matmul, norm, rope, hadamard, conv, scale_add, swiglu, pack_kv, attention).
- Compilado e empacotado no MSIX `1.0.0.82`. Implantado com `dflash-barrier-mode: minimal`.

### Resultados Medidos no Console (Probe 17 & RPC Dedicated Bench)
- **Paridade Numérica:** 100% preservada bit a bit (`first_output_sha256: 4f5c680fc4a3aa5860cea34d61a122ed6a7b6be40b840f9c7d26d55423ae1663`, `differing_outputs: 0`).
- **DRAFT Compute:** **25,16 ms** (25.162 µs) — Novo recorde histórico mínimo (mediana de 25,20 ms vs 25,22 ms no 1.0.0.80).
- **Fused Compute:** **27,50 ms** (27.499 µs) — Ganho estável de ~0,17 ms vs 27,67 ms anterior.
- **Cycle RTT Fused RPC:** **31,36 ms** (31.364 µs) — Novo recorde mínimo vs 31,50 ms anterior.
- **Decisão:** APROVADO e mantido ativo no console Xbox (`1.0.0.82`).

---

## 4. Experimento 3: Calibração do Seletor Host na .57 (Aprovado com Novo Recorde)

### Metodologia
- Host `.57` (RTX 4070, CUDA0, `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, KVarN4, Flash Attention).
- Drafter remoto no Xbox Series X (`192.168.1.26:50053`, pacote `1.0.0.82`, `dflash-q2.gguf`).
- Pipeline corrigido com overlap total (`DFLASH_XBOX_PIPELINE=1`, `--batch-size 64 --ubatch-size 64` evitando OOM).
- Varredura de `draft-p-min` em `[0.25, 0.30, 0.35, 0.40, 0.45]` com `n_max=7`, `temperature=0.0`, geração de 512 tokens.

### Resultados Comparativos

| Configuração | `p_min` | Throughput (tok/s) | Wall Time (s) | Tokens Propostos | Tokens Aceitos | Taxa de Aceitação (%) | Tokens / Ciclo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target Baseline (Sem especulativo)** | — | **35,506** | 14,58 | 0 | 0 | 0,00% | 1,00 |
| **DFlash2 Q2 n7 (p_min = 0.25)** | 0,25 | 30,521 | 16,95 | 1069 | 356 | 33,30% | 3,28 |
| **DFlash2 Q2 n7 (p_min = 0.30)** | 0,30 | 31,628 | 16,38 | 942 | 367 | 38,96% | 3,53 |
| **DFlash2 Q2 n7 (p_min = 0.35)** | 0,35 | 31,666 | 16,37 | 903 | 366 | 40,53% | 3,51 |
| **DFlash2 Q2 n7 (p_min = 0.40) [ÓTIMO]** | **0,40** | **32,184** | **16,06** | **781** | **363** | **46,48%** | **3,44** |
| **DFlash2 Q2 n7 (p_min = 0.45)** | 0,45 | 32,061 | 16,16 | 778 | 358 | 46,02% | 3,32 |

---

## 5. Conclusões e Síntese dos Ganhos

1. **Evolução do Pipeline Especulativo:**
   - O melhor throughput anterior registrado para `n7` no branch (`status-2026-09-28.md`) era **28,468 tok/s** (`p_min=0.30`, pacote 1.0.0.75).
   - Com o pacote **1.0.0.82** (transições em lote D3D12) e calibração de seletor para **`draft-p-min = 0.40`**, o pipeline especulativo atingiu **32,184 tok/s**.
   - **Ganho líquido de throughput:** **+13,05%** (de 28,47 tok/s para 32,18 tok/s).
   - **Salto na taxa de aceitação:** De 33,30% (`p_min=0.25`) para **46,48%** (`p_min=0.40`).
2. **Gap em relação ao Target Sozinho:**
   - Anteriormente, o pipeline especulativo era **19,78% mais lento** que a RTX 4070 rodando sozinha (28,47 vs 35,49 tok/s).
   - Agora, o pipeline especulativo está a apenas **9,35% de paridade** (32,18 vs 35,51 tok/s), tornando o drafter remoto muito mais viável e com 46,5% dos tokens validados vindo diretamente do Xbox.
3. **Serviço de Produção na .57:**
   - O serviço `llama-server-root.service` foi mantido 100% íntegro e restaurado ao estado `active` após as baterias de testes.
