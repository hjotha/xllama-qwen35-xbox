# Plan 002: Steer e Correções para o OpenCode — MTP Rev 32 no Xbox

Data: 2026-10-04. Prioridade: P1. Executor: OpenCode (optiplex/.157, tmux 0).
Branch: `spike/beellama-pin` em `/home/hjotha/worktrees/xllama-mtp`.

---

## 1. O que realmente aconteceu no travamento da Rev 31 (Diagnóstico Forense)

A conclusão de que o `build-uwp.ps1` "parou de gerar o AppPackages" ou que houve falta de espaço em disco na `.193` é **INCORRETA**:

1. **Espaço em disco na `.193`:** O disco C: possui **32.86 GB livres**. Espaço não é problema.
2. **A causa da ausência da Rev 31:**
   - O `relaunch27.ps1` foi executado às 23:00:33 tentando iniciar o build da Rev 31.
   - A tarefa agendada `XllamaMtpBuild` no Windows está configurada com `MultipleInstances: IgnoreNew`.
   - Às 23:00:33, a compilação da **Revisão 30 ainda estava em execução** (iniciou às 22:36:13 e só concluiu às 23:02:50).
   - O Task Scheduler do Windows **ignorou silenciosamente** a chamada `.Run($null)` para a Rev 31 porque a tarefa anterior ainda estava ativa.
   - Portanto, a compilação da Revisão 31 **NUNCA OCORREU**.
   - O loop de espera no bash deu timeout procurando por `xllama_1.6.0.31_x64_Test` porque nada foi construído.

---

## 2. Avaliação das Entregas do Steer Anterior (Codex Plan 001)

| Ponto do Steer Codex                             | Status na Rev 30 / 31                 | Diagnóstico                                                                                  |
| ------------------------------------------------ | ------------------------------------- | -------------------------------------------------------------------------------------------- |
| **F1 Double Trim no Reject**                     | **ATENDIDO** (`7635525`)              | `tail_already_trimmed` suprime o 2º trim após rejeição.                                      |
| **F2 `n_rs_seq` no Target Context**              | **ATENDIDO** (`7635525`)              | Derivado da profundidade MTP; fim do abort no depth 4.                                       |
| **F3 KV Tail Cleanup no Draft**                  | **ATENDIDO** (`7635525`)              | `llama_memory_seq_rm` restaurado em `mtp_draft.cpp:205`.                                     |
| **F4 CPU Threadpools no Draft**                  | **ATENDIDO** (`7635525`)              | Compartilha o `GgufCpuThreadpools` com o target context.                                     |
| **F5 Batch Combinado `[anchor, drafts]`**        | **IMPLEMENTADO COM BUGS** (`eb610e6`) | Estrutura correta, mas introduziu **off-by-one no carry** e **omissão do `first_token_ms`**. |
| **F6 Eliminar Overhead de `top_prob` / Softmax** | **NÃO ATENDIDO**                      | `top_prob` continua calculando 152.064 exponenciais escalares em CPU por draft candidate!    |

---

## 3. Correções Obrigatórias de Código para a Revisão 32

### Correção A: Off-by-one Crítico no Carry State (`last_decoded_pos()`)

- **Arquivo:** `src/bridge/decode_loop.h` (linha ~295)
- **Problema:**
  ```cpp
  auto last_decoded_pos = [&]() -> llama_pos {
      return llama_memory_seq_pos_max(mm_carry, 0) - 1; // ERRADO!
  };
  ```
  No modo unmasked (`cparams.embeddings_nextn_masked == false`), `llama_get_embeddings_nextn_ith(p.ctx, i)` indexa a posição do token no KV cache.
  `llama_memory_seq_pos_max(mem, 0)` retorna a **maior posição já decodificada** na sequência (ex: 49).
  Como o anchor (`token`) **ainda não foi decodificado**, a última posição decodificada é exatamente `seq_pos_max` (49).
  Ao subtrair 1 (`seq_pos_max - 1`), o código pegava a posição **48** (o penúltimo token), condicionando o MTP no estado oculto errado!
- **Correção:**
  ```cpp
  auto last_decoded_pos = [&]() -> llama_pos {
      return llama_memory_seq_pos_max(mm_carry, 0);
  };
  ```

### Correção B: Eliminar o Gargalo de CPU no `top_prob`

- **Arquivo:** `src/bridge/mtp_draft.cpp` (linha ~279)
- **Problema:** O vocabulário do Qwen 3.5 possui **152.064 tokens**. Executar `std::exp(double)` 152.064 vezes em CPU a cada depth de draft consome vários milissegundos por token. Além disso, mesmo quando `m_params.p_min <= 0.0f` (filtragem desativada), `top_prob` estava sendo chamado cegamente!
- **Correção:**
  Executar `top_prob` **apenas** quando `m_params.p_min > 0.0f`:
  ```cpp
  if (m_params.p_min > 0.0f) {
      const float p = top_prob(m_ctx, i_last);
      if (p < m_params.p_min)
          break;
  }
  ```

### Correção C: Registrar `first_token_ms` no `lead_in_batch`

- **Arquivo:** `src/bridge/decode_loop.h` (linha ~473)
- **Problema:** Quando `lead_in_batch == true`, `out.first_token_ms` não é medido se o primeiro token emitido for o anchor via MTP.
- **Correção:** Ao emitir o primeiro token no `commit_draft` (ou antes de `feed[0]`), se `out.first_token_ms == 0.0`, registrar o tempo a partir de `p.decode_start`.

---

## 4. Procedimento de Build e Validação da Revisão 32

O OpenCode deve executar rigorosamente este roteiro:

1. **Aplicar as correções A, B e C** nos arquivos locais.
2. **Validar compilação no host:**
   ```bash
   clang-format -i src/bridge/decode_loop.h src/bridge/mtp_draft.cpp
   cmake --build build/linux-test -j$(nproc)
   ctest --test-dir build/linux-test
   ```
3. **Commit e Push:**
   ```bash
   git commit -am "fix(mtp): correct carry position off-by-one and bypass top_prob when p_min <= 0"
   git push origin spike/beellama-pin
   ```
4. **Disparo Seguro na `.193` (BuildRevision 32):**
   - Garantir que não há build anterior rodando:
     ```powershell
     $svc = New-Object -ComObject 'Schedule.Service'
     $svc.Connect()
     $t = $svc.GetFolder('\').GetTask('XllamaMtpBuild')
     if ($t.State -eq 4) {
         $t.Stop(0)
         Start-Sleep -Seconds 5
     }
     $reg = $t.Definition
     $reg.Actions.Item(1).Arguments = '-File "C:\Users\hjotha\build\xllama-q5k\Build-Local.ps1" -BuildRevision 32'
     $svc.GetFolder('\').RegisterTaskDefinition('XllamaMtpBuild', $reg, 4, $null, $null, 3, $null)
     ```
   - No repositório `C:\Users\hjotha\projects\xllama-qbox`:
     - Fazer `git pull` da branch `spike/beellama-pin`.
     - Executar `./scripts/apply-uwp-patches.sh`.
   - Limpar `C:\Users\hjotha\build\xllama-q5k\exit.txt`.
   - Iniciar a tarefa: `$t.Run($null)`.
   - Aguardar a conclusão monitorando `build-local.log` até `=== DONE ===` e `exit.txt == 0`.
5. **Deploy no Xbox Series S/X:**
   ```bash
   set -a
   source ~/.config/xllama/xbox-env
   set +a
   ./scripts/deploy.sh /tmp/opencode/xllama-1632.msix
   ```
6. **Benchmark Obrigatório de 4 Runs:**
   ```bash
   # 1. Baseline sem MTP
   ./scripts/bench-xbox-ort.sh qwen35-4b-mtp --runs 4 --n-predict 64 --gpu-layers 99 --ctx 2048 --ubatch 64 --ignore-eog --out /tmp/opencode/bench-baseline-r32.csv

   # 2. MTP Depth 2
   ./scripts/bench-xbox-ort.sh qwen35-4b-mtp --runs 4 --n-predict 64 --gpu-layers 99 --ctx 2048 --ubatch 64 --ignore-eog --mtp 2 --out /tmp/opencode/bench-mtp2-r32.csv

   # 3. MTP Depth 4
   ./scripts/bench-xbox-ort.sh qwen35-4b-mtp --runs 4 --n-predict 64 --gpu-layers 99 --ctx 2048 --ubatch 64 --ignore-eog --mtp 4 --out /tmp/opencode/bench-mtp4-r32.csv
   ```
7. **Critério de Sucesso:**
   - Taxa de aceitação elevada.
   - Paridade de texto com baseline.
   - Throughput (tok/s) no hardware Xbox superando o baseline de ~18.4 tok/s.
