# T18 — teste interno de falha mid-blob em `save_state` (plano 009 · demonstração de isolamento PRÉ-EXECUÇÃO + desenho)

Data: 2026-10-12. Autorização do usuário: concluir a lacuna de falha durante
a escrita com o menor teste reversível e isolado, **demonstrando antes de
executar** que só um slot/request descartáveis de teste podem receber a
falha. Nada aqui toca dispositivo, `.57`, snapshots válidos, tolerâncias
ou push/merge.

## 1. Veículo de injeção — por que NÃO existe knob global

- Verificado no código atual: `grep -rn "slot-fault" src/ include/` →
  **0 ocorrências**. Nenhum knob de arquivo, variável de ambiente ou hook
  existe em produção; o desenho anterior de knob (`slot-fault.target.txt`)
  **não foi implementado** e foi descartado justamente por poder capturar
  request alheio.
- O veículo é `setrlimit(RLIMIT_FSIZE, L)` — atributo **por processo**
  do kernel Linux, aplicado só ao processo do teste e restaurado por
  RAII. Não há arquivo, variável global, caminho compartilhado ou
  endpoint ao qual um request alheio possa "colar-se": rlimit não é
  observável nem ajustável por outro processo (exceto via ptrace, que
  não existe aqui). Nenhum lado de produção é alterado — a mudança é
  somente `tests/test_session.cpp` (arquivo já registrado em
  `tests/CMakeLists.txt:17`).
- **Identificação interna do request pela camada UWP não é
  indispensável**: o teste não passa pela camada de API — ele chama
  `Session::save_state` diretamente na mesma thread do processo de
  teste. O "request de teste" é identificável por construção: dentro da
  janela armada existe exatamente **uma thread** e ela só executa
  `save_state` do próprio teste; o binário não tem servidor HTTP, não
  faz rede e não faz fork. Nenhuma mudança em `uwp/` ou em
  `include/xllama/session.h` é necessária (API pública intocada).

## 2. Slot e request descartáveis exclusivos

- Diretório: `temp_directory_path() / "xllama-t18-midwrite-<pid>-<ns>"`,
  criado pelo teste com nome único (pid + timestamp) — nunca é
  `LocalState/slots`, nunca é o `xllama-kv-state.bin` fixo do teste
  round-trip existente (sem colisão mesmo com reordenação de casos).
- Arquivos: apenas `ref.kv` (snapshot de referência criado pelo próprio
  teste) e `fault.kv` (alvo descartável). O teste não conhece nem abre
  nenhum outro caminho.
- Dispositivo: o teste roda no host Linux; **zero chamadas de rede**
  (nenhum curl/API/deploy no caso de teste). rev244 no Xbox permanece
  intocado por construção — será reconferido ao final.

## 3. Proteção de snapshots válidos

- O único snapshot alvo de um save com falha é o `ref.kv` do próprio
  teste. Seus bytes são carregados na memória **antes** da janela
  armada e comparados por igualdade byte a byte após **cada** tentativa
  falha (`CHECK(read(ref) == ref_bytes)`).
- Mecânica de proteção é a do código de produção: escrita vai só no
  tmp (`session.cpp:1588`), o caminho de falha remove o tmp
  (`:1608-11`) e `rename` (`:1614`) — a única mutação do arquivo final
  — nunca acontece quando o save falha.
- Nenhum snapshot de baseline/dispositivo é aberto, listado ou escrito.

## 4. Limite de bytes e de espaço

- Referência: um save completo em `ref.kv` antes da janela → `S` =
  `file_size(ref)`. O header (`"xllama-kv\t<n_state>\t<n_tokens>\t…"`,
  `session.cpp:1432`, `:1597`) fornece `n_state` (tamanho do blob).
- Limite de bytes: `pre_bound = 512 + 8·n_ctx` (header ≤512 B + ids de
  token sobre-limitados) e **`L = pre_bound + n_state/2`**, com as
  travas `n_state > 4·pre_bound` e `L < S`. Consequência: `pre < L <
pre + n_state` — a falha cai **estritamente dentro da região do
  blob**, nunca no header/ids (não é falha de abertura nem de header).
  `L` é registrado no log do teste como evidência do limite.
- Limite de espaço: durante a janela o kernel **impede** qualquer
  escrita além de `L` (o tmp não pode crescer além disso); após cada
  falha o caminho de produção remove o tmp ⇒ dique líquido adicional = 0. Nenhum preenchimento de disco é usado (nada de ENOSPC por
  capacidade). Pós-janela o diretório deve conter exatamente
  `{ref.kv}` — asserção explícita.

## 5. Limpeza verificável e reversibilidade

- RAII na janela: `~Arm()` restaura `RLIMIT_FSIZE` original (guardado
  com `getrlimit`) e a disposição anterior de `SIGXFSZ` (guardada com
  `sigaction`); `SIGXFSZ` é ignorado só dentro da janela, senão o
  kernel mataria o processo de teste — esse comportamento é exatamente
  o que o teste NÃO exercita (morte de processo, §7).
- RAII do caso: `~Cleanup()` executa `remove_all(dir)` em qualquer
  saída, inclusive exceção de `REQUIRE`; o caso também assere
  explicitamente `!exists(dir)` ao final.
- Verificação externa pós-execução: `ls /tmp/xllama-t18-midwrite-*`
  deve voltar vazio.

## 6. Desenho do teste (caso novo em `tests/test_session.cpp`)

Guardas: `#ifdef __linux__` (setrlimit é POSIX; em outras plataformas o
caso retorna), opt-in `XLLAMA_TEST_MODEL` (mesmo padrão dos casos
`#170b/#169` já existentes no arquivo).

1. Diretório único + `Cleanup` RAII (remove dir em qualquer saída).
2. `Session::create(sp)` (`n_ctx=2048`) + `generate` curto em ASCII
   puro → resident KV (`save_state` recusa sem `m_kv_tokens`).
3. Save de referência → `ref.kv`; `S = file_size`; parse do header →
   `n_state`; travas `n_state > 4·pre_bound`, `L = pre_bound +
n_state/2`, `L < S`; bytes do ref guardados em memória.
4. **Janela armada** (RAII `Arm`):
   a. `save_state(fault.kv)` ⇒ `false`, `err == "write failed (disk
full?)"` (mensagem exata de `session.cpp:1610`), `fault.kv`
   não existe, `fault.kv.tmp` não existe.
   b. `save_state(ref.kv)` sobre o snapshot válido ⇒ `false`, mesma
   mensagem, `ref.kv` existe e é **byte-idêntico** ao original,
   `ref.kv.tmp` não existe.
5. Fim da janela (restauração RAII). Asserções de espaço: dir contém
   exatamente `{ref.kv}`.
6. Save subsequente **sem** o limite ⇒ `true`; `file_size == S` e
   conteúdo byte-idêntico ao da referência (mesmo estado re-gravado).
7. Integridade: nova sessão `load_state(ref.kv)` ⇒ `true` +
   `generate` de continuação `success` (o arquivo é utilizável).
8. Limpeza: `remove_all(dir)`; `!exists(dir)`.

Falha esperada do caminho de produção (sem variação): `fwrite` parcial
(`io_chunked`, `session.cpp:1464-75` retorna false quando `got != want`)
⇒ `ok=false` ⇒ `fclose` (`:1605`) ⇒ `remove(tmp)` (`:1608-11`) ⇒
`*err = "write failed (disk full?)"` ⇒ `return false`. `rename` nunca
é atingido.

## 7. Distinção obrigatória (não declarar ambas provadas)

- **Provado com este teste**: falha de I/O **simulada** no meio do blob
  — o caminho de falha real do `save_state` completa (tmp removido,
  `false` retornado, snapshot prévio byte-idêntico, save posterior
  íntegro).
- **NÃO provado — morte de processo**: o teste executa o caminho de
  falha até o fim; um SIGKILL no meio da escrita **não** rodaria
  `remove(tmp)` e deixaria `.tmp` parcial. Essa variante segue
  coberta apenas estruturalmente: o loader abre só o caminho final
  (`load_state`), `.tmp` é inerte (B11) e `KvStore::prune`
  (`kv_store.cpp:78-98`) varre tmp órfãos.
- **NÃO provado — request da API Xbox**: o teste host não exercita o
  handler HTTP (`uwp/`), o mapeamento para 400, NTFS nem a
  concorrência do servidor LAN; isso continua coberto apenas pela
  evidência de dispositivo já existente (T16/F1/F2: 400, snapshot
  preservado, `.tmp` limpo). Host ≠ dispositivo.

## 8. Preservação e plano de execução

- Mudança somente em `tests/test_session.cpp` ⇒ build **apenas**
  `linux-test` (host). Nenhum binário de dispositivo é alterado:
  **rev244 Final**, baselines, snapshots e rollback (rev243/rev238)
  preservados por construção; nenhum gate de dispositivo é reexecutado
  porque o produto não muda.
- Fixture de teste: `stories15M-q4_0.gguf` (19.077.344 B, GGUF v3,
  verificado por magic) baixado de
  `huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/…`
  para `/tmp/opencode/models/` (fora do repositório).
- Evidências a capturar: saída do caso (MESSAGEs com `L`, `S`,
  `n_state`), `ctest` completo (regressão), `/tmp` vazio após,
  `git status` de `src/`+`include/` = 0, PFN/versão do Xbox inalterados.
- Sem push, sem merge, sem kills por tentativa e erro, sem `.57`, sem
  tolerâncias alteradas.

## 9. Observabilidade cooperativa — plano (execução focal, autorização 1414)

**Motivo**: `L=181254` sozinha prova apenas o _limite_, não que bytes
foram _efetivamente persistidos_ no `.tmp` antes de `save_state`
removê-lo. Duas rotas foram recusadas pelo usuário e ficam registradas
(sem tentativas alternativas): (i) `rm -rf /tmp/xllama-t18-midwrite-*`
(cleanup dos 3 órfãos SIGILL — **preservados como evidência**);
(ii) `strace`/leitura de `/proc/sys/kernel/yama/ptrace_scope` (sem
ptrace por qualquer rota, sem alterar yama/permissões).

**Mecanismo (somente teste, sem nova privilégio, sem escrita falsa)**:
um **filho observador** criado pelo próprio caso de teste, _antes_ de
armar o rlimit:

1. `pipe` de prontidão + `pipe` de relatório; `fork`;
2. o filho faz `inotify_init`/`inotify_add_watch` no diretório exclusivo
   (`IN_CREATE|IN_MODIFY|IN_DELETE`), sinaliza prontidão e entra em
   `poll` com _watchdog_ de 10 s (nunca trava a suíte);
3. ao ver `IN_CREATE` de `fault.kv.tmp`/`ref.kv.tmp`, o filho **abre o
   arquivo pelo nome e mantém o fd** — depois do `unlink` de
   `save_state` o inode continua vivo via fd, então `fstat(fd)`
   reporta o tamanho final persistido de forma determinística;
   fallback `stat` pelo nome; tamanhos máximos são rastreados a cada
   evento; o `unlink` é registrado via `IN_DELETE`;
4. relatório por pipe (`max`/`open`/`del` por tmp + inesperados), o
   filho ` _exit` (nunca desenrola doctest);
5. o pai, após a janela armada, lê o relatório, `waitpid` e assere:
   `0 < max ≤ L` para **cada** tmp, `del=1` para cada, `unexpected=0`.

O salvamento real é feito integralmente por `save_state` (rotina de
produção intacta — nenhuma escrita é substituída ou simulada pelo
observador); o filho só **observa**. Ele é _forked antes do armamento_,
herda os limites originais, e `RLIMIT_FSIZE`/`SIGXFSZ` continuam
restritos ao processo do teste.

**Isolamento (a documentar no recibo)**: `setrlimit` é **por processo**
(semântica POSIX, `man setrlimit`) e a disposição de `SIGXFSZ` é
**por processo** (`man signal`); só o processo `xllama-tests` é alterado
e o `Arm` os restaura por RAII. O processo host do caso não tem
`save` concorrente (doctest sequencial, uma thread executa
`save_state`; as threads ggml não escrevem nos caminhos de slot e o
`generate` termina antes do armamento) e **nenhum request Xbox** (o caso
não tem código de rede; fixture local; dispositivo nunca é contactado).
Nenhuma outra execução paralela é afetada: limites não são compartilhados
entre processos, não existe knob global (`grep slot-fault` = 0 em
`src/`+`include/`), e os únicos artefatos compartilhados são a fixture
(somente leitura) e o diretório exclusivo (pid+timestamp).

**Limites de bytes/espaço (recalculados)**: `L` por processo durante a
janela; o observador não cria arquivos (só `pipe`s); dique adicional de
disco = 0 (ambos os `.tmp` removidos — agora _observados_ via
`IN_DELETE`, não assumidos); pós-janela o diretório contém exatamente
`{ref.kv}`; limpeza RAII do diretório ao final.

**Se não observável**: se a corrida `IN_CREATE`→`open` perder (relatório
`open=0`/`max=0`), isso será registrado como lacuna exata (corrida
inode entre `fclose`+`remove` e o `open` do observador), sem rotas de
novo privilégio.

### 9.1 Revisão pós-execução — corrida `IN_CREATE` perdida

A primeira implementação abria o `.tmp` em resposta ao `IN_CREATE`.
Run 1 observou `fault.kv.tmp max=181254 open=1 del=1` e
`ref.kv.tmp max=181254 open=1 del=1` (`unexpected=0`), mas **run 2
perdeu a corrida** (`open=0 max=0 del=1` nos dois — o pai concluiu
`fclose`+`remove` antes de o filho ser agendado). Mecanismo revisado:
**giro de vantagem** — o filho tenta `open()` dos nomes já conhecidos
(`fault.kv.tmp`/`ref.kv.tmp`) continuamente a partir do sinal de
prontidão, **antes de o pai armar**; o primeiro `open` bem-sucedido
fixa o inode e o fd sobrevive ao `unlink`, tornando a captura de
tamanho determinística (margem: latência de `syscall` do filho
~sub-µs vs janela `open`→`unlink` do pai ~centenas de µs de escritas
em buffer). inotify continua responsável por `IN_DELETE`, tamanhos
com o nome ainda existente, nomes inesperados e `IN_Q_OVERFLOW`;
prazo de 10 s impede travamento. Ambas as runs ficam registradas no
`T18-host-midwrite-receipt.txt`.

### 9.2 Revisão final — pré-criação do `.tmp` (determinístico por construção)

O giro de vantagem também perdeu (run 3 de 3: `open=0 max=0 del=1`) —
mesmo partindo antes do pai, o filho pode ficar fora da CPU durante a
janela de ~centenas de µs. Verificado no código de produção:
`save_state` faz `open_state_file(tmp, "wb")` (`src/bridge/session.cpp:1589`)
**sem `remove` prévio e sem `O_EXCL`**, e o caminho de falha só
`remove(tmp)` o _nome_ (`:1609`). Mecanismo final:

1. o filho **pré-cria** `fault.kv.tmp` e `ref.kv.tmp` (vazios,
   `O_CREAT|O_RDWR`) e guarda os fds **antes** de assinalar prontidão;
2. o handshake do pipe fixa a ordem: filho cria → sinaliza → pai arma →
   `save_state` trunca **esses mesmos inodes** com `"wb"` e grava neles;
3. o caminho de falha desvincula só o nome; os fds mantêm o inode vivo,
   então `fstat(fd)` reporta o tamanho final a qualquer instante — inclusive
   processando `IN_DELETE` atrasado. **Sem corrida de escalonamento.**
4. inotify continua reportando `IN_DELETE`, `IN_Q_OVERFLOW` e nomes
   inesperados; prazo de 10 s.

Custo em espaço: arquivos pré-criados vazios (0 blocos) até o pai gravar;
nenhum arquivo novo além dos dois `.tmp` já previstos pelo teste.
