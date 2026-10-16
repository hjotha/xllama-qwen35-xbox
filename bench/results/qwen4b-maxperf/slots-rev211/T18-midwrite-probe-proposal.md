# Proposta — probe de falha após N bytes, vinculado a slot e request (T18 · para decisão; NÃO implementado)

Data: 2026-10-12 · Revisão 2 (correção da alegação de exclusividade de
request). Fecha a lacuna exa do subgate de atomicidade do #170c (falha de
I/O no meio da gravação do blob) de forma **determinística**, após
T16/T17/T17b/T18 terem esgotado os meios externos seguros: kills por
tempo caem após o rename; 60 polls do portal nunca viram o `.tmp`
parcial; encher disco é proibido. **Nada foi implementado nem executado;
esta etapa é só o desenho e a análise de risco.**

## 1. Evidência de código (estado atual)

- `save_state` (`src/bridge/session.cpp:1560`) recebe **apenas o path**
  (`const std::string& path`), não o id — o id vem do chamador via
  `KvStore::path_for` (`src/bridge/kv_store.cpp:63-67`,
  `dir/(id + ".kv")`). Qualquer amarração do disparo é, portanto, por
  **basename do path**, não por argumento. A interface é
  `virtual bool save_state(const std::string& path, std::string* err =
  nullptr)` (`include/xllama/session.h:200`); único override neste
  worktree: `session.cpp:1560`.
- Escrita: tmp = `path + ".tmp"` (`:1588`); header (`:1596-98`); tokens
  (`:1600-01`); **blob do target ~58 MB** via `io_chunked` em chunks de
  8 MB (`:1464`, `:1602`); trailer MTP (`:1603-04`); `fclose` (`:1605`).
- Falha de escrita ⇒ `remove(tmp)` + `return false` (`:1608-11`) — a API
  devolve 400. `rename(tmp, path)` é a **única** mutação do arquivo final
  (`:1614`). `load_state` abre só `path` (o tmp nunca é lido).
- **Sem mutex** em `session.cpp` nem em `kv_store.cpp`: a serialização de
  requests vive na camada LAN (`uwp/`), **não vendorizada** neste worktree
  — logo a proposta **não pode** depender dela (ver §3.3).
- `KvStore::valid_id` (`kv_store.cpp:49-61`) restringe o id a
  `[A-Za-z0-9_-]` (≤64); `KvStore::erase` (`:69-76`) apaga `<id>.kv` e
  `<id>.kv.tmp`; `KvStore::prune` (`:78-98`) remove todo `.tmp` órfão do
  diretório.
- Precedente de knob por arquivo, default off: `kv_reuse_diag.txt`
  (`:694`), `chat-dump.flag` (`:1512`) — `open_state_file(
  resolve_local_path("…"), "rb")`.
- Ids usados pelos gates/baselines deste plano (não podem ser alvo):
  `t2b-n1 t2b-n2 t6 t6c t6d t7 t9-x t10 t11 t12 t13a t13c t15 t16
  t16-nokv t17 t17ref t18 t19`.

## 2. Mudança exata proposta (NÃO APLICADA — awaits decisão)

Knob de nome fixo `slot-fault.target.txt`, cujo **conteúdo** carrega
`"<basename-alvo> <bytes>"`. O knob é **aberto para todo save `*.kv`**
(a comparação precisa do conteúdo — não existe checagem pelo nome do
arquivo, que é fixo); apenas saves cujo basename casa exatamente podem
prosseguir, e o disparo passa por uma **reivindicação atômica por rename**
(ver §3.3): no máximo um save por arm dispara. Os bytes são escritos no
tmp e o save aborta pelo caminho de falha real (`:1608-11`). ≈40 linhas,
todas em `session.cpp`.

```diff
+    // TEST-ONLY atomicity fault probe (plan 009 / T18; PROPOSED, not
+    // implemented). Armed by the LOCALSTATE file "slot-fault.target.txt"
+    // whose content is "<target-basename> <bytes>": the offset N at which
+    // to abort the save while it writes the target blob. The knob is
+    // FIXED-name, so it is opened (read-only) on every *.kv save — the
+    // slot binding needs its content and is checked AFTER the read; a
+    // non-matching save never claims nor consumes it. The fault can only
+    // fire when (a) the save path basename matches the armed basename
+    // exactly (and ends in .kv, never .tmp) AND (b) this save wins the
+    // atomic rename claim below. Default off: no file -> zero behaviour
+    // change. The abort reuses the existing "write failed" path (remove
+    // tmp + false) and never touches the previous <id>.kv (only the
+    // rename does that).
+    static size_t probe_fault_bytes_for(const std::string& path) {
+        const size_t slash = path.find_last_of("/\\");
+        const std::string target =
+            slash == std::string::npos ? path : path.substr(slash + 1);
+        // Disarm anything that is not "<target>.kv" (guards *.tmp etc.).
+        if (target.size() < 4 || target.substr(target.size() - 3) != ".kv")
+            return 0;
+        const std::string knob = resolve_local_path("slot-fault.target.txt");
+        FILE* k = open_state_file(knob, "rb");   // read-only, pre-match
+        if (!k)
+            return 0;
+        char b[64] = {};
+        const size_t n = std::fread(b, 1, sizeof(b) - 1, k);
+        std::fclose(k);
+        (void)n;
+        // arm file layout: "<target-basename> <bytes>\n" (plain char
+        // buffers on purpose: sscanf must not write into std::string SSO).
+        char armed_target[80] = {}, bytes_txt[80] = {};
+        if (std::sscanf(b, "%79s %79s", armed_target, bytes_txt) != 2)
+            return 0;
+        if (target != armed_target)
+            return 0;   // armed for a DIFFERENT slot -> never claim
+        size_t v = 0;
+        for (const char* c = bytes_txt; *c; ++c) {
+            if (*c < '0' || *c > '9')
+                return 0;
+            v = v * 10 + static_cast<size_t>(*c - '0');
+        }
+        if (v == 0)
+            return 0;
+        // ATOMIC CLAIM: rename to a unique claim name is a single
+        // namespace operation (POSIX rename(2); MoveFileEx on NTFS). Of
+        // any number of concurrent saves of THIS slot, exactly one moves
+        // the knob; every other contender fails here -> no fault.
+        std::error_code ec0;
+        std::filesystem::rename(knob, knob + ".claimed", ec0);
+        if (ec0)
+            return 0;   // lost the claim (or stale/absent knob)
+        return v;
+    }
```
e no corpo de `save_state`, substituindo a escrita única do blob (`:1602`):
```diff
-        ok = ok && io_chunked(fp, blob.data(), n_state, /*write=*/true);
+        const size_t fault = probe_fault_bytes_for(path); // 0 = off
+        if (fault > 0 && fault < static_cast<size_t>(n_state)) {
+            // TEST-ONLY: write only up to the fault offset, then take the
+            // normal failure path below (remove tmp, return false -> HTTP 400).
+            ok = ok && io_chunked(fp, blob.data(), fault, /*write=*/true);
+            ok = false;
+            log_output("[xllama] session: fault probe — abort save after " +
+                       std::to_string(fault) + " bytes (test-only)\n");
+        } else {
+            ok = ok && io_chunked(fp, blob.data(), n_state, /*write=*/true);
+        }
```

## 3. Isolamento, limpeza, limites e riscos

### 3.1 Só o slot armado (por basename do path)

O knob tem nome fixo e é **aberto antes da comparação** (a leitura é o
próprio ato de casar); saves de outro id (p.ex. `lfm`, `t2b-n1`, `t17`)
leem, comparam `target != armed_target` e retornam 0 — **não reivindicam,
não consomem, não disparam**. Ids inválidos nem chegam ao save
(`KvStore::valid_id`); `.tmp`/outros artefatos são desarmados pelo teste
de extensão antes mesmo da abertura do knob. Saves de outros slots não
modificam o knob nem os arquivos de slot.

### 3.2 Snapshot anterior nunca alcançado

A falha acontece **antes** do `rename` (`:1614`); o caminho de falha
remove apenas o tmp do teste. O `<id>.kv` previamente válido fica
byte-idêntico (é o que F1 do T18 demonstrou com sha). Um snapshot válido
de **outro** id não é sequer aberto. Restrição operacional adicional: armar
**apenas** id de teste sem baseline (evitar `t2b-n1`, `t6`… listados em
§1) para que um snapshot de gate nunca seja o alvo.

### 3.3 Saves concorrentes — reivindicação atômica e o que ela NÃO garante

**Revisão 2 — a versão anterior estava errada**: um "consumo por
leitura+remove" **não** é atômico — dois saves concorrentes do mesmo
slot podem ler o conteúdo antes de qualquer `remove` e ambos disparar.
Com essa lógica **não se pode declarar exclusividade de request**; ela
foi retirada da proposta.

**Reivindicação atômica proposta (verificável)**: após o casar o basename,
o probe faz `std::filesystem::rename(knob, knob + ".claimed", ec)`.
`rename` é uma operação única de namespace (POSIX `rename(2)`; NTFS via
`MoveFileEx`) — de qualquer número de saves concorrentes do **mesmo**
slot, exatamente um move o knob; todos os demais recebem `ec != 0` e
retornam 0 (sem falha injetada). Verificação no teste: o `.claimed`
presente após o disparo prova que a reivindicação ocorreu, e dois requests
simultâneos do mesmo slot devem produzir **exatamente um** log
`fault probe` (passo 5 do §4).

**Limitação restante — vinculação ao request de teste (fora deste
worktree)**: o claim garante "no máximo **um** save por arm", mas não
"exatamente **o** request de teste" — `save_state` só vê o path
(`session.h:200` / `session.cpp:1560`), então qualquer save do mesmo slot
pode reivindicar, e a serialização por request só existe na camada LAN
(`uwp/`), não vendorizada aqui. A vinculação por identidade de request
exige mudança **fora deste worktree**, com duas variantes exatas:
- **(a) Estender a interface**: `Session::save_state` ganharia um
  parâmetro `const std::string& request_nonce = ""`
  (`include/xllama/session.h:200`; override em `session.cpp:1560`); o
  handler do `POST /v1/slots/save` em `uwp/` passaria um header de teste
  (ex.: `X-Test-Nonce`); o knob conteria `"<basename> <bytes> <nonce>"` e
  o probe casaria os três — só o request que souber o nonce (o de teste)
  poderia reivindicar.
- **(b) Mover o claim para `uwp/`**: o handler já serializa requests e
  poderia reivindicar/validar o nonce antes de chamar `save_state`, mas a
  injeção da falha no meio do blob ainda precisaria de um canal até
  `session.cpp` — ou seja, em qualquer variante há toque em `uwp/` +
  `session.h`, fora do escopo desta etapa.

Enquanto essa mudança não for aprovada, a proposta oferece: binding por
basename de slot + claim atômico (no máximo um disparo por arm) +
procedimento que envia um único request e confere os demais ids intactos
(§4). A exclusividade **por identidade de request** fica registrada como
pendência fora do worktree, não como propriedade do probe.

### 3.4 Limpeza

- O knob reivindicado vira `slot-fault.target.txt.claimed` (o original
  deixa de existir — desarme visível: `fetch` do knob dá 404 e o
  `.claimed` está presente).
- Setup do teste: apagar `.claimed` órfão de runs anteriores **antes** de
  armar, para que o `rename` caia sempre em destino inexistente.
- Limpeza explícita pós-teste: apagar `slot-fault.target.txt` (se sobrou)
  **e** `.claimed`; apagar `<id>.kv.tmp` residual via `KvStore::erase`/API
  (`.tmp` já é removido pelo caminho de falha), confirmar listagem
  `{"ok":true,"slots":[…]}` e diretório `slots` sem `.tmp` (B11/T18 já
  provam que tmp é inerte). Não toca em outros slots, flags ou baselines.

### 3.5 Limites da prova (o que o teste NÃO provaria)

- Simula **falha de I/O durante a escrita** (o blob é truncado e o save
  aborta), não **morte de processo (SIGKILL)** no meio do write; a
  variante morte continua coberta apenas estruturalmente (loader só lê o
  caminho final + B11 + prune de tmp).
- Cobre só o caminho de falha por `ok=false` do mesmo slot; não reexercita
  os casos T16 (open/rename/fingerprint) já cobertos.
- Não mede tempo/atomicidade sob carga; é um probe de falha única.
- **Não** prova exclusividade por identidade de request (§3.3) — só "no
  máximo um por arm" e "só no slot armado".

### 3.6 Riscos

1. **Knob esquecido armado** ⇒ o próximo save cujo basename case
   reivindica (knob some) e falha limpo (400), sem corrupção; saves de
   outros slots não casam; mitigação: checagem final de ausência de knob
   e `.claimed`.
2. **Rename sobre `.claimed` stale** ⇒ no POSIX/Windows o destino é
   sobrescrito; por isso o setup apaga `.claimed` antes de armar (§3.4),
   tornando o claim de cada run distinto.
3. **Corrida leitura↔armamento**: se o upload do knob colidisse com a
   leitura, o parse falharia (`sscanf`/basename) ⇒ 0, sem disparo —
   procedimento: armar, verificar conteúdo, **então** enviar o request.
4. **Mudança em caminho quente de produção**, ainda que atrás de flag ⇒
   exige build Iteration monotônico (ex.: rev245) + gates de binding
   (bateria ON/OFF, vround/termgate/api-recall) antes de qualquer Final.
5. **Escopo**: probe test-only; se o maintainer preferir, pode ser
   compilado sob `#ifdef XLLAMA_TEST_PROBES` para não existir em release.
6. Risco de dano **praticamente nulo** com as travas acima: só o tmp do
   slot armado é escrito; nenhum arquivo final é mutado no caminho de
   falha.

## 4. Teste que o probe provaria (se aprovado)

Para um id de teste novo (ex.: `t20`) já com snapshot válido:
1. Setup: apagar `.claimed` órfão; armar `slot-fault.target.txt` =
   `"t20.kv 31457280"`; verificar o conteúdo; um save de outro id
   qualquer antes ⇒ **não** dispara e o knob permanece intacto (prova do
   isolamento §3.1).
2. Save `t20` ⇒ HTTP 400 + log `fault probe — abort save after 31457280`;
   nenhum `t20.kv` novo e **nenhum** `t20.kv.tmp` residual; knob virou
   `.claimed` (prova do claim §3.3).
3. Snapshot anterior de `t20` restaura byte-idêntico; listagem da loja
   inalterada.
4. Save subsequente de `t20` (knob reivindicado) bem-sucedido e
   restaurável.
5. **Contenção (verificação do claim)**: rearmar o knob e enviar **dois**
   requests de save de `t20` em paralelo ⇒ exatamente **um** log
   `fault probe` (o outro save completa sem injeção) — evidência
   observável de "no máximo um por arm".
Fecha deterministicamente a lacuna "falha de I/O durante a escrita",
sem kills, sem encher disco e sem tocar produção.

## 5. Decisão solicitada

Implementar o diff em uma build **Iteration** (rev245) e rodar os
passos do §4, ou manter a lacuna registrada conforme o T18? A
vinculação por identidade de request (§3.3 variantes a/b) é decisão
separada, pois toca `uwp/` + `include/xllama/session.h`, fora deste
worktree. Nada foi implementado/executado; alterações locais e testes
OpenCode em andamento seguem preservados; sem push/merge.
