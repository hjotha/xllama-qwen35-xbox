# qwen35-4b GPU decode — diagnóstico e plano do shader que falta

Data: 2026-10-02. Host: optiplex. Alvo: Xbox Series X `192.168.1.26`.
Branch: `feat/qwen35-4b-manifest` (base `64c17ca`). **Sem commit** — falta
validação de console do kernel novo.

## 1. Premissa a descartar: não existe HLSL de "dflash rpc"

**Corrigido em 2026-10-02 (12:25).** A busca original rodou só em `/home/hjotha`
e chegou à conclusão errada de que não existe shader HLSL de dflash nesta
máquina. Os shaders existem, no repositório `xllama-research-source` na
`192.168.1.193`, branch `research/dflash-xbox-rpc`:

```
C:\Users\hjotha\projects\xllama-research-source\shaders\dflash\
  quant_k_decode.hlsli  q5k_matmul.hlsl  q4k_matmul.hlsl  q3k_matmul.hlsl
  q2k_q8k_matmul.hlsl  q3k_q8k_matmul.hlsl  q6k_q8k_matmul.hlsl  q8k_quantize.hlsl
  dflash_draft_quant_q8.hlsl  dflash_attention.hlsl  sync_ops.hlsl  ggml_float.hlsli
C:\Users\hjotha\projects\xllama-research-source\shaders\rkva\
  rkva_flash_attn_kvarn4.hlsl  rkva_store_kvarn4.hlsl  rkva_wht.hlsl  ...
```

Veredito de reuso (§6.2): **o layout Q5_K é a única parte reusável**; a
arquitetura de kernel não é. Inventário no `/home/hjotha` continua verdadeiro
para aquele host:

| Onde procurado (só `/home/hjotha`)                    | Resultado                                                 |
| ----------------------------------------------------- | --------------------------------------------------------- |
| `beellama.cpp/src/models/dflash.cpp` (69 829 B)       | **0** ocorrências de `hlsl`, `d3d12`, `shader`, `compute` |
| `beellama.cpp/ggml/src/ggml-rpc/`                     | `ggml-rpc.cpp`, `transport*.cpp` — nenhum shader          |
| `find /home/hjotha -name '*.hlsl' -o -name '*.hlsli'` | só os 9 arquivos de `shaders/` do fork                    |

DFlash aqui é **arquitetura de draft model** para speculative decoding
(`--spec-type draft-dflash`, `--spec-draft-model`), com cache KVarN. O doc
`docs/quickstart-qwen36-dflash.md` é explícito: _"CUDA is the runtime-qualified
DFlash draft-KVarN backend for this release. HIP/ROCm, Vulkan … remain
unqualified."_ Não há rota de shader para portar. Nada a integrar.

## 2. O GPU decode no Xbox já existe no fork — não é inviável

`docs/gguf-gpu-decode.md` §D2 (CI `1.6.0.1138`) mediu o backend `d3d12`:

| Modelo          | decode CPU → GPU              | prefill | peak RAM         | H9                 |
| --------------- | ----------------------------- | ------- | ---------------- | ------------------ |
| qwen25-coder-3b | 14.4 → **22.8** tok/s (1.59×) | 1.32×   | **FAIL** +539 MB | **FAIL** 6/8 → 5/8 |
| lfm25-1.2b      | 39.6 → **62.5** tok/s (1.58×) | 1.74×   | **FAIL** +265 MB | PASS               |

`D2 = FAIL` é **só** peak RAM e qualidade (H9). O ganho de velocidade é real e
medido; por isso o backend é opt-in (`gguf_gpu_layers.txt`), não ausente.

O motivo de o console ter ficado em `backend=cpu` é outro: o pacote instalado
era o release upstream `1.6.0.1046`, **anterior** ao código D2b. Confirmado por
strings: o binário upstream não tem `gguf gpu layers:`, `D3D12_Host` nem
`D3D12_Weights`.

## 3. O binário com o backend existe pronto (CI do fork)

| Item     | Valor                                                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Run      | `36966362855`, workflow `build-uwp`, branch `main`, variante `unified`                                                               |
| Artefato | `xllama-appx` (id `11209837750`), MSIX `xllama_1.6.0.1_x64.msix`                                                                     |
| sha256   | `947e7a46e33f4857a6768de0b48b23a941cf164e37d271174bfbac08f7f3f0ae`                                                                   |
| Prova    | strings no `xllama.exe`: `D3D12_Host`, `D3D12_Weights`, `xllama-d3d12-v01`, `[xllama] gguf gpu layers:`, `src/bridge/ggml_d3d12.cpp` |

**Nenhum build de 40 min foi necessário** — o artefato do CI já é o main do
fork com D2b dentro.

### 3.1 Bloqueio encontrado na instalação (reproduzível)

Instalou (`code=0 reason='Success'`), mas **não sobe**: `start-app` →
`HTTP 400 {"ErrorCode":-2147024894}` = `0x80070002 ERROR_FILE_NOT_FOUND`.

Causa, documentada no próprio manifest do pacote:

```xml
<!-- Required by the Linux uwp-crossbuild /MD store-CRT path (VCRUNTIME140_APP).
     Device Portal installs without it and then fails launch as 0x80070002. -->
<PackageDependency Name="Microsoft.VCLibs.140.00" MinVersion="14.0.33519.0" ... />
```

O SDK injeta esse `MinVersion` automaticamente (o repo tem `14.0.0.0` em
`uwp/AppxManifest.xml`; o MSIX sai com `14.0.33519.0`) porque o `xllama.exe`
importa `VCRUNTIME140_APP.dll` (caminho `/MD` store-CRT — verificado por
strings). O console tem um `VCLibs.140.00` mais antigo, e **não dá para
atualizar**: toda tentativa devolve

```
error 0x80073D02: Unable to install because the following apps need to be
closed Microsoft.Xbox.DevHome_1.0.2607.19001_x64__8wekyb3d8bbwe
```

`stop-app` no DevHome devolve HTTP 200 e ele volta — é app de sistema do Xbox.

### 3.2 O que foi tentado no contorno (e onde parou)

1. Empacotar de novo com `MinVersion="14.0.0.0"` — **funcionou**, verificado:
   `BEFORE: MinVersion="14.0.33519.0"` → `AFTER: MinVersion="14.0.0.0"`,
   `makeappx pack` OK.
2. Assinar o MSIX reempacotado — **bloqueado na .193**. A conta não consegue
   criar container de chave CryptoAPI, então nenhum caminho de assinatura
   funciona:
   - `New-SelfSignedCertificate` → `CertEnroll::CX509Enrollment::_CreateRequest:
Access denied. 0x80090010 (NTE_PERM)`
   - `makecert.exe -sv` → `Error: Can't create the key of the subject`
   - `signtool sign /f <pfx>` → `No certificates were found that met all the
given criteria`, em 6 PFXs OpenSSL (com e sem EKU, PBES2 e legacy) e nos
     signtool de `10.0.19041.0` e `10.0.26100.0`
   - `Import-Certificate Cert:\CurrentUser\Root` → `UI is not allowed in this
operation` (prompt de confiança bloqueado em sessão SSH)
3. Console **restaurado** ao estado anterior: `1.6.0.1046` reinstalado por cima
   (1046 > 1, sem wipe), GGUF re-subido, override e `api.flag` de volta,
   `qwen35-4b` ativo e respondendo na API.

### 3.3 Como destravar de verdade

Qualquer um destes, na ordem de custo:

1. **Assinar numa máquina Windows com sessão desktop** (não SSH): abrir a
   .193 no console/RDP e rodar `scripts/build-uwp.ps1`, que já tem o caminho de
   certificado do próprio projeto. O `makecert`/`New-SelfSignedCertificate`
   funciona em sessão desktop — o MSIX de `29/09` na `.193` prova que já
   buildou lá.
2. **VCLibs mais novo no console**: atualizar o SO do Xbox para uma build com
   `VCLibs.140.00` ≥ `14.0.33519.0`, o que satisfaz a dependência sem mexer em
   assinatura.
3. **Build `/MT` (CRT estático)**: tira o `VCRUNTIME140_APP.dll` e com ele a
   dependência. Muda o caminho de build do CI, não é local do manifest.

## 4. Qual shader falta para o 4B — análise de tipos do GGUF

`Qwen3.5-4B-Q4_K_M.gguf` (2 740 937 888 B, 426 tensores, arch `qwen35`,
32 blocos, embedding 2560, GQA 16/4, `full_attention_interval=4` + chaves
`ssm.*` → modelo **híbrido SSM + atenção**).

`d3d12_weight_type_supported()` aceita **só** `Q4_0`, `Q4_K`, `Q6_K`
(`src/bridge/ggml_d3d12.cpp:22`).

| Tipo     | Pesos de matmul | Share     | Kernel                   |
| -------- | --------------- | --------- | ------------------------ |
| Q4_K     | 1361.0 MB       | 49.9%     | existe (`mmv_q4_k.hlsl`) |
| Q6_K     | 841.9 MB        | 30.8%     | existe (`mmv_q6_k.hlsl`) |
| **Q5_K** | **519.0 MB**    | **19.0%** | **não existe**           |
| Q8_0     | 4.2 MB          | 0.2%      | não existe               |
| F32      | 3.1 MB          | 0.1%      | n/a                      |

**Cobertura hoje: 80,7%** dos bytes de peso que são matmul.

Os 48 tensores Q5_K sem kernel são exatamente dois por bloco:

| Tensor            | Shape       | x   | Total    |
| ----------------- | ----------- | --- | -------- |
| `attn_qkv.weight` | 2560 × 8192 | 24  | 346.0 MB |
| `ssm_out.weight`  | 4096 × 2560 | 24  | 173.0 MB |

### 4.1 O shader a criar

**`shaders/ggml_d3d12_mmv_q5_k.hlsl`** — um arquivo. Derivação direta do
`mmv_q4_k.hlsl` (mesmo lane map `tid>>4` / `tid&15`, `il = itid>>2`,
`ir = itid&3`, mesmo `nchunk`/`IN_FLIGHT`, mesmo `xload`/`reduce_store`):

|              | Q4_K                       | Q5_K                                                      |
| ------------ | -------------------------- | --------------------------------------------------------- |
| bytes/bloco  | 144                        | **176**                                                   |
| layout       | `dm(4) scales(12) qs(128)` | `dm(4) scales(12) qh(32) qs(128)`                         |
| offset do qs | 16                         | **48**                                                    |
| bits         | 4                          | **5** (+ 1 bit de `qh[l]`, máscara `1 << (element >> 5)`) |
| dequant      | `d*sc*q − dmin*m`          | `d*sc*q − dmin*m` (mesma forma; **sem** offset −16)       |

O header de 4 bytes é igual (`half2 d/dmin` + 12 bytes de scale, `hdr.y/z/w`),
e `get_scale_min_k4()` é **reutilizado sem alteração** — Q5_K usa o mesmo
layout de 6 bits empacotados em `scales[12]`. Diferenças reais no laço: offset
`qs_byte = 48u + il*32u + ir*8u`, stride `blk * 176u`, e uma segunda carga
`qh_byte = 16u + ir*8u` — os 32 bytes de `qh` servem os quatro sub-blocos de 64
elementos, então `dequantize_row_q5_K` indexa `qh[l]` com `l = element & 31` e
mascara `1 << (element >> 5)`: os oito pesos de uma thread usam **um** índice
de bit só (`2*il` na metade baixa, `2*il+1` na alta), e `il` não entra no offset
do byte.

Passos:

1. `shaders/ggml_d3d12_mmv_q5_k.hlsl`
2. `scripts/compile-gpugemv-shader.sh mmv_q5_k` → `shaders/generated/ggml_d3d12_mmv_q5_k_t{64,128}_dxil.h`
3. `d3d12_weight_type_supported()`: acrescentar `GGML_TYPE_Q5_K`
4. dispatch do blob por `d3d12_mm_threads(k)`, igual aos outros três
5. emulação host em `ggml_d3d12.cpp` (o bloco de testes exige paridade com o
   dequantizador do ggml) + casos em `tests/test_ggml_d3d12.cpp`
6. gate: `scripts/bench-d3d12-selftest.sh` (D2a) antes de qualquer produto

**Ganho esperado: 80,7% → 99,7%** dos bytes de matmul na GPU.

### 4.2 O que "todo em GPU" ainda exige depois do Q5_K

O backend hoje é **matmul-only** (`dev_supports_op` aceita `MUL_MAT` + as views
`NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE`). Fora da GPU ficam, por token:

- `RMS_NORM` (177 tensores F32 de norm), `ROPE` (com `rope.dimension_sections`,
  rotary parcial), `SOFT_MAX`, os `ADD`/`MUL` de bias e máscara
- `GET_ROWS` em `token_embd.weight` (Q6_K, tied com `lm_head`) — por isso a
  duplicata de ~244 MiB que o D2 mediu
- **a parte SSM/Mamba**: `ssm.conv_kernel=4`, `ssm.state_size=128`,
  `ssm.inner_size=4096` → convolução causal + scan sequencial, que é uma
  classe de op que o backend nem tem shape para hoje
- KV cache e atenção (`offload_kqv = false` por desenho, `llama_gpu.h`)

Ou seja: o Q5_K é o gargalo de matmul e o próximo passo correto; "tudo na GPU"
é um D3 maior, não um shader a mais.

## 5. Estado do console ao fim do turno

`GianlucaMazza.xllama_1.6.0.1046_x64__pj67f1fcj4n14`, app no ar,
`http://192.168.1.26:11434` servindo, `qwen35-4b` ativo e gerando
(`"content":"OK"`). `backend=cpu` — o MSIX com D3D12 não roda neste console
pelo bloqueio §3.1.

Revalidado em 2026-10-02 12:20 (mesmo estado):
`GET /v1/models` → `{"id":"qwen35-4b","active":true}` e um
`POST /v1/chat/completions` devolvendo `"content":"OK"`.

## 6. Kernel Q5_K — implementado e validado no host (2026-10-02 13:00)

Passos 1 a 5 do plano §4.1 feitos nesta worktree. O passo 6 (gate D2a no
console) continua bloqueado por §3.1 e **não** foi medido.

### 6.1 O que mudou

| Arquivo                                                  | Mudança                                                                                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `shaders/ggml_d3d12_mmv_q5_k.hlsl`                       | novo: lane map do Q4_K, stride 176, `qs` em 48, `qh` em 16 com máscara `0x01010101` por bit                                    |
| `shaders/generated/ggml_d3d12_mmv_q5_k_t{64,128}_dxil.h` | novos: 10 304 B e 10 520 B de DXIL                                                                                             |
| `scripts/compile-gpugemv-shader.sh`                      | alvo `mmv_q5_k`                                                                                                                |
| `src/bridge/ggml_d3d12.cpp`                              | `d3d12_weight_type_supported` aceita Q5_K; ramo Q5_K no emulador host; `kPsoQ5K` + blob nas tabelas; 4 novos casos no selftest |
| `include/xllama/ggml_d3d12.h`, `docs/gguf-gpu-decode.md` | lista de tipos                                                                                                                 |
| `tests/test_ggml_d3d12.cpp`                              | Q5_K suportado; 4 formas (1024, 2560, 4096, 5120) × 2 strides                                                                  |

**Bug pego pelos testes na primeira passada:** o byte de `qh` foi escrito como
`16u + il*32u + ir*8u`. Errado — `qh` tem 32 bytes e `dequantize_row_q5_K`
indexa por `element & 31`, então o offset é `16u + ir*8u` e os quatro `il`
compartilham as mesmas duas palavras. Corrigido; a paridade com o dequantizador
do ggml passou de `rel_err ≈ 1.0` (totalmente errado) para `< 1e-4`.

### 6.2 Shaders dflash da `.193`: o que serve e o que não

| Arquivo                                                                               | Reuso no `qwen35`                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shaders/dflash/quant_k_decode.hlsli` (`dflash_q5k`)                                  | **layout sim**: `qs` em 48, `qh` em 16, bit `1 << (element >> 5)`, nibble por `element & 32`. Foi a referência que confirmou o bit único e a ausência do offset −16. Algoritmo não: 1 lane por peso com `dflash_load_u8` (um `Load` de dword por byte) → ~5,8× de amplificação de leitura contra os 176 B do bloco.                 |
| `shaders/dflash/dflash_draft_quant_q8.hlsl`                                           | `decode_ints` com `DFLASH_Q5_K` confirma o mesmo layout por um segundo caminho. É o caminho **int8 com Q8_K** (`WaveActiveSum`, `ggml_round_f32`): throughput bem maior, mas exige quantizar as ativações para Q8_K e casa com o `gpugemv` de draft, não com o `mmv` f32×f32 deste backend. Candidato para um D3, não para este PR. |
| `shaders/dflash/quant_k_matmul.hlsli`, `q*_matmul.hlsl`                               | assinatura de root diferente (`b0={n,k_dim,nb,pad0}`, `X` em `t1` SRV, `Dispatch(N,…)`, 256 threads, 1 linha por grupo) contra a do fork (`b0` de 8 constantes, `X` em `u1` UAV, `NUM_ROWS=4`, 64/128 threads). Não é drop-in.                                                                                                      |
| `shaders/rkva/rkva_flash_attn_kvarn4.hlsl`, `rkva_store_kvarn4.hlsl`, `rkva_wht.hlsl` | nada a reusar agora: são KV-cache/atenção, e o backend ainda nem tem shape para `offload_kqv` (§4.2). Relevantíssimo para o D3.                                                                                                                                                                                                     |

Os shaders `rkva_*` estão validados em hardware contra baseline CUDA
(`cos=1.0`, commit `bc361c3`), ou seja, é a evidência mais forte que existe
neste ecossistema de que a abordagem funciona no Series S — só não é o
problema de hoje.

### 6.3 Validação executada

| Gate                                                                         | Resultado                                                                                                                  |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `cmake --preset linux-test` + build `-j$(nproc)`                             | **OK** (100%)                                                                                                              |
| `xllama-tests --test-case='*d3d12*'`                                         | **11/11 casos, 112 asserções**                                                                                             |
| `ctest` completo                                                             | 243/244 — só falha `test_session.cpp:13` (SIGILL), **pré-existente**: reproduzido com `git stash` no commit base `64c17ca` |
| `clang-format 22.1.5 --dry-run --Werror`                                     | OK                                                                                                                         |
| `shellcheck 0.11.0 scripts/compile-gpugemv-shader.sh`                        | OK                                                                                                                         |
| `prettier 3.9.6 --check docs/gguf-gpu-decode.md`                             | OK                                                                                                                         |
| `check-coherence.py`, `check-release-metadata.py`                            | OK                                                                                                                         |
| `generate-benchmark-summary.py --check`, `build-research-package.py --check` | OK                                                                                                                         |
| `unittest` dos 5 módulos de CI                                               | 22 testes OK                                                                                                               |
| DXIL com `dxc` 1.9.2609                                                      | blobs Q5_K reproduzem byte-a-byte o mesmo compilador dos blobs Q4_K/Q6_K commitados (`d33cf8c0…`, `7da98de9…`)             |

O `dxc.exe` 1.8.2502.11 do Windows SDK na `.193` produz bytes diferentes
(`f39c2ef5…`), como avisa o cabeçalho do script; por isso o blob foi gerado com
1.9.2609, o mesmo que reproduz os blobs existentes.

**Falta o gate D2a** (`scripts/bench-d3d12-selftest.sh`): exige o MSIX com D3D12,
ou seja §3.1 destravado. Nenhum tok/s de Q5_K foi medido e nada foi adicionado
a `docs/benchmarks.md`.
