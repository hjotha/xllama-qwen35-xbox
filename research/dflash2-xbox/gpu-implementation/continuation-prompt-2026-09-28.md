# DFlash2 Xbox GPU - arquivos do plano e prompt de continuacao (2026-09-28)

Todos os caminhos relativos a `C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\`.

## Nucleo do plano GPU (F0-F7)

| Arquivo | Conteudo |
|---|---|
| `dflash-gpu-implementation-plan-2026-09-28.md` | Plano F0-F7, arquitetura, criterios |
| `gpu-implementation\f0-f2-results-2026-09-28.md` | F0-F2 (capacidades, kernels) |
| `gpu-implementation\f3-f4-results-2026-09-28.md` | F3/F4 + causa raiz F4 (flip Q8, correlacao 0.999991) + F5 |
| `gpu-implementation\f6-real-production-campaign-2026-09-28.md` | Campanha F6 real (21,26 vs 12,92 tok/s, saida identica) |
| `gpu-implementation\f7-nmax-sweep-2026-09-28.md` | Sweep n_max (otimo=4; 7 = OOM no target) |
| `gpu-implementation\f7-verdict-2026-09-28.md` | Veredito F7: limitado por banda de pesos (14,2 GB/s), condicoes de viabilidade |

## Contexto/historico referenciado pelo plano

`trim-rewind-optimization-2026-09-28.md`, `profiling-swift-xbox-dflash-2026-09-27.md`,
`consolidation-2026-09-27.md`, `current-status-2026-09-27.md`,
`swift-dflash-parameters-2026-09-27.md`, `dflash2-remote-pipeline-audit-2026-09-27.md`,
`dflash2-xbox-lan-2026-09-27.md`, `gokaya-baselines-2026-09-27\status.md`.

Repo privado com tudo: `https://github.com/hjotha/xllama-dflash-xbox` (branch main).

## Prompt para o proximo agente

```
Tarefa: otimizar os kernels GEMV do DFlash2 GPU (Xbox) ate banda efetiva de pesos >=20 GB/s —
alvo: DRAFT <=46 ms/ciclo com 5 posicoes (hoje 64,5 ms @ 14,2 GB/s). Ler primeiro
research\dflash2-xbox\gpu-implementation\f7-verdict-2026-09-28.md e f3-f4-results-2026-09-28.md.

Ambiente: worktree C:\Users\hjotha\worktrees\xllama-dflash-xbox (branch research/dflash-xbox-rpc;
repo privado github.com/hjotha/xllama-dflash-xbox). Console Xbox 192.168.1.26 (Device Portal 11443);
prod na .57 (.157 = ponte SSH; senha amdk6x).

Codigo-alvo: shaders\dflash\q3k_q8k_matmul.hlsl e dflash_draft_quant_q8.hlsl — trocar a carga
byte-por-lane (ByteAddressBuffer) por cargas u32 agrupadas em groupshared por bloco de 256 pesos
(+ variante wave32), mantendo a MESMA aritmetica inteira GGML (dot exato) para nao mudar a saida.

Ciclo de validacao por iteracao:
1) python research\dflash2-xbox\gpu-implementation\q3k_q8k_compile.py && draft_compile_quant_q8.py
2) build: research\dflash2-xbox\build-dflash-rpc.cmd (bump em uwp\dflash-rpc\Package.appxmanifest)
3) sign: signtool.exe sign /fd SHA256 /sha1 CD6D10EC8332955BC5C1074BC20A297D0625F1F4 /s My <msix>
4) install: python research\dflash2-xbox\xbox_portal.py "/api/app/packagemanager/package?package=<msix>"
   --method POST --upload <msix>  (Kits: C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64)
5) probe: run_gpu_probe.py --host 192.168.1.26 --port 50053 --opcode 17 --output <json> (numeros
   devem permanecer consistentes com draft-1.0.0.4x.json)
6) bench: dflash_dedicated_bench.py --host 192.168.1.26 --port 50053 --warmup 10 --iterations 20
   --drafts 4 (meta: draft_xbox_compute_us <= 46.000; comparar com 64.516 atual)
7) real: LocalState\dflash-backend.txt=d3d12 + relaunch; na .57: bash /tmp/dflash-gpu-test.sh <label> 4
   (pausa/restaura llama-server-root.service; baseline: d3d12 21,26 tok/s, Swift sozinho 34,99 tok/s)

Regras: nao commitar sem validacao completa; resultado real vs falhas na resposta; atualizar o repo
privado (push) ao final. Estado atual: console em backend cpu (padrao), prod .57 saudavel.
```
