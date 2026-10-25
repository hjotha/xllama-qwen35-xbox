# DFlash2 Xbox GPU - arquivos do plano e prompt de continuacao (2026-09-28)

Todos os caminhos relativos a `C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\`.

## Nucleo do plano GPU (F0-F7)

| Arquivo                                                        | Conteudo                                                                       |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `dflash-gpu-implementation-plan-2026-09-28.md`                 | Plano F0-F7, arquitetura, criterios                                            |
| `gpu-implementation\f0-f2-results-2026-09-28.md`               | F0-F2 (capacidades, kernels)                                                   |
| `gpu-implementation\f3-f4-results-2026-09-28.md`               | F3/F4 + causa raiz F4 (flip Q8, correlacao 0.999991) + F5                      |
| `gpu-implementation\f6-real-production-campaign-2026-09-28.md` | Campanha F6 real (21,26 vs 12,92 tok/s, saida identica)                        |
| `gpu-implementation\f7-nmax-sweep-2026-09-28.md`               | Sweep n_max (otimo=4; 7 = OOM no target)                                       |
| `gpu-implementation\f7-verdict-2026-09-28.md`                  | Veredito F7: limitado por banda de pesos (14,2 GB/s), condicoes de viabilidade |

## Contexto/historico referenciado pelo plano

`trim-rewind-optimization-2026-09-28.md`, `profiling-swift-xbox-dflash-2026-09-27.md`,
`consolidation-2026-09-27.md`, `current-status-2026-09-27.md`,
`swift-dflash-parameters-2026-09-27.md`, `dflash2-remote-pipeline-audit-2026-09-27.md`,
`dflash2-xbox-lan-2026-09-27.md`, `gokaya-baselines-2026-09-27\status.md`.

## Guia Mestre Completo (SSOT)

Consulte o documento mestre consolidado com todos os comandos, rotas, tabelas de benchmark e instruções passo a passo:
👉 [`research\dflash2-xbox\CONTINUATION.md`](../CONTINUATION.md)

Repo privado com tudo: `git@github.com:hjotha/xllama-dflash-xbox.git` (branch `research/dflash-xbox-rpc`).

## Prompt para o próximo agente

```
Tarefa: Implementar Melhorias 1 e 2 de Decodificação Especulativa DFlash2 (Xbox GPU + .57)
para superar o baseline do Swift isolado (34,99 tok/s). Estado atual: Q3_K_M (25,75 tok/s),
Q4_K_M GPU nativo (23,70 tok/s), app 1.0.0.47 instalado e validado.
Ler primeiro: research\dflash2-xbox\CONTINUATION.md e gpu-implementation\plan-acceptance-pipelining-2026-09-28.md.

Ambiente: worktree C:\Users\hjotha\worktrees\xllama-dflash-xbox (branch research/dflash-xbox-rpc;
repo privado github.com/hjotha/xllama-dflash-xbox). Console Xbox 192.168.1.26 (Device Portal 11443);
prod na .57 (.157 = ponte SSH; autenticação SSH e sudo configurada fora do repositório).

Próximos passos priorizados:
1) Melhoria 1: Varredura de --draft-p-min (0.05 a 0.80) na .57 com n_max=4 e n_max=5
   (alvo: n_accept >= 4.0 tokens/ciclo -> TPS >= 36.7 tok/s sem alterar C++).
2) Melhoria 2: Fusão RPC (DFLASH_RPC_SYNC_AND_DRAFT = 6) em uwp/dflash-rpc/dedicated.cpp e
   speculative.cpp (alvo: eliminar 1 RTT de socket, economizar 5-7 ms).
3) Melhoria 3: Pipelining assíncrono (pre-draft no Xbox durante verificação na 4070 -> TPS >= 40).

Ciclo de validação:
- Compilar MSIX: research\dflash2-xbox\build-dflash-rpc.cmd (bump em uwp\dflash-rpc\Package.appxmanifest)
- Assinar: signtool.exe sign /fd SHA256 /sha1 CD6D10EC8332955BC5C1074BC20A297D0625F1F4 /s My <msix>
- Deploy: python research\dflash2-xbox\xbox_portal.py "/api/app/packagemanager/package?package=<msix>" --method POST --upload <msix>
- Bench dedicado: python research\dflash2-xbox\dflash_dedicated_bench.py --host 192.168.1.26 --port 50053 --warmup 10 --iterations 20 --drafts 4
- Teste real .57: bash /tmp/dflash-gpu-test.sh <label> 4 [draft_path] [draft_p_min]
  (pausa/restaura llama-server-root.service automaticamente)

Regras: não commitar no main; manter backend do Xbox em 'cpu' no repouso; garantir serviço .57 ativo.
```
