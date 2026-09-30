"""Render the measured A/B/C DFlash profile as a readable Markdown report."""

import json
import pathlib


ROOT = pathlib.Path(__file__).with_name("profiling-2026-09-27")
DATA = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
MODES = (("real", "A: DFlash real no Xbox"),
         ("xbox", "B: RPC Xbox + MOCK_DRAFT"),
         ("local", "C: replay local na .57"))
STAGES = (
    ("request_prepare", "1. Preparar request DRAFT na .57"),
    ("draft_serialize", "2. Serializar DRAFT"),
    ("draft_rpc_transport", "3/9/10. Transporte e sockets do DRAFT, agregado"),
    ("xbox_receive", "4. Receber payload DRAFT no Xbox"),
    ("xbox_prepare", "5. Preparar DFlash no Xbox"),
    ("xbox_compute", "6. Inferencia DFlash DRAFT no Xbox"),
    ("xbox_postprocess", "8. Copiar/serializar resposta Xbox"),
    ("xbox_send", "9. Enviar resposta pelo socket Xbox"),
    ("host_response_copy", "10. Copiar resposta para o seletor na .57"),
    ("host_selector", "7. Seletor na .57"),
    ("host_proposals", "11. Processar propostas"),
    ("batch_prepare", "18. Preparar batch de verificacao"),
    ("target_submit", "12a. Submeter verificacao target"),
    ("target_sync", "12b. Aguardar RTX 4070"),
    ("hidden_capture", "16a. Capturar features para SYNC"),
    ("sync_serialize", "16b. Serializar SYNC"),
    ("sync_rpc_transport", "16c. Transporte e sockets do SYNC"),
    ("sync_xbox_prepare", "17a. Preparar SYNC no Xbox"),
    ("sync_xbox_compute", "17b. Inferencia SYNC no Xbox"),
    ("trim_rpc", "15a. TRIM RPC do KV Xbox"),
    ("rewind_rpc", "15b. Rewind RPC do KV Xbox"),
    ("sample", "13. Sampling/aceitacao"),
    ("commit", "14. Commit dos tokens aceitos"),
    ("kv_remove", "15c. Remocao KV target na .57"),
    ("emit_tokens", "14b. Processar e emitir tokens finais"),
    ("other_host_wait", "18. Espera, barreira e overhead restante"),
)


def p(mode, stage, field="mean_ms"):
    return DATA["modes"][mode]["phases"][stage][field]


def table(mode):
    out = ["| Etapa | Média ms | Mediana ms | p95 ms | Mín ms | Máx ms | % ciclo |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in STAGES:
        stats = DATA["modes"][mode]["phases"][key]
        out.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
            label, stats["mean_ms"], stats["median_ms"], stats["p95_ms"],
            stats["min_ms"], stats["max_ms"], stats["percent_cycle"]))
    return "\n".join(out)


def main():
    a, b, c = (DATA["modes"][mode] for mode, _ in MODES)
    derived = DATA["derived"]
    sha = a["request"]["content_sha256"]
    lines = [
        "# Profiling Swift/ISTA + Xbox DFlash RPC (2026-09-27)",
        "",
        "## Configuração e validade",
        "",
        "- Target: Swift 1.5 / ISTA Qwen3.8-27B GSQ-RCO IQ3_XXS-mtp na `.57`, RTX 4070, contexto 8192, KV target `kvarn4`, paralelismo 1.",
        "- Drafter: DFlash2 Q3_K_M no `HJotha.XboxDFlashRpc` 1.0.0.9, CPU, KV Q4_0; output head e seletor na `.57`.",
        "- Prompt: `Write a short paragraph explaining why the sky appears blue.`; seed 1234; temperatura 1.0, top-p 0.95, top-k 20, `cache_prompt=false`, `ignore_eos=true`, `n_max=4`, `p_min=0`, `n_predict=512`.",
        "- A: DFlash real; B: Xbox reproduz os hidden states capturados em A, preserva RPC e elimina `llama_decode` em DRAFT e SYNC; C: a `.57` reproduz os mesmos hidden states sem RPC por ciclo. Posição, token âncora e tamanho do bloco são validados no replay.",
        f"- Os três modos completaram 512 tokens e os mesmos 182 ciclos, 728 propostas, 329 aceites (45.19%), sem falha RPC; hash SHA-256 do texto final: `{sha}`. A trajetória de posição/propostas/aceites também é idêntica.",
        "- Tabelas por etapa usam 180 ciclos completos estáveis: excluem o primeiro, que inclui prefill, e o último, que não tem início de ciclo seguinte. TPS vem do tempo de geração integral dos 512 tokens. Um ensaio por modo, em ordem A/B/C; variação de clock e temperatura entre ensaios não foi controlada por repetições.",
        "",
        "## Resultado direto",
        "",
        "| Modo | Geração 512 tokens | TPS final | Ciclo completo médio | Mediana | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, label in MODES:
        req = DATA["modes"][mode]["request"]
        full = DATA["modes"][mode]["phases"]["full_cycle_us"]
        lines.append(f"| {label} | {req['predicted_ms']/1000:.3f} s | {req['tps']:.2f} | "
                     f"{full['mean_ms']:.3f} ms | {full['median_ms']:.3f} ms | {full['p95_ms']:.3f} ms |")
    lines += [
        "",
        f"O rendimento real é **{derived['tokens_per_cycle']:.3f} tokens finais por ciclo**. O baseline Swift sozinho anterior foi 34.99 tok/s, então o ciclo completo deve cair abaixo de **{derived['break_even_cycle_ms']:.3f} ms** com a aceitação atual.",
        "",
        "## Onde estão os ~149 ms fora do DRAFT RPC",
        "",
        f"Em A, ciclo completo {p('real','full_cycle_us'):.3f} ms menos DRAFT RPC {p('real','draft_rpc_us'):.3f} ms = **{p('real','non_draft_us'):.3f} ms**. A divisão direta é:",
        "",
        "| Etapa fora do DRAFT RPC | Média ms | % do ciclo A |",
        "| --- | ---: | ---: |",
        f"| Verificação target na RTX 4070 | {p('real','target_verify_us'):.3f} | {p('real','target_verify_us','percent_cycle'):.2f}% |",
        f"| TRIM RPC | {p('real','trim_rpc'):.3f} | {p('real','trim_rpc','percent_cycle'):.2f}% |",
        f"| Rewind RPC (139 de 180 ciclos) | {p('real','rewind_rpc'):.3f} | {p('real','rewind_rpc','percent_cycle'):.2f}% |",
        f"| SYNC RPC, inclusive compute Xbox | {p('real','sync_rpc_us'):.3f} | {p('real','sync_rpc_us','percent_cycle'):.2f}% |",
        f"| Seletor host | {p('real','host_selector'):.3f} | {p('real','host_selector','percent_cycle'):.2f}% |",
        f"| Sampling | {p('real','sample'):.3f} | {p('real','sample','percent_cycle'):.2f}% |",
        f"| Demais preparações, commit e espera | {p('real','non_draft_us')-sum(p('real',k) for k in ('target_verify_us','trim_rpc','rewind_rpc','sync_rpc_us','host_selector','sample')):.3f} | — |",
        "",
        "## Custos isolados e decisão",
        "",
        f"- **DFlash no Xbox:** DRAFT `llama_decode` {p('real','xbox_compute'):.3f} ms/ciclo (p95 {p('real','xbox_compute','p95_ms'):.3f}); SYNC `llama_decode` {p('real','sync_xbox_compute'):.3f} ms. Total direto **{p('real','xbox_compute_total_us'):.3f} ms/ciclo**. A−B mediu {derived['real_minus_mock_xbox_ms_per_cycle']:.3f} ms/ciclo, incluindo pequenas variações nas demais etapas.",
        f"- **Rede/RPC:** soma direta dos componentes sem compute em A **{p('real','rpc_transport_total_us'):.3f} ms/ciclo** (DRAFT {p('real','draft_rpc_transport'):.3f}, SYNC {p('real','sync_rpc_transport'):.3f}, TRIM {p('real','trim_rpc'):.3f}, rewind {p('real','rewind_rpc'):.3f}). B−C mediu {derived['mock_xbox_minus_local_ms_per_cycle']:.3f} ms/ciclo.",
        f"- **Target verification:** `llama_decode` + sincronização RTX 4070 **{p('real','target_verify_us'):.3f} ms/ciclo**, p95 {p('real','target_verify_us','p95_ms'):.3f}. Submit {p('real','target_submit'):.3f}; espera da GPU {p('real','target_sync'):.3f} ms.",
        f"- **Commit/KV/SYNC:** commit + remoção de KV target {p('real','commit_kv_us'):.3f} ms. Captura + serialização SYNC {p('real','hidden_capture')+p('real','sync_serialize'):.3f} ms. SYNC RPC {p('real','sync_rpc_us'):.3f} ms (compute Xbox {p('real','sync_xbox_compute'):.3f}); TRIM + rewind {p('real','control_rpc_us'):.3f} ms/ciclo.",
        f"- **Piso medido com DFlash=0 e RPC preservado:** B, {p('xbox','full_cycle_us'):.3f} ms/ciclo e {b['request']['tps']:.2f} tok/s. O RPC atual sozinho ainda fica abaixo do baseline 34.99 tok/s.",
        f"- **Piso medido sem RPC por ciclo:** C, {p('local','full_cycle_us'):.3f} ms/ciclo e {c['request']['tps']:.2f} tok/s. É o teto observado com a mesma aceitação se compute remoto e RPC por ciclo fossem nulos; não é limite absoluto do hardware.",
        f"- Para superar o baseline, A precisa reduzir pelo menos {p('real','full_cycle_us')-derived['break_even_cycle_ms']:.3f} ms/ciclo. Mesmo B precisa reduzir {p('xbox','full_cycle_us')-derived['break_even_cycle_ms']:.3f} ms/ciclo. Eliminar apenas o compute Xbox não basta.",
        "",
        "## Três maiores gargalos",
        "",
        f"1. Compute DFlash Xbox, DRAFT + SYNC: **{p('real','xbox_compute_total_us'):.3f} ms/ciclo** ({p('real','xbox_compute_total_us','percent_cycle'):.2f}%).",
        f"2. TRIM + rewind do KV por RPC: **{p('real','control_rpc_us'):.3f} ms/ciclo** ({p('real','control_rpc_us','percent_cycle'):.2f}%). Em B, sem compute, TRIM ainda recebe o payload de 4 bytes no Xbox em média {b['xbox_trim_payload_receive']['mean_ms']:.3f} ms. O cliente faz dois `send()` sem TCP_NODELAY; atraso Nagle/ACK é uma inferência forte, não uma captura de pacotes.",
        f"3. Verificação target RTX 4070: **{p('real','target_verify_us'):.3f} ms/ciclo** ({p('real','target_verify_us','percent_cycle'):.2f}%). É o principal piso de compute no host após remover o drafter remoto.",
        "",
        "A soma dos estágios exclusivos mais `other_host_wait` fecha 100% do ciclo. `other_host_wait` em A foi "
        f"{p('real','other_host_wait'):.3f} ms ({p('real','other_host_wait','percent_cycle'):.2f}%), portanto não há uma barreira grande sem atribuição.",
        "",
        "## Limites da separação de rede",
        "",
        "Os relógios monotônicos do host e Xbox não são sincronizados. Os tempos de cada lado e o round-trip permitem atribuir o conjunto rede/socket, mas não dividir fisicamente cada direção com precisão sub-ms. `header_wait` no host inclui inferência Xbox; não é uma medida só da volta. DRAFT host `send()` e leitura do payload aparecem abaixo apenas como diagnósticos, pois já pertencem ao transporte agregado.",
        "",
        "| Diagnóstico A | Média ms | Mediana ms | p95 ms | Mín ms | Máx ms | % ciclo |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("draft_host_send_us", "Host, duas chamadas send() do DRAFT"),
                       ("draft_host_payload_read_us", "Host, leitura do payload de resposta"),
                       ("draft_header_wait_including_compute_us", "Host, espera cabeçalho inclusive compute")):
        s = a["phases"][key]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
            label, s["mean_ms"], s["median_ms"], s["p95_ms"], s["min_ms"], s["max_ms"], s["percent_cycle"]))
    for mode, label in MODES:
        lines += ["", f"## Etapas exclusivas: {label}", "",
                  "Valores por 180 ciclos estáveis. Percentual calculado sobre a média do ciclo completo do respectivo modo; as linhas somam 100% com arredondamento.",
                  "", table(mode)]
    lines += [
        "", "## Artefatos e estado final", "",
        "- `profiling-2026-09-27/cycles.csv`: 540 linhas de dados (180 ciclos estáveis × 3 modos), tempos por etapa e ciclo.",
        "- `profiling-2026-09-27/analysis.json`: estatísticas completas e diagnósticos.",
        "- `profiling-2026-09-27/dflash-profile-{real,xbox,local}.jsonl`: 182 ciclos brutos por modo; logs do servidor e Xbox no mesmo diretório.",
        "- Build `.57` instrumentado em `/home/hjotha/beellama.cpp/build-dflash-xbox-dl`, sem substituir `/home/hjotha/beellama.cpp/build/bin/llama-server` da produção. O teste em 59567 terminou; o roteador de produção voltou a `active` e `/health` respondeu `200 {\"status\":\"ok\"}`. Na última checagem o worker carregado era Bonsai 2, selecionado pelo roteador de dois modelos; Swift não estava carregado naquele instante.",
        "- O app Xbox 1.0.0.9 permanece instalado, com modo mock desativado pelo controle RPC no início de C. Instrumentação e scripts ficaram nas branches de pesquisa; nenhum commit, merge ou push foi feito.",
        "",
    ]
    output = pathlib.Path(__file__).with_name("profiling-swift-xbox-dflash-2026-09-27.md")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
