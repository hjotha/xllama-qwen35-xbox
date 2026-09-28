"""Render the measured TRIM optimization report from verified artifacts."""
from pathlib import Path
import json
import statistics

from analyze_profile import stats, read_jsonl

root = Path(__file__).with_name('trim-optimization-2026-09-28')
data = json.loads((root / 'analysis.json').read_text())
runs = data['runs']
old = runs['trim-before-real']
real = data['fused_repeats']['real']
mock = data['fused_repeats']['xbox']
mean = lambda run, phase: run['phases'][phase]['mean_ms']
lines = [
    '# Xbox DFlash RPC: TRIM and rewind optimization', '',
    'Measured on 2026-09-28. Xbox app `HJotha.XboxDFlashRpc` version `1.0.0.10`.', '',
    f"The original control round trips cost {mean(old, 'control_rpc_us'):.3f} ms/cycle. "
    'The optimized path folds both suffix removals into the existing SYNC request. '
    'Its standalone TRIM and rewind round trips are eliminated; KV work remains on the Xbox inside SYNC.', '',
    f"Real generation improved from {old['request']['tps']:.3f} to {real['tps_mean']:.3f} tok/s "
    f"({(real['tps_mean']/old['request']['tps']-1)*100:.2f}%). "
    f"Mean cycle latency fell from {mean(old, 'full_cycle_us'):.3f} to {mean(real, 'full_cycle_us'):.3f} ms.", '',
    '## Changes', '',
    '- Enable TCP_NODELAY on the Linux client; keep the header with its payload using MSG_MORE where available.',
    '- Add optional protocol-v2 opcode 7 (SYNC_TRIM). An empty request probes support without changing KV.',
    '- Remove the union of pending draft and rewind suffixes once before decoding the same target features. '
    'A rewind to zero retains the original full reset behavior.',
    '- Keep old SYNC/TRIM/RESET handlers and fallback when an older app rejects the capability probe. '
    'Keep the accept-time TRIM for drafts without a subsequent feature SYNC.',
    '- Preserve DRAFT anchor replacement, models, selector, target verification, sampling, and KV types.',
    '- DFLASH_XBOX_LEGACY_TRIM=1 reproduces separate control requests with the optimized TCP client.', '',
    '## Fixed workload and method', '',
    '`n_max=4`, `p_min=0`, `n_predict=512`, seed 1234, temperature 1.0, top_p 0.95, top_k 20, '
    'cache_prompt=false, ignore_eos=true. Prompt: '
    '`Write a short paragraph explaining why the sky appears blue.`', '',
    'Target: `/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, '
    'RTX 4070 on 192.168.1.57, target KVarN4 KV, context 8192. '
    'Drafter: `/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf`, Xbox CPU, Q4_0 KV.', '',
    'The original and optimized host sources were rebuilt into separate snapshots using the same core '
    'and backend libraries. All full comparisons use the same Xbox 1.0.0.10 app; its original v2 '
    'handlers remain available. Production was stopped with explicit user authorization and restored by EXIT cleanup.', '',
    'Each original/TCP-only/local condition has one run. The fused real and Xbox-mock conditions each '
    'have three runs. Per-cycle statistics exclude the first cycle (prefill accounting) and final cycle '
    '(no following draft timestamp). TPS uses the full 512-token request.', '',
    '## Runs', '',
    '| Run | TPS | Mean cycle ms | TRIM ms | Rewind ms | Cycles | Accepted / proposed |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for name, r in runs.items():
    lines.append(f"| {name} | {r['request']['tps']:.3f} | {mean(r, 'full_cycle_us'):.3f} | "
                 f"{mean(r, 'trim_rpc_us'):.3f} | {mean(r, 'rewind_rpc_us'):.3f} | "
                 f"{r['cycles']} | {r['accepted']} / {r['proposed']} |")
lines += ['', 'All 11 requests have identical output SHA-256 and identical position/proposal/acceptance trajectories.', '',
          f"Output SHA-256: `{old['request']['content_sha256']}`.", '',
          '## Real and mock limits', '',
          f"Fused real TPS across three runs: mean {real['tps_mean']:.3f}, "
          f"range {real['tps_min']:.3f}–{real['tps_max']:.3f}.", '',
          f"Fused Xbox mock: {mean(mock, 'full_cycle_us'):.3f} ms/cycle, "
          f"{mock['tps_mean']:.3f} tok/s (range {mock['tps_min']:.3f}–{mock['tps_max']:.3f}). "
          'This mock removes DRAFT and SYNC compute while preserving the real RPC path and captured acceptance.', '',
          f"The prior standalone Swift baseline was 34.99 tok/s; it was not rerun here. "
          f"At {data['tokens_per_cycle']:.6f} final request tokens/cycle, break-even is "
          f"{data['break_even_cycle_ms']:.3f} ms/cycle. "
          'The remaining real pipeline is dominated by Xbox inference and target verification.', '',
          '## Small-control transport experiment', '']
fused_kv = [data['xbox_server'][f'trim-fused-real-{i}']['fused_kv']['mean_ms'] for i in (1, 2, 3)]
lines += [f"Actual Xbox KV removal inside fused SYNC averaged {statistics.mean(fused_kv):.6f} ms/request "
          'across the three real runs. This work is included in SYNC preparation; zero standalone '
          'TRIM/rewind latency does not mean the KV operation was skipped.', '']
for name in ['trim-before.jsonl', 'trim-after-old-app.jsonl']:
    rows = [r for r in read_jsonl(root / name) if 'roundtrip_us' in r][1:]
    s = stats([r['roundtrip_us'] for r in rows])
    lines.append(f"- {name}: 99 steady calls, mean {s['mean_ms']:.3f} ms, "
                 f"median {s['median_ms']:.3f}, p95 {s['p95_ms']:.3f}, "
                 f"min {s['min_ms']:.3f}, max {s['max_ms']:.3f}.")
lines += ['', 'Both microbenchmarks used the same original Xbox 1.0.0.9 app. The new client also '
          'successfully fell back after the unsupported capability probe. The result is consistent with '
          'the split-write Nagle/delayed-ACK stall; no packet capture was used to distinguish the individual TCP effects.', '']
lines += [f"Most of the throughput gain comes from the TCP fix. TCP-only real generation measured "
          f"{runs['trim-tcp-real']['request']['tps']:.3f} tok/s versus {real['tps_mean']:.3f} for the full optimization; "
          'that small difference is within the observed real-run variation. Fusion removes the control round trips '
          'and their overhead, but is not responsible for the entire 33% speedup.', '']
for title, r in [('Original real', old), ('TCP-only real', runs['trim-tcp-real']),
                 ('Fused real, pooled three runs', real), ('Fused Xbox mock, pooled three runs', mock)]:
    lines += [f'## {title}: stage statistics', '',
              'RPC totals include their server compute; nested diagnostic rows overlap and must not be added together.', '',
              '| Stage | Mean ms | Median ms | p95 ms | Min ms | Max ms | % cycle |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage, s in r['phases'].items():
        lines.append(f"| {stage} | {s['mean_ms']:.3f} | {s['median_ms']:.3f} | {s['p95_ms']:.3f} | "
                     f"{s['min_ms']:.3f} | {s['max_ms']:.3f} | {s['percent_cycle']:.2f} |")
    lines += ['']
lines += ['## Validation and artifacts', '',
          '- Linux research server and Xbox Release MSIX builds succeeded. Existing compiler warnings are retained in build logs.',
          '- Two existing DFlash static regressions passed.',
          '- Real Xbox protocol tests compare hidden output bit for bit after cuts before/equal/after the draft anchor, '
          'reset, a 64-row SYNC, five rejected malformed requests, and a nonmutating capability probe.',
          '- All benchmark requests returned 512 tokens; exact trajectories and text are asserted by analyze_trim.py.',
          '- The final real DRAFT capture is byte-for-byte identical to the previous profiling capture: '
          '18,639,712 bytes, SHA-256 `8edb11cb003f3b973b442daa0750e18025ef05bfedc9ce5c55d65c8d4080d0ce`.',
          '- Raw host traces, request summaries, completion JSON, Xbox logs, source snapshots, and cycles.csv are retained.',
          '- Git diff --check and Python/shell syntax checks passed. No model-wide quality suite was claimed.', '',
          'Worktree: `C:\\Users\\hjotha\\worktrees\\xllama-dflash-xbox`, branch `research/dflash-xbox-rpc`. '
          'Host checkout: `/home/hjotha/beellama.cpp`, branch `research/dflash2-xbox-rpc` (existing llama.cpp direct-checkout exception). '
          'No commit, merge or push: the broader research changes already present remain uncommitted.', '']
lines += ['Base xllama commit: `4bad1d58d3c38dbab3d484e1dab5a611cc20c58e`. '
          'Base llama.cpp commit on both hosts: `57e515b8c0801926d298395cc900a7e154248940`. '
          'The measured code includes pre-existing research modifications plus `optimization.patch`; '
          '`source-hashes.json` records the exact before/after sources.', '',
          '## Final operational state', '',
          '- Comparison script finished with `TRIM_RUN_EXIT=0`.',
          '- Production `llama-server-root.service` restored to `active`; `/health` returned HTTP 200 with `{"status":"ok"}`.',
          '- No listener remained on test port 59567.',
          '- Xbox process `DFlashRpc.exe`, PID 3976, was running package `HJotha.XboxDFlashRpc_1.0.0.10_x64__3hvcqmcd1t14m`.',
          '- Final Xbox control request explicitly disabled mock: op 6, nTokens 0, status 0, mock false.', '']
output = root.parent / 'trim-rewind-optimization-2026-09-28.md'
output.write_text('\n'.join(lines), encoding='utf-8')
print(output.resolve())
