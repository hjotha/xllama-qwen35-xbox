"""Compare original, TCP-only and fused TRIM/SYNC runs with exact trajectory checks."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from analyze_profile import TRACE, read_jsonl, stats

ROOT = Path(__file__).with_name('trim-optimization-2026-09-28')
LABELS = ['trim-before-real', 'trim-before-xbox', 'trim-tcp-real', 'trim-tcp-xbox']
LABELS += [f'trim-fused-{mode}-{i}' for i in (1, 2, 3) for mode in ('real', 'xbox')]
LABELS += ['trim-fused-local']


def main():
    runs, cycles, reference = {}, [], None
    for label in LABELS:
        rows = read_jsonl(ROOT / f'dflash-profile-{label}.jsonl')
        summary = json.loads((ROOT / f'dflash-profile-{label}-summary.json').read_text())
        trajectory = [(r['target_position'], r['n_proposed'], r['n_accepted']) for r in rows]
        identity = (summary['content_sha256'], trajectory)
        if reference is None:
            reference = identity
        assert reference == identity, f'text or acceptance changed: {label}'
        assert not any(r['rpc_failed'] for r in rows), label
        assert all(r.get('fused_sync', False) == label.startswith('trim-fused-') for r in rows), label
        events = []
        for line in (ROOT / f'dflash-profile-{label}-server.log').read_text(errors='replace').splitlines():
            match = TRACE.match(line)
            if match:
                stage, begin, end, n = match.groups()
                events.append((stage, int(begin), int(end)))
        measurements = []
        for i in range(1, len(rows) - 1):
            row = rows[i]
            start, end = row['draft_start_tick_us'], rows[i + 1]['draft_start_tick_us']
            trace = {}
            for stage, begin, finish in events:
                if start <= begin < end:
                    trace[stage] = trace.get(stage, 0) + finish - begin
            required = ('target_submit', 'target_sync', 'sample_accept', 'commit_prompt', 'kv_remove')
            assert all(k in trace for k in required), (label, i, trace)
            phases = {
                'full_cycle_us': end - start,
                'draft_rpc_us': row['draft_rpc_us'],
                'draft_compute_us': row['xbox_compute_us'],
                'sync_rpc_us': row['sync_rpc_us'],
                'sync_compute_us': row['sync_xbox_compute_us'],
                'sync_prepare_including_fused_kv_us': row['sync_xbox_prepare_us'],
                'trim_rpc_us': row['trim_rpc_us'],
                'rewind_rpc_us': row['rewind_rpc_us'],
                'control_rpc_us': row['trim_rpc_us'] + row['rewind_rpc_us'],
                'target_verify_us': trace['target_submit'] + trace['target_sync'],
                'selector_us': row['selector_us'],
                'sample_us': trace['sample_accept'],
                'commit_kv_us': trace['commit_prompt'] + trace['kv_remove'],
                'xbox_compute_total_us': row['xbox_compute_us'] + row['sync_xbox_compute_us'],
                'rpc_noncompute_us': row['draft_rpc_us'] + row['sync_rpc_us']
                    + row['trim_rpc_us'] + row['rewind_rpc_us']
                    - row['xbox_compute_us'] - row['sync_xbox_compute_us'],
            }
            cycles.append({'run': label, 'cycle': i, 'position': row['target_position'],
                           'proposed': row['n_proposed'], 'accepted': row['n_accepted'], **phases})
            measurements.append(phases)
        total = statistics.mean(m['full_cycle_us'] for m in measurements)
        runs[label] = {
            'request': summary,
            'cycles': len(rows), 'steady_cycles': len(measurements),
            'proposed': sum(r['n_proposed'] for r in rows),
            'accepted': sum(r['n_accepted'] for r in rows),
            'phases': {k: stats([m[k] for m in measurements], total) for k in measurements[0]},
        }
    groups = {}
    for mode in ('real', 'xbox'):
        labels = [f'trim-fused-{mode}-{i}' for i in (1, 2, 3)]
        selected = [r for r in cycles if r['run'] in labels]
        total = statistics.mean(r['full_cycle_us'] for r in selected)
        groups[mode] = {
            'runs': 3, 'tps_mean': statistics.mean(runs[k]['request']['tps'] for k in labels),
            'tps_min': min(runs[k]['request']['tps'] for k in labels),
            'tps_max': max(runs[k]['request']['tps'] for k in labels),
            'phases': {k: stats([r[k] for r in selected], total)
                       for k in selected[0] if k.endswith('_us')},
        }
    result = {'same_text_and_trajectory': True, 'runs': runs, 'fused_repeats': groups,
              'tokens_per_cycle': 512 / len(reference[1]),
              'swift_previous_tps': 34.99,
              'break_even_cycle_ms': 512 / len(reference[1]) / 34.99 * 1000}
    xbox = [r for r in read_jsonl(ROOT / 'xbox-dflash-dedicated.jsonl') if r.get('phase') == 'request']
    markers = [i for i, r in enumerate(xbox) if r.get('op') == 6]
    # Eleven benchmark connections followed by the EXIT cleanup disabling mock.
    markers = markers[-(len(LABELS) + 1):]
    assert len(markers) == len(LABELS) + 1, markers
    server = {}
    for i, label in enumerate(LABELS):
        segment = xbox[markers[i] + 1:markers[i + 1]]
        drafts = [r for r in segment if r['op'] == 4]
        assert len(drafts) == (0 if label.endswith('local') else runs[label]['cycles']), (label, len(drafts))
        assert all(r['status'] == 0 for r in segment), label
        fused = [r for r in segment if r['op'] == 7 and r['nTokens'] > 0]
        server[label] = {'requests_by_op': {str(op): sum(r['op'] == op for r in segment) for op in range(1, 8)},
                         'fused_kv': stats([r['kvUs'] for r in fused]) if fused else None}
    result['xbox_server'] = server
    (ROOT / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (ROOT / 'cycles.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=cycles[0])
        writer.writeheader()
        writer.writerows(cycles)
    print(json.dumps({'same_text_and_trajectory': True, 'runs': {label: {
        'tps': r['request']['tps'], 'cycle_ms': r['phases']['full_cycle_us']['mean_ms'],
        'trim_ms': r['phases']['trim_rpc_us']['mean_ms'],
        'rewind_ms': r['phases']['rewind_rpc_us']['mean_ms'],
        'accepted': r['accepted'], 'cycles': r['cycles'],
    } for label, r in runs.items()}}, indent=2))


if __name__ == '__main__':
    main()
