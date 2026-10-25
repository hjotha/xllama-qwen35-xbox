import json
import sys
from collections import defaultdict

def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\hjotha\worktrees\xllama-dflash-xbox\scratch\probe17_batch8.json'
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    probe = data['probe']
    trace = probe.get('firstBlockTrace')
    if not trace:
        print("No firstBlockTrace found!")
        return
    checkpoints = trace['checkpoints']
    total_interval = sum(c.get('gpuIntervalMs', 0) for c in checkpoints)
    print(f"File: {filename}")
    print(f"Batch: {trace.get('batch')}")
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Sum of intervals: {total_interval:.4f} ms")

    # Categorize by operation base name
    cat_times = defaultdict(float)
    cat_counts = defaultdict(int)

    for c in checkpoints:
        name = c['name']
        dt = c.get('gpuIntervalMs', 0.0)
        # remove layer suffix if any (e.g. noise_norm-0 -> noise_norm)
        base = name.split('-')[0]
        cat_times[base] += dt
        cat_counts[base] += 1

    print("\n=== AGGREGATED BY OPERATION ===")
    print(f"{'Operation':20s} {'Count':6s} {'Total (ms)':12s} {'Mean (ms)':10s} {'% of Total':10s}")
    for base, t in sorted(cat_times.items(), key=lambda x: x[1], reverse=True):
        cnt = cat_counts[base]
        pct = (t / total_interval) * 100 if total_interval > 0 else 0
        mean = t / cnt if cnt > 0 else 0
        print(f"{base:20s} {cnt:6d} {t:12.4f} {mean:10.4f} {pct:9.2f}%")

    print("\n=== INDIVIDUAL CHECKPOINTS (>0.03 ms) ===")
    print(f"{'Idx':3s} {'Checkpoint Name':22s} {'Interval (ms)':14s}")
    for i, c in enumerate(checkpoints):
        dt = c.get('gpuIntervalMs', 0.0)
        if dt > 0.03:
            print(f"{i:3d} {c['name']:22s} {dt:14.4f}")

if __name__ == '__main__':
    main()
