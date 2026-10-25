import json

p75 = json.load(open('research/dflash2-xbox/gpu-implementation/q2-n7-performance-20260928/q2-n8-profile-75.json'))
p76 = json.load(open('scratch/probe17_76.json'))

chk75 = p75['probe']['firstBlockTrace']['checkpoints']
chk76 = p76['probe']['firstBlockTrace']['checkpoints']

gpu_total_75 = sum(c.get('gpuIntervalMs', 0) for c in chk75)
gpu_total_76 = sum(c.get('gpuIntervalMs', 0) for c in chk76)

ffn_75 = sum(c.get('gpuIntervalMs', 0) for c in chk75 if 'ffn' in c['name'] or 'swiglu' in c['name'])
ffn_76 = sum(c.get('gpuIntervalMs', 0) for c in chk76 if 'ffn' in c['name'] or 'swiglu' in c['name'])

print(f"Total GPU Timestamp Time (firstBlockTrace):")
print(f"  Package 75: {gpu_total_75:.3f} ms")
print(f"  Package 76: {gpu_total_76:.3f} ms")
print(f"  Diff:       {gpu_total_76 - gpu_total_75:+.3f} ms ({(gpu_total_76 - gpu_total_75)/gpu_total_75*100:+.1f}%)")

print(f"\nFFN GPU Timestamp Time:")
print(f"  Package 75: {ffn_75:.3f} ms")
print(f"  Package 76: {ffn_76:.3f} ms")
print(f"  Diff:       {ffn_76 - ffn_75:+.3f} ms ({(ffn_76 - ffn_75)/ffn_75*100:+.1f}%)")
