import json

with open("scratch/probe17_b3.json", "r", encoding="utf-8") as f:
    d = json.load(f)

trace = d["probe"]["firstBlockTrace"]["checkpoints"]
print(f"Total checkpoints: {len(trace)}")
found_divergence = False
for i, cp in enumerate(trace):
    name = cp["name"]
    avail = cp.get("available", False)
    ma = cp.get("maxAbs", 0)
    cos = cp.get("cosine", 1)
    if avail:
        c_rms = cp.get("cpuRms", 0)
        g_rms = cp.get("gpuRms", 0)
        # tolerance is 1e-3
        status = "OK" if ma < 1e-3 and cos > 0.9999 else "DIVERGED"
        if status == "DIVERGED" and not found_divergence:
            print(f">>> FIRST DIVERGENCE AT CHECKPOINT {i}: {name} <<<")
            found_divergence = True
        print(f"{i:02d} {name:<22}: {status:<8} maxAbs={ma:<12} cosine={cos:<10.6f} cpuRms={c_rms:<10.4f} gpuRms={g_rms:<10.4f}")
