import json
import sys

def main():
    path83 = "scratch/probe17_baseline_test.json"
    path84 = "scratch/probe17_84.json"

    with open(path83, "r", encoding="utf-8") as f:
        d83 = json.load(f)
    with open(path84, "r", encoding="utf-8") as f:
        d84 = json.load(f)

    p83 = d83.get("probe", d83.get("result", {}))
    p84 = d84.get("probe", d84.get("result", {}))

    cases83 = p83.get("cases", [])
    cases84 = p84.get("cases", [])

    print(f"Total Cases: .83={len(cases83)} | .84={len(cases84)}")
    print(f"{'Batch':<6} {'Stage':<18} {'.83 GPU (ms)':<14} {'.84 GPU (ms)':<14} {'Delta':<10} {'%':<8} {'MaxAbs':<8} {'Cosine':<8}")
    print("-" * 88)

    sum83 = 0.0
    sum84 = 0.0
    mismatches = 0

    for c83, c84 in zip(cases83, cases84):
        b = c83["batch"]
        st = c83["stage"]
        g83 = c83["gpuMs"]
        g84 = c84["gpuMs"]
        sum83 += g83
        sum84 += g84
        delta = g84 - g83
        pct = (delta / g83) * 100 if g83 else 0
        ma = c84.get("maxAbs", -1)
        cos = c84.get("cosine", -1)
        if ma > 0:
            mismatches += 1
        print(f"{b:<6} {st:<18} {g83:<14.3f} {g84:<14.3f} {delta:<+10.3f} {pct:<+7.2f}% {ma:<8} {cos:<8.6f}")

    print("-" * 88)
    n = len(cases83) or 1
    print(f"Media Total GPU: .83={sum83/n:.3f} ms | .84={sum84/n:.3f} ms | Diff: {(sum84-sum83)/n:+.3f} ms ({((sum84-sum83)/sum83)*100:+.2f}%)")
    print(f"Mismatches: {mismatches} / {len(cases84)}")

if __name__ == "__main__":
    main()
