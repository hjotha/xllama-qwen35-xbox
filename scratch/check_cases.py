import json
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else "scratch/probe17_84.json"

with open(filename, encoding="utf-8") as f:
    d = json.load(f)

cases = d["probe"]["cases"]
print(f"File: {filename}, Total cases: {len(cases)}, Overall passed: {d['probe']['passed']}")
for i, c in enumerate(cases):
    print(f"Case {i:02d}: batch={c['batch']} stage={c['stage']:<18} maxAbs={c['maxAbs']:<10.6f} cos={c['cosine']:<10.6f} mismatches={c['mismatches']:<5} pass={c['withinProvisionalTolerance']}")
