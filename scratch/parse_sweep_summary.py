import json

data = json.load(open(r"scratch\experiment3_sweep_results.json", "r", encoding="utf-8"))
print(f"{'Label':<30} | {'p_min':<6} | {'tok/s':<8} | {'Wall(s)':<8} | {'Draft':<6} | {'Accept':<6} | {'Ratio%':<8} | {'Tok/Cyc':<8}")
print("-" * 95)
for d in data:
    label = d["label"]
    p_min = d["p_min"]
    tps = d["tps"]
    wall_s = d["wall_s"]
    draft_n = d.get("draft_n", 0)
    accepted = d.get("draft_n_accepted", 0)
    ratio = d.get("accept_ratio_pct", 0)
    tpc = d.get("tokens_per_cycle_estimated", 0)
    print(f"{label:30} | {p_min:6.2f} | {tps:8.3f} | {wall_s:8.2f} | {draft_n:6d} | {accepted:6d} | {ratio:7.2f}% | {tpc:8.2f}")
