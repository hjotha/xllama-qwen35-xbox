#!/usr/bin/env python3
"""Ordered scheduler split adjacency from a sched2 census log (plan 007 audit).

Parses the GGML_LOG_DEBUG stream into graphs -> splits -> nodes in emission
order, then reports, per graph class (prefill chunks by index rule / decode
rest), the ordered backend runs and the op pairs at each CPU<->D3D12 boundary.
Usage: split-adjacency.py <log> [n_prefill_graphs]
"""
import re
import sys
from collections import Counter

SPLIT = re.compile(r'## SPLIT #(\d+): (\w+) # (\d+) inputs')
NODE = re.compile(r'node #\s*(\d+) \(\s*([A-Z_0-9]+)\)')


def parse(path):
    txt = open(path, errors="replace").read()
    # keep only the last run
    starts = [m.start() for m in re.finditer(r'\[xllama\] bench: prompt\.txt', txt)]
    if starts:
        txt = txt[starts[-1]:]
    # tokens in order: split headers and node ops
    toks = []
    for m in re.finditer(r'## SPLIT #(\d+): (\w+) # (\d+) inputs|node #\s*(\d+) \(\s*([A-Z_0-9]+)\)', txt):
        if m.group(1) is not None:
            toks.append(("split", m.group(1), m.group(2)))
        else:
            toks.append(("node", m.group(5), None))
    graphs = []
    cur = None
    for t in toks:
        if t[0] == "split":
            if t[1] == "0":
                cur = []
                graphs.append(cur)
            if cur is not None:
                cur.append({"backend": t[2], "ops": []})
        elif cur is not None and cur:
            cur[-1]["ops"].append(t[1])
    return graphs


def boundaries(graph):
    """Ordered (left_op, right_op) pairs where the backend changes."""
    out = []
    for a, b in zip(graph, graph[1:]):
        if a["backend"] != b["backend"]:
            l = a["ops"][-1] if a["ops"] else "-"
            r = b["ops"][0] if b["ops"] else "-"
            out.append((a["backend"], b["backend"], l, r))
    return out


def report(path, n_prefill):
    graphs = parse(path)
    print(f"\n=== {path}  graphs={len(graphs)} prefill_chunks={n_prefill}")
    for label, gs in (("prefill", graphs[:n_prefill]), ("decode", graphs[n_prefill:])):
        if not gs:
            continue
        sp = [len(g) for g in gs]
        bnd = Counter()
        for g in gs:
            for bl, br, l, r in boundaries(g):
                bnd[(bl, br, l + "->" + r)] += 1
        print(f"  {label}: graphs={len(gs)} splits/g={sum(sp)/len(sp):.1f} "
              f"transitions/g={sum(len(boundaries(g)) for g in gs)/len(gs):.1f}")
        for (bl, br, pair), n in bnd.most_common(12):
            print(f"    {bl:>5}->{br:<5} {pair:<28} x{n}")
    return graphs


if __name__ == "__main__":
    path = sys.argv[1]
    n_prefill = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    report(path, n_prefill)
