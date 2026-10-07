#!/usr/bin/env python3
"""Assert the exact FA2 grid and retain full-token equality plus paired rates."""
import csv, hashlib, json, statistics, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
prefix = sys.argv[1]
out = BASE / (prefix + "-grid-summary.json")
assert not out.exists(), "retain summaries; choose a fresh prefix"
result = []
for prompt in ["std", "chat", "code"]:
 for n in [64,256]:
  records = {}
  streams = {}
  for arm in ["seq", "mtp"]:
   stem=f"{prefix}-{prompt}{n}-{arm}"
   meta=json.loads((BASE/(stem+".json")).read_text())
   assert meta["status"]=="complete", stem
   rows=list(csv.DictReader((BASE/(stem+".csv")).open()))
   assert len(rows)==meta["requested"]["runs"]-1 and len(rows)>=1
   assert all(int(r["n_gen_tok"])==n for r in rows), "fixed output length mismatch: "+stem
   dumps=[]
   for run in range(1,meta["requested"]["runs"]+1):
    raw=(BASE/f"{stem}.run{run}.tokens").read_bytes()
    ids=[int(s) for s in raw.decode().splitlines() if s.strip().isdigit()]
    assert len(ids)==n, "token count mismatch: "+stem
    dumps.append(raw)
   assert len(set(dumps))==1, "intra-arm greedy mismatch"
   records[arm]=meta
   streams[arm]=dumps
  assert streams["seq"][0]==streams["mtp"][0], "full-ID mismatch"
  assert records["seq"]["package_sha256"]==records["mtp"]["package_sha256"]
  assert records["seq"]["package_pfn"]==records["mtp"]["package_pfn"]
  seq=records["seq"]["median"]["decode_tok_s"];mtp=records["mtp"]["median"]["decode_tok_s"]
  assert any(s["accepted"]>0 for s in records["mtp"]["stats"][1:]), "inactive MTP"
  result.append({"prompt":prompt,"tokens":n,"measurements_per_arm":records["seq"]["rows"],"parity_repetitions_per_arm":len(streams["seq"]),"seq_tps":seq,"mtp_tps":mtp,"gain_pct":100*(mtp/seq-1),"token_sha256":hashlib.sha256(streams["seq"][0]).hexdigest(),"package_sha256":records["seq"]["package_sha256"],"package_pfn":records["seq"]["package_pfn"],"mtp_stats":records["mtp"]["stats"]})
out.write_text(json.dumps({"status":"PASS_EXACT_GRID_FULL_IDS","cells":result,"limitation":"Small fixed sample; descriptive throughput, no statistical product certification."},indent=2)+"\n")
print(json.dumps(result))
