import csv,json,pathlib,statistics,hashlib
b=pathlib.Path(__file__).resolve().parent;old=b.parent/'final247-direct'
metrics=['prefill_ms','ttft_ms','decode_tok_s','peak_ws_mb']
details=[];inputs=[]
for block in ['b1','b2']:
 for p in sorted((b/'blocks'/block/'csv').glob('*.csv')):
  q=old/'blocks'/block/'csv'/p.name
  rows={247:list(csv.DictReader(q.open())),255:list(csv.DictReader(p.open()))}
  assert all(len(x)==2 and all('prof0' in y['host'] for y in x) for x in rows.values())
  d={'block':block,'cell_arm':p.stem,'metrics':{}}
  for m in metrics:
   vals={v:[float(x[m]) for x in rs] for v,rs in rows.items()}
   means={v:statistics.mean(x) for v,x in vals.items()}
   d['metrics'][m]={'samples':vals,'means':means,'change_percent':100*(means[255]/means[247]-1) if means[247] else None}
  details.append(d)
  inputs.extend({'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in [p,q])
assert len(details)==48
out={'comparison':'Final255 vs Final247, identical opt-in mode2, profile OFF, sequential campaigns; source change includes identity and diagnostic instrumentation besides query policy','pairs':48,'metrics':{m:{'median_change_percent':statistics.median(d['metrics'][m]['change_percent'] for d in details),'min_change_percent':min(d['metrics'][m]['change_percent'] for d in details),'max_change_percent':max(d['metrics'][m]['change_percent'] for d in details)} for m in metrics},'details':details,'inputs':inputs}
(b/'final-cost-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['metrics'],indent=2))
