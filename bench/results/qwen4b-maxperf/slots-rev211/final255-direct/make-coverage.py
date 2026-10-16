import pathlib,json,hashlib,datetime
b=pathlib.Path(__file__).resolve().parent
stages=['numeric','cold','api','slots-on','slots-off','aligned','stability','concurrency','fault','fingerprint','trailer','termgate','vround','b1','b2']
t=(b/'stages.txt').read_text()
assert all(s+' PASS ' in t for s in stages)
perf=json.loads((b/'performance-summary.json').read_text());assert perf['positive_pairs']==24 and perf['overlap_pairs']==0
cold=json.loads((b/'cold-prefill/strict-receipt.json').read_text());assert len(cold['comparisons'])==5 and all(x['aligned_C_R_equal'] and x['cold_F_R_equal'] for x in cold['comparisons'].values())
paths=['package-verification.json','install-receipt.txt','launch-receipt.txt','build-local.log','final-build-proof.json','numeric-summary.json','stages.txt','performance-summary.json','final-cost-comparison.json','cold-prefill/strict-receipt.json','fingerprint-device/preservation-proof.json','fingerprint-device/receipt.json','fault-contention/actual-write-observation.json','native-state-final255.json','cold-prefill/capture-integrity.json']
paths += [s+'-output.txt' for s in stages if s != 'numeric']
files=[]
for name in paths:
 p=b/name;assert p.is_file(),name
 files.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
x={'revision':255,'mode':'Final/LTCG','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages_passed':stages,'strict_cold_five_files_exact':True,'performance':{k:v for k,v in perf.items() if k not in ['detail','inputs']},'mode_contract':'Existing opt-in flashattn.txt=2 throughout normal API, automatic reuse and all gates. Diagnostic capture independent and absent except first-FA probes. Native modes remain native.','limits':['Bounded tested trajectories, not universal model/precision equivalence','No forced process death, power-loss or fsync durability claim','Fingerprint configuration identity, not cryptographic model/adapter weights identity','Bounded stability cycle, not long soak','Historical rollback210 expected package not located; not substituted or installed'],'evidence':files}
(b/'coverage-final255.json').write_text(json.dumps(x,indent=2)+'\n')
print('Final255 coverage receipt written')
