import csv,hashlib,json,pathlib,statistics
B=pathlib.Path(__file__).resolve().parent
pairs=[];inputs=[];rows_total=0
for block in ('b1','b2'):
    d=B/'blocks'/block
    assert (d/'done').exists()
    files=sorted((d/'csv').glob('*-sw0.csv'));assert len(files)==12
    for off in files:
        cell=off.stem.removesuffix('-sw0')
        values={};ws={}
        for arm in (0,1):
            p=d/'csv'/f'{cell}-sw{arm}.csv'
            rows=list(csv.DictReader(p.open()));assert len(rows)==2 and {r['run_index'] for r in rows}=={'2','3'}
            assert all('prof0' in r['host'] for r in rows)
            rows_total+=len(rows)
            values[arm]=[float(r['decode_tok_s']) for r in rows]
            ws[arm]=[float(r['peak_ws_mb']) for r in rows]
            inputs.append({'path':str(p.relative_to(B)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        offmean,onmean=map(lambda a:statistics.mean(values[a]),(0,1))
        pairs.append({'block':block,'cell':cell,'off_tok_s':offmean,'on_tok_s':onmean,'gain_percent':100*(onmean/offmean-1),'ranges_overlap':not(max(values[0])<min(values[1]) or max(values[1])<min(values[0])),'ws_mb':ws,'baseline':'parity_only_no_historical_baseline' if cell in ('std256','std256-mtp') else 'historical_digest_and_arm_parity'})
assert rows_total==96 and len(pairs)==24
g=[x['gain_percent'] for x in pairs]
summary={'revision':255,'mode':'Final/LTCG','profile':False,'measured_rows':rows_total,'pairs':len(pairs),'positive_pairs':sum(x>0 for x in g),'median_gain_percent':statistics.median(g),'mean_gain_percent':statistics.mean(g),'population_sd_percent':statistics.pstdev(g),'min_gain_percent':min(g),'max_gain_percent':max(g),'overlap_pairs':sum(x['ranges_overlap'] for x in pairs),'detail':pairs,'inputs':inputs}
(B/'performance-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('detail','inputs')},indent=2))
