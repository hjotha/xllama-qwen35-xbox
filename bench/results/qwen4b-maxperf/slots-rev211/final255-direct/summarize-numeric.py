import csv,json,pathlib
b=pathlib.Path(__file__).resolve().parent
out={}
for name,expected in [('d3d12be-rev255.csv',50),('d3d12sc-rev255.csv',282)]:
 rows=list(csv.DictReader((b/name).open()))
 assert len(rows)==expected,(name,len(rows),expected)
 assert all(row['ok']=='1' for row in rows),name
 out[name]={'rows':len(rows),'all_ok':True}
(b/'numeric-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
