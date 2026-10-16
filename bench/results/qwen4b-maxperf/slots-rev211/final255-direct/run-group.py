import pathlib,subprocess,json,datetime,sys
b=pathlib.Path(__file__).resolve().parent;r=b.parents[4]
groups={'functional':['cold','api','slots-on','slots-off','aligned','stability','concurrency','fault','fingerprint','trailer','termgate','vround'],'b1':['b1'],'b2':['b2']}
group=sys.argv[1];stages=groups[group]
(b/('owned-'+group)).mkdir()
def status(**x):
 x['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(b/'campaign-status.json').write_text(json.dumps(x,indent=2)+'\n')
for stage in stages:
 status(state='running',stage=stage,group=group)
 with (b/(stage+'-output.txt')).open('wb') as out:
  result=subprocess.run(['bash',str(b/'run-stage.sh'),stage,'confirmed-20261013'],cwd=r,stdout=out,stderr=subprocess.STDOUT)
 if result.returncode:
  status(state='failed',stage=stage,exit_code=result.returncode);raise SystemExit(result.returncode)
 with (b/'campaign-receipts.txt').open('a') as f:f.write(stage+' PASS '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
status(state='group_passed',group=group)
print('PASS group',group)
