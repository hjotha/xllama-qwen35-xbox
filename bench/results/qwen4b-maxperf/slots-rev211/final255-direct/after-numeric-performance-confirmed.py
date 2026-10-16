import pathlib,os,select,json,subprocess,datetime
b=pathlib.Path(__file__).resolve().parent;r=b.parents[4]
(b/'owned-perf-sequence-confirmed').mkdir()
pid=1864715
try:
 cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
 assert b'owned-numeric' in cmd and b'final255-direct' in cmd,cmd
 fd=os.pidfd_open(pid)
 poll=select.poll();poll.register(fd,select.POLLIN)
 print('Waiting on the verified numeric process exit event',flush=True)
 assert poll.poll(1800000),'numeric process event deadline exceeded'
 os.close(fd)
except FileNotFoundError:
 pass
assert 'numeric PASS ' in (b/'stages.txt').read_text()
n=json.loads((b/'numeric-summary.json').read_text())
assert n['d3d12be-rev255.csv']=={'rows':50,'all_ok':True}
assert n['d3d12sc-rev255.csv']=={'rows':282,'all_ok':True}
for group in ['b1','b2']:
 subprocess.run(['python3',str(b/'run-group.py'),group],cwd=r,check=True)
for script in ['summarize-blocks.py','compare-final.py']:
 subprocess.run(['python3',str(b/script)],cwd=r,check=True)
print('Numeric and both performance blocks completed; native parser remains outside this sequence.',flush=True)
