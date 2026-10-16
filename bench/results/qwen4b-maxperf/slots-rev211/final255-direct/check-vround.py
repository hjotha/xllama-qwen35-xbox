from pathlib import Path
import hashlib
base=Path(__file__).resolve().parent/'vround'
ref=base.parents[2]/'enabled-rev209'/'vround'
for cell,count in (('chat256-mtp',93),('code256-mtp',87)):
 for sw in (0,1):
  name=f'{cell}-sw{sw}'
  a=(base/(name+'.schedule-canon.txt')).read_bytes()
  b=(ref/(name+'.schedule-canon.txt')).read_bytes()
  assert a==b, name+' canonical baseline mismatch'
  assert len(a.splitlines())==count,(name,len(a.splitlines()))
  for run in (1,2):
   ta=(base/(name+f'.run{run}.tokens')).read_bytes()
   tb=(ref/(name+f'.run{run}.tokens')).read_bytes()
   assert ta==tb,(name,run,'full token mismatch')
  print(name,'canonical and both token files byte-equal to rev209',hashlib.sha256(a).hexdigest())
print('PASS: 4 canonical schedules, 8 token files, counts 93/87')
