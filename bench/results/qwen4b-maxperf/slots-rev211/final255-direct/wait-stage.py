import ctypes,select,os,pathlib,json
b=pathlib.Path(__file__).resolve().parent;p=b/'campaign-status.json'
lib=ctypes.CDLL(None);fd=lib.inotify_init1(0);assert fd>=0
assert lib.inotify_add_watch(fd,os.fsencode(p),8)>=0
if select.select([fd],[],[],45)[0]:os.read(fd,4096)
os.close(fd);print(p.read_text());print((b/'campaign-receipts.txt').read_text()[-650:])
state=json.loads(p.read_text());stage=state.get('stage','')
if stage in ['b1','b2']:
 files=list((b/'blocks'/stage/'csv').glob('*.csv'))
 print('completed CSVs',len(files),'of24')
 out=b/(stage+'-output.txt')
 if out.exists():print(out.read_text()[-500:])
