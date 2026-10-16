import hashlib,json,pathlib,subprocess,urllib.request,urllib.error
B=pathlib.Path(__file__).resolve().parent
D=B/'fingerprint-device';D.mkdir(exist_ok=True)
PFN='GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14'
S='r255identity';BAD='r255incompatible'
def deploy(*args):return subprocess.check_output(['bash','scripts/deploy.sh',*args],timeout=90)
def req(name,route,body=None):
    request=urllib.request.Request('http://192.168.1.26:11434'+route,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=120) as r:code,data=r.status,json.load(r)
    except urllib.error.HTTPError as e:code,data=e.code,json.load(e)
    (D/(name+'.json')).write_text(json.dumps({'http':code,'body':data},indent=2)+'\n')
    return code,data
chat={'model':'qwen35-4b-mtp','temperature':0,'max_tokens':2,'messages':[{'role':'user','content':'Reply exactly: ready'}]}
assert req('pre','/v1/slots')[1]['slots']==[]
try:
    assert req('seed','/v1/chat/completions',chat)[0]==200
    assert req('save','/v1/slots/save',{'slot':S})[0]==200
    deploy('fetch-file',PFN,S+'.kv',str(D/'original.kv'),'slots')
    original=(D/'original.kv').read_bytes()
    header,body=original.split(b'\n',1);fields=header.split(b'\t')
    identity=fields[3].decode()
    assert identity.startswith('v3 '),identity
    words=identity.split(' ')
    required=['scaling','orig','base','scale','ext','attn','fast','slow','fa_query']
    for field in required:assert sum(w.startswith(field+'=') for w in words)==1
    for field in ['version','legacy-policy']+required:
        changed=words.copy()
        if field=='version':changed[0]='v2'
        elif field=='legacy-policy':changed=[w for w in changed if not w.startswith('fa_query=')]
        else:
            i=next(i for i,w in enumerate(changed) if w.startswith(field+'='))
            value=changed[i].split('=',1)[1]
            changed[i]=field+'='+ (str(int(value)+1) if field in ('scaling','orig','fa_query') else f'{int(value,16)^1:08x}')
        altered=fields[:3]+[' '.join(changed).encode()]
        path=D/(field+'.kv')
        path.write_bytes(b'\t'.join(altered)+b'\n'+body)
        assert path.read_bytes().split(b'\n',1)[1]==body
        deploy('upload-file',str(path),PFN,'slots',BAD+'.kv')
        code,data=req(field+'-refusal','/v1/slots/restore',{'slot':BAD})
        assert code==400 and data.get('error',{}).get('message')=='slot restore failed: state file belongs to a different model or context',(field,code,data)
        deploy('fetch-file',PFN,BAD+'.kv',str(D/(field+'-after.kv')),'slots')
        assert (D/(field+'-after.kv')).read_bytes()==path.read_bytes(), 'Rejected snapshot changed or disappeared'
        assert req(field+'-usable','/v1/chat/completions',chat)[0]==200
        assert req(field+'-original','/v1/slots/restore',{'slot':S})[0]==200
    deploy('fetch-file',PFN,S+'.kv',str(D/'after.kv'),'slots')
    assert (D/'after.kv').read_bytes()==original
    (D/'receipt.json').write_text(json.dumps({'mode':'Final/LTCG','revision':255,'refusals':11,'post_refusal_chat_ok':11,'original_restores':11,'snapshot_sha256':hashlib.sha256(original).hexdigest(),'identity':identity,'scope':'Real Xbox handler refusal for each identity field; actual changed-GGUF model identity validated separately on host.'},indent=2)+'\n')
    print('PASS: Xbox 11/11 identity/version refusals, usable chat and original restore; original bytes preserved')
finally:
    for slot in (S,BAD):req('delete-'+slot,'/v1/slots/delete',{'slot':slot})
    assert req('post','/v1/slots')[1]['slots']==[]
