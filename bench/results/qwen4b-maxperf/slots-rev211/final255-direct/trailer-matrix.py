import hashlib,json,pathlib,subprocess,urllib.request,urllib.error
D=pathlib.Path(__file__).resolve().parent/"trailer-matrix";D.mkdir(exist_ok=False)
PFN="GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14";API="http://192.168.1.26:11434"
def deploy(*a): return subprocess.check_output(["bash","scripts/deploy.sh",*a],timeout=90)
def req(name,path,body=None):
    r=urllib.request.Request(API+path,data=None if body is None else json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(r,timeout=120) as o: code=o.status;data=json.load(o)
    except urllib.error.HTTPError as e:code=e.code;data=json.load(e)
    (D/(name+".json")).write_text(json.dumps({"http":code,"body":data},indent=2))
    return code,data
assert deploy("pfn").decode().strip()==PFN
assert req("before","/v1/slots")[1]["slots"]==[]
src=D.parent/"fault-contention/before.kv"; b=bytearray(src.read_bytes())
nl=b.index(b"\n"); h=b[:nl].decode().split("\t");off=nl+1+int(h[2])*4+int(h[1])
assert b[off:off+8]==b"XLMTPST1"
variants={}
v=bytearray(b);v[off]^=255;variants["magic"]=v
v=bytearray(b);v[off+8]^=255;variants["version"]=v
v=bytearray(b);v[-1]^=255;variants["hash"]=v
variants["truncated"]=b[:off+40];variants["absent"]=b[:off]
summary={"source_sha256":hashlib.sha256(b).hexdigest(),"trailer_offset":off,"mode":"rev255 Final/LTCG","cases":[]}
for name,data in variants.items():
    slot="r255trailer"+name
    p=D/(name+".kv");p.write_bytes(data)
    try:
        assert req(name+"-pre","/v1/chat/completions",{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":1,"messages":[{"role":"user","content":"Different trajectory: "+name}]})[0]==200
        before=deploy("get-log",PFN).splitlines()
        deploy("upload-file",str(p),PFN,"slots",slot+".kv")
        code,j=req(name+"-restore","/v1/slots/restore",{"slot":slot})
        assert code==200 and j.get("ok") is True
        assert req(name+"-after","/v1/chat/completions",{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":8,"messages":[{"role":"user","content":"Say ready."}]})[0]==200
        log=b"\n".join(deploy("get-log",PFN).splitlines()[len(before):])
        (D/(name+".log")).write_bytes(log)
        assert b"MTP mirror restored from trailer" not in log
        if name!="absent":assert b"MTP trailer refused" in log,log[-4000:]
        assert b"KV state loaded" in log
        summary["cases"].append({"case":name,"http":code,"behavior":"target restored; invalid/absent drafter trailer fallback, API usable","sha256":hashlib.sha256(data).hexdigest()})
        print("PASS",name,flush=True)
    finally:
        assert req(name+"-delete","/v1/slots/delete",{"slot":slot})[0]==200
assert req("after","/v1/slots")[1]["slots"]==[]
(D/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")

