import re, concurrent.futures, hashlib, json, pathlib, secrets, subprocess, urllib.request, urllib.error
D=pathlib.Path(__file__).resolve().parent/"fault-contention"
D.mkdir(exist_ok=True)
PFN="GianlucaMazza.xllama_1.6.0.255_x64__pj67f1fcj4n14"
S="r255faultcon"
OTHER="r255faultother"
nonce=secrets.token_hex(12)
def deploy(*args):
    return subprocess.check_output(["bash","scripts/deploy.sh",*args],timeout=90)
def req(name,route,body=None,n=None):
    h={"Content-Type":"application/json"}
    if n is not None: h["X-Test-Nonce"]=n
    request=urllib.request.Request("http://192.168.1.26:11434"+route,data=None if body is None else json.dumps(body).encode(),headers=h)
    try:
        with urllib.request.urlopen(request,timeout=90) as r: code=r.status; data=json.load(r)
    except urllib.error.HTTPError as e: code=e.code; data=json.load(e)
    (D/(name+".json")).write_text(json.dumps({"http":code,"body":data},indent=2)+"\n")
    return code,data
def fetch(name,out,sub=""):
    deploy("fetch-file",PFN,name,str(D/out),sub)
    return (D/out).read_bytes()
assert req("pre","/v1/slots")[1]["slots"]==[]
listing=json.loads(deploy("list-localstate",PFN))
assert not any(i["Name"].startswith("slot-fault.target.txt") for i in listing["Items"])
assert req("seed","/v1/chat/completions",{"model":"qwen35-4b-mtp","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]})[0]==200
try:
    assert req("save-before","/v1/slots/save",{"slot":S})[0]==200
    before=fetch(S+".kv","before.kv","slots")
    arm=(S+".kv 200000 "+nonce+"\n").encode()
    (D/"arm.txt").write_bytes(arm)
    deploy("upload-file",str(D/"arm.txt"),PFN,"","slot-fault.target.txt")
    assert fetch("slot-fault.target.txt","readback.txt")==arm
    for name,slot,n in (("no-nonce",S,None),("wrong-nonce",S,nonce+"x"),("other-slot",OTHER,nonce)):
        assert req(name,"/v1/slots/save",{"slot":slot},n)[0]==200
        assert fetch("slot-fault.target.txt",name+"-arm.txt")==arm
    assert fetch(S+".kv","prefault.kv","slots")==before
    print("PASS: exact slot/nonce isolation and snapshot identity before contention",flush=True)
    l0=len(deploy("get-log",PFN).splitlines())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(req,"contender-"+str(i),"/v1/slots/save",{"slot":S},nonce) for i in range(2)]
        results=[f.result() for f in futures]
    assert sorted(r[0] for r in results)==[200,400],results
    after=fetch(S+".kv","after.kv","slots")
    assert after==before
    items=json.loads(deploy("list-localstate",PFN,"slots"))["Items"]
    assert not any(i["Name"].endswith(".tmp") for i in items)
    assert fetch("slot-fault.target.txt.claimed","claimed.txt")==arm
    lines=deploy("get-log",PFN).splitlines()[l0:]
    (D/"fault-window.txt").write_bytes(b"\n".join(lines)+b"\n")
    assert sum(b"fault probe" in l and b"abort save after 200000 bytes" in l for l in lines)==1
    observed=[re.search(rb"fwrite_bytes=(\d+) write_ok=(\d+) flush_rc=(-?\d+) tmp_size=(\d+) expected_size=(\d+)",line) for line in lines if b"fault probe" in line]
    assert len(observed)==1 and observed[0] is not None, "No independent write-size observation"
    written,write_ok,flush_rc,size,expected=map(int,observed[0].groups())
    header,body=before.split(b"\n",1)
    fields=header.split(b"\t")
    prefix=len(header)+1+int(fields[2])*4
    assert written==200000 and write_ok==1 and flush_rc==0
    assert size==expected==prefix+written
    assert prefix < size < prefix+int(fields[1])
    (D/"actual-write-observation.json").write_text(json.dumps({"fwrite_return_bytes":written,"fflush_return":flush_rc,"observed_tmp_file_size":size,"header_plus_token_bytes":prefix,"strictly_inside_target_blob":True,"durability_claim":False},indent=2)+"\n")
    assert req("restore-after","/v1/slots/restore",{"slot":S})[0]==200
    print("PASS: two concurrent exact-nonce saves -> one HTTP400 and one HTTP200, one fault log, no tmp, restore succeeds",flush=True)
    print("snapshot_sha256="+hashlib.sha256(before).hexdigest(),flush=True)
finally:
    for slot in (S,OTHER): req("delete-"+slot,"/v1/slots/delete",{"slot":slot})
    listing=json.loads(deploy("list-localstate",PFN))
    for name in ("slot-fault.target.txt","slot-fault.target.txt.claimed"):
        if any(i["Name"]==name for i in listing["Items"]): deploy("delete-file",PFN,name)
    assert req("post","/v1/slots")[1]["slots"]==[]
