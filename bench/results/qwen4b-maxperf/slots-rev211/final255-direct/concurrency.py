import concurrent.futures, json, pathlib, time, urllib.request
base = pathlib.Path(__file__).resolve().parent / "concurrency"
base.mkdir(exist_ok=True)
url = "http://192.168.1.26:11434"
def req(name, route, body=None):
    start=time.monotonic()
    r=urllib.request.Request(url+route, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=90) as response:
        result=json.load(response)
        status=response.status
    (base/(name+".json")).write_text(json.dumps({"http":status,"elapsed_ms":(time.monotonic()-start)*1000,"body":result},indent=2)+"\n")
    return result
slots=("r255conca","r255concb")
assert req("pre","/v1/slots")["slots"] == [], "Nonempty store; abort before any writes"
seed={"model":"qwen35-4b-mtp","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"The northern depot cipher is BRAVO-7. Acknowledge with: noted."}]}
assert req("seed","/v1/chat/completions",seed)["choices"][0]["message"]["content"] == "noted"
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(req,"save-a","/v1/slots/save",{"slot":slots[0]}),pool.submit(req,"save-b","/v1/slots/save",{"slot":slots[1]}),pool.submit(req,"list-concurrent","/v1/slots"),pool.submit(req,"chat-concurrent","/v1/chat/completions",seed)]
        results=[task.result() for task in tasks]
    assert all(result["ok"] for result in results[:3])
    assert results[3]["choices"][0]["message"]["content"] == "noted"
    for slot in slots:
        assert req("restore-"+slot,"/v1/slots/restore",{"slot":slot})["ok"]
    print("PASS: simultaneous save/save/list/chat 4/4, both saved states restore")
finally:
    for slot in slots:
        req("delete-"+slot,"/v1/slots/delete",{"slot":slot})
    assert req("post","/v1/slots")["slots"] == []
