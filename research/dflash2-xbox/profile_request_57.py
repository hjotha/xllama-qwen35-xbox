"""Run the fixed 512-token Swift/DFlash comparison against the isolated server."""

import hashlib
import json
import pathlib
import sys
import time
import urllib.request


def main() -> int:
    mode = sys.argv[1]
    output = pathlib.Path(f"/tmp/dflash-profile-{mode}-completion.json")
    request_body = {
        "prompt": "Write a short paragraph explaining why the sky appears blue.",
        "n_predict": 512,
        "seed": 1234,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "cache_prompt": False,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:59567/completion",
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=360) as response:
        raw = response.read()
        status = response.status
    wall_s = time.perf_counter() - start
    output.write_bytes(raw)
    payload = json.loads(raw)
    timings = payload.get("timings", {})
    print(json.dumps({
        "mode": mode,
        "http_status": status,
        "wall_s": wall_s,
        "tokens": timings.get("predicted_n"),
        "tps": timings.get("predicted_per_second"),
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_ms": timings.get("prompt_ms"),
        "content_sha256": hashlib.sha256(payload.get("content", "").encode()).hexdigest(),
        "output": str(output),
    }, separators=(",", ":")))
    if status != 200 or timings.get("predicted_n") != 512:
        raise RuntimeError("benchmark did not finish 512 tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
