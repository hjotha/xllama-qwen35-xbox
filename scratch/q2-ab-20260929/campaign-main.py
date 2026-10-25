"""Run on .57: compare target-only and Xbox sequential/pipeline, then restore service.

Requires the Xbox already running the selected GPU backend/model. This measures
performance and greedy output agreement; it does not certify hidden-state parity.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import ssl
import statistics
import subprocess
import time
import urllib.request


def health(port, output):
    subprocess.run(["curl", "-fsS", "--retry", "90", "--retry-delay", "1",
                    "--retry-connrefused", "--retry-all-errors", "--max-time", "3",
                    f"http://127.0.0.1:{port}/health", "-o", str(output)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xbox-package", required=True)
    parser.add_argument("--xbox-model", default="dflash-q2.gguf")
    parser.add_argument("--draft-model", default="/home/hjotha/models/Qwen3.8-27B-DFlash2-Q2_K.gguf")
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-max", type=int, choices=(1, 2, 4, 6, 7), default=4)
    parser.add_argument("--modes", nargs="+", choices=("target", "gpu-separate", "gpu-sequential", "gpu-pipeline"),
                        default=("target", "gpu-sequential", "gpu-pipeline"))
    parser.add_argument("--counts", nargs="+", type=int, default=(512, 128))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    service = "llama-server-root.service"
    root = Path("/home/hjotha/beellama.cpp")
    server = root / "build-dflash-xbox-dl/bin/llama-server"
    target = Path("/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
    draft = Path(args.draft_model)
    lock = open("/tmp/dflash-q2-layout-ab-20260929/campaign.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run(["systemctl", "is-active", "--quiet", service], check=True)
    # Read the actual startup controls, not a label inferred from host --ngl.
    controls = {}
    for name in ("dflash-backend.txt", "dflash-model.txt"):
        url = ("https://192.168.1.26:11443/api/filesystem/apps/file?knownfolderid=LocalAppData"
               f"&packagefullname=HJotha.XboxDFlashRpc_{args.xbox_package}_x64__3hvcqmcd1t14m"
               f"&path=%5CLocalState&filename={name}")
        with urllib.request.urlopen(url, context=ssl._create_unverified_context(), timeout=15) as response:
            controls[name] = response.read().decode().strip()
    assert controls == {"dflash-backend.txt": "d3d12", "dflash-model.txt": args.xbox_model}, controls
    metadata = {
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "version": subprocess.run([str(server), "--version"], capture_output=True, text=True).stderr,
        "binary_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
        "runtime_library_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted({p.resolve() for p in server.parent.glob("libllama*.so*")})
        },
        "source_status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True),
        "speculative_source_sha256": hashlib.sha256((root / "common/speculative.cpp").read_bytes()).hexdigest(),
        "target": str(target), "draft": str(draft), "xbox_package": args.xbox_package,
        "xbox_backend": controls["dflash-backend.txt"], "xbox_model": controls["dflash-model.txt"], "runs": [],
    }
    results = []
    process = None
    log = None
    try:
        subprocess.run(["systemctl", "stop", service], check=True)
        for mode in args.modes:
            env = dict(os.environ)
            for key in list(env):
                if key.startswith("DFLASH_XBOX_"):
                    del env[key]
            metrics_path = output / f"{mode}-metrics.jsonl"
            command = [str(server), "--model", str(target), "--device", "CUDA0",
                       "--n-gpu-layers", "99", "--fit", "off", "--load-mode", "none",
                       "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
                       "--flash-attn", "on", "--ctx-size", "8192", "--parallel", "1",
                       "--batch-size", "64", "--ubatch-size", "64",
                       "--host", "127.0.0.1", "--port", "59567"]
            if mode == "target":
                command += ["--spec-type", "none"]
            else:
                command += ["--spec-type", "draft-dflash", "--spec-draft-model", str(draft),
                            "--spec-draft-n-max", str(args.n_max), "--draft-p-min", str(args.p_min),
                            "--spec-draft-ngl", "0", "--spec-draft-device", "none"]
                env.update(DFLASH_XBOX_RPC="192.168.1.26:50053",
                           DFLASH_XBOX_SEPARATE_DRAFT="1" if mode == "gpu-separate" else "0",
                           DFLASH_XBOX_PIPELINE="1" if mode == "gpu-pipeline" else "0",
                           DFLASH_XBOX_METRICS=str(metrics_path), DFLASH_XBOX_TRACE="1")
            metadata["runs"].append({"mode": mode, "command": command,
                                     "env": {k: v for k, v in env.items() if k.startswith("DFLASH_XBOX_")}})
            (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
            log = (output / f"{mode}-server.log").open("wb")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({"mode": mode, "phase": "starting", "pid": process.pid}), flush=True)
            health(59567, output / f"{mode}-health.json")
            for request_index, count in enumerate(args.counts, 1):
                temperature = 0.0 if count == 128 else args.temperature
                before = len(metrics_path.read_text().splitlines()) if metrics_path.exists() else 0
                body = dict(prompt="Write a short paragraph explaining why the sky appears blue.",
                            n_predict=count, seed=1234, temperature=temperature,
                            top_p=0.95, top_k=20, min_p=0.0, repeat_penalty=1.0,
                            cache_prompt=False, ignore_eos=True, stream=False)
                label = f"{mode}-n{count}-t{temperature:g}-r{request_index}"
                (output / f"{label}-request.json").write_text(json.dumps(body, indent=2))
                request = urllib.request.Request("http://127.0.0.1:59567/completion",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=360) as response:
                    raw = response.read()
                wall = time.perf_counter() - started
                (output / f"{label}-completion.json").write_bytes(raw)
                payload = json.loads(raw)
                timings = payload["timings"]
                assert timings["predicted_n"] == count, timings
                rows = [json.loads(line) for line in metrics_path.read_text().splitlines()[before:]] if metrics_path.exists() else []
                active = [row for row in rows if row.get("n_proposed", 0) > 0]
                result = dict(mode=mode, count=count, temperature=temperature, wall_s=wall,
                              timings=timings, content_sha256=hashlib.sha256(payload.get("content", "").encode()).hexdigest(),
                              cycles=len(active), rpc_failures=sum(bool(row.get("rpc_failed")) for row in rows),
                              pipeline_used=sum(bool(row.get("pipeline_used")) for row in rows),
                              pipeline_discarded=sum(bool(row.get("pipeline_discarded")) for row in rows),
                              pipeline_attempts=max((row.get("pipeline_attempts", 0) for row in rows), default=0),
                              xbox_compute_ms_mean=statistics.mean(row["xbox_compute_us"] / 1000 for row in active) if active else None)
                result["metrics_lines"] = [before, before + len(rows)]
                result["cycle_mean_ms"] = {
                    key: statistics.mean(row[key] for row in active) / 1000
                    for key in ("total_cycle_us", "draft_rpc_us", "sync_rpc_us", "xbox_compute_us",
                                "sync_xbox_compute_us", "target_verify_us", "selector_us", "pipeline_wait_us")
                } if active else {}
                results.append(result)
                (output / "results.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process = None
            log.close()
            log = None
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log is not None:
            log.close()
        subprocess.run(["systemctl", "start", service], check=True)
        health(8090, output / "restored-health.json")
        subprocess.run(["systemctl", "is-active", service], check=True)
        print(json.dumps({"production_restored": True}), flush=True)


if __name__ == "__main__":
    main()
