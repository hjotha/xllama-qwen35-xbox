#!/usr/bin/env python3
"""Sweep draft-p-min, n_max, and temperature to maximize acceptance rate on DFlash2 Xbox GPU.

Runs on host .57 with safety trap for llama-server-root.service.
"""

import hashlib
import json
import os
import pathlib
import signal
import re
import urllib.error
import subprocess
import sys
import time
import urllib.request

PROD_SERVICE = "llama-server-root.service"
SERVER_BIN = "/home/hjotha/beellama.cpp/build-dflash-xbox-dl/bin/llama-server"
TARGET_MODEL = "/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
DRAFT_MODEL = "/home/hjotha/models/Qwen3.8-27B-DFlash2-Q8_0.gguf"
TEST_PORT = 59567
RPC_ENDPOINT = "192.168.1.26:50053"

CURRENT_PROC = None


def stop_process():
    global CURRENT_PROC
    if CURRENT_PROC is None:
        return
    proc = CURRENT_PROC
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    proc.wait(timeout=5)
    CURRENT_PROC = None


def service_command(action):
    password = os.environ.get("DFLASH_SUDO_PASSWORD")
    subprocess.run(["sudo", "-S" if password else "-n", "systemctl", action, PROD_SERVICE],
                   input=(password + "\n").encode() if password else None,
                   check=True, timeout=30)


def service_active():
    result = subprocess.run(["systemctl", "is-active", PROD_SERVICE],
                            capture_output=True, text=True, timeout=10)
    if result.stdout.strip() not in ("active", "inactive", "failed"):
        raise RuntimeError(f"Cannot determine production service state: {result.stdout} {result.stderr}")
    return result.stdout.strip() == "active"


def wait_for_health(port, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port == TEST_PORT and CURRENT_PROC is not None and CURRENT_PROC.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    return False


def server_command(args, n_max, p_min):
    cmd = [SERVER_BIN, "--model", TARGET_MODEL, "--spec-type", args.spec_type,
           "--device", "CUDA0", "--n-gpu-layers", "99", "--fit", "off",
           "--load-mode", "none", "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
           "--flash-attn", "on", "--ctx-size", "8192", "--parallel", "1",
           "--host", "127.0.0.1", "--port", str(TEST_PORT)]
    env = os.environ.copy()
    env.pop("DFLASH_SUDO_PASSWORD", None)
    for key in list(env):
        if key.startswith("DFLASH_XBOX_"):
            env.pop(key)
    if args.spec_type != "none":
        cmd += ["--spec-draft-model", args.draft_model, "--spec-draft-n-max", str(n_max),
                "--draft-p-min", str(p_min), "--spec-draft-ngl", "0", "--spec-draft-device", "none"]
        env.update(DFLASH_XBOX_RPC=RPC_ENDPOINT, DFLASH_XBOX_TRACE="1")
        if os.environ.get("DFLASH_XBOX_SEPARATE_DRAFT") == "1":
            env["DFLASH_XBOX_SEPARATE_DRAFT"] = "1"
    env["GGML_MTP_PROF"] = "1"
    return cmd, env


def run_single_benchmark(args, n_max, p_min, temperature, label):
    global CURRENT_PROC
    log_path = pathlib.Path(f"/tmp/dflash-sweep-{label}.log")
    metrics_path = pathlib.Path(f"/tmp/dflash-sweep-{label}-metrics.jsonl")
    cmd, env = server_command(args, n_max, p_min)
    if args.spec_type != "none":
        metrics_path.write_text("", encoding="utf-8")
        env["DFLASH_XBOX_METRICS"] = str(metrics_path)
    request_body = {
        "prompt": args.prompt, "n_predict": 512, "seed": 1234,
        "temperature": temperature, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
        "cache_prompt": False, "ignore_eos": True, "stream": False,
    }
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            CURRENT_PROC = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        if not wait_for_health(TEST_PORT):
            raise RuntimeError(f"Server failed to start on port {TEST_PORT}; check {log_path}")
        req = urllib.request.Request(
            f"http://127.0.0.1:{TEST_PORT}/completion",
            data=json.dumps(request_body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.load(resp)
        wall_s = time.perf_counter() - t0
        timings = payload.get("timings", {})
        predicted_n = timings.get("predicted_n", 0)
        predicted_ms = timings.get("predicted_ms", 0.0)
        tps = timings.get("predicted_per_second", 0.0)
        if "error" in payload or predicted_n <= 0 or predicted_ms <= 0 or tps <= 0:
            raise RuntimeError(f"Invalid completion: {payload}")
        draft_n = timings.get("draft_n", 0)
        accepted = timings.get("draft_n_accepted", 0)
        content = payload.get("content", "")
        # These are estimates: metrics JSONL is the source for actual RPC cycle counts.
        cycles = predicted_n - accepted
        return {
            "label": label, "spec_type": args.spec_type, "draft_model": args.draft_model if args.spec_type != "none" else None,
            "n_max": n_max, "p_min": p_min, "temperature": temperature,
            "wall_s": round(wall_s, 3), "tps": round(tps, 3), "tokens": predicted_n,
            "predicted_ms": round(predicted_ms, 2), "draft_n": draft_n, "draft_n_accepted": accepted,
            "accept_ratio_pct": round(100 * accepted / draft_n, 2) if draft_n else 0,
            "n_cycles_estimated": cycles, "tokens_per_cycle_estimated": predicted_n / cycles if cycles > 0 else None,
            "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "request": request_body, "generation_settings": payload.get("generation_settings"),
            "server_command": cmd, "log_path": str(log_path),
            "metrics_path": str(metrics_path) if args.spec_type != "none" else None,
        }
    finally:
        stop_process()


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full", "pmin", "n5", "greedy"), default="quick")
    parser.add_argument("--draft-model", default=DRAFT_MODEL,
                        help="Host draft GGUF; Xbox loaded quantization is configured independently")
    parser.add_argument("--spec-type", choices=("draft-dflash", "none"), default="draft-dflash")
    parser.add_argument("--n-max", type=int)
    parser.add_argument("--p-min", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--label", default="sweep")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/tmp/dflash-sweep-results.jsonl"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt", default="Write a short paragraph explaining why the sky appears blue.")
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.n_max is not None and args.n_max < 1):
        parser.error("repeats and n-max must be positive")
    if args.p_min is not None and not 0 <= args.p_min <= 1:
        parser.error("p-min must be between 0 and 1")
    if args.temperature is not None and not 0 <= args.temperature <= 10:
        parser.error("temperature must be between 0 and 10")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("label must use letters, numbers, underscores, dots or hyphens")
    if args.spec_type != "none" and any(x is not None for x in (args.n_max, args.p_min, args.temperature)):
        if any(x is None for x in (args.n_max, args.p_min, args.temperature)):
            parser.error("single config requires n-max, p-min and temperature together")
    return args


def configurations(args):
    all_configs = {
        "quick": [
            (4, 0.00, 1.0, "n4_p00_t10"),
            (4, 0.20, 1.0, "n4_p20_t10"),
            (4, 0.40, 1.0, "n4_p40_t10"),
            (4, 0.60, 1.0, "n4_p60_t10"),
            (4, 0.70, 1.0, "n4_p70_t10"),
            (5, 0.00, 1.0, "n5_p00_t10"),
            (5, 0.50, 1.0, "n5_p50_t10"),
            (4, 0.00, 0.0, "n4_p00_t00"),
        ],
        "pmin": [
            (4, 0.00, 1.0, "n4_p00_t10"),
            (4, 0.10, 1.0, "n4_p10_t10"),
            (4, 0.20, 1.0, "n4_p20_t10"),
            (4, 0.30, 1.0, "n4_p30_t10"),
            (4, 0.40, 1.0, "n4_p40_t10"),
            (4, 0.50, 1.0, "n4_p50_t10"),
            (4, 0.60, 1.0, "n4_p60_t10"),
            (4, 0.70, 1.0, "n4_p70_t10"),
            (4, 0.80, 1.0, "n4_p80_t10"),
        ],
        "n5": [
            (5, 0.00, 1.0, "n5_p00_t10"),
            (5, 0.20, 1.0, "n5_p20_t10"),
            (5, 0.40, 1.0, "n5_p40_t10"),
            (5, 0.60, 1.0, "n5_p60_t10"),
            (5, 0.70, 1.0, "n5_p70_t10"),
        ],
        "greedy": [
            (4, 0.00, 0.0, "n4_p00_t00"),
            (4, 0.30, 0.0, "n4_p30_t00"),
            (4, 0.60, 0.0, "n4_p60_t00"),
            (5, 0.00, 0.0, "n5_p00_t00"),
            (5, 0.50, 0.0, "n5_p50_t00"),
        ],
        "full": [
            (4, 0.00, 1.0, "n4_p00_t10"),
            (4, 0.10, 1.0, "n4_p10_t10"),
            (4, 0.20, 1.0, "n4_p20_t10"),
            (4, 0.30, 1.0, "n4_p30_t10"),
            (4, 0.40, 1.0, "n4_p40_t10"),
            (4, 0.50, 1.0, "n4_p50_t10"),
            (4, 0.60, 1.0, "n4_p60_t10"),
            (4, 0.70, 1.0, "n4_p70_t10"),
            (4, 0.80, 1.0, "n4_p80_t10"),
            (5, 0.00, 1.0, "n5_p00_t10"),
            (5, 0.30, 1.0, "n5_p30_t10"),
            (5, 0.50, 1.0, "n5_p50_t10"),
            (5, 0.65, 1.0, "n5_p65_t10"),
            (5, 0.75, 1.0, "n5_p75_t10"),
            (4, 0.00, 0.0, "n4_p00_t00"),
            (4, 0.30, 0.0, "n4_p30_t00"),
            (4, 0.60, 0.0, "n4_p60_t00"),
        ],
    }

    if args.spec_type == "none":
        return [(0, 0, args.temperature if args.temperature is not None else 1.0, "baseline")]
    if args.n_max is not None:
        return [(args.n_max, args.p_min, args.temperature, "single")]
    return all_configs[args.mode]


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main(argv=None):
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, interrupted)
    was_active = False
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        was_active = service_active()
        if was_active:
            service_command("stop")
        for repeat in range(1, args.repeats + 1):
            for n_max, p_min, temp, config_label in configurations(args):
                label = f"{args.label}-{config_label}-r{repeat}"
                result = run_single_benchmark(args, n_max, p_min, temp, label)
                result["repeat"] = repeat
                with args.output.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(result) + "\n")
                print(f"{label}: {result['tps']:.3f} tok/s, acceptance {result['accept_ratio_pct']:.2f}%", flush=True)
    finally:
        # A second SIGTERM must not interrupt service restoration.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            stop_process()
        finally:
            if was_active:
                service_command("start")
                if not wait_for_health(8090, timeout=60):
                    raise RuntimeError("Production service did not recover /health in 60s")
                print("[RESTORE] Production service is healthy.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
