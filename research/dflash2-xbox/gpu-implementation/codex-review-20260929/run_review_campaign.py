"""Run on .57 beside sweep_acceptance.py; receive the sudo password on stdin."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import sweep_acceptance as sweep


def main():
    os.environ["DFLASH_SUDO_PASSWORD"] = sys.stdin.readline().rstrip("\n")
    os.environ["DFLASH_XBOX_PIPELINE"] = "1"
    output = Path(__file__).resolve().parent
    results = output / "results.jsonl"
    if results.exists():
        raise RuntimeError("Preserve the previous results; choose a new run directory")
    manifest = {
        "started_unix": time.time(), "pipeline": "1", "temperature": 0.0,
        "xbox_package": "1.0.0.82", "xbox_model": "dflash-q2.gguf",
        "purpose": "Diagnostic performance only; GPU parity gate is failing",
        "server_sha256": hashlib.sha256(Path(sweep.SERVER_BIN).read_bytes()).hexdigest(),
        "sweep_sha256": hashlib.sha256(Path(sweep.__file__).read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "-C", "/home/hjotha/beellama.cpp", "rev-parse", "HEAD"], text=True).strip(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    signal.signal(signal.SIGTERM, sweep.interrupted)
    was_active = sweep.service_active()
    configs = [("target", "none", 0, 0.0),
               ("n4p25", "draft-dflash", 4, 0.25),
               ("n7p40", "draft-dflash", 7, 0.40)]
    try:
        if was_active:
            sweep.service_command("stop")
        print("Production stopped for controlled validation", flush=True)
        for round_index in range(3):
            order = configs[round_index:] + configs[:round_index]
            for name, spec_type, n_max, p_min in order:
                label = f"codex-review-20260929-{name}-r{round_index + 1}"
                args = sweep.parse_args(["--spec-type", spec_type, "--temperature", "0"]
                                        if spec_type == "none" else
                                        ["--n-max", str(n_max), "--p-min", str(p_min),
                                         "--temperature", "0"])
                print("START " + label, flush=True)
                result = sweep.run_single_benchmark(args, n_max, p_min, 0.0, label)
                result.update(round=round_index + 1, variant=name, pipeline="1")
                with results.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result) + "\n")
                print(json.dumps({k: result[k] for k in
                                  ("label", "tps", "wall_s", "tokens", "draft_n",
                                   "draft_n_accepted", "content_sha256")}), flush=True)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            sweep.stop_process()
        finally:
            if was_active:
                sweep.service_command("start")
                if not sweep.wait_for_health(8090, timeout=90):
                    raise RuntimeError("Production health did not recover")
            restored = {"was_active": was_active, "active": sweep.service_active(),
                        "healthy": sweep.wait_for_health(8090, timeout=5)}
            (output / "restored.json").write_text(json.dumps(restored))
            print("RESTORED " + json.dumps(restored), flush=True)
    assert len(results.read_text().splitlines()) == 9


if __name__ == "__main__":
    main()
