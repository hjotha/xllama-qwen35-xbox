"""Run Experiment 3 sweep of draft-p-min on .57 against Xbox package 1.0.0.82."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, r"C:\Users\hjotha\worktrees\xllama-dflash-xbox")
sys.path.insert(0, r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox")
from profile_ssh import authorized_password, connect

P_MIN_VALUES = [0.25, 0.30, 0.35, 0.40, 0.45]
RESULTS = []

print("Connecting to 192.168.1.57...", flush=True)
with connect("192.168.1.57") as client:
    # Remove previous results file
    client.exec_command("rm -f /tmp/dflash-sweep-results.jsonl")

    # 1. Run target-only baseline
    print("\n--- Running Target-Only Baseline (No Speculative) ---", flush=True)
    cmd_target = (
        f"DFLASH_SUDO_PASSWORD={authorized_password()} python3 /tmp/sweep_acceptance.py "
        f"--spec-type none --label target_baseline --repeats 1"
    )
    stdin, stdout, stderr = client.exec_command(cmd_target, timeout=300)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    print(out, flush=True)
    if err and "password for hjotha" not in err:
        print(f"STDERR: {err}", flush=True)

    # 2. Sweep p-min values
    for p_min in P_MIN_VALUES:
        label = f"n7_p{int(p_min*100):02d}_t00"
        print(f"\n--- Running Sweep p-min = {p_min:.2f} (n_max = 7, temp = 0.0) ---", flush=True)
        cmd_sweep = (
            f"DFLASH_SUDO_PASSWORD={authorized_password()} python3 /tmp/sweep_acceptance.py "
            f"--spec-type draft-dflash --draft-model /home/hjotha/models/Qwen3.8-27B-DFlash2-Q8_0.gguf "
            f"--n-max 7 --p-min {p_min} --temperature 0.0 --label {label} --repeats 1"
        )
        stdin, stdout, stderr = client.exec_command(cmd_sweep, timeout=300)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(out, flush=True)
        if err and "password for hjotha" not in err:
            print(f"STDERR: {err}", flush=True)

    # 3. Read results from /tmp/dflash-sweep-results.jsonl via SFTP
    sftp = client.open_sftp()
    try:
        remote_data = sftp.file("/tmp/dflash-sweep-results.jsonl").read().decode("utf-8")
        for line in remote_data.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    parsed = json.loads(line)
                    RESULTS.append(parsed)
                except Exception as e:
                    print(f"Error parsing line: {e}", flush=True)
    except Exception as e:
        print(f"Error reading results file: {e}", flush=True)

    # 4. Check production service status
    stdin, stdout, stderr = client.exec_command("systemctl is-active llama-server-root.service", timeout=30)
    prod_status = stdout.read().decode().strip()
    print(f"\nFinal llama-server-root.service status: {prod_status}", flush=True)

out_file = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\scratch\experiment3_sweep_results.json")
out_file.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
print(f"\nSaved {len(RESULTS)} results to {out_file}", flush=True)
