"""Run package 1.0.0.78 benchmark campaign on .57 and stream/save results."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, r"C:\Users\hjotha\worktrees\xllama-dflash-xbox")
sys.path.insert(0, r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox")
from profile_ssh import authorized_password, connect

output_dir_remote = "/tmp/dflash-q2-n7-pkg78-20260929"
cmd = (
    f"sudo -S python3 -u /tmp/run_pipeline_campaign_78.py --output-dir {output_dir_remote} "
    f"--xbox-package 1.0.0.78 --xbox-model dflash-q2.gguf "
    f"--draft-model /home/hjotha/models/Qwen3.8-27B-DFlash2-Q8_0.gguf "
    f"--n-max 7 --p-min 0.3 --temperature 0.0 "
    f"--modes target gpu-sequential gpu-pipeline "
    f"--counts 512 128"
)

print(f"Connecting to 192.168.1.57...", flush=True)
with connect("192.168.1.57") as client:
    print(f"Executing: {cmd}", flush=True)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=1800)
    stdin.write(authorized_password() + "\n")
    stdin.flush()

    # Stream stdout line by line
    for line in iter(stdout.readline, ""):
        print(line, end="", flush=True)

    err = stderr.read().decode("utf-8", errors="replace")
    if err:
        print(f"\nSTDERR:\n{err}", flush=True)

    exit_status = stdout.channel.recv_exit_status()
    print(f"\nExit status: {exit_status}", flush=True)

    if exit_status == 0:
        local_dir = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\gpu-implementation\q2-n7-performance-20260929-pkg78")
        local_dir.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            for item in sftp.listdir(output_dir_remote):
                try:
                    sftp.get(f"{output_dir_remote}/{item}", str(local_dir / item))
                    print(f"Downloaded {item}", flush=True)
                except Exception as e:
                    print(f"Failed {item}: {e}", flush=True)
    sys.exit(exit_status)
