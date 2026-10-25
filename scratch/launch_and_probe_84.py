import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox")
PORTAL = ROOT / r"research\dflash2-xbox\xbox_portal.py"
PFN = "HJotha.XboxDFlashRpc_1.0.0.84_x64__3hvcqmcd1t14m"

def portal(route, method="GET", timeout="120"):
    cmd = ["python", str(PORTAL), route, "--method", method, "--timeout", timeout]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

def main():
    print("=== LAUNCH & PROBE 1.0.0.84 ===")

    # 1. Stop any running app instance
    print("1. Stopping app...")
    pkg64 = base64.b64encode(PFN.encode()).decode()
    res = portal(f"/api/taskmanager/app?package={pkg64}", "DELETE")
    print("Stop response:", res)
    time.sleep(2)

    # 2. Launch 1.0.0.84
    print("2. Launching 1.0.0.84...")
    aumid = "HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App"
    aumid64 = base64.b64encode(aumid.encode()).decode()
    launch_res = portal(f"/api/taskmanager/app?appid={aumid64}", "POST")
    print("Launch response:", launch_res)

    # 3. Wait for port 50053
    print("3. Waiting for port 50053 to open...")
    opened = False
    for i in range(35):
        time.sleep(1)
        try:
            with socket.create_connection(("192.168.1.26", 50053), timeout=1):
                print(f"Port 50053 is OPEN and READY after {i+1}s!")
                opened = True
                break
        except Exception:
            pass

    if not opened:
        print("ERROR: Port 50053 did not open in time")
        sys.exit(1)

    # 4. Execute Probe 17
    print("4. Executing Probe 17 on Xbox (this takes ~15-20s)...")
    out_json = ROOT / "scratch/probe17_84.json"
    cmd_probe = ["python", str(ROOT / "research/dflash2-xbox/gpu-implementation/run_gpu_probe.py"),
                 "--host", "192.168.1.26", "--port", "50053", "--opcode", "17", "--output", str(out_json)]
    probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, encoding="utf-8")
    print("Probe output:", probe_res.stdout)
    if probe_res.stderr:
        print("Probe stderr:", probe_res.stderr)

    # 5. Analyze
    print("5. Comparing against baseline .83...")
    subprocess.run(["python", str(ROOT / "scratch/analyze_probe17.py")])

if __name__ == "__main__":
    main()
