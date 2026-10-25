import base64
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox")
MSIX = ROOT / r"uwp\dflash-rpc\AppPackages\DFlashRpc\DFlashRpc_1.0.0.84_x64_Test\DFlashRpc_1.0.0.84_x64.msix"
PORTAL = ROOT / r"research\dflash2-xbox\xbox_portal.py"
SIGNTOOL = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
CERT_SHA1 = "CD6D10EC8332955BC5C1074BC20A297D0625F1F4"

def run_portal(route, method="GET", upload=None):
    cmd = ["python", str(PORTAL), route, "--method", method]
    if upload:
        cmd.extend(["--upload", str(upload)])
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

def main():
    print("=== ORQUESTRADOR DE VALIDAÇÃO 1.0.0.84 ===")

    # 1. Sign package
    print("1. Signing package...")
    cmd_sign = [SIGNTOOL, "sign", "/fd", "SHA256", "/sha1", CERT_SHA1, "/s", "My", str(MSIX)]
    res = subprocess.run(cmd_sign, capture_output=True, text=True)
    if res.returncode != 0:
        print("Sign failed:", res.stderr, res.stdout)
        sys.exit(1)
    print("Sign SUCCESS!")

    # 2. STOP RUNNING APP FIRST so package manager doesn't hit 0x80070005 (Access is denied)
    print("2. Stopping any running app instance...")
    pkg = "HJotha.XboxDFlashRpc_1.0.0.84_x64__3hvcqmcd1t14m"
    pkg64 = base64.b64encode(pkg.encode()).decode()
    stop_res = run_portal(f"/api/taskmanager/app?package={pkg64}", "DELETE")
    print("Stop response:", stop_res)
    time.sleep(2)

    # 3. Upload and deploy
    print("3. Uploading and updating package on Xbox...")
    res_deploy = run_portal(f"/api/app/packagemanager/package?package={MSIX.name}", "POST", upload=MSIX)
    print("Deploy response:", res_deploy)

    print("Waiting for package manager state...")
    for i in range(35):
        time.sleep(2)
        st = run_portal("/api/app/packagemanager/state")
        print(f"State ({i}):", st)
        if '"Success": true' in st or '"Success":true' in st:
            print("Package installation confirmed SUCCESS!")
            break

    # 4. Launch new app
    print("4. Launching 1.0.0.84...")
    aumid = "HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App"
    aumid64 = base64.b64encode(aumid.encode()).decode()
    launch_res = run_portal(f"/api/taskmanager/app?appid={aumid64}", "POST")
    print("Launch response:", launch_res)

    # 5. Wait for port 50053
    print("5. Waiting for port 50053 to open...")
    opened = False
    for i in range(25):
        time.sleep(1)
        try:
            with socket.create_connection(("192.168.1.26", 50053), timeout=1):
                print("Port 50053 is OPEN and READY!")
                opened = True
                break
        except Exception:
            pass

    if not opened:
        print("ERROR: Port 50053 did not open in time")
        sys.exit(1)

    # 6. Execute Probe 17
    print("6. Executing Probe 17 on Xbox...")
    out_json = ROOT / "scratch/probe17_84.json"
    cmd_probe = ["python", str(ROOT / "research/dflash2-xbox/gpu-implementation/run_gpu_probe.py"),
                 "--host", "192.168.1.26", "--port", "50053", "--opcode", "17", "--output", str(out_json)]
    probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, encoding="utf-8")
    print("Probe output:", probe_res.stdout)
    if probe_res.stderr:
        print("Probe stderr:", probe_res.stderr)

    # 7. Parse and compare results
    print("7. Comparing against baseline...")
    cmd_analyze = ["python", str(ROOT / "scratch/analyze_probe17.py")]
    subprocess.run(cmd_analyze)

if __name__ == "__main__":
    main()
