import base64
import json
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
PFN = "HJotha.XboxDFlashRpc_1.0.0.84_x64__3hvcqmcd1t14m"

def portal(route, method="GET", upload=None, timeout="120"):
    cmd = ["python", str(PORTAL), route, "--method", method, "--timeout", timeout]
    if upload:
        cmd.extend(["--upload", str(upload)])
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

def upload_file(local_path, remote_name):
    print(f"Uploading {remote_name} ({Path(local_path).stat().st_size} bytes)...")
    t0 = time.time()
    route = f"/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname={PFN}&path=%5CLocalState&filename={remote_name}"
    res = portal(route, "POST", upload=local_path, timeout="180")
    print(f"Uploaded {remote_name} in {time.time()-t0:.2f}s: {res}")

def main():
    print("=== DEPLOY & PROBE 1.0.0.84 ===")

    # 1. Sign
    print("1. Signing package...")
    cmd_sign = [SIGNTOOL, "sign", "/fd", "SHA256", "/sha1", CERT_SHA1, "/s", "My", str(MSIX)]
    res = subprocess.run(cmd_sign, capture_output=True, text=True)
    if res.returncode != 0:
        print("Sign failed:", res.stderr, res.stdout)
        sys.exit(1)
    print("Sign SUCCESS!")

    # 2. Stop running app
    print("2. Stopping any running app instance...")
    pkg64 = base64.b64encode(PFN.encode()).decode()
    portal(f"/api/taskmanager/app?package={pkg64}", "DELETE")
    time.sleep(2)

    # 3. Ensure DefaultUWPContentTypeToGame
    print("3. Ensuring DefaultUWPContentTypeToGame...")
    cmd = ["python", str(PORTAL), "/ext/settings/DefaultUWPContentTypeToGame", "--method", "PUT",
           "--json-file", str(ROOT / "research/dflash2-xbox/enable-game-default.json")]
    subprocess.run(cmd, capture_output=True)

    # 4. Uninstall old package
    print("4. Uninstalling old package...")
    portal(f"/api/app/packagemanager/package?package={PFN}", "DELETE")
    for _ in range(15):
        time.sleep(2)
        st = portal("/api/app/packagemanager/state")
        if not st or '"Success": true' in st or '"Success":true' in st:
            print("Uninstall complete!")
            break

    # 5. Install 1.0.0.84
    print("5. Installing 1.0.0.84 MSIX...")
    inst_res = portal(f"/api/app/packagemanager/package?package={MSIX.name}", "POST", upload=MSIX)
    print("Install triggered:", inst_res)
    for i in range(35):
        time.sleep(2)
        st = portal("/api/app/packagemanager/state")
        print(f"Install state ({i}): {st}")
        if '"Success": true' in st or '"Success":true' in st:
            print("Install complete!")
            break

    # 6. Upload LocalState files
    print("6. Uploading LocalState files...")
    upload_file(r"C:\tmp\dflash-model.txt", "dflash-model.txt")
    upload_file(r"C:\tmp\dflash-backend.txt", "dflash-backend.txt")
    upload_file(ROOT / r"research\dflash2-xbox\gpu-implementation\f0-2026-09-28\reference.cap", "reference.cap")
    upload_file(r"C:\tmp\dflash-q2.gguf", "dflash-q2.gguf")

    # 7. Launch
    print("7. Launching 1.0.0.84...")
    aumid = "HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App"
    aumid64 = base64.b64encode(aumid.encode()).decode()
    launch_res = portal(f"/api/taskmanager/app?appid={aumid64}", "POST")
    print("Launch response:", launch_res)

    # 8. Wait for port 50053
    print("8. Waiting for port 50053 to open...")
    opened = False
    for i in range(30):
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

    # 9. Execute Probe 17
    print("9. Executing Probe 17 on Xbox...")
    out_json = ROOT / "scratch/probe17_84.json"
    cmd_probe = ["python", str(ROOT / "research/dflash2-xbox/gpu-implementation/run_gpu_probe.py"),
                 "--host", "192.168.1.26", "--port", "50053", "--opcode", "17", "--output", str(out_json)]
    probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, encoding="utf-8")
    print("Probe output:", probe_res.stdout)
    if probe_res.stderr:
        print("Probe stderr:", probe_res.stderr)

    # 10. Analyze
    print("10. Comparing against baseline...")
    subprocess.run(["python", str(ROOT / "scratch/analyze_probe17.py")])

if __name__ == "__main__":
    main()
