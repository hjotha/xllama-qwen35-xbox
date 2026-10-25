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
AUMID = "HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def portal(route, method="GET", upload=None, timeout="180"):
    cmd = ["python", str(PORTAL), route, "--method", method, "--timeout", str(timeout)]
    if upload:
        cmd.extend(["--upload", str(upload)])
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

def upload_file(local_path, remote_name):
    size_mb = Path(local_path).stat().st_size / (1024 * 1024)
    log(f"Uploading {remote_name} ({size_mb:.2f} MB)...")
    t0 = time.time()
    route = f"/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname={PFN}&path=%5CLocalState&filename={remote_name}"
    res = portal(route, "POST", upload=local_path, timeout="300")
    log(f"Uploaded {remote_name} in {time.time()-t0:.2f}s: {res}")
    if '"Success": false' in res or '"Success":false' in res or "http_status" in res and "400" in res:
        raise RuntimeError(f"Upload failed: {res}")

def main():
    log("=== FULL DEPLOY AND TEST PACKAGE 1.0.0.84 ===")

    # 1. Sign
    log("1. Signing package...")
    cmd_sign = [SIGNTOOL, "sign", "/fd", "SHA256", "/sha1", CERT_SHA1, "/s", "My", str(MSIX)]
    res = subprocess.run(cmd_sign, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"Sign failed: {res.stderr} {res.stdout}")
        sys.exit(1)
    log("Sign SUCCESS!")

    # 2. Ensure DefaultUWPContentTypeToGame
    log("2. Ensuring DefaultUWPContentTypeToGame...")
    cmd = ["python", str(PORTAL), "/ext/settings/DefaultUWPContentTypeToGame", "--method", "PUT",
           "--json-file", str(ROOT / "research/dflash2-xbox/enable-game-default.json")]
    subprocess.run(cmd, capture_output=True)

    # 3. Install 1.0.0.84
    log(f"3. Installing {MSIX.name} ({MSIX.stat().st_size / (1024*1024):.2f} MB)...")
    inst_res = portal(f"/api/app/packagemanager/package?package={MSIX.name}", "POST", upload=MSIX, timeout="180")
    log(f"Install response: {inst_res}")

    for i in range(40):
        time.sleep(2)
        st = portal("/api/app/packagemanager/state")
        log(f"Install state ({i}): {st}")
        if '"Success": true' in st or '"Success":true' in st:
            log("Install COMPLETE!")
            break

    # 4. Check if files in LocalState already exist
    log("4. Checking LocalState...")
    files_res = portal(f"/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname={PFN}&path=%5CLocalState")
    log(f"LocalState query: {files_res}")

    # If LocalState not created yet, launch app once to force Windows to create LocalState
    if "Failed to get package origin" in files_res or "cannot find the path" in files_res or "500" in files_res:
        log("LocalState missing, launching app once to initialize...")
        aumid64 = base64.b64encode(AUMID.encode()).decode()
        portal(f"/api/taskmanager/app?appid={aumid64}", "POST")
        time.sleep(3)
        # Stop app
        pkg64 = base64.b64encode(PFN.encode()).decode()
        portal(f"/api/taskmanager/app?package={pkg64}", "DELETE")
        time.sleep(2)
        files_res = portal(f"/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname={PFN}&path=%5CLocalState")

    # 5. Upload required files only if missing or size mismatch
    log("5. Uploading asset files...")
    existing_files = {}
    try:
        data = json.loads(files_res)
        for item in data.get("data", {}).get("Files", []):
            existing_files[item.get("Name", "")] = item.get("FileSize", 0)
    except Exception:
        pass
    log(f"Existing files in LocalState: {existing_files}")

    def upload_if_needed(local_path, remote_name):
        local_size = Path(local_path).stat().st_size
        if existing_files.get(remote_name) == local_size:
            log(f"Skipping {remote_name} (already present with size {local_size} bytes)")
            return
        upload_file(local_path, remote_name)

    upload_if_needed(r"C:\tmp\dflash-model.txt", "dflash-model.txt")
    upload_if_needed(r"C:\tmp\dflash-backend.txt", "dflash-backend.txt")
    upload_if_needed(ROOT / r"research\dflash2-xbox\gpu-implementation\f0-2026-09-28\reference.cap", "reference.cap")
    upload_if_needed(r"C:\tmp\dflash-q2.gguf", "dflash-q2.gguf")

    # 6. Stop and Launch App
    log("6. Stopping any existing instances...")
    pkg64 = base64.b64encode(PFN.encode()).decode()
    portal(f"/api/taskmanager/app?package={pkg64}", "DELETE")
    time.sleep(2)

    log("7. Launching 1.0.0.84...")
    aumid64 = base64.b64encode(AUMID.encode()).decode()
    launch_res = portal(f"/api/taskmanager/app?appid={aumid64}", "POST")
    log(f"Launch response: {launch_res}")

    # 7. Wait for port 50053
    log("8. Waiting for port 50053 to open...")
    opened = False
    for i in range(40):
        time.sleep(1)
        try:
            with socket.create_connection(("192.168.1.26", 50053), timeout=1):
                log(f"Port 50053 is OPEN and READY after {i+1}s!")
                opened = True
                break
        except Exception:
            pass

    if not opened:
        log("ERROR: Port 50053 did not open in time")
        sys.exit(1)

    # 8. Run Probe 17
    log("9. Executing Probe 17 on Xbox Series X...")
    out_json = ROOT / "scratch/probe17_84.json"
    cmd_probe = ["python", str(ROOT / "research/dflash2-xbox/gpu-implementation/run_gpu_probe.py"),
                 "--host", "192.168.1.26", "--port", "50053", "--opcode", "17", "--output", str(out_json)]
    t_probe = time.time()
    probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, encoding="utf-8")
    log(f"Probe 17 completed in {time.time()-t_probe:.2f}s!")
    log(f"Probe output: {probe_res.stdout}")
    if probe_res.stderr:
        log(f"Probe stderr: {probe_res.stderr}")

    # 9. Check 20 cases
    log("10. Evaluating all 20 cases against tolerance...")
    cmd_check = ["python", str(ROOT / "scratch/check_cases.py"), str(out_json)]
    subprocess.run(cmd_check)

    # 10. Performance comparison against baseline .83
    log("11. Performance comparison against baseline .83...")
    subprocess.run(["python", str(ROOT / "scratch/analyze_probe17.py")])

if __name__ == "__main__":
    main()
