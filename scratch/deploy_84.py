import base64
import json
import subprocess
import time
import socket

def run_portal(route, method="GET", upload=None):
    cmd = ["python", r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\xbox_portal.py",
           route, "--method", method]
    if upload:
        cmd.extend(["--upload", upload])
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

def main():
    print("1. Stopping running instances...")
    for ver in ["1.0.0.82", "1.0.0.83", "1.0.0.84"]:
        pkg = f"HJotha.XboxDFlashRpc_{ver}_x64__3hvcqmcd1t14m"
        pkg64 = base64.b64encode(pkg.encode()).decode()
        run_portal(f"/api/taskmanager/app?package={pkg64}", "DELETE")

    time.sleep(2)
    print("2. Uploading and installing 1.0.0.84...")
    msix = r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\AppPackages\DFlashRpc\DFlashRpc_1.0.0.84_x64_Test\DFlashRpc_1.0.0.84_x64.msix"
    out = run_portal("/api/app/packagemanager/package?package=DFlashRpc_1.0.0.84_x64.msix", "POST", upload=msix)
    print("Install response:", out)

    print("3. Waiting for package manager state to clear...")
    for _ in range(30):
        time.sleep(2)
        state = run_portal("/api/app/packagemanager/state")
        if not state:
            print("Install complete!")
            break
        print("State:", state)

    time.sleep(2)
    print("4. Launching 1.0.0.84...")
    aumid = "HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App"
    aumid64 = base64.b64encode(aumid.encode()).decode()
    start_out = run_portal(f"/api/taskmanager/app?appid={aumid64}", "POST")
    print("Launch response:", start_out)

    print("5. Waiting for port 50053 to listen...")
    for i in range(30):
        time.sleep(1)
        try:
            with socket.create_connection(("192.168.1.26", 50053), timeout=1):
                print("Port 50053 is UP and listening!")
                return
        except Exception:
            pass
    print("WARNING: Timed out waiting for port 50053")

if __name__ == "__main__":
    main()
