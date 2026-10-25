import base64
import json
import subprocess
import time
import socket
from pathlib import Path

def portal(route, method="GET", upload=None):
    cmd = ["python", r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\xbox_portal.py",
           route, "--method", method]
    if upload:
        cmd.extend(["--upload", upload])
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

def main():
    print("1. Stopping running app...")
    pkg = "HJotha.XboxDFlashRpc_1.0.0.84_x64__3hvcqmcd1t14m"
    pkg64 = base64.b64encode(pkg.encode()).decode()
    portal(f"/api/taskmanager/app?package={pkg64}", "DELETE")
    time.sleep(2)

    print("2. Ensuring DefaultUWPContentTypeToGame is True...")
    portal("/ext/settings/DefaultUWPContentTypeToGame", "PUT", upload=None)
    cmd = ["python", r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\xbox_portal.py",
           "/ext/settings/DefaultUWPContentTypeToGame", "--method", "PUT",
           "--json-file", r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\enable-game-default.json"]
    subprocess.run(cmd, capture_output=True)
    setting = json.loads(portal("/ext/settings/DefaultUWPContentTypeToGame"))
    print("Setting:", setting["data"]["Value"])

    print("3. Uninstalling package...")
    del_resp = portal(f"/api/app/packagemanager/package?package={pkg}", "DELETE")
    print("Uninstall response:", del_resp)

    for _ in range(15):
        time.sleep(2)
        st = portal("/api/app/packagemanager/state")
        print("Uninstall state:", st)
        if not st or '"Success": true' in st or '"Success":true' in st:
            break

    print("4. Installing 1.0.0.84 with Game default...")
    msix = r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\AppPackages\DFlashRpc\DFlashRpc_1.0.0.84_x64_Test\DFlashRpc_1.0.0.84_x64.msix"
    inst_resp = portal("/api/app/packagemanager/package?package=DFlashRpc_1.0.0.84_x64.msix", "POST", upload=msix)
    print("Install response:", inst_resp)

    for i in range(30):
        time.sleep(2)
        st = portal("/api/app/packagemanager/state")
        print(f"Install state ({i}):", st)
        if '"Success": true' in st or '"Success":true' in st:
            print("Installation complete!")
            break

    time.sleep(2)
    print("5. Launching...")
    aumid = "HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App"
    aumid64 = base64.b64encode(aumid.encode()).decode()
    launch_resp = portal(f"/api/taskmanager/app?appid={aumid64}", "POST")
    print("Launch response:", launch_resp)

    print("6. Waiting for port 50053...")
    for i in range(25):
        time.sleep(1)
        try:
            with socket.create_connection(("192.168.1.26", 50053), timeout=1):
                print("Port 50053 is OPEN!")
                return
        except Exception:
            pass
    print("Warning: Port 50053 did not open in time")

if __name__ == "__main__":
    main()
