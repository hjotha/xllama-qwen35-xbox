import subprocess
import json
import time
from pathlib import Path

ROOT = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox")
PORTAL = ROOT / r"research\dflash2-xbox\xbox_portal.py"
MSIX = ROOT / r"uwp\dflash-rpc\AppPackages\DFlashRpc\DFlashRpc_1.0.0.84_x64_Test\DFlashRpc_1.0.0.84_x64.msix"

def portal(route, method="GET", upload=None, timeout="180"):
    cmd = ["python", str(PORTAL), route, "--method", method, "--timeout", str(timeout)]
    if upload:
        cmd.extend(["--upload", str(upload)])
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

def main():
    print(f"1. Setting DefaultUWPContentTypeToGame to True...", flush=True)
    res_set = portal("/ext/settings/DefaultUWPContentTypeToGame", "PUT", timeout="15")
    cmd = ["python", str(PORTAL), "/ext/settings/DefaultUWPContentTypeToGame", "--method", "PUT",
           "--json-file", str(ROOT / "research/dflash2-xbox/enable-game-default.json")]
    subprocess.run(cmd, capture_output=True)

    print(f"2. Uploading and installing {MSIX.name} ({MSIX.stat().st_size / (1024*1024):.2f} MB)...", flush=True)
    t0 = time.time()
    inst = portal(f"/api/app/packagemanager/package?package={MSIX.name}", "POST", upload=str(MSIX), timeout="180")
    print(f"Install response ({time.time()-t0:.2f}s): {inst}", flush=True)

    print("3. Waiting for deployment to complete...", flush=True)
    for i in range(40):
        time.sleep(2)
        st = portal("/api/app/packagemanager/state")
        print(f"[{i:02d}] State: {st}", flush=True)
        if '"Success": true' in st or '"Success":true' in st:
            print(">>> INSTALLATION CONFIRMED SUCCESSFUL! <<<", flush=True)
            break
        elif '"Success": false' in st or '"Success":false' in st:
            print(">>> INSTALLATION FAILED! <<<", flush=True)
            return

    # Check installed packages
    pkgs = portal("/api/app/packagemanager/packages")
    try:
        data = json.loads(pkgs)
        found = [p for p in data["data"]["InstalledPackages"] if "DFlashRpc" in p.get("PackageFullName", "")]
        print(f"Installed DFlashRpc packages: {len(found)}")
        for p in found:
            print(f" - {p['PackageFullName']} | Users: {p.get('RegisteredUsers')}")
    except Exception as e:
        print("Error parsing packages:", e)

if __name__ == "__main__":
    main()
