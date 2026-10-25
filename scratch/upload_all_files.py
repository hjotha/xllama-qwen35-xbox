import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox")
PORTAL = ROOT / r"research\dflash2-xbox\xbox_portal.py"
PFN = "HJotha.XboxDFlashRpc_1.0.0.84_x64__3hvcqmcd1t14m"

def upload(local_path, remote_name):
    print(f"Uploading {remote_name} ({Path(local_path).stat().st_size} bytes)...")
    t0 = time.time()
    cmd = ["python", str(PORTAL),
           f"/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname={PFN}&path=%5CLocalState&filename={remote_name}",
           "--method", "POST", "--upload", str(local_path), "--timeout", "180"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(f"Failed to upload {remote_name}:", res.stderr, res.stdout)
        sys.exit(1)
    print(f"Uploaded {remote_name} in {time.time()-t0:.2f}s: {res.stdout.strip()}")

def main():
    upload(r"C:\tmp\dflash-model.txt", "dflash-model.txt")
    upload(r"C:\tmp\dflash-backend.txt", "dflash-backend.txt")
    upload(ROOT / r"research\dflash2-xbox\gpu-implementation\f0-2026-09-28\reference.cap", "reference.cap")
    upload(r"C:\tmp\dflash-q2.gguf", "dflash-q2.gguf")
    print("All 4 files uploaded successfully!")

if __name__ == "__main__":
    main()
