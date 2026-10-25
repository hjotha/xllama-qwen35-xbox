import base64
import subprocess
import time

def main():
    pkg = "HJotha.XboxDFlashRpc_1.0.0.82_x64__3hvcqmcd1t14m"
    pkg64 = base64.b64encode(pkg.encode()).decode()
    subprocess.run(["python", r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\xbox_portal.py",
                    f"/api/taskmanager/app?package={pkg64}", "--method", "DELETE"])

    pkg_new = "HJotha.XboxDFlashRpc_1.0.0.83_x64__3hvcqmcd1t14m"
    pkg64_new = base64.b64encode(pkg_new.encode()).decode()
    subprocess.run(["python", r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\xbox_portal.py",
                    f"/api/taskmanager/app?package={pkg64_new}", "--method", "DELETE"])

    time.sleep(2)
    aumid = "HJotha.XboxDFlashRpc_3hvcqmcd1t14m!App"
    aumid64 = base64.b64encode(aumid.encode()).decode()
    print("Starting 1.0.0.83...")
    subprocess.run(["python", r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\research\dflash2-xbox\xbox_portal.py",
                    f"/api/taskmanager/app?appid={aumid64}", "--method", "POST"])
    print("Done!")

if __name__ == "__main__":
    main()
