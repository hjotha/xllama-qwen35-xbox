#Requires -Version 5.1
# Runs build-uwp.ps1 (unified backend, so the D3D12 matmul kernel is compiled in)
# from the interactive desktop session, where CryptoAPI can create the key
# container for the self-signed test certificate. An SSH session cannot.
$repo = "C:\Users\hjotha\projects\xllama-qbox"
$log = "$repo\build-q5k.log"

function Log($m) {
    Add-Content -Path $log -Value ((Get-Date -Format "HH:mm:ss") + " " + $m)
}

Set-Content -Path $log -Value ("=== START " + (Get-Date -Format s) +
    " session=" + (Get-Process -Id $PID).SessionId +
    " user=" + (whoami) + " ===")

Set-Location $repo

Log "step 1/3: apply-uwp-patches.sh"
$p1 = Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" `
    -ArgumentList "scripts/apply-uwp-patches.sh" -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$repo\patches.out" -RedirectStandardError "$repo\patches.err"
Log "patches exit=$($p1.ExitCode)"
if ($p1.ExitCode -ne 0) {
    Log "ABORT: patches failed"
    Get-Content "$repo\patches.err" -Tail 40 | ForEach-Object { Log $_ }
    Set-Content -Path "$repo\build-q5k.exit" -Value "patches:$($p1.ExitCode)"
    exit 1
}

Log "step 2/3: build-uwp.ps1 -Backend unified -BuildRevision 1"
$p2 = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-ExecutionPolicy", "Bypass", "-NoProfile", "-File",
    "$repo\scripts\build-uwp.ps1", "-Configuration", "Release", "-Platform", "x64",
    "-Backend", "unified", "-BuildRevision", "1" `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$repo\msbuild.out" -RedirectStandardError "$repo\msbuild.err"
Log "build-uwp exit=$($p2.ExitCode)"

Log "step 3/3: package inventory"
Get-ChildItem "$repo\uwp\AppPackages" -Filter "*.msix" -Recurse -EA SilentlyContinue |
    ForEach-Object { Log ("MSIX " + $_.FullName + " " + $_.Length + " " + $_.LastWriteTime) }
if ($p2.ExitCode -eq 0) {
    $m = Get-ChildItem "$repo\uwp\AppPackages" -Filter "*.msix" -Recurse -EA SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($m) {
        Log "sha256=" + (Get-FileHash $m.FullName -Algorithm SHA256).Hash
        Set-Content -Path "$repo\build-q5k.msix" -Value $m.FullName
    }
}
Get-Content "$repo\msbuild.err" -Tail 40 -EA SilentlyContinue | ForEach-Object { Log ("ERR " + $_) }

Set-Content -Path "$repo\build-q5k.exit" -Value "$($p2.ExitCode)"
Log "=== DONE ==="
