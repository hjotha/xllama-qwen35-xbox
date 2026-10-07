#Requires -Version 5.1
# Local xllama UWP build with the Q5_K D3D12 kernel, for the .193 desktop session.
#
# Must run from a logged-on desktop. The only SSH-reachable account
# (hpbeats\hjotha) has no interactive token, and its key store refuses every
# key operation over SSH (New-SelfSignedCertificate NTE_PERM, signtool "no
# certificates", GetRSAPrivateKey "Invalid provider type"). Session 1 belongs
# to hpbeats\hjoth, where the desktop CryptoAPI context works -- the same
# InteractiveToken scheme as the CodexXboxProbe* tasks that built here on 9/26.
#
# Usage (after an RDP/console logon, or via the XllamaMtpBuild task):
#   .\scripts\run-xbox-build.ps1 -BuildRevision 120
param(
    [Parameter(Mandatory)][ValidateRange(1,65535)][int]$BuildRevision,
    [switch]$Final,
    [string]$WorkRoot = "C:\Users\hjotha\build\xllama-q5k"
)

$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$work = $WorkRoot
$sdk  = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
$makeappx = "$sdk\makeappx.exe"
$signtool = "$sdk\signtool.exe"
$log = "$work\build-local.log"

New-Item -ItemType Directory -Force -Path $work | Out-Null
function Log($m) { Add-Content -Path $log -Value ((Get-Date -Format "HH:mm:ss") + " " + $m) }
function Write-Exit($result) {
    Set-Content -Path "$work\exit.tmp" -Value $result
    Move-Item -Path "$work\exit.tmp" -Destination "$work\exit.txt" -Force
}

$buildLock = New-Object System.Threading.Mutex($false, "Local\XllamaXboxBuild")
if (-not $buildLock.WaitOne(0)) { throw "Another xllama Xbox build owns the cache." }
Set-Content -Path $log -Value ("=== START " + (Get-Date -Format s) +
    " session=" + (Get-Process -Id $PID).SessionId + " user=" + (whoami) + " ===")
Set-Location $repo
Remove-Item "$work\exit.txt" -Force -EA SilentlyContinue
$buildStarted = Get-Date
Log ("mode=" + $(if ($Final) { "final" } else { "iteration" }))

# 1. AppContainer guards on the llama.cpp submodule (required by llamacpp/unified).
# A scheduled task has no Python on PATH (the Store alias is a stub), and
# apply-uwp-patches.sh resolves the llama-mmap.cpp conflict with python3.
$pyDir = "C:\Users\hjotha\AppData\Local\Programs\Python\Python314"
if (Test-Path (Join-Path $pyDir "python.exe")) { $env:PATH = "$pyDir;$env:PATH" }
Log "1/4 apply-uwp-patches.sh"
$p1 = Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" `
    -ArgumentList "scripts/apply-uwp-patches.sh" -WorkingDirectory $repo `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$work\patches.out" -RedirectStandardError "$work\patches.err"
Log "patches exit=$($p1.ExitCode)"
if ($p1.ExitCode -ne 0) {
    Log "ABORT no patches"
    Get-Content "$work\patches.err" -Tail 20 -EA SilentlyContinue | ForEach-Object { Log $_ }
    Write-Exit "patches:$($p1.ExitCode)"
    exit 1
}

# 2. Build. -Backend unified compiles src/bridge/ggml_d3d12.cpp (the Q5_K kernel).
# pwsh, not powershell: build-uwp.ps1 tests $IsWindows, which PS 5.1 lacks
# (CI runs pwsh too).
# git safe.directory: the llama.cpp submodule is owned by hpbeats\hjotha but
# this build runs as hpbeats\hjoth. GenerateGgmlVersion.cmake shells out to
# git, which refuses foreign-owned repos ("dubious ownership"). The env form
# is honored without touching any global config.
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "safe.directory"
$env:GIT_CONFIG_VALUE_0 = "*"
# MSVC env: a bare scheduled task has no compiler on PATH (TRK0005).
# Import the VS2022 x64 developer environment, like ilammy/msvc-dev-cmd on CI.
$vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
$vsEnv = cmd /c "`"$vsDevCmd`" -arch=x64 -no_logo >nul 2>&1 && set" 2>$null
foreach ($line in $vsEnv) {
    if ($line -match '^(PATH|INCLUDE|LIB|LIBPATH)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}
Log ("VS dev env: cl.exe na PATH = " + [bool](Get-Command cl.exe -EA SilentlyContinue))
# C1033 guard: the ggml-uwp PDB goes to uwp\x64\Release\ggml-uwp\, which no
# target creates (cl.exe does not create /Fd directories). A directory wiped
# by cleanup -- or never created -- fails every compile with C1033.
foreach ($d in @("uwp\x64\Release\ggml-uwp", "uwp\xllama\x64\Release",
                 "uwp\ggml-uwp\x64\Release")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repo $d) | Out-Null
}
Log "dirs de saida garantidos"
Log "2/4 build-uwp.ps1 -Backend unified -BuildRevision $BuildRevision"
$modeArgs = @()
if (-not $Final) { $modeArgs += "-Iteration" }
$modeArgs += @("-WindowsSdkVersion", (Split-Path (Split-Path $sdk -Parent) -Leaf))
Log ("build flags=" + ($modeArgs -join ' '))
$pwsh = "C:\Users\hjotha\scoop\shims\pwsh.exe"
& $pwsh -ExecutionPolicy Bypass -NoProfile -File "$repo\scripts\build-uwp.ps1" `
    -Configuration Release -Platform x64 -Backend unified -BuildRevision $BuildRevision @modeArgs `
    > "$work\msbuild.out" 2> "$work\msbuild.err"
$buildExit = $LASTEXITCODE
Log "build exit=$buildExit"
Get-Content "$work\msbuild.err" -Tail 25 -EA SilentlyContinue | ForEach-Object { Log ("ERR " + $_) }
if ($buildExit -ne 0) { Log "ABORT no build"; Write-Exit "build:$buildExit"; exit 1 }

$msix = Get-ChildItem "$repo\uwp\AppPackages" -Filter "*.msix" -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $msix -or $msix.LastWriteTime -lt $buildStarted) {
    Log "ABORT: no fresh package was produced"
    Write-Exit "stale-package"
    exit 1
}
Log "msix=$($msix.FullName) size=$($msix.Length)"
Log "sha256=$((Get-FileHash $msix.FullName -Algorithm SHA256).Hash)"

# 3. The MSIX SDK reinjects PackageDependency Microsoft.VCLibs.140.00
#    MinVersion="14.0.33519.0" no matter what uwp/AppxManifest.xml says, and the
#    console's VCLibs is older, so launch dies with 0x80070002 ERROR_FILE_NOT_FOUND.
#    Unpack, lower the MinVersion, repack, re-sign -- the cert build-uwp.ps1 made.
$unpack = "$work\unpacked"
Remove-Item -Recurse -Force $unpack -EA SilentlyContinue
Log "3/4 makeappx unpack"
& $makeappx unpack /p $msix.FullName /d $unpack *>&1 | Select-Object -Last 2 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Write-Exit "unpack:$LASTEXITCODE"; exit 1 }

$mf = Join-Path $unpack "AppxManifest.xml"
$before = ([regex]::Match([System.IO.File]::ReadAllText($mf),
    'Microsoft\.VCLibs\.140\.00" MinVersion="([0-9.]*)"')).Groups[1].Value
# Edit the bytes, not the text: Get-Content|Set-Content round-trips through the
# console codepage and rewrote the em dash (U+2014) in an unrelated comment as
# three bytes, which makeappx then rejected with "Illegal xml character" on a
# line the script never touched. ReadAllText/WriteAllText keeps the document
# intact and the manifest must stay byte-identical apart from MinVersion.
$mfText = [System.IO.File]::ReadAllText($mf, [System.Text.UTF8Encoding]::new($false))
$mfText = $mfText -replace '(Microsoft\.VCLibs\.140\.00" MinVersion=")[0-9.]+(")', '${1}14.0.0.0${2}'
[System.IO.File]::WriteAllText($mf, $mfText, [System.Text.UTF8Encoding]::new($false))
$after = ([regex]::Match([System.IO.File]::ReadAllText($mf),
    'Microsoft\.VCLibs\.140\.00" MinVersion="([0-9.]*)"')).Groups[1].Value
Log "VCLibs MinVersion: $before -> $after"

$fixed = "$work\xllama-q5k-fixed.msix"
Remove-Item $fixed -Force -EA SilentlyContinue
Log "4/4 makeappx pack + signtool sign"
& $makeappx pack /d $unpack /p $fixed /o *>&1 | Select-Object -Last 3 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Write-Exit "pack:$LASTEXITCODE"; exit 1 }

$pfx = "$repo\uwp\xllama-test.pfx"
if (Test-Path $pfx) {
    & $signtool sign /f $pfx /p "xllama-test" /fd sha256 $fixed *>&1 | Select-Object -Last 4 | ForEach-Object { Log $_ }
    $signExit = $LASTEXITCODE
    if ($signExit -ne 0) {
        Log "ABORT signtool sign exit=$signExit"
        Write-Exit "sign:$signExit"
        exit 1
    }
    $verifyOutput = & $signtool verify /pa $fixed *>&1
    $verifyExit = $LASTEXITCODE
    $verifyOutput | Select-Object -Last 8 | ForEach-Object { Log $_ }
    if ($verifyExit -ne 0) {
        $verifyText = ($verifyOutput | Out-String)
        $untrustedRoot = ($verifyText -match 'terminated in a root') -and ($verifyText -match 'certificate which is not trusted by the trust provider')
        $expectedCertPath = Join-Path $repo 'uwp\xllama-test.cer'
        $expectedCert = if (Test-Path $expectedCertPath) { New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($expectedCertPath) } else { $null }
        $auth = Get-AuthenticodeSignature -FilePath $fixed
        $actualThumb = if ($auth.SignerCertificate) { $auth.SignerCertificate.Thumbprint } else { '' }
        $rejectedStatuses = @('HashMismatch', 'NotSigned', 'NotSupportedFileFormat', 'Incompatible')
        $isRejectedStatus = $rejectedStatuses -contains ([string]$auth.Status)
        $hasSigner = ($auth.SignerCertificate -ne $null)
        $thumbMatches = ($expectedCert -ne $null) -and ($actualThumb -eq $expectedCert.Thumbprint) -and ($actualThumb -ne '')
        $statusOk = @('Valid', 'UnknownError', 'NotTrusted') -contains ([string]$auth.Status)

        if (-not $untrustedRoot -or -not $hasSigner -or -not $thumbMatches -or $isRejectedStatus -or -not $statusOk) {
            Log "ABORT signature verify exit=$verifyExit status=$($auth.Status) signer=$actualThumb expected=$(if($expectedCert){$expectedCert.Thumbprint}else{''})"
            Write-Exit "verify:$verifyExit"
            exit 1
        }
        Log "verify warning: untrusted self-signed root on build host; signer matches expected certificate (status=$($auth.Status))"
    } else {
        Log "signtool verify exit=0"
    }
} else {
    Log "ABORT: test certificate missing"
    Write-Exit "missing-certificate"
    exit 1
}
Log "sha256-fixed=$((Get-FileHash $fixed -Algorithm SHA256).Hash)"
Set-Content "$work\fixed-msix-path.txt" $fixed
Log ("elapsed_seconds=" + [math]::Round(((Get-Date)-$buildStarted).TotalSeconds,2))
Log "=== DONE ==="
Write-Exit "0"
