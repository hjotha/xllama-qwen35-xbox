$ErrorActionPreference = 'Stop'
$wt = Join-Path $env:USERPROFILE 'worktrees\xllama-dflash-xbox'
$proj = Join-Path $wt 'uwp\dflash-rpc'
$manifest = Join-Path $proj 'Package.appxmanifest'
[xml]$xml = Get-Content $manifest
$ver = $xml.Package.Identity.Version
Write-Output "MANIFEST_VERSION=$ver"
if ($ver -eq '1.0.0.89') {
  $txt = [IO.File]::ReadAllText($manifest)
  $txt = $txt.Replace('Version="1.0.0.89"', 'Version="1.0.0.90"')
  [IO.File]::WriteAllText($manifest, $txt)
  Write-Output 'BUMPED_TO_1.0.0.90'
} elseif ($ver -ne '1.0.0.90') {
  Write-Output "UNEXPECTED_VERSION=$ver"
  exit 2
} else {
  Write-Output 'ALREADY_1.0.0.90'
}
$msbuild = 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\msbuild.exe'
$vcx = Join-Path $proj 'DFlashRpc.vcxproj'
$log = Join-Path $proj 'build-90.log'
& $msbuild $vcx /p:Configuration=Release /p:Platform=x64 /m /v:minimal /fl "/flp:logfile=$log;verbosity=normal"
Write-Output "MSBUILD_EXIT=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit 1 }
$msix = Join-Path $proj 'AppPackages\DFlashRpc\DFlashRpc_1.0.0.90_x64_Test\DFlashRpc_1.0.0.90_x64.msix'
if (!(Test-Path $msix)) { Write-Output 'MSIX_NOT_FOUND'; exit 1 }
$h = Get-FileHash $msix -Algorithm SHA256
$len = (Get-Item $msix).Length
Write-Output "MSIX_OK len=$len sha256=$($h.Hash)"
Write-Output "MSIX_PATH=$msix"
