$ErrorActionPreference = 'Stop'
$wt = Join-Path $env:USERPROFILE 'worktrees\xllama-dflash-xbox'
$proj = Join-Path $wt 'uwp\dflash-rpc'
$c = New-SelfSignedCertificate -Subject 'CN=Hjotha Xbox Probe' -Type CodeSigningCert -KeyExportPolicy Exportable -KeyLength 2048 -KeyAlgorithm RSA -HashAlgorithm SHA256 -NotAfter (Get-Date).AddYears(2) -CertStoreLocation Cert:\CurrentUser\My -FriendlyName 'Hjotha Xbox Probe 2026-10'
Write-Output ('NEW_THUMBPRINT=' + $c.Thumbprint)
$pwd = ConvertTo-SecureString -String 'xbox90sign' -Force -AsPlainText
$cerPath = Join-Path $proj 'hjotha-xbox-2026.cer'
Export-Certificate -Cert $c -FilePath $cerPath -Force | Out-Null
$pfxPath = Join-Path $env:TEMP 'hjotha-xbox-2026.pfx'
Export-PfxCertificate -Cert $c -FilePath $pfxPath -Password $pwd -Force | Out-Null
Write-Output 'EXPORT_OK'
$msix = Join-Path $proj 'AppPackages\DFlashRpc\DFlashRpc_1.0.0.90_x64_Test\DFlashRpc_1.0.0.90_x64.msix'
& 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe' sign /fd SHA256 /f $pfxPath /p xbox90sign $msix
Write-Output ('SIGN_EXIT=' + $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) { exit 1 }
& 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe' verify /pa $msix
Write-Output ('VERIFY_EXIT=' + $LASTEXITCODE)
