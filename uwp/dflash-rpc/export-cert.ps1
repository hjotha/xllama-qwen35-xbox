$ErrorActionPreference = 'Stop'
$c = Get-ChildItem Cert:/CurrentUser/My | Where-Object Subject -like '*Hjotha*' | Select-Object -First 1
$pwd = ConvertTo-SecureString -String 'xbox90sign' -Force -AsPlainText
try {
  Export-PfxCertificate -Cert $c -FilePath "$env:TEMP\hjotha-xbox.pfx" -Password $pwd -Force | Out-Null
  Write-Output 'PFX_EXPORT_OK'
} catch {
  Write-Output ('PFX_EXPORT_FAIL: ' + $_.Exception.Message)
}
try {
  Export-Certificate -Cert $c -FilePath "$env:TEMP\hjotha-xbox.cer" -Force | Out-Null
  Write-Output 'CER_EXPORT_OK'
} catch {
  Write-Output ('CER_EXPORT_FAIL: ' + $_.Exception.Message)
}
