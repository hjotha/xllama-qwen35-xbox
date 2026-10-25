param([int]$BuildRevision,[switch]$Final,[string]$WorkRoot)
$ErrorActionPreference='Stop'
$w=$WorkRoot;$log="$w\build-local.log"
function Log($s){Add-Content $log ((Get-Date).ToUniversalTime().ToString('o')+' '+$s)}
function Done($s){Set-Content "$w\exit.tmp" $s;Move-Item "$w\exit.tmp" "$w\exit.txt" -Force}
Set-Content $log 'Final247 packaging-only; compiled artifact preserved'
try{
 if($BuildRevision -ne 247 -or -not $Final){throw 'Wrong mode/version'}
 $old='C:\Users\hjotha\tmp\xllama-rev246-crt\xllama_1.6.0.246_x64.msix'
 if((Get-FileHash $old).Hash -ne '1536B30DD226DFEE21BE097FF2C475D1809D1C1C1DF016699B67C29964071ECF'){throw 'Source package mismatch'}
 Add-Type -AssemblyName System.IO.Compression.FileSystem
 $zip=[IO.Compression.ZipFile]::OpenRead($old)
 try{
  foreach($dll in @('MSVCP140.dll','MSVCP140_1.dll','VCRUNTIME140.dll','VCRUNTIME140_1.dll')){
   $e=$zip.GetEntry($dll);if(-not $e){throw "Missing approved CRT $dll"}
   $dest="$w\unpacked\$dll"
   [IO.Compression.ZipFileExtensions]::ExtractToFile($e,$dest,$true)
   Log ("CRT "+$dll+" SHA256="+(Get-FileHash $dest).Hash)
  }
 }finally{$zip.Dispose()}
 [xml]$m=Get-Content "$w\unpacked\AppxManifest.xml"
 if($m.Package.Identity.Version -ne '1.6.0.247' -or $m.Package.Identity.Publisher -ne 'CN=xllama-dev'){throw 'Manifest mismatch'}
 Log ("EXE SHA256="+(Get-FileHash "$w\unpacked\xllama.exe").Hash)
 $sdk='C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64'
 $fixed="$w\xllama_1.6.0.247_x64.msix"
 if(Test-Path $fixed){throw 'Package already exists; inspect rather than overwrite'}
 & "$sdk\makeappx.exe" pack /d "$w\unpacked" /p $fixed /o *>&1|ForEach-Object {Log $_}
 if($LASTEXITCODE -ne 0){throw "pack:$LASTEXITCODE"}
 $thumb='21E98358E40980181C8795919FE3C7D455E2B7FC'
 $cert=Get-Item ("Cert:\CurrentUser\My\"+$thumb)
 if(-not $cert.HasPrivateKey -or $cert.Subject -ne 'CN=xllama-dev'){throw 'Existing certificate unavailable'}
 & "$sdk\signtool.exe" sign /sha1 $thumb /s My /fd sha256 $fixed *>&1|ForEach-Object {Log $_}
 if($LASTEXITCODE -ne 0){throw "sign:$LASTEXITCODE"}
 $a=Get-AuthenticodeSignature $fixed
 Log ("signature status="+$a.Status+" signer="+$a.SignerCertificate.Thumbprint)
 if($a.SignerCertificate.Thumbprint -ne $thumb -or @('Valid','UnknownError','NotTrusted') -notcontains [string]$a.Status){throw 'Signature mismatch'}
 Log ("SIGNED SHA256="+(Get-FileHash $fixed).Hash)
 Set-Content "$w\fixed-msix-path.txt" $fixed
 Done '0'
}catch{Log ("ERROR "+$_);Done 'package-failed';exit 1}
