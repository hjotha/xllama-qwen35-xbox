$dxc = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe"
Set-Location C:\Users\hjotha\worktrees\xllama-mtp-fast\shaders
& $dxc -T cs_6_2 -E main -Gis -Od -denorm preserve -Fo silu13.dxil ggml_d3d12_silu_noguard.hlsl *> silu13-compile.log
Write-Output ("s=" + $LASTEXITCODE)
Write-Output ("silu13 size: " + (Get-Item silu13.dxil).Length)
Write-Output ("silu11 size: " + (Get-Item C:\Users\hjotha\worktrees\xllama-mtp-fast\silu11.dxil).Length)
$h11 = (Get-FileHash C:\Users\hjotha\worktrees\xllama-mtp-fast\silu11.dxil -Algorithm SHA256).Hash
$h13 = (Get-FileHash silu13.dxil -Algorithm SHA256).Hash
Write-Output ("silu11 sha256: " + $h11)
Write-Output ("silu13 sha256: " + $h13)
Write-Output ("identical: " + ($h11 -eq $h13))
