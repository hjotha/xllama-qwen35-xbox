$dxc = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe"
Set-Location C:\Users\hjotha\worktrees\xllama-mtp-fast\shaders
Remove-Item silu.dxil,swiglu_probe.dxil,silu-compile.log,swiglu-compile.log -Force -ErrorAction SilentlyContinue
& $dxc -T cs_6_2 -E main -Gis -Od -denorm preserve -Fo silu.dxil ggml_d3d12_silu.hlsl *> silu-compile.log
Write-Output ("s=" + $LASTEXITCODE)
& $dxc -T cs_6_2 -E main -Gis -Od -denorm preserve -Fo swiglu_probe.dxil ggml_d3d12_swiglu_probe.hlsl *> swiglu-compile.log
Write-Output ("w=" + $LASTEXITCODE)
Write-Output "--- silu messages ---"
Select-String -Path silu-compile.log -Pattern "error|warning" | Select-Object -First 8 | ForEach-Object { $_.Line }
Write-Output "--- swiglu messages ---"
Select-String -Path swiglu-compile.log -Pattern "error|warning" | Select-Object -First 8 | ForEach-Object { $_.Line }
Get-ChildItem silu.dxil,swiglu_probe.dxil -ErrorAction SilentlyContinue | ForEach-Object { $_.Name + " " + $_.Length }
