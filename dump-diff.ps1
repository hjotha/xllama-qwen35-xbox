$dxc = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe"
Set-Location C:\Users\hjotha\worktrees\xllama-mtp-fast
& $dxc -dumpbin silu11.dxil > silu11.asm.txt 2>&1
& $dxc -dumpbin silu12.dxil > silu12.asm.txt 2>&1
Write-Output ("sizes: " + (Get-Item silu11.asm.txt).Length + " / " + (Get-Item silu12.asm.txt).Length)
$d = Compare-Object (Get-Content silu11.asm.txt) (Get-Content silu12.asm.txt)
Write-Output ("diff lines: " + $d.Count)
$d | Select-Object -First 80 | ForEach-Object { $_.SideIndicator + " " + $_.InputObject }
