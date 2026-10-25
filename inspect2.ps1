$dxc = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe"
Set-Location C:\Users\hjotha\worktrees\xllama-mtp-fast
& $dxc -dumpbin silu11.dxil > silu11.asm.txt 2>&1
$l11 = Get-Content silu11.asm.txt
$l12 = Get-Content silu12.asm.txt
Write-Output "=== silu11 (rev203) first fptoui context ==="
for ($i=0; $i -lt $l11.Count; $i++) { if ($l11[$i] -match "fptoui") { $s=[Math]::Max(0,$i-14); $e=[Math]::Min($l11.Count-1,$i+20); $l11[$s..$e] | ForEach-Object { $_ }; break } }
Write-Output "=== silu12 (rev204) first fptoui context ==="
for ($i=0; $i -lt $l12.Count; $i++) { if ($l12[$i] -match "fptoui") { $s=[Math]::Max(0,$i-14); $e=[Math]::Min($l12.Count-1,$i+20); $l12[$s..$e] | ForEach-Object { $_ }; break } }
