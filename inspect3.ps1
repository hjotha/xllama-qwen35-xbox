Set-Location C:\Users\hjotha\worktrees\xllama-mtp-fast
foreach ($f in @("silu11.asm.txt","silu12.asm.txt")) {
  $c = Get-Content $f
  $fdiv = ($c | Select-String "fdiv double").Count
  $fptrunc = ($c | Select-String "fptrunc").Count
  $fptoui = ($c | Select-String "fptoui").Count
  $fcmp = ($c | Select-String "fcmp").Count
  $fmul = ($c | Select-String "fmul double").Count
  $fadd = ($c | Select-String "fadd double").Count
  $fpext = ($c | Select-String "fpext").Count
  Write-Output ("$f : lines=" + $c.Count + " fdiv=" + $fdiv + " fptrunc=" + $fptrunc + " fptoui=" + $fptoui + " fcmp=" + $fcmp + " fmul_d=" + $fmul + " fadd_d=" + $fadd + " fpext=" + $fpext)
}
Write-Output "=== silu11: all fdiv double lines and 3 lines after ==="
$c = Get-Content silu11.asm.txt
for ($i=0;$i -lt $c.Count;$i++){ if ($c[$i] -match "fdiv double"){ $c[$i..([Math]::Min($i+4,$c.Count-1))] | ForEach-Object { $_ }; Write-Output "---" } }
