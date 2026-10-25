Set-Location C:\Users\hjotha\worktrees\xllama-mtp-fast
$lines = Get-Content silu12.asm.txt
Write-Output ("total lines: " + $lines.Count)
$idx = 0
foreach ($l in $lines) {
  if ($l -match "8\.388608|713623846352979|2\.8147|1\.175494|fptoui|fptrunc double") {
    Write-Output ("[" + $idx + "] " + $l.Trim())
  }
  $idx++
}
