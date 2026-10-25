$ErrorActionPreference = 'Stop'
$p = Join-Path $env:USERPROFILE 'worktrees\xllama-dflash-xbox\uwp\ggml-dflash-uwp.vcxproj'
$t = [IO.File]::ReadAllText($p)
if ($t.Contains('10.0.22621.0')) {
  $t = $t.Replace('10.0.22621.0', '10.0.26100.0')
  [IO.File]::WriteAllText($p, $t)
  Write-Output 'RETARGETED_TO_26100'
} else {
  Write-Output 'NO_22621_FOUND'
}
