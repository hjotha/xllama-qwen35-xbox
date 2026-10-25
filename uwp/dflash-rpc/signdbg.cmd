@echo off
set D=%USERPROFILE%\worktrees\xllama-dflash-xbox\uwp\dflash-rpc
set MSIX=%D%\AppPackages\DFlashRpc\DFlashRpc_1.0.0.90_x64_Test\DFlashRpc_1.0.0.90_x64.msix
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe" sign /v /debug /fd SHA256 /f "%D%\hjotha-xbox-2026.pfx" /p xbox90sign "%MSIX%"
echo SIGN_EXIT=%errorlevel%
