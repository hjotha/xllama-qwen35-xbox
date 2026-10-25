@echo off
set URL=https://192.168.1.26:11443
set DIR=%USERPROFILE%\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\AppPackages\DFlashRpc\DFlashRpc_1.0.0.90_x64_Test
set MSIX=%DIR%\DFlashRpc_1.0.0.90_x64.msix
curl.exe -k -s -m 30 -D "%TEMP%\xbox-hdrs90.txt" -o NUL %URL%/
set CSRF=
for /f "tokens=2 delims==;" %%a in ('findstr /i "CSRF-Token" "%TEMP%\xbox-hdrs90.txt"') do set CSRF=%%a
if "%CSRF%"=="" echo NO_CSRF_TOKEN & exit /b 2
set DEPS=
for %%f in ("%DIR%\Dependencies\*.appx") do call set DEPS=%%DEPS%% -F "file=@%%f;type=application/octet-stream"
curl.exe -k -s -m 600 -H "X-CSRF-Token: %CSRF%" -X POST -F "file=@%MSIX%;type=application/octet-stream" %DEPS% "%URL%/api/app/packagemanager/package?package=DFlashRpc_1.0.0.90_x64.msix"
echo.
