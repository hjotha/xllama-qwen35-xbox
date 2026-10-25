@echo off
set URL=https://192.168.1.26:11443
set CER=%USERPROFILE%\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\hjotha-xbox-2026.cer
curl.exe -k -s -m 30 -D "%TEMP%\xbox-hdrs90c.txt" -o NUL %URL%/
set CSRF=
for /f "tokens=2 delims==;" %%a in ('findstr /i "CSRF-Token" "%TEMP%\xbox-hdrs90c.txt"') do set CSRF=%%a
if "%CSRF%"=="" echo NO_CSRF_TOKEN & exit /b 2
curl.exe -k -s -m 120 -H "X-CSRF-Token: %CSRF%" -X POST -F "file=@%CER%;type=application/octet-stream" "%URL%/api/app/packagemanager/certificate?package=hjotha-xbox-2026.cer"
echo.
