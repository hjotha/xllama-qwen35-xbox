@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\msbuild.exe" %USERPROFILE%\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\DFlashRpc.vcxproj /p:Configuration=Release /p:Platform=x64 /m /v:minimal /fl "/flp:logfile=%USERPROFILE%\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\build-90.log;verbosity=normal"
echo MSBUILD_EXIT=%errorlevel%
exit /b %errorlevel%
