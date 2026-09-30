@echo off
setlocal
set "VSROOT=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
"%VSROOT%\MSBuild\Current\Bin\MSBuild.exe" "%~dp0..\..\uwp\dflash-rpc\DFlashRpc.vcxproj" /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143 /p:VCToolsInstallDir=C:\BuildTools\VC\Tools\MSVC\14.44.35207\ /p:VCToolsVersion=14.44.35207 /p:WindowsTargetPlatformVersion=10.0.26100.0 /p:AppxBundle=Never /m:1 /v:minimal %*
exit /b %errorlevel%
