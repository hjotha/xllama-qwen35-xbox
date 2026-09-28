@echo off
setlocal
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
pushd "%~dp0..\..\.."
cl /nologo /W4 /WX /EHsc /std:c++17 /utf-8 /DNOMINMAX ^
 /Iuwp\dflash-rpc /Fo"%TEMP%\dflash-draft-sha256-test.obj" ^
 /Fe"%TEMP%\dflash-draft-sha256-test.exe" ^
 research\dflash2-xbox\gpu-implementation\draft_test_sha256.cpp
if errorlevel 1 (set "RESULT=%errorlevel%") else (
 "%TEMP%\dflash-draft-sha256-test.exe" %*
 set "RESULT=%errorlevel%"
)
popd
exit /b %RESULT%
