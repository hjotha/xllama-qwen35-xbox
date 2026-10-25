@echo off
setlocal
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
pushd "%~dp0..\..\.."
cl /nologo /Zs /W4 /WX /EHsc /std:c++17 /utf-8 /bigobj /arch:AVX2 ^
 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS ^
 /DGGML_USE_CPU /DGGML_STATIC /DGGML_SCHED_MAX_COPIES=4 ^
 /Iinclude /Ishaders\generated /Illama.cpp\include /Illama.cpp\src ^
 /Illama.cpp\ggml\include /Illama.cpp\ggml\src /Iuwp\generated ^
 uwp\dflash-rpc\dflash-gpu-engine.cpp ^
 uwp\dflash-rpc\dflash-gpu-draft-engine.cpp ^
 uwp\dflash-rpc\dflash-gpu-draft-probe.cpp
set "RESULT=%errorlevel%"
popd
exit /b %RESULT%
