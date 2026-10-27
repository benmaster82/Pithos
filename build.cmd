@echo off
rem Build pithos on Windows. Uses gcc (MSYS2 MinGW-w64) if available, otherwise MSVC.
rem   build.cmd          -> build\pithos.exe
rem   build.cmd msvc     -> force MSVC
rem   build.cmd test     -> also build the test programs
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not exist build mkdir build

set SRC=src\main.c src\model.c src\gguf.c src\tokenizer.c src\sampler.c src\quant.c src\platform.c src\quant_neon.c src\gpu_opencl.c src\json.c src\pcache.c src\server.c src\proc.c src\tools.c
set WANT=%1

if /i "%WANT%"=="msvc" goto msvc
where gcc >nul 2>nul && goto gcc
if exist C:\msys64\mingw64\bin\gcc.exe (
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
    goto gcc
)
if exist C:\msys64\ucrt64\bin\gcc.exe (
    set "PATH=C:\msys64\ucrt64\bin;%PATH%"
    goto gcc
)
goto msvc

:gcc
echo [pithos] compilatore: gcc
set CFLAGS=-O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
gcc %CFLAGS% -mavx2 -mfma -mf16c -c src\quant_avx2.c -o build\quant_avx2.o || goto fail
gcc %CFLAGS% -municode %SRC% build\quant_avx2.o -o build\pithos.exe -lm -lws2_32 || goto fail
if /i "%WANT%"=="test" (
    gcc %CFLAGS% tests\test_kernels.c src\gguf.c src\quant.c src\platform.c build\quant_avx2.o -o build\test_kernels.exe -lm || goto fail
    gcc %CFLAGS% tests\test_gpu.c src\model.c src\gguf.c src\quant.c src\platform.c src\gpu_opencl.c build\quant_avx2.o -o build\test_gpu.exe -lm || goto fail
)
goto ok

:msvc
where cl >nul 2>nul && goto msvc_ready
set "VSINST=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
if not exist "%VSINST%\vswhere.exe" goto nocompiler
for /f "delims=" %%i in ('"%VSINST%\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VSDIR=%%i"
if not defined VSDIR goto nocompiler
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul || goto nocompiler
:msvc_ready
echo [pithos] compilatore: MSVC
set CLFLAGS=/nologo /O2 /Oi /GL /std:c11 /W3 /D_CRT_SECURE_NO_WARNINGS
cl %CLFLAGS% /arch:AVX2 /c src\quant_avx2.c /Fobuild\quant_avx2.obj || goto fail
cl %CLFLAGS% %SRC% build\quant_avx2.obj /Fobuild\ /Febuild\pithos.exe /link /LTCG || goto fail
if /i "%2"=="test" (
    cl %CLFLAGS% tests\test_kernels.c src\gguf.c src\quant.c src\platform.c build\quant_avx2.obj /Fobuild\ /Febuild\test_kernels.exe /link /LTCG || goto fail
)
goto ok

:nocompiler
echo [pithos] nessun compilatore C trovato: installa MSYS2 (pacman -S mingw-w64-x86_64-gcc) o Visual Studio Build Tools.
exit /b 1
:fail
echo [pithos] compilazione fallita
exit /b 1
:ok
echo [pithos] ok: build\pithos.exe
exit /b 0
