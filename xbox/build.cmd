@echo off
rem Build the Xbox (UWP) package: build-xbox\Pithos.appx, signed with build-xbox\pithos_xbox.pfx.
rem Needs Visual Studio 2022 with "MSVC v143" and "C++ Universal Windows Platform tools",
rem and the Windows SDK 10.0.26100. Run xbox\make_cert.ps1 once to create the signing certificate.
setlocal
cd /d "%~dp0.."
set "VS=C:\Program Files\Microsoft Visual Studio\2022\Community"
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2^>nul`) do set "VS=%%i"
call "%VS%\VC\Auxiliary\Build\vcvarsall.bat" x64 uwp 10.0.26100.0 >nul 2>nul || goto nocompiler
where cl >nul 2>nul || goto nocompiler

set OBJ=build-xbox\uwp
set LAYOUT=build-xbox\layout
if not exist %OBJ% mkdir %OBJ%
if exist %LAYOUT% rmdir /s /q %LAYOUT%
mkdir %LAYOUT%\Assets

set CFLAGS=/nologo /c /O2 /MD /std:c11 /W3 /D_CRT_SECURE_NO_WARNINGS /DWINAPI_FAMILY=WINAPI_FAMILY_APP /DPITHOS_UWP
echo [xbox] compiling the engine
cl %CFLAGS% src\main.c src\model.c src\gguf.c src\tokenizer.c src\sampler.c src\quant.c src\platform.c src\quant_neon.c src\gpu_opencl.c src\gpu_d3d12.c src\gpu_vulkan.c src\json.c src\pcache.c src\server.c src\proc.c src\tools.c src\agent.c src\audio.c src\speech.c src\mp3.c /Fo%OBJ%\ || goto fail
cl %CFLAGS% /arch:AVX src\quant_avx.c /Fo%OBJ%\ || goto fail
cl %CFLAGS% /arch:AVX2 src\quant_avx2.c /Fo%OBJ%\ || goto fail
echo [xbox] compiling the GPU probe shaders
fxc /nologo /O3 /T cs_5_0 /E flops /Vn g_flops50 /Fh %OBJ%\cs_flops50.h xbox\gpu_probe.hlsl >nul || goto fail
fxc /nologo /O3 /T cs_5_0 /E copy /Vn g_copy50 /Fh %OBJ%\cs_copy50.h xbox\gpu_probe.hlsl >nul || goto fail
fxc /nologo /O3 /T cs_4_0 /E flops /Vn g_flops40 /Fh %OBJ%\cs_flops40.h xbox\gpu_probe.hlsl >nul || goto fail
fxc /nologo /O3 /T cs_4_0 /E copy /Vn g_copy40 /Fh %OBJ%\cs_copy40.h xbox\gpu_probe.hlsl >nul || goto fail
(echo #include "cs_flops50.h"& echo #include "cs_copy50.h"& echo #include "cs_flops40.h"& echo #include "cs_copy40.h") > %OBJ%\gpu_probe_cs.h
echo [xbox] compiling the app shell
cl /nologo /c /O2 /MD /std:c++20 /EHsc /bigobj /DWINAPI_FAMILY=WINAPI_FAMILY_APP /I "%WindowsSdkDir%Include\%WindowsSDKVersion%cppwinrt" /I %OBJ% xbox\app.cpp xbox\gpu_probe.cpp /Fo%OBJ%\ || goto fail
link /nologo /APPCONTAINER /SUBSYSTEM:WINDOWS /OUT:%LAYOUT%\pithos_xbox.exe %OBJ%\*.obj WindowsApp.lib || goto fail

copy /y xbox\Package.appxmanifest %LAYOUT%\AppxManifest.xml >nul
copy /y xbox\Assets\*.png %LAYOUT%\Assets\ >nul
copy /y web\brand\pandora-mark.svg %LAYOUT%\Assets\pithos-mark.svg >nul
echo [xbox] packaging
makeappx pack /o /h SHA256 /d %LAYOUT% /p build-xbox\Pithos.appx >nul || goto fail
if not exist build-xbox\pithos_xbox.pfx (
    echo [xbox] build-xbox\pithos_xbox.pfx is missing: run  powershell -ExecutionPolicy Bypass -File xbox\make_cert.ps1
    goto fail
)
signtool sign /q /fd SHA256 /f build-xbox\pithos_xbox.pfx /p pithos build-xbox\Pithos.appx || goto fail
echo [xbox] OK: build-xbox\Pithos.appx
exit /b 0

:nocompiler
echo [xbox] Visual Studio 2022 with the C++ UWP tools was not found
exit /b 1
:fail
echo [xbox] build failed
exit /b 1
