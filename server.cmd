@echo off
rem Start the pithos server with the web chat and open it in the browser.
rem Extra options are passed through, e.g.  server.cmd --port 8095 --no-gpu
cd /d "%~dp0"
if not exist build\pithos.exe call build.cmd || exit /b 1
build\pithos.exe --server --open %*
