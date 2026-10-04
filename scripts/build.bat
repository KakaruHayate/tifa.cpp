@echo on
call "%~dp0vcvars.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0.."
cmake --build build --config Release --target tifa_ggml_cli
echo BUILDRC=%ERRORLEVEL%
