@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0.."
cmake --build build --config Release --target tifa_ggml_cli
echo BUILDRC=%ERRORLEVEL%
