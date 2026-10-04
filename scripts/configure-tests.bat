@echo on
call "%~dp0vcvars.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0.."
cmake -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIFA_GGML_BUILD_TESTS=ON -DTIFA_GGML_BUILD_CLI=OFF
echo CMRC=%ERRORLEVEL%
