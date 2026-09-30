@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0.."
cmake -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIFA_GGML_BUILD_TESTS=ON -DTIFA_GGML_BUILD_CLI=OFF
echo CMRC=%ERRORLEVEL%
