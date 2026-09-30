@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
echo VCVARS=%ERRORLEVEL%
cd /d "%~dp0.."
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIFA_GGML_VULKAN=ON -DTIFA_GGML_BUILD_CLI=ON -DTIFA_GGML_BUILD_TESTS=OFF -Wno-dev
echo CMRC=%ERRORLEVEL%
rem Tests build (GoogleTest via FetchContent):
rem   cmake -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIFA_GGML_BUILD_TESTS=ON
