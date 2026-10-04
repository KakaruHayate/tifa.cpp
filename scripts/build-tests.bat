@echo on
call "%~dp0vcvars.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0.."
cmake --build build-tests --config Release
echo BUILDRC=%ERRORLEVEL%
