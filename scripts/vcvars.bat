@echo off
rem Enter the x64 MSVC developer environment.  Visual Studio is located through
rem vswhere (installed with every VS/Build Tools) so no machine-specific install
rem path is baked into the scripts.  Must run without setlocal: vcvars64.bat has
rem to modify the caller's environment.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [vcvars] vswhere not found at "%VSWHERE%"; install Visual Studio Build Tools. 1>&2
  exit /b 1
)
set "VSDIR="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo [vcvars] no Visual Studio install with the C++ toolset was found. 1>&2
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat"
exit /b %ERRORLEVEL%
