@echo off
setlocal
if not defined VSDEVCMD if defined PIXEL_REFINE_VSDEVCMD set "VSDEVCMD=%PIXEL_REFINE_VSDEVCMD%"
if not defined VSDEVCMD (
  echo ERROR: set VSDEVCMD or PIXEL_REFINE_VSDEVCMD to VsDevCmd.bat.
  exit /b 2
)
call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
pushd "%~dp0..\..\..\test_algorithm\taichi_upstream\stable-v1.7.4-development"
cmake --build build\pr-vk --target taichi_c_api -j8
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
