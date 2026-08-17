@echo off
setlocal
set "MSVC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
set "MSVC_LIB=%MSVC%\lib\x64"
set "MSVC_INC=%MSVC%\include"
set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt"
set "SDK_INC_UM=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um"
set "SDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"
set "OMP_REDIST=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC\14.51.36231\debug_nonredist\x64\Microsoft.VC145.OpenMP.LLVM"

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "OUT=%SRC%\qwnrun_msvc.exe"

clang -O3 -mavx2 -mf16c -mfma -fopenmp ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" ^
    "%SRC%\qwnrun.c" "%SRC%\qwn_runtime_config.c" "%SRC%\qwanto_decode.c" "%SRC%\qwanto_native.c" "%SRC%\qwanto_kernels.c" "%SRC%\qwanto_turboquant.c" "%SRC%\qwanto_thinking.c" "%SRC%\qwanto_speculative.c" "%SRC%\qwanto_agentic.c" "%SRC%\qwanto_autopilot.c" "%SRC%\qwanto_gpu.c" "%SRC%\qwanto_bitdecoding.c" "%SRC%\qwanto_jetspec.c" "%SRC%\qwanto_talon.c" "%SRC%\qwanto_sliminfer.c" "%SRC%\qwanto_pquant.c" "%SRC%\qwanto_littlebit.c" "%SRC%\qwn_paged_kv.c" ^
    "%MSVC_LIB%\libomp.lib" "%SDK_LIB%\psapi.lib" ^
    -o "%OUT%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD FAILED & exit /b 1)

if not exist "%OMP_REDIST%\libomp140.x86_64.dll" (
    echo BUILD FAILED: LLVM OpenMP runtime DLL was not found at "%OMP_REDIST%\libomp140.x86_64.dll" 1>&2
    exit /b 1
)
copy /Y "%OMP_REDIST%\libomp140.x86_64.dll" "%SRC%\libomp140.x86_64.dll" >nul

echo Built: %OUT%
echo LLVM OpenMP runtime DLL copied alongside the binary.
endlocal
