@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
pushd "%ROOT%"

where cl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :msvc

where gcc.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :gcc

echo [ERROR] Neither MSVC cl.exe nor MinGW-w64 gcc.exe was found. 1>&2
echo [INFO] Open a VS Developer Command Prompt or install MinGW-w64. 1>&2
popd
exit /b 1

:msvc
echo [INFO] Building qwnrun with MSVC OpenMP/AVX2...
cl.exe /nologo /O2 /Oi /Ot /openmp /arch:AVX2 /fp:fast ^
  /D_CRT_SECURE_NO_WARNINGS /DNDEBUG /DCOLI_CUDA /I"c" ^
  c/qwnrun.c c/qwanto_decode.c c/qwanto_native.c c/qwanto_kernels.c c/qwanto_turboquant.c c/qwanto_thinking.c c/qwanto_speculative.c c/qwanto_agentic.c c/qwanto_autopilot.c c/qwn_paged_kv.c c/backend_loader.c ^
  /Fe:c/qwnrun.exe /link /OPT:REF /OPT:ICF vcomp140.lib
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] MSVC qwnrun build failed. 1>&2
  popd
  exit /b 1
)
echo [OK] Built c/qwnrun.exe with MSVC OpenMP.
popd
exit /b 0

:gcc
echo [INFO] Building qwnrun with MinGW-w64 GCC/libgomp...
gcc.exe -std=c11 -O3 -mavx2 -mf16c -mfma -fopenmp -ffast-math ^
  -D_CRT_SECURE_NO_WARNINGS -DNDEBUG -DCOLI_CUDA -Ic ^
  c/qwnrun.c c/qwanto_decode.c c/qwanto_native.c c/qwanto_kernels.c c/qwanto_turboquant.c c/qwanto_thinking.c c/qwanto_speculative.c c/qwanto_agentic.c c/qwanto_autopilot.c c/qwn_paged_kv.c c/backend_loader.c ^
  -o c/qwnrun.exe -lgomp -lm -lpsapi
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] GCC qwnrun build failed. 1>&2
  popd
  exit /b 1
)
echo [OK] Built c/qwnrun.exe with GCC OpenMP.
popd
exit /b 0
