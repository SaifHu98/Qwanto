@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
pushd "%ROOT%"

where nvcc.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [INFO] CUDA Toolkit/nvcc not found; qwnrun will use the OpenMP CPU fallback.
  popd
  exit /b 0
)

where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
  )
)

echo [INFO] Building c/qwn_cuda.dll with nvcc...
nvcc -O3 -std=c++17 --shared -DQWN_CUDA_BUILDING_DLL ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_80,code=sm_80 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -Xcompiler="/O2 /MD /D_CRT_SECURE_NO_WARNINGS" ^
  c/cuda/qwn_hypervsq_cuda.cu -o c/qwn_cuda.dll -lcublas -lcudart
if %ERRORLEVEL% NEQ 0 (
  echo [WARN] CUDA DLL build failed; qwnrun remains usable through CPU fallback. 1>&2
  popd
  exit /b 1
)
echo [OK] Built c/qwn_cuda.dll. qwnrun will load it dynamically when present.
echo [INFO] Building the legacy multi-GPU backend used by COLI_CUDA...
nvcc -O3 -std=c++17 --shared -DCOLI_CUDA_BUILDING_DLL ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_80,code=sm_80 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -Xcompiler="/O2 /MD /D_CRT_SECURE_NO_WARNINGS" ^
  c/backend_cuda.cu -o c/coli_cuda.dll -lcublas -lcudart
if %ERRORLEVEL% NEQ 0 (
  echo [WARN] Legacy multi-GPU CUDA backend failed; qwn_cuda.dll remains available. 1>&2
  popd
  exit /b 1
)
echo [OK] Built c/coli_cuda.dll with multi-GPU support.
popd
exit /b 0
