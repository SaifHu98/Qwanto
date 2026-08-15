@echo off
setlocal
set "MSVC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
set "MSVC_LIB=%MSVC%\lib\x64"
set "MSVC_INC=%MSVC%\include"
set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt"
set "SDK_INC_UM=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um"
set "SDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"

set "SRC=D:\EcoUni\qwanto\c"
set "OUT=D:\EcoUni\qwanto\c\test_agentic.exe"

clang -O3 -march=x86-64-v3 -fopenmp ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_agentic.c" "%SRC%\qwanto_agentic.c" "%SRC%\qwanto_decode.c" "%SRC%\qwanto_thinking.c" "%SRC%\qwanto_turboquant.c" "%SRC%\qwanto_speculative.c" "%SRC%\qwanto_kernels.c" "%SRC%\qwanto_native.c" "%SRC%\qwn_paged_kv.c" ^
    "%MSVC_LIB%\libomp.lib" "%SDK_LIB%\psapi.lib" ^
    -o "%OUT%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD FAILED & exit /b 1)
echo Built: %OUT%
endlocal
