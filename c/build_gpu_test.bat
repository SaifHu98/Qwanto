@echo off
setlocal
set "MSVC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231"
set "MSVC_LIB=%MSVC%\lib\x64"
set "MSVC_INC=%MSVC%\include"
set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt"
set "SDK_INC_UM=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um"
set "SDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"

set "SRC=D:\EcoUni\qwanto\c"
set "OUT1=D:\EcoUni\qwanto\c\test_gpu_detection.exe"
set "OUT2=D:\EcoUni\qwanto\c\test_gpu_kernels.exe"
set "OUT3=D:\EcoUni\qwanto\c\test_bitdecoding.exe"
set "OUT4=D:\EcoUni\qwanto\c\test_jetspec.exe"
set "OUT5=D:\EcoUni\qwanto\c\test_talon.exe"
set "OUT6=D:\EcoUni\qwanto\c\test_sliminfer.exe"
set "OUT7=D:\EcoUni\qwanto\c\test_pquant.exe"
set "OUT8=D:\EcoUni\qwanto\c\test_littlebit.exe"
set "OUT9=D:\EcoUni\qwanto\c\test_unified_5000.exe"

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_gpu_detection.c" "%SRC%\qwanto_gpu.c" "%SRC%\qwanto_bitdecoding.c" ^
    -o "%OUT1%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD DETECTION FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_gpu_kernels.c" "%SRC%\qwanto_gpu.c" "%SRC%\qwanto_bitdecoding.c" ^
    -o "%OUT2%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD KERNELS FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_bitdecoding.c" "%SRC%\qwanto_gpu.c" "%SRC%\qwanto_bitdecoding.c" ^
    -o "%OUT3%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD BITDECODING FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_jetspec.c" "%SRC%\qwanto_jetspec.c" ^
    -o "%OUT4%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD JETSPEC FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_talon.c" "%SRC%\qwanto_talon.c" ^
    -o "%OUT5%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD TALON FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_sliminfer.c" "%SRC%\qwanto_sliminfer.c" ^
    -o "%OUT6%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD SLIMINFER FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_pquant.c" "%SRC%\qwanto_pquant.c" ^
    -o "%OUT7%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD PQUANT FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_littlebit.c" "%SRC%\qwanto_littlebit.c" ^
    -o "%OUT8%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD LITTLEBIT FAILED & exit /b 1)

clang -O3 -march=x86-64-v3 -mavxvnni -Wno-deprecated-declarations ^
    -I"%MSVC_INC%" -I"%SDK_INC%" -I"%SDK_INC_UM%" -I"%SRC%" ^
    "%SRC%\tests\test_unified_5000.c" "%SRC%\qwanto_pquant.c" "%SRC%\qwanto_littlebit.c" "%SRC%\qwanto_bitdecoding.c" "%SRC%\qwanto_jetspec.c" "%SRC%\qwanto_talon.c" "%SRC%\qwanto_sliminfer.c" "%SRC%\qwanto_autopilot.c" "%SRC%\qwanto_thinking.c" "%SRC%\qwanto_decode.c" "%SRC%\qwanto_native.c" "%SRC%\qwanto_kernels.c" "%SRC%\qwanto_turboquant.c" "%SRC%\qwn_paged_kv.c" ^
    -o "%OUT9%" ^
    -Xlinker /LIBPATH:"%MSVC_LIB%" -Xlinker /LIBPATH:"%SDK_LIB%"

if %ERRORLEVEL% NEQ 0 (echo BUILD UNIFIED 5000 FAILED & exit /b 1)

echo Built: %OUT1%
echo Built: %OUT2%
echo Built: %OUT3%
echo Built: %OUT4%
echo Built: %OUT5%
echo Built: %OUT6%
echo Built: %OUT7%
echo Built: %OUT8%
echo Built: %OUT9%
endlocal
