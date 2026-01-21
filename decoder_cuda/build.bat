@echo off
REM Visual Data Diode - CUDA Decoder Build Script
REM Requires: Visual Studio 2022, CUDA Toolkit 13.1+, OpenCV via vcpkg

echo Building VDD CUDA Decoder...
echo.

REM Set CUDA path
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM Set vcpkg path
set "VCPKG_ROOT=C:\vcpkg"

REM Check for CMake
where cmake >nul 2>&1
if errorlevel 1 (
    set "PATH=C:\Program Files\CMake\bin;%PATH%"
)

where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: CMake not found in PATH
    echo Please install CMake and add to PATH
    exit /b 1
)

cmake --version
echo.

REM Check for CUDA
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: CUDA nvcc not found in PATH
    echo Please ensure CUDA Toolkit is installed
    exit /b 1
)

nvcc --version
echo.

REM Save current directory
set "SCRIPT_DIR=%~dp0"

REM Find Visual Studio and set up environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Return to script directory
cd /d "%SCRIPT_DIR%"

REM Clean and create build directory
if exist build rmdir /s /q build
mkdir build
cd build

REM Configure with CMake using Ninja
echo Configuring with CMake (Ninja)...
cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    ..

if errorlevel 1 (
    echo ERROR: CMake configuration failed
    cd ..
    exit /b 1
)

REM Build
echo.
echo Building...
ninja

if errorlevel 1 (
    echo ERROR: Build failed
    cd ..
    exit /b 1
)

echo.
echo ============================================================
echo Build successful!
echo Executable: %CD%\Release\vdd_decode.exe
echo ============================================================
cd ..
