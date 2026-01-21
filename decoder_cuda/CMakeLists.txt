cmake_minimum_required(VERSION 3.18)
project(vdd_decode LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# vcpkg integration
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
elseif(EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# CUDA architecture - RTX 3050 Ti is SM 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${OpenCV_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
)

# Source files
set(CUDA_SOURCES
    src/kernels/sync_detect.cu
    src/kernels/grid_extract.cu
    src/kernels/bit_pack.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/cuda_decoder.cpp
    src/video_reader.cpp
    src/block_assembler.cpp
    src/fec_decoder.cpp
)

# Build executable
add_executable(vdd_decode
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Link libraries
target_link_libraries(vdd_decode
    CUDA::cudart
    ${OpenCV_LIBS}
)

# Copy test data
add_custom_command(TARGET vdd_decode POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build complete: vdd_decode.exe"
)

# Install
install(TARGETS vdd_decode DESTINATION bin)
