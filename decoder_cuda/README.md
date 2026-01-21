# Visual Data Diode - CUDA Decoder

High-performance CUDA-accelerated decoder for visual data diode videos.

## Performance Target

- **Current Python decoder**: ~3 FPS (bottleneck: grid extraction)
- **CUDA decoder target**: 60+ FPS (20x improvement)
- **Expected throughput**: 100+ KB/s (vs ~1 KB/s Python)

## Prerequisites

1. **NVIDIA GPU**: RTX 3050 Ti or newer (Compute Capability 8.6+)
2. **CUDA Toolkit 11.8+**: Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. **Visual Studio 2022**: With C++ desktop development workload
4. **CMake 3.18+**: Download from [cmake.org](https://cmake.org/download/)
5. **OpenCV 4.x**: Download from [opencv.org](https://opencv.org/releases/)
6. **zlib**: Included with Visual Studio or download separately

## Installation Steps

### 1. Install CUDA Toolkit

1. Download CUDA Toolkit 11.8+ from NVIDIA
2. Run installer (express installation recommended)
3. Verify installation:
   ```
   nvcc --version
   ```

### 2. Install OpenCV

Option A: vcpkg (recommended):
```
vcpkg install opencv4:x64-windows
```

Option B: Manual installation:
1. Download OpenCV from opencv.org
2. Extract to C:\opencv
3. Set environment variable: `OpenCV_DIR=C:\opencv\build`

### 3. Build the Decoder

```batch
cd decoder_cuda
build.bat
```

Or manually with CMake:
```batch
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Usage

```batch
vdd_decode.exe --input captured.avi --output decoded.bin --profile standard
```

Options:
- `--input, -i`: Input video file (required)
- `--output, -o`: Output file or directory (required)
- `--profile, -p`: Encoding profile (conservative, standard, aggressive, ultra)
- `--help, -h`: Show help

## Profiles

| Profile | Cell Size | Capacity/Block | Best For |
|---------|-----------|----------------|----------|
| conservative | 16px | ~1.8 KB | Noisy capture |
| standard | 10px | ~4.8 KB | Normal use |
| aggressive | 8px | ~7.5 KB | High quality capture |
| ultra | 6px | ~13 KB | Perfect capture |

## Architecture

```
CUDA Decoder Pipeline:
1. Video Read (OpenCV) → RGB Frame
2. Sync Detect (CUDA)  → Frame Bounds
3. Grid Extract (CUDA) → Cell Values (2-bit)
4. Bit Pack (CUDA)     → Packed Bytes
5. Block Decode (CPU)  → Header + Payload + CRC
6. FEC Correct (CPU)   → Error Recovery
7. Assemble (CPU)      → Output File
```

## File Structure

```
decoder_cuda/
├── CMakeLists.txt          # Build configuration
├── build.bat               # Windows build script
├── include/
│   ├── constants.h         # Protocol constants
│   ├── block_header.h      # Block data structures
│   └── cuda_decoder.h      # Decoder interface
├── src/
│   ├── main.cpp            # CLI entry point
│   ├── cuda_decoder.cpp    # Main decoder class
│   ├── video_reader.cpp    # OpenCV video input
│   ├── block_assembler.cpp # File reconstruction
│   ├── fec_decoder.cpp     # Reed-Solomon FEC
│   └── kernels/
│       ├── sync_detect.cu  # Border detection
│       ├── grid_extract.cu # Cell sampling
│       └── bit_pack.cu     # Byte packing
└── tests/
    └── test_decoder.cpp    # Unit tests
```

## Troubleshooting

### CUDA not found
- Ensure CUDA Toolkit is installed
- Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`

### OpenCV not found
- Set `OpenCV_DIR` environment variable
- Or use vcpkg: `vcpkg install opencv4:x64-windows`

### Build errors
- Ensure Visual Studio 2022 is installed with C++ workload
- Check CUDA compute capability matches your GPU (modify `CMAKE_CUDA_ARCHITECTURES`)
