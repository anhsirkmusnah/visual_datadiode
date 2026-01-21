/**
 * Visual Data Diode - CUDA Sync Detection Kernel
 *
 * Detects cyan/magenta sync border to find frame boundaries.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace vdd {

// Check if pixel is cyan-ish (low R, high G, high B)
__device__ inline bool is_cyan(uint8_t r, uint8_t g, uint8_t b) {
    return (r < 150) && (g > 150) && (b > 150) && (g > r + 50) && (b > r + 50);
}

// Check if pixel is magenta-ish (high R, low G, high B)
__device__ inline bool is_magenta(uint8_t r, uint8_t g, uint8_t b) {
    return (r > 150) && (g < 150) && (b > 150) && (r > g + 50) && (b > g + 50);
}

// Kernel to detect sync border pixels and find bounds
// Each block handles a row/column and uses atomic operations to find min/max
__global__ void sync_detect_kernel(
    const uint8_t* frame,
    int width,
    int height,
    int* min_x,
    int* max_x,
    int* min_y,
    int* max_y,
    int* sync_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx >= total_pixels) return;

    int y = idx / width;
    int x = idx % width;

    // Get RGB values (assuming RGB format, 3 bytes per pixel)
    int pixel_idx = (y * width + x) * 3;
    uint8_t r = frame[pixel_idx];
    uint8_t g = frame[pixel_idx + 1];
    uint8_t b = frame[pixel_idx + 2];

    // Check if sync color
    if (is_cyan(r, g, b) || is_magenta(r, g, b)) {
        atomicMin(min_x, x);
        atomicMax(max_x, x);
        atomicMin(min_y, y);
        atomicMax(max_y, y);
        atomicAdd(sync_count, 1);
    }
}

// Kernel to verify corner markers
// Checks 3x3 corner regions for white/black pattern
__global__ void corner_verify_kernel(
    const uint8_t* frame,
    int width,
    int height,
    int left,
    int top,
    int right,
    int bottom,
    int cell_size,
    int* corner_matches
) {
    // Each thread verifies one corner
    int corner_idx = threadIdx.x;
    if (corner_idx >= 4) return;

    // Corner positions
    int corner_x, corner_y;
    const uint8_t* pattern;

    // Corner patterns (3x3, row-major)
    const uint8_t CORNER_TL[9] = {1,1,1, 1,0,1, 1,1,0};
    const uint8_t CORNER_TR[9] = {0,1,1, 1,0,1, 1,1,1};
    const uint8_t CORNER_BL[9] = {1,1,0, 1,0,1, 1,1,1};
    const uint8_t CORNER_BR[9] = {0,1,0, 1,0,1, 1,1,1};

    switch (corner_idx) {
        case 0:  // Top-left
            corner_x = left;
            corner_y = top;
            pattern = CORNER_TL;
            break;
        case 1:  // Top-right
            corner_x = right - cell_size * 3;
            corner_y = top;
            pattern = CORNER_TR;
            break;
        case 2:  // Bottom-left
            corner_x = left;
            corner_y = bottom - cell_size * 3;
            pattern = CORNER_BL;
            break;
        case 3:  // Bottom-right
            corner_x = right - cell_size * 3;
            corner_y = bottom - cell_size * 3;
            pattern = CORNER_BR;
            break;
        default:
            return;
    }

    // Check 3x3 cell pattern
    int matches = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // Sample center of cell
            int sample_x = corner_x + j * cell_size + cell_size / 2;
            int sample_y = corner_y + i * cell_size + cell_size / 2;

            if (sample_x >= 0 && sample_x < width && sample_y >= 0 && sample_y < height) {
                int pixel_idx = (sample_y * width + sample_x) * 3;
                uint8_t gray = (frame[pixel_idx] + frame[pixel_idx + 1] + frame[pixel_idx + 2]) / 3;

                bool expected_white = pattern[i * 3 + j] == 1;
                bool is_white = gray > 128;

                if (expected_white == is_white) {
                    matches++;
                }
            }
        }
    }

    // Require at least 7/9 matches
    if (matches >= 7) {
        atomicAdd(corner_matches, 1);
    }
}

// Host wrapper for sync detection
extern "C" void launch_sync_detect(
    const uint8_t* d_frame,
    int width,
    int height,
    int* d_bounds,  // [min_x, max_x, min_y, max_y, sync_count]
    cudaStream_t stream
) {
    int total_pixels = width * height;
    int threads_per_block = 256;
    int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    // Initialize bounds to extreme values
    int h_init[5] = {width, 0, height, 0, 0};  // min_x=width, max_x=0, min_y=height, max_y=0, count=0
    cudaMemcpyAsync(d_bounds, h_init, 5 * sizeof(int), cudaMemcpyHostToDevice, stream);

    sync_detect_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_frame, width, height,
        &d_bounds[0], &d_bounds[1], &d_bounds[2], &d_bounds[3], &d_bounds[4]
    );
}

// Host wrapper for corner verification
extern "C" void launch_corner_verify(
    const uint8_t* d_frame,
    int width,
    int height,
    int left,
    int top,
    int right,
    int bottom,
    int cell_size,
    int* d_corner_matches,
    cudaStream_t stream
) {
    // Initialize corner matches
    cudaMemsetAsync(d_corner_matches, 0, sizeof(int), stream);

    // Launch with 4 threads (one per corner)
    corner_verify_kernel<<<1, 4, 0, stream>>>(
        d_frame, width, height, left, top, right, bottom, cell_size, d_corner_matches
    );
}

}  // namespace vdd
