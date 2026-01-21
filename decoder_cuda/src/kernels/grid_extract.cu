/**
 * Visual Data Diode - CUDA Grid Extraction Kernel
 *
 * Extracts cell values from frame and quantizes to 2-bit values.
 * This is the main bottleneck in the Python decoder (~40ms per frame)
 * and should run in <1ms on GPU.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace vdd {

// Gray level thresholds for quantization
__constant__ uint8_t d_gray_thresholds[3] = {43, 128, 213};

// Quantize gray value to 2-bit value (0-3)
__device__ inline int quantize_gray(uint8_t gray) {
    if (gray < d_gray_thresholds[0]) return 0;
    if (gray < d_gray_thresholds[1]) return 1;
    if (gray < d_gray_thresholds[2]) return 2;
    return 3;
}

/**
 * Grid extraction kernel
 *
 * Each thread extracts one cell value by sampling the cell center.
 * Processes only interior cells (skips sync border).
 *
 * Input: RGB frame
 * Output: 2-bit values for each interior cell
 */
__global__ void grid_extract_kernel(
    const uint8_t* frame,
    int frame_width,
    int frame_height,
    int left,
    int top,
    int right,
    int bottom,
    int grid_width,
    int grid_height,
    int cell_size,
    int border_cells,
    uint8_t* cell_values  // Output: one byte per cell (only lower 2 bits used)
) {
    // Calculate interior dimensions
    int interior_width = grid_width - 2 * border_cells;
    int interior_height = grid_height - 2 * border_cells;
    int total_interior_cells = interior_width * interior_height;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_interior_cells) return;

    // Convert linear index to interior grid position
    int interior_row = idx / interior_width;
    int interior_col = idx % interior_width;

    // Convert to full grid position (add border offset)
    int grid_row = interior_row + border_cells;
    int grid_col = interior_col + border_cells;

    // Calculate frame bounds dimensions
    int frame_region_width = right - left;
    int frame_region_height = bottom - top;

    // Calculate actual cell size in pixels
    float cell_w = (float)frame_region_width / grid_width;
    float cell_h = (float)frame_region_height / grid_height;

    // Calculate cell center in frame coordinates
    int center_x = (int)(left + (grid_col + 0.5f) * cell_w);
    int center_y = (int)(top + (grid_row + 0.5f) * cell_h);

    // Clamp to frame bounds
    center_x = min(max(center_x, 0), frame_width - 1);
    center_y = min(max(center_y, 0), frame_height - 1);

    // Sample pixel (RGB format)
    int pixel_idx = (center_y * frame_width + center_x) * 3;
    uint8_t r = frame[pixel_idx];
    uint8_t g = frame[pixel_idx + 1];
    uint8_t b = frame[pixel_idx + 2];

    // Convert to grayscale (simple average)
    uint8_t gray = (r + g + b) / 3;

    // Quantize to 2-bit value
    cell_values[idx] = (uint8_t)quantize_gray(gray);
}

/**
 * Alternative kernel with 2x2 center sampling for noise reduction
 * Samples a small region at cell center and uses majority voting
 */
__global__ void grid_extract_robust_kernel(
    const uint8_t* frame,
    int frame_width,
    int frame_height,
    int left,
    int top,
    int right,
    int bottom,
    int grid_width,
    int grid_height,
    int cell_size,
    int border_cells,
    uint8_t* cell_values
) {
    int interior_width = grid_width - 2 * border_cells;
    int interior_height = grid_height - 2 * border_cells;
    int total_interior_cells = interior_width * interior_height;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_interior_cells) return;

    int interior_row = idx / interior_width;
    int interior_col = idx % interior_width;
    int grid_row = interior_row + border_cells;
    int grid_col = interior_col + border_cells;

    int frame_region_width = right - left;
    int frame_region_height = bottom - top;
    float cell_w = (float)frame_region_width / grid_width;
    float cell_h = (float)frame_region_height / grid_height;

    // Calculate cell center
    float center_x_f = left + (grid_col + 0.5f) * cell_w;
    float center_y_f = top + (grid_row + 0.5f) * cell_h;

    // Sample 2x2 area around center
    int samples[4];
    int offsets[4][2] = {{-1, -1}, {0, -1}, {-1, 0}, {0, 0}};

    for (int s = 0; s < 4; s++) {
        int x = (int)(center_x_f + offsets[s][0]);
        int y = (int)(center_y_f + offsets[s][1]);
        x = min(max(x, 0), frame_width - 1);
        y = min(max(y, 0), frame_height - 1);

        int pixel_idx = (y * frame_width + x) * 3;
        uint8_t gray = (frame[pixel_idx] + frame[pixel_idx + 1] + frame[pixel_idx + 2]) / 3;
        samples[s] = quantize_gray(gray);
    }

    // Majority voting (most common value)
    int counts[4] = {0, 0, 0, 0};
    for (int s = 0; s < 4; s++) {
        counts[samples[s]]++;
    }

    int best = 0;
    int best_count = counts[0];
    for (int i = 1; i < 4; i++) {
        if (counts[i] > best_count) {
            best = i;
            best_count = counts[i];
        }
    }

    cell_values[idx] = (uint8_t)best;
}

// Host wrapper for grid extraction
extern "C" void launch_grid_extract(
    const uint8_t* d_frame,
    int frame_width,
    int frame_height,
    int left,
    int top,
    int right,
    int bottom,
    int grid_width,
    int grid_height,
    int cell_size,
    int border_cells,
    uint8_t* d_cell_values,
    bool use_robust,
    cudaStream_t stream
) {
    int interior_width = grid_width - 2 * border_cells;
    int interior_height = grid_height - 2 * border_cells;
    int total_cells = interior_width * interior_height;

    int threads_per_block = 256;
    int num_blocks = (total_cells + threads_per_block - 1) / threads_per_block;

    if (use_robust) {
        grid_extract_robust_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            d_frame, frame_width, frame_height,
            left, top, right, bottom,
            grid_width, grid_height, cell_size, border_cells,
            d_cell_values
        );
    } else {
        grid_extract_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            d_frame, frame_width, frame_height,
            left, top, right, bottom,
            grid_width, grid_height, cell_size, border_cells,
            d_cell_values
        );
    }
}

}  // namespace vdd
