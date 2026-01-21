/**
 * Visual Data Diode - CUDA Bit Packing Kernel
 *
 * Packs 2-bit cell values into bytes (4 cells per byte).
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace vdd {

/**
 * Bit packing kernel
 *
 * Packs 4 consecutive 2-bit cell values into one byte.
 * Each thread packs 4 cells into 1 byte.
 *
 * Input: cell_values (one byte per cell, only lower 2 bits used)
 * Output: packed_bytes (4 cells per byte)
 */
__global__ void bit_pack_kernel(
    const uint8_t* cell_values,
    int num_cells,
    uint8_t* packed_bytes
) {
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_bytes = num_cells / 4;

    if (byte_idx >= num_bytes) return;

    int cell_start = byte_idx * 4;

    // Pack 4 cells into one byte
    // Cell order: [bits 7-6, bits 5-4, bits 3-2, bits 1-0]
    uint8_t packed = 0;

    if (cell_start < num_cells)
        packed |= (cell_values[cell_start] & 0x03) << 6;
    if (cell_start + 1 < num_cells)
        packed |= (cell_values[cell_start + 1] & 0x03) << 4;
    if (cell_start + 2 < num_cells)
        packed |= (cell_values[cell_start + 2] & 0x03) << 2;
    if (cell_start + 3 < num_cells)
        packed |= (cell_values[cell_start + 3] & 0x03);

    packed_bytes[byte_idx] = packed;
}

/**
 * Combined extraction + packing kernel for maximum efficiency
 * Reads cells in groups of 4 and directly outputs packed bytes.
 */
__global__ void grid_to_bytes_kernel(
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
    uint8_t* packed_bytes
) {
    // Constants for quantization
    const uint8_t thresh0 = 43;
    const uint8_t thresh1 = 128;
    const uint8_t thresh2 = 213;

    int interior_width = grid_width - 2 * border_cells;
    int interior_height = grid_height - 2 * border_cells;
    int total_cells = interior_width * interior_height;
    int num_bytes = total_cells / 4;

    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx >= num_bytes) return;

    int frame_region_width = right - left;
    int frame_region_height = bottom - top;
    float cell_w = (float)frame_region_width / grid_width;
    float cell_h = (float)frame_region_height / grid_height;

    uint8_t packed = 0;

    // Process 4 cells
    for (int i = 0; i < 4; i++) {
        int cell_idx = byte_idx * 4 + i;
        if (cell_idx >= total_cells) break;

        // Convert to grid position
        int interior_row = cell_idx / interior_width;
        int interior_col = cell_idx % interior_width;
        int grid_row = interior_row + border_cells;
        int grid_col = interior_col + border_cells;

        // Calculate cell center
        int center_x = (int)(left + (grid_col + 0.5f) * cell_w);
        int center_y = (int)(top + (grid_row + 0.5f) * cell_h);
        center_x = min(max(center_x, 0), frame_width - 1);
        center_y = min(max(center_y, 0), frame_height - 1);

        // Sample pixel
        int pixel_idx = (center_y * frame_width + center_x) * 3;
        uint8_t gray = (frame[pixel_idx] + frame[pixel_idx + 1] + frame[pixel_idx + 2]) / 3;

        // Quantize
        int bits;
        if (gray < thresh0) bits = 0;
        else if (gray < thresh1) bits = 1;
        else if (gray < thresh2) bits = 2;
        else bits = 3;

        // Pack into byte
        packed |= (bits << (6 - i * 2));
    }

    packed_bytes[byte_idx] = packed;
}

// Host wrapper for bit packing
extern "C" void launch_bit_pack(
    const uint8_t* d_cell_values,
    int num_cells,
    uint8_t* d_packed_bytes,
    cudaStream_t stream
) {
    int num_bytes = num_cells / 4;
    int threads_per_block = 256;
    int num_blocks = (num_bytes + threads_per_block - 1) / threads_per_block;

    bit_pack_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_cell_values, num_cells, d_packed_bytes
    );
}

// Host wrapper for combined extraction + packing
extern "C" void launch_grid_to_bytes(
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
    uint8_t* d_packed_bytes,
    cudaStream_t stream
) {
    int interior_width = grid_width - 2 * border_cells;
    int interior_height = grid_height - 2 * border_cells;
    int num_bytes = (interior_width * interior_height) / 4;

    int threads_per_block = 256;
    int num_blocks = (num_bytes + threads_per_block - 1) / threads_per_block;

    grid_to_bytes_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_frame, frame_width, frame_height,
        left, top, right, bottom,
        grid_width, grid_height, cell_size, border_cells,
        d_packed_bytes
    );
}

}  // namespace vdd
