#pragma once

#include <cstdint>

namespace vdd {

// Frame dimensions
constexpr int FRAME_WIDTH = 1920;
constexpr int FRAME_HEIGHT = 1080;

// Grayscale encoding levels
constexpr uint8_t GRAY_LEVELS[4] = {0, 85, 170, 255};
constexpr uint8_t GRAY_THRESHOLDS[3] = {43, 128, 213};

// Sync border colors (RGB)
constexpr uint8_t COLOR_CYAN[3] = {0, 255, 255};
constexpr uint8_t COLOR_MAGENTA[3] = {255, 0, 255};
constexpr uint8_t COLOR_WHITE[3] = {255, 255, 255};
constexpr uint8_t COLOR_BLACK[3] = {0, 0, 0};

// Corner marker patterns (3x3, row-major, 1=white, 0=black)
constexpr uint8_t CORNER_TOP_LEFT[9] = {1,1,1, 1,0,1, 1,1,0};
constexpr uint8_t CORNER_TOP_RIGHT[9] = {0,1,1, 1,0,1, 1,1,1};
constexpr uint8_t CORNER_BOTTOM_LEFT[9] = {1,1,0, 1,0,1, 1,1,1};
constexpr uint8_t CORNER_BOTTOM_RIGHT[9] = {0,1,0, 1,0,1, 1,1,1};

// Block header format
constexpr int HEADER_SIZE = 24;
constexpr int CRC_SIZE = 4;

// FEC parameters
constexpr int RS_SYMBOL_SIZE = 8;
constexpr int RS_DEFAULT_NSYM = 32;
constexpr float DEFAULT_FEC_RATIO = 0.10f;

// Encoding profile parameters
struct EncodingProfile {
    const char* name;
    int cell_size;
    int grid_width;
    int grid_height;
    int interior_width;
    int interior_height;
    int border_width;
    int payload_bytes;

    constexpr EncodingProfile(const char* n, int cs) :
        name(n),
        cell_size(cs),
        grid_width(FRAME_WIDTH / cs),
        grid_height(FRAME_HEIGHT / cs),
        interior_width(FRAME_WIDTH / cs - 4),  // 2 cells border on each side
        interior_height(FRAME_HEIGHT / cs - 4),
        border_width(2),
        payload_bytes((FRAME_WIDTH / cs - 4) * (FRAME_HEIGHT / cs - 4) / 4)  // 4 cells per byte
    {}
};

// Standard profiles
constexpr EncodingProfile PROFILE_CONSERVATIVE("conservative", 16);
constexpr EncodingProfile PROFILE_STANDARD("standard", 10);
constexpr EncodingProfile PROFILE_AGGRESSIVE("aggressive", 8);
constexpr EncodingProfile PROFILE_ULTRA("ultra", 6);

// Block header flags
enum BlockFlags : uint8_t {
    BLOCK_FLAG_NONE = 0x00,
    BLOCK_FLAG_FIRST = 0x01,
    BLOCK_FLAG_LAST = 0x02,
    BLOCK_FLAG_ENCRYPTED = 0x04,
    BLOCK_FLAG_COMPRESSED = 0x08
};

// Quantize gray value to 2-bit value (0-3)
#ifdef __CUDACC__
__host__ __device__
#endif
inline int gray_to_bits(uint8_t gray) {
    if (gray < GRAY_THRESHOLDS[0]) return 0;
    if (gray < GRAY_THRESHOLDS[1]) return 1;
    if (gray < GRAY_THRESHOLDS[2]) return 2;
    return 3;
}

}  // namespace vdd
