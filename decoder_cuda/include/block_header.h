#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include "constants.h"

namespace vdd {

#pragma pack(push, 1)
struct BlockHeader {
    uint32_t session_id;      // Offset 0
    uint32_t block_index;     // Offset 4
    uint32_t total_blocks;    // Offset 8
    uint32_t file_size;       // Offset 12
    uint16_t payload_size;    // Offset 16
    uint8_t flags;            // Offset 18
    uint8_t reserved;         // Offset 19
    uint16_t sequence;        // Offset 20
    uint16_t prev_crc16;      // Offset 22
    // Total: 24 bytes

    bool is_first_block() const { return flags & BLOCK_FLAG_FIRST; }
    bool is_last_block() const { return flags & BLOCK_FLAG_LAST; }
    bool is_encrypted() const { return flags & BLOCK_FLAG_ENCRYPTED; }

    static BlockHeader from_bytes(const uint8_t* data) {
        BlockHeader header;
        std::memcpy(&header, data, sizeof(BlockHeader));
        return header;
    }
};
#pragma pack(pop)

static_assert(sizeof(BlockHeader) == 24, "BlockHeader must be 24 bytes");

// File metadata stored in block 0 payload
struct FileMetadata {
    uint8_t file_hash[32];    // SHA-256 hash
    uint32_t filename_len;
    // Followed by filename bytes

    static constexpr int MIN_SIZE = 36;  // 32 + 4
};

// Decoded block with data
struct DecodedBlock {
    BlockHeader header;
    std::vector<uint8_t> payload;
    uint32_t crc32;
    bool crc_valid;
    bool fec_corrected;
    int fec_corrections;

    DecodedBlock() : crc32(0), crc_valid(false), fec_corrected(false), fec_corrections(0) {}
};

}  // namespace vdd
