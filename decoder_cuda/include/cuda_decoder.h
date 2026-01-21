#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <map>

#include "constants.h"
#include "block_header.h"

namespace vdd {

// Frame bounds detected by sync
struct FrameBounds {
    int left;
    int top;
    int right;
    int bottom;
    bool valid;
    float confidence;
};

// Decode result for a single frame
struct FrameDecodeResult {
    bool synced;
    bool decoded;
    bool crc_valid;
    int block_index;
    int fec_corrections;
};

// Statistics for video decoding
struct DecodeStats {
    int total_frames;
    int synced_frames;
    int decoded_blocks;
    int unique_blocks;
    int crc_errors;
    int fec_corrections;
    double processing_time;
    double fps;
};

// CUDA decoder class
class CudaDecoder {
public:
    CudaDecoder(const EncodingProfile& profile = PROFILE_STANDARD);
    ~CudaDecoder();

    // Initialize CUDA resources
    bool initialize();

    // Release CUDA resources
    void release();

    // Decode a single frame (RGB format)
    // Returns true if a valid block was decoded
    FrameDecodeResult decode_frame(const uint8_t* frame_rgb, int width, int height);

    // Get decoded block data
    bool get_block(int block_index, DecodedBlock& block) const;

    // Get all received blocks
    const std::map<int, DecodedBlock>& get_blocks() const { return blocks_; }

    // Check if all blocks received
    bool is_complete() const;

    // Get missing block indices
    std::vector<int> get_missing_blocks() const;

    // Assemble file from blocks
    std::vector<uint8_t> assemble_file();

    // Get file hash from block 0 metadata (32 bytes SHA-256)
    bool get_expected_hash(uint8_t* hash_out) const;

    // Get original filename from block 0 metadata
    std::string get_original_filename() const;

    // Get stats
    DecodeStats get_stats() const;

    // Reset decoder state
    void reset();

private:
    // Profile
    EncodingProfile profile_;

    // CUDA memory
    uint8_t* d_frame_;           // Device frame buffer
    uint8_t* d_grid_;            // Device grid buffer
    uint8_t* d_packed_bytes_;    // Device packed bytes
    int* d_bounds_;              // Device frame bounds [left, top, right, bottom, valid]
    int* d_cell_values_;         // Device cell values (2-bit per cell)

    // Host memory
    std::vector<uint8_t> h_packed_bytes_;

    // Block storage
    std::map<int, DecodedBlock> blocks_;
    int total_blocks_;
    uint32_t session_id_;

    // Stats
    int frames_processed_;
    int frames_synced_;
    int crc_errors_;
    int fec_corrections_;

    // CUDA stream
    cudaStream_t stream_;

    // Internal methods
    bool detect_sync(const uint8_t* d_frame, FrameBounds& bounds);
    bool extract_grid(const uint8_t* d_frame, const FrameBounds& bounds, uint8_t* d_grid);
    bool pack_bytes(const uint8_t* d_grid, uint8_t* d_bytes);
    bool decode_block(const uint8_t* bytes, int size, DecodedBlock& block);
    bool verify_crc(const uint8_t* data, int size, uint32_t expected_crc);
};

// Video decoder that wraps CudaDecoder with video file reading
class VideoDecoder {
public:
    VideoDecoder(const std::string& video_path, const EncodingProfile& profile = PROFILE_STANDARD);
    ~VideoDecoder();

    // Decode entire video
    bool decode_all(std::function<void(int, int, int)> progress_callback = nullptr);

    // Get decoder
    CudaDecoder& decoder() { return *decoder_; }

    // Get assembled file data
    std::vector<uint8_t> get_file_data();

    // Save decoded file
    bool save_file(const std::string& output_path);

    // Get stats
    DecodeStats get_stats() const;

private:
    std::string video_path_;
    EncodingProfile profile_;
    std::unique_ptr<CudaDecoder> decoder_;

    // Video info
    int total_frames_;
    double fps_;
    int width_;
    int height_;
};

}  // namespace vdd
