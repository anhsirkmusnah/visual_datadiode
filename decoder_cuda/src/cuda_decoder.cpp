/**
 * Visual Data Diode - CUDA Decoder Implementation
 *
 * High-performance decoder using CUDA for frame processing.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>

#include "cuda_decoder.h"
#include "constants.h"
#include "block_header.h"

// External CUDA kernel launchers
extern "C" {
    void launch_sync_detect(
        const uint8_t* d_frame,
        int width,
        int height,
        int* d_bounds,
        cudaStream_t stream
    );

    void launch_corner_verify(
        const uint8_t* d_frame,
        int width,
        int left,
        int top,
        int right,
        int bottom,
        int cell_size,
        int* d_corner_matches,
        cudaStream_t stream
    );

    void launch_grid_to_bytes(
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
    );
}

namespace vdd {

// CRC32 from fec_decoder.cpp
class CRC32 {
public:
    static uint32_t compute(const uint8_t* data, size_t length);
    static bool verify(const uint8_t* data, size_t length, uint32_t expected_crc);
};

CudaDecoder::CudaDecoder(const EncodingProfile& profile)
    : profile_(profile)
    , d_frame_(nullptr)
    , d_grid_(nullptr)
    , d_packed_bytes_(nullptr)
    , d_bounds_(nullptr)
    , d_cell_values_(nullptr)
    , total_blocks_(0)
    , session_id_(0)
    , frames_processed_(0)
    , frames_synced_(0)
    , crc_errors_(0)
    , fec_corrections_(0)
    , stream_(nullptr)
{
}

CudaDecoder::~CudaDecoder() {
    release();
}

bool CudaDecoder::initialize() {
    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (code " << err << ")" << std::endl;
        std::cerr << "This may be due to driver/runtime version mismatch." << std::endl;
        std::cerr << "Compiled with CUDA toolkit, but driver may be older." << std::endl;
        return false;
    }
    if (device_count == 0) {
        std::cerr << "No CUDA devices available" << std::endl;
        return false;
    }

    // Select device 0
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;

    // Create stream
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Allocate device memory
    int frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3;
    int grid_size = profile_.grid_width * profile_.grid_height;
    int interior_cells = profile_.interior_width * profile_.interior_height;
    int packed_size = interior_cells / 4;

    err = cudaMalloc(&d_frame_, frame_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate frame buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_packed_bytes_, packed_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate packed bytes buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_bounds_, 6 * sizeof(int));  // min_x, max_x, min_y, max_y, sync_count, corner_matches
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate bounds buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Allocate host memory
    h_packed_bytes_.resize(packed_size);

    std::cout << "CUDA Decoder initialized" << std::endl;
    std::cout << "  Profile: " << profile_.name << " (" << profile_.cell_size << "px cells)" << std::endl;
    std::cout << "  Grid: " << profile_.grid_width << "x" << profile_.grid_height << std::endl;
    std::cout << "  Interior: " << profile_.interior_width << "x" << profile_.interior_height << std::endl;
    std::cout << "  Payload: " << packed_size << " bytes/block" << std::endl;

    return true;
}

void CudaDecoder::release() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (d_frame_) {
        cudaFree(d_frame_);
        d_frame_ = nullptr;
    }
    if (d_packed_bytes_) {
        cudaFree(d_packed_bytes_);
        d_packed_bytes_ = nullptr;
    }
    if (d_bounds_) {
        cudaFree(d_bounds_);
        d_bounds_ = nullptr;
    }
}

FrameDecodeResult CudaDecoder::decode_frame(const uint8_t* frame_rgb, int width, int height) {
    FrameDecodeResult result = {false, false, false, -1, 0};
    frames_processed_++;

    // Upload frame to device
    int frame_size = width * height * 3;
    cudaError_t err = cudaMemcpyAsync(d_frame_, frame_rgb, frame_size,
                                       cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        return result;
    }

    // Detect sync
    FrameBounds bounds;
    if (!detect_sync(d_frame_, bounds)) {
        return result;
    }

    result.synced = true;
    frames_synced_++;

    // Extract and pack bytes (combined kernel)
    launch_grid_to_bytes(
        d_frame_,
        width, height,
        bounds.left, bounds.top, bounds.right, bounds.bottom,
        profile_.grid_width, profile_.grid_height,
        profile_.cell_size, profile_.border_width,
        d_packed_bytes_,
        stream_
    );

    // Download packed bytes
    int packed_size = profile_.interior_width * profile_.interior_height / 4;
    err = cudaMemcpyAsync(h_packed_bytes_.data(), d_packed_bytes_, packed_size,
                          cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        return result;
    }

    // Synchronize
    cudaStreamSynchronize(stream_);

    // Decode block
    DecodedBlock block;
    if (decode_block(h_packed_bytes_.data(), packed_size, block)) {
        result.decoded = true;
        result.crc_valid = block.crc_valid;
        result.block_index = block.header.block_index;
        result.fec_corrections = block.fec_corrections;

        if (block.crc_valid) {
            // Check if this is a new block
            if (blocks_.find(block.header.block_index) == blocks_.end()) {
                blocks_[block.header.block_index] = block;

                // Update session info from first block
                if (block.header.block_index == 0 || total_blocks_ == 0) {
                    total_blocks_ = block.header.total_blocks;
                    session_id_ = block.header.session_id;
                }
            }
        } else {
            crc_errors_++;
        }

        if (block.fec_corrections > 0) {
            fec_corrections_ += block.fec_corrections;
        }
    }

    return result;
}

bool CudaDecoder::detect_sync(const uint8_t* d_frame, FrameBounds& bounds) {
    // Launch sync detection kernel
    launch_sync_detect(d_frame, FRAME_WIDTH, FRAME_HEIGHT, d_bounds_, stream_);

    // Download bounds
    int h_bounds[5];
    cudaMemcpyAsync(h_bounds, d_bounds_, 5 * sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    int min_x = h_bounds[0];
    int max_x = h_bounds[1];
    int min_y = h_bounds[2];
    int max_y = h_bounds[3];
    int sync_count = h_bounds[4];

    // Check for valid sync
    if (sync_count < 100 || max_x <= min_x || max_y <= min_y) {
        bounds.valid = false;
        return false;
    }

    // Verify bounds make sense
    int width = max_x - min_x;
    int height = max_y - min_y;

    if (width < 100 || height < 100) {
        bounds.valid = false;
        return false;
    }

    bounds.left = min_x;
    bounds.top = min_y;
    bounds.right = max_x;
    bounds.bottom = max_y;
    bounds.valid = true;
    bounds.confidence = std::min(1.0f, sync_count / 1000.0f);

    return true;
}

bool CudaDecoder::decode_block(const uint8_t* bytes, int size, DecodedBlock& block) {
    // Minimum size: header + crc
    if (size < HEADER_SIZE + CRC_SIZE) {
        return false;
    }

    // Parse header
    block.header = BlockHeader::from_bytes(bytes);

    // Validate header
    if (block.header.payload_size > size - HEADER_SIZE - CRC_SIZE) {
        return false;
    }

    if (block.header.block_index >= block.header.total_blocks) {
        return false;
    }

    // Extract payload
    int payload_offset = HEADER_SIZE;
    int payload_size = block.header.payload_size;
    block.payload.assign(bytes + payload_offset, bytes + payload_offset + payload_size);

    // Extract CRC (after payload)
    int crc_offset = payload_offset + payload_size;
    std::memcpy(&block.crc32, bytes + crc_offset, 4);

    // Verify CRC (over header + payload)
    uint32_t computed_crc = CRC32::compute(bytes, payload_offset + payload_size);
    block.crc_valid = (computed_crc == block.crc32);

    return true;
}

bool CudaDecoder::get_block(int block_index, DecodedBlock& block) const {
    auto it = blocks_.find(block_index);
    if (it == blocks_.end()) {
        return false;
    }
    block = it->second;
    return true;
}

bool CudaDecoder::is_complete() const {
    if (total_blocks_ == 0) return false;
    return blocks_.size() >= static_cast<size_t>(total_blocks_);
}

std::vector<int> CudaDecoder::get_missing_blocks() const {
    std::vector<int> missing;
    for (int i = 0; i < total_blocks_; i++) {
        if (blocks_.find(i) == blocks_.end()) {
            missing.push_back(i);
        }
    }
    return missing;
}

std::vector<uint8_t> CudaDecoder::assemble_file() {
    std::vector<uint8_t> data;

    if (total_blocks_ == 0 || blocks_.empty()) {
        return data;
    }

    int file_size = 0;
    if (blocks_.count(0)) {
        file_size = blocks_.at(0).header.file_size;
    }

    for (int i = 0; i < total_blocks_; i++) {
        auto it = blocks_.find(i);
        if (it == blocks_.end()) {
            std::cerr << "Missing block " << i << std::endl;
            continue;
        }

        const auto& block = it->second;
        const auto& payload = block.payload;

        // First block contains metadata to skip
        if (i == 0) {
            if (payload.size() >= 36) {
                uint32_t filename_len;
                std::memcpy(&filename_len, payload.data() + 32, 4);
                size_t metadata_size = 36 + filename_len;
                if (metadata_size < payload.size()) {
                    data.insert(data.end(),
                                payload.begin() + metadata_size,
                                payload.end());
                }
            }
        } else {
            data.insert(data.end(), payload.begin(), payload.end());
        }
    }

    // Trim to file size
    if (file_size > 0 && data.size() > static_cast<size_t>(file_size)) {
        data.resize(file_size);
    }

    return data;
}

bool CudaDecoder::get_expected_hash(uint8_t* hash_out) const {
    auto it = blocks_.find(0);
    if (it == blocks_.end()) {
        return false;
    }

    const auto& payload = it->second.payload;
    if (payload.size() < 32) {
        return false;
    }

    // First 32 bytes of block 0 payload is SHA-256 hash
    std::memcpy(hash_out, payload.data(), 32);
    return true;
}

std::string CudaDecoder::get_original_filename() const {
    auto it = blocks_.find(0);
    if (it == blocks_.end()) {
        return "";
    }

    const auto& payload = it->second.payload;
    if (payload.size() < 36) {
        return "";
    }

    // Filename length at offset 32, filename starts at 36
    uint32_t filename_len;
    std::memcpy(&filename_len, payload.data() + 32, 4);

    if (payload.size() < 36 + filename_len) {
        return "";
    }

    return std::string(reinterpret_cast<const char*>(payload.data() + 36), filename_len);
}

DecodeStats CudaDecoder::get_stats() const {
    DecodeStats stats;
    stats.total_frames = frames_processed_;
    stats.synced_frames = frames_synced_;
    stats.decoded_blocks = static_cast<int>(blocks_.size());
    stats.unique_blocks = static_cast<int>(blocks_.size());
    stats.crc_errors = crc_errors_;
    stats.fec_corrections = fec_corrections_;
    stats.processing_time = 0;  // Calculated by caller
    stats.fps = 0;
    return stats;
}

void CudaDecoder::reset() {
    blocks_.clear();
    total_blocks_ = 0;
    session_id_ = 0;
    frames_processed_ = 0;
    frames_synced_ = 0;
    crc_errors_ = 0;
    fec_corrections_ = 0;
}

// CRC32 implementation
uint32_t CRC32::compute(const uint8_t* data, size_t length) {
    static uint32_t table[256];
    static bool initialized = false;

    if (!initialized) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int j = 0; j < 8; j++) {
                crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
            }
            table[i] = crc;
        }
        initialized = true;
    }

    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

bool CRC32::verify(const uint8_t* data, size_t length, uint32_t expected_crc) {
    return compute(data, length) == expected_crc;
}

}  // namespace vdd
