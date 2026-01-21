/**
 * Visual Data Diode - Block Assembler
 *
 * Assembles decoded blocks into the original file.
 */

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>

#include "block_header.h"

namespace vdd {

class BlockAssembler {
public:
    BlockAssembler() : total_blocks_(0), file_size_(0), session_id_(0) {}

    // Add a decoded block
    bool add_block(const DecodedBlock& block) {
        if (!block.crc_valid) {
            return false;
        }

        int idx = block.header.block_index;

        // First block initializes session info
        if (blocks_.empty() || idx == 0) {
            session_id_ = block.header.session_id;
            total_blocks_ = block.header.total_blocks;
            file_size_ = block.header.file_size;
        }

        // Verify session
        if (block.header.session_id != session_id_) {
            std::cerr << "Session ID mismatch: " << block.header.session_id
                      << " != " << session_id_ << std::endl;
            return false;
        }

        // Store or update block
        if (blocks_.find(idx) == blocks_.end()) {
            blocks_[idx] = block;
            return true;
        }

        // Already have this block
        return false;
    }

    // Check if all blocks received
    bool is_complete() const {
        if (total_blocks_ == 0) return false;
        return blocks_.size() >= static_cast<size_t>(total_blocks_);
    }

    // Get missing block indices
    std::vector<int> get_missing() const {
        std::vector<int> missing;
        for (int i = 0; i < total_blocks_; i++) {
            if (blocks_.find(i) == blocks_.end()) {
                missing.push_back(i);
            }
        }
        return missing;
    }

    // Get completion percentage
    float get_completion() const {
        if (total_blocks_ == 0) return 0.0f;
        return static_cast<float>(blocks_.size()) / total_blocks_ * 100.0f;
    }

    // Assemble file data from blocks
    std::vector<uint8_t> assemble() const {
        std::vector<uint8_t> data;

        for (int i = 0; i < total_blocks_; i++) {
            auto it = blocks_.find(i);
            if (it == blocks_.end()) {
                std::cerr << "Missing block " << i << std::endl;
                continue;
            }

            const auto& block = it->second;
            const auto& payload = block.payload;

            // First block contains metadata
            if (i == 0) {
                // Skip metadata: 32 bytes hash + 4 bytes filename_len + filename
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

        // Trim to actual file size
        if (data.size() > static_cast<size_t>(file_size_)) {
            data.resize(file_size_);
        }

        return data;
    }

    // Extract file metadata from block 0
    bool get_file_info(std::string& filename, std::vector<uint8_t>& file_hash) const {
        auto it = blocks_.find(0);
        if (it == blocks_.end()) {
            return false;
        }

        const auto& payload = it->second.payload;
        if (payload.size() < 36) {
            return false;
        }

        // Extract hash (32 bytes)
        file_hash.assign(payload.begin(), payload.begin() + 32);

        // Extract filename
        uint32_t filename_len;
        std::memcpy(&filename_len, payload.data() + 32, 4);

        if (36 + filename_len <= payload.size()) {
            filename.assign(
                reinterpret_cast<const char*>(payload.data() + 36),
                filename_len
            );
            return true;
        }

        return false;
    }

    // Save assembled file
    bool save(const std::string& output_path) const {
        auto data = assemble();
        if (data.empty()) {
            std::cerr << "No data to save" << std::endl;
            return false;
        }

        std::ofstream file(output_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open output file: " << output_path << std::endl;
            return false;
        }

        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();

        std::cout << "Saved " << data.size() << " bytes to " << output_path << std::endl;
        return true;
    }

    // Reset state
    void reset() {
        blocks_.clear();
        total_blocks_ = 0;
        file_size_ = 0;
        session_id_ = 0;
    }

    // Getters
    int total_blocks() const { return total_blocks_; }
    int received_blocks() const { return static_cast<int>(blocks_.size()); }
    int file_size() const { return file_size_; }
    uint32_t session_id() const { return session_id_; }
    const std::map<int, DecodedBlock>& blocks() const { return blocks_; }

private:
    std::map<int, DecodedBlock> blocks_;
    int total_blocks_;
    int file_size_;
    uint32_t session_id_;
};

}  // namespace vdd
