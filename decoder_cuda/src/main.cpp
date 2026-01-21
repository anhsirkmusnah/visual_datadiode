/**
 * Visual Data Diode - CUDA Decoder CLI
 *
 * Command-line tool to decode visual data diode videos.
 *
 * Usage:
 *   vdd_decode --input video.avi --output ./received/
 *   vdd_decode --input video.avi --output decoded.bin --profile standard
 */

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "cuda_decoder.h"
#include "constants.h"
#include "sha256.h"

namespace fs = std::filesystem;

// Forward declarations from video_reader.cpp
namespace vdd {
class VideoReader {
public:
    VideoReader();
    ~VideoReader();
    bool open(const std::string& path);
    void close();
    bool read_frame_rgb(uint8_t* buffer, int buffer_size);
    bool is_open() const;
    int total_frames() const;
    double fps() const;
    int width() const;
    int height() const;
    int frame_size() const;

private:
    cv::VideoCapture cap_;
    bool is_open_;
    int total_frames_;
    double fps_;
    int width_;
    int height_;
};

// Implementation
VideoReader::VideoReader() : is_open_(false), total_frames_(0), fps_(0), width_(0), height_(0) {}
VideoReader::~VideoReader() { close(); }

bool VideoReader::open(const std::string& path) {
    cap_.open(path);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open video: " << path << std::endl;
        return false;
    }
    total_frames_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Video opened: " << path << std::endl;
    std::cout << "  Resolution: " << width_ << "x" << height_ << std::endl;
    std::cout << "  FPS: " << fps_ << ", Frames: " << total_frames_ << std::endl;
    std::cout << "  Duration: " << (total_frames_ / fps_) << "s" << std::endl;

    is_open_ = true;
    return true;
}

void VideoReader::close() {
    if (is_open_) {
        cap_.release();
        is_open_ = false;
    }
}

bool VideoReader::read_frame_rgb(uint8_t* buffer, int buffer_size) {
    cv::Mat bgr;
    if (!cap_.read(bgr)) return false;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    int frame_size = rgb.total() * rgb.elemSize();
    if (buffer_size < frame_size) return false;

    std::memcpy(buffer, rgb.data, frame_size);
    return true;
}

bool VideoReader::is_open() const { return is_open_; }
int VideoReader::total_frames() const { return total_frames_; }
double VideoReader::fps() const { return fps_; }
int VideoReader::width() const { return width_; }
int VideoReader::height() const { return height_; }
int VideoReader::frame_size() const { return width_ * height_ * 3; }
}

void print_usage(const char* program) {
    std::cout << "Visual Data Diode - CUDA Decoder\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --input, -i <path>     Input video file (required)\n";
    std::cout << "  --output, -o <path>    Output file or directory (required)\n";
    std::cout << "  --profile, -p <name>   Encoding profile: conservative, standard, aggressive, ultra\n";
    std::cout << "                         (default: standard)\n";
    std::cout << "  --help, -h             Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program << " -i captured.avi -o decoded.bin\n";
    std::cout << "  " << program << " -i captured.avi -o ./output/ -p conservative\n";
}

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string output_path;
    std::string profile_name = "standard";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) input_path = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--profile" || arg == "-p") {
            if (i + 1 < argc) profile_name = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Validate arguments
    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: --input and --output are required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Select profile
    vdd::EncodingProfile profile = vdd::PROFILE_STANDARD;
    if (profile_name == "conservative") {
        profile = vdd::PROFILE_CONSERVATIVE;
    } else if (profile_name == "standard") {
        profile = vdd::PROFILE_STANDARD;
    } else if (profile_name == "aggressive") {
        profile = vdd::PROFILE_AGGRESSIVE;
    } else if (profile_name == "ultra") {
        profile = vdd::PROFILE_ULTRA;
    } else {
        std::cerr << "Unknown profile: " << profile_name << std::endl;
        return 1;
    }

    // Check input file
    if (!fs::exists(input_path)) {
        std::cerr << "Input file not found: " << input_path << std::endl;
        return 1;
    }

    // Open video
    vdd::VideoReader reader;
    if (!reader.open(input_path)) {
        return 1;
    }

    // Initialize CUDA decoder
    vdd::CudaDecoder decoder(profile);
    if (!decoder.initialize()) {
        std::cerr << "Failed to initialize CUDA decoder" << std::endl;
        return 1;
    }

    // Allocate frame buffer
    std::vector<uint8_t> frame_buffer(reader.frame_size());

    // Decode loop
    std::cout << "\nDecoding " << reader.total_frames() << " frames...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    int last_progress = -1;

    while (reader.read_frame_rgb(frame_buffer.data(), frame_buffer.size())) {
        auto result = decoder.decode_frame(frame_buffer.data(), reader.width(), reader.height());

        // Print progress
        int progress = decoder.get_stats().total_frames * 100 / reader.total_frames();
        if (progress != last_progress && progress % 10 == 0) {
            last_progress = progress;
            std::cout << "  " << progress << "% - "
                      << decoder.get_blocks().size() << " blocks decoded\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Get stats
    auto stats = decoder.get_stats();
    double fps = stats.total_frames / (duration.count() / 1000.0);

    std::cout << "\nDecoding complete:\n";
    std::cout << "  Frames: " << stats.total_frames << " total, " << stats.synced_frames << " synced\n";
    std::cout << "  Blocks: " << stats.unique_blocks << "/" << decoder.get_blocks().size();
    if (!decoder.get_blocks().empty()) {
        std::cout << " (session " << decoder.get_blocks().begin()->second.header.session_id << ")";
    }
    std::cout << "\n";
    std::cout << "  CRC errors: " << stats.crc_errors << ", FEC corrections: " << stats.fec_corrections << "\n";
    std::cout << "  Time: " << (duration.count() / 1000.0) << "s (" << fps << " FPS)\n";

    // Check completeness
    if (!decoder.is_complete()) {
        auto missing = decoder.get_missing_blocks();
        std::cout << "Warning: Missing " << missing.size() << " blocks\n";
        if (missing.size() <= 10) {
            std::cout << "  Missing: ";
            for (int idx : missing) std::cout << idx << " ";
            std::cout << "\n";
        }
    }

    // Assemble and save
    auto data = decoder.assemble_file();
    if (data.empty()) {
        std::cerr << "No data to save\n";
        return 1;
    }

    // Get expected hash from metadata
    uint8_t expected_hash[32];
    bool has_expected_hash = decoder.get_expected_hash(expected_hash);

    // Compute actual hash
    uint8_t actual_hash[32];
    vdd::SHA256::hash(data, actual_hash);

    // Verify integrity
    bool integrity_ok = false;
    if (has_expected_hash) {
        integrity_ok = vdd::SHA256::compare(expected_hash, actual_hash);
        std::cout << "\nIntegrity check: " << (integrity_ok ? "PASSED" : "FAILED") << "\n";
        if (!integrity_ok) {
            std::cout << "  Expected: " << vdd::SHA256::to_hex(expected_hash, 32) << "\n";
            std::cout << "  Actual:   " << vdd::SHA256::to_hex(actual_hash, 32) << "\n";
        }
    } else {
        std::cout << "\nIntegrity check: SKIPPED (no hash in metadata)\n";
    }

    // Determine output filename
    std::string final_output = output_path;
    if (fs::is_directory(output_path)) {
        // Use original filename from metadata if available
        std::string orig_name = decoder.get_original_filename();
        if (orig_name.empty()) {
            orig_name = fs::path(input_path).stem().string() + "_decoded.bin";
        }
        final_output = (fs::path(output_path) / orig_name).string();
    }

    // Save file
    std::ofstream out_file(final_output, std::ios::binary);
    if (!out_file) {
        std::cerr << "Failed to open output file: " << final_output << std::endl;
        return 1;
    }
    out_file.write(reinterpret_cast<const char*>(data.data()), data.size());
    out_file.close();

    std::cout << "\nSaved " << data.size() << " bytes to " << final_output << "\n";
    std::cout << "SHA-256: " << vdd::SHA256::to_hex(actual_hash, 32) << "\n";

    // Calculate throughput
    double video_duration = reader.total_frames() / reader.fps();
    double throughput_bps = (data.size() * 8) / video_duration;
    std::cout << "Effective throughput: " << (throughput_bps / 1000) << " kbps\n";

    // Return error code if integrity check failed
    if (has_expected_hash && !integrity_ok) {
        std::cerr << "ERROR: File integrity verification failed!\n";
        return 2;
    }

    return 0;
}
