/**
 * Visual Data Diode - Video Reader
 *
 * Reads video frames using OpenCV.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace vdd {

class VideoReader {
public:
    VideoReader() : is_open_(false), total_frames_(0), fps_(0), width_(0), height_(0) {}

    ~VideoReader() {
        close();
    }

    bool open(const std::string& path) {
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

    void close() {
        if (is_open_) {
            cap_.release();
            is_open_ = false;
        }
    }

    // Read next frame in BGR format (OpenCV default)
    bool read_frame_bgr(cv::Mat& frame) {
        if (!is_open_) return false;
        return cap_.read(frame);
    }

    // Read next frame and convert to RGB
    bool read_frame_rgb(cv::Mat& frame) {
        cv::Mat bgr;
        if (!read_frame_bgr(bgr)) return false;
        cv::cvtColor(bgr, frame, cv::COLOR_BGR2RGB);
        return true;
    }

    // Read frame into provided RGB buffer
    bool read_frame_rgb(uint8_t* buffer, int buffer_size) {
        cv::Mat bgr;
        if (!read_frame_bgr(bgr)) return false;

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        int frame_size = rgb.total() * rgb.elemSize();
        if (buffer_size < frame_size) {
            std::cerr << "Buffer too small: " << buffer_size << " < " << frame_size << std::endl;
            return false;
        }

        std::memcpy(buffer, rgb.data, frame_size);
        return true;
    }

    // Get current frame position
    int get_position() const {
        return static_cast<int>(cap_.get(cv::CAP_PROP_POS_FRAMES));
    }

    // Seek to frame
    bool seek(int frame_idx) {
        if (!is_open_) return false;
        return cap_.set(cv::CAP_PROP_POS_FRAMES, frame_idx);
    }

    // Getters
    bool is_open() const { return is_open_; }
    int total_frames() const { return total_frames_; }
    double fps() const { return fps_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int frame_size() const { return width_ * height_ * 3; }

private:
    cv::VideoCapture cap_;
    bool is_open_;
    int total_frames_;
    double fps_;
    int width_;
    int height_;
};

}  // namespace vdd
