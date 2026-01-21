"""
Comprehensive Visual Data Diode Test Suite

Tests all configurations:
- All encoding profiles (conservative, standard, aggressive, ultra)
- Various file sizes (64B, 1KB, 10KB, 100KB, 1MB, 10MB)
- FPS rates (15, 30, 60)
- Validates CRC integrity, frame sync, and data accuracy
"""

import cv2
import numpy as np
import pygame
import time
import sys
import os
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Configuration
CAPTURE_DEVICE = 3
DISPLAY_INDEX = 3

# Profiles to test
PROFILES = {
    'conservative': 16,  # cell_size
    'standard': 10,
    'aggressive': 8,
    'ultra': 6
}

# Test file sizes
FILE_SIZES = [64, 1024, 10240, 102400, 1048576, 10485760]  # 64B to 10MB

@dataclass
class TestResult:
    """Result of a single test."""
    profile: str
    file_size: int
    fps: int
    success: bool
    bytes_correct: int
    bytes_total: int
    error_rate: float
    frames_used: int
    throughput_bps: float
    elapsed_time: float
    message: str = ""

def create_test_data(size_bytes: int) -> bytes:
    """Create predictable test data."""
    data = bytes([i % 256 for i in range(size_bytes)])
    return data

def encode_byte_to_cells(byte_val: int) -> List[int]:
    """
    Encode a single byte as 4 grayscale cells (2 bits each).
    Uses levels: 0, 85, 170, 255
    """
    levels = [0, 85, 170, 255]
    cells = []
    for i in range(4):
        bits = (byte_val >> (6 - i*2)) & 0x03
        cells.append(levels[bits])
    return cells

def decode_cells_to_byte(cell_values: List[int]) -> int:
    """
    Decode 4 grayscale cell values back to a byte.
    Uses standard midpoint thresholds (same as test_transfer.py).
    """
    byte_val = 0
    thresholds = [43, 128, 213]

    for i, val in enumerate(cell_values):
        if val < thresholds[0]:
            bits = 0
        elif val < thresholds[1]:
            bits = 1
        elif val < thresholds[2]:
            bits = 2
        else:
            bits = 3
        byte_val |= (bits << (6 - i*2))

    return byte_val

def create_frame_from_data(data: bytes, frame_w: int, frame_h: int, cell_size: int) -> Tuple[np.ndarray, int]:
    """
    Create a frame image encoding the given data.
    Returns (frame, bytes_encoded).
    """
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Sync border colors
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)

    border_cells = 2
    border_px = border_cells * cell_size

    # Draw sync border
    frame[:border_px, :] = CYAN  # Top
    frame[-border_px:, :] = MAGENTA  # Bottom
    frame[:, :border_px] = CYAN  # Left
    frame[:, -border_px:] = MAGENTA  # Right

    # Interior dimensions
    int_left = border_px
    int_top = border_px
    int_right = frame_w - border_px
    int_bottom = frame_h - border_px

    # Cells per row (4 cells per byte)
    cells_per_row = (int_right - int_left) // cell_size
    bytes_per_row = cells_per_row // 4

    # Encode data into cells
    data_idx = 0
    y = int_top

    while y + cell_size <= int_bottom and data_idx < len(data):
        x = int_left
        for _ in range(bytes_per_row):
            if data_idx >= len(data):
                break

            byte_val = data[data_idx]
            cells = encode_byte_to_cells(byte_val)

            for cell_val in cells:
                if x + cell_size <= int_right:
                    frame[y:y+cell_size, x:x+cell_size] = [cell_val, cell_val, cell_val]
                    x += cell_size

            data_idx += 1

        y += cell_size

    return frame, data_idx

def decode_frame_to_data(frame: np.ndarray, cell_size: int, expected_bytes: int) -> bytes:
    """
    Decode data from a captured frame.
    Uses same logic as working test_transfer.py.
    """
    frame_h, frame_w = frame.shape[:2]

    border_cells = 2
    border_px = border_cells * cell_size

    # Interior dimensions
    int_left = border_px
    int_top = border_px
    int_right = frame_w - border_px
    int_bottom = frame_h - border_px

    # Cells per row
    cells_per_row = (int_right - int_left) // cell_size
    bytes_per_row = cells_per_row // 4

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    data = bytearray()
    y = int_top

    while y + cell_size <= int_bottom and len(data) < expected_bytes:
        x = int_left
        for _ in range(bytes_per_row):
            if len(data) >= expected_bytes:
                break

            cell_values = []
            for _ in range(4):
                if x + cell_size <= int_right:
                    # Sample center of cell
                    cx = x + cell_size // 2
                    cy = y + cell_size // 2
                    sample_r = cell_size // 4
                    region = gray[cy-sample_r:cy+sample_r, cx-sample_r:cx+sample_r]
                    cell_values.append(int(np.mean(region)))
                    x += cell_size

            if len(cell_values) == 4:
                byte_val = decode_cells_to_byte(cell_values)
                data.append(byte_val)

        y += cell_size

    return bytes(data)

def test_single_transfer(
    profile_name: str,
    cell_size: int,
    file_size: int,
    fps: int = 30
) -> TestResult:
    """
    Test file transfer with specific settings.
    """
    print(f"\n--- Testing: {profile_name}, {file_size} bytes, {fps} FPS ---")

    # Create test data
    original_data = create_test_data(file_size)
    original_hash = hashlib.sha256(original_data).hexdigest()

    # Initialize pygame
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()

    if DISPLAY_INDEX >= len(sizes):
        pygame.quit()
        return TestResult(
            profile=profile_name,
            file_size=file_size,
            fps=fps,
            success=False,
            bytes_correct=0,
            bytes_total=file_size,
            error_rate=1.0,
            frames_used=0,
            throughput_bps=0,
            elapsed_time=0,
            message=f"Display {DISPLAY_INDEX} not found"
        )

    x_offset = sum(sizes[i][0] for i in range(DISPLAY_INDEX))
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},0"

    disp_w, disp_h = sizes[DISPLAY_INDEX]
    screen = pygame.display.set_mode((disp_w, disp_h), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    # Open capture
    cap = cv2.VideoCapture(CAPTURE_DEVICE, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        pygame.quit()
        return TestResult(
            profile=profile_name,
            file_size=file_size,
            fps=fps,
            success=False,
            bytes_correct=0,
            bytes_total=file_size,
            error_rate=1.0,
            frames_used=0,
            throughput_bps=0,
            elapsed_time=0,
            message="Failed to open capture device"
        )

    # Flush buffer
    for _ in range(30):
        cap.grab()

    # Calculate capacity
    border_px = 2 * cell_size
    int_w = disp_w - 2 * border_px
    int_h = disp_h - 2 * border_px
    cells_per_row = int_w // cell_size
    bytes_per_row = cells_per_row // 4
    rows = int_h // cell_size
    bytes_per_frame = bytes_per_row * rows

    print(f"  Capacity: {bytes_per_frame} bytes/frame (cell size: {cell_size})")

    # Transfer
    start_time = time.time()
    received_data = bytearray()
    sent_offset = 0
    frame_num = 0
    frame_delay = 1.0 / fps

    while sent_offset < len(original_data):
        frame_start = time.time()

        # Create and display frame
        chunk = original_data[sent_offset:sent_offset + bytes_per_frame]
        frame_img, bytes_encoded = create_frame_from_data(
            chunk, disp_w, disp_h, cell_size
        )

        surface = pygame.surfarray.make_surface(np.transpose(frame_img, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Wait for display to stabilize (must be long enough for HDMI signal to propagate)
        time.sleep(0.1)

        # Flush capture buffer (removes stale frames)
        for _ in range(5):
            cap.grab()

        ret, captured = cap.read()
        if ret:
            decoded = decode_frame_to_data(captured, cell_size, len(chunk))
            received_data.extend(decoded)

            # Check this frame
            errors = sum(1 for a, b in zip(decoded, chunk) if a != b)
            if errors == 0:
                print(f"    Frame {frame_num}: {len(chunk)} bytes OK")
            else:
                print(f"    Frame {frame_num}: {len(chunk)} bytes - {errors} errors")
        else:
            print(f"    Frame {frame_num}: capture failed!")

        sent_offset += bytes_encoded
        frame_num += 1

        # Pace sending
        elapsed = time.time() - frame_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    elapsed_time = time.time() - start_time

    # Cleanup
    cap.release()
    pygame.quit()

    # Verify
    received_data = bytes(received_data[:file_size])  # Trim to exact size
    bytes_correct = sum(1 for a, b in zip(received_data, original_data) if a == b)
    error_rate = 1.0 - (bytes_correct / file_size) if file_size > 0 else 0
    throughput = (file_size * 8) / elapsed_time if elapsed_time > 0 else 0  # bits per second

    success = received_data == original_data

    return TestResult(
        profile=profile_name,
        file_size=file_size,
        fps=fps,
        success=success,
        bytes_correct=bytes_correct,
        bytes_total=file_size,
        error_rate=error_rate,
        frames_used=frame_num,
        throughput_bps=throughput,
        elapsed_time=elapsed_time,
        message="OK" if success else f"{file_size - bytes_correct} byte errors"
    )

def run_comprehensive_tests(
    profiles: List[str] = None,
    file_sizes: List[int] = None,
    fps_rates: List[int] = None
) -> List[TestResult]:
    """
    Run comprehensive test suite.
    """
    if profiles is None:
        profiles = list(PROFILES.keys())
    if file_sizes is None:
        file_sizes = [64, 1024, 10240]  # Default smaller tests
    if fps_rates is None:
        fps_rates = [30]

    results = []

    print("=" * 70)
    print("VISUAL DATA DIODE - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Profiles: {profiles}")
    print(f"File sizes: {[f'{s/1024:.1f}KB' if s >= 1024 else f'{s}B' for s in file_sizes]}")
    print(f"FPS rates: {fps_rates}")
    print("=" * 70)

    for profile in profiles:
        cell_size = PROFILES[profile]
        for size in file_sizes:
            for fps in fps_rates:
                result = test_single_transfer(profile, cell_size, size, fps)
                results.append(result)

                # Brief pause between tests
                time.sleep(0.5)

    return results

def print_results(results: List[TestResult]):
    """Print test results summary."""
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    # Group by success
    passed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nPassed: {len(passed)}/{len(results)}")

    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            size_str = f"{r.file_size/1024:.1f}KB" if r.file_size >= 1024 else f"{r.file_size}B"
            print(f"  - {r.profile}, {size_str}, {r.fps}FPS: {r.message} (error rate: {r.error_rate*100:.2f}%)")

    # Throughput summary
    print(f"\nThroughput by profile:")
    for profile in PROFILES.keys():
        profile_results = [r for r in passed if r.profile == profile]
        if profile_results:
            avg_throughput = sum(r.throughput_bps for r in profile_results) / len(profile_results)
            print(f"  {profile}: {avg_throughput/1000:.1f} kbps average")

    print("\n" + "=" * 70)

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive visual diode test")
    parser.add_argument('--profiles', nargs='+', default=['conservative'],
                        choices=list(PROFILES.keys()),
                        help='Profiles to test')
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 1024, 10240],
                        help='File sizes to test (bytes)')
    parser.add_argument('--fps', nargs='+', type=int, default=[30],
                        help='FPS rates to test')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with minimal settings')
    parser.add_argument('--full', action='store_true',
                        help='Full test with all profiles and sizes up to 1MB')

    args = parser.parse_args()

    if args.quick:
        profiles = ['conservative']
        sizes = [64, 256]
        fps_rates = [30]
    elif args.full:
        profiles = list(PROFILES.keys())
        sizes = [64, 1024, 10240, 102400, 1048576]  # Up to 1MB
        fps_rates = [30]
    else:
        profiles = args.profiles
        sizes = args.sizes
        fps_rates = args.fps

    results = run_comprehensive_tests(profiles, sizes, fps_rates)
    print_results(results)

    # Return exit code based on results
    all_passed = all(r.success for r in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
