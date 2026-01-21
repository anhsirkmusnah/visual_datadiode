"""
Visual Data Diode - Record and Decode Test Suite

Tests the record-then-decode approach with various configurations
to find optimal settings for reliable transfer.
"""

import cv2
import numpy as np
import pygame
import time
import sys
import os
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    EncodingProfile, FRAME_WIDTH, FRAME_HEIGHT
)

# Configuration
CAPTURE_DEVICE = 3
DISPLAY_INDEX = 3

# Test profiles
PROFILES = {
    'conservative': PROFILE_CONSERVATIVE,  # 16px cells
    'standard': PROFILE_STANDARD,           # 10px cells
    'aggressive': PROFILE_AGGRESSIVE,       # 8px cells
    'ultra': PROFILE_ULTRA                  # 6px cells
}


@dataclass
class TestConfig:
    """Configuration for a single test."""
    profile_name: str
    cell_size: int
    fps: int
    repeat_count: int
    file_size: int


@dataclass
class TestResult:
    """Result of a single test."""
    config: TestConfig
    success: bool
    bytes_correct: int
    bytes_total: int
    error_rate: float
    frames_recorded: int
    frames_synced: int
    blocks_decoded: int
    blocks_expected: int
    record_time: float
    decode_time: float
    throughput_bps: float
    message: str = ""


def create_test_data(size_bytes: int) -> bytes:
    """Create predictable test data."""
    return bytes([i % 256 for i in range(size_bytes)])


def create_test_video(
    data: bytes,
    output_path: str,
    profile: EncodingProfile,
    fps: int,
    repeat_count: int,
    sync_frames: int = 30,
    end_frames: int = 60
) -> Tuple[bool, int]:
    """
    Create a test video file with encoded data.

    Returns:
        (success, total_frames)
    """
    from sender.video_prerender import VideoPreRenderer

    # Write data to temp file
    temp_file = Path(output_path).with_suffix('.bin')
    temp_file.write_bytes(data)

    try:
        renderer = VideoPreRenderer(
            output_path=output_path,
            profile=profile,
            fps=fps,
            repeat_count=repeat_count,
            sync_frames=sync_frames,
            end_frames=end_frames
        )

        stats = renderer.prerender_file(str(temp_file))
        return True, stats.total_frames

    except Exception as e:
        print(f"Pre-render error: {e}")
        return False, 0

    finally:
        temp_file.unlink(missing_ok=True)


def play_and_record(
    video_path: str,
    record_path: str,
    display_index: int,
    capture_device: int,
    record_fps: int = 60
) -> Tuple[bool, int, float]:
    """
    Play video on display while recording from capture.

    Returns:
        (success, frames_recorded, duration)
    """
    from receiver.video_recorder import VideoRecorder

    # Open video
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False, 0, 0

    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Playing: {total_frames} frames @ {video_fps:.1f} FPS")

    # Initialize pygame
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()

    if display_index >= len(sizes):
        video_cap.release()
        pygame.quit()
        return False, 0, 0

    x_offset = sum(sizes[i][0] for i in range(display_index))
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},0"

    disp_w, disp_h = sizes[display_index]
    screen = pygame.display.set_mode((disp_w, disp_h), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    # Open capture
    capture_cap = cv2.VideoCapture(capture_device, cv2.CAP_MSMF)
    capture_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture_cap.set(cv2.CAP_PROP_FPS, record_fps)
    capture_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture_cap.isOpened():
        video_cap.release()
        pygame.quit()
        return False, 0, 0

    # Flush stale frames
    for _ in range(10):
        capture_cap.grab()

    # Initialize recorder
    recorder = VideoRecorder(
        output_path=record_path,
        width=1920,
        height=1080,
        fps=record_fps
    )

    if not recorder.start():
        video_cap.release()
        capture_cap.release()
        pygame.quit()
        return False, 0, 0

    # Play and record
    frame_time = 1.0 / video_fps
    start_time = time.time()

    try:
        while True:
            frame_start = time.perf_counter()

            # Read video frame
            ret, frame = video_cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display
            if frame_rgb.shape[1] != disp_w or frame_rgb.shape[0] != disp_h:
                frame_rgb = cv2.resize(frame_rgb, (disp_w, disp_h))

            surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Capture and record
            cap_ret, captured = capture_cap.read()
            if cap_ret:
                recorder.add_frame(captured)

            # Timing
            elapsed = time.perf_counter() - frame_start
            sleep_time = frame_time - elapsed - 0.001
            if sleep_time > 0:
                time.sleep(sleep_time)

            while time.perf_counter() - frame_start < frame_time:
                pass

            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    break

    finally:
        pygame.quit()
        video_cap.release()
        capture_cap.release()

    stats = recorder.stop()
    duration = time.time() - start_time

    return True, stats.frames_recorded, duration


def decode_recording(
    record_path: str,
    profile: EncodingProfile
) -> Tuple[bytes, int, int, int, float]:
    """
    Decode a recorded video.

    Returns:
        (decoded_data, frames_synced, blocks_decoded, blocks_expected, decode_time)
    """
    from receiver.video_decoder import VideoDecoder

    start_time = time.time()

    decoder = VideoDecoder(
        video_path=record_path,
        profile=profile
    )

    stats, data = decoder.decode_all()
    decode_time = time.time() - start_time

    return data, stats.synced_frames, stats.unique_blocks, stats.total_blocks_expected, decode_time


def run_single_test(config: TestConfig, temp_dir: str) -> TestResult:
    """Run a single test with given configuration."""
    profile = PROFILES[config.profile_name]

    print(f"\n{'='*60}")
    print(f"Testing: {config.profile_name} ({config.cell_size}px), "
          f"{config.fps} FPS, repeat={config.repeat_count}, "
          f"{config.file_size/1024:.1f}KB")
    print('='*60)

    # Create test data
    original_data = create_test_data(config.file_size)
    expected_hash = hashlib.sha256(original_data).hexdigest()

    # Paths
    prerender_path = os.path.join(temp_dir, f"prerender_{config.profile_name}")
    record_path = os.path.join(temp_dir, f"record_{config.profile_name}")

    # Step 1: Pre-render
    print("\n1. Pre-rendering...")
    success, total_frames = create_test_video(
        data=original_data,
        output_path=prerender_path,
        profile=profile,
        fps=config.fps,
        repeat_count=config.repeat_count
    )

    if not success:
        return TestResult(
            config=config,
            success=False,
            bytes_correct=0,
            bytes_total=config.file_size,
            error_rate=1.0,
            frames_recorded=0,
            frames_synced=0,
            blocks_decoded=0,
            blocks_expected=0,
            record_time=0,
            decode_time=0,
            throughput_bps=0,
            message="Pre-render failed"
        )

    # Find actual video file
    prerender_video = None
    for ext in ['.mkv', '.avi']:
        p = Path(prerender_path).with_suffix(ext)
        if p.exists():
            prerender_video = str(p)
            break

    if not prerender_video:
        return TestResult(
            config=config,
            success=False,
            bytes_correct=0,
            bytes_total=config.file_size,
            error_rate=1.0,
            frames_recorded=0,
            frames_synced=0,
            blocks_decoded=0,
            blocks_expected=0,
            record_time=0,
            decode_time=0,
            throughput_bps=0,
            message="Pre-rendered video not found"
        )

    # Step 2: Play and Record
    print("\n2. Playing and recording...")
    success, frames_recorded, record_time = play_and_record(
        video_path=prerender_video,
        record_path=record_path,
        display_index=DISPLAY_INDEX,
        capture_device=CAPTURE_DEVICE,
        record_fps=60
    )

    if not success:
        return TestResult(
            config=config,
            success=False,
            bytes_correct=0,
            bytes_total=config.file_size,
            error_rate=1.0,
            frames_recorded=0,
            frames_synced=0,
            blocks_decoded=0,
            blocks_expected=0,
            record_time=record_time,
            decode_time=0,
            throughput_bps=0,
            message="Play/record failed"
        )

    # Find recorded video
    record_video = None
    for ext in ['.mkv', '.avi']:
        p = Path(record_path).with_suffix(ext)
        if p.exists():
            record_video = str(p)
            break

    if not record_video:
        return TestResult(
            config=config,
            success=False,
            bytes_correct=0,
            bytes_total=config.file_size,
            error_rate=1.0,
            frames_recorded=frames_recorded,
            frames_synced=0,
            blocks_decoded=0,
            blocks_expected=0,
            record_time=record_time,
            decode_time=0,
            throughput_bps=0,
            message="Recorded video not found"
        )

    # Step 3: Decode
    print("\n3. Decoding recording...")
    decoded_data, frames_synced, blocks_decoded, blocks_expected, decode_time = decode_recording(
        record_path=record_video,
        profile=profile
    )

    # Verify
    decoded_data = decoded_data[:config.file_size]  # Trim to exact size
    bytes_correct = sum(1 for a, b in zip(decoded_data, original_data) if a == b)
    error_rate = 1.0 - (bytes_correct / config.file_size) if config.file_size > 0 else 0

    total_time = record_time + decode_time
    throughput = (config.file_size * 8) / total_time if total_time > 0 else 0

    success = decoded_data == original_data

    return TestResult(
        config=config,
        success=success,
        bytes_correct=bytes_correct,
        bytes_total=config.file_size,
        error_rate=error_rate,
        frames_recorded=frames_recorded,
        frames_synced=frames_synced,
        blocks_decoded=blocks_decoded,
        blocks_expected=blocks_expected,
        record_time=record_time,
        decode_time=decode_time,
        throughput_bps=throughput,
        message="OK" if success else f"{config.file_size - bytes_correct} byte errors"
    )


def run_optimization_tests() -> List[TestResult]:
    """
    Run tests with various configurations to find optimal settings.
    """
    # Test configurations
    configs = []

    # Test different cell sizes with standard settings
    for profile_name, profile in PROFILES.items():
        configs.append(TestConfig(
            profile_name=profile_name,
            cell_size=profile.cell_size,
            fps=30,
            repeat_count=2,
            file_size=10240  # 10KB
        ))

    # Test different FPS rates with conservative profile
    for fps in [15, 30, 60]:
        configs.append(TestConfig(
            profile_name='conservative',
            cell_size=16,
            fps=fps,
            repeat_count=2,
            file_size=10240
        ))

    # Test different repeat counts
    for repeat in [1, 2, 3]:
        configs.append(TestConfig(
            profile_name='standard',
            cell_size=10,
            fps=30,
            repeat_count=repeat,
            file_size=10240
        ))

    # Test larger file sizes with best settings
    for size in [102400, 1048576]:  # 100KB, 1MB
        configs.append(TestConfig(
            profile_name='conservative',
            cell_size=16,
            fps=30,
            repeat_count=2,
            file_size=size
        ))

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for config in configs:
            result = run_single_test(config, temp_dir)
            results.append(result)

            # Brief pause between tests
            time.sleep(1)

    return results


def print_results(results: List[TestResult]):
    """Print test results summary."""
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    # Table header
    print(f"\n{'Profile':<12} {'Cell':<6} {'FPS':<6} {'Rep':<5} {'Size':<10} "
          f"{'Status':<8} {'Errors':<10} {'Throughput':<12}")
    print("-"*80)

    for r in results:
        size_str = f"{r.config.file_size/1024:.1f}KB" if r.config.file_size >= 1024 else f"{r.config.file_size}B"
        status = "PASS" if r.success else "FAIL"
        errors = f"{r.error_rate*100:.2f}%" if not r.success else "0%"
        throughput = f"{r.throughput_bps/1000:.1f} kbps"

        print(f"{r.config.profile_name:<12} {r.config.cell_size:<6} {r.config.fps:<6} "
              f"{r.config.repeat_count:<5} {size_str:<10} {status:<8} {errors:<10} {throughput:<12}")

    # Summary
    passed = sum(1 for r in results if r.success)
    print(f"\nTotal: {passed}/{len(results)} tests passed")

    # Best configuration
    successful = [r for r in results if r.success]
    if successful:
        best = max(successful, key=lambda r: r.throughput_bps)
        print(f"\nBest configuration: {best.config.profile_name}, "
              f"{best.config.cell_size}px cells, {best.config.fps} FPS, "
              f"repeat={best.config.repeat_count}")
        print(f"  Throughput: {best.throughput_bps/1000:.1f} kbps")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Record and decode optimization tests")
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with minimal settings')
    parser.add_argument('--profile', choices=list(PROFILES.keys()),
                        help='Test specific profile only')
    parser.add_argument('--size', type=int, default=10240,
                        help='File size to test (bytes)')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Frame repeat count (default: 3)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Playback FPS (default: 30)')

    args = parser.parse_args()

    if args.quick:
        # Single quick test
        config = TestConfig(
            profile_name='conservative',
            cell_size=16,
            fps=args.fps,
            repeat_count=args.repeat,
            file_size=args.size
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_single_test(config, temp_dir)

        print(f"\nResult: {'PASS' if result.success else 'FAIL'}")
        print(f"  Errors: {result.error_rate*100:.2f}%")
        print(f"  Throughput: {result.throughput_bps/1000:.1f} kbps")

    else:
        results = run_optimization_tests()
        print_results(results)


if __name__ == "__main__":
    main()
