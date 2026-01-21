#!/usr/bin/env python3
"""
Visual Data Diode - Pipeline Test Script

Tests the complete encode-decode pipeline with various file sizes and profiles.
Run this to verify the system works correctly before deployment.
"""

import os
import sys
import time
import hashlib
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_encode import BatchEncoder
from receiver.video_decoder import VideoDecoder
from shared import (
    PROFILE_CONSERVATIVE, PROFILE_STANDARD,
    PROFILE_AGGRESSIVE, PROFILE_ULTRA
)


def create_test_file(size_bytes: int) -> bytes:
    """Create deterministic test data."""
    return bytes([i % 256 for i in range(size_bytes)])


def compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data).hexdigest()


def test_encode_decode(
    profile,
    file_size: int,
    fps: int = 60,
    repeat: int = 2,
    verbose: bool = True
) -> dict:
    """
    Test encode-decode cycle with given parameters.

    Returns:
        dict with test results
    """
    result = {
        'profile': profile.name,
        'file_size': file_size,
        'fps': fps,
        'repeat': repeat,
        'success': False,
        'encode_time': 0,
        'decode_time': 0,
        'video_size': 0,
        'blocks': 0,
        'frames': 0,
        'throughput_kbps': 0,
        'error': None
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        original_data = create_test_file(file_size)
        original_hash = compute_hash(original_data)

        input_file = Path(temp_dir) / "test_input.bin"
        input_file.write_bytes(original_data)

        video_path = Path(temp_dir) / "test_video"

        try:
            # Encode
            if verbose:
                print(f"\n{'='*60}")
                print(f"Testing: {profile.name}, {file_size/1024:.1f}KB, {fps}FPS, repeat={repeat}")
                print('='*60)

            encoder = BatchEncoder(
                profile=profile,
                fps=fps,
                repeat_count=repeat
            )

            encode_start = time.time()
            stats = encoder.encode_file(str(input_file), str(video_path))
            result['encode_time'] = time.time() - encode_start
            result['video_size'] = stats.output_size
            result['blocks'] = stats.total_blocks
            result['frames'] = stats.total_frames

            # Find video file
            video_file = None
            for ext in ['.avi', '.mp4', '.mkv']:
                p = video_path.with_suffix(ext)
                if p.exists():
                    video_file = p
                    break

            if not video_file:
                result['error'] = "Video file not created"
                return result

            # Decode
            if verbose:
                print(f"\nDecoding...")

            decode_start = time.time()
            decoder = VideoDecoder(str(video_file), profile=profile)
            decode_stats, decoded_data = decoder.decode_all()
            decoder.close()  # Explicitly close to release file handle
            result['decode_time'] = time.time() - decode_start

            # Verify
            decoded_data = decoded_data[:file_size]
            decoded_hash = compute_hash(decoded_data)

            result['success'] = (decoded_hash == original_hash)

            if result['success']:
                video_duration = result['frames'] / fps
                result['throughput_kbps'] = (file_size * 8) / video_duration / 1000
            else:
                # Count correct bytes
                correct = sum(1 for a, b in zip(decoded_data, original_data) if a == b)
                result['error'] = f"Hash mismatch: {correct}/{file_size} bytes correct"

        except Exception as e:
            result['error'] = str(e)

    return result


def run_all_tests(verbose: bool = True):
    """Run comprehensive test suite."""

    profiles = [
        PROFILE_CONSERVATIVE,
        PROFILE_STANDARD,
    ]

    # Test configurations: (file_size, fps, repeat)
    configs = [
        (1024, 60, 2),      # 1 KB - quick test
        (10240, 60, 2),     # 10 KB - standard test
        (102400, 60, 2),    # 100 KB - medium test
    ]

    results = []

    print("\n" + "="*70)
    print("VISUAL DATA DIODE - PIPELINE TEST SUITE")
    print("="*70)

    for profile in profiles:
        for file_size, fps, repeat in configs:
            result = test_encode_decode(
                profile=profile,
                file_size=file_size,
                fps=fps,
                repeat=repeat,
                verbose=verbose
            )
            results.append(result)

            status = "PASS" if result['success'] else "FAIL"
            print(f"\n  Result: {status}")
            if result['error']:
                print(f"  Error: {result['error']}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    print(f"\n{'Profile':<15} {'Size':<10} {'FPS':<6} {'Rep':<5} {'Status':<8} {'Throughput':<12}")
    print("-"*70)

    passed = 0
    for r in results:
        size_str = f"{r['file_size']/1024:.1f}KB"
        status = "PASS" if r['success'] else "FAIL"
        throughput = f"{r['throughput_kbps']:.1f} kbps" if r['success'] else "N/A"

        print(f"{r['profile']:<15} {size_str:<10} {r['fps']:<6} {r['repeat']:<5} {status:<8} {throughput:<12}")

        if r['success']:
            passed += 1

    print("-"*70)
    print(f"Total: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\nAll tests passed! System is ready for deployment.")
    else:
        print("\nSome tests failed. Please check the errors above.")

    return results


def quick_test():
    """Run a single quick test to verify system works."""
    print("Running quick verification test...")

    result = test_encode_decode(
        profile=PROFILE_STANDARD,
        file_size=5000,
        fps=60,
        repeat=2,
        verbose=True
    )

    if result['success']:
        print("\n" + "="*60)
        print("QUICK TEST PASSED")
        print("="*60)
        print(f"  Encode time: {result['encode_time']:.1f}s")
        print(f"  Decode time: {result['decode_time']:.1f}s")
        print(f"  Throughput: {result['throughput_kbps']:.1f} kbps")
        return True
    else:
        print("\n" + "="*60)
        print("QUICK TEST FAILED")
        print("="*60)
        print(f"  Error: {result['error']}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the visual data diode pipeline")
    parser.add_argument('--quick', action='store_true', help='Run quick verification test only')
    parser.add_argument('--profile', choices=['conservative', 'standard', 'aggressive', 'ultra'],
                        help='Test specific profile only')
    parser.add_argument('--size', type=int, default=10240, help='File size for single test')

    args = parser.parse_args()

    if args.quick:
        success = quick_test()
        sys.exit(0 if success else 1)
    elif args.profile:
        profiles = {
            'conservative': PROFILE_CONSERVATIVE,
            'standard': PROFILE_STANDARD,
            'aggressive': PROFILE_AGGRESSIVE,
            'ultra': PROFILE_ULTRA
        }
        result = test_encode_decode(
            profile=profiles[args.profile],
            file_size=args.size,
            verbose=True
        )
        sys.exit(0 if result['success'] else 1)
    else:
        results = run_all_tests()
        passed = sum(1 for r in results if r['success'])
        sys.exit(0 if passed == len(results) else 1)
