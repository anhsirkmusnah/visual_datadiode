#!/usr/bin/env python3
"""
Visual Data Diode - Integrity Test Suite

Tests the complete encode-decode pipeline with integrity verification.
Creates test files of various sizes, encodes them, decodes with CUDA decoder,
and verifies SHA-256 hash matches.

Usage:
    python test_integrity.py --size 10MB
    python test_integrity.py --size 500MB
    python test_integrity.py --size 1GB
    python test_integrity.py --all
"""

import os
import sys
import time
import hashlib
import argparse
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_encode import BatchEncoder
from shared import PROFILE_STANDARD, PROFILE_CONSERVATIVE


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    size_bytes: int
    encode_time: float
    decode_time: float
    video_frames: int
    video_duration: float
    input_hash: str
    output_hash: str
    integrity_passed: bool
    decode_fps: float
    throughput_kbps: float
    error: Optional[str] = None


class IntegrityTestSuite:
    """Runs integrity tests on the visual data diode system."""

    # Size presets
    SIZE_PRESETS = {
        '10MB': 10 * 1024 * 1024,
        '50MB': 50 * 1024 * 1024,
        '100MB': 100 * 1024 * 1024,
        '500MB': 500 * 1024 * 1024,
        '1GB': 1024 * 1024 * 1024,
    }

    def __init__(self, test_dir: str = "test_results", profile: str = "standard"):
        """Initialize test suite."""
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Select profile
        if profile == "conservative":
            self.profile = PROFILE_CONSERVATIVE
        else:
            self.profile = PROFILE_STANDARD

        # Find CUDA decoder
        self.decoder_path = self._find_decoder()
        if not self.decoder_path:
            raise RuntimeError("CUDA decoder not found. Build it first.")

        print(f"Test directory: {self.test_dir}")
        print(f"Profile: {self.profile.name}")
        print(f"Decoder: {self.decoder_path}")

    def _find_decoder(self) -> Optional[Path]:
        """Find the CUDA decoder executable."""
        candidates = [
            Path(__file__).parent / "decoder_cuda/build/Release/vdd_decode.exe",
            Path(__file__).parent / "decoder_cuda/build/vdd_decode.exe",
            Path(__file__).parent / "decoder_cuda/build/Debug/vdd_decode.exe",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def create_test_file(self, size_bytes: int, name: str) -> Tuple[Path, str]:
        """
        Create a test file with random data.

        Returns: (file_path, sha256_hash)
        """
        file_path = self.test_dir / f"{name}_input.bin"

        print(f"Creating test file: {file_path.name} ({size_bytes / 1024 / 1024:.1f} MB)")

        # Generate random data in chunks
        chunk_size = 1024 * 1024  # 1 MB chunks
        hasher = hashlib.sha256()

        with open(file_path, 'wb') as f:
            remaining = size_bytes
            while remaining > 0:
                chunk = os.urandom(min(chunk_size, remaining))
                f.write(chunk)
                hasher.update(chunk)
                remaining -= len(chunk)

                # Progress
                progress = (size_bytes - remaining) * 100 // size_bytes
                if progress % 10 == 0:
                    print(f"\r  Generating: {progress}%", end="", flush=True)

        print(f"\r  Generated: {size_bytes:,} bytes")

        return file_path, hasher.hexdigest()

    def encode_file(self, input_path: Path, output_name: str) -> Tuple[Path, float, int, float]:
        """
        Encode a file to video.

        Returns: (video_path, encode_time, total_frames, video_duration)
        """
        output_path = self.test_dir / output_name

        print(f"Encoding to video...")

        encoder = BatchEncoder(
            profile=self.profile,
            fps=60,
            repeat_count=2,  # Each block shown twice for redundancy
            sync_frames=30,
            end_frames=60,
        )

        start_time = time.time()
        stats = encoder.encode_file(str(input_path), str(output_path))
        encode_time = time.time() - start_time

        video_path = Path(stats.output_file)

        return video_path, encode_time, stats.total_frames, stats.video_duration

    def decode_file(self, video_path: Path, output_dir: Path) -> Tuple[Path, float, str, bool]:
        """
        Decode a video using the CUDA decoder.

        Returns: (output_path, decode_time, output_hash, integrity_passed)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Decoding with CUDA decoder...")

        start_time = time.time()

        # Run CUDA decoder
        result = subprocess.run(
            [
                str(self.decoder_path),
                "-i", str(video_path),
                "-o", str(output_dir),
                "-p", self.profile.name
            ],
            capture_output=True,
            text=True
        )

        decode_time = time.time() - start_time

        # Print decoder output
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        # Check for integrity result in output
        integrity_passed = "Integrity check: PASSED" in result.stdout

        # Find output file
        output_files = list(output_dir.glob("*"))
        if not output_files:
            return None, decode_time, "", False

        output_path = output_files[0]

        # Compute hash of output
        hasher = hashlib.sha256()
        with open(output_path, 'rb') as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        output_hash = hasher.hexdigest()

        return output_path, decode_time, output_hash, integrity_passed

    def run_test(self, size_name: str, size_bytes: int) -> TestResult:
        """Run a complete encode-decode-verify test."""
        print(f"\n{'='*60}")
        print(f"TEST: {size_name} ({size_bytes / 1024 / 1024:.1f} MB)")
        print(f"{'='*60}")

        test_subdir = self.test_dir / size_name
        test_subdir.mkdir(parents=True, exist_ok=True)

        try:
            # Create test file
            input_path, input_hash = self.create_test_file(size_bytes, size_name)
            print(f"  Input SHA-256: {input_hash}")

            # Encode
            video_path, encode_time, total_frames, video_duration = self.encode_file(
                input_path, f"{size_name}_video"
            )
            print(f"  Video: {video_path.name} ({total_frames} frames, {video_duration:.1f}s)")
            print(f"  Encode time: {encode_time:.1f}s")

            # Decode
            output_dir = test_subdir / "decoded"
            output_path, decode_time, output_hash, integrity_passed = self.decode_file(
                video_path, output_dir
            )

            if output_path is None:
                return TestResult(
                    name=size_name,
                    size_bytes=size_bytes,
                    encode_time=encode_time,
                    decode_time=decode_time,
                    video_frames=total_frames,
                    video_duration=video_duration,
                    input_hash=input_hash,
                    output_hash="",
                    integrity_passed=False,
                    decode_fps=0,
                    throughput_kbps=0,
                    error="No output file produced"
                )

            # Calculate metrics
            decode_fps = total_frames / decode_time if decode_time > 0 else 0
            throughput_kbps = (size_bytes * 8 / video_duration / 1000) if video_duration > 0 else 0

            # Verify hash match
            hash_match = input_hash == output_hash

            print(f"\n  Output SHA-256: {output_hash}")
            print(f"  Hash match: {'YES' if hash_match else 'NO'}")
            print(f"  Integrity check: {'PASSED' if integrity_passed else 'FAILED'}")
            print(f"  Decode FPS: {decode_fps:.1f}")
            print(f"  Throughput: {throughput_kbps:.1f} kbps")

            # Move files to test subdir for inspection
            shutil.move(str(input_path), str(test_subdir / input_path.name))
            shutil.move(str(video_path), str(test_subdir / video_path.name))

            return TestResult(
                name=size_name,
                size_bytes=size_bytes,
                encode_time=encode_time,
                decode_time=decode_time,
                video_frames=total_frames,
                video_duration=video_duration,
                input_hash=input_hash,
                output_hash=output_hash,
                integrity_passed=integrity_passed and hash_match,
                decode_fps=decode_fps,
                throughput_kbps=throughput_kbps,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return TestResult(
                name=size_name,
                size_bytes=size_bytes,
                encode_time=0,
                decode_time=0,
                video_frames=0,
                video_duration=0,
                input_hash="",
                output_hash="",
                integrity_passed=False,
                decode_fps=0,
                throughput_kbps=0,
                error=str(e)
            )

    def run_all_tests(self, sizes: list) -> list:
        """Run tests for all specified sizes."""
        results = []

        for size_name in sizes:
            if size_name in self.SIZE_PRESETS:
                size_bytes = self.SIZE_PRESETS[size_name]
            else:
                # Parse size like "100MB"
                if size_name.upper().endswith('GB'):
                    size_bytes = int(float(size_name[:-2]) * 1024 * 1024 * 1024)
                elif size_name.upper().endswith('MB'):
                    size_bytes = int(float(size_name[:-2]) * 1024 * 1024)
                elif size_name.upper().endswith('KB'):
                    size_bytes = int(float(size_name[:-2]) * 1024)
                else:
                    size_bytes = int(size_name)

            result = self.run_test(size_name, size_bytes)
            results.append(result)

        return results

    def print_summary(self, results: list):
        """Print summary of all test results."""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")

        all_passed = True

        for r in results:
            status = "PASS" if r.integrity_passed else "FAIL"
            if r.error:
                status = f"ERROR: {r.error[:30]}"

            print(f"\n{r.name}:")
            print(f"  Size: {r.size_bytes / 1024 / 1024:.1f} MB")
            print(f"  Encode time: {r.encode_time:.1f}s")
            print(f"  Decode time: {r.decode_time:.1f}s")
            print(f"  Decode FPS: {r.decode_fps:.1f}")
            print(f"  Throughput: {r.throughput_kbps:.1f} kbps")
            print(f"  Status: {status}")

            if not r.integrity_passed:
                all_passed = False
                if r.input_hash != r.output_hash:
                    print(f"  Input hash:  {r.input_hash}")
                    print(f"  Output hash: {r.output_hash}")

        print(f"\n{'='*80}")
        print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print(f"{'='*80}")

        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Visual Data Diode Integrity Test Suite")
    parser.add_argument(
        '--size',
        action='append',
        help='Size to test (10MB, 100MB, 500MB, 1GB, or custom like 50MB)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all preset sizes (10MB, 500MB, 1GB)'
    )
    parser.add_argument(
        '--small',
        action='store_true',
        help='Run small test only (10MB)'
    )
    parser.add_argument(
        '--profile',
        choices=['conservative', 'standard'],
        default='standard',
        help='Encoding profile (default: standard)'
    )
    parser.add_argument(
        '--output-dir',
        default='test_results',
        help='Output directory for test files'
    )

    args = parser.parse_args()

    # Determine which sizes to test
    if args.all:
        sizes = ['10MB', '500MB', '1GB']
    elif args.small:
        sizes = ['10MB']
    elif args.size:
        sizes = args.size
    else:
        sizes = ['10MB']  # Default to small test

    print(f"\n{'='*60}")
    print("VISUAL DATA DIODE - INTEGRITY TEST SUITE")
    print(f"{'='*60}")
    print(f"Tests to run: {', '.join(sizes)}")
    print(f"Profile: {args.profile}")
    print(f"Output dir: {args.output_dir}")

    try:
        suite = IntegrityTestSuite(args.output_dir, args.profile)
        results = suite.run_all_tests(sizes)
        all_passed = suite.print_summary(results)

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
