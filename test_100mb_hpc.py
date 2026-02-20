#!/usr/bin/env python3
"""
Visual Data Diode - 100 MB HPC/CUDA Test

End-to-end test with:
- 100 MB test file
- HPC parallel encoding (multiprocessing)
- CUDA GPU decoding (if available)

Usage:
    python test_100mb_hpc.py
    python test_100mb_hpc.py --size 50  # 50 MB instead
    python test_100mb_hpc.py --no-cuda  # Disable CUDA
"""

import os
import sys
import argparse
import time
import hashlib
import tempfile
from pathlib import Path
from multiprocessing import cpu_count

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    ENHANCED_PROFILE_CONSERVATIVE,
    check_cuda_available, get_cuda_info,
    compute_file_hash, check_fec_available, check_crypto_available
)
from shared.enhanced_encoding import EnhancedEncodingProfile

# Create a very robust profile with 48px cells (much more reliable at cost of throughput)
ENHANCED_PROFILE_ROBUST = EnhancedEncodingProfile("enhanced_very_robust", cell_size=48)


def create_test_file(size_mb: int, output_path: str) -> str:
    """
    Create a test file with random data.

    Args:
        size_mb: File size in megabytes
        output_path: Output file path

    Returns:
        SHA-256 hash of the file
    """
    print(f"Creating {size_mb} MB test file: {output_path}")

    # Generate random data in chunks
    chunk_size = 1024 * 1024  # 1 MB
    total_bytes = size_mb * chunk_size

    hasher = hashlib.sha256()

    import numpy as np
    np.random.seed(42)  # Fixed seed for reproducible tests

    with open(output_path, 'wb') as f:
        bytes_written = 0
        while bytes_written < total_bytes:
            # Use random data in safe range (16-239) to avoid extreme luma issues
            chunk = np.random.randint(16, 240, size=chunk_size, dtype=np.uint8).tobytes()
            f.write(chunk)
            hasher.update(chunk)
            bytes_written += chunk_size

            # Progress
            pct = bytes_written * 100 // total_bytes
            if pct % 10 == 0 and bytes_written % (10 * chunk_size) == 0:
                print(f"  {pct}% created...")

    file_hash = hasher.hexdigest()
    print(f"  File created: {output_path}")
    print(f"  SHA-256: {file_hash}")

    return file_hash


def test_hpc_encoding(input_path: str, output_path: str, workers: int = None, batch_size: int = 100, redundancy: int = 3, crf: int = 0) -> dict:
    """
    Test HPC parallel encoding.

    Args:
        input_path: Input file path
        output_path: Output video path
        workers: Number of worker processes
        batch_size: Batch size for processing

    Returns:
        Encoding stats
    """
    from encode_parallel_hpc import HPCParallelEncoder

    print(f"\n{'='*60}")
    print("HPC PARALLEL ENCODING TEST")
    print(f"{'='*60}")

    # Use ROBUST profile (48px cells) with 30% FEC for strong error correction
    encoder = HPCParallelEncoder(
        profile=ENHANCED_PROFILE_ROBUST,
        fps=30,
        crf=crf,  # 0=lossless, 18-23=fast lossy
        workers=workers or max(1, cpu_count() - 1),
        batch_size=batch_size,
        fec_ratio=0.30,  # 30% FEC parity - stronger error correction
        redundancy=redundancy  # Configurable redundancy (1-3x)
    )

    result = encoder.encode_file(input_path, output_path)

    return result


def test_cuda_decoding(video_path: str, output_dir: str, use_cuda: bool = True, batch_size: int = 32) -> dict:
    """
    Test CUDA-accelerated decoding.

    Args:
        video_path: Input video path
        output_dir: Output directory
        use_cuda: Whether to use CUDA
        batch_size: Batch size for GPU processing

    Returns:
        Decoding stats
    """
    from process_enhanced import EnhancedVideoProcessor

    print(f"\n{'='*60}")
    print("DECODING TEST" + (" (CUDA)" if use_cuda and check_cuda_available() else " (CPU)"))
    print(f"{'='*60}")

    processor = EnhancedVideoProcessor(
        video_path=video_path,
        output_dir=output_dir,
        fec_ratio=0.30,  # 30% FEC parity - must match encoder
        profile=ENHANCED_PROFILE_ROBUST
    )

    # Override decoder if CUDA available
    if use_cuda and check_cuda_available():
        from shared import CUDAFrameDecoder
        processor.decoder = CUDAFrameDecoder(processor.profile)
        print(f"Using CUDA decoder on GPU: {get_cuda_info()['device_name']}")
    else:
        print("Using CPU decoder")

    result = processor.process()

    return {
        'success': result.success,
        'files_decoded': len(result.files_decoded),
        'total_frames': result.total_frames,
        'processing_time': result.processing_time,
        'message': result.message,
        'decoded_files': result.files_decoded
    }


def verify_output(original_hash: str, decoded_path: str) -> bool:
    """
    Verify decoded file matches original.

    Args:
        original_hash: SHA-256 hash of original file
        decoded_path: Path to decoded file

    Returns:
        True if hashes match
    """
    print(f"\nVerifying: {decoded_path}")

    if not os.path.exists(decoded_path):
        print("  ERROR: Decoded file not found!")
        return False

    # Compute hash of decoded file
    hasher = hashlib.sha256()
    with open(decoded_path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)

    decoded_hash = hasher.hexdigest()

    print(f"  Original hash: {original_hash}")
    print(f"  Decoded hash:  {decoded_hash}")

    if original_hash == decoded_hash:
        print("  MATCH - File transferred successfully!")
        return True
    else:
        print("  MISMATCH - Data corruption detected!")
        return False


def main():
    parser = argparse.ArgumentParser(description="100 MB HPC/CUDA test for Visual Data Diode")
    parser.add_argument('--size', type=int, default=100, help='Test file size in MB (default: 100)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for encoding')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA decoding')
    parser.add_argument('--keep-files', action='store_true', help='Keep test files after completion')
    parser.add_argument('--redundancy', type=int, default=3, help='Block redundancy (1-3, default: 3)')
    parser.add_argument('--crf', type=int, default=0, help='H.264 CRF (0=lossless, 18-23=fast lossy, default: 0)')

    args = parser.parse_args()

    print("="*60)
    print("VISUAL DATA DIODE - 100 MB HPC/CUDA TEST")
    print("="*60)
    print()

    # System info
    print("System Configuration:")
    print(f"  CPU cores: {cpu_count()}")
    print(f"  Workers: {args.workers or max(1, cpu_count() - 1)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  CUDA available: {check_cuda_available()}")
    if check_cuda_available():
        cuda_info = get_cuda_info()
        print(f"  GPU: {cuda_info['device_name']}")
        print(f"  GPU Memory: {cuda_info['memory_free'] / 1024**3:.1f} GB free")
    print(f"  FEC available: {check_fec_available()}")
    print(f"  Crypto available: {check_crypto_available()}")
    print()

    # Profile info
    profile = ENHANCED_PROFILE_ROBUST
    print("Encoding Profile:")
    info = profile.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="vdd_test_")
    print(f"Temp directory: {temp_dir}")
    print()

    try:
        # Create test file
        input_path = os.path.join(temp_dir, f"test_{args.size}mb.bin")
        original_hash = create_test_file(args.size, input_path)

        # HPC Encoding
        video_path = os.path.join(temp_dir, "encoded.mp4")
        encode_result = test_hpc_encoding(
            input_path,
            video_path,
            workers=args.workers,
            batch_size=args.batch_size,
            redundancy=args.redundancy,
            crf=args.crf
        )

        # Decoding
        output_dir = os.path.join(temp_dir, "decoded")
        os.makedirs(output_dir, exist_ok=True)

        decode_result = test_cuda_decoding(
            video_path,
            output_dir,
            use_cuda=not args.no_cuda,
            batch_size=32
        )

        # Verification
        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}")

        success = False
        if decode_result['files_decoded'] > 0:
            for decoded_file in decode_result['decoded_files']:
                decoded_path = decoded_file.output_path
                if verify_output(original_hash, decoded_path):
                    success = True
                    break
        else:
            print("No files decoded!")

        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Test file size: {args.size} MB")
        print(f"  Encoding time: {encode_result['encode_time']:.1f}s")
        print(f"  Encoding throughput: {encode_result['input_size'] / encode_result['encode_time'] / 1024 / 1024:.2f} MB/s")
        print(f"  Decoding time: {decode_result['processing_time']:.1f}s")
        print(f"  Files decoded: {decode_result['files_decoded']}")
        print(f"  Video duration: {encode_result['video_duration']:.1f}s")
        print(f"  Video throughput: {encode_result['throughput_kbps']:.1f} kbps")
        print()

        if success:
            print("TEST PASSED - File transferred and verified successfully!")
        else:
            print("TEST FAILED - Data corruption or incomplete transfer!")

        return 0 if success else 1

    finally:
        # Cleanup
        if not args.keep_files:
            print(f"\nCleaning up temp directory: {temp_dir}")
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        else:
            print(f"\nTest files kept in: {temp_dir}")


if __name__ == "__main__":
    sys.exit(main())
