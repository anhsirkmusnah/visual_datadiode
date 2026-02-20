#!/usr/bin/env python3
"""
Visual Data Diode - HPC Parallel Encoder

High-performance encoder using:
- Multiprocessing (ProcessPoolExecutor) for true parallelism
- Enhanced 8-luma + 4-color encoding
- NVENC/QuickSync/libx264 for H.264 encoding
- Direct pipe to FFmpeg (no temp files)
- Batch processing for maximum throughput

Optimized for multi-core systems with NVIDIA GPU.

Usage:
    python encode_parallel_hpc.py input_file.bin output.mp4
    python encode_parallel_hpc.py input_file.bin output.mp4 --workers 16 --batch-size 100
"""

import os
import sys
import argparse
import hashlib
import time
import struct
import subprocess
import tempfile
import mmap
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import threading
import queue

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    ENHANCED_PROFILE_CONSERVATIVE, ENHANCED_PROFILE_STANDARD,
    EnhancedEncodingProfile,
    EnhancedFrameEncoder,
    pack_data_for_enhanced_frame,
    FRAME_WIDTH, FRAME_HEIGHT,
    Block, BlockHeader, BlockFlags, FileMetadata,
    SimpleFEC, compute_file_hash, crc16,
    HEADER_SIZE, CRC_SIZE, DEFAULT_FEC_RATIO
)


def detect_hw_encoder() -> Tuple[str, str]:
    """Detect best available H.264 encoder."""
    encoders = [
        ('h264_nvenc', 'NVIDIA NVENC'),
        ('h264_qsv', 'Intel QuickSync'),
        ('h264_amf', 'AMD AMF'),
        ('libx264', 'CPU libx264'),
    ]

    for codec, name in encoders:
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True, timeout=5
            )
            if codec in result.stdout:
                test_cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.1',
                    '-c:v', codec, '-f', 'null', '-'
                ]
                test = subprocess.run(test_cmd, capture_output=True, timeout=10)
                if test.returncode == 0:
                    return codec, name
        except:
            continue

    return 'libx264', 'CPU libx264 (fallback)'


# Worker function for multiprocessing (must be at module level)
def _encode_block_worker(args: Tuple) -> Tuple[int, bytes]:
    """
    Worker function for encoding a single block.
    Returns just the packed block data (not full frame) to save memory.

    Args:
        args: Tuple of (block_idx, payload, header_data, profile_dict)

    Returns:
        Tuple of (block_idx, packed_block_data)
    """
    block_idx, payload, header_data, profile_name, cell_size, fec_ratio = args

    # Recreate profile in worker process
    profile = EnhancedEncodingProfile(profile_name, cell_size)
    fec = SimpleFEC(fec_ratio)

    # Calculate capacities
    total_raw_bytes = profile.payload_bytes

    # Create header
    header = BlockHeader(
        session_id=header_data['session_id'],
        block_index=block_idx,
        total_blocks=header_data['total_blocks'],
        file_size=header_data['file_size'],
        payload_size=len(payload),
        flags=header_data['flags'],
        sequence=block_idx & 0xFFFF,
        prev_crc16=0
    )

    # Pack block structure: header + payload + CRC
    import zlib
    header_bytes = header.pack()
    crc = zlib.crc32(header_bytes + payload) & 0xFFFFFFFF
    crc_bytes = struct.pack('<I', crc)
    block_data = header_bytes + payload + crc_bytes

    # Calculate fixed block_data_size (must match decoder)
    block_data_fixed_size = total_raw_bytes
    for _ in range(10):
        parity_needed = fec.parity_size(block_data_fixed_size)
        new_size = total_raw_bytes - parity_needed
        if new_size == block_data_fixed_size:
            break
        block_data_fixed_size = new_size

    # Pad block_data to fixed size before FEC (so decoder can use fixed extraction)
    if len(block_data) < block_data_fixed_size:
        block_data = block_data + bytes(block_data_fixed_size - len(block_data))

    # Compute FEC on padded block_data
    if fec.available and fec_ratio > 0:
        _, parity = fec.encode(block_data)
    else:
        parity = b''

    # Final packed data: padded block_data + parity
    packed = block_data + parity
    # Should already be exactly total_raw_bytes, but pad if needed
    if len(packed) < total_raw_bytes:
        packed = packed + bytes(total_raw_bytes - len(packed))

    # Return just the packed block data (not full frame) - saves ~99.99% memory
    return block_idx, packed


def _block_data_to_frame(packed: bytes, block_idx: int, profile: EnhancedEncodingProfile, encoder: EnhancedFrameEncoder) -> bytes:
    """
    Convert packed block data to a full frame.

    Args:
        packed: Packed block data bytes
        block_idx: Block index for frame encoding
        profile: Encoding profile
        encoder: Frame encoder

    Returns:
        Frame bytes (RGB24)
    """
    # Split into luma and color
    luma_data, color_data = pack_data_for_enhanced_frame(packed, profile)

    # Encode frame
    frame = encoder.encode_data_frame(luma_data, color_data, block_idx)

    return frame.tobytes()


class HPCParallelEncoder:
    """
    High-performance parallel encoder using multiprocessing.

    Features:
    - ProcessPoolExecutor for true parallelism (bypasses GIL)
    - Batch processing for efficient memory usage
    - Enhanced 8-luma + 4-color encoding
    - Ordered output with minimal buffering
    """

    def __init__(
        self,
        profile: EnhancedEncodingProfile = ENHANCED_PROFILE_CONSERVATIVE,
        fps: int = 30,
        crf: int = 18,
        workers: int = None,
        batch_size: int = 100,
        sync_frames: int = 15,
        calibration_frames: int = 5,
        end_frames: int = 30,
        fec_ratio: float = DEFAULT_FEC_RATIO,
        redundancy: int = 1
    ):
        self.profile = profile
        self.fps = fps
        self.crf = crf
        self.workers = workers or max(1, cpu_count() - 1)
        self.batch_size = batch_size
        self.sync_frames = sync_frames
        self.calibration_frames = calibration_frames
        self.end_frames = end_frames
        self.fec_ratio = fec_ratio
        self.redundancy = max(1, redundancy)  # How many times to send each block

        # Initialize encoder for sync/calibration frames
        self.encoder = EnhancedFrameEncoder(profile)

        # Calculate capacity - FEC covers header + payload + CRC (block_data)
        self.total_raw_bytes = profile.payload_bytes
        self.fec = SimpleFEC(fec_ratio)

        # Calculate block_data_size iteratively
        # block_data = header + payload + CRC
        # total = block_data + parity + padding = total_raw_bytes
        block_data_size = self.total_raw_bytes
        for _ in range(10):
            parity_needed = self.fec.parity_size(block_data_size)
            new_block_data_size = self.total_raw_bytes - parity_needed
            if new_block_data_size == block_data_size:
                break
            block_data_size = new_block_data_size

        self.block_data_size = block_data_size
        self.fec_parity_size = self.fec.parity_size(block_data_size)
        # block_data = header + payload + CRC, so payload = block_data - header - CRC
        self.block_capacity = block_data_size - HEADER_SIZE - CRC_SIZE

        # Detect hardware encoder
        self.hw_encoder, self.encoder_name = detect_hw_encoder()

        print(f"HPCParallelEncoder initialized:")
        print(f"  Profile: {profile.name} ({profile.cell_size}px cells)")
        print(f"  Workers: {self.workers} threads")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Encoder: {self.encoder_name}")
        print(f"  Total raw bytes: {self.total_raw_bytes}")
        print(f"  Block data size: {self.block_data_size} (header+payload+CRC)")
        print(f"  FEC parity size: {self.fec_parity_size}")
        print(f"  Capacity: {self.block_capacity} bytes/block")
        print(f"  FPS: {fps}, CRF: {crf}")
        if self.redundancy > 1:
            print(f"  Redundancy: {self.redundancy}x (each block sent {self.redundancy} times)")

    def encode_file(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None
    ) -> Dict:
        """
        Encode file using parallel processing.

        Args:
            input_path: Input file path
            output_path: Output video path
            progress_callback: Optional callback(current, total, fps)

        Returns:
            Dict with encoding stats
        """
        input_path = Path(input_path)
        output_path = Path(output_path).with_suffix('.mp4')

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        start_time = time.time()

        # File info
        file_size = input_path.stat().st_size
        file_hash_bytes = compute_file_hash(str(input_path))
        file_hash = file_hash_bytes.hex()
        filename = input_path.name
        session_id = hash(str(input_path) + str(time.time())) & 0xFFFFFFFF

        # Calculate blocks
        filename_bytes = filename.encode('utf-8')
        metadata_size = 36 + len(filename_bytes)
        block0_capacity = self.block_capacity - metadata_size

        if file_size <= block0_capacity:
            total_blocks = 1
        else:
            remaining = file_size - block0_capacity
            total_blocks = 1 + (remaining + self.block_capacity - 1) // self.block_capacity

        # Account for redundancy: each block is sent redundancy times
        total_data_frames = total_blocks * self.redundancy
        total_frames = self.calibration_frames + self.sync_frames + total_data_frames + self.end_frames

        print(f"\nEncoding: {input_path.name}")
        print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"  Hash: {file_hash}")
        print(f"  Blocks: {total_blocks:,}")
        print(f"  Frames: {total_frames:,}")
        print(f"  Est. duration: {total_frames/self.fps:.1f}s")
        print(f"  Output: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Start FFmpeg
        # Capture stderr for error reporting
        ffmpeg_cmd = self._build_ffmpeg_cmd(output_path)
        ffmpeg = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        frames_written = 0

        try:
            # Write calibration frames
            print(f"  Writing {self.calibration_frames} calibration frames...")
            calibration_frame = self.encoder.create_calibration_frame()
            for _ in range(self.calibration_frames):
                ffmpeg.stdin.write(calibration_frame.tobytes())
                frames_written += 1

            # Write sync frames
            print(f"  Writing {self.sync_frames} sync frames...")
            for i in range(self.sync_frames):
                sync_frame = self.encoder.create_sync_frame(i)
                ffmpeg.stdin.write(sync_frame.tobytes())
                frames_written += 1

            # Read file data
            print(f"  Reading input file...")
            with open(input_path, 'rb') as f:
                file_data = f.read()

            # Prepare all block data
            print(f"  Preparing {total_blocks:,} blocks...")
            blocks_args = self._prepare_blocks(
                file_data, file_hash_bytes, filename, session_id, total_blocks
            )

            # Parallel frame generation with batched output
            # Use ThreadPoolExecutor for better Windows compatibility
            print(f"  Encoding frames with {self.workers} threads...")

            # For interleaved redundancy, we store BLOCK DATA (not frames) in memory
            # Block data is ~283 bytes vs ~6 MB for frames = 99.99% memory savings
            # Then generate frames on-the-fly during writing

            # Store block data in memory (532K blocks × 283 bytes = ~150 MB for 100 MB file)
            all_block_data = [None] * total_blocks
            block_data_size = self.total_raw_bytes  # ~283 bytes per block

            print(f"  Block data buffer: {total_blocks * block_data_size / 1024 / 1024:.1f} MB")

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Process in batches for progress reporting
                for batch_start in range(0, total_blocks, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, total_blocks)
                    batch_args = blocks_args[batch_start:batch_end]

                    # Submit batch
                    futures = {
                        executor.submit(_encode_block_worker, args): args[0]
                        for args in batch_args
                    }

                    # Collect results - store just block data (not frames)
                    for future in as_completed(futures):
                        block_idx, packed_data = future.result()
                        all_block_data[block_idx] = packed_data

                    # Progress during encoding phase
                    pct = batch_end * 100 // total_blocks
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    eta = (total_blocks - batch_end) / rate if rate > 0 else 0
                    print(f"    {pct}% - {batch_end:,}/{total_blocks:,} encoded ({rate:.0f} blocks/s, ETA {eta:.0f}s)")

            # Now write frames in interleaved passes for redundancy
            # Pass 1: blocks 0, 1, 2, ..., N-1
            # Pass 2: blocks 0, 1, 2, ..., N-1 (if redundancy >= 2)
            # Pass 3: blocks 0, 1, 2, ..., N-1 (if redundancy >= 3)
            # This spreads redundant copies temporally for independent failure modes
            print(f"  Writing {total_data_frames:,} data frames ({self.redundancy} passes)...")

            for pass_num in range(self.redundancy):
                if self.redundancy > 1:
                    print(f"    Pass {pass_num + 1}/{self.redundancy}...")
                for block_idx in range(total_blocks):
                    # Generate frame from block data on-the-fly
                    packed_data = all_block_data[block_idx]
                    frame_bytes = _block_data_to_frame(packed_data, block_idx, self.profile, self.encoder)
                    ffmpeg.stdin.write(frame_bytes)
                    frames_written += 1

                    # Progress callback
                    if progress_callback and frames_written % 100 == 0:
                        elapsed = time.time() - start_time
                        fps_rate = frames_written / elapsed if elapsed > 0 else 0
                        progress_callback(frames_written, total_frames, fps_rate)

                    # Periodic progress during writing
                    if frames_written % 10000 == 0:
                        pct = frames_written * 100 // total_data_frames
                        print(f"      {pct}% - {frames_written:,}/{total_data_frames:,} frames written")

            # Free block data memory
            del all_block_data

            # Write end frames
            print(f"  Writing {self.end_frames} end frames...")
            end_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            for _ in range(self.end_frames):
                ffmpeg.stdin.write(end_frame.tobytes())
                frames_written += 1

        except BrokenPipeError as e:
            print(f"\n  ERROR: FFmpeg pipe broke during frame writing!")
            print(f"  Frames written: {frames_written:,}/{total_frames:,}")
            # Try to get FFmpeg's error output
            try:
                ffmpeg.stdin.close()
            except:
                pass
            # Use communicate() to avoid deadlock
            _, stderr_bytes = ffmpeg.communicate()
            stderr_output = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''
            if stderr_output:
                # Print last 500 chars of stderr
                print(f"  FFmpeg stderr (last 500 chars):\n{stderr_output[-500:]}")
            raise RuntimeError(f"FFmpeg pipe broke: {e}. Check disk space and FFmpeg errors above.")
        finally:
            try:
                ffmpeg.stdin.close()
            except:
                pass
            print("  Waiting for FFmpeg to finalize...")
            sys.stdout.flush()
            # Use communicate() instead of wait() to avoid deadlock with stderr pipe
            _, stderr_output = ffmpeg.communicate()

        if ffmpeg.returncode != 0:
            stderr_str = stderr_output.decode('utf-8', errors='replace') if stderr_output else ''
            print(f"  FFmpeg exited with code: {ffmpeg.returncode}")
            if stderr_str:
                print(f"  FFmpeg stderr (last 500 chars):\n{stderr_str[-500:]}")

        encode_time = time.time() - start_time
        output_size = output_path.stat().st_size if output_path.exists() else 0
        video_duration = total_frames / self.fps

        print(f"\n{'='*60}")
        print(f"HPC ENCODING COMPLETE")
        print(f"{'='*60}")
        print(f"  Time: {encode_time:.1f}s")
        print(f"  Speed: {total_blocks / encode_time:.1f} blocks/s ({file_size/encode_time/1024/1024:.2f} MB/s)")
        print(f"  Video: {output_size/1024/1024:.1f} MB, {video_duration:.1f}s duration")
        print(f"  Throughput: {file_size * 8 / video_duration / 1000:.1f} kbps")
        print(f"  Hash: {file_hash}")

        return {
            'input_file': str(input_path),
            'input_size': file_size,
            'input_hash': file_hash,
            'output_file': str(output_path),
            'output_size': output_size,
            'total_blocks': total_blocks,
            'encode_time': encode_time,
            'video_duration': video_duration,
            'throughput_kbps': file_size * 8 / video_duration / 1000
        }

    def _build_ffmpeg_cmd(self, output_path: Path) -> List[str]:
        """Build FFmpeg command line."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
            '-pix_fmt', 'rgb24', '-r', str(self.fps),
            '-i', '-',
            '-c:v', self.hw_encoder,
        ]

        if self.hw_encoder == 'h264_nvenc':
            cmd.extend(['-preset', 'p4', '-rc', 'vbr', '-cq', str(self.crf)])
        elif self.hw_encoder == 'h264_qsv':
            cmd.extend(['-preset', 'medium', '-global_quality', str(self.crf)])
        else:
            if self.crf == 0:
                # Truly lossless encoding with libx264 requires specific flags
                # Use qp=0 and lossless=1 for mathematically lossless output
                cmd.extend(['-preset', 'ultrafast', '-qp', '0'])
                cmd.extend(['-x264-params', 'lossless=1'])
            else:
                cmd.extend(['-preset', 'fast', '-crf', str(self.crf)])

        # Use yuv444p for better quality (especially for lossless)
        pix_fmt = 'yuv444p' if self.crf == 0 else 'yuv420p'
        cmd.extend(['-pix_fmt', pix_fmt, str(output_path)])

        return cmd

    def _prepare_blocks(
        self,
        file_data: bytes,
        file_hash: bytes,
        filename: str,
        session_id: int,
        total_blocks: int
    ) -> List[Tuple]:
        """Prepare all block arguments for parallel processing."""
        blocks_args = []
        offset = 0

        for block_idx in range(total_blocks):
            is_first = (block_idx == 0)
            is_last = (block_idx == total_blocks - 1)

            flags = BlockFlags.NONE
            if is_first:
                flags |= BlockFlags.FIRST_BLOCK
            if is_last:
                flags |= BlockFlags.LAST_BLOCK

            if is_first:
                metadata = FileMetadata(file_hash=file_hash, filename=filename)
                metadata_bytes = metadata.pack()
                capacity = self.block_capacity - len(metadata_bytes)
                payload = metadata_bytes + file_data[offset:offset + capacity]
                offset += capacity
            else:
                payload = file_data[offset:offset + self.block_capacity]
                offset += self.block_capacity

            header_data = {
                'session_id': session_id,
                'total_blocks': total_blocks,
                'file_size': len(file_data),
                'flags': flags
            }

            # Arguments for worker function
            args = (
                block_idx,
                payload,
                header_data,
                self.profile.name,
                self.profile.cell_size,
                self.fec_ratio
            )
            blocks_args.append(args)

        return blocks_args


def main():
    parser = argparse.ArgumentParser(description="HPC Parallel encoder for Visual Data Diode")
    parser.add_argument('input', help='Input file to encode')
    parser.add_argument('output', help='Output video file (.mp4)')
    parser.add_argument('--profile', choices=['enhanced_conservative', 'enhanced_standard'],
                        default='enhanced_conservative', help='Encoding profile')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS (default: 30)')
    parser.add_argument('--crf', type=int, default=18, help='H.264 CRF quality (default: 18)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')

    args = parser.parse_args()

    profiles = {
        'enhanced_conservative': ENHANCED_PROFILE_CONSERVATIVE,
        'enhanced_standard': ENHANCED_PROFILE_STANDARD
    }

    encoder = HPCParallelEncoder(
        profile=profiles[args.profile],
        fps=args.fps,
        crf=args.crf,
        workers=args.workers,
        batch_size=args.batch_size
    )

    encoder.encode_file(args.input, args.output)


if __name__ == "__main__":
    main()
