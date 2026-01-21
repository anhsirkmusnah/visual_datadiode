#!/usr/bin/env python3
"""
Visual Data Diode - Parallel Encoder

High-performance encoder using:
- Multiprocessing for frame generation (all CPU cores)
- NVENC/QuickSync/libx264 for H.264 encoding
- Direct pipe to FFmpeg (no temp files)

Optimized for Intel i9 + NVIDIA GPU systems.

Usage:
    python encode_parallel.py input_file.bin output.mp4
    python encode_parallel.py input_file.bin output.mp4 --workers 16
"""

import os
import sys
import argparse
import hashlib
import time
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    EncodingProfile, PROFILE_CONSERVATIVE, PROFILE_STANDARD,
    PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    FRAME_WIDTH, FRAME_HEIGHT,
    GRAY_LEVELS, bits_to_gray,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_WHITE, COLOR_BLACK,
    CORNER_TOP_LEFT, CORNER_TOP_RIGHT, CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT,
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


# Pre-computed lookup table
GRAY_LUT = np.array([GRAY_LEVELS[i] for i in range(4)], dtype=np.uint8)


class FrameGenerator:
    """Thread-safe frame generator."""

    def __init__(self, profile: EncodingProfile, fec_ratio: float = DEFAULT_FEC_RATIO):
        self.profile = profile
        self.cell_size = profile.cell_size
        self.grid_width = FRAME_WIDTH // self.cell_size
        self.grid_height = FRAME_HEIGHT // self.cell_size
        self.border_width = 2

        self.interior_left = self.border_width
        self.interior_right = self.grid_width - self.border_width
        self.interior_top = self.border_width
        self.interior_bottom = self.grid_height - self.border_width
        self.interior_width = self.interior_right - self.interior_left
        self.interior_height = self.interior_bottom - self.interior_top

        self.total_cells = self.interior_width * self.interior_height
        self.raw_bytes = self.total_cells // 4

        self.fec = SimpleFEC(fec_ratio)
        self.fec_parity_size = self.fec.parity_size(self.raw_bytes)
        self.block_capacity = self.raw_bytes - HEADER_SIZE - CRC_SIZE - self.fec_parity_size

        # Pre-generate sync grid
        self.sync_grid = self._generate_sync_grid()

    def _generate_sync_grid(self) -> np.ndarray:
        """Generate sync border pattern."""
        grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                is_border = (row < self.border_width or row >= self.grid_height - self.border_width or
                            col < self.border_width or col >= self.grid_width - self.border_width)
                if is_border:
                    if row < 3 and col < 3:
                        continue
                    if row < 3 and col >= self.grid_width - 3:
                        continue
                    if row >= self.grid_height - 3 and col < 3:
                        continue
                    if row >= self.grid_height - 3 and col >= self.grid_width - 3:
                        continue

                    if (row + col) % 2 == 0:
                        grid[row, col] = COLOR_MAGENTA
                    else:
                        grid[row, col] = COLOR_CYAN

        corners = [
            (0, 0, CORNER_TOP_LEFT),
            (0, self.grid_width - 3, CORNER_TOP_RIGHT),
            (self.grid_height - 3, 0, CORNER_BOTTOM_LEFT),
            (self.grid_height - 3, self.grid_width - 3, CORNER_BOTTOM_RIGHT)
        ]
        for start_row, start_col, pattern in corners:
            for i in range(3):
                for j in range(3):
                    color = COLOR_WHITE if pattern[i * 3 + j] else COLOR_BLACK
                    grid[start_row + i, start_col + j] = color

        return grid

    def bytes_to_cells(self, data: bytes) -> np.ndarray:
        """Convert bytes to gray values using vectorized operations."""
        arr = np.frombuffer(data, dtype=np.uint8)
        cells = np.empty(len(arr) * 4, dtype=np.uint8)
        cells[0::4] = GRAY_LUT[(arr >> 6) & 0x03]
        cells[1::4] = GRAY_LUT[(arr >> 4) & 0x03]
        cells[2::4] = GRAY_LUT[(arr >> 2) & 0x03]
        cells[3::4] = GRAY_LUT[arr & 0x03]
        return cells

    def encode_block(self, block_idx: int, payload: bytes, header_data: dict) -> np.ndarray:
        """Encode a single block to frame."""
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

        block = Block(header=header, payload=payload)
        full_data = header.pack() + payload
        _, parity = self.fec.encode(full_data)
        block.fec_parity = parity
        packed = block.pack(include_fec=True)

        # Create grid
        grid = self.sync_grid.copy()
        cells = self.bytes_to_cells(packed)

        # Fill interior
        cell_idx = 0
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if cell_idx < len(cells):
                    gray = cells[cell_idx]
                    grid[row, col] = (gray, gray, gray)
                    cell_idx += 1
                else:
                    grid[row, col] = (128, 128, 128)

        # Expand to frame
        frame = np.repeat(grid, self.cell_size, axis=0)
        frame = np.repeat(frame, self.cell_size, axis=1)
        return frame[:FRAME_HEIGHT, :FRAME_WIDTH]

    def create_sync_frame(self) -> np.ndarray:
        """Create sync-only frame."""
        grid = self.sync_grid.copy()
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if (row + col) % 2 == 0:
                    grid[row, col] = (64, 64, 64)
                else:
                    grid[row, col] = (192, 192, 192)
        frame = np.repeat(grid, self.cell_size, axis=0)
        frame = np.repeat(frame, self.cell_size, axis=1)
        return frame[:FRAME_HEIGHT, :FRAME_WIDTH]

    def create_end_frame(self) -> np.ndarray:
        """Create end frame."""
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


class ParallelEncoder:
    """High-performance parallel encoder using ThreadPoolExecutor."""

    def __init__(
        self,
        profile: EncodingProfile = PROFILE_STANDARD,
        fps: int = 30,
        crf: int = 18,
        workers: int = None,
        sync_frames: int = 15,
        end_frames: int = 30,
        fec_ratio: float = DEFAULT_FEC_RATIO
    ):
        self.profile = profile
        self.fps = fps
        self.crf = crf
        # Use more threads than cores since numpy releases GIL
        self.workers = workers or (os.cpu_count() or 8) * 2
        self.sync_frames = sync_frames
        self.end_frames = end_frames
        self.fec_ratio = fec_ratio

        self.generator = FrameGenerator(profile, fec_ratio)
        self.hw_encoder, self.encoder_name = detect_hw_encoder()

        print(f"ParallelEncoder initialized:")
        print(f"  Profile: {profile.name} ({self.generator.cell_size}px cells)")
        print(f"  Workers: {self.workers} threads")
        print(f"  Encoder: {self.encoder_name}")
        print(f"  Capacity: {self.generator.block_capacity} bytes/block")
        print(f"  FPS: {fps}, CRF: {crf}")

    def encode_file(self, input_path: str, output_path: str):
        """Encode file using parallel processing."""
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
        block0_capacity = self.generator.block_capacity - metadata_size

        if file_size <= block0_capacity:
            total_blocks = 1
        else:
            remaining = file_size - block0_capacity
            total_blocks = 1 + (remaining + self.generator.block_capacity - 1) // self.generator.block_capacity

        total_frames = self.sync_frames + total_blocks + self.end_frames

        print(f"\nEncoding: {input_path.name}")
        print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"  Hash: {file_hash}")
        print(f"  Blocks: {total_blocks:,}")
        print(f"  Frames: {total_frames:,}")
        print(f"  Est. duration: {total_frames/self.fps:.1f}s")
        print(f"  Output: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Start FFmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
            '-pix_fmt', 'rgb24', '-r', str(self.fps),
            '-i', '-',
            '-c:v', self.hw_encoder,
        ]

        if self.hw_encoder == 'h264_nvenc':
            ffmpeg_cmd.extend(['-preset', 'p4', '-rc', 'vbr', '-cq', str(self.crf)])
        elif self.hw_encoder == 'h264_qsv':
            ffmpeg_cmd.extend(['-preset', 'medium', '-global_quality', str(self.crf)])
        else:
            ffmpeg_cmd.extend(['-preset', 'fast', '-crf', str(self.crf)])

        ffmpeg_cmd.extend(['-pix_fmt', 'yuv420p', str(output_path)])

        ffmpeg = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        # Write sync frames
        print(f"  Writing {self.sync_frames} sync frames...")
        sync_frame = self.generator.create_sync_frame()
        for _ in range(self.sync_frames):
            ffmpeg.stdin.write(sync_frame.tobytes())

        # Read file data
        print(f"  Reading input...")
        with open(input_path, 'rb') as f:
            file_data = f.read()

        # Prepare block data
        print(f"  Preparing {total_blocks:,} blocks...")
        blocks_data = []
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
                metadata = FileMetadata(file_hash=file_hash_bytes, filename=filename)
                metadata_bytes = metadata.pack()
                capacity = self.generator.block_capacity - len(metadata_bytes)
                payload = metadata_bytes + file_data[offset:offset + capacity]
                offset += capacity
            else:
                payload = file_data[offset:offset + self.generator.block_capacity]
                offset += self.generator.block_capacity

            header_data = {
                'session_id': session_id,
                'total_blocks': total_blocks,
                'file_size': file_size,
                'flags': flags
            }
            blocks_data.append((block_idx, payload, header_data))

        # Parallel frame generation with ordered output
        print(f"  Encoding frames with {self.workers} threads...")

        frames_written = 0
        frame_buffer = {}
        next_frame = 0
        buffer_lock = threading.Lock()

        def process_block(block_info):
            block_idx, payload, header_data = block_info
            frame = self.generator.encode_block(block_idx, payload, header_data)
            return block_idx, frame

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all blocks
            futures = {executor.submit(process_block, bd): bd[0] for bd in blocks_data}

            for future in as_completed(futures):
                block_idx, frame = future.result()

                with buffer_lock:
                    frame_buffer[block_idx] = frame

                    # Write frames in order
                    while next_frame in frame_buffer:
                        ffmpeg.stdin.write(frame_buffer[next_frame].tobytes())
                        del frame_buffer[next_frame]
                        next_frame += 1
                        frames_written += 1

                        if frames_written % 2000 == 0:
                            pct = frames_written * 100 // total_blocks
                            elapsed = time.time() - start_time
                            rate = frames_written / elapsed
                            eta = (total_blocks - frames_written) / rate if rate > 0 else 0
                            print(f"    {pct}% - {frames_written:,}/{total_blocks:,} ({rate:.0f} fps, ETA {eta:.0f}s)")

        # Write remaining buffered frames
        while next_frame < total_blocks:
            if next_frame in frame_buffer:
                ffmpeg.stdin.write(frame_buffer[next_frame].tobytes())
                del frame_buffer[next_frame]
            next_frame += 1

        # Write end frames
        print(f"  Writing {self.end_frames} end frames...")
        end_frame = self.generator.create_end_frame()
        for _ in range(self.end_frames):
            ffmpeg.stdin.write(end_frame.tobytes())

        # Close FFmpeg
        ffmpeg.stdin.close()
        ffmpeg.wait()

        if ffmpeg.returncode != 0:
            stderr = ffmpeg.stderr.read().decode()
            print(f"  FFmpeg warning: {stderr[:300]}")

        encode_time = time.time() - start_time
        output_size = output_path.stat().st_size if output_path.exists() else 0
        video_duration = total_frames / self.fps

        print(f"\n{'='*50}")
        print(f"ENCODING COMPLETE")
        print(f"{'='*50}")
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
        }


def main():
    parser = argparse.ArgumentParser(description="Parallel encoder for Visual Data Diode")
    parser.add_argument('input', help='Input file to encode')
    parser.add_argument('output', help='Output video file (.mp4)')
    parser.add_argument('--profile', choices=['conservative', 'standard', 'aggressive', 'ultra'],
                        default='standard', help='Encoding profile')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS (default: 30)')
    parser.add_argument('--crf', type=int, default=18, help='H.264 CRF quality (default: 18)')
    parser.add_argument('--workers', type=int, default=None, help='Number of threads')

    args = parser.parse_args()

    profiles = {
        'conservative': PROFILE_CONSERVATIVE,
        'standard': PROFILE_STANDARD,
        'aggressive': PROFILE_AGGRESSIVE,
        'ultra': PROFILE_ULTRA
    }

    encoder = ParallelEncoder(
        profile=profiles[args.profile],
        fps=args.fps,
        crf=args.crf,
        workers=args.workers
    )

    encoder.encode_file(args.input, args.output)


if __name__ == "__main__":
    main()
