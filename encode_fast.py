#!/usr/bin/env python3
"""
Visual Data Diode - Fast Encoder

Optimized encoder using FFmpeg for efficient H.264 output.
Designed for HDMI capture at 1080p 30fps.

Usage:
    python encode_fast.py input_file.bin output.mp4
    python encode_fast.py input_file.bin output.mp4 --profile standard --crf 23
"""

import os
import sys
import argparse
import hashlib
import time
import struct
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
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


@dataclass
class EncodeResult:
    """Encoding result statistics."""
    input_file: str
    input_size: int
    output_file: str
    output_size: int
    total_blocks: int
    total_frames: int
    encode_time: float
    video_duration: float
    input_hash: str

    @property
    def compression_ratio(self) -> float:
        return self.output_size / self.input_size if self.input_size > 0 else 0

    @property
    def throughput_kbps(self) -> float:
        return (self.input_size * 8 / self.video_duration / 1000) if self.video_duration > 0 else 0


class FastEncoder:
    """
    Fast encoder using FFmpeg for efficient H.264 output.

    Optimized for:
    - 30 FPS playback (configurable)
    - Low file size with H.264 CRF encoding
    - Direct pipe to FFmpeg (no intermediate files)
    """

    def __init__(
        self,
        profile: EncodingProfile = PROFILE_STANDARD,
        fps: int = 30,
        crf: int = 18,  # H.264 quality (0=lossless, 18=recommended for data, 51=worst)
        repeat_count: int = 1,  # No redundancy by default
        sync_frames: int = 15,  # Half second at 30fps
        end_frames: int = 30,
        fec_ratio: float = DEFAULT_FEC_RATIO
    ):
        self.profile = profile
        self.fps = fps
        self.crf = crf
        self.repeat_count = repeat_count
        self.sync_frames = sync_frames
        self.end_frames = end_frames
        self.fec_ratio = fec_ratio

        # Grid dimensions
        self.cell_size = profile.cell_size
        self.grid_width = FRAME_WIDTH // self.cell_size
        self.grid_height = FRAME_HEIGHT // self.cell_size
        self.border_width = 2

        # Interior bounds
        self.interior_left = self.border_width
        self.interior_right = self.grid_width - self.border_width
        self.interior_top = self.border_width
        self.interior_bottom = self.grid_height - self.border_width
        self.interior_width = self.interior_right - self.interior_left
        self.interior_height = self.interior_bottom - self.interior_top

        # Capacity
        self.total_cells = self.interior_width * self.interior_height
        self.raw_bytes = self.total_cells // 4

        # FEC
        self.fec = SimpleFEC(fec_ratio)
        self.fec_parity_size = self.fec.parity_size(self.raw_bytes)
        self.block_capacity = self.raw_bytes - HEADER_SIZE - CRC_SIZE - self.fec_parity_size

        # Pre-generate sync pattern
        self._generate_sync_pattern()

        print(f"FastEncoder initialized:")
        print(f"  Profile: {profile.name} ({self.cell_size}px cells)")
        print(f"  Grid: {self.grid_width}x{self.grid_height}")
        print(f"  Capacity: {self.block_capacity} bytes/block")
        print(f"  FPS: {fps}, CRF: {crf}")

    def _generate_sync_pattern(self):
        """Pre-generate sync border pattern."""
        self.sync_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self._is_border_cell(row, col):
                    if self._is_corner_cell(row, col):
                        continue
                    if (row + col) % 2 == 0:
                        self.sync_grid[row, col] = COLOR_MAGENTA
                    else:
                        self.sync_grid[row, col] = COLOR_CYAN

        # Corner markers
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
                    self.sync_grid[start_row + i, start_col + j] = color

    def _is_border_cell(self, row: int, col: int) -> bool:
        return (row < self.border_width or
                row >= self.grid_height - self.border_width or
                col < self.border_width or
                col >= self.grid_width - self.border_width)

    def _is_corner_cell(self, row: int, col: int) -> bool:
        if row < 3 and col < 3:
            return True
        if row < 3 and col >= self.grid_width - 3:
            return True
        if row >= self.grid_height - 3 and col < 3:
            return True
        if row >= self.grid_height - 3 and col >= self.grid_width - 3:
            return True
        return False

    def _grid_to_frame(self, grid: np.ndarray) -> np.ndarray:
        """Expand grid to full frame."""
        frame = np.repeat(grid, self.cell_size, axis=0)
        frame = np.repeat(frame, self.cell_size, axis=1)
        if frame.shape[0] < FRAME_HEIGHT or frame.shape[1] < FRAME_WIDTH:
            padded = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame
            frame = padded
        else:
            frame = frame[:FRAME_HEIGHT, :FRAME_WIDTH]
        return frame

    def _bytes_to_cells(self, data: bytes) -> list:
        """Convert bytes to gray cell values."""
        cells = []
        for byte in data:
            cells.append(bits_to_gray((byte >> 6) & 0x03))
            cells.append(bits_to_gray((byte >> 4) & 0x03))
            cells.append(bits_to_gray((byte >> 2) & 0x03))
            cells.append(bits_to_gray(byte & 0x03))
        return cells

    def _encode_block_to_frame(self, block: Block) -> np.ndarray:
        """Encode block to frame."""
        grid = self.sync_grid.copy()
        block_bytes = block.pack(include_fec=True)
        cells = self._bytes_to_cells(block_bytes)

        cell_idx = 0
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if cell_idx < len(cells):
                    gray = cells[cell_idx]
                    grid[row, col] = (gray, gray, gray)
                    cell_idx += 1
                else:
                    grid[row, col] = (128, 128, 128)

        return self._grid_to_frame(grid)

    def _create_sync_frame(self) -> np.ndarray:
        """Create sync-only frame."""
        grid = self.sync_grid.copy()
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if (row + col) % 2 == 0:
                    grid[row, col] = (64, 64, 64)
                else:
                    grid[row, col] = (192, 192, 192)
        return self._grid_to_frame(grid)

    def _create_end_frame(self) -> np.ndarray:
        """Create end frame (black)."""
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    def encode_file(self, input_path: str, output_path: str) -> EncodeResult:
        """
        Encode file to video. Uses OpenCV for encoding, then FFmpeg for H.264 conversion.
        """
        input_path = Path(input_path)
        output_path = Path(output_path).with_suffix('.mp4')

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        start_time = time.time()

        # Get file info
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

        total_data_frames = total_blocks * self.repeat_count
        total_frames = self.sync_frames + total_data_frames + self.end_frames

        print(f"\nEncoding: {input_path.name}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Hash: {file_hash}")
        print(f"  Blocks: {total_blocks}")
        print(f"  Frames: {total_frames}")
        print(f"  Output: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary MJPG file (OpenCV compatible)
        temp_avi = output_path.with_suffix('.temp.avi')
        writer = cv2.VideoWriter(
            str(temp_avi),
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer")

        frame_count = 0

        # Write sync frames
        sync_frame = self._create_sync_frame()
        sync_bgr = cv2.cvtColor(sync_frame, cv2.COLOR_RGB2BGR)
        for _ in range(self.sync_frames):
            writer.write(sync_bgr)
            frame_count += 1

        # Encode data blocks
        with open(input_path, 'rb') as f:
            sequence = 0
            prev_crc16_val = 0

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
                    capacity = self.block_capacity - len(metadata_bytes)
                    payload_data = f.read(capacity)
                    full_payload = metadata_bytes + payload_data
                else:
                    full_payload = f.read(self.block_capacity)

                header = BlockHeader(
                    session_id=session_id,
                    block_index=block_idx,
                    total_blocks=total_blocks,
                    file_size=file_size,
                    payload_size=len(full_payload),
                    flags=flags,
                    sequence=sequence,
                    prev_crc16=prev_crc16_val
                )

                block = Block(header=header, payload=full_payload)
                block_data = header.pack() + full_payload
                _, parity = self.fec.encode(block_data)
                block.fec_parity = parity

                sequence = (sequence + 1) & 0xFFFF
                prev_crc16_val = crc16(block_data)

                # Encode frame
                frame = self._encode_block_to_frame(block)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Write frame(s)
                for _ in range(self.repeat_count):
                    writer.write(frame_bgr)
                    frame_count += 1

                # Progress
                if (block_idx + 1) % 500 == 0 or is_last:
                    pct = (block_idx + 1) * 100 // total_blocks
                    print(f"  {pct}% - Block {block_idx + 1}/{total_blocks}")

        # Write end frames
        end_frame = self._create_end_frame()
        for _ in range(self.end_frames):
            writer.write(end_frame)
            frame_count += 1

        writer.release()

        # Convert to H.264 with FFmpeg
        print(f"  Converting to H.264 (CRF {self.crf})...")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', str(temp_avi),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', str(self.crf),
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg warning: {result.stderr[:200]}")

        # Clean up temp file
        temp_avi.unlink(missing_ok=True)

        encode_time = time.time() - start_time
        output_size = output_path.stat().st_size
        video_duration = total_frames / self.fps

        print(f"  Complete: {encode_time:.1f}s")
        print(f"  Video size: {output_size / 1024 / 1024:.1f} MB")
        print(f"  Duration: {video_duration:.1f}s")
        print(f"  Throughput: {file_size * 8 / video_duration / 1000:.1f} kbps")

        return EncodeResult(
            input_file=str(input_path),
            input_size=file_size,
            output_file=str(output_path),
            output_size=output_size,
            total_blocks=total_blocks,
            total_frames=total_frames,
            encode_time=encode_time,
            video_duration=video_duration,
            input_hash=file_hash
        )


def main():
    parser = argparse.ArgumentParser(description="Fast encoder for Visual Data Diode")
    parser.add_argument('input', help='Input file to encode')
    parser.add_argument('output', help='Output video file (.mp4)')
    parser.add_argument('--profile', choices=['conservative', 'standard', 'aggressive', 'ultra'],
                        default='standard', help='Encoding profile')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS (default: 30)')
    parser.add_argument('--crf', type=int, default=23, help='H.264 CRF quality 0-51 (default: 23, lower=better)')
    parser.add_argument('--repeat', type=int, default=1, help='Frame repeat count (default: 1)')

    args = parser.parse_args()

    profiles = {
        'conservative': PROFILE_CONSERVATIVE,
        'standard': PROFILE_STANDARD,
        'aggressive': PROFILE_AGGRESSIVE,
        'ultra': PROFILE_ULTRA
    }

    encoder = FastEncoder(
        profile=profiles[args.profile],
        fps=args.fps,
        crf=args.crf,
        repeat_count=args.repeat
    )

    result = encoder.encode_file(args.input, args.output)

    print(f"\n{'='*50}")
    print("ENCODING COMPLETE")
    print(f"{'='*50}")
    print(f"Input:  {result.input_file} ({result.input_size:,} bytes)")
    print(f"Output: {result.output_file} ({result.output_size:,} bytes)")
    print(f"Hash:   {result.input_hash}")
    print(f"Ratio:  {result.compression_ratio:.2f}x")
    print(f"Rate:   {result.throughput_kbps:.1f} kbps")


if __name__ == "__main__":
    main()
