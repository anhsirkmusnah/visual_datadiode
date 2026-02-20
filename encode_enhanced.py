#!/usr/bin/env python3
"""
Visual Data Diode - Enhanced Encoder

Uses 8 luma levels + 4 color states for ~75% more throughput.
Optimized for HDMI-to-MJPEG capture (MS2109 and similar).

Usage:
    python encode_enhanced.py input_file.bin output.mp4
    python encode_enhanced.py input_file.bin output.mp4 --crf 23
"""

import os
import sys
import argparse
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    ENHANCED_PROFILE_CONSERVATIVE,
    EnhancedFrameEncoder,
    pack_data_for_enhanced_frame,
    FRAME_WIDTH, FRAME_HEIGHT,
    Block, BlockHeader, BlockFlags, FileMetadata,
    SimpleFEC, compute_file_hash, crc16,
    HEADER_SIZE, CRC_SIZE, DEFAULT_FEC_RATIO
)


@dataclass
class EnhancedEncodeResult:
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


class EnhancedFastEncoder:
    """
    Fast encoder using enhanced 8-luma + 4-color encoding.

    Provides ~75% more throughput compared to base 2-bit grayscale encoding.
    """

    def __init__(
        self,
        fps: int = 30,
        crf: int = 18,
        repeat_count: int = 1,
        sync_frames: int = 15,
        end_frames: int = 30,
        calibration_frames: int = 10,  # Extra: calibration for color
        fec_ratio: float = DEFAULT_FEC_RATIO
    ):
        self.fps = fps
        self.crf = crf
        self.repeat_count = repeat_count
        self.sync_frames = sync_frames
        self.end_frames = end_frames
        self.calibration_frames = calibration_frames
        self.fec_ratio = fec_ratio

        # Use enhanced encoding profile
        self.profile = ENHANCED_PROFILE_CONSERVATIVE
        self.encoder = EnhancedFrameEncoder(self.profile)

        # Capacity from enhanced profile
        capacity = self.encoder.get_capacity()
        self.luma_bytes = capacity['luma_bytes_per_frame']
        self.color_bytes = capacity['color_bytes_per_frame']
        self.total_raw_bytes = capacity['total_bytes_per_frame']

        # FEC - apply to combined data
        self.fec = SimpleFEC(fec_ratio)
        self.fec_parity_size = self.fec.parity_size(self.total_raw_bytes)
        self.block_capacity = self.total_raw_bytes - HEADER_SIZE - CRC_SIZE - self.fec_parity_size

        print(f"EnhancedFastEncoder initialized:")
        print(f"  Profile: {self.profile.name} ({self.profile.cell_size}px cells)")
        print(f"  Grid: {self.profile.grid_width}x{self.profile.grid_height}")
        print(f"  Luma capacity: {self.luma_bytes} bytes/frame")
        print(f"  Color capacity: {self.color_bytes} bytes/frame")
        print(f"  Total capacity: {self.total_raw_bytes} bytes/frame")
        print(f"  Block capacity: {self.block_capacity} bytes/block (after headers/FEC)")
        print(f"  FPS: {fps}, CRF: {crf}")

    def _encode_block_to_frame(self, block: Block, frame_number: int = 0) -> np.ndarray:
        """Encode block to frame using enhanced encoding."""
        block_bytes = block.pack(include_fec=True)

        # Pad to total capacity
        if len(block_bytes) < self.total_raw_bytes:
            block_bytes = block_bytes + bytes(self.total_raw_bytes - len(block_bytes))
        elif len(block_bytes) > self.total_raw_bytes:
            block_bytes = block_bytes[:self.total_raw_bytes]

        # Split into luma and color data
        luma_data, color_data = pack_data_for_enhanced_frame(block_bytes, self.profile)

        # Encode to frame
        return self.encoder.encode_data_frame(luma_data, color_data, frame_number)

    def _create_sync_frame(self, frame_number: int = 0) -> np.ndarray:
        """Create sync-only frame with identifying marker."""
        return self.encoder.create_sync_frame(frame_number)

    def _create_calibration_frame(self) -> np.ndarray:
        """Create calibration frame with all luma levels and color states."""
        return self.encoder.create_calibration_frame()

    def _create_end_frame(self) -> np.ndarray:
        """Create end frame (black)."""
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    def encode_file(self, input_path: str, output_path: str) -> EnhancedEncodeResult:
        """
        Encode file to video with enhanced encoding.
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
        total_frames = self.calibration_frames + self.sync_frames + total_data_frames + self.end_frames

        print(f"\nEncoding: {input_path.name}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Hash: {file_hash}")
        print(f"  Blocks: {total_blocks}")
        print(f"  Frames: {total_frames} (cal: {self.calibration_frames}, sync: {self.sync_frames}, data: {total_data_frames}, end: {self.end_frames})")
        print(f"  Output: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use direct FFmpeg pipe for lossless encoding (MJPG intermediate is lossy!)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', str(self.crf),
            '-pix_fmt', 'yuv444p' if self.crf == 0 else 'yuv420p',
            str(output_path)
        ]

        import os
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=0x08000000 if os.name == 'nt' else 0  # CREATE_NO_WINDOW on Windows
        )

        frame_count = 0

        def write_frame(frame):
            """Write RGB frame to FFmpeg pipe."""
            ffmpeg_proc.stdin.write(frame.tobytes())

        # Write calibration frames
        print("  Writing calibration frames...")
        cal_frame = self._create_calibration_frame()
        for _ in range(self.calibration_frames):
            write_frame(cal_frame)
            frame_count += 1

        # Write sync frames
        print("  Writing sync frames...")
        for i in range(self.sync_frames):
            sync_frame = self._create_sync_frame(i)
            write_frame(sync_frame)
            frame_count += 1

        # Encode data blocks
        print("  Encoding data blocks...")
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
                frame = self._encode_block_to_frame(block, frame_count)

                # Write frame(s)
                for _ in range(self.repeat_count):
                    write_frame(frame)
                    frame_count += 1

                # Progress
                if (block_idx + 1) % 200 == 0 or is_last:
                    pct = (block_idx + 1) * 100 // total_blocks
                    print(f"    {pct}% - Block {block_idx + 1}/{total_blocks}")

        # Write end frames
        print("  Writing end frames...")
        end_frame = self._create_end_frame()
        for _ in range(self.end_frames):
            write_frame(end_frame)
            frame_count += 1

        # Close FFmpeg pipe and wait for encoding to finish
        print(f"  Finalizing H.264 encoding (CRF {self.crf})...")
        ffmpeg_proc.stdin.close()
        stdout, stderr = ffmpeg_proc.communicate()

        if ffmpeg_proc.returncode != 0:
            print(f"  FFmpeg warning: {stderr.decode()[-500:]}")

        encode_time = time.time() - start_time
        output_size = output_path.stat().st_size
        video_duration = total_frames / self.fps

        print(f"  Complete: {encode_time:.1f}s")
        print(f"  Video size: {output_size / 1024 / 1024:.1f} MB")
        print(f"  Duration: {video_duration:.1f}s")
        print(f"  Throughput: {file_size * 8 / video_duration / 1000:.1f} kbps")

        return EnhancedEncodeResult(
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
    parser = argparse.ArgumentParser(description="Enhanced encoder for Visual Data Diode (8 luma + 4 color)")
    parser.add_argument('input', help='Input file to encode')
    parser.add_argument('output', help='Output video file (.mp4)')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS (default: 30)')
    parser.add_argument('--crf', type=int, default=18, help='H.264 CRF quality 0-51 (default: 18, lower=better)')
    parser.add_argument('--repeat', type=int, default=1, help='Frame repeat count (default: 1)')
    parser.add_argument('--calibration', type=int, default=10, help='Number of calibration frames (default: 10)')

    args = parser.parse_args()

    encoder = EnhancedFastEncoder(
        fps=args.fps,
        crf=args.crf,
        repeat_count=args.repeat,
        calibration_frames=args.calibration
    )

    result = encoder.encode_file(args.input, args.output)

    print(f"\n{'='*60}")
    print("ENHANCED ENCODING COMPLETE")
    print(f"{'='*60}")
    print(f"Input:  {result.input_file} ({result.input_size:,} bytes)")
    print(f"Output: {result.output_file} ({result.output_size:,} bytes)")
    print(f"Hash:   {result.input_hash}")
    print(f"Ratio:  {result.compression_ratio:.2f}x")
    print(f"Rate:   {result.throughput_kbps:.1f} kbps")


if __name__ == "__main__":
    main()
