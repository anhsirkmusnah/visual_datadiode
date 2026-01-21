#!/usr/bin/env python3
"""
Visual Data Diode - Batch Encoder

Converts a folder of files into playable MP4 videos for visual transmission.
Each input file becomes one output video that can be played in any media player.

Usage:
    python batch_encode.py ./input_folder ./output_videos
    python batch_encode.py ./input_folder ./output_videos --profile standard --fps 60
"""

import os
import sys
import argparse
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

# Add parent to path for shared imports
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
class EncodeStats:
    """Statistics from encoding a file."""
    input_file: str
    input_size: int
    output_file: str
    output_size: int
    total_blocks: int
    total_frames: int
    encode_time: float
    video_duration: float  # seconds

    @property
    def data_rate_kbps(self) -> float:
        """Effective data rate in kbps."""
        if self.video_duration > 0:
            return (self.input_size * 8) / self.video_duration / 1000
        return 0.0


class BatchEncoder:
    """
    Encodes files to MP4 videos for visual transmission.

    Features:
    - H.264 codec for universal playback
    - Configurable FPS (default 60)
    - Configurable encoding profile
    - Progress callback support
    """

    # Codec options (in order of preference)
    # MJPG is universally supported and works with CUDA decoder
    # For maximum quality, use FFmpeg to convert to H.264 CRF 0 after encoding
    CODEC_OPTIONS = [
        ('MJPG', 'avi', 'Motion JPEG'),       # Universal compatibility
        ('XVID', 'avi', 'XVID'),              # Fallback
    ]

    def __init__(
        self,
        profile: EncodingProfile = PROFILE_STANDARD,
        fps: int = 60,
        repeat_count: int = 2,
        sync_frames: int = 30,
        end_frames: int = 60,
        fec_ratio: float = DEFAULT_FEC_RATIO
    ):
        """
        Initialize batch encoder.

        Args:
            profile: Encoding profile (cell size)
            fps: Output video FPS
            repeat_count: Times to repeat each data frame
            sync_frames: Number of sync-only frames at start
            end_frames: Number of black frames at end
            fec_ratio: FEC overhead ratio
        """
        self.profile = profile
        self.fps = fps
        self.repeat_count = repeat_count
        self.sync_frames = sync_frames
        self.end_frames = end_frames
        self.fec_ratio = fec_ratio

        # Pre-compute grid dimensions
        self.cell_size = profile.cell_size
        self.grid_width = FRAME_WIDTH // self.cell_size
        self.grid_height = FRAME_HEIGHT // self.cell_size
        self.border_width = 2  # cells

        # Interior bounds
        self.interior_left = self.border_width
        self.interior_right = self.grid_width - self.border_width
        self.interior_top = self.border_width
        self.interior_bottom = self.grid_height - self.border_width
        self.interior_width = self.interior_right - self.interior_left
        self.interior_height = self.interior_bottom - self.interior_top

        # Calculate payload capacity
        self.total_cells = self.interior_width * self.interior_height
        self.raw_bytes = self.total_cells // 4  # 4 cells per byte

        # FEC setup
        self.fec = SimpleFEC(fec_ratio)
        self.fec_parity_size = self.fec.parity_size(self.raw_bytes)
        self.block_capacity = self.raw_bytes - HEADER_SIZE - CRC_SIZE - self.fec_parity_size

        # Pre-generate sync pattern
        self._generate_sync_pattern()

        print(f"BatchEncoder initialized:")
        print(f"  Profile: {profile.name} ({self.cell_size}px cells)")
        print(f"  Grid: {self.grid_width}x{self.grid_height} cells")
        print(f"  Interior: {self.interior_width}x{self.interior_height} cells")
        print(f"  Capacity: {self.block_capacity} bytes/block")
        print(f"  FPS: {fps}, Repeat: {repeat_count}")

    def _generate_sync_pattern(self):
        """Pre-generate the sync border pattern as a grid."""
        self.sync_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self._is_border_cell(row, col):
                    if self._is_corner_cell(row, col):
                        continue  # Filled by corner markers
                    # Alternating cyan/magenta
                    if (row + col) % 2 == 0:
                        self.sync_grid[row, col] = COLOR_MAGENTA
                    else:
                        self.sync_grid[row, col] = COLOR_CYAN

        # Fill corner markers
        self._fill_corners()

    def _is_border_cell(self, row: int, col: int) -> bool:
        """Check if cell is in sync border."""
        return (row < self.border_width or
                row >= self.grid_height - self.border_width or
                col < self.border_width or
                col >= self.grid_width - self.border_width)

    def _is_corner_cell(self, row: int, col: int) -> bool:
        """Check if cell is in a corner marker region."""
        if row < 3 and col < 3:
            return True
        if row < 3 and col >= self.grid_width - 3:
            return True
        if row >= self.grid_height - 3 and col < 3:
            return True
        if row >= self.grid_height - 3 and col >= self.grid_width - 3:
            return True
        return False

    def _fill_corners(self):
        """Fill corner marker patterns."""
        corners = [
            (0, 0, CORNER_TOP_LEFT),
            (0, self.grid_width - 3, CORNER_TOP_RIGHT),
            (self.grid_height - 3, 0, CORNER_BOTTOM_LEFT),
            (self.grid_height - 3, self.grid_width - 3, CORNER_BOTTOM_RIGHT)
        ]

        for start_row, start_col, pattern in corners:
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    color = COLOR_WHITE if pattern[idx] else COLOR_BLACK
                    self.sync_grid[start_row + i, start_col + j] = color

    def _grid_to_frame(self, grid: np.ndarray) -> np.ndarray:
        """Expand cell grid to full frame resolution."""
        frame = np.repeat(grid, self.cell_size, axis=0)
        frame = np.repeat(frame, self.cell_size, axis=1)

        # Pad to exact frame size if needed
        if frame.shape[0] < FRAME_HEIGHT or frame.shape[1] < FRAME_WIDTH:
            padded = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame
            frame = padded
        else:
            frame = frame[:FRAME_HEIGHT, :FRAME_WIDTH]

        return frame

    def _bytes_to_cells(self, data: bytes) -> List[int]:
        """Convert bytes to cell grayscale values (4 cells per byte)."""
        cells = []
        for byte in data:
            cells.append(bits_to_gray((byte >> 6) & 0x03))
            cells.append(bits_to_gray((byte >> 4) & 0x03))
            cells.append(bits_to_gray((byte >> 2) & 0x03))
            cells.append(bits_to_gray(byte & 0x03))
        return cells

    def _encode_block_to_frame(self, block: Block) -> np.ndarray:
        """Encode a block to a full resolution frame."""
        # Start with sync pattern
        grid = self.sync_grid.copy()

        # Pack block data
        block_bytes = block.pack(include_fec=True)

        # Convert to cells
        cells = self._bytes_to_cells(block_bytes)

        # Fill interior
        cell_idx = 0
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if cell_idx < len(cells):
                    gray = cells[cell_idx]
                    grid[row, col] = (gray, gray, gray)
                    cell_idx += 1
                else:
                    grid[row, col] = (128, 128, 128)  # Padding

        return self._grid_to_frame(grid)

    def _create_sync_frame(self) -> np.ndarray:
        """Create a sync-only frame (alternating pattern in interior)."""
        grid = self.sync_grid.copy()

        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if (row + col) % 2 == 0:
                    grid[row, col] = (64, 64, 64)
                else:
                    grid[row, col] = (192, 192, 192)

        return self._grid_to_frame(grid)

    def _create_end_frame(self) -> np.ndarray:
        """Create an end-of-transmission frame (all black)."""
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    def _select_codec(self, output_path: Path) -> Tuple[str, str, str]:
        """Select best available codec."""
        for fourcc, ext, name in self.CODEC_OPTIONS:
            test_path = str(output_path.parent / f"_test_codec.{ext}")
            try:
                writer = cv2.VideoWriter(
                    test_path,
                    cv2.VideoWriter_fourcc(*fourcc),
                    self.fps,
                    (FRAME_WIDTH, FRAME_HEIGHT)
                )
                if writer.isOpened():
                    # Write test frame
                    test_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    writer.write(test_frame)
                    writer.release()
                    Path(test_path).unlink(missing_ok=True)
                    return fourcc, ext, name
                writer.release()
            except Exception:
                pass
            finally:
                Path(test_path).unlink(missing_ok=True)

        # Fallback
        return 'XVID', 'avi', 'XVID (fallback)'

    def encode_file(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> EncodeStats:
        """
        Encode a single file to video.

        Args:
            input_path: Path to input file
            output_path: Path for output video (extension may be changed)
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            EncodeStats with encoding results
        """
        input_path = Path(input_path)
        output_path = Path(output_path).with_suffix('')  # Remove extension

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        start_time = time.time()

        # Get file info
        file_size = input_path.stat().st_size
        file_hash = compute_file_hash(str(input_path))
        filename = input_path.name
        session_id = hash(str(input_path) + str(time.time())) & 0xFFFFFFFF

        # Calculate blocks needed
        filename_bytes = filename.encode('utf-8')
        metadata_size = 36 + len(filename_bytes)  # hash + filename_len + filename
        block0_capacity = self.block_capacity - metadata_size

        if file_size <= block0_capacity:
            total_blocks = 1
        else:
            remaining = file_size - block0_capacity
            total_blocks = 1 + (remaining + self.block_capacity - 1) // self.block_capacity

        # Calculate total frames
        total_data_frames = total_blocks * self.repeat_count
        total_frames = self.sync_frames + total_data_frames + self.end_frames

        # Select codec
        fourcc, ext, codec_name = self._select_codec(output_path)
        final_output = output_path.with_suffix(f'.{ext}')

        print(f"\nEncoding: {input_path.name}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Blocks: {total_blocks}")
        print(f"  Frames: {total_frames} ({self.sync_frames} sync + {total_data_frames} data + {self.end_frames} end)")
        print(f"  Codec: {codec_name}")
        print(f"  Output: {final_output}")

        # Create video writer
        final_output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(final_output),
            cv2.VideoWriter_fourcc(*fourcc),
            self.fps,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer with {codec_name}")

        frame_num = 0

        # Write sync frames
        sync_frame = self._create_sync_frame()
        sync_frame_bgr = cv2.cvtColor(sync_frame, cv2.COLOR_RGB2BGR)

        for _ in range(self.sync_frames):
            writer.write(sync_frame_bgr)
            frame_num += 1
            if progress_callback:
                progress_callback(frame_num, total_frames)

        # Read and encode file data
        with open(input_path, 'rb') as f:
            sequence = 0
            prev_crc16 = 0

            for block_idx in range(total_blocks):
                is_first = (block_idx == 0)
                is_last = (block_idx == total_blocks - 1)

                # Build flags
                flags = BlockFlags.NONE
                if is_first:
                    flags |= BlockFlags.FIRST_BLOCK
                if is_last:
                    flags |= BlockFlags.LAST_BLOCK

                # Read payload
                if is_first:
                    # Include metadata
                    metadata = FileMetadata(file_hash=file_hash, filename=filename)
                    metadata_bytes = metadata.pack()
                    capacity = self.block_capacity - len(metadata_bytes)
                    payload_data = f.read(capacity)
                    full_payload = metadata_bytes + payload_data
                else:
                    full_payload = f.read(self.block_capacity)

                # Create header
                header = BlockHeader(
                    session_id=session_id,
                    block_index=block_idx,
                    total_blocks=total_blocks,
                    file_size=file_size,
                    payload_size=len(full_payload),
                    flags=flags,
                    sequence=sequence,
                    prev_crc16=prev_crc16
                )

                # Create block with FEC
                block = Block(header=header, payload=full_payload)
                block_data = header.pack() + full_payload
                _, parity = self.fec.encode(block_data)
                block.fec_parity = parity

                # Update sequence tracking
                sequence = (sequence + 1) & 0xFFFF
                prev_crc16 = crc16(block_data)

                # Encode to frame
                frame_rgb = self._encode_block_to_frame(block)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Write frame (repeated)
                for _ in range(self.repeat_count):
                    writer.write(frame_bgr)
                    frame_num += 1
                    if progress_callback:
                        progress_callback(frame_num, total_frames)

                # Progress
                if (block_idx + 1) % 100 == 0 or is_last:
                    pct = (block_idx + 1) * 100 // total_blocks
                    print(f"  {pct}% - Block {block_idx + 1}/{total_blocks}")

        # Write end frames
        end_frame = self._create_end_frame()

        for _ in range(self.end_frames):
            writer.write(end_frame)
            frame_num += 1
            if progress_callback:
                progress_callback(frame_num, total_frames)

        writer.release()

        encode_time = time.time() - start_time
        output_size = final_output.stat().st_size
        video_duration = total_frames / self.fps

        print(f"  Complete: {encode_time:.1f}s, {output_size / 1024 / 1024:.1f} MB")

        return EncodeStats(
            input_file=str(input_path),
            input_size=file_size,
            output_file=str(final_output),
            output_size=output_size,
            total_blocks=total_blocks,
            total_frames=total_frames,
            encode_time=encode_time,
            video_duration=video_duration
        )

    def encode_folder(
        self,
        input_folder: str,
        output_folder: str,
        pattern: str = "*"
    ) -> List[EncodeStats]:
        """
        Encode all files in a folder.

        Args:
            input_folder: Path to input folder
            output_folder: Path for output videos
            pattern: Glob pattern for files (default: all)

        Returns:
            List of EncodeStats for each file
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)

        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        output_folder.mkdir(parents=True, exist_ok=True)

        # Find files
        files = sorted(input_folder.glob(pattern))
        files = [f for f in files if f.is_file()]

        if not files:
            print(f"No files found in {input_folder}")
            return []

        print(f"\n{'='*60}")
        print(f"BATCH ENCODE: {len(files)} files")
        print(f"Input: {input_folder}")
        print(f"Output: {output_folder}")
        print(f"{'='*60}")

        results = []
        total_input_size = 0
        total_output_size = 0
        total_time = 0

        for i, input_file in enumerate(files):
            print(f"\n[{i+1}/{len(files)}]", end="")

            output_name = input_file.stem + "_encoded"
            output_path = output_folder / output_name

            try:
                stats = self.encode_file(str(input_file), str(output_path))
                results.append(stats)

                total_input_size += stats.input_size
                total_output_size += stats.output_size
                total_time += stats.encode_time

            except Exception as e:
                print(f"  ERROR: {e}")

        # Summary
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"  Files: {len(results)}/{len(files)} successful")
        print(f"  Input size: {total_input_size / 1024 / 1024:.1f} MB")
        print(f"  Output size: {total_output_size / 1024 / 1024:.1f} MB")
        print(f"  Total time: {total_time:.1f}s")
        if results:
            avg_rate = sum(s.data_rate_kbps for s in results) / len(results)
            print(f"  Avg data rate: {avg_rate:.1f} kbps")
        print(f"{'='*60}")

        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Batch encode files to visual transmission videos"
    )
    parser.add_argument(
        'input',
        help='Input file or folder'
    )
    parser.add_argument(
        'output',
        help='Output file or folder'
    )
    parser.add_argument(
        '--profile',
        choices=['conservative', 'standard', 'aggressive', 'ultra'],
        default='standard',
        help='Encoding profile (default: standard)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='Output FPS (default: 60)'
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=2,
        help='Frame repeat count (default: 2)'
    )
    parser.add_argument(
        '--pattern',
        default='*',
        help='File pattern for folder mode (default: *)'
    )

    args = parser.parse_args()

    # Select profile
    profiles = {
        'conservative': PROFILE_CONSERVATIVE,
        'standard': PROFILE_STANDARD,
        'aggressive': PROFILE_AGGRESSIVE,
        'ultra': PROFILE_ULTRA
    }
    profile = profiles[args.profile]

    # Create encoder
    encoder = BatchEncoder(
        profile=profile,
        fps=args.fps,
        repeat_count=args.repeat
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        # Folder mode
        encoder.encode_folder(str(input_path), str(output_path), args.pattern)
    else:
        # Single file mode
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoder.encode_file(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
