#!/usr/bin/env python3
"""
Visual Data Diode - Enhanced Video Processor

Processes videos encoded with the enhanced 8-luma + 4-color encoding.

Usage:
    python process_enhanced.py input.mp4 output_dir/
"""

import os
import sys
import argparse
import time
import struct
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    ENHANCED_PROFILE_CONSERVATIVE,
    EnhancedFrameDecoder,
    unpack_data_from_enhanced_frame,
    FRAME_WIDTH, FRAME_HEIGHT,
    Block, BlockHeader,
    SimpleFEC, crc16,
    HEADER_SIZE, CRC_SIZE, DEFAULT_FEC_RATIO,
    COLOR_BLACK
)
from receiver.assembler import BlockAssembler


class ProcessorState(Enum):
    SCANNING = "scanning"
    RECEIVING = "receiving"
    END_DETECTED = "end"
    GAP = "gap"


@dataclass
class DecodedFile:
    filename: str
    output_path: str
    file_size: int
    hash_valid: Optional[bool]
    blocks_received: int
    total_blocks: int


@dataclass
class ProcessResult:
    success: bool
    files_decoded: List[DecodedFile]
    total_frames: int
    processing_time: float
    message: str


class EnhancedVideoProcessor:
    """
    Processes videos encoded with enhanced 8-luma + 4-color encoding.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        fec_ratio: float = DEFAULT_FEC_RATIO,
        profile = None
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.fec_ratio = fec_ratio

        os.makedirs(output_dir, exist_ok=True)

        # Enhanced decoder - use provided profile or default
        self.profile = profile if profile is not None else ENHANCED_PROFILE_CONSERVATIVE
        self.decoder = EnhancedFrameDecoder(self.profile)

        # Calculate capacity (MUST match encoder exactly)
        # Use payload_bytes from profile for consistency
        self.luma_bytes = self.profile.luma_bits_per_frame // 8
        self.color_bytes = self.profile.color_bits_per_frame // 8
        self.total_raw_bytes = self.profile.payload_bytes  # Use payload_bytes, not luma+color sum!

        # FEC - use same calculation as encoder for consistent parity size
        # FEC covers header + payload + CRC (block_data)
        self.fec = SimpleFEC(fec_ratio)

        # Calculate block_data_size iteratively (must match encoder)
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

        # State
        self._state = ProcessorState.SCANNING
        self._calibrated = False
        self._current_frame = 0
        self._consecutive_black_frames = 0
        self._consecutive_unsynced_frames = 0
        self._black_threshold = 30
        self._end_frame_threshold = 5

        # Decoding state
        self._assembler: Optional[BlockAssembler] = None
        self._decoded_files: List[DecodedFile] = []
        self._session_id: Optional[int] = None
        self._total_blocks: Optional[int] = None
        self._received_blocks = {}
        self._duplicate_count = 0
        self._failed_decodes = 0

        print(f"EnhancedVideoProcessor initialized:")
        print(f"  Profile: {self.profile.name}")
        print(f"  Luma bytes: {self.luma_bytes}")
        print(f"  Color bytes: {self.color_bytes}")
        print(f"  Total raw bytes: {self.total_raw_bytes}")
        print(f"  Block data size: {self.block_data_size} (header+payload+CRC)")
        print(f"  FEC parity size: {self.fec_parity_size}")
        print(f"  Block capacity: {self.block_capacity}")

    def process(self) -> ProcessResult:
        """Process the video file."""
        start_time = time.time()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return ProcessResult(
                success=False,
                files_decoded=[],
                total_frames=0,
                processing_time=0,
                message=f"Failed to open video: {self.video_path}"
            )

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"\nProcessing: {self.video_path}")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}, Frames: {total_frames}")

            self._state = ProcessorState.SCANNING
            self._current_frame = 0
            calibration_count = 0
            data_frames = 0

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self._current_frame += 1

                # Extract grid from frame
                grid = self._extract_grid(frame)

                # Process based on state
                if self._state == ProcessorState.SCANNING:
                    result = self._handle_scanning(grid)
                    if result and result.is_calibration_frame:
                        calibration_count += 1
                elif self._state == ProcessorState.RECEIVING:
                    result = self._handle_receiving(grid, frame)
                    if result and not result.is_calibration_frame:
                        data_frames += 1
                elif self._state == ProcessorState.GAP:
                    result = self._handle_gap(grid)

                # Progress
                if self._current_frame % 100 == 0:
                    received = len(self._received_blocks)
                    total = self._total_blocks or 0
                    print(f"  Frame {self._current_frame}/{total_frames}: state={self._state.value}, blocks={received}/{total}")

            # Finalize
            if self._state == ProcessorState.RECEIVING:
                self._finalize_current_file()

            elapsed = time.time() - start_time

            print(f"\nProcessing complete:")
            print(f"  Calibration frames: {calibration_count}")
            print(f"  Data frames: {data_frames}")
            print(f"  Files decoded: {len(self._decoded_files)}")
            print(f"  Time: {elapsed:.1f}s")

            return ProcessResult(
                success=True,
                files_decoded=self._decoded_files,
                total_frames=total_frames,
                processing_time=elapsed,
                message=f"Decoded {len(self._decoded_files)} file(s)"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ProcessResult(
                success=False,
                files_decoded=self._decoded_files,
                total_frames=0,
                processing_time=time.time() - start_time,
                message=f"Processing error: {str(e)}"
            )
        finally:
            cap.release()

    def _extract_grid(self, frame: np.ndarray) -> np.ndarray:
        """Extract cell grid from frame by sampling cell centers."""
        grid = np.zeros((self.profile.grid_height, self.profile.grid_width, 3), dtype=np.uint8)
        cell_size = self.profile.cell_size

        for row in range(self.profile.grid_height):
            for col in range(self.profile.grid_width):
                cy = row * cell_size + cell_size // 2
                cx = col * cell_size + cell_size // 2

                if cy < frame.shape[0] and cx < frame.shape[1]:
                    grid[row, col] = frame[cy, cx]

        return grid

    def _handle_scanning(self, grid: np.ndarray):
        """Handle frame in scanning state."""
        result = self.decoder.decode_grid(grid)

        if result.is_calibration_frame:
            print(f"[Frame {self._current_frame}] Calibration frame detected")
            if result.calibration and result.calibration.is_valid:
                self._calibrated = True
                print(f"  Calibration valid: luma_sep={result.calibration.luma_separation:.1f}")
            return result

        if result.is_sync_frame:
            # Sync frames are expected during scanning - just skip them
            return result

        # Check if this looks like a data frame (high confidence)
        if result.success and result.luma_confidence > 0.5:
            print(f"[Frame {self._current_frame}] Data frame detected, starting reception")
            self._start_new_file()
            self._state = ProcessorState.RECEIVING
            self._process_data_frame(result)
            return result

        return result

    def _handle_receiving(self, grid: np.ndarray, frame: np.ndarray):
        """Handle frame in receiving state."""
        # Check for black frame (end pattern)
        if self._is_black_frame(frame):
            self._consecutive_black_frames += 1
            if self._consecutive_black_frames >= self._end_frame_threshold:
                print(f"[Frame {self._current_frame}] End pattern detected")
                self._finalize_current_file()
                self._state = ProcessorState.GAP
                self._consecutive_black_frames = 0
                return None
        else:
            self._consecutive_black_frames = 0

        # Decode frame
        result = self.decoder.decode_grid(grid)

        if result.is_calibration_frame:
            # Calibration frame in middle of stream - just update calibration
            if result.calibration and result.calibration.is_valid:
                self._calibrated = True
            return result

        if result.is_sync_frame:
            # Sync frames are not data - skip them but don't count as failures
            return result

        if result.success and result.luma_data:
            self._consecutive_unsynced_frames = 0
            self._process_data_frame(result)
        else:
            self._consecutive_unsynced_frames += 1
            self._failed_decodes += 1

            if self._consecutive_unsynced_frames > 30:
                print(f"[Frame {self._current_frame}] Lost sync after 30 frames")
                self._finalize_current_file()
                self._state = ProcessorState.SCANNING
                self._consecutive_unsynced_frames = 0

        return result

    def _handle_gap(self, grid: np.ndarray):
        """Handle frame in gap state."""
        # Check if this is a black frame first (end sequence continues)
        if self._is_black_grid(grid):
            return None

        result = self.decoder.decode_grid(grid)

        if result.is_calibration_frame:
            print(f"[Frame {self._current_frame}] New calibration after gap")
            if result.calibration and result.calibration.is_valid:
                self._calibrated = True
            return result

        if result.is_sync_frame:
            # Sync frames during gap - skip them
            return result

        if result.success and result.luma_confidence > 0.5:
            print(f"[Frame {self._current_frame}] New file detected after gap")
            self._start_new_file()
            self._state = ProcessorState.RECEIVING
            self._process_data_frame(result)
            return result

        return result

    def _is_black_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is mostly black."""
        return np.mean(frame) < self._black_threshold

    def _is_black_grid(self, grid: np.ndarray) -> bool:
        """Check if grid is mostly black (for gap detection)."""
        return np.mean(grid) < self._black_threshold

    def _start_new_file(self):
        """Initialize state for a new file."""
        self._assembler = BlockAssembler(output_dir=self.output_dir)
        self._session_id = None
        self._total_blocks = None
        self._received_blocks = {}
        self._duplicate_count = 0
        self._failed_decodes = 0
        self._consecutive_black_frames = 0
        self._consecutive_unsynced_frames = 0

    def _process_data_frame(self, result):
        """Process decoded data frame into block."""
        if not result.luma_data or not result.color_data:
            return

        # Combine luma and color data
        block_bytes = unpack_data_from_enhanced_frame(result.luma_data, result.color_data)

        # Try to parse block
        try:
            block = self._parse_block(block_bytes)
            if block:
                self._add_block(block)
            else:
                self._failed_decodes += 1
                print(f"  [Frame] Failed to parse block: header invalid")
        except Exception as e:
            self._failed_decodes += 1
            print(f"  [Frame] Exception parsing block: {e}")

    def _parse_block(self, data: bytes) -> Optional[Block]:
        """Parse raw bytes into a Block."""
        if len(data) < HEADER_SIZE + CRC_SIZE:
            return None

        # Block format (from encoder):
        # [header (24)] [payload (N)] [CRC (4)] [FEC parity (M)] [padding]
        # FEC covers: header + payload + CRC (block_data)

        # First parse header to get payload_size
        header = BlockHeader.unpack(data[:HEADER_SIZE])
        if header is None:
            return None

        # CRC is right after payload
        payload_size = header.payload_size
        if payload_size <= 0 or payload_size > self.block_capacity:
            payload_size = min(self.block_capacity, len(data) - HEADER_SIZE - CRC_SIZE)

        crc_start = HEADER_SIZE + payload_size
        parity_start = crc_start + CRC_SIZE

        # Try FEC correction if available
        # Use fixed block_data_size (not dependent on header.payload_size which might be corrupted)
        if self.fec.available and self.fec_parity_size > 0 and len(data) >= self.block_data_size + self.fec_parity_size:
            # Extract block_data and parity at fixed positions
            block_data = data[:self.block_data_size]
            parity = data[self.block_data_size:self.block_data_size + self.fec_parity_size]

            try:
                corrected, corrections = self.fec.decode(block_data, parity)
                if corrected is not None and corrections >= 0:
                    # FEC correction successful
                    data = corrected + parity + data[self.block_data_size + self.fec_parity_size:]
                    if corrections > 0:
                        # Track FEC corrections (print summary at end)
                        if not hasattr(self, '_fec_corrections'):
                            self._fec_corrections = 0
                            self._fec_blocks_corrected = 0
                        self._fec_corrections += corrections
                        self._fec_blocks_corrected += 1
                    # Re-parse header from corrected data
                    header = BlockHeader.unpack(data[:HEADER_SIZE])
                    if header is None:
                        return None
                    payload_size = header.payload_size
                    crc_start = HEADER_SIZE + payload_size
                elif corrections == -1:
                    # FEC failed - block has too many errors to correct
                    # Skip this block (don't add corrupted data)
                    if not hasattr(self, '_fec_failures'):
                        self._fec_failures = 0
                    self._fec_failures += 1
                    if self._fec_failures <= 3:
                        print(f"  [FEC] Too many errors in block {header.block_index}")
                    return None  # Skip corrupted block
            except Exception as e:
                if not hasattr(self, '_fec_exceptions'):
                    self._fec_exceptions = 0
                self._fec_exceptions += 1
                if self._fec_exceptions <= 3:
                    print(f"  [FEC] Exception for block {header.block_index}: {e}")

        # Extract payload
        payload = data[HEADER_SIZE:crc_start]

        # Verify CRC
        import zlib
        crc_stored = struct.unpack('<I', data[crc_start:crc_start + 4])[0] if len(data) > crc_start + 4 else 0
        crc_computed = zlib.crc32(header.pack() + payload) & 0xFFFFFFFF
        if crc_stored != crc_computed:
            if not hasattr(self, '_crc_errors'):
                self._crc_errors = 0
            self._crc_errors += 1
            # CRC failed - block data is corrupted, skip it
            return None

        return Block(header=header, payload=payload)

    def _add_block(self, block: Block):
        """Add decoded block to assembler."""
        block_idx = block.header.block_index

        # Validate block - reject obviously invalid blocks
        if block.header.total_blocks == 0:
            # Invalid block - total_blocks should never be 0
            self._failed_decodes += 1
            return

        if block.header.session_id == 0 and block.header.total_blocks == 0:
            # Invalid block - likely decoded from noise/artifacts
            self._failed_decodes += 1
            return

        # Track session
        if self._session_id is None:
            self._session_id = block.header.session_id
            self._total_blocks = block.header.total_blocks
            print(f"  Session: {self._session_id:08x}, Total blocks: {self._total_blocks}")

        # Check session consistency
        if block.header.session_id != self._session_id:
            # Different session - ignore (could be noise)
            self._failed_decodes += 1
            return

        # Check for duplicate
        if block_idx in self._received_blocks:
            self._duplicate_count += 1
            return

        self._received_blocks[block_idx] = True

        # Add to assembler
        if self._assembler:
            self._assembler.add_block(block)

    def _finalize_current_file(self):
        """Finalize and save current file."""
        if not self._assembler:
            return

        received = len(self._received_blocks)
        total = self._total_blocks or 0

        crc_errors = getattr(self, '_crc_errors', 0)
        fec_corrections = getattr(self, '_fec_corrections', 0)
        fec_blocks = getattr(self, '_fec_blocks_corrected', 0)
        print(f"\nFinalizing file:")
        print(f"  Blocks: {received}/{total}")
        print(f"  Duplicates: {self._duplicate_count}")
        print(f"  Failed decodes: {self._failed_decodes}")
        print(f"  CRC errors: {crc_errors}")
        if fec_blocks > 0:
            print(f"  FEC: corrected {fec_corrections} errors in {fec_blocks} blocks")

        if received == 0:
            print("  No blocks received, skipping")
            return

        # Try assembly
        if not self._assembler.is_complete():
            # Check if mostly complete
            if total > 0 and received / total >= 0.9:
                print(f"  Attempting partial assembly ({received/total*100:.1f}%)")
                self._fill_missing_blocks()
            else:
                print(f"  Too many missing blocks ({received}/{total})")
                return

        result = self._assembler.assemble()

        if result.output_path:
            decoded = DecodedFile(
                filename=result.filename or "unknown",
                output_path=result.output_path,
                file_size=result.file_size,
                hash_valid=result.file_hash_valid,
                blocks_received=received,
                total_blocks=total
            )
            self._decoded_files.append(decoded)
            print(f"  Saved: {result.filename} ({result.file_size} bytes)")

    def _fill_missing_blocks(self):
        """Fill missing blocks with zeros for partial assembly."""
        if not self._assembler or not self._total_blocks:
            return

        # Find most common block size
        from collections import Counter
        sizes = [len(v) for v in self._assembler.blocks.values()]
        if not sizes:
            return

        common_size = Counter(sizes).most_common(1)[0][0]

        # Fill missing
        for idx in range(self._total_blocks):
            if idx not in self._assembler.blocks:
                self._assembler.blocks[idx] = bytes(common_size)


def main():
    parser = argparse.ArgumentParser(description="Process enhanced-encoded video")
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output directory')

    args = parser.parse_args()

    processor = EnhancedVideoProcessor(
        video_path=args.input,
        output_dir=args.output
    )

    result = processor.process()

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Files decoded: {len(result.files_decoded)}")
    print(f"Time: {result.processing_time:.1f}s")

    for f in result.files_decoded:
        print(f"  - {f.filename}: {f.file_size} bytes, {f.blocks_received}/{f.total_blocks} blocks")


if __name__ == "__main__":
    main()
