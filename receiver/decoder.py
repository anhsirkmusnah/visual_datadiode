"""
Visual Data Diode - Frame Decoder

Decodes binary data from captured frame grids.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    Block, BlockHeader,
    EncodingProfile, DEFAULT_PROFILE,
    GRAY_THRESHOLDS, gray_to_bits,
    HEADER_SIZE, CRC_SIZE,
    SimpleFEC
)


@dataclass
class DecodeResult:
    """Result of frame decoding."""
    success: bool
    block: Optional[Block]
    crc_valid: bool
    fec_corrected: int  # Number of errors corrected (-1 if failed)
    confidence: float  # Cell decode confidence
    message: str = ""


class FrameDecoder:
    """
    Decodes blocks from cell grids.

    Converts grayscale cell values back to binary data,
    with FEC error correction and CRC verification.
    """

    def __init__(
        self,
        profile: EncodingProfile = DEFAULT_PROFILE,
        fec_ratio: float = 0.10
    ):
        """
        Initialize decoder.

        Args:
            profile: Encoding profile
            fec_ratio: Expected FEC ratio
        """
        self.profile = profile
        self.cell_size = profile.cell_size
        self.grid_width = profile.grid_width
        self.grid_height = profile.grid_height
        self.border_width = profile.sync_border_width

        # Interior bounds (data area)
        self.interior_left = self.border_width
        self.interior_right = self.grid_width - self.border_width
        self.interior_top = self.border_width
        self.interior_bottom = self.grid_height - self.border_width

        # FEC decoder
        self.fec = SimpleFEC(fec_ratio)

        # Adaptive thresholds (can be calibrated)
        self.thresholds = list(GRAY_THRESHOLDS)

        # Statistics
        self.total_cells_decoded = 0
        self.low_confidence_cells = 0

    def decode_grid(self, grid: np.ndarray) -> DecodeResult:
        """
        Decode a cell grid to extract block data.

        Args:
            grid: Cell grid (grid_height, grid_width, 3)

        Returns:
            DecodeResult with decoded block or error info
        """
        # Step 1: Extract interior cells
        interior = grid[
            self.interior_top:self.interior_bottom,
            self.interior_left:self.interior_right
        ]

        # Step 2: Convert cells to grayscale values
        gray_values = np.mean(interior, axis=2).astype(np.float32)

        # Step 3: Quantize to bit values
        bits, confidence = self._quantize_cells(gray_values)

        # Step 4: Pack bits into bytes
        raw_bytes = self._bits_to_bytes(bits)

        if len(raw_bytes) < HEADER_SIZE + CRC_SIZE:
            return DecodeResult(
                success=False,
                block=None,
                crc_valid=False,
                fec_corrected=-1,
                confidence=confidence,
                message="Insufficient data decoded"
            )

        # Step 5: Unpack and verify block
        try:
            # Try to unpack header first
            header = BlockHeader.unpack(raw_bytes[:HEADER_SIZE])

            # Calculate expected sizes
            payload_end = HEADER_SIZE + header.payload_size
            crc_end = payload_end + CRC_SIZE

            if len(raw_bytes) < crc_end:
                return DecodeResult(
                    success=False,
                    block=None,
                    crc_valid=False,
                    fec_corrected=-1,
                    confidence=confidence,
                    message=f"Data too short: {len(raw_bytes)} < {crc_end}"
                )

            # Extract components
            fec_size = len(raw_bytes) - crc_end
            block, crc_valid = Block.unpack(raw_bytes, fec_size)

            if crc_valid:
                return DecodeResult(
                    success=True,
                    block=block,
                    crc_valid=True,
                    fec_corrected=0,
                    confidence=confidence,
                    message="Decoded successfully"
                )

            # Step 6: Try FEC correction
            if fec_size > 0 and self.fec.available:
                data_to_correct = raw_bytes[:crc_end]
                parity = raw_bytes[crc_end:crc_end + fec_size]

                corrected, errors = self.fec.decode(data_to_correct, parity)

                if corrected is not None:
                    # Re-unpack with corrected data
                    block, crc_valid = Block.unpack(
                        corrected + parity, fec_size
                    )

                    return DecodeResult(
                        success=crc_valid,
                        block=block if crc_valid else None,
                        crc_valid=crc_valid,
                        fec_corrected=errors,
                        confidence=confidence,
                        message=f"FEC corrected {errors} errors" if crc_valid else "FEC correction failed"
                    )

            return DecodeResult(
                success=False,
                block=block,
                crc_valid=False,
                fec_corrected=-1,
                confidence=confidence,
                message="CRC failed, FEC could not correct"
            )

        except Exception as e:
            return DecodeResult(
                success=False,
                block=None,
                crc_valid=False,
                fec_corrected=-1,
                confidence=confidence,
                message=f"Decode error: {str(e)}"
            )

    def _quantize_cells(
        self, gray_values: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Quantize grayscale values to 2-bit values.

        Args:
            gray_values: 2D array of grayscale values (0-255)

        Returns:
            (bits_array, confidence)
        """
        h, w = gray_values.shape
        bits = np.zeros((h, w), dtype=np.uint8)
        confidences = []

        for i in range(h):
            for j in range(w):
                val = gray_values[i, j]
                bit_val, conf = self._quantize_single(val)
                bits[i, j] = bit_val
                confidences.append(conf)
                self.total_cells_decoded += 1
                if conf < 0.7:
                    self.low_confidence_cells += 1

        avg_confidence = np.mean(confidences) if confidences else 0.0
        return bits, avg_confidence

    def _quantize_single(self, value: float) -> Tuple[int, float]:
        """
        Quantize a single grayscale value.

        Returns:
            (bit_value, confidence)
        """
        # Find which level this value is closest to
        levels = [0, 85, 170, 255]
        distances = [abs(value - level) for level in levels]
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # Confidence based on distance to threshold
        # Perfect: 0, threshold boundary: 42.5
        confidence = 1.0 - min(min_dist / 42.5, 1.0)

        return min_idx, confidence

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """
        Pack 2-bit cell values into bytes.

        Args:
            bits: 2D array of 2-bit values (0-3)

        Returns:
            Packed bytes
        """
        # Flatten row-major
        flat = bits.flatten()

        # Pack 4 cells per byte
        num_bytes = len(flat) // 4
        result = bytearray(num_bytes)

        for i in range(num_bytes):
            idx = i * 4
            byte_val = (
                (flat[idx] << 6) |
                (flat[idx + 1] << 4) |
                (flat[idx + 2] << 2) |
                flat[idx + 3]
            )
            result[i] = byte_val

        return bytes(result)

    def calibrate_thresholds(self, sample_grids: List[np.ndarray]):
        """
        Calibrate quantization thresholds from sample grids.

        Args:
            sample_grids: List of grids with known patterns
        """
        # Collect grayscale values from all grids
        all_values = []

        for grid in sample_grids:
            interior = grid[
                self.interior_top:self.interior_bottom,
                self.interior_left:self.interior_right
            ]
            gray = np.mean(interior, axis=2).flatten()
            all_values.extend(gray)

        if not all_values:
            return

        # Find clusters (expect 4)
        values = np.array(all_values)

        # Simple k-means-like clustering
        centers = [0.0, 85.0, 170.0, 255.0]

        for _ in range(10):  # Iterate to refine centers
            # Assign values to nearest center
            assignments = [[] for _ in range(4)]
            for v in values:
                distances = [abs(v - c) for c in centers]
                cluster = np.argmin(distances)
                assignments[cluster].append(v)

            # Update centers
            for i in range(4):
                if assignments[i]:
                    centers[i] = np.mean(assignments[i])

        # Update thresholds as midpoints between centers
        self.thresholds = [
            (centers[0] + centers[1]) / 2,
            (centers[1] + centers[2]) / 2,
            (centers[2] + centers[3]) / 2
        ]

    def get_statistics(self) -> dict:
        """Get decoding statistics."""
        return {
            'total_cells': self.total_cells_decoded,
            'low_confidence_cells': self.low_confidence_cells,
            'low_confidence_ratio': (
                self.low_confidence_cells / self.total_cells_decoded
                if self.total_cells_decoded > 0 else 0.0
            )
        }

    def reset_statistics(self):
        """Reset decoding statistics."""
        self.total_cells_decoded = 0
        self.low_confidence_cells = 0


class StreamDecoder:
    """
    Decodes a stream of frames, handling duplicates and ordering.
    """

    def __init__(
        self,
        profile: EncodingProfile = DEFAULT_PROFILE,
        fec_ratio: float = 0.10
    ):
        self.decoder = FrameDecoder(profile, fec_ratio)

        # Tracking state
        self.session_id: Optional[int] = None
        self.total_blocks: Optional[int] = None
        self.received_blocks: dict = {}  # block_index -> DecodeResult
        self.duplicate_count = 0
        self.failed_decodes = 0

    def process_grid(self, grid: np.ndarray) -> Optional[DecodeResult]:
        """
        Process a single grid from the stream.

        Args:
            grid: Cell grid

        Returns:
            DecodeResult if new block decoded, None if duplicate/failed
        """
        result = self.decoder.decode_grid(grid)

        if not result.success or result.block is None:
            self.failed_decodes += 1
            return result

        block = result.block
        block_idx = block.header.block_index

        # Check session
        if self.session_id is None:
            self.session_id = block.header.session_id
            self.total_blocks = block.header.total_blocks
        elif block.header.session_id != self.session_id:
            # New session started
            self.reset()
            self.session_id = block.header.session_id
            self.total_blocks = block.header.total_blocks

        # Check for duplicate
        if block_idx in self.received_blocks:
            self.duplicate_count += 1
            return None

        # Store result
        self.received_blocks[block_idx] = result
        return result

    def get_progress(self) -> Tuple[int, int]:
        """
        Get reception progress.

        Returns:
            (blocks_received, total_blocks)
        """
        received = len(self.received_blocks)
        total = self.total_blocks or 0
        return received, total

    def get_missing_blocks(self) -> List[int]:
        """Get list of missing block indices."""
        if self.total_blocks is None:
            return []

        return [
            i for i in range(self.total_blocks)
            if i not in self.received_blocks
        ]

    def is_complete(self) -> bool:
        """Check if all blocks received."""
        if self.total_blocks is None:
            return False

        return len(self.received_blocks) == self.total_blocks

    def reset(self):
        """Reset decoder state."""
        self.session_id = None
        self.total_blocks = None
        self.received_blocks.clear()
        self.duplicate_count = 0
        self.failed_decodes = 0
        self.decoder.reset_statistics()
