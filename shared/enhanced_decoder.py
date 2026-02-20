"""
Visual Data Diode - Enhanced Frame Decoder

Decodes data using 8 luma levels + 6 color states.
Includes calibration frame detection and analysis.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .enhanced_encoding import (
    EnhancedEncodingProfile,
    ENHANCED_LUMA_LEVELS,
    ENHANCED_LUMA_THRESHOLDS,
    ColorState,
    ENHANCED_COLOR_YCBCR,
    CalibrationData,
    luma_to_bits_enhanced,
    CALIBRATION_MAGIC,
    SYNC_MAGIC,
    ENHANCED_PROFILE_CONSERVATIVE,
)


@dataclass
class EnhancedDecodeResult:
    """Result of enhanced frame decoding."""
    success: bool
    is_calibration_frame: bool
    is_sync_frame: bool = False
    luma_data: Optional[bytes] = None
    color_data: Optional[bytes] = None
    luma_confidence: float = 0.0
    color_confidence: float = 0.0
    calibration: Optional[CalibrationData] = None
    message: str = ""


class EnhancedFrameDecoder:
    """
    Decodes frames using enhanced luma + color encoding.

    Features:
    - 8 grayscale level detection (3 bits per cell)
    - 6 color state detection (2 bits per 2x2 group)
    - Calibration frame detection and analysis
    - Adaptive thresholds from calibration
    """

    def __init__(self, profile: EnhancedEncodingProfile = ENHANCED_PROFILE_CONSERVATIVE):
        """
        Initialize decoder.

        Args:
            profile: Enhanced encoding profile
        """
        self.profile = profile
        self.cell_size = profile.cell_size
        self.grid_width = profile.grid_width
        self.grid_height = profile.grid_height
        self.border_width = profile.sync_border_width

        # Interior bounds
        self.interior_left = self.border_width
        self.interior_right = self.grid_width - self.border_width
        self.interior_top = self.border_width
        self.interior_bottom = self.grid_height - self.border_width

        # Color group dimensions
        self.color_group_size = profile.color_group_size

        # Calibration data (updated after processing calibration frame)
        self.calibration: Optional[CalibrationData] = None

        # Statistics
        self.frames_decoded = 0
        self.calibration_frames = 0

    def decode_grid(self, grid: np.ndarray) -> EnhancedDecodeResult:
        """
        Decode a cell grid to extract luma and color data.

        Args:
            grid: Cell grid (grid_height, grid_width, 3) - RGB values at cell centers

        Returns:
            EnhancedDecodeResult with decoded data
        """
        # Extract interior region
        interior = grid[
            self.interior_top:self.interior_bottom,
            self.interior_left:self.interior_right
        ]

        # Check if this is a calibration frame
        if self._is_calibration_frame(interior):
            calibration = self._analyze_calibration_frame(grid)
            if calibration and calibration.is_valid:
                self.calibration = calibration
                self.calibration_frames += 1

            return EnhancedDecodeResult(
                success=True,
                is_calibration_frame=True,
                is_sync_frame=False,
                luma_data=None,
                color_data=None,
                luma_confidence=1.0,
                color_confidence=1.0,
                calibration=calibration,
                message="Calibration frame processed"
            )

        # Check if this is a sync frame
        if self._is_sync_frame(interior):
            return EnhancedDecodeResult(
                success=True,
                is_calibration_frame=False,
                is_sync_frame=True,
                luma_data=None,
                color_data=None,
                luma_confidence=1.0,
                color_confidence=1.0,
                calibration=None,
                message="Sync frame detected"
            )

        # Decode luma (grayscale) data
        luma_data, luma_conf = self._decode_luma(interior)

        # Decode color data
        color_data, color_conf = self._decode_color(interior)

        self.frames_decoded += 1

        return EnhancedDecodeResult(
            success=True,
            is_calibration_frame=False,
            is_sync_frame=False,
            luma_data=luma_data,
            color_data=color_data,
            luma_confidence=luma_conf,
            color_confidence=color_conf,
            calibration=None,
            message=f"Decoded: luma_conf={luma_conf:.2f}, color_conf={color_conf:.2f}"
        )

    def _is_calibration_frame(self, interior: np.ndarray) -> bool:
        """Check if interior contains calibration frame marker."""
        # Check first row, rightmost 8 cells for alternating 0/7 pattern
        first_row = interior[0, -8:]  # Last 8 cells of first row

        if len(first_row) < 8:
            return False

        detected_levels = []
        for i in range(8):
            # Get luma value (proper Y calculation)
            r, g, b = first_row[i]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            level = luma_to_bits_enhanced(gray)
            detected_levels.append(level)

        # Encoder places magic[i] at position -(8-i), so:
        # first_row[-8] = magic[0], first_row[-7] = magic[1], ..., first_row[-1] = magic[7]
        # Reading first_row[-8:] gives [magic[0], magic[1], ..., magic[7]] = CALIBRATION_MAGIC
        expected = CALIBRATION_MAGIC

        # Check against magic pattern (allow some tolerance)
        matches = sum(1 for a, b in zip(detected_levels, expected) if a == b)
        return matches >= 6  # Allow 2 mismatches

    def _is_sync_frame(self, interior: np.ndarray) -> bool:
        """Check if interior contains sync frame marker."""
        # Check first row, rightmost 8 cells for sync pattern [3,3,3,3,4,4,4,4]
        first_row = interior[0, -8:]  # Last 8 cells of first row

        if len(first_row) < 8:
            return False

        detected_levels = []
        for i in range(8):
            # Get luma value (proper Y calculation)
            r, g, b = first_row[i]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            level = luma_to_bits_enhanced(gray)
            detected_levels.append(level)

        # Check against sync magic pattern (allow 1 mismatch for robustness)
        matches = sum(1 for a, b in zip(detected_levels, SYNC_MAGIC) if a == b)
        return matches >= 7

    def _analyze_calibration_frame(self, grid: np.ndarray) -> Optional[CalibrationData]:
        """
        Analyze calibration frame to extract calibration data.

        Args:
            grid: Full grid including border

        Returns:
            CalibrationData or None if analysis fails
        """
        calibration = CalibrationData()

        # Interior region
        interior = grid[
            self.interior_top:self.interior_bottom,
            self.interior_left:self.interior_right
        ]

        interior_h = interior.shape[0]
        interior_w = interior.shape[1]

        # Luma section is top half
        luma_section_h = interior_h // 2
        strip_h = luma_section_h // 8

        # === ANALYZE LUMA LEVELS ===
        for level in range(8):
            # Each strip spans full width but first 3 columns have identifier pattern
            y_start = level * strip_h
            y_end = (level + 1) * strip_h

            # Skip first 4 columns (identifier pattern uses 3, plus margin)
            # Also skip last column (magic marker)
            x_start = 5
            x_end = interior_w - 2

            if y_start >= y_end or x_start >= x_end:
                continue

            region = interior[y_start:y_end, x_start:x_end]

            # Calculate proper luma Y = 0.299R + 0.587G + 0.114B
            r = region[:, :, 0].astype(float)
            g = region[:, :, 1].astype(float)
            b = region[:, :, 2].astype(float)
            luma_values = 0.299 * r + 0.587 * g + 0.114 * b

            mean_val = float(np.mean(luma_values))
            std_val = float(np.std(luma_values))

            calibration.luma_calibration[level] = (mean_val, std_val)

        # === ANALYZE COLOR STATES ===
        color_y_start = luma_section_h
        strip_w = interior_w // 6

        for state in range(6):
            x_start = state * strip_w + strip_w // 4
            x_end = (state + 1) * strip_w - strip_w // 4

            # Skip identifier rows at top
            y_start = color_y_start + 4
            y_end = interior_h

            if y_start >= y_end or x_start >= x_end:
                continue

            region = interior[y_start:y_end, x_start:x_end]

            # Convert to YCbCr for color analysis
            r = region[:, :, 0].astype(float)
            g = region[:, :, 1].astype(float)
            b = region[:, :, 2].astype(float)

            # Y = 0.299R + 0.587G + 0.114B
            y = 0.299 * r + 0.587 * g + 0.114 * b
            # Cb = 128 + 0.564*(B-Y)
            cb = 128 + 0.564 * (b - y)
            # Cr = 128 + 0.713*(R-Y)
            cr = 128 + 0.713 * (r - y)

            cb_mean = float(np.mean(cb))
            cr_mean = float(np.mean(cr))
            std_val = float(np.sqrt(np.var(cb) + np.var(cr)))

            calibration.color_calibration[state] = (cb_mean, cr_mean, std_val)

        # Compute thresholds
        calibration.compute_thresholds()

        return calibration

    def _decode_luma(self, interior: np.ndarray) -> Tuple[bytes, float]:
        """
        Decode luma data from interior region.

        Args:
            interior: Interior cell grid (interior_h, interior_w, 3)

        Returns:
            (luma_bytes, confidence)
        """
        tribits = []
        confidences = []

        for row in range(interior.shape[0]):
            for col in range(interior.shape[1]):
                # Compute proper Y (luma) from RGB
                # Y = 0.299*R + 0.587*G + 0.114*B
                r, g, b = interior[row, col]
                gray = 0.299 * r + 0.587 * g + 0.114 * b

                # Decode to 3-bit value
                if self.calibration and self.calibration.is_valid:
                    level = self.calibration.decode_luma(gray)
                else:
                    level = luma_to_bits_enhanced(gray)

                tribits.append(level)

                # Calculate confidence
                if self.calibration and self.calibration.luma_thresholds:
                    # Distance to nearest threshold
                    min_dist = min(abs(gray - t) for t in self.calibration.luma_thresholds)
                    conf = min(1.0, min_dist / 18.0)  # 18 is half the level spacing
                else:
                    expected = ENHANCED_LUMA_LEVELS[level]
                    conf = 1.0 - min(1.0, abs(gray - expected) / 18.0)

                confidences.append(conf)

        # Pack tribits into bytes
        luma_bytes = self._tribits_to_bytes(tribits)

        # Truncate to expected length (tribit packing may produce 1 extra byte)
        expected_luma_bytes = self.profile.luma_bits_per_frame // 8
        if len(luma_bytes) > expected_luma_bytes:
            luma_bytes = luma_bytes[:expected_luma_bytes]

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return luma_bytes, avg_confidence

    def _decode_color(self, interior: np.ndarray) -> Tuple[bytes, float]:
        """
        Decode color data from interior region.

        Args:
            interior: Interior cell grid (interior_h, interior_w, 3)

        Returns:
            (color_bytes, confidence)
        """
        dibits = []
        confidences = []

        # Process 2x2 cell groups
        groups_h = interior.shape[0] // self.color_group_size
        groups_w = interior.shape[1] // self.color_group_size

        for group_row in range(groups_h):
            for group_col in range(groups_w):
                # Extract 2x2 cell group
                y1 = group_row * self.color_group_size
                y2 = y1 + self.color_group_size
                x1 = group_col * self.color_group_size
                x2 = x1 + self.color_group_size

                group = interior[y1:y2, x1:x2]

                # Compute CbCr per cell, then average (more robust than averaging RGB)
                cb_values = []
                cr_values = []
                for cy in range(self.color_group_size):
                    for cx in range(self.color_group_size):
                        r, g, b = group[cy, cx]
                        y = 0.299 * r + 0.587 * g + 0.114 * b
                        cb_values.append(128 + 0.564 * (b - y))
                        cr_values.append(128 + 0.713 * (r - y))

                cb = float(np.mean(cb_values))
                cr = float(np.mean(cr_values))

                # Decode color state (returns 0-3 directly)
                if self.calibration and self.calibration.color_centroids:
                    state, conf = self.calibration.decode_color(cb, cr)
                else:
                    state, conf = self._decode_color_default(cb, cr)

                dibits.append(state)
                confidences.append(conf)

        # Pack dibits into bytes
        color_bytes = self._dibits_to_bytes(dibits)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return color_bytes, avg_confidence

    def _decode_color_default(self, cb: float, cr: float) -> Tuple[int, float]:
        """
        Decode color state using default thresholds.

        Only considers states 0-3 since those are the only ones used for data encoding.

        Args:
            cb: Cb chrominance value
            cr: Cr chrominance value

        Returns:
            (state, confidence)
        """
        # Color offsets in CbCr space (must match encoder's COLOR_CBCR_OFFSETS)
        COLOR_OFFSETS = {
            0: (-30, 60),   # RED: Yellow-ish Cb, Red Cr
            1: (-50, 10),   # YELLOW: Yellow Cb, slight red Cr
            2: (-20, -50),  # GREEN: Slight yellow Cb, Cyan Cr
            3: (30, -60),   # CYAN: Blue Cb, Cyan Cr
        }

        # Find closest color state in CbCr space
        min_dist = float('inf')
        best_state = 0

        for state in range(4):  # Only check states 0-3
            cb_offset, cr_offset = COLOR_OFFSETS[state]
            expected_cb = 128 + cb_offset
            expected_cr = 128 + cr_offset

            dist = (cb - expected_cb) ** 2 + (cr - expected_cr) ** 2
            if dist < min_dist:
                min_dist = dist
                best_state = state

        # Confidence based on distance
        confidence = max(0.0, 1.0 - (min_dist ** 0.5) / 100)

        return best_state, confidence

    def _tribits_to_bytes(self, tribits: List[int]) -> bytes:
        """
        Convert list of 3-bit values to bytes.

        Packing: 8 tribits into 3 bytes (24 bits)
        """
        result = bytearray()
        bit_buffer = 0
        bits_in_buffer = 0

        for tribit in tribits:
            bit_buffer = (bit_buffer << 3) | (tribit & 0x07)
            bits_in_buffer += 3

            while bits_in_buffer >= 8:
                bits_in_buffer -= 8
                byte_val = (bit_buffer >> bits_in_buffer) & 0xFF
                result.append(byte_val)

        # Handle remaining bits (pad with zeros)
        if bits_in_buffer > 0:
            byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
            result.append(byte_val)

        return bytes(result)

    def _dibits_to_bytes(self, dibits: List[int]) -> bytes:
        """
        Convert list of 2-bit values to bytes.

        Packing: 4 dibits per byte
        """
        result = bytearray()

        for i in range(0, len(dibits), 4):
            byte_val = 0
            for j in range(4):
                if i + j < len(dibits):
                    byte_val |= (dibits[i + j] & 0x03) << (6 - j * 2)
            result.append(byte_val)

        return bytes(result)

    def get_statistics(self) -> dict:
        """Get decoding statistics."""
        return {
            'frames_decoded': self.frames_decoded,
            'calibration_frames': self.calibration_frames,
            'has_calibration': self.calibration is not None and self.calibration.is_valid,
            'luma_separation': self.calibration.luma_separation if self.calibration else 0,
            'color_separation': self.calibration.color_separation if self.calibration else 0,
        }

    def reset_calibration(self):
        """Reset calibration data."""
        self.calibration = None

    def set_calibration(self, calibration: CalibrationData):
        """Set calibration data manually."""
        self.calibration = calibration


class EnhancedStreamDecoder:
    """
    Stateful stream decoder for enhanced encoding.

    Handles calibration frames automatically and maintains decode state.
    """

    def __init__(self, profile: EnhancedEncodingProfile = ENHANCED_PROFILE_CONSERVATIVE):
        self.decoder = EnhancedFrameDecoder(profile)
        self.profile = profile

        # State
        self.calibrated = False
        self.frames_since_calibration = 0
        self.total_frames = 0
        self.successful_decodes = 0
        self.failed_decodes = 0

        # Data buffer
        self.luma_buffer = bytearray()
        self.color_buffer = bytearray()

    def process_grid(self, grid: np.ndarray) -> EnhancedDecodeResult:
        """
        Process a grid from the stream.

        Args:
            grid: Cell grid

        Returns:
            EnhancedDecodeResult
        """
        self.total_frames += 1

        result = self.decoder.decode_grid(grid)

        if result.is_calibration_frame:
            if result.calibration and result.calibration.is_valid:
                self.calibrated = True
                self.frames_since_calibration = 0
            return result

        self.frames_since_calibration += 1

        if result.success:
            self.successful_decodes += 1

            if result.luma_data:
                self.luma_buffer.extend(result.luma_data)
            if result.color_data:
                self.color_buffer.extend(result.color_data)
        else:
            self.failed_decodes += 1

        return result

    def get_data(self) -> Tuple[bytes, bytes]:
        """Get accumulated data."""
        return bytes(self.luma_buffer), bytes(self.color_buffer)

    def get_combined_data(self) -> bytes:
        """Get combined luma + color data."""
        return bytes(self.luma_buffer) + bytes(self.color_buffer)

    def clear_buffers(self):
        """Clear data buffers."""
        self.luma_buffer.clear()
        self.color_buffer.clear()

    def get_statistics(self) -> dict:
        """Get stream statistics."""
        stats = self.decoder.get_statistics()
        stats.update({
            'calibrated': self.calibrated,
            'frames_since_calibration': self.frames_since_calibration,
            'total_frames': self.total_frames,
            'successful_decodes': self.successful_decodes,
            'failed_decodes': self.failed_decodes,
            'success_rate': (
                self.successful_decodes / self.total_frames
                if self.total_frames > 0 else 0.0
            ),
            'luma_buffer_size': len(self.luma_buffer),
            'color_buffer_size': len(self.color_buffer),
        })
        return stats

    def reset(self):
        """Reset stream decoder state."""
        self.decoder.reset_calibration()
        self.calibrated = False
        self.frames_since_calibration = 0
        self.total_frames = 0
        self.successful_decodes = 0
        self.failed_decodes = 0
        self.luma_buffer.clear()
        self.color_buffer.clear()
