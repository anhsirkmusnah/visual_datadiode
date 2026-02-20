"""
Visual Data Diode - Enhanced Frame Encoder

Encodes data using 8 luma levels + 6 color states.
Includes calibration frame generation.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .enhanced_encoding import (
    EnhancedEncodingProfile,
    ENHANCED_LUMA_LEVELS,
    ColorState,
    ENHANCED_COLOR_RGB,
    get_color_rgb,
    CALIBRATION_MAGIC,
    SYNC_MAGIC,
    ENHANCED_PROFILE_CONSERVATIVE,
)
from .constants import (
    COLOR_CYAN, COLOR_MAGENTA, COLOR_WHITE, COLOR_BLACK,
    CORNER_TOP_LEFT, CORNER_TOP_RIGHT, CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT,
    FRAME_WIDTH, FRAME_HEIGHT,
)


class EnhancedFrameEncoder:
    """
    Encodes data frames using enhanced luma + color encoding.

    Features:
    - 8 grayscale levels (3 bits per cell)
    - 6 color states per 2x2 cell group (~2 bits per group)
    - Calibration frame generation
    - Sync border with cyan/magenta pattern
    """

    def __init__(self, profile: EnhancedEncodingProfile = ENHANCED_PROFILE_CONSERVATIVE):
        """
        Initialize encoder.

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

    def create_calibration_frame(self) -> np.ndarray:
        """
        Create a calibration frame with all luma levels and color states.

        Layout:
        - Top section: All 8 luma levels (horizontal strips)
        - Bottom section: All 6 color states (vertical strips)
        - Each section uses multiple rows/columns for reliability

        Returns:
            RGB frame (FRAME_HEIGHT, FRAME_WIDTH, 3)
        """
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Draw sync border
        self._draw_sync_border(frame)

        # Draw corner markers
        self._draw_corners(frame)

        # Interior region in CELLS (for alignment with decoder)
        interior_cells_w = self.interior_right - self.interior_left
        interior_cells_h = self.interior_bottom - self.interior_top

        # Split interior: top half for luma, bottom half for colors (in cells)
        luma_section_cells = interior_cells_h // 2
        color_section_cells = interior_cells_h - luma_section_cells

        # === LUMA CALIBRATION SECTION ===
        # 8 horizontal strips, each showing one luma level
        strip_cells_h = luma_section_cells // 8

        for level in range(8):
            # Calculate in cells, then convert to pixels
            cell_y_start = self.interior_top + level * strip_cells_h
            cell_y_end = self.interior_top + (level + 1) * strip_cells_h

            y_start = cell_y_start * self.cell_size
            y_end = cell_y_end * self.cell_size
            x_start = self.interior_left * self.cell_size
            x_end = self.interior_right * self.cell_size

            gray_val = ENHANCED_LUMA_LEVELS[level]

            # Fill strip with solid grayscale
            frame[y_start:y_end, x_start:x_end] = [gray_val, gray_val, gray_val]

            # Add level identifier pattern at the start (for verification)
            # Pattern: level number in binary using black/white cells
            pattern_x = x_start
            for bit in range(3):
                bit_val = (level >> (2 - bit)) & 1
                color = [255, 255, 255] if bit_val else [0, 0, 0]
                frame[y_start:y_end, pattern_x:pattern_x + self.cell_size] = color
                pattern_x += self.cell_size

        # === COLOR CALIBRATION SECTION ===
        # 6 vertical strips, each showing one color state with mid-luma
        color_cell_y_start = self.interior_top + luma_section_cells
        color_y_start = color_cell_y_start * self.cell_size
        interior_x = self.interior_left * self.cell_size
        interior_w = interior_cells_w * self.cell_size
        strip_w = interior_w // 6

        mid_luma = 3  # Mid-range luma level for best color visibility
        interior_y_end = self.interior_bottom * self.cell_size

        for state in range(6):
            x_start = interior_x + state * strip_w
            x_end = interior_x + (state + 1) * strip_w

            color_rgb = get_color_rgb(ColorState(state), mid_luma)
            frame[color_y_start:interior_y_end, x_start:x_end] = color_rgb

            # Add state identifier pattern at the top of each strip
            pattern_y = color_y_start
            for bit in range(3):
                bit_val = (state >> (2 - bit)) & 1
                color = [255, 255, 255] if bit_val else [0, 0, 0]
                frame[pattern_y:pattern_y + self.cell_size, x_start:x_end] = color
                pattern_y += self.cell_size

        # === MAGIC MARKER ===
        # Place marker in first row of interior (8 consecutive cells)
        # Use cell-aligned coordinates for reliable detection
        marker_y = self.interior_top * self.cell_size  # First row of interior in pixels
        for i, level in enumerate(CALIBRATION_MAGIC):
            if i >= 8:  # Only need 8 cells for marker
                break
            # Place in rightmost 8 cells of first row
            col_from_right = 7 - i
            marker_x = interior_x + interior_w - (col_from_right + 1) * self.cell_size
            gray = ENHANCED_LUMA_LEVELS[level]
            frame[marker_y:marker_y + self.cell_size, marker_x:marker_x + self.cell_size] = [gray, gray, gray]

        return frame

    def encode_data_frame(
        self,
        luma_data: bytes,
        color_data: bytes,
        frame_number: int = 0
    ) -> np.ndarray:
        """
        Encode data into a frame using luma + color encoding.

        Args:
            luma_data: Bytes to encode in luma channel (3 bits per cell)
            color_data: Bytes to encode in color channel (2 bits per 2x2 group)
            frame_number: Frame sequence number (for sync pattern alternation)

        Returns:
            RGB frame (FRAME_HEIGHT, FRAME_WIDTH, 3)
        """
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Draw sync border
        self._draw_sync_border(frame, frame_number)

        # Draw corner markers
        self._draw_corners(frame)

        # Unpack luma data (3 bits per cell)
        luma_bits = self._bytes_to_tribits(luma_data)

        # Unpack color data (2 bits per group)
        color_bits = self._bytes_to_dibits(color_data)

        # Encode interior cells
        luma_idx = 0
        color_idx = 0

        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                # Get luma value
                if luma_idx < len(luma_bits):
                    luma_level = luma_bits[luma_idx]
                else:
                    luma_level = 0
                luma_idx += 1

                # Determine color group
                group_row = (row - self.interior_top) // self.color_group_size
                group_col = (col - self.interior_left) // self.color_group_size
                group_idx = group_row * self.profile.color_groups_width + group_col

                # Get color state for this group
                if group_idx < len(color_bits):
                    color_state = color_bits[group_idx] % 6  # Map 0-3 to 0-5 (use first 4 states)
                else:
                    color_state = 0

                # Get RGB color for this cell
                rgb = get_color_rgb(ColorState(color_state), luma_level)

                # Fill cell
                y1 = row * self.cell_size
                y2 = (row + 1) * self.cell_size
                x1 = col * self.cell_size
                x2 = (col + 1) * self.cell_size

                frame[y1:y2, x1:x2] = rgb

        return frame

    def encode_grayscale_frame(
        self,
        data: bytes,
        frame_number: int = 0
    ) -> np.ndarray:
        """
        Encode data using only grayscale (8 levels, no color).

        Useful for testing or fallback mode.

        Args:
            data: Bytes to encode (3 bits per cell)
            frame_number: Frame sequence number

        Returns:
            RGB frame
        """
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Draw sync border
        self._draw_sync_border(frame, frame_number)

        # Draw corner markers
        self._draw_corners(frame)

        # Unpack data (3 bits per cell)
        tribits = self._bytes_to_tribits(data)

        # Encode interior cells
        idx = 0
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if idx < len(tribits):
                    level = tribits[idx]
                else:
                    level = 0
                idx += 1

                gray = ENHANCED_LUMA_LEVELS[level]

                y1 = row * self.cell_size
                y2 = (row + 1) * self.cell_size
                x1 = col * self.cell_size
                x2 = (col + 1) * self.cell_size

                frame[y1:y2, x1:x2] = [gray, gray, gray]

        return frame

    def create_sync_frame(self, frame_number: int = 0) -> np.ndarray:
        """
        Create a sync frame with identifying marker.

        Sync frames use a mid-gray fill with a magic marker pattern
        in the rightmost 8 cells of the first interior row.

        Args:
            frame_number: Frame sequence number (for border pattern alternation)

        Returns:
            RGB frame (FRAME_HEIGHT, FRAME_WIDTH, 3)
        """
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Draw sync border
        self._draw_sync_border(frame, frame_number)

        # Draw corner markers
        self._draw_corners(frame)

        # Fill interior with mid-gray (level 4)
        mid_gray = ENHANCED_LUMA_LEVELS[4]
        interior_x = self.interior_left * self.cell_size
        interior_y = self.interior_top * self.cell_size
        interior_w = (self.interior_right - self.interior_left) * self.cell_size
        interior_h = (self.interior_bottom - self.interior_top) * self.cell_size

        frame[interior_y:interior_y + interior_h, interior_x:interior_x + interior_w] = [mid_gray, mid_gray, mid_gray]

        # Place sync magic marker in rightmost 8 cells of first interior row
        marker_y = self.interior_top * self.cell_size
        interior_w_cells = self.interior_right - self.interior_left

        for i, level in enumerate(SYNC_MAGIC):
            if i >= 8:
                break
            # Place from right side (same as calibration marker)
            col_from_right = 7 - i
            marker_x = interior_x + (interior_w_cells - col_from_right - 1) * self.cell_size
            gray = ENHANCED_LUMA_LEVELS[level]
            frame[marker_y:marker_y + self.cell_size, marker_x:marker_x + self.cell_size] = [gray, gray, gray]

        return frame

    def _draw_sync_border(self, frame: np.ndarray, frame_number: int = 0):
        """Draw cyan/magenta sync border."""
        # Alternate colors based on row/column parity
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                # Check if in border region
                in_top = row < self.border_width
                in_bottom = row >= self.grid_height - self.border_width
                in_left = col < self.border_width
                in_right = col >= self.grid_width - self.border_width

                if in_top or in_bottom or in_left or in_right:
                    # Skip corner regions (handled separately)
                    if (in_top or in_bottom) and (in_left or in_right):
                        continue

                    # Checkerboard pattern
                    parity = (row + col + frame_number) % 2
                    color = COLOR_CYAN if parity == 0 else COLOR_MAGENTA

                    y1 = row * self.cell_size
                    y2 = (row + 1) * self.cell_size
                    x1 = col * self.cell_size
                    x2 = (col + 1) * self.cell_size

                    frame[y1:y2, x1:x2] = color

    def _draw_corners(self, frame: np.ndarray):
        """Draw corner marker patterns."""
        corner_cells = 3  # 3x3 cells per corner

        corners = [
            (0, 0, CORNER_TOP_LEFT),
            (self.grid_width - corner_cells, 0, CORNER_TOP_RIGHT),
            (0, self.grid_height - corner_cells, CORNER_BOTTOM_LEFT),
            (self.grid_width - corner_cells, self.grid_height - corner_cells, CORNER_BOTTOM_RIGHT),
        ]

        for start_col, start_row, pattern in corners:
            for i in range(3):
                for j in range(3):
                    is_white = pattern[i * 3 + j] == 1
                    color = COLOR_WHITE if is_white else COLOR_BLACK

                    row = start_row + i
                    col = start_col + j

                    y1 = row * self.cell_size
                    y2 = (row + 1) * self.cell_size
                    x1 = col * self.cell_size
                    x2 = (col + 1) * self.cell_size

                    frame[y1:y2, x1:x2] = color

    def _bytes_to_tribits(self, data: bytes) -> List[int]:
        """
        Convert bytes to list of 3-bit values (0-7).

        Packing: 8 tribits from 3 bytes (24 bits = 8 * 3 bits)
        """
        tribits = []
        bit_buffer = 0
        bits_in_buffer = 0

        for byte in data:
            bit_buffer = (bit_buffer << 8) | byte
            bits_in_buffer += 8

            while bits_in_buffer >= 3:
                bits_in_buffer -= 3
                tribit = (bit_buffer >> bits_in_buffer) & 0x07
                tribits.append(tribit)

        # Handle remaining bits (pad with zeros)
        if bits_in_buffer > 0:
            tribit = (bit_buffer << (3 - bits_in_buffer)) & 0x07
            tribits.append(tribit)

        return tribits

    def _bytes_to_dibits(self, data: bytes) -> List[int]:
        """
        Convert bytes to list of 2-bit values (0-3).

        Packing: 4 dibits per byte
        """
        dibits = []
        for byte in data:
            dibits.append((byte >> 6) & 0x03)
            dibits.append((byte >> 4) & 0x03)
            dibits.append((byte >> 2) & 0x03)
            dibits.append(byte & 0x03)
        return dibits

    def get_capacity(self) -> dict:
        """Get encoding capacity information."""
        return {
            'luma_bits_per_frame': self.profile.luma_bits_per_frame,
            'color_bits_per_frame': self.profile.color_bits_per_frame,
            'total_bits_per_frame': self.profile.total_bits_per_frame,
            'luma_bytes_per_frame': self.profile.luma_bits_per_frame // 8,
            'color_bytes_per_frame': self.profile.color_bits_per_frame // 8,
            'total_bytes_per_frame': self.profile.payload_bytes,
        }


def pack_data_for_enhanced_frame(
    data: bytes,
    profile: EnhancedEncodingProfile
) -> Tuple[bytes, bytes]:
    """
    Pack data bytes into luma and color components.

    Splits input data proportionally between luma and color channels.

    Args:
        data: Input data bytes
        profile: Encoding profile

    Returns:
        (luma_data, color_data) tuple
    """
    # Use payload_bytes for total capacity (may be 1 more than luma+color due to rounding)
    total_bytes = profile.payload_bytes
    luma_bytes = profile.luma_bits_per_frame // 8
    # Color gets the remainder to account for rounding
    color_bytes = total_bytes - luma_bytes

    if len(data) > total_bytes:
        data = data[:total_bytes]
    elif len(data) < total_bytes:
        data = data + bytes(total_bytes - len(data))

    # Split: luma gets first portion, color gets rest
    luma_data = data[:luma_bytes]
    color_data = data[luma_bytes:]

    return luma_data, color_data


def unpack_data_from_enhanced_frame(
    luma_data: bytes,
    color_data: bytes
) -> bytes:
    """
    Unpack luma and color data back into original bytes.

    Args:
        luma_data: Decoded luma bytes
        color_data: Decoded color bytes

    Returns:
        Combined data bytes
    """
    return luma_data + color_data
