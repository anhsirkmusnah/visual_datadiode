"""
Visual Data Diode - Frame Encoder

Encodes binary blocks into cell grids for visual rendering.
"""

import numpy as np
from typing import Tuple, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    Block, EncodingProfile, DEFAULT_PROFILE,
    GRAY_LEVELS, bits_to_gray,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_WHITE, COLOR_BLACK,
    CORNER_TOP_LEFT, CORNER_TOP_RIGHT, CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT,
    FRAME_WIDTH, FRAME_HEIGHT
)


class FrameEncoder:
    """
    Encodes binary blocks into visual cell grids.

    Produces numpy arrays representing frames ready for rendering.
    """

    def __init__(self, profile: EncodingProfile = DEFAULT_PROFILE):
        """
        Initialize encoder.

        Args:
            profile: Encoding profile defining cell size
        """
        self.profile = profile
        self.cell_size = profile.cell_size
        self.grid_width = profile.grid_width
        self.grid_height = profile.grid_height

        # Pre-calculate interior bounds
        self.border_width = profile.sync_border_width
        self.interior_left = self.border_width
        self.interior_right = self.grid_width - self.border_width
        self.interior_top = self.border_width
        self.interior_bottom = self.grid_height - self.border_width

        # Pre-generate sync border pattern
        self._generate_sync_pattern()

    def _generate_sync_pattern(self):
        """Pre-generate the sync border pattern."""
        self.sync_pattern = np.zeros(
            (self.grid_height, self.grid_width, 3),
            dtype=np.uint8
        )

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self._is_border_cell(row, col):
                    if self._is_corner_cell(row, col):
                        # Will be filled by corner markers
                        continue
                    # Alternating cyan/magenta
                    if (row + col) % 2 == 0:
                        color = COLOR_MAGENTA
                    else:
                        color = COLOR_CYAN
                    self.sync_pattern[row, col] = color

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
        # Top-left corner
        if row < 3 and col < 3:
            return True
        # Top-right corner
        if row < 3 and col >= self.grid_width - 3:
            return True
        # Bottom-left corner
        if row >= self.grid_height - 3 and col < 3:
            return True
        # Bottom-right corner
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
                    self.sync_pattern[start_row + i, start_col + j] = color

    def encode_block(self, block: Block) -> np.ndarray:
        """
        Encode a block into a cell grid.

        Args:
            block: Block to encode

        Returns:
            numpy array of shape (grid_height, grid_width, 3) with RGB values
        """
        # Start with sync pattern
        grid = self.sync_pattern.copy()

        # Pack block data into bytes
        block_bytes = block.pack(include_fec=True)

        # Convert bytes to cells
        cells = self._bytes_to_cells(block_bytes)

        # Fill interior with data cells
        cell_idx = 0
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if cell_idx < len(cells):
                    gray_value = cells[cell_idx]
                    grid[row, col] = (gray_value, gray_value, gray_value)
                    cell_idx += 1
                else:
                    # Padding: use mid-gray to indicate unused
                    grid[row, col] = (128, 128, 128)

        return grid

    def encode_sync_only(self) -> np.ndarray:
        """
        Generate a sync-only frame (no data).

        Used for initial synchronization.
        """
        grid = self.sync_pattern.copy()

        # Fill interior with alternating pattern for sync frames
        for row in range(self.interior_top, self.interior_bottom):
            for col in range(self.interior_left, self.interior_right):
                if (row + col) % 2 == 0:
                    grid[row, col] = (64, 64, 64)
                else:
                    grid[row, col] = (192, 192, 192)

        return grid

    def encode_end_frame(self) -> np.ndarray:
        """
        Generate an end-of-transmission frame.

        All black to signal completion.
        """
        return np.zeros(
            (self.grid_height, self.grid_width, 3),
            dtype=np.uint8
        )

    def _bytes_to_cells(self, data: bytes) -> List[int]:
        """
        Convert bytes to cell grayscale values.

        Each byte produces 4 cells (2 bits each).
        """
        cells = []
        for byte in data:
            # MSB first
            cells.append(bits_to_gray((byte >> 6) & 0x03))
            cells.append(bits_to_gray((byte >> 4) & 0x03))
            cells.append(bits_to_gray((byte >> 2) & 0x03))
            cells.append(bits_to_gray(byte & 0x03))
        return cells

    def grid_to_frame(self, grid: np.ndarray) -> np.ndarray:
        """
        Expand cell grid to full frame resolution.

        Args:
            grid: Cell grid (grid_height, grid_width, 3)

        Returns:
            Frame (FRAME_HEIGHT, FRAME_WIDTH, 3)
        """
        # Use numpy repeat for efficient scaling
        frame = np.repeat(grid, self.cell_size, axis=0)
        frame = np.repeat(frame, self.cell_size, axis=1)

        # Ensure exact dimensions (handle rounding)
        frame = frame[:FRAME_HEIGHT, :FRAME_WIDTH]

        # Pad if needed
        if frame.shape[0] < FRAME_HEIGHT or frame.shape[1] < FRAME_WIDTH:
            padded = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame
            frame = padded

        return frame

    def encode_to_frame(self, block: Block) -> np.ndarray:
        """
        Encode a block directly to full frame resolution.

        Args:
            block: Block to encode

        Returns:
            Frame (FRAME_HEIGHT, FRAME_WIDTH, 3)
        """
        grid = self.encode_block(block)
        return self.grid_to_frame(grid)

    def get_capacity(self) -> Tuple[int, int]:
        """
        Get cell capacity.

        Returns:
            (total_interior_cells, bytes_capacity)
        """
        interior_cells = (self.interior_right - self.interior_left) * \
                         (self.interior_bottom - self.interior_top)
        byte_capacity = interior_cells // 4  # 4 cells per byte
        return interior_cells, byte_capacity


class TestPatternGenerator:
    """
    Generates test patterns for calibration and debugging.
    """

    def __init__(self, profile: EncodingProfile = DEFAULT_PROFILE):
        self.encoder = FrameEncoder(profile)

    def grayscale_gradient(self) -> np.ndarray:
        """Generate horizontal grayscale gradient pattern."""
        grid = self.encoder.sync_pattern.copy()

        for row in range(self.encoder.interior_top, self.encoder.interior_bottom):
            for col in range(self.encoder.interior_left, self.encoder.interior_right):
                # Gradient from left to right
                progress = (col - self.encoder.interior_left) / \
                           (self.encoder.interior_right - self.encoder.interior_left)
                gray = int(progress * 255)
                grid[row, col] = (gray, gray, gray)

        return self.encoder.grid_to_frame(grid)

    def level_test(self) -> np.ndarray:
        """Generate 4-level test pattern."""
        grid = self.encoder.sync_pattern.copy()

        height = self.encoder.interior_bottom - self.encoder.interior_top
        width = self.encoder.interior_right - self.encoder.interior_left

        for row in range(self.encoder.interior_top, self.encoder.interior_bottom):
            for col in range(self.encoder.interior_left, self.encoder.interior_right):
                # Divide into 4 horizontal bands
                rel_row = row - self.encoder.interior_top
                band = int(rel_row / height * 4)
                band = min(band, 3)
                gray = GRAY_LEVELS[band]
                grid[row, col] = (gray, gray, gray)

        return self.encoder.grid_to_frame(grid)

    def checkerboard(self, cell_size: int = 2) -> np.ndarray:
        """Generate checkerboard pattern for alignment testing."""
        grid = self.encoder.sync_pattern.copy()

        for row in range(self.encoder.interior_top, self.encoder.interior_bottom):
            for col in range(self.encoder.interior_left, self.encoder.interior_right):
                rel_row = row - self.encoder.interior_top
                rel_col = col - self.encoder.interior_left
                if ((rel_row // cell_size) + (rel_col // cell_size)) % 2 == 0:
                    grid[row, col] = (0, 0, 0)
                else:
                    grid[row, col] = (255, 255, 255)

        return self.encoder.grid_to_frame(grid)

    def all_levels(self) -> np.ndarray:
        """Generate pattern with all 4 levels in stripes."""
        grid = self.encoder.sync_pattern.copy()

        width = self.encoder.interior_right - self.encoder.interior_left

        for row in range(self.encoder.interior_top, self.encoder.interior_bottom):
            for col in range(self.encoder.interior_left, self.encoder.interior_right):
                rel_col = col - self.encoder.interior_left
                # 4 vertical stripes
                stripe = int(rel_col / width * 4)
                stripe = min(stripe, 3)
                gray = GRAY_LEVELS[stripe]
                grid[row, col] = (gray, gray, gray)

        return self.encoder.grid_to_frame(grid)
