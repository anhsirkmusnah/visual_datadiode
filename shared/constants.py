"""
Visual Data Diode - Shared Constants

Defines all protocol constants, color values, and configuration defaults.
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Tuple

# =============================================================================
# Resolution and Frame Settings
# =============================================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEFAULT_FPS = 20

# =============================================================================
# Encoding Profiles
# =============================================================================

@dataclass(frozen=True)
class EncodingProfile:
    """Defines cell size and resulting grid dimensions."""
    name: str
    cell_size: int

    @property
    def grid_width(self) -> int:
        return FRAME_WIDTH // self.cell_size

    @property
    def grid_height(self) -> int:
        return FRAME_HEIGHT // self.cell_size

    @property
    def total_cells(self) -> int:
        return self.grid_width * self.grid_height

    @property
    def sync_border_width(self) -> int:
        return 2  # cells

    @property
    def interior_width(self) -> int:
        return self.grid_width - 2 * self.sync_border_width

    @property
    def interior_height(self) -> int:
        return self.grid_height - 2 * self.sync_border_width

    @property
    def header_cells(self) -> int:
        """One row for header."""
        return self.interior_width

    @property
    def payload_cells(self) -> int:
        """All interior cells minus header row."""
        return self.interior_width * (self.interior_height - 1)

    @property
    def payload_bytes(self) -> int:
        """Raw payload capacity in bytes (2 bits per cell)."""
        return self.payload_cells * 2 // 8


PROFILE_CONSERVATIVE = EncodingProfile("conservative", 16)
PROFILE_STANDARD = EncodingProfile("standard", 10)
PROFILE_AGGRESSIVE = EncodingProfile("aggressive", 8)

DEFAULT_PROFILE = PROFILE_STANDARD

# =============================================================================
# Grayscale Encoding Levels
# =============================================================================

# 4 grayscale levels for 2 bits per cell
GRAY_LEVELS = [0, 85, 170, 255]

# Thresholds for decoding (midpoints between levels)
GRAY_THRESHOLDS = [43, 128, 213]

def gray_to_bits(gray_value: int) -> int:
    """Convert grayscale value to 2-bit value (0-3)."""
    if gray_value < GRAY_THRESHOLDS[0]:
        return 0
    elif gray_value < GRAY_THRESHOLDS[1]:
        return 1
    elif gray_value < GRAY_THRESHOLDS[2]:
        return 2
    else:
        return 3

def bits_to_gray(bits: int) -> int:
    """Convert 2-bit value (0-3) to grayscale value."""
    return GRAY_LEVELS[bits & 0x03]

# =============================================================================
# Sync Border Colors (RGB tuples)
# =============================================================================

COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Corner marker patterns (3x3 grids, W=White, B=Black)
# Each is a list of 9 values (row-major): 1=White, 0=Black
CORNER_TOP_LEFT = [
    1, 1, 1,
    1, 0, 1,
    1, 1, 0
]

CORNER_TOP_RIGHT = [
    0, 1, 1,
    1, 0, 1,
    1, 1, 1
]

CORNER_BOTTOM_LEFT = [
    1, 1, 0,
    1, 0, 1,
    1, 1, 1
]

CORNER_BOTTOM_RIGHT = [
    0, 1, 0,
    1, 0, 1,
    1, 1, 1
]

# =============================================================================
# Block Header Format
# =============================================================================

HEADER_SIZE = 20  # bytes

class BlockFlags(IntFlag):
    """Flags for block header."""
    NONE = 0x00
    FIRST_BLOCK = 0x01
    LAST_BLOCK = 0x02
    ENCRYPTED = 0x04
    COMPRESSED = 0x08

# Header field offsets and sizes
HEADER_SESSION_ID_OFFSET = 0
HEADER_SESSION_ID_SIZE = 4

HEADER_BLOCK_INDEX_OFFSET = 4
HEADER_BLOCK_INDEX_SIZE = 4

HEADER_TOTAL_BLOCKS_OFFSET = 8
HEADER_TOTAL_BLOCKS_SIZE = 4

HEADER_FILE_SIZE_OFFSET = 12
HEADER_FILE_SIZE_SIZE = 4

HEADER_PAYLOAD_SIZE_OFFSET = 16
HEADER_PAYLOAD_SIZE_SIZE = 2

HEADER_FLAGS_OFFSET = 18
HEADER_FLAGS_SIZE = 1

HEADER_RESERVED_OFFSET = 19
HEADER_RESERVED_SIZE = 1

# =============================================================================
# FEC Configuration
# =============================================================================

# Reed-Solomon parameters
RS_SYMBOL_SIZE = 8  # bits (GF(2^8))
RS_MAX_CODEWORD = 255  # n for RS(255, k)
RS_DEFAULT_NSYM = 32  # 2t = 32, can correct 16 symbol errors

# FEC overhead ratio (default 10%)
DEFAULT_FEC_RATIO = 0.10

# =============================================================================
# CRC Configuration
# =============================================================================

CRC_SIZE = 4  # bytes (CRC-32)

# =============================================================================
# Timing Configuration
# =============================================================================

SYNC_FRAME_COUNT = 10  # Number of sync-only frames at start
END_FRAME_COUNT = 10   # Number of black frames at end
DEFAULT_REPEAT_COUNT = 2  # Transmit each block this many times

# =============================================================================
# File Metadata (Block 0)
# =============================================================================

FILE_HASH_SIZE = 32  # SHA-256
FILENAME_LENGTH_SIZE = 4
AES_NONCE_SIZE = 12  # For AES-GCM
AES_TAG_SIZE = 16    # For AES-GCM

# =============================================================================
# Cell Sampling Configuration
# =============================================================================

# Sample center portion of cell to avoid edge effects
CELL_SAMPLE_RATIO = 0.6  # Sample inner 60% of cell

# =============================================================================
# Protocol Version
# =============================================================================

PROTOCOL_VERSION = 1
PROTOCOL_MAGIC = b'VDDP'  # Visual Data Diode Protocol
