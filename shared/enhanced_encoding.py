"""
Visual Data Diode - Enhanced Encoding

8 luma levels (3 bits/cell) + 6 color states (per 2x2 cell group)
Optimized for MJPEG capture with YCbCr 4:2:0 subsampling.

Total capacity: ~3.5 bits per cell (+75% over base 2-bit encoding)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from enum import IntEnum

# =============================================================================
# Enhanced Luma Encoding (8 levels = 3 bits per cell)
# =============================================================================

# 8 grayscale levels with equal spacing
# Spacing of ~36 between levels for good separation
ENHANCED_LUMA_LEVELS = [0, 36, 73, 109, 146, 182, 219, 255]

# Thresholds for decoding (midpoints between levels)
ENHANCED_LUMA_THRESHOLDS = [18, 54, 91, 127, 164, 200, 237]

def luma_to_bits_enhanced(luma_value: float) -> int:
    """Convert luma value to 3-bit value (0-7)."""
    for i, thresh in enumerate(ENHANCED_LUMA_THRESHOLDS):
        if luma_value < thresh:
            return i
    return 7

def bits_to_luma_enhanced(bits: int) -> int:
    """Convert 3-bit value (0-7) to luma grayscale value."""
    return ENHANCED_LUMA_LEVELS[bits & 0x07]


# =============================================================================
# Enhanced Color Encoding (6 states optimized for MJPEG/YCbCr 4:2:0)
# =============================================================================

# 6 colors forming a hexagon in CbCr color space for maximum separation
# Each color has similar luminance (~128) to not interfere with luma encoding
# Format: (R, G, B) tuples

class ColorState(IntEnum):
    """6 color states for enhanced encoding."""
    RED = 0      # High Cr, Low Cb
    YELLOW = 1   # Medium Cr, Low Cb
    GREEN = 2    # Low Cr, Medium Cb
    CYAN = 3     # Low Cr, High Cb
    BLUE = 4     # Medium Cr, High Cb
    MAGENTA = 5  # High Cr, High Cb


# RGB colors with matched luminance (~128) for each state
# These are optimized for separation in YCbCr space
# Y = 0.299R + 0.587G + 0.114B
ENHANCED_COLOR_RGB = {
    ColorState.RED:     (255, 80, 80),    # Warm red, Y≈120
    ColorState.YELLOW:  (220, 220, 60),   # Yellow, Y≈200
    ColorState.GREEN:   (80, 200, 80),    # Green, Y≈160
    ColorState.CYAN:    (80, 220, 220),   # Cyan, Y≈185
    ColorState.BLUE:    (100, 100, 255),  # Blue, Y≈115
    ColorState.MAGENTA: (220, 80, 220),   # Magenta, Y≈130
}

# YCbCr approximate values for each color (for decoder reference)
# Cb = 128 + 0.564*(B-Y), Cr = 128 + 0.713*(R-Y)
ENHANCED_COLOR_YCBCR = {
    ColorState.RED:     (120, 90, 220),   # Low Cb, High Cr
    ColorState.YELLOW:  (200, 50, 140),   # Low Cb, Medium Cr
    ColorState.GREEN:   (160, 80, 75),    # Medium Cb, Low Cr
    ColorState.CYAN:    (185, 145, 55),   # High Cb, Low Cr
    ColorState.BLUE:    (115, 200, 115),  # High Cb, Medium Cr
    ColorState.MAGENTA: (130, 175, 195),  # High Cb, High Cr
}


def get_color_rgb(state: ColorState, luma_level: int) -> Tuple[int, int, int]:
    """
    Get RGB color for a cell with given color state and luma level.

    IMPORTANT: This function ensures the output RGB has the correct Y (luma)
    while applying the color tint in the CbCr (chrominance) domain.

    For MJPEG/YCbCr encoding, we must preserve luma accurately.

    Args:
        state: Color state (0-5)
        luma_level: Luma level (0-7)

    Returns:
        RGB tuple with precise luma and color tint
    """
    target_y = ENHANCED_LUMA_LEVELS[luma_level]

    # Define color offsets in CbCr space (relative to neutral)
    # These define how much to shift from gray in the blue-yellow and red-cyan axes
    # Cb offset (blue-yellow): positive = blue, negative = yellow
    # Cr offset (red-cyan): positive = red, negative = cyan
    # Color offsets balanced for both luma accuracy and color separation
    COLOR_CBCR_OFFSETS = {
        ColorState.RED:     (-30, 60),   # Yellow-ish Cb, Red Cr
        ColorState.YELLOW:  (-50, 10),   # Yellow Cb, slight red Cr
        ColorState.GREEN:   (-20, -50),  # Slight yellow Cb, Cyan Cr
        ColorState.CYAN:    (30, -60),   # Blue Cb, Cyan Cr
        ColorState.BLUE:    (60, -10),   # Blue Cb, slight cyan Cr
        ColorState.MAGENTA: (30, 50),    # Blue Cb, Red Cr
    }

    cb_offset, cr_offset = COLOR_CBCR_OFFSETS[state]

    # Scale color saturation at extremes to preserve luma accuracy while maintaining color detection
    # MIN_SATURATION = 0.7 gives good balance between color detection and luma accuracy
    MIN_SATURATION = 0.7
    if target_y < 32:
        saturation = MIN_SATURATION + (1.0 - MIN_SATURATION) * (target_y / 32.0)
    elif target_y > 223:
        saturation = MIN_SATURATION + (1.0 - MIN_SATURATION) * ((255 - target_y) / 32.0)
    else:
        saturation = 1.0

    cb_offset = int(cb_offset * saturation)
    cr_offset = int(cr_offset * saturation)

    # Convert YCbCr to RGB
    # Y = 0.299R + 0.587G + 0.114B
    # Cb = 128 + 0.564(B - Y)  =>  B = Y + (Cb - 128) / 0.564
    # Cr = 128 + 0.713(R - Y)  =>  R = Y + (Cr - 128) / 0.713

    # Target Cb and Cr (128 is neutral)
    cb = 128 + cb_offset
    cr = 128 + cr_offset

    # Convert back to RGB
    r = target_y + (cr - 128) / 0.713
    b = target_y + (cb - 128) / 0.564
    # G is derived from Y = 0.299R + 0.587G + 0.114B
    # G = (Y - 0.299R - 0.114B) / 0.587
    g = (target_y - 0.299 * r - 0.114 * b) / 0.587

    # Clamp to valid RGB range
    r = int(max(0, min(255, r)))
    g = int(max(0, min(255, g)))
    b = int(max(0, min(255, b)))

    return (r, g, b)


def detect_color_state(rgb: Tuple[float, float, float]) -> Tuple[ColorState, float]:
    """
    Detect color state from RGB values.

    Args:
        rgb: RGB tuple (can be float averages)

    Returns:
        (ColorState, confidence)
    """
    r, g, b = rgb

    # Convert to approximate CbCr
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 + 0.564 * (b - y)
    cr = 128 + 0.713 * (r - y)

    # Find closest color state in CbCr space
    min_dist = float('inf')
    best_state = ColorState.RED

    for state, (sy, scb, scr) in ENHANCED_COLOR_YCBCR.items():
        # Distance in CbCr plane (ignore Y for color detection)
        dist = (cb - scb) ** 2 + (cr - scr) ** 2
        if dist < min_dist:
            min_dist = dist
            best_state = state

    # Confidence based on distance (closer = higher confidence)
    # Max expected distance is ~100^2 = 10000 for opposite colors
    confidence = max(0.0, 1.0 - (min_dist / 10000) ** 0.5)

    return best_state, confidence


# =============================================================================
# Enhanced Encoding Profile
# =============================================================================

@dataclass
class EnhancedEncodingProfile:
    """
    Enhanced encoding profile with 8 luma levels + 6 color states.

    Color is encoded per 2x2 cell group (4 cells share one color).
    """
    name: str
    cell_size: int
    color_group_size: int = 2  # 2x2 cells share one color

    # Frame dimensions
    frame_width: int = 1920
    frame_height: int = 1080

    @property
    def grid_width(self) -> int:
        return self.frame_width // self.cell_size

    @property
    def grid_height(self) -> int:
        return self.frame_height // self.cell_size

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
    def interior_cells(self) -> int:
        return self.interior_width * self.interior_height

    @property
    def color_groups_width(self) -> int:
        """Number of color groups horizontally."""
        return self.interior_width // self.color_group_size

    @property
    def color_groups_height(self) -> int:
        """Number of color groups vertically."""
        return self.interior_height // self.color_group_size

    @property
    def total_color_groups(self) -> int:
        """Total number of 2x2 color groups."""
        return self.color_groups_width * self.color_groups_height

    @property
    def luma_bits_per_frame(self) -> int:
        """Bits from luma encoding (3 bits per cell)."""
        return self.interior_cells * 3

    @property
    def color_bits_per_frame(self) -> int:
        """
        Bits from color encoding.
        6 states = log2(6) ≈ 2.58 bits per group.
        For byte alignment: pack 3 groups into 7 bits (6^3=216 < 2^8=256).
        Effective: 7/3 ≈ 2.33 bits per group.
        """
        # Conservative: use 2 bits per group (4 of 6 states used for data)
        # This wastes 2 states but simplifies encoding
        return self.total_color_groups * 2

    @property
    def total_bits_per_frame(self) -> int:
        return self.luma_bits_per_frame + self.color_bits_per_frame

    @property
    def payload_bytes(self) -> int:
        """Total payload capacity in bytes."""
        return self.total_bits_per_frame // 8

    def get_info(self) -> dict:
        """Get profile information."""
        return {
            'name': self.name,
            'cell_size': self.cell_size,
            'grid': f'{self.grid_width}x{self.grid_height}',
            'interior': f'{self.interior_width}x{self.interior_height}',
            'interior_cells': self.interior_cells,
            'color_groups': self.total_color_groups,
            'luma_bits': self.luma_bits_per_frame,
            'color_bits': self.color_bits_per_frame,
            'total_bits': self.total_bits_per_frame,
            'payload_bytes': self.payload_bytes,
        }


# Pre-defined enhanced profiles
ENHANCED_PROFILE_CONSERVATIVE = EnhancedEncodingProfile("enhanced_conservative", 16)
ENHANCED_PROFILE_STANDARD = EnhancedEncodingProfile("enhanced_standard", 10)


# =============================================================================
# Calibration Data Structure
# =============================================================================

@dataclass
class CalibrationData:
    """
    Calibration data learned from calibration frame.

    Maps expected values to actually received values for accurate decoding.
    """
    # Luma calibration: expected level -> (received_mean, received_std)
    luma_calibration: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # Color calibration: expected state -> (received_cb_mean, received_cr_mean, std)
    color_calibration: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)

    # Computed thresholds for decoding
    luma_thresholds: List[float] = field(default_factory=list)
    color_centroids: List[Tuple[float, float]] = field(default_factory=list)

    # Calibration quality metrics
    luma_separation: float = 0.0  # Average separation between luma levels
    color_separation: float = 0.0  # Average separation between color states

    is_valid: bool = False

    def compute_thresholds(self):
        """Compute optimal thresholds from calibration data."""
        if len(self.luma_calibration) < 8:
            return

        # Compute luma thresholds as midpoints between measured levels
        measured_levels = [self.luma_calibration[i][0] for i in range(8)]
        self.luma_thresholds = []
        for i in range(7):
            threshold = (measured_levels[i] + measured_levels[i + 1]) / 2
            self.luma_thresholds.append(threshold)

        # Compute average luma separation
        separations = [measured_levels[i + 1] - measured_levels[i] for i in range(7)]
        self.luma_separation = sum(separations) / len(separations) if separations else 0

        # Compute color centroids
        if len(self.color_calibration) >= 6:
            self.color_centroids = [
                (self.color_calibration[i][0], self.color_calibration[i][1])
                for i in range(6)
            ]

            # Compute average color separation
            min_dists = []
            for i in range(6):
                cb1, cr1 = self.color_centroids[i]
                min_dist = float('inf')
                for j in range(6):
                    if i != j:
                        cb2, cr2 = self.color_centroids[j]
                        dist = ((cb1 - cb2) ** 2 + (cr1 - cr2) ** 2) ** 0.5
                        min_dist = min(min_dist, dist)
                min_dists.append(min_dist)
            self.color_separation = sum(min_dists) / len(min_dists) if min_dists else 0

        self.is_valid = self.luma_separation > 10 and len(self.luma_thresholds) == 7

    def decode_luma(self, value: float) -> int:
        """Decode luma value using calibrated thresholds."""
        if not self.luma_thresholds:
            return luma_to_bits_enhanced(value)

        for i, thresh in enumerate(self.luma_thresholds):
            if value < thresh:
                return i
        return 7

    def decode_color(self, cb: float, cr: float) -> Tuple[int, float]:
        """Decode color state using calibrated centroids.

        Only considers states 0-3 since those are the only ones used for data encoding.
        """
        if not self.color_centroids:
            # Fallback to default detection
            rgb_approx = (cr, 128, cb)  # Very rough approximation
            state, conf = detect_color_state(rgb_approx)
            return int(state) % 4, conf  # Map to 0-3

        # Find closest centroid (only check states 0-3)
        min_dist = float('inf')
        best_state = 0

        for i, (ccb, ccr) in enumerate(self.color_centroids):
            if i > 3:  # Only use first 4 states
                continue
            dist = (cb - ccb) ** 2 + (cr - ccr) ** 2
            if dist < min_dist:
                min_dist = dist
                best_state = i

        # Confidence based on distance
        confidence = max(0.0, 1.0 - (min_dist ** 0.5) / 100)

        return best_state, confidence


# =============================================================================
# Calibration Frame Marker
# =============================================================================

# Special pattern in first row to identify calibration frame
CALIBRATION_MAGIC = [0, 7, 0, 7, 0, 7, 0, 7]  # Alternating min/max luma

# Special pattern to identify sync frames (different from calibration)
# Pattern: [3, 3, 3, 3, 4, 4, 4, 4] - two groups of mid-gray levels
SYNC_MAGIC = [3, 3, 3, 3, 4, 4, 4, 4]

def is_calibration_frame_marker(first_row_luma: List[int]) -> bool:
    """Check if first row contains calibration magic pattern."""
    if len(first_row_luma) < 8:
        return False
    return first_row_luma[:8] == CALIBRATION_MAGIC


def is_sync_frame_marker(first_row_luma: List[int]) -> bool:
    """Check if first row contains sync magic pattern."""
    if len(first_row_luma) < 8:
        return False
    # Check for sync pattern (allow 1 mismatch for robustness)
    matches = sum(1 for a, b in zip(first_row_luma[:8], SYNC_MAGIC) if a == b)
    return matches >= 7
