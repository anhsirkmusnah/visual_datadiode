"""
Visual Data Diode - Frame Synchronization

Detects sync borders and corner markers for frame alignment.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    EncodingProfile, DEFAULT_PROFILE,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_WHITE, COLOR_BLACK,
    CORNER_TOP_LEFT, CORNER_TOP_RIGHT, CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT,
    FRAME_WIDTH, FRAME_HEIGHT
)


@dataclass
class SyncResult:
    """Result of sync detection."""
    is_synced: bool
    confidence: float  # 0.0 to 1.0
    corners: Optional[List[Tuple[int, int]]]  # [TL, TR, BL, BR] pixel coords
    cell_bounds: Optional[Tuple[int, int, int, int]]  # (left, top, right, bottom)
    message: str = ""


class FrameSync:
    """
    Detects sync borders and aligns frames for decoding.

    Uses the cyan/magenta border pattern and corner markers
    to locate the data grid within a captured frame.
    """

    def __init__(self, profile: EncodingProfile = DEFAULT_PROFILE):
        """
        Initialize sync detector.

        Args:
            profile: Encoding profile defining cell size
        """
        self.profile = profile
        self.cell_size = profile.cell_size
        self.grid_width = profile.grid_width
        self.grid_height = profile.grid_height
        self.border_width = profile.sync_border_width

        # Color detection thresholds
        self.cyan_hue_range = (80, 100)  # HSV hue for cyan
        self.magenta_hue_range = (140, 160)  # HSV hue for magenta

        # Minimum confidence for sync
        self.min_confidence = 0.7

        # Cache for corner templates
        self._corner_templates = None

    def detect_sync(self, frame: np.ndarray) -> SyncResult:
        """
        Detect sync pattern in a frame.

        Args:
            frame: RGB frame (FRAME_HEIGHT, FRAME_WIDTH, 3)

        Returns:
            SyncResult with detection info
        """
        # Step 1: Detect border colors
        border_mask = self._detect_border_colors(frame)

        if np.sum(border_mask) < 100:
            return SyncResult(
                is_synced=False,
                confidence=0.0,
                corners=None,
                cell_bounds=None,
                message="No sync colors detected"
            )

        # Step 2: Find frame boundaries from border
        bounds = self._find_frame_bounds(border_mask)

        if bounds is None:
            return SyncResult(
                is_synced=False,
                confidence=0.0,
                corners=None,
                cell_bounds=None,
                message="Could not determine frame boundaries"
            )

        left, top, right, bottom = bounds

        # Step 3: Detect corner markers
        corners = self._detect_corners(frame, bounds)

        if corners is None:
            # Try with just bounds
            confidence = self._calculate_confidence(frame, bounds, None)

            if confidence >= self.min_confidence:
                return SyncResult(
                    is_synced=True,
                    confidence=confidence,
                    corners=None,
                    cell_bounds=bounds,
                    message="Synced (no corner verification)"
                )

            return SyncResult(
                is_synced=False,
                confidence=confidence,
                corners=None,
                cell_bounds=bounds,
                message="Corner markers not detected"
            )

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(frame, bounds, corners)

        return SyncResult(
            is_synced=confidence >= self.min_confidence,
            confidence=confidence,
            corners=corners,
            cell_bounds=bounds,
            message="Synced" if confidence >= self.min_confidence else "Low confidence"
        )

    def _detect_border_colors(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect cyan and magenta pixels using fast downsampled detection.

        Returns:
            Binary mask of sync-colored pixels (at original resolution)
        """
        return self._detect_border_colors_fast(frame)

    def _detect_border_colors_fast(self, frame: np.ndarray) -> np.ndarray:
        """
        Fast border detection using downsampling and border-only checking.
        """
        h, w = frame.shape[:2]

        # Downsample for faster detection (4x4 = 16x speedup)
        scale = 4
        small = frame[::scale, ::scale, :]
        sh, sw = small.shape[:2]

        # Only check border regions of downsampled image
        border_check = 30  # 30 pixels at 1/4 scale = 120 at full scale

        # Create small mask
        small_mask = np.zeros((sh, sw), dtype=bool)

        # Check border strips
        # Top strip
        region = small[:border_check, :]
        small_mask[:border_check, :] = self._detect_sync_colors_rgb(region)

        # Bottom strip
        region = small[sh - border_check:, :]
        small_mask[sh - border_check:, :] = self._detect_sync_colors_rgb(region)

        # Left strip (excluding corners already done)
        region = small[border_check:sh - border_check, :border_check]
        small_mask[border_check:sh - border_check, :border_check] = self._detect_sync_colors_rgb(region)

        # Right strip
        region = small[border_check:sh - border_check, sw - border_check:]
        small_mask[border_check:sh - border_check, sw - border_check:] = self._detect_sync_colors_rgb(region)

        # Upscale mask back to original size using nearest neighbor
        mask = np.repeat(np.repeat(small_mask, scale, axis=0), scale, axis=1)

        # Trim to original size (in case of rounding)
        mask = mask[:h, :w]

        return mask

    def _detect_sync_colors_rgb(self, region: np.ndarray) -> np.ndarray:
        """Fast RGB-based sync color detection for a region."""
        r = region[:, :, 0].astype(np.int16)
        g = region[:, :, 1].astype(np.int16)
        b = region[:, :, 2].astype(np.int16)

        # Cyan: low R, high G, high B (0, 255, 255)
        cyan_mask = (r < 150) & (g > 150) & (b > 150) & ((g - r) > 50) & ((b - r) > 50)

        # Magenta: high R, low G, high B (255, 0, 255)
        magenta_mask = (r > 150) & (g < 150) & (b > 150) & ((r - g) > 50) & ((b - g) > 50)

        return cyan_mask | magenta_mask

    def _detect_border_colors_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Full frame RGB-based border detection (slower, for fallback)."""
        return self._detect_sync_colors_rgb(frame)

    def _find_frame_bounds(
        self, border_mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find frame boundaries from border mask.

        Returns:
            (left, top, right, bottom) or None
        """
        # Find rows and columns with significant border pixels
        row_sums = np.sum(border_mask, axis=1)
        col_sums = np.sum(border_mask, axis=0)

        # Threshold for "significant" border presence
        threshold = max(10, border_mask.shape[1] * 0.02)

        # Find top/bottom
        top_rows = np.where(row_sums > threshold)[0]
        if len(top_rows) == 0:
            return None

        top = int(top_rows[0])
        bottom = int(top_rows[-1])

        # Find left/right
        left_cols = np.where(col_sums > threshold)[0]
        if len(left_cols) == 0:
            return None

        left = int(left_cols[0])
        right = int(left_cols[-1])

        # Validate bounds
        if right - left < 100 or bottom - top < 100:
            return None

        return (left, top, right, bottom)

    def _detect_corners(
        self, frame: np.ndarray, bounds: Tuple[int, int, int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Detect corner marker patterns.

        Returns:
            List of 4 corner positions [(x,y), ...] or None
        """
        left, top, right, bottom = bounds

        # Calculate expected corner positions (3x3 cells in each corner)
        corner_size = self.cell_size * 3

        corner_regions = [
            (left, top, "TL", CORNER_TOP_LEFT),
            (right - corner_size, top, "TR", CORNER_TOP_RIGHT),
            (left, bottom - corner_size, "BL", CORNER_BOTTOM_LEFT),
            (right - corner_size, bottom - corner_size, "BR", CORNER_BOTTOM_RIGHT)
        ]

        corners = []

        for x, y, name, pattern in corner_regions:
            # Extract region
            region = frame[y:y + corner_size, x:x + corner_size]

            if region.shape[0] < corner_size or region.shape[1] < corner_size:
                return None

            # Check if pattern matches
            if self._match_corner_pattern(region, pattern):
                corners.append((x, y))
            else:
                return None  # All corners must match

        return corners

    def _match_corner_pattern(
        self, region: np.ndarray, pattern: List[int]
    ) -> bool:
        """
        Check if region matches corner pattern.

        Args:
            region: Image region (corner_size x corner_size x 3)
            pattern: Expected pattern (9 values, 1=white, 0=black)

        Returns:
            True if pattern matches
        """
        cell_size = region.shape[0] // 3

        matches = 0

        for i in range(3):
            for j in range(3):
                # Sample center of cell
                y1 = i * cell_size + cell_size // 4
                y2 = (i + 1) * cell_size - cell_size // 4
                x1 = j * cell_size + cell_size // 4
                x2 = (j + 1) * cell_size - cell_size // 4

                cell_region = region[y1:y2, x1:x2]
                mean_value = np.mean(cell_region)

                expected_white = pattern[i * 3 + j] == 1

                if expected_white and mean_value > 128:
                    matches += 1
                elif not expected_white and mean_value < 128:
                    matches += 1

        return matches >= 7  # Allow 2 mismatches

    def _calculate_confidence(
        self,
        frame: np.ndarray,
        bounds: Tuple[int, int, int, int],
        corners: Optional[List[Tuple[int, int]]]
    ) -> float:
        """
        Calculate sync confidence score.

        Returns:
            Confidence value from 0.0 to 1.0
        """
        confidence = 0.0
        left, top, right, bottom = bounds

        # Check aspect ratio (should be close to 1280/720 = 1.78)
        width = right - left
        height = bottom - top
        aspect = width / height if height > 0 else 0

        expected_aspect = FRAME_WIDTH / FRAME_HEIGHT

        if 0.9 < aspect / expected_aspect < 1.1:
            confidence += 0.3
        elif 0.8 < aspect / expected_aspect < 1.2:
            confidence += 0.15

        # Check size (should be close to expected)
        expected_width = self.grid_width * self.cell_size
        expected_height = self.grid_height * self.cell_size

        if 0.9 < width / expected_width < 1.1:
            confidence += 0.2
        if 0.9 < height / expected_height < 1.1:
            confidence += 0.2

        # Corners detected
        if corners is not None and len(corners) == 4:
            confidence += 0.3

        return min(1.0, confidence)

    def extract_grid(
        self, frame: np.ndarray, sync_result: SyncResult
    ) -> Optional[np.ndarray]:
        """
        Extract the cell grid from a synced frame by sampling cell centers.

        Directly samples the center pixel of each cell to preserve discrete
        gray levels. This is essential for accurate data recovery.

        Args:
            frame: RGB frame
            sync_result: Result from detect_sync

        Returns:
            Grid array of shape (grid_height, grid_width, 3)
        """
        if not sync_result.is_synced or sync_result.cell_bounds is None:
            return None

        left, top, right, bottom = sync_result.cell_bounds

        # Calculate cell size from bounds
        width = right - left
        height = bottom - top

        cell_w = width / self.grid_width
        cell_h = height / self.grid_height

        # Vectorized cell center sampling - fast and preserves discrete values
        cols = np.arange(self.grid_width)
        rows = np.arange(self.grid_height)

        # Calculate center pixel coordinates for each cell
        cx = (left + (cols + 0.5) * cell_w).astype(int)
        cy = (top + (rows + 0.5) * cell_h).astype(int)

        # Clip to frame bounds
        cx = np.clip(cx, 0, frame.shape[1] - 1)
        cy = np.clip(cy, 0, frame.shape[0] - 1)

        # Sample single pixel at each cell center (vectorized)
        grid = frame[cy[:, None], cx[None, :]]

        return grid


class AdaptiveSync:
    """
    Adaptive sync detector that adjusts to capture conditions.

    Learns optimal detection parameters from initial frames.
    """

    def __init__(self, profile: EncodingProfile = DEFAULT_PROFILE):
        self.base_sync = FrameSync(profile)
        self.calibrated = False

        # Calibration results
        self.color_offsets = np.zeros(3)  # RGB offset
        self.brightness_scale = 1.0

    def calibrate(self, frames: List[np.ndarray]) -> bool:
        """
        Calibrate from sample frames.

        Args:
            frames: List of captured frames with sync pattern

        Returns:
            True if calibration successful
        """
        if not frames:
            return False

        # Detect sync in each frame
        good_frames = []
        for frame in frames:
            result = self.base_sync.detect_sync(frame)
            if result.is_synced:
                good_frames.append((frame, result))

        if len(good_frames) < 3:
            return False

        # Analyze detected colors
        # (Placeholder for more sophisticated calibration)
        self.calibrated = True
        return True

    def detect_sync(self, frame: np.ndarray) -> SyncResult:
        """
        Detect sync with calibration applied.

        Args:
            frame: RGB frame

        Returns:
            SyncResult
        """
        if self.calibrated:
            # Apply calibration
            frame = self._apply_calibration(frame)

        return self.base_sync.detect_sync(frame)

    def _apply_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply calibration corrections to frame."""
        corrected = frame.astype(np.float32)
        corrected = corrected * self.brightness_scale + self.color_offsets
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return corrected
