"""
Visual Data Diode - CUDA-Accelerated Frame Decoder

GPU-accelerated decoding using CuPy for massive parallelism.
Falls back to CPU implementation if CUDA is not available.

Features:
- Batch frame processing on GPU
- Vectorized luma and color decoding
- Optional hardware video decoding via NVDEC
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Check for CUDA availability
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    CUDA_DEVICE_NAME = cp.cuda.Device().name.decode() if cp.cuda.is_available() else "N/A"
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = "N/A"
    cp = None

from .enhanced_encoding import (
    EnhancedEncodingProfile,
    ENHANCED_LUMA_LEVELS,
    ENHANCED_LUMA_THRESHOLDS,
    ColorState,
    CalibrationData,
    luma_to_bits_enhanced,
    CALIBRATION_MAGIC,
    SYNC_MAGIC,
    ENHANCED_PROFILE_CONSERVATIVE,
)
from .enhanced_decoder import EnhancedDecodeResult, EnhancedFrameDecoder


def check_cuda_available() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    return CUDA_AVAILABLE


def get_cuda_info() -> dict:
    """Get CUDA device information."""
    if not CUDA_AVAILABLE:
        return {
            'available': False,
            'device_name': 'N/A',
            'memory_total': 0,
            'memory_free': 0,
        }

    try:
        device = cp.cuda.Device()
        mem_info = device.mem_info
        return {
            'available': True,
            'device_name': CUDA_DEVICE_NAME,
            'memory_total': mem_info[1],
            'memory_free': mem_info[0],
        }
    except:
        return {
            'available': False,
            'device_name': 'Error',
            'memory_total': 0,
            'memory_free': 0,
        }


class CUDAFrameDecoder:
    """
    CUDA-accelerated frame decoder.

    Uses GPU for parallel processing of luma and color decoding.
    Falls back to CPU if CUDA is not available.
    """

    def __init__(self, profile: EnhancedEncodingProfile = ENHANCED_PROFILE_CONSERVATIVE):
        """
        Initialize CUDA decoder.

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

        # Calibration data
        self.calibration: Optional[CalibrationData] = None

        # Use CUDA if available
        self.use_cuda = CUDA_AVAILABLE

        # Pre-allocate GPU arrays for luma thresholds
        if self.use_cuda:
            self._luma_thresholds_gpu = cp.array(ENHANCED_LUMA_THRESHOLDS, dtype=cp.float32)
            self._luma_levels_gpu = cp.array(ENHANCED_LUMA_LEVELS, dtype=cp.float32)

            # Color offsets (must match encoder)
            self._color_offsets_gpu = cp.array([
                [-30, 60],   # RED
                [-50, 10],   # YELLOW
                [-20, -50],  # GREEN
                [30, -60],   # CYAN
            ], dtype=cp.float32)

        # CPU fallback decoder
        self.cpu_decoder = EnhancedFrameDecoder(profile)

        # Statistics
        self.frames_decoded = 0
        self.gpu_frames = 0
        self.cpu_frames = 0

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

        # Check for calibration frame (use CPU - infrequent)
        if self._is_calibration_frame(interior):
            result = self.cpu_decoder.decode_grid(grid)
            if result.calibration and result.calibration.is_valid:
                self.calibration = result.calibration
                self._update_gpu_calibration()
            return result

        # Check for sync frame (use CPU - infrequent)
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

        # Use GPU for data frames
        if self.use_cuda:
            try:
                return self._decode_gpu(interior)
            except Exception as e:
                # Fall back to CPU on GPU error
                self.cpu_frames += 1
                return self.cpu_decoder.decode_grid(grid)
        else:
            self.cpu_frames += 1
            return self.cpu_decoder.decode_grid(grid)

    def decode_batch(self, grids: List[np.ndarray]) -> List[EnhancedDecodeResult]:
        """
        Decode multiple grids in batch on GPU.

        Args:
            grids: List of cell grids

        Returns:
            List of EnhancedDecodeResult
        """
        if not self.use_cuda or len(grids) == 0:
            return [self.decode_grid(g) for g in grids]

        results = []

        # Separate calibration/sync frames from data frames
        data_indices = []
        interiors = []

        for i, grid in enumerate(grids):
            interior = grid[
                self.interior_top:self.interior_bottom,
                self.interior_left:self.interior_right
            ]

            if self._is_calibration_frame(interior):
                result = self.cpu_decoder.decode_grid(grid)
                if result.calibration and result.calibration.is_valid:
                    self.calibration = result.calibration
                    self._update_gpu_calibration()
                results.append((i, result))
            elif self._is_sync_frame(interior):
                results.append((i, EnhancedDecodeResult(
                    success=True,
                    is_calibration_frame=False,
                    is_sync_frame=True,
                    luma_data=None,
                    color_data=None,
                    luma_confidence=1.0,
                    color_confidence=1.0,
                    calibration=None,
                    message="Sync frame detected"
                )))
            else:
                data_indices.append(i)
                interiors.append(interior)

        # Batch process data frames on GPU
        if interiors:
            batch_results = self._decode_batch_gpu(interiors)
            for idx, result in zip(data_indices, batch_results):
                results.append((idx, result))

        # Sort by original index and extract results
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _decode_gpu(self, interior: np.ndarray) -> EnhancedDecodeResult:
        """Decode single frame on GPU."""
        self.frames_decoded += 1
        self.gpu_frames += 1

        # Transfer to GPU
        interior_gpu = cp.asarray(interior, dtype=cp.float32)

        # Decode luma
        luma_data, luma_conf = self._decode_luma_gpu(interior_gpu)

        # Decode color
        color_data, color_conf = self._decode_color_gpu(interior_gpu)

        return EnhancedDecodeResult(
            success=True,
            is_calibration_frame=False,
            is_sync_frame=False,
            luma_data=luma_data,
            color_data=color_data,
            luma_confidence=float(luma_conf),
            color_confidence=float(color_conf),
            calibration=None,
            message=f"GPU decoded: luma_conf={luma_conf:.2f}, color_conf={color_conf:.2f}"
        )

    def _decode_batch_gpu(self, interiors: List[np.ndarray]) -> List[EnhancedDecodeResult]:
        """Decode batch of frames on GPU."""
        batch_size = len(interiors)
        if batch_size == 0:
            return []

        self.frames_decoded += batch_size
        self.gpu_frames += batch_size

        # Stack interiors into batch tensor
        stacked = np.stack(interiors, axis=0)  # (batch, H, W, 3)
        batch_gpu = cp.asarray(stacked, dtype=cp.float32)

        results = []
        for i in range(batch_size):
            interior_gpu = batch_gpu[i]

            # Decode luma
            luma_data, luma_conf = self._decode_luma_gpu(interior_gpu)

            # Decode color
            color_data, color_conf = self._decode_color_gpu(interior_gpu)

            results.append(EnhancedDecodeResult(
                success=True,
                is_calibration_frame=False,
                is_sync_frame=False,
                luma_data=luma_data,
                color_data=color_data,
                luma_confidence=float(luma_conf),
                color_confidence=float(color_conf),
                calibration=None,
                message=f"GPU batch decoded"
            ))

        return results

    def _decode_luma_gpu(self, interior_gpu: 'cp.ndarray') -> Tuple[bytes, float]:
        """Decode luma data on GPU."""
        # Compute Y (luma) from RGB: Y = 0.299*R + 0.587*G + 0.114*B
        r = interior_gpu[:, :, 0]
        g = interior_gpu[:, :, 1]
        b = interior_gpu[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Flatten for processing
        gray_flat = gray.ravel()

        # Vectorized luma level detection
        thresholds = self._luma_thresholds_gpu

        # Compare against all thresholds at once
        # Result: level = sum(gray >= threshold) for each threshold
        comparisons = gray_flat[:, None] >= thresholds[None, :]  # (N, 7)
        levels = cp.sum(comparisons, axis=1).astype(cp.uint8)  # (N,)

        # Calculate confidence
        closest_level_values = self._luma_levels_gpu[levels]
        diffs = cp.abs(gray_flat - closest_level_values)
        confidences = 1.0 - cp.minimum(diffs / 18.0, 1.0)
        avg_confidence = float(cp.mean(confidences))

        # Transfer to CPU and pack tribits to bytes
        levels_cpu = cp.asnumpy(levels).tolist()
        luma_bytes = self._tribits_to_bytes(levels_cpu)

        # Truncate to expected length
        expected_luma_bytes = self.profile.luma_bits_per_frame // 8
        if len(luma_bytes) > expected_luma_bytes:
            luma_bytes = luma_bytes[:expected_luma_bytes]

        return luma_bytes, avg_confidence

    def _decode_color_gpu(self, interior_gpu: 'cp.ndarray') -> Tuple[bytes, float]:
        """Decode color data on GPU."""
        h, w = interior_gpu.shape[:2]
        groups_h = h // self.color_group_size
        groups_w = w // self.color_group_size

        # Reshape to extract 2x2 groups
        # interior_gpu shape: (H, W, 3)
        # Need to reshape to (groups_h, 2, groups_w, 2, 3)
        reshaped = interior_gpu[:groups_h * 2, :groups_w * 2, :].reshape(
            groups_h, self.color_group_size,
            groups_w, self.color_group_size,
            3
        )

        # Compute Y for each cell in group
        r = reshaped[:, :, :, :, 0]
        g = reshaped[:, :, :, :, 1]
        b = reshaped[:, :, :, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b

        # Compute Cb and Cr
        cb = 128 + 0.564 * (b - y)
        cr = 128 + 0.713 * (r - y)

        # Average over 2x2 groups
        cb_avg = cp.mean(cb, axis=(1, 3))  # (groups_h, groups_w)
        cr_avg = cp.mean(cr, axis=(1, 3))

        # Flatten
        cb_flat = cb_avg.ravel()
        cr_flat = cr_avg.ravel()

        # Compute distance to each color state
        color_offsets = self._color_offsets_gpu  # (4, 2) - [cb_offset, cr_offset]
        expected_cb = 128 + color_offsets[:, 0]  # (4,)
        expected_cr = 128 + color_offsets[:, 1]  # (4,)

        # Distances: (N, 4)
        dist_cb = (cb_flat[:, None] - expected_cb[None, :]) ** 2
        dist_cr = (cr_flat[:, None] - expected_cr[None, :]) ** 2
        distances = dist_cb + dist_cr

        # Find closest state
        states = cp.argmin(distances, axis=1).astype(cp.uint8)
        min_distances = cp.min(distances, axis=1)

        # Confidence
        confidences = cp.maximum(0.0, 1.0 - cp.sqrt(min_distances) / 100)
        avg_confidence = float(cp.mean(confidences))

        # Transfer to CPU and pack dibits to bytes
        states_cpu = cp.asnumpy(states).tolist()
        color_bytes = self._dibits_to_bytes(states_cpu)

        return color_bytes, avg_confidence

    def _is_calibration_frame(self, interior: np.ndarray) -> bool:
        """Check if interior contains calibration frame marker."""
        first_row = interior[0, -8:]

        if len(first_row) < 8:
            return False

        detected_levels = []
        for i in range(8):
            r, g, b = first_row[i]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            level = luma_to_bits_enhanced(gray)
            detected_levels.append(level)

        matches = sum(1 for a, b in zip(detected_levels, CALIBRATION_MAGIC) if a == b)
        return matches >= 6

    def _is_sync_frame(self, interior: np.ndarray) -> bool:
        """Check if interior contains sync frame marker."""
        first_row = interior[0, -8:]

        if len(first_row) < 8:
            return False

        detected_levels = []
        for i in range(8):
            r, g, b = first_row[i]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            level = luma_to_bits_enhanced(gray)
            detected_levels.append(level)

        matches = sum(1 for a, b in zip(detected_levels, SYNC_MAGIC) if a == b)
        return matches >= 7

    def _update_gpu_calibration(self):
        """Update GPU arrays with calibration data."""
        if not self.use_cuda or not self.calibration:
            return

        if self.calibration.luma_thresholds:
            self._luma_thresholds_gpu = cp.array(
                self.calibration.luma_thresholds, dtype=cp.float32
            )

        if self.calibration.color_centroids:
            # Update color offsets from calibration
            centroids = np.array(self.calibration.color_centroids[:4])  # Only first 4
            offsets = centroids - 128  # Convert to offsets
            self._color_offsets_gpu = cp.array(offsets, dtype=cp.float32)

    def _tribits_to_bytes(self, tribits: List[int]) -> bytes:
        """Convert list of 3-bit values to bytes."""
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

        if bits_in_buffer > 0:
            byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
            result.append(byte_val)

        return bytes(result)

    def _dibits_to_bytes(self, dibits: List[int]) -> bytes:
        """Convert list of 2-bit values to bytes."""
        result = bytearray()

        for i in range(0, len(dibits), 4):
            byte_val = 0
            for j in range(4):
                if i + j < len(dibits):
                    byte_val |= (dibits[i + j] & 0x03) << (6 - j * 2)
            result.append(byte_val)

        return bytes(result)

    def get_statistics(self) -> dict:
        """Get decoder statistics."""
        return {
            'frames_decoded': self.frames_decoded,
            'gpu_frames': self.gpu_frames,
            'cpu_frames': self.cpu_frames,
            'cuda_available': self.use_cuda,
            'device_name': CUDA_DEVICE_NAME if self.use_cuda else 'CPU',
            'has_calibration': self.calibration is not None and self.calibration.is_valid,
        }


class CUDAVideoProcessor:
    """
    High-performance video processor using CUDA acceleration.

    Features:
    - Hardware video decoding (NVDEC) when available
    - Batch frame processing on GPU
    - Parallel block assembly
    """

    def __init__(
        self,
        profile: EnhancedEncodingProfile = ENHANCED_PROFILE_CONSERVATIVE,
        batch_size: int = 32,
        use_nvdec: bool = True
    ):
        """
        Initialize CUDA video processor.

        Args:
            profile: Encoding profile
            batch_size: Number of frames to process in batch
            use_nvdec: Whether to use NVDEC for video decoding
        """
        self.profile = profile
        self.batch_size = batch_size
        self.use_nvdec = use_nvdec and CUDA_AVAILABLE

        # Initialize decoder
        self.decoder = CUDAFrameDecoder(profile)

        # Statistics
        self.total_frames = 0
        self.process_time = 0.0

    def process_video(
        self,
        video_path: str,
        progress_callback=None
    ) -> List[EnhancedDecodeResult]:
        """
        Process video file with CUDA acceleration.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(current, total)

        Returns:
            List of decode results
        """
        import cv2
        import time

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        batch = []

        start_time = time.time()

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                grid = self._extract_grid(frame)
                batch.append(grid)

                self.total_frames += 1

                # Process batch when full
                if len(batch) >= self.batch_size:
                    batch_results = self.decoder.decode_batch(batch)
                    results.extend(batch_results)
                    batch = []

                    if progress_callback:
                        progress_callback(self.total_frames, total_frames)

            # Process remaining frames
            if batch:
                batch_results = self.decoder.decode_batch(batch)
                results.extend(batch_results)

        finally:
            cap.release()

        self.process_time = time.time() - start_time

        return results

    def _extract_grid(self, frame: np.ndarray) -> np.ndarray:
        """Extract cell grid from frame."""
        cell_size = self.profile.cell_size
        grid_h = self.profile.grid_height
        grid_w = self.profile.grid_width

        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for row in range(grid_h):
            for col in range(grid_w):
                cy = row * cell_size + cell_size // 2
                cx = col * cell_size + cell_size // 2

                if cy < frame.shape[0] and cx < frame.shape[1]:
                    grid[row, col] = frame[cy, cx]

        return grid

    def get_statistics(self) -> dict:
        """Get processing statistics."""
        fps = self.total_frames / self.process_time if self.process_time > 0 else 0
        stats = self.decoder.get_statistics()
        stats.update({
            'total_frames_processed': self.total_frames,
            'process_time': self.process_time,
            'fps': fps,
        })
        return stats
