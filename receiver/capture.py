"""
Visual Data Diode - Frame Capture

Captures frames from USB HDMI capture device using OpenCV.
Optimized for MS2130 chipset with YUV422 support.
"""

import numpy as np
from typing import Optional, Tuple, List, Generator
import threading
import queue
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import FRAME_WIDTH, FRAME_HEIGHT

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class FrameCapture:
    """
    Captures frames from a USB HDMI capture device.

    Uses OpenCV with DirectShow backend on Windows for best compatibility
    with MS2130-based capture cards.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = 30,
        use_mjpeg: bool = False
    ):
        """
        Initialize capture device.

        Args:
            device_index: Camera/capture device index
            width: Capture width
            height: Capture height
            fps: Target capture FPS
            use_mjpeg: Use MJPEG format (vs YUV422)
        """
        if not OPENCV_AVAILABLE:
            raise ImportError(
                "OpenCV not available. Install with: pip install opencv-python"
            )

        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.use_mjpeg = use_mjpeg

        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0
        self.actual_fps = 0.0
        self._fps_start_time = 0.0
        self._fps_frame_count = 0

    def open(self) -> bool:
        """
        Open the capture device.

        Returns:
            True if successful
        """
        try:
            # Use DirectShow backend on Windows for MS2130 compatibility
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                # Try without DirectShow
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                return False

            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Set format
            if self.use_mjpeg:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            else:
                # Try YUV422 (YUYV)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))

            # Reduce buffer size for lower latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Capture opened: {int(actual_width)}x{int(actual_height)} @ {actual_fps} FPS")

            return True

        except Exception as e:
            print(f"Failed to open capture device: {e}")
            return False

    def close(self):
        """Close the capture device."""
        self.stop_capture()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start_capture(self):
        """Start continuous capture in background thread."""
        if self._running:
            return

        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError("Failed to open capture device")

        self._running = True
        self._fps_start_time = time.time()
        self._fps_frame_count = 0

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop_capture(self):
        """Stop continuous capture."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self):
        """Background capture loop."""
        while self._running:
            ret, frame = self.cap.read()

            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize if needed
                if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (self.width, self.height),
                        interpolation=cv2.INTER_LINEAR
                    )

                # Update last frame
                with self._lock:
                    self._last_frame = frame_rgb

                # Add to queue
                try:
                    self._frame_queue.put_nowait(frame_rgb)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame_rgb)
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass

                self.frames_captured += 1
                self._fps_frame_count += 1

                # Update FPS every second
                elapsed = time.time() - self._fps_start_time
                if elapsed >= 1.0:
                    self.actual_fps = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._fps_start_time = time.time()

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next captured frame.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            RGB frame as numpy array, or None if timeout
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get most recent frame (may skip frames).

        Returns:
            RGB frame as numpy array, or None if no frame
        """
        with self._lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame directly (blocking).

        Returns:
            RGB frame as numpy array, or None if failed
        """
        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR
                )

            return frame_rgb

        return None

    def frames(self, count: int = -1) -> Generator[np.ndarray, None, None]:
        """
        Generator for captured frames.

        Args:
            count: Number of frames to capture (-1 for infinite)

        Yields:
            RGB frames as numpy arrays
        """
        captured = 0

        while count < 0 or captured < count:
            frame = self.get_frame(timeout=1.0)
            if frame is not None:
                yield frame
                captured += 1
            elif not self._running:
                break

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get number of frames in queue."""
        return self._frame_queue.qsize()


def list_capture_devices() -> List[dict]:
    """
    List available capture devices.

    Returns:
        List of device info dictionaries
    """
    if not OPENCV_AVAILABLE:
        return []

    devices = []

    # Try to open devices 0-9
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            devices.append({
                'index': i,
                'width': int(width),
                'height': int(height),
                'fps': fps,
                'description': f"Device {i}: {int(width)}x{int(height)} @ {fps} FPS"
            })

            cap.release()

    return devices


def check_opencv_available() -> bool:
    """Check if OpenCV is available."""
    return OPENCV_AVAILABLE


class CaptureCalibrator:
    """
    Calibration helper for capture device.

    Helps adjust settings for optimal capture quality.
    """

    def __init__(self, capture: FrameCapture):
        self.capture = capture

    def measure_latency(self, num_samples: int = 10) -> float:
        """
        Measure capture latency.

        Returns average time between frame requests and responses.
        """
        latencies = []

        for _ in range(num_samples):
            start = time.time()
            frame = self.capture.read_frame()
            end = time.time()

            if frame is not None:
                latencies.append(end - start)

        if latencies:
            return sum(latencies) / len(latencies)
        return 0.0

    def measure_color_accuracy(
        self, frame: np.ndarray, expected_colors: List[Tuple[int, int, int]]
    ) -> dict:
        """
        Measure how accurately colors are captured.

        Args:
            frame: Captured frame
            expected_colors: List of expected RGB colors

        Returns:
            Dictionary with color accuracy metrics
        """
        # This would analyze regions of known color
        # For now, return placeholder
        return {
            'mean_error': 0.0,
            'max_error': 0.0,
            'recommendations': []
        }

    def analyze_noise(self, num_frames: int = 30) -> dict:
        """
        Analyze frame-to-frame noise.

        Captures multiple frames of a static scene and measures variation.
        """
        frames = []

        for _ in range(num_frames):
            frame = self.capture.read_frame()
            if frame is not None:
                frames.append(frame.astype(np.float32))

        if len(frames) < 2:
            return {'noise_level': 0.0, 'recommendations': []}

        # Stack and compute per-pixel std deviation
        stack = np.stack(frames, axis=0)
        noise = np.std(stack, axis=0)
        mean_noise = np.mean(noise)

        recommendations = []
        if mean_noise > 5.0:
            recommendations.append("High noise detected. Consider using larger cell sizes.")
        if mean_noise > 10.0:
            recommendations.append("Very high noise. Check cable connections and capture settings.")

        return {
            'noise_level': float(mean_noise),
            'noise_map': noise,
            'recommendations': recommendations
        }
