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
        device_name: str = None,
        width: int = 1920,  # Request max resolution
        height: int = 1080,
        fps: int = 60,  # Request max FPS
        use_mjpeg: bool = False
    ):
        """
        Initialize capture device.

        Args:
            device_index: Camera/capture device index
            device_name: Device name (preferred over index if provided)
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
        self.device_name = device_name
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
        Open the capture device with optimized settings for high FPS.
        Uses MSMF backend which achieves best FPS on Windows USB capture devices.

        Returns:
            True if successful
        """
        try:
            backend_name = None

            # MSMF (Media Foundation) is MUCH faster than DirectShow for USB capture
            # Testing showed: DSHOW=4.6 FPS vs MSMF=59.9 FPS on Pibox VC9811T
            print(f"Opening device {self.device_index} with MSMF backend...")
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
            if self.cap.isOpened():
                backend_name = "MSMF"
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Fallback to DirectShow only if MSMF fails
            if not self.cap.isOpened():
                print(f"MSMF failed, trying DirectShow...")
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    backend_name = "DSHOW"
                    # Try MJPEG for better FPS with DirectShow
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.cap.isOpened():
                print(f"Failed to open device {self.device_index} ({self.device_name})")
                return False

            print(f"Opened device {self.device_index} ({self.device_name}) via {backend_name}")

            # Minimize buffer for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Capture configured: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            # Quick FPS test
            print("  Testing capture speed...")
            import time as t
            start = t.time()
            test_frames = 0
            for _ in range(30):
                ret, _ = self.cap.read()
                if ret:
                    test_frames += 1
            test_elapsed = t.time() - start
            test_fps = test_frames / test_elapsed if test_elapsed > 0 else 0
            print(f"  Test capture: {test_fps:.1f} FPS")

            if test_fps < 15:
                print(f"  WARNING: Low FPS detected!")
                if backend_name == "DSHOW":
                    print(f"  Try using MSMF backend for better performance")

            return True

        except Exception as e:
            print(f"Failed to open capture device: {e}")
            import traceback
            traceback.print_exc()
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
        """Background capture loop optimized for high FPS."""
        # Flush stale frames from buffer
        for _ in range(5):
            self.cap.grab()

        while self._running:
            # read() is actually fine for MSMF - internally optimized
            ret, frame = self.cap.read()

            if not ret or frame is None:
                continue

            # Minimal processing - just convert color space
            # OpenCV captures in BGR, we need RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Only resize if actually needed
            h, w = frame_rgb.shape[:2]
            if w != self.width or h != self.height:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST
                )

            # Update last frame (for get_latest_frame)
            with self._lock:
                self._last_frame = frame_rgb

            # Queue management - drop old frames to prevent lag buildup
            try:
                self._frame_queue.put_nowait(frame_rgb)
            except queue.Full:
                # Clear queue and add fresh frame
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(frame_rgb)
                except queue.Full:
                    pass
                self.frames_dropped += 1

            self.frames_captured += 1
            self._fps_frame_count += 1

            # Update FPS counter
            now = time.time()
            elapsed = now - self._fps_start_time
            if elapsed >= 1.0:
                self.actual_fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._fps_start_time = now

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


def _get_dshow_device_names() -> List[str]:
    """
    Get DirectShow device names using ffmpeg or PowerShell.

    Returns:
        List of device names in order
    """
    import subprocess
    import re

    # Try PowerShell first (faster, no ffmpeg dependency)
    try:
        result = subprocess.run(
            ['powershell', '-Command',
             'Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq "Camera" -or $_.PNPClass -eq "Image" -or ($_.Name -like "*capture*" -or $_.Name -like "*video*" -or $_.Name -like "*USB3*" -or $_.Name -like "*HDMI*") } | Select-Object -ExpandProperty Name'],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000  # CREATE_NO_WINDOW
        )
        if result.returncode == 0 and result.stdout.strip():
            names = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
            if names:
                return names
    except Exception:
        pass

    # Fallback to ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=0x08000000  # CREATE_NO_WINDOW
        )

        output = result.stderr
        video_devices = []

        for line in output.split('\n'):
            match = re.search(r'\[dshow.*?\]\s+"([^"]+)"\s+\(video\)', line)
            if match:
                name = match.group(1)
                video_devices.append(name)

        return video_devices

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return []


def list_capture_devices() -> List[dict]:
    """
    List available capture devices, prioritizing 1080p capable devices.

    Returns:
        List of device info dictionaries sorted by resolution (highest first)
    """
    if not OPENCV_AVAILABLE:
        return []

    devices = []

    # Scan devices using MSMF (faster and better FPS)
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            # Try to set 1080p to detect capable devices
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Mark 1080p devices as likely capture cards
            is_capture_card = (w >= 1920 and h >= 1080)
            name = f"Device {i}" + (" (1080p)" if is_capture_card else "")

            devices.append({
                'index': i,
                'name': name,
                'width': w,
                'height': h,
                'fps': fps,
                'backend': 'MSMF',
                'description': f"{name} - {w}x{h}",
                'is_capture_card': is_capture_card
            })

    # Sort by resolution (highest first) so capture cards appear at top
    devices.sort(key=lambda x: (x['is_capture_card'], x['width'] * x['height']), reverse=True)

    return devices


def open_capture_device(device_info: dict, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> Optional[cv2.VideoCapture]:
    """
    Open a capture device using the best available method.

    Args:
        device_info: Device info dictionary from list_capture_devices()
        width: Desired capture width
        height: Desired capture height

    Returns:
        Opened VideoCapture or None
    """
    name = device_info.get('name', '')
    index = device_info.get('index', 0)

    # Try opening by name with DSHOW first (most reliable for USB capture)
    cap = cv2.VideoCapture(f'video={name}', cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    # Try MSMF by index
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    # Try DSHOW by index
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    # Try default backend
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    return None


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


class VideoFileCapture:
    """
    Captures frames from a recorded video file.

    Provides the same interface as FrameCapture for compatibility.
    """

    def __init__(
        self,
        video_path: str,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        playback_fps: int = 0  # 0 = use video's native FPS
    ):
        """
        Initialize video file capture.

        Args:
            video_path: Path to the video file
            width: Target width (frames will be resized if different)
            height: Target height (frames will be resized if different)
            playback_fps: Playback speed (0 = native FPS)
        """
        if not OPENCV_AVAILABLE:
            raise ImportError(
                "OpenCV not available. Install with: pip install opencv-python"
            )

        self.video_path = video_path
        self.width = width
        self.height = height
        self.playback_fps = playback_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Video info
        self.total_frames = 0
        self.video_fps = 0.0
        self.video_width = 0
        self.video_height = 0
        self.current_frame = 0

        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0
        self.actual_fps = 0.0
        self._fps_start_time = 0.0
        self._fps_frame_count = 0

    def open(self) -> bool:
        """
        Open the video file.

        Returns:
            True if successful
        """
        try:
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                print(f"Failed to open video file: {self.video_path}")
                return False

            # Get video info
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.playback_fps <= 0:
                self.playback_fps = self.video_fps

            print(f"Video opened: {self.video_path}")
            print(f"  Resolution: {self.video_width}x{self.video_height}")
            print(f"  FPS: {self.video_fps}, Total frames: {self.total_frames}")
            print(f"  Playback FPS: {self.playback_fps}")

            return True

        except Exception as e:
            print(f"Failed to open video file: {e}")
            return False

    def close(self):
        """Close the video file."""
        self.stop_capture()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start_capture(self):
        """Start reading frames in background thread."""
        if self._running:
            return

        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError("Failed to open video file")

        self._running = True
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self.current_frame = 0

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop_capture(self):
        """Stop reading frames."""
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
        """Background frame reading loop."""
        frame_delay = 1.0 / self.playback_fps if self.playback_fps > 0 else 0

        while self._running:
            frame_start = time.time()

            ret, frame = self.cap.read()

            if not ret or frame is None:
                # End of video - loop or stop
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                continue

            self.current_frame += 1

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

            # Control playback speed
            frame_time = time.time() - frame_start
            if frame_delay > frame_time:
                time.sleep(frame_delay - frame_time)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next frame from video.

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

    def get_progress(self) -> Tuple[int, int]:
        """Get current position in video."""
        return self.current_frame, self.total_frames

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get number of frames in queue."""
        return self._frame_queue.qsize()
