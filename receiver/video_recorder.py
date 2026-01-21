"""
Visual Data Diode - Video Recorder

Records the incoming video stream to a file for later decoding.
This approach allows capture at maximum FPS without processing overhead.
"""

import cv2
import numpy as np
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class RecordingStats:
    """Statistics for video recording."""
    frames_recorded: int = 0
    frames_dropped: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    file_size_bytes: int = 0

    @property
    def duration(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time if self.start_time > 0 else 0

    @property
    def actual_fps(self) -> float:
        if self.duration > 0:
            return self.frames_recorded / self.duration
        return 0.0


class VideoRecorder:
    """
    Records video stream to file for later decoding.

    Uses a separate thread for writing to avoid frame drops.
    Supports multiple codecs with preference for lossless/high-quality.
    """

    # Codec options in order of preference
    # Note: FFV1 can fail with certain dimensions, prefer HFYU
    CODEC_OPTIONS = [
        ('HFYU', 'avi', 'Lossless HuffYUV'),     # Lossless, fast
        ('MJPG', 'avi', 'Motion JPEG'),          # Good quality, smaller
        ('XVID', 'avi', 'XVID'),                  # Fallback
    ]

    def __init__(
        self,
        output_path: str,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        codec: str = None,
        on_frame_recorded: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize video recorder.

        Args:
            output_path: Path to save video (extension will be adjusted based on codec)
            width: Frame width
            height: Frame height
            fps: Target FPS for recording
            codec: Preferred codec (None = auto-select best available)
            on_frame_recorded: Callback when frame is written
        """
        self.base_output_path = Path(output_path).with_suffix('')
        self.width = width
        self.height = height
        self.fps = fps
        self.preferred_codec = codec
        self.on_frame_recorded = on_frame_recorded

        self.writer: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[Path] = None
        self.codec_name: str = ""

        self._running = False
        self._write_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=300)  # Buffer ~5 seconds at 60fps
        self._lock = threading.Lock()

        self.stats = RecordingStats()

    def _select_codec(self) -> tuple:
        """Select best available codec."""
        if self.preferred_codec:
            for fourcc, ext, name in self.CODEC_OPTIONS:
                if fourcc == self.preferred_codec:
                    return fourcc, ext, name

        # Test each codec
        for fourcc, ext, name in self.CODEC_OPTIONS:
            test_path = str(self.base_output_path) + f"_test.{ext}"
            try:
                writer = cv2.VideoWriter(
                    test_path,
                    cv2.VideoWriter_fourcc(*fourcc),
                    self.fps,
                    (self.width, self.height)
                )
                if writer.isOpened():
                    writer.release()
                    Path(test_path).unlink(missing_ok=True)
                    return fourcc, ext, name
            except Exception:
                pass
            finally:
                Path(test_path).unlink(missing_ok=True)

        # Fallback to MJPG
        return 'MJPG', 'avi', 'Motion JPEG (fallback)'

    def start(self) -> bool:
        """
        Start recording.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        # Select codec
        fourcc, ext, name = self._select_codec()
        self.codec_name = name
        self.output_path = self.base_output_path.with_suffix(f'.{ext}')

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            print(f"Failed to open video writer with {name}")
            return False

        print(f"Recording to: {self.output_path}")
        print(f"Codec: {name}, Resolution: {self.width}x{self.height}, FPS: {self.fps}")

        # Reset stats
        self.stats = RecordingStats()
        self.stats.start_time = time.time()

        # Start write thread
        self._running = True
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thread.start()

        return True

    def stop(self) -> RecordingStats:
        """
        Stop recording and finalize file.

        Returns:
            Recording statistics
        """
        self._running = False

        # Wait for write thread to finish
        if self._write_thread:
            self._write_thread.join(timeout=5.0)
            self._write_thread = None

        # Close writer
        if self.writer:
            self.writer.release()
            self.writer = None

        self.stats.end_time = time.time()

        # Get file size
        if self.output_path and self.output_path.exists():
            self.stats.file_size_bytes = self.output_path.stat().st_size

        print(f"Recording complete: {self.stats.frames_recorded} frames, "
              f"{self.stats.duration:.1f}s, {self.stats.actual_fps:.1f} FPS")

        if self.stats.frames_dropped > 0:
            print(f"Warning: {self.stats.frames_dropped} frames dropped")

        return self.stats

    def add_frame(self, frame: np.ndarray, is_bgr: bool = True):
        """
        Add a frame to be recorded.

        Args:
            frame: Frame to record
            is_bgr: True if frame is BGR (from OpenCV), False if RGB
        """
        if not self._running:
            return

        # Convert RGB to BGR if needed (OpenCV VideoWriter expects BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3 and not is_bgr:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Resize if needed
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        # Add to queue
        try:
            self._frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            self.stats.frames_dropped += 1

    def _write_loop(self):
        """Background thread for writing frames."""
        while self._running or not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get(timeout=0.1)

                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                    self.stats.frames_recorded += 1

                    if self.on_frame_recorded:
                        try:
                            self.on_frame_recorded(self.stats.frames_recorded)
                        except Exception:
                            pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Write error: {e}")

    @property
    def is_recording(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._frame_queue.qsize()


class EndFrameDetector:
    """
    Detects end-of-transmission frames.

    Looks for a specific pattern (e.g., all black frames or
    a specific color sequence) to determine when recording should stop.
    """

    def __init__(
        self,
        consecutive_frames: int = 30,  # ~1 second at 30fps
        black_threshold: float = 10.0   # Mean pixel value threshold
    ):
        """
        Initialize detector.

        Args:
            consecutive_frames: Number of consecutive end frames needed
            black_threshold: Maximum mean pixel value to consider "black"
        """
        self.consecutive_frames = consecutive_frames
        self.black_threshold = black_threshold

        self._black_count = 0
        self._detected = False

    def check_frame(self, frame: np.ndarray) -> bool:
        """
        Check if this frame is an end frame.

        Args:
            frame: Frame to check

        Returns:
            True if end-of-transmission detected
        """
        if self._detected:
            return True

        # Check if frame is mostly black
        mean_value = np.mean(frame)

        if mean_value < self.black_threshold:
            self._black_count += 1

            if self._black_count >= self.consecutive_frames:
                self._detected = True
                return True
        else:
            self._black_count = 0

        return False

    def reset(self):
        """Reset detector state."""
        self._black_count = 0
        self._detected = False

    @property
    def is_detected(self) -> bool:
        return self._detected


class RecordingSession:
    """
    Complete recording session that combines capture, recording, and end detection.
    """

    def __init__(
        self,
        capture_device: int,
        output_path: str,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        auto_stop: bool = True,
        on_progress: Optional[Callable[[int, float], None]] = None,
        on_complete: Optional[Callable[[str, RecordingStats], None]] = None
    ):
        """
        Initialize recording session.

        Args:
            capture_device: OpenCV capture device index
            output_path: Path to save video
            width: Capture/record width
            height: Capture/record height
            fps: Target FPS
            auto_stop: Automatically stop when end frames detected
            on_progress: Callback (frames, elapsed_time)
            on_complete: Callback (output_path, stats)
        """
        self.capture_device = capture_device
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_stop = auto_stop
        self.on_progress = on_progress
        self.on_complete = on_complete

        self.recorder = VideoRecorder(
            output_path=output_path,
            width=width,
            height=height,
            fps=fps
        )

        self.end_detector = EndFrameDetector()

        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start recording session."""
        if self._running:
            return True

        # Open capture device with MSMF for best FPS
        print(f"Opening capture device {self.capture_device} with MSMF...")
        self.cap = cv2.VideoCapture(self.capture_device, cv2.CAP_MSMF)

        if not self.cap.isOpened():
            print("Failed to open capture device")
            return False

        # Configure capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Capture configured: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")

        # Flush stale frames
        for _ in range(10):
            self.cap.grab()

        # Start recorder
        if not self.recorder.start():
            self.cap.release()
            return False

        # Start capture thread
        self._running = True
        self.end_detector.reset()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def stop(self) -> tuple:
        """
        Stop recording session.

        Returns:
            (output_path, stats)
        """
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        stats = self.recorder.stop()
        output_path = str(self.recorder.output_path)

        if self.on_complete:
            self.on_complete(output_path, stats)

        return output_path, stats

    def _capture_loop(self):
        """Capture frames and add to recorder."""
        last_progress = 0

        while self._running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                continue

            # Add to recorder (frame is BGR from OpenCV)
            self.recorder.add_frame(frame)

            # Check for end frames
            if self.auto_stop and self.end_detector.check_frame(frame):
                print("End-of-transmission detected!")
                self._running = False
                break

            # Progress callback every second
            elapsed = time.time() - self.recorder.stats.start_time
            if int(elapsed) > last_progress:
                last_progress = int(elapsed)
                if self.on_progress:
                    self.on_progress(self.recorder.stats.frames_recorded, elapsed)

    @property
    def is_running(self) -> bool:
        return self._running


if __name__ == "__main__":
    # Test recording
    import sys

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"Recording from device {device} for {duration} seconds...")

    session = RecordingSession(
        capture_device=device,
        output_path="test_recording",
        auto_stop=False
    )

    if session.start():
        time.sleep(duration)
        output_path, stats = session.stop()

        print(f"\nRecording saved to: {output_path}")
        print(f"Stats: {stats.frames_recorded} frames, {stats.actual_fps:.1f} FPS")
        print(f"File size: {stats.file_size_bytes / 1024 / 1024:.1f} MB")
    else:
        print("Failed to start recording")
