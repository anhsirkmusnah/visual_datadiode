"""
Visual Data Diode - Frame Timing

Provides precise frame timing for consistent transmission.
"""

import time
from typing import Callable, Optional
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import DEFAULT_FPS


@dataclass
class TimingStats:
    """Statistics for frame timing."""
    frames_sent: int = 0
    total_time: float = 0.0
    min_frame_time: float = float('inf')
    max_frame_time: float = 0.0
    avg_frame_time: float = 0.0
    late_frames: int = 0  # Frames that took longer than target

    @property
    def actual_fps(self) -> float:
        if self.total_time > 0:
            return self.frames_sent / self.total_time
        return 0.0


class FrameTimer:
    """
    Provides consistent frame timing using busy-wait for precision.

    Uses time.perf_counter() for high-resolution timing.
    """

    def __init__(self, fps: int = DEFAULT_FPS):
        """
        Initialize timer.

        Args:
            fps: Target frames per second
        """
        self.fps = fps
        self.frame_time = 1.0 / fps
        self._next_frame_time = 0.0
        self._started = False
        self._paused = False
        self._pause_time = 0.0

        # Statistics
        self.stats = TimingStats()
        self._last_frame_start = 0.0

    def start(self):
        """Start timing from now."""
        self._next_frame_time = time.perf_counter() + self.frame_time
        self._started = True
        self._paused = False
        self._last_frame_start = time.perf_counter()
        self.stats = TimingStats()

    def wait_for_next_frame(self) -> float:
        """
        Wait until it's time for the next frame.

        Uses hybrid approach: sleep for most of wait, busy-wait for precision.

        Returns:
            Actual time waited in seconds
        """
        if not self._started:
            self.start()

        now = time.perf_counter()
        wait_time = self._next_frame_time - now

        if wait_time > 0:
            # Sleep for most of the wait (leave 2ms for busy-wait)
            sleep_time = wait_time - 0.002
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Busy-wait for the remainder
            while time.perf_counter() < self._next_frame_time:
                pass

            actual_wait = wait_time
        else:
            # We're late
            self.stats.late_frames += 1
            actual_wait = 0

        # Update for next frame
        frame_end = time.perf_counter()
        frame_duration = frame_end - self._last_frame_start

        # Update statistics
        self.stats.frames_sent += 1
        self.stats.total_time += frame_duration
        self.stats.min_frame_time = min(self.stats.min_frame_time, frame_duration)
        self.stats.max_frame_time = max(self.stats.max_frame_time, frame_duration)
        self.stats.avg_frame_time = self.stats.total_time / self.stats.frames_sent

        self._last_frame_start = frame_end
        self._next_frame_time = frame_end + self.frame_time

        return actual_wait

    def pause(self):
        """Pause timing (for UI interaction, etc.)."""
        if not self._paused:
            self._paused = True
            self._pause_time = time.perf_counter()

    def resume(self):
        """Resume timing after pause."""
        if self._paused:
            pause_duration = time.perf_counter() - self._pause_time
            self._next_frame_time += pause_duration
            self._paused = False

    def reset(self):
        """Reset timer and statistics."""
        self._started = False
        self._paused = False
        self.stats = TimingStats()

    def set_fps(self, fps: int):
        """Change target FPS."""
        self.fps = fps
        self.frame_time = 1.0 / fps

    @property
    def is_running(self) -> bool:
        """Check if timer is running (started and not paused)."""
        return self._started and not self._paused


class TransmissionController:
    """
    High-level transmission control with timing.

    Manages the frame transmission loop with proper timing,
    pause/resume, and progress tracking.
    """

    def __init__(
        self,
        fps: int = DEFAULT_FPS,
        on_frame: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Initialize controller.

        Args:
            fps: Target frames per second
            on_frame: Callback for each frame (receives frame index)
            on_complete: Callback when transmission completes
            on_error: Callback on error
        """
        self.timer = FrameTimer(fps)
        self.on_frame = on_frame
        self.on_complete = on_complete
        self.on_error = on_error

        self._running = False
        self._paused = False
        self._current_frame = 0
        self._total_frames = 0

    def start(self, total_frames: int, start_frame: int = 0):
        """
        Start transmission.

        Args:
            total_frames: Total frames to transmit
            start_frame: Starting frame index (for resume)
        """
        self._total_frames = total_frames
        self._current_frame = start_frame
        self._running = True
        self._paused = False
        self.timer.start()

    def stop(self):
        """Stop transmission."""
        self._running = False

    def pause(self):
        """Pause transmission."""
        if self._running and not self._paused:
            self._paused = True
            self.timer.pause()

    def resume(self):
        """Resume transmission."""
        if self._running and self._paused:
            self._paused = False
            self.timer.resume()

    def step(self) -> bool:
        """
        Process one frame.

        Returns:
            True if should continue, False if complete or stopped
        """
        if not self._running or self._paused:
            return self._running

        if self._current_frame >= self._total_frames:
            self._running = False
            if self.on_complete:
                self.on_complete()
            return False

        try:
            # Wait for proper timing
            self.timer.wait_for_next_frame()

            # Call frame callback
            if self.on_frame:
                self.on_frame(self._current_frame)

            self._current_frame += 1
            return True

        except Exception as e:
            self._running = False
            if self.on_error:
                self.on_error(e)
            return False

    def run(self):
        """Run transmission loop until complete."""
        while self.step():
            pass

    @property
    def progress(self) -> float:
        """Get progress as fraction (0.0 to 1.0)."""
        if self._total_frames == 0:
            return 0.0
        return self._current_frame / self._total_frames

    @property
    def frames_remaining(self) -> int:
        """Get number of frames remaining."""
        return max(0, self._total_frames - self._current_frame)

    @property
    def estimated_time_remaining(self) -> float:
        """Get estimated time remaining in seconds."""
        return self.frames_remaining * self.timer.frame_time

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._running and self._paused

    @property
    def is_complete(self) -> bool:
        return not self._running and self._current_frame >= self._total_frames


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
