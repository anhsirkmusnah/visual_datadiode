"""
Visual Data Diode - Frame Renderer

Renders frames to HDMI output using pygame for fullscreen display.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from pathlib import Path
import threading
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import FRAME_WIDTH, FRAME_HEIGHT

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class FrameRenderer:
    """
    Renders frames to a display using pygame.

    Supports both windowed and fullscreen modes.
    Can target specific displays for HDMI output.
    """

    def __init__(
        self,
        fullscreen: bool = True,
        display_index: int = -1,
        window_title: str = "Visual Data Diode - Sender"
    ):
        """
        Initialize renderer.

        Args:
            fullscreen: Use fullscreen mode
            display_index: Display to use (-1 for default, 0 for primary, 1 for secondary)
            window_title: Window title (for windowed mode)
        """
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame not available. Install with: pip install pygame"
            )

        self.fullscreen = fullscreen
        self.display_index = display_index
        self.window_title = window_title
        self.screen: Optional[pygame.Surface] = None
        self._initialized = False
        self._running = False
        self._lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None

    def initialize(self) -> bool:
        """
        Initialize pygame and create display.

        Returns:
            True if successful
        """
        if self._initialized:
            return True

        try:
            pygame.init()
            pygame.display.set_caption(self.window_title)

            # Get display info
            num_displays = pygame.display.get_num_displays()

            if self.display_index >= num_displays:
                print(f"Warning: Display {self.display_index} not found, using default")
                self.display_index = -1

            # Set up display
            if self.fullscreen:
                # Position window on target display before going fullscreen
                if self.display_index >= 0 and num_displays > 1:
                    # Get display bounds
                    try:
                        display_bounds = pygame.display.get_desktop_sizes()
                        if self.display_index < len(display_bounds):
                            # Calculate offset
                            x_offset = sum(
                                display_bounds[i][0]
                                for i in range(self.display_index)
                            )
                            import os
                            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},0"
                    except Exception:
                        pass

                flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
                self.screen = pygame.display.set_mode(
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    flags
                )
            else:
                self.screen = pygame.display.set_mode(
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    pygame.HWSURFACE | pygame.DOUBLEBUF
                )

            # Hide cursor in fullscreen
            if self.fullscreen:
                pygame.mouse.set_visible(False)

            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize renderer: {e}")
            return False

    def shutdown(self):
        """Shutdown pygame and cleanup."""
        self._running = False
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self.screen = None

    def render_frame(self, frame: np.ndarray):
        """
        Render a frame to the display.

        Args:
            frame: RGB frame as numpy array (FRAME_HEIGHT, FRAME_WIDTH, 3)
        """
        if not self._initialized:
            self.initialize()

        if self.screen is None:
            return

        with self._lock:
            self._current_frame = frame

        # Convert numpy array to pygame surface
        # pygame expects (width, height) format for surfarray
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )

        # Blit to screen
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Process events to prevent "not responding"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False

    def render_black(self):
        """Render a black frame."""
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.render_frame(frame)

    def is_running(self) -> bool:
        """Check if renderer is still running (not quit)."""
        return self._initialized and self._running

    def get_display_info(self) -> dict:
        """Get information about available displays."""
        if not pygame.get_init():
            pygame.init()

        info = {
            'num_displays': pygame.display.get_num_displays(),
            'displays': []
        }

        try:
            sizes = pygame.display.get_desktop_sizes()
            for i, size in enumerate(sizes):
                info['displays'].append({
                    'index': i,
                    'width': size[0],
                    'height': size[1]
                })
        except Exception:
            pass

        return info


class RenderThread:
    """
    Thread-safe frame rendering with queue.

    Allows main thread to queue frames while renderer displays them.
    """

    def __init__(self, renderer: FrameRenderer, fps: int = 20):
        """
        Initialize render thread.

        Args:
            renderer: FrameRenderer instance
            fps: Target frames per second
        """
        self.renderer = renderer
        self.fps = fps
        self.frame_time = 1.0 / fps

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_queue = []
        self._queue_lock = threading.Lock()
        self._frame_event = threading.Event()

        # Stats
        self.frames_rendered = 0
        self.actual_fps = 0.0

    def start(self):
        """Start the render thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the render thread."""
        self._running = False
        self._frame_event.set()  # Wake up thread
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def queue_frame(self, frame: np.ndarray):
        """
        Queue a frame for rendering.

        Args:
            frame: RGB frame array
        """
        with self._queue_lock:
            self._frame_queue.append(frame.copy())
            self._frame_event.set()

    def get_queue_size(self) -> int:
        """Get number of frames waiting in queue."""
        with self._queue_lock:
            return len(self._frame_queue)

    def _render_loop(self):
        """Main render loop."""
        self.renderer.initialize()
        self.renderer._running = True

        last_time = time.time()
        frame_count = 0

        while self._running and self.renderer._running:
            # Wait for frame or timeout
            self._frame_event.wait(timeout=self.frame_time)

            # Get next frame
            frame = None
            with self._queue_lock:
                if self._frame_queue:
                    frame = self._frame_queue.pop(0)
                    if not self._frame_queue:
                        self._frame_event.clear()

            if frame is not None:
                self.renderer.render_frame(frame)
                self.frames_rendered += 1
                frame_count += 1

            # Calculate FPS every second
            now = time.time()
            elapsed = now - last_time
            if elapsed >= 1.0:
                self.actual_fps = frame_count / elapsed
                frame_count = 0
                last_time = now

        self.renderer.shutdown()


def check_pygame_available() -> bool:
    """Check if pygame is available."""
    return PYGAME_AVAILABLE


def list_displays() -> list:
    """List available displays."""
    if not PYGAME_AVAILABLE:
        return []

    pygame.init()
    try:
        displays = []
        for i, size in enumerate(pygame.display.get_desktop_sizes()):
            displays.append({
                'index': i,
                'width': size[0],
                'height': size[1],
                'description': f"Display {i}: {size[0]}x{size[1]}"
            })
        return displays
    except Exception:
        return [{'index': 0, 'width': 1920, 'height': 1080, 'description': 'Default'}]
    finally:
        pygame.quit()
