"""
Visual Data Diode - Sender Main Application

Main entry point for the sender application.
Coordinates UI, chunking, encoding, and rendering.
"""

import threading
import time
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    EncodingProfile, DEFAULT_PROFILE, DEFAULT_FPS,
    SYNC_FRAME_COUNT, END_FRAME_COUNT, DEFAULT_REPEAT_COUNT,
    derive_key
)
from .chunker import FileChunker
from .encoder import FrameEncoder
from .renderer import FrameRenderer
from .timing import FrameTimer, format_duration
from .ui import SenderUI


class SenderApplication:
    """
    Main sender application.

    Coordinates all sender components.
    """

    def __init__(self):
        self.ui = SenderUI()
        self.chunker: Optional[FileChunker] = None
        self.encoder: Optional[FrameEncoder] = None
        self.renderer: Optional[FrameRenderer] = None
        self.timer: Optional[FrameTimer] = None

        # State
        self._running = False
        self._paused = False
        self._transmission_thread: Optional[threading.Thread] = None

        # Progress tracking
        self.current_block = 0
        self.total_blocks = 0
        self.bytes_sent = 0
        self.start_time = 0.0

        # Settings
        self.profile = DEFAULT_PROFILE
        self.fps = DEFAULT_FPS
        self.repeat_count = DEFAULT_REPEAT_COUNT
        self.file_path: Optional[str] = None

        # Connect UI callbacks
        self._setup_ui_callbacks()

    def _setup_ui_callbacks(self):
        """Connect UI callbacks to handler methods."""
        self.ui.on_start = self._on_start
        self.ui.on_pause = self._on_pause
        self.ui.on_resume = self._on_resume
        self.ui.on_stop = self._on_stop
        self.ui.on_restart = self._on_restart

    def _on_start(self, settings: dict):
        """Handle start transmission."""
        self.file_path = settings['file_path']
        self.profile = settings['profile']
        self.fps = settings['fps']
        self.repeat_count = settings['repeat_count']

        # Setup encryption if enabled
        encryption_key = None
        if settings['encrypt'] and settings['password']:
            key, salt = derive_key(settings['password'])
            encryption_key = key

        # Initialize components
        try:
            self.chunker = FileChunker(
                profile=self.profile,
                encrypt=settings['encrypt'],
                encryption_key=encryption_key
            )

            # Prepare file
            total_blocks, file_size, file_hash = self.chunker.prepare_file(self.file_path)
            self.total_blocks = total_blocks

            self.encoder = FrameEncoder(profile=self.profile)
            self.renderer = FrameRenderer(
                fullscreen=True,
                display_index=settings['display_index']
            )
            self.timer = FrameTimer(fps=self.fps)

            # Start transmission in background thread
            self._running = True
            self._paused = False
            self.current_block = 0
            self.bytes_sent = 0
            self.start_time = time.time()

            self._transmission_thread = threading.Thread(
                target=self._transmission_loop,
                daemon=True
            )
            self._transmission_thread.start()

            # Start progress update timer
            self._schedule_progress_update()

        except Exception as e:
            self.ui.show_complete(False, f"Failed to start: {str(e)}")

    def _on_pause(self):
        """Handle pause."""
        self._paused = True
        if self.timer:
            self.timer.pause()

    def _on_resume(self):
        """Handle resume."""
        self._paused = False
        if self.timer:
            self.timer.resume()

    def _on_stop(self):
        """Handle stop."""
        self._running = False
        if self.renderer:
            self.renderer.shutdown()

    def _on_restart(self):
        """Handle restart."""
        # Stop current transmission
        self._running = False
        time.sleep(0.2)  # Allow thread to stop

        # Restart from beginning
        self.current_block = 0
        self.bytes_sent = 0
        self.start_time = time.time()
        self._running = True
        self._paused = False

        if self.timer:
            self.timer.reset()

        self._transmission_thread = threading.Thread(
            target=self._transmission_loop,
            daemon=True
        )
        self._transmission_thread.start()

    def _transmission_loop(self):
        """Main transmission loop (runs in background thread)."""
        try:
            # Initialize renderer
            if not self.renderer.initialize():
                self._signal_error("Failed to initialize display")
                return

            # Send sync frames first
            sync_frame = self.encoder.encode_sync_only()
            sync_frame_full = self.encoder.grid_to_frame(sync_frame)

            self.timer.start()

            for _ in range(SYNC_FRAME_COUNT):
                if not self._running:
                    return

                while self._paused:
                    time.sleep(0.1)
                    if not self._running:
                        return

                self.renderer.render_frame(sync_frame_full)
                self.timer.wait_for_next_frame()

            # Transmit blocks with repeats
            for repeat in range(self.repeat_count):
                for block in self.chunker.generate_blocks():
                    if not self._running:
                        return

                    while self._paused:
                        time.sleep(0.1)
                        if not self._running:
                            return

                    # Encode and render block
                    frame = self.encoder.encode_to_frame(block)
                    self.renderer.render_frame(frame)
                    self.timer.wait_for_next_frame()

                    # Update progress (only on first pass)
                    if repeat == 0:
                        self.current_block = block.header.block_index + 1
                        self.bytes_sent += block.header.payload_size

            # Send end frames
            end_frame = self.encoder.encode_end_frame()
            end_frame_full = self.encoder.grid_to_frame(end_frame)

            for _ in range(END_FRAME_COUNT):
                if not self._running:
                    return

                self.renderer.render_frame(end_frame_full)
                self.timer.wait_for_next_frame()

            # Complete
            self._signal_complete()

        except Exception as e:
            self._signal_error(str(e))

        finally:
            if self.renderer:
                self.renderer.shutdown()

    def _schedule_progress_update(self):
        """Schedule periodic progress updates."""
        if not self._running:
            return

        elapsed = time.time() - self.start_time
        actual_fps = self.timer.stats.actual_fps if self.timer else self.fps

        try:
            self.ui.update_progress(
                self.current_block,
                self.total_blocks,
                self.bytes_sent,
                elapsed,
                actual_fps
            )
        except Exception:
            pass  # UI might be closing

        if self._running:
            self.ui.root.after(100, self._schedule_progress_update)

    def _signal_complete(self):
        """Signal transmission complete."""
        self._running = False
        elapsed = time.time() - self.start_time

        def show():
            self.ui.show_complete(
                True,
                f"Transmission complete!\n\n"
                f"Blocks sent: {self.total_blocks:,}\n"
                f"Bytes sent: {self.bytes_sent:,}\n"
                f"Time: {format_duration(elapsed)}\n"
                f"Average rate: {self.bytes_sent / elapsed / 1024:.1f} KB/s"
            )

        self.ui.root.after(100, show)

    def _signal_error(self, message: str):
        """Signal transmission error."""
        self._running = False

        def show():
            self.ui.show_complete(False, message)

        self.ui.root.after(100, show)

    def run(self):
        """Run the application."""
        self.ui.run()


def main():
    """Entry point for sender application."""
    app = SenderApplication()
    app.run()


if __name__ == "__main__":
    main()
