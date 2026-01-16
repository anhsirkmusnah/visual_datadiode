"""
Visual Data Diode - Receiver Main Application

Main entry point for the receiver application.
Coordinates capture, sync, decoding, and file assembly.
"""

import threading
import time
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import EncodingProfile, DEFAULT_PROFILE
from .capture import FrameCapture
from .sync import FrameSync
from .decoder import FrameDecoder, StreamDecoder
from .assembler import BlockAssembler
from .ui import ReceiverUI


class ReceiverApplication:
    """
    Main receiver application.

    Coordinates all receiver components.
    """

    def __init__(self):
        self.ui = ReceiverUI()
        self.capture: Optional[FrameCapture] = None
        self.sync: Optional[FrameSync] = None
        self.decoder: Optional[StreamDecoder] = None
        self.assembler: Optional[BlockAssembler] = None

        # State
        self._running = False
        self._receive_thread: Optional[threading.Thread] = None

        # Statistics
        self.start_time = 0.0
        self.frames_processed = 0
        self.blocks_received = 0
        self.crc_errors = 0
        self.fec_corrections = 0

        # Settings
        self.profile = DEFAULT_PROFILE
        self.output_dir = ""
        self.password: Optional[str] = None

        # Connect UI callbacks
        self._setup_ui_callbacks()

    def _setup_ui_callbacks(self):
        """Connect UI callbacks."""
        self.ui.on_start = self._on_start
        self.ui.on_stop = self._on_stop
        self.ui.on_save = self._on_save

    def _on_start(self, settings: dict):
        """Handle start reception."""
        self.profile = settings['profile']
        self.output_dir = settings['output_dir']
        self.password = settings['password']

        # Initialize components
        try:
            self.capture = FrameCapture(
                device_index=settings['device_index'],
                use_mjpeg=False  # Prefer YUV422 for MS2130
            )

            self.sync = FrameSync(profile=self.profile)
            self.decoder = StreamDecoder(profile=self.profile)
            self.assembler = BlockAssembler(
                output_dir=self.output_dir,
                password=self.password
            )

            # Open capture device
            if not self.capture.open():
                self.ui.show_complete(False, message="Failed to open capture device")
                return

            # Start capture
            self.capture.start_capture()

            # Reset statistics
            self.start_time = time.time()
            self.frames_processed = 0
            self.blocks_received = 0
            self.crc_errors = 0
            self.fec_corrections = 0

            # Start receive thread
            self._running = True
            self._receive_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True
            )
            self._receive_thread.start()

            # Start progress update timer
            self._schedule_progress_update()

        except Exception as e:
            self.ui.show_complete(False, message=f"Failed to start: {str(e)}")

    def _on_stop(self):
        """Handle stop reception."""
        self._running = False

        if self.capture:
            self.capture.stop_capture()
            self.capture.close()

    def _on_save(self):
        """Handle save file."""
        if self.assembler:
            status = self.assembler.assemble()

            if status.complete:
                self.ui.show_complete(
                    success=True,
                    filename=status.filename or "received_file",
                    file_size=status.file_size,
                    hash_valid=status.file_hash_valid,
                    message=status.message
                )
            else:
                self.ui.show_complete(
                    success=False,
                    message=status.message
                )

    def _receive_loop(self):
        """Main receive loop."""
        show_preview = self.ui.show_preview_var.get()

        while self._running:
            # Get frame from capture
            frame = self.capture.get_frame(timeout=0.5)

            if frame is None:
                continue

            self.frames_processed += 1

            # Update preview (throttled)
            if show_preview and self.frames_processed % 5 == 0:
                try:
                    self.ui.root.after(0, lambda f=frame: self.ui.update_preview(f))
                except Exception:
                    pass

            # Detect sync
            sync_result = self.sync.detect_sync(frame)

            # Update sync status
            try:
                self.ui.root.after(
                    0,
                    lambda s=sync_result: self.ui.update_sync_status(
                        s.is_synced, s.confidence
                    )
                )
            except Exception:
                pass

            if not sync_result.is_synced:
                continue

            # Extract grid
            grid = self.sync.extract_grid(frame, sync_result)

            if grid is None:
                continue

            # Decode block
            result = self.decoder.process_grid(grid)

            if result is None:
                # Duplicate block
                continue

            if result.success and result.block:
                # Add to assembler
                success, msg = self.assembler.add_block(result.block)

                if success:
                    self.blocks_received += 1

                    if result.fec_corrected > 0:
                        self.fec_corrections += result.fec_corrected

                # Check if complete
                if self.assembler.is_complete():
                    self._signal_complete()
                    break

            elif not result.crc_valid:
                self.crc_errors += 1

        # Cleanup
        if self.capture:
            self.capture.stop_capture()

    def _schedule_progress_update(self):
        """Schedule periodic progress updates."""
        if not self._running:
            return

        elapsed = time.time() - self.start_time
        fps = self.frames_processed / elapsed if elapsed > 0 else 0

        # Get decoder progress
        received, total = 0, 0
        if self.decoder:
            received, total = self.decoder.get_progress()

        try:
            self.ui.update_progress(
                received,
                total,
                self.crc_errors,
                self.fec_corrections,
                elapsed,
                fps
            )
        except Exception:
            pass

        if self._running:
            self.ui.root.after(200, self._schedule_progress_update)

    def _signal_complete(self):
        """Signal reception complete."""
        self._running = False

        def do_complete():
            if self.assembler:
                status = self.assembler.get_status()

                # Auto-save
                result = self.assembler.assemble()

                self.ui.show_complete(
                    success=result.complete,
                    filename=result.filename or "received_file",
                    file_size=result.file_size,
                    hash_valid=result.file_hash_valid,
                    message=result.message
                )

        try:
            self.ui.root.after(100, do_complete)
        except Exception:
            pass

    def run(self):
        """Run the application."""
        self.ui.run()


def main():
    """Entry point for receiver application."""
    app = ReceiverApplication()
    app.run()


if __name__ == "__main__":
    main()
