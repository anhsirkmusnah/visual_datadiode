"""
Visual Data Diode - Receiver Main Application

Main entry point for the receiver application.
Processes recorded video files to extract encoded data.
"""

import threading
import time
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import EncodingProfile, DEFAULT_PROFILE
from .video_processor import VideoProcessor, ProcessorProgress, DecodedFile, ProcessorResult
from .ui import ReceiverUI


class ReceiverApplication:
    """
    Main receiver application.

    Processes video files to extract encoded data files.
    """

    def __init__(self):
        self.ui = ReceiverUI()
        self.processor: Optional[VideoProcessor] = None

        # State
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None

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

    def _on_start(self, settings: dict):
        """Handle start processing."""
        self.profile = settings['profile']
        self.output_dir = settings['output_dir']
        self.password = settings['password']
        video_path = settings['video_path']

        # Create processor
        self.processor = VideoProcessor(
            video_path=video_path,
            output_dir=self.output_dir,
            profile=self.profile,
            password=self.password,
            on_progress=self._on_progress,
            on_file_complete=self._on_file_complete
        )

        # Start processing in background thread
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._processing_thread.start()

    def _on_stop(self):
        """Handle stop processing."""
        self._running = False
        if self.processor:
            self.processor.stop()

    def _processing_loop(self):
        """Main processing loop (runs in background thread)."""
        try:
            result = self.processor.process()
            self._signal_complete(result)

        except Exception as e:
            self._signal_error(str(e))

    def _on_progress(self, progress: ProcessorProgress):
        """Handle progress update from processor."""
        if not self._running:
            return

        try:
            self.ui.root.after(0, lambda p=progress: self.ui.update_progress(p))
        except Exception:
            pass

    def _on_file_complete(self, decoded_file: DecodedFile):
        """Handle file completion from processor."""
        try:
            file_info = {
                'filename': decoded_file.filename,
                'output_path': decoded_file.output_path,
                'file_size': decoded_file.file_size,
                'hash_valid': decoded_file.hash_valid,
                'blocks_received': decoded_file.blocks_received,
                'total_blocks': decoded_file.total_blocks
            }
            self.ui.root.after(0, lambda f=file_info: self.ui.add_decoded_file(f))
        except Exception:
            pass

    def _signal_complete(self, result: ProcessorResult):
        """Signal processing complete."""
        self._running = False

        def show():
            self.ui.show_complete(
                success=result.success,
                message=result.message,
                files_decoded=len(result.files_decoded),
                elapsed=result.processing_time
            )

        try:
            self.ui.root.after(100, show)
        except Exception:
            pass

    def _signal_error(self, message: str):
        """Signal processing error."""
        self._running = False

        def show():
            self.ui.show_complete(False, message=message)

        try:
            self.ui.root.after(100, show)
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
