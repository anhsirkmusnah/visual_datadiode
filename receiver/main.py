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
from .binary_receiver import BinaryReceiver


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

        # Binary capture state
        self._binary_receiver: Optional[BinaryReceiver] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running = False

        # Connect UI callbacks
        self._setup_ui_callbacks()

    def _setup_ui_callbacks(self):
        """Connect UI callbacks."""
        self.ui.on_start = self._on_start
        self.ui.on_stop = self._on_stop
        self.ui.on_capture_start = self._on_capture_start
        self.ui.on_capture_stop = self._on_capture_stop

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

    # ── Binary Capture handlers ──────────────────────────────────────

    def _on_capture_start(self, settings: dict):
        """Handle Start Capture from Binary Stream tab."""
        self._capture_running = True

        def on_progress(stats_dict):
            if not self._capture_running:
                return
            try:
                self.ui.root.after(0, lambda s=stats_dict: self.ui.update_capture_progress(s))
            except Exception:
                pass

        def on_file_complete(result_dict):
            try:
                self.ui.root.after(0, lambda r=result_dict: self.ui.add_capture_file(r))
            except Exception:
                pass

        self._binary_receiver = BinaryReceiver(
            output_dir=settings['output_dir'],
            device_index=settings['device_index'],
            fec_ratio=settings['fec_ratio'],
            idle_timeout=settings['idle_timeout'],
            on_progress=on_progress,
            on_file_complete=on_file_complete,
        )

        def capture_thread():
            try:
                success = self._binary_receiver.run()
                if self._capture_running:
                    msg = "File received and verified!" if success else "Capture finished (check results)"
                    self._signal_capture_complete(success is not False, msg)
            except Exception as e:
                self._signal_capture_complete(False, str(e))

        self._capture_thread = threading.Thread(target=capture_thread, daemon=True)
        self._capture_thread.start()

    def _on_capture_stop(self):
        """Handle Stop from Binary Stream tab."""
        self._capture_running = False
        if self._binary_receiver:
            self._binary_receiver.stop()

    def _signal_capture_complete(self, success: bool, message: str):
        """Signal binary capture complete to UI."""
        self._capture_running = False

        def show():
            self.ui.show_capture_complete(success, message)

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
