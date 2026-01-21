"""
Visual Data Diode - Sender Main Application

Main entry point for the sender application.
Coordinates UI, chunking, encoding, and video file output.
"""

import threading
import time
import os
from typing import Optional, List
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
from .video_encoder import VideoEncoder, BatchVideoEncoder
from .timing import format_duration
from .ui import SenderUI


class SenderApplication:
    """
    Main sender application.

    Encodes files to video files instead of displaying on screen.
    """

    def __init__(self):
        self.ui = SenderUI()
        self.chunker: Optional[FileChunker] = None
        self.encoder: Optional[FrameEncoder] = None
        self.video_encoder: Optional[VideoEncoder] = None

        # State
        self._running = False
        self._encoding_thread: Optional[threading.Thread] = None

        # Progress tracking
        self.current_file_index = 0
        self.total_files = 0
        self.current_frame = 0
        self.total_frames = 0
        self.start_time = 0.0
        self.file_start_time = 0.0

        # Settings
        self.profile = DEFAULT_PROFILE
        self.fps = DEFAULT_FPS
        self.repeat_count = DEFAULT_REPEAT_COUNT
        self.file_paths: List[str] = []
        self.output_dir: str = ""

        # Connect UI callbacks
        self._setup_ui_callbacks()

    def _setup_ui_callbacks(self):
        """Connect UI callbacks to handler methods."""
        self.ui.on_start = self._on_start
        self.ui.on_stop = self._on_stop

    def _on_start(self, settings: dict):
        """Handle start encoding."""
        self.file_paths = settings['file_paths']
        self.output_dir = settings['output_dir']
        self.profile = settings['profile']
        self.fps = settings['fps']
        self.repeat_count = settings['repeat_count']
        self.add_audio = settings['add_audio']
        self.resolution_width = settings.get('resolution_width', 1920)
        self.resolution_height = settings.get('resolution_height', 1080)

        # Setup encryption if enabled
        self.encryption_key = None
        if settings['encrypt'] and settings['password']:
            key, salt = derive_key(settings['password'])
            self.encryption_key = key
        self.encrypt = settings['encrypt']

        # Initialize state
        self._running = True
        self.current_file_index = 0
        self.total_files = len(self.file_paths)
        self.start_time = time.time()

        # Start encoding in background thread
        self._encoding_thread = threading.Thread(
            target=self._encoding_loop,
            daemon=True
        )
        self._encoding_thread.start()

        # Start progress update timer
        self._schedule_progress_update()

    def _on_stop(self):
        """Handle stop encoding."""
        self._running = False

    def _get_output_path(self, input_path: str) -> str:
        """Generate output video path for an input file."""
        basename = os.path.basename(input_path)
        name, _ = os.path.splitext(basename)
        output_name = f"{name}_encoded.mp4"
        return os.path.join(self.output_dir, output_name)

    def _calculate_total_frames(self, total_blocks: int) -> int:
        """Calculate total frames for a file."""
        data_frames = total_blocks * self.repeat_count
        return SYNC_FRAME_COUNT + data_frames + END_FRAME_COUNT

    def _encoding_loop(self):
        """Main encoding loop (runs in background thread)."""
        completed_files = []

        try:
            # Process each file
            for file_index, file_path in enumerate(self.file_paths):
                if not self._running:
                    break

                self.current_file_index = file_index
                self.file_start_time = time.time()

                try:
                    self._encode_single_file(file_path)
                    output_path = self._get_output_path(file_path)
                    completed_files.append((file_path, output_path))

                    # Notify UI of file completion
                    self._signal_file_complete(file_path, output_path)

                except Exception as e:
                    self._signal_error(f"Error encoding {os.path.basename(file_path)}: {str(e)}")
                    if not self._running:
                        break

            # All files complete
            if self._running:
                self._signal_complete(completed_files)

        except Exception as e:
            self._signal_error(str(e))

    def _encode_single_file(self, file_path: str):
        """Encode a single file to video."""
        # Initialize chunker for this file
        self.chunker = FileChunker(
            profile=self.profile,
            encrypt=self.encrypt,
            encryption_key=self.encryption_key
        )

        # Prepare file and get block count
        total_blocks, file_size, file_hash = self.chunker.prepare_file(file_path)

        # Calculate total frames
        self.total_frames = self._calculate_total_frames(total_blocks)
        self.current_frame = 0

        # Initialize frame encoder
        self.encoder = FrameEncoder(profile=self.profile)

        # Create video encoder
        output_path = self._get_output_path(file_path)
        self.video_encoder = VideoEncoder(
            output_path=output_path,
            width=self.resolution_width,
            height=self.resolution_height,
            fps=self.fps,
            add_audio=self.add_audio
        )

        try:
            # Open video encoder
            self.video_encoder.open(total_frames=self.total_frames)

            # Write sync frames
            sync_frame = self.encoder.encode_sync_only()
            sync_frame_full = self.encoder.grid_to_frame(sync_frame)

            for _ in range(SYNC_FRAME_COUNT):
                if not self._running:
                    return

                self.video_encoder.write_frame(sync_frame_full)
                self.current_frame += 1

            # Write data frames with repeats
            for repeat in range(self.repeat_count):
                for block in self.chunker.generate_blocks():
                    if not self._running:
                        return

                    # Encode block to frame
                    frame = self.encoder.encode_to_frame(block)
                    self.video_encoder.write_frame(frame)
                    self.current_frame += 1

            # Write end frames
            end_frame = self.encoder.encode_end_frame()
            end_frame_full = self.encoder.grid_to_frame(end_frame)

            for _ in range(END_FRAME_COUNT):
                if not self._running:
                    return

                self.video_encoder.write_frame(end_frame_full)
                self.current_frame += 1

        finally:
            # Always close video encoder
            if self.video_encoder:
                self.video_encoder.close()
                self.video_encoder = None

    def _schedule_progress_update(self):
        """Schedule periodic progress updates."""
        if not self._running:
            return

        elapsed = time.time() - self.file_start_time if self.file_start_time > 0 else 0

        try:
            current_file_name = ""
            if self.current_file_index < len(self.file_paths):
                current_file_name = os.path.basename(self.file_paths[self.current_file_index])

            self.ui.update_progress(
                current_file_index=self.current_file_index,
                total_files=self.total_files,
                current_file_name=current_file_name,
                current_frame=self.current_frame,
                total_frames=self.total_frames,
                elapsed_time=elapsed
            )
        except Exception:
            pass  # UI might be closing

        if self._running:
            self.ui.root.after(100, self._schedule_progress_update)

    def _signal_file_complete(self, file_path: str, output_path: str):
        """Signal single file encoding complete."""
        def show():
            self.ui.show_file_complete(file_path, output_path)

        self.ui.root.after(10, show)

    def _signal_complete(self, completed_files: list):
        """Signal all encoding complete."""
        self._running = False
        elapsed = time.time() - self.start_time

        total_size = sum(os.path.getsize(p) for p, _ in completed_files)
        output_size = sum(
            os.path.getsize(o) if os.path.exists(o) else 0
            for _, o in completed_files
        )

        def show():
            message = (
                f"Encoding complete!\n\n"
                f"Files encoded: {len(completed_files)}\n"
                f"Input size: {self._format_size(total_size)}\n"
                f"Output size: {self._format_size(output_size)}\n"
                f"Total time: {format_duration(elapsed)}\n"
                f"\nOutput directory:\n{self.output_dir}"
            )
            self.ui.show_complete(True, message)

        self.ui.root.after(100, show)

    def _signal_error(self, message: str):
        """Signal encoding error."""
        self._running = False

        def show():
            self.ui.show_complete(False, message)

        self.ui.root.after(100, show)

    def _format_size(self, size: int) -> str:
        """Format file size as human-readable string."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"

    def run(self):
        """Run the application."""
        self.ui.run()


def main():
    """Entry point for sender application."""
    app = SenderApplication()
    app.run()


if __name__ == "__main__":
    main()
