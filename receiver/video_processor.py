"""
Visual Data Diode - Video File Processor

Processes recorded video files that may contain multiple encoded files.
Detects file boundaries using sync/end patterns and decodes each file.
"""

import numpy as np
import time
import os
from typing import Optional, List, Callable, Tuple, Generator
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    EncodingProfile, DEFAULT_PROFILE,
    SYNC_FRAME_COUNT, END_FRAME_COUNT,
    COLOR_BLACK
)
from .sync import FrameSync, SyncResult
from .decoder import FrameDecoder, StreamDecoder, DecodeResult
from .assembler import BlockAssembler, AssemblyStatus

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class ProcessorState(Enum):
    """State of the video processor."""
    SCANNING = "scanning"       # Looking for sync pattern (start of file)
    RECEIVING = "receiving"     # Decoding data frames
    END_DETECTED = "end"        # End pattern detected
    GAP = "gap"                 # In gap between files


@dataclass
class DecodedFile:
    """Information about a decoded file."""
    index: int                  # File index in video (0-based)
    filename: str               # Original filename (from metadata)
    output_path: str            # Path where file was saved
    file_size: int              # Size in bytes
    hash_valid: Optional[bool]  # Whether hash verified
    start_frame: int            # Frame where file started
    end_frame: int              # Frame where file ended
    blocks_received: int        # Number of blocks decoded
    total_blocks: int           # Total blocks expected
    crc_errors: int             # CRC errors encountered
    fec_corrections: int        # FEC corrections made


@dataclass
class ProcessorProgress:
    """Progress information for the processor."""
    state: ProcessorState
    current_frame: int
    total_frames: int
    files_found: int
    files_decoded: int
    current_file_blocks: int
    current_file_total_blocks: int
    gap_frames: int             # Frames skipped as unrecognized


@dataclass
class ProcessorResult:
    """Final result of video processing."""
    success: bool
    files_decoded: List[DecodedFile]
    total_frames: int
    processing_time: float
    gap_frames: int
    message: str


class VideoProcessor:
    """
    Processes video files containing encoded data.

    Detects file boundaries using sync/end patterns,
    decodes each file, and saves to output directory.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        profile: EncodingProfile = DEFAULT_PROFILE,
        password: Optional[str] = None,
        on_progress: Optional[Callable[[ProcessorProgress], None]] = None,
        on_file_complete: Optional[Callable[[DecodedFile], None]] = None
    ):
        """
        Initialize video processor.

        Args:
            video_path: Path to input video file
            output_dir: Directory for decoded files
            profile: Encoding profile
            password: Decryption password (optional)
            on_progress: Progress callback
            on_file_complete: File completion callback
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")

        self.video_path = video_path
        self.output_dir = output_dir
        self.profile = profile
        self.password = password
        self.on_progress = on_progress
        self.on_file_complete = on_file_complete

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Video properties
        self.total_frames = 0
        self.video_fps = 0.0
        self.video_width = 0
        self.video_height = 0

        # Processing state
        self._running = False
        self._state = ProcessorState.SCANNING
        self._current_frame = 0
        self._gap_frames = 0

        # Components
        self._sync = FrameSync(profile)
        self._decoder: Optional[StreamDecoder] = None
        self._assembler: Optional[BlockAssembler] = None

        # Results
        self._decoded_files: List[DecodedFile] = []
        self._current_file_start = 0
        self._current_file_index = 0
        self._crc_errors = 0
        self._fec_corrections = 0

        # End detection
        self._consecutive_black_frames = 0
        self._consecutive_unsynced_frames = 0
        self._black_threshold = 30  # Pixel value threshold for black
        self._end_frame_threshold = 5  # Consecutive black frames to detect end

    def process(self) -> ProcessorResult:
        """
        Process the entire video file.

        Returns:
            ProcessorResult with all decoded files
        """
        start_time = time.time()

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return ProcessorResult(
                success=False,
                files_decoded=[],
                total_frames=0,
                processing_time=0,
                gap_frames=0,
                message=f"Failed to open video: {self.video_path}"
            )

        try:
            # Get video properties
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"Processing video: {self.video_path}")
            print(f"  Resolution: {self.video_width}x{self.video_height}")
            print(f"  FPS: {self.video_fps}, Frames: {self.total_frames}")

            self._running = True
            self._state = ProcessorState.SCANNING
            self._current_frame = 0
            self._gap_frames = 0

            # Process frames
            while self._running:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self._current_frame += 1

                # Process frame based on state
                self._process_frame(frame)

                # Report progress
                if self.on_progress and self._current_frame % 30 == 0:
                    progress = self._get_progress()
                    self.on_progress(progress)

            # Finalize any in-progress file
            if self._state == ProcessorState.RECEIVING:
                self._finalize_current_file()

            elapsed = time.time() - start_time

            return ProcessorResult(
                success=True,
                files_decoded=self._decoded_files,
                total_frames=self.total_frames,
                processing_time=elapsed,
                gap_frames=self._gap_frames,
                message=f"Decoded {len(self._decoded_files)} file(s)"
            )

        except Exception as e:
            return ProcessorResult(
                success=False,
                files_decoded=self._decoded_files,
                total_frames=self.total_frames,
                processing_time=time.time() - start_time,
                gap_frames=self._gap_frames,
                message=f"Processing error: {str(e)}"
            )
        finally:
            cap.release()

    def stop(self):
        """Stop processing."""
        self._running = False

    def _process_frame(self, frame: np.ndarray):
        """Process a single frame based on current state."""
        if self._state == ProcessorState.SCANNING:
            self._handle_scanning(frame)
        elif self._state == ProcessorState.RECEIVING:
            self._handle_receiving(frame)
        elif self._state == ProcessorState.GAP:
            self._handle_gap(frame)

    def _handle_scanning(self, frame: np.ndarray):
        """Handle frame in scanning state - looking for sync pattern."""
        # Detect sync pattern
        sync_result = self._sync.detect_sync(frame)

        if sync_result.is_synced:
            # Found sync - start receiving
            print(f"[Frame {self._current_frame}] Sync detected, confidence: {sync_result.confidence:.2f}")
            self._start_new_file()
            self._state = ProcessorState.RECEIVING
            self._consecutive_unsynced_frames = 0

            # Try to decode this frame (might be sync-only or data)
            grid = self._sync.extract_grid(frame, sync_result)
            if grid is not None:
                self._decode_grid(grid)
        else:
            # Not synced - count as gap
            self._gap_frames += 1

    def _handle_receiving(self, frame: np.ndarray):
        """Handle frame in receiving state - decoding data."""
        # Check for end pattern (black frame)
        if self._is_black_frame(frame):
            self._consecutive_black_frames += 1

            if self._consecutive_black_frames >= self._end_frame_threshold:
                # End pattern detected
                print(f"[Frame {self._current_frame}] End pattern detected")
                self._finalize_current_file()
                self._state = ProcessorState.GAP
                self._consecutive_black_frames = 0
                return
        else:
            self._consecutive_black_frames = 0

        # Detect sync
        sync_result = self._sync.detect_sync(frame)

        if sync_result.is_synced:
            self._consecutive_unsynced_frames = 0

            # Extract grid and decode
            grid = self._sync.extract_grid(frame, sync_result)
            if grid is not None:
                self._decode_grid(grid)
        else:
            self._consecutive_unsynced_frames += 1

            # Too many unsynced frames - might have missed end
            if self._consecutive_unsynced_frames > 30:
                print(f"[Frame {self._current_frame}] Lost sync, finalizing file")
                self._finalize_current_file()
                self._state = ProcessorState.SCANNING
                self._consecutive_unsynced_frames = 0

    def _handle_gap(self, frame: np.ndarray):
        """Handle frame in gap state - between files."""
        # Check for new sync pattern
        sync_result = self._sync.detect_sync(frame)

        if sync_result.is_synced:
            # New file starting
            print(f"[Frame {self._current_frame}] New sync detected after gap")
            self._start_new_file()
            self._state = ProcessorState.RECEIVING

            # Try to decode
            grid = self._sync.extract_grid(frame, sync_result)
            if grid is not None:
                self._decode_grid(grid)
        else:
            # Still in gap
            self._gap_frames += 1

    def _is_black_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is mostly black (end pattern)."""
        # Check average brightness
        mean_val = np.mean(frame)
        return mean_val < self._black_threshold

    def _start_new_file(self):
        """Initialize state for a new file."""
        self._decoder = StreamDecoder(profile=self.profile)
        self._assembler = BlockAssembler(
            output_dir=self.output_dir,
            password=self.password
        )
        self._current_file_start = self._current_frame
        self._crc_errors = 0
        self._fec_corrections = 0
        self._consecutive_black_frames = 0
        self._consecutive_unsynced_frames = 0

    def _decode_grid(self, grid: np.ndarray):
        """Decode a grid and add to current file."""
        if self._decoder is None or self._assembler is None:
            return

        result = self._decoder.process_grid(grid)

        if result is None:
            # Duplicate block
            return

        if result.success and result.block:
            # Add to assembler
            success, msg = self._assembler.add_block(result.block)

            if result.fec_corrected > 0:
                self._fec_corrections += result.fec_corrected

        if not result.crc_valid:
            self._crc_errors += 1

    def _finalize_current_file(self):
        """Finalize and save the current file."""
        if self._assembler is None or self._decoder is None:
            return

        # Get decoder progress
        received, total = self._decoder.get_progress()

        if received == 0:
            print(f"  No blocks received, skipping file")
            return

        # Check if complete
        if not self._assembler.is_complete():
            missing = self._assembler.get_status().missing_blocks
            print(f"  Incomplete: {received}/{total} blocks, missing {len(missing)}")
            # Still try to assemble what we have if most blocks received
            if received < total * 0.9:
                print(f"  Skipping file (too many missing blocks)")
                return

        # Assemble file
        result = self._assembler.assemble()

        if result.output_path:
            # Create decoded file record
            decoded = DecodedFile(
                index=self._current_file_index,
                filename=result.filename or f"file_{self._current_file_index}",
                output_path=result.output_path,
                file_size=result.file_size,
                hash_valid=result.file_hash_valid,
                start_frame=self._current_file_start,
                end_frame=self._current_frame,
                blocks_received=result.blocks_received,
                total_blocks=result.total_blocks,
                crc_errors=self._crc_errors,
                fec_corrections=self._fec_corrections
            )

            self._decoded_files.append(decoded)
            self._current_file_index += 1

            print(f"  Saved: {result.filename} ({result.file_size} bytes)")

            # Callback
            if self.on_file_complete:
                self.on_file_complete(decoded)

    def _get_progress(self) -> ProcessorProgress:
        """Get current progress."""
        current_blocks = 0
        total_blocks = 0

        if self._decoder:
            current_blocks, total_blocks = self._decoder.get_progress()

        return ProcessorProgress(
            state=self._state,
            current_frame=self._current_frame,
            total_frames=self.total_frames,
            files_found=len(self._decoded_files) + (1 if self._state == ProcessorState.RECEIVING else 0),
            files_decoded=len(self._decoded_files),
            current_file_blocks=current_blocks,
            current_file_total_blocks=total_blocks,
            gap_frames=self._gap_frames
        )


def process_video_file(
    video_path: str,
    output_dir: str,
    profile: EncodingProfile = DEFAULT_PROFILE,
    password: Optional[str] = None,
    on_progress: Optional[Callable[[ProcessorProgress], None]] = None
) -> ProcessorResult:
    """
    Convenience function to process a video file.

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        profile: Encoding profile
        password: Decryption password
        on_progress: Progress callback

    Returns:
        ProcessorResult
    """
    processor = VideoProcessor(
        video_path=video_path,
        output_dir=output_dir,
        profile=profile,
        password=password,
        on_progress=on_progress
    )

    return processor.process()


def get_video_info(video_path: str) -> Optional[dict]:
    """
    Get information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info or None if failed
    """
    if not OPENCV_AVAILABLE:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        info = {
            'path': video_path,
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        return info
    finally:
        cap.release()
