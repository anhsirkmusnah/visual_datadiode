"""
Visual Data Diode - Video Decoder

Decodes data from a recorded video file.
This allows processing at any speed without real-time constraints.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    EncodingProfile, DEFAULT_PROFILE,
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    GRAY_THRESHOLDS
)
from .sync import FrameSync, SyncResult
from .decoder import FrameDecoder, StreamDecoder, DecodeResult


@dataclass
class VideoDecodeStats:
    """Statistics from video decoding."""
    total_frames: int = 0
    synced_frames: int = 0
    decoded_blocks: int = 0
    failed_decodes: int = 0
    duplicate_blocks: int = 0
    crc_errors: int = 0
    fec_corrections: int = 0
    processing_time: float = 0.0
    unique_blocks: int = 0
    total_blocks_expected: int = 0

    @property
    def sync_rate(self) -> float:
        if self.total_frames > 0:
            return self.synced_frames / self.total_frames
        return 0.0

    @property
    def decode_rate(self) -> float:
        if self.synced_frames > 0:
            return self.decoded_blocks / self.synced_frames
        return 0.0

    @property
    def fps(self) -> float:
        if self.processing_time > 0:
            return self.total_frames / self.processing_time
        return 0.0


class VideoDecoder:
    """
    Decodes data from a recorded video file.

    Processes each frame to extract data blocks, then assembles them
    into the original file.
    """

    def __init__(
        self,
        video_path: str,
        profile: EncodingProfile = DEFAULT_PROFILE,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
        on_block_decoded: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize video decoder.

        Args:
            video_path: Path to recorded video file
            profile: Encoding profile to use for decoding
            on_progress: Callback (current_frame, total_frames, blocks_decoded)
            on_block_decoded: Callback (block_index, total_blocks)
        """
        self.video_path = Path(video_path)
        self.profile = profile
        self.on_progress = on_progress
        self.on_block_decoded = on_block_decoded

        self.sync = FrameSync(profile=profile)
        self.stream_decoder = StreamDecoder(profile=profile)

        self.cap: Optional[cv2.VideoCapture] = None
        self.stats = VideoDecodeStats()

        # Video info
        self.total_frames = 0
        self.video_fps = 0.0
        self.video_width = 0
        self.video_height = 0

    def open(self) -> bool:
        """
        Open video file for processing.

        Returns:
            True if successful
        """
        if not self.video_path.exists():
            print(f"Video file not found: {self.video_path}")
            return False

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video opened: {self.video_path}")
        print(f"  Resolution: {self.video_width}x{self.video_height}")
        print(f"  FPS: {self.video_fps:.1f}, Total frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames / self.video_fps:.1f}s")

        return True

    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def decode_all(self) -> Tuple[VideoDecodeStats, bytes]:
        """
        Decode all frames and extract data.

        Returns:
            (stats, assembled_data)
        """
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                return self.stats, b''

        # Reset state
        self.stream_decoder.reset()
        self.stats = VideoDecodeStats()
        start_time = time.time()

        print(f"Decoding {self.total_frames} frames...")

        frame_idx = 0
        last_progress_pct = -1

        while True:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                break

            self.stats.total_frames += 1
            frame_idx += 1

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect sync
            sync_result = self.sync.detect_sync(frame_rgb)

            if not sync_result.is_synced:
                continue

            self.stats.synced_frames += 1

            # Extract grid
            grid = self.sync.extract_grid(frame_rgb, sync_result)

            if grid is None:
                continue

            # Decode block
            result = self.stream_decoder.process_grid(grid)

            if result is None:
                # Duplicate
                self.stats.duplicate_blocks += 1
                continue

            if result.success and result.block:
                self.stats.decoded_blocks += 1
                self.stats.unique_blocks = len(self.stream_decoder.received_blocks)

                if result.fec_corrected > 0:
                    self.stats.fec_corrections += result.fec_corrected

                if self.stream_decoder.total_blocks:
                    self.stats.total_blocks_expected = self.stream_decoder.total_blocks

                if self.on_block_decoded:
                    self.on_block_decoded(
                        result.block.header.block_index,
                        self.stream_decoder.total_blocks or 0
                    )
            else:
                self.stats.failed_decodes += 1
                if not result.crc_valid:
                    self.stats.crc_errors += 1

            # Progress callback
            progress_pct = int(frame_idx * 100 / self.total_frames)
            if progress_pct != last_progress_pct:
                last_progress_pct = progress_pct
                if self.on_progress:
                    self.on_progress(
                        frame_idx,
                        self.total_frames,
                        self.stats.unique_blocks
                    )

                # Print progress every 10%
                if progress_pct % 10 == 0:
                    print(f"  {progress_pct}% - {self.stats.unique_blocks} blocks decoded")

        self.stats.processing_time = time.time() - start_time

        # Assemble data
        data = self._assemble_data()

        # Final stats
        print(f"\nDecoding complete:")
        print(f"  Frames: {self.stats.total_frames} total, {self.stats.synced_frames} synced")
        print(f"  Blocks: {self.stats.unique_blocks}/{self.stats.total_blocks_expected}")
        print(f"  CRC errors: {self.stats.crc_errors}, FEC corrections: {self.stats.fec_corrections}")
        print(f"  Processing time: {self.stats.processing_time:.1f}s ({self.stats.fps:.1f} FPS)")

        return self.stats, data

    def _assemble_data(self) -> bytes:
        """Assemble decoded blocks into file data."""
        from shared import FileMetadata, BlockFlags

        if not self.stream_decoder.total_blocks:
            return b''

        # Check for missing blocks
        missing = self.stream_decoder.get_missing_blocks()

        if missing:
            print(f"Warning: Missing {len(missing)} blocks: {missing[:10]}...")

        # Assemble in order
        data = bytearray()

        for idx in range(self.stream_decoder.total_blocks):
            if idx in self.stream_decoder.received_blocks:
                result = self.stream_decoder.received_blocks[idx]
                if result.block:
                    payload = result.block.payload

                    # Block 0 has metadata prefix that needs to be stripped
                    if idx == 0:
                        try:
                            encrypted = bool(result.block.header.flags & BlockFlags.ENCRYPTED)
                            metadata, consumed = FileMetadata.unpack(payload, encrypted)
                            # Skip metadata, keep only file data
                            payload = payload[consumed:]
                        except Exception as e:
                            print(f"Warning: Could not unpack metadata from block 0: {e}")

                    data.extend(payload)
            else:
                # Missing block - fill with zeros (could use FEC for recovery)
                print(f"Missing block {idx}")

        return bytes(data)

    def get_file_info(self) -> dict:
        """Get decoded file information."""
        if not self.stream_decoder.received_blocks:
            return {}

        # Get info from first block
        if 0 in self.stream_decoder.received_blocks:
            result = self.stream_decoder.received_blocks[0]
            if result.block:
                header = result.block.header
                return {
                    'session_id': header.session_id,
                    'total_blocks': header.total_blocks,
                    'file_size': header.file_size,
                    'blocks_received': len(self.stream_decoder.received_blocks),
                    'complete': self.stream_decoder.is_complete()
                }

        return {}


class AutoProfileDecoder:
    """
    Automatically detects the encoding profile from the video.

    Tries different cell sizes and selects the one that decodes successfully.
    """

    PROFILES = [
        PROFILE_CONSERVATIVE,  # 16px cells
        PROFILE_STANDARD,      # 10px cells
        PROFILE_AGGRESSIVE,    # 8px cells
        PROFILE_ULTRA,         # 6px cells
    ]

    def __init__(self, video_path: str):
        self.video_path = video_path

    def detect_profile(self, sample_frames: int = 100) -> Optional[EncodingProfile]:
        """
        Detect the encoding profile by trying to decode sample frames.

        Args:
            sample_frames: Number of frames to sample

        Returns:
            Best matching profile or None
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // sample_frames)

        best_profile = None
        best_score = 0

        for profile in self.PROFILES:
            score = self._test_profile(cap, profile, sample_frames, sample_interval)

            print(f"  {profile.name}: score={score}")

            if score > best_score:
                best_score = score
                best_profile = profile

            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()

        if best_score > 0:
            print(f"Detected profile: {best_profile.name} (score={best_score})")
            return best_profile

        return None

    def _test_profile(
        self,
        cap: cv2.VideoCapture,
        profile: EncodingProfile,
        sample_frames: int,
        sample_interval: int
    ) -> int:
        """Test a profile and return success score."""
        sync = FrameSync(profile=profile)
        decoder = StreamDecoder(profile=profile)

        success_count = 0
        frames_tested = 0

        for i in range(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * sample_interval)
            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sync_result = sync.detect_sync(frame_rgb)

            if sync_result.is_synced:
                grid = sync.extract_grid(frame_rgb, sync_result)

                if grid is not None:
                    result = decoder.decoder.decode_grid(grid)

                    if result.success:
                        success_count += 1

            frames_tested += 1

        return success_count

    def decode(
        self,
        profile: EncodingProfile = None,
        on_progress: Optional[Callable] = None
    ) -> Tuple[VideoDecodeStats, bytes]:
        """
        Decode video with auto-detected or specified profile.

        Args:
            profile: Profile to use (None = auto-detect)
            on_progress: Progress callback

        Returns:
            (stats, data)
        """
        if profile is None:
            print("Auto-detecting encoding profile...")
            profile = self.detect_profile()

            if profile is None:
                print("Could not detect profile")
                return VideoDecodeStats(), b''

        decoder = VideoDecoder(
            self.video_path,
            profile=profile,
            on_progress=on_progress
        )

        return decoder.decode_all()


def decode_recorded_video(
    video_path: str,
    output_path: str = None,
    profile: EncodingProfile = None
) -> Tuple[bool, str]:
    """
    Convenience function to decode a recorded video.

    Args:
        video_path: Path to recorded video
        output_path: Path to save decoded file (None = auto from video name)
        profile: Encoding profile (None = auto-detect)

    Returns:
        (success, message)
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return False, f"Video not found: {video_path}"

    if output_path is None:
        output_path = video_path.with_suffix('.bin')

    # Decode
    auto_decoder = AutoProfileDecoder(str(video_path))
    stats, data = auto_decoder.decode(profile=profile)

    if not data:
        return False, "No data decoded"

    # Check completeness
    info = {}
    if stats.total_blocks_expected > 0:
        completeness = stats.unique_blocks / stats.total_blocks_expected
        if completeness < 1.0:
            return False, f"Incomplete: {stats.unique_blocks}/{stats.total_blocks_expected} blocks ({completeness*100:.1f}%)"

    # Save data
    with open(output_path, 'wb') as f:
        f.write(data)

    return True, f"Decoded {len(data)} bytes to {output_path}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_decoder.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    success, message = decode_recorded_video(video_path, output_path)
    print(message)
    sys.exit(0 if success else 1)
