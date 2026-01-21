"""
Visual Data Diode - Video Pre-renderer

Pre-renders all frames to a video file for perfect playback timing.
This ensures consistent frame delivery without real-time encoding overhead.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    EncodingProfile, DEFAULT_PROFILE,
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_FPS,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_BLACK
)
from .encoder import FrameEncoder
from .chunker import FileChunker


@dataclass
class PreRenderStats:
    """Statistics from pre-rendering."""
    total_frames: int = 0
    total_blocks: int = 0
    file_size: int = 0
    render_time: float = 0.0
    output_size_bytes: int = 0

    @property
    def frames_per_second(self) -> float:
        if self.render_time > 0:
            return self.total_frames / self.render_time
        return 0.0


class VideoPreRenderer:
    """
    Pre-renders transmission frames to a video file.

    Creates a video file that can be played back with perfect timing
    for reliable data transmission.
    """

    # Codec options for pre-rendering (prefer reliable lossless)
    # Note: FFV1 can fail with certain frame dimensions, so prefer HFYU/MJPG
    CODEC_OPTIONS = [
        ('HFYU', 'avi', 'Lossless HuffYUV'),
        ('MJPG', 'avi', 'Motion JPEG (95%)'),
        ('XVID', 'avi', 'XVID'),
    ]

    def __init__(
        self,
        output_path: str,
        profile: EncodingProfile = DEFAULT_PROFILE,
        fps: int = DEFAULT_FPS,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        codec: str = None,
        sync_frames: int = 30,   # Sync-only frames at start
        end_frames: int = 60,    # Black frames at end
        repeat_count: int = 1,   # Times to repeat each data frame
        on_progress: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize pre-renderer.

        Args:
            output_path: Path for output video
            profile: Encoding profile
            fps: Output video FPS
            width: Frame width
            height: Frame height
            codec: Preferred codec (None = auto)
            sync_frames: Number of sync-only frames at start
            end_frames: Number of black frames at end
            repeat_count: Times to repeat each data frame
            on_progress: Callback (current_frame, total_frames)
        """
        self.base_output_path = Path(output_path).with_suffix('')
        self.profile = profile
        self.fps = fps
        self.width = width
        self.height = height
        self.preferred_codec = codec
        self.sync_frames = sync_frames
        self.end_frames = end_frames
        self.repeat_count = repeat_count
        self.on_progress = on_progress

        self.encoder = FrameEncoder(profile=profile)
        _, self.payload_capacity = self.encoder.get_capacity()

        self.writer: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[Path] = None
        self.codec_name: str = ""

        self.stats = PreRenderStats()

    def _select_codec(self) -> tuple:
        """Select best available codec."""
        if self.preferred_codec:
            for fourcc, ext, name in self.CODEC_OPTIONS:
                if fourcc.strip() == self.preferred_codec.strip():
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
                    # Write a test frame
                    test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    writer.write(test_frame)
                    writer.release()
                    Path(test_path).unlink(missing_ok=True)
                    return fourcc, ext, name
                writer.release()
            except Exception:
                pass
            finally:
                Path(test_path).unlink(missing_ok=True)

        # Fallback
        return 'MJPG', 'avi', 'Motion JPEG (fallback)'

    def _create_sync_frame(self) -> np.ndarray:
        """Create a sync-only frame (no data, just sync pattern)."""
        # Use encoder to create proper sync frame with corner markers
        grid = self.encoder.encode_sync_only()
        return self.encoder.grid_to_frame(grid)

    def _create_end_frame(self) -> np.ndarray:
        """Create an end-of-transmission frame (all black)."""
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def prerender_file(self, input_path: str) -> PreRenderStats:
        """
        Pre-render a file to video.

        Args:
            input_path: Path to file to encode

        Returns:
            Rendering statistics
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Select codec
        fourcc, ext, name = self._select_codec()
        self.codec_name = name
        self.output_path = self.base_output_path.with_suffix(f'.{ext}')

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file info
        file_size = input_path.stat().st_size

        # Create chunker
        chunker = FileChunker(
            profile=self.profile,
            fec_ratio=0.1
        )

        # Prepare file
        total_blocks, file_size_actual, file_hash = chunker.prepare_file(str(input_path))
        total_data_frames = total_blocks * self.repeat_count
        total_frames = self.sync_frames + total_data_frames + self.end_frames

        print(f"Pre-rendering: {input_path}")
        print(f"  File size: {file_size:,} bytes")
        print(f"  Blocks: {total_blocks}")
        print(f"  Frames: {total_frames} ({self.sync_frames} sync + {total_data_frames} data + {self.end_frames} end)")
        print(f"  Codec: {name}")
        print(f"  Output: {self.output_path}")

        # Initialize writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer with {name}")

        # Reset stats
        self.stats = PreRenderStats()
        self.stats.file_size = file_size
        self.stats.total_blocks = total_blocks
        start_time = time.time()

        frame_num = 0

        # Write sync frames
        print("  Writing sync frames...")
        sync_frame = self._create_sync_frame()
        sync_frame_bgr = cv2.cvtColor(sync_frame, cv2.COLOR_RGB2BGR)

        for _ in range(self.sync_frames):
            self.writer.write(sync_frame_bgr)
            frame_num += 1
            self._report_progress(frame_num, total_frames)

        # Write data frames
        print("  Writing data frames...")
        for block in chunker.generate_blocks():
            # Encode block to full resolution frame
            frame_rgb = self.encoder.encode_to_frame(block)

            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Write frame (repeat_count times)
            for _ in range(self.repeat_count):
                self.writer.write(frame_bgr)
                frame_num += 1
                self._report_progress(frame_num, total_frames)

        # Write end frames
        print("  Writing end frames...")
        end_frame = self._create_end_frame()

        for _ in range(self.end_frames):
            self.writer.write(end_frame)
            frame_num += 1
            self._report_progress(frame_num, total_frames)

        # Finalize
        self.writer.release()
        self.writer = None

        self.stats.total_frames = frame_num
        self.stats.render_time = time.time() - start_time
        self.stats.output_size_bytes = self.output_path.stat().st_size

        print(f"\nPre-render complete:")
        print(f"  Frames: {self.stats.total_frames}")
        print(f"  Time: {self.stats.render_time:.1f}s ({self.stats.frames_per_second:.1f} fps)")
        print(f"  Output size: {self.stats.output_size_bytes / 1024 / 1024:.1f} MB")

        return self.stats

    def _report_progress(self, current: int, total: int):
        """Report progress."""
        if self.on_progress:
            self.on_progress(current, total)

        # Print progress every 10%
        pct = int(current * 100 / total)
        if pct % 10 == 0 and current > 0:
            prev_pct = int((current - 1) * 100 / total)
            if prev_pct % 10 != 0:
                print(f"    {pct}%")

    def prerender_data(self, data: bytes, session_name: str = "session") -> PreRenderStats:
        """
        Pre-render raw data to video.

        Args:
            data: Raw bytes to encode
            session_name: Name for the session

        Returns:
            Rendering statistics
        """
        # Write data to temp file
        temp_path = self.base_output_path.parent / f"_temp_{session_name}.bin"
        temp_path.write_bytes(data)

        try:
            return self.prerender_file(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)


class VideoPlayer:
    """
    Plays a pre-rendered video file through the display.

    Handles pygame display and precise frame timing.
    """

    def __init__(
        self,
        video_path: str,
        display_index: int = 0,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_complete: Optional[Callable[[], None]] = None
    ):
        """
        Initialize player.

        Args:
            video_path: Path to pre-rendered video
            display_index: Display to use
            on_progress: Callback (current_frame, total_frames)
            on_complete: Callback when playback completes
        """
        self.video_path = Path(video_path)
        self.display_index = display_index
        self.on_progress = on_progress
        self.on_complete = on_complete

        self.cap: Optional[cv2.VideoCapture] = None
        self.screen = None
        self._running = False

        # Video info
        self.total_frames = 0
        self.video_fps = 0.0
        self.video_width = 0
        self.video_height = 0

    def open(self) -> bool:
        """Open video file."""
        if not self.video_path.exists():
            print(f"Video not found: {self.video_path}")
            return False

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video loaded: {self.video_path}")
        print(f"  {self.video_width}x{self.video_height} @ {self.video_fps:.1f} FPS")
        print(f"  {self.total_frames} frames, {self.total_frames/self.video_fps:.1f}s duration")

        return True

    def play(self) -> bool:
        """
        Play video through display.

        Returns:
            True if playback completed successfully
        """
        import pygame

        if not self.cap:
            if not self.open():
                return False

        # Initialize pygame
        pygame.init()
        sizes = pygame.display.get_desktop_sizes()

        if self.display_index >= len(sizes):
            print(f"Display {self.display_index} not found")
            pygame.quit()
            return False

        # Position on target display
        import os
        x_offset = sum(sizes[i][0] for i in range(self.display_index))
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},0"

        disp_w, disp_h = sizes[self.display_index]
        self.screen = pygame.display.set_mode((disp_w, disp_h), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)

        print(f"Playing on display {self.display_index} ({disp_w}x{disp_h})")

        frame_time = 1.0 / self.video_fps
        self._running = True
        frame_num = 0

        try:
            while self._running:
                frame_start = time.perf_counter()

                # Check for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self._running = False
                            break

                if not self._running:
                    break

                # Read frame
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    # End of video
                    break

                frame_num += 1

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize if needed
                if frame_rgb.shape[1] != disp_w or frame_rgb.shape[0] != disp_h:
                    frame_rgb = cv2.resize(frame_rgb, (disp_w, disp_h))

                # Display
                surface = pygame.surfarray.make_surface(
                    np.transpose(frame_rgb, (1, 0, 2))
                )
                self.screen.blit(surface, (0, 0))
                pygame.display.flip()

                # Progress callback
                if self.on_progress:
                    self.on_progress(frame_num, self.total_frames)

                # Precise timing
                elapsed = time.perf_counter() - frame_start
                sleep_time = frame_time - elapsed - 0.001  # Leave 1ms for overhead

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Busy wait for precise timing
                while time.perf_counter() - frame_start < frame_time:
                    pass

        finally:
            pygame.quit()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset for replay

        print(f"Playback complete: {frame_num}/{self.total_frames} frames")

        if self.on_complete:
            self.on_complete()

        return frame_num >= self.total_frames

    def stop(self):
        """Stop playback."""
        self._running = False

    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None


def prerender_and_play(
    input_file: str,
    output_video: str = None,
    display_index: int = 0,
    profile: EncodingProfile = DEFAULT_PROFILE,
    fps: int = 30,
    repeat_count: int = 2
) -> bool:
    """
    Convenience function to pre-render and play a file.

    Args:
        input_file: File to transmit
        output_video: Path for pre-rendered video (None = auto)
        display_index: Display to play on
        profile: Encoding profile
        fps: Playback FPS
        repeat_count: Times to repeat each frame

    Returns:
        True if successful
    """
    input_path = Path(input_file)

    if output_video is None:
        output_video = str(input_path.with_suffix('')) + "_prerendered"

    # Pre-render
    renderer = VideoPreRenderer(
        output_path=output_video,
        profile=profile,
        fps=fps,
        repeat_count=repeat_count
    )

    try:
        stats = renderer.prerender_file(input_file)
    except Exception as e:
        print(f"Pre-render failed: {e}")
        return False

    # Play
    player = VideoPlayer(
        video_path=str(renderer.output_path),
        display_index=display_index
    )

    return player.play()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-render and play file")
    parser.add_argument('input_file', help='File to transmit')
    parser.add_argument('--display', type=int, default=3, help='Display index')
    parser.add_argument('--profile', choices=['conservative', 'standard', 'aggressive', 'ultra'],
                        default='conservative', help='Encoding profile')
    parser.add_argument('--fps', type=int, default=30, help='Output FPS')
    parser.add_argument('--repeat', type=int, default=2, help='Frame repeat count')
    parser.add_argument('--render-only', action='store_true', help='Only render, don\'t play')

    args = parser.parse_args()

    # Select profile
    profiles = {
        'conservative': PROFILE_CONSERVATIVE,
        'standard': PROFILE_STANDARD,
        'aggressive': PROFILE_AGGRESSIVE,
        'ultra': PROFILE_ULTRA
    }
    profile = profiles[args.profile]

    if args.render_only:
        renderer = VideoPreRenderer(
            output_path=args.input_file + "_prerendered",
            profile=profile,
            fps=args.fps,
            repeat_count=args.repeat
        )
        renderer.prerender_file(args.input_file)
    else:
        success = prerender_and_play(
            args.input_file,
            display_index=args.display,
            profile=profile,
            fps=args.fps,
            repeat_count=args.repeat
        )
        sys.exit(0 if success else 1)
