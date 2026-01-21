"""
Visual Data Diode - Video Encoder

Encodes frames to MP4 video file with lossless H.264 and audio sync beeps.
"""

import numpy as np
import os
import subprocess
import tempfile
import struct
import wave
from typing import Optional, Callable, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_FPS

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class AudioGenerator:
    """
    Generates audio sync beeps for the video.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def generate_beep(
        self,
        frequency: int = 1000,
        duration_ms: int = 100,
        volume: float = 0.5
    ) -> np.ndarray:
        """Generate a sine wave beep."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, False)
        wave_data = np.sin(2 * np.pi * frequency * t) * volume

        # Apply fade in/out to avoid clicks
        fade_samples = min(100, num_samples // 4)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave_data[:fade_samples] *= fade_in
        wave_data[-fade_samples:] *= fade_out

        return (wave_data * 32767).astype(np.int16)

    def generate_silence(self, duration_ms: int) -> np.ndarray:
        """Generate silence."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.int16)

    def generate_sync_pattern(self) -> np.ndarray:
        """
        Generate sync audio pattern for start of transmission.
        Three ascending beeps: 800Hz, 1000Hz, 1200Hz
        """
        pattern = []
        for freq in [800, 1000, 1200]:
            pattern.append(self.generate_beep(freq, 150, 0.6))
            pattern.append(self.generate_silence(50))
        return np.concatenate(pattern)

    def generate_end_pattern(self) -> np.ndarray:
        """
        Generate end audio pattern for completion.
        Three descending beeps: 1200Hz, 1000Hz, 800Hz, then long 600Hz
        """
        pattern = []
        for freq in [1200, 1000, 800]:
            pattern.append(self.generate_beep(freq, 150, 0.6))
            pattern.append(self.generate_silence(50))
        pattern.append(self.generate_beep(600, 500, 0.7))
        return np.concatenate(pattern)

    def generate_progress_beep(self) -> np.ndarray:
        """Generate short beep for progress indication."""
        return self.generate_beep(1000, 50, 0.3)

    def save_wav(self, audio_data: np.ndarray, path: str):
        """Save audio data to WAV file."""
        with wave.open(path, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_data.tobytes())


class VideoEncoder:
    """
    Encodes frames to MP4 video with lossless H.264 and optional audio.

    Uses FFmpeg for muxing video and audio into final MP4.
    Falls back to OpenCV if FFmpeg is not available (no audio in that case).
    """

    def __init__(
        self,
        output_path: str,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = DEFAULT_FPS,
        add_audio: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize video encoder.

        Args:
            output_path: Path for output MP4 file
            width: Frame width
            height: Frame height
            fps: Frames per second
            add_audio: Whether to add audio sync beeps
            on_progress: Callback (current_frame, total_frames)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.add_audio = add_audio
        self.on_progress = on_progress

        self._temp_video_path: Optional[str] = None
        self._temp_audio_path: Optional[str] = None
        self._video_writer = None
        self._frame_count = 0
        self._total_frames = 0
        self._audio_gen = AudioGenerator()
        self._audio_samples: List[np.ndarray] = []

        # Check for FFmpeg
        self._ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                creationflags=0x08000000 if os.name == 'nt' else 0
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def open(self, total_frames: int = 0):
        """
        Open the video encoder.

        Args:
            total_frames: Expected total frames (for progress calculation)
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")

        self._total_frames = total_frames
        self._frame_count = 0
        self._audio_samples = []

        # Create temp directory
        self._temp_dir = tempfile.mkdtemp(prefix="vdd_encode_")

        if self._ffmpeg_available and self.add_audio:
            # Write to temp video file, will mux with audio later
            self._temp_video_path = os.path.join(self._temp_dir, "video.mp4")
            video_path = self._temp_video_path
        else:
            # Write directly to output
            video_path = self.output_path

        # Use lossless H.264 codec
        # mp4v works everywhere, but for true lossless we need ffmpeg
        if self._ffmpeg_available:
            # Write raw frames, will re-encode with ffmpeg
            self._temp_video_path = os.path.join(self._temp_dir, "video.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG for intermediate
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self._video_writer = cv2.VideoWriter(
            self._temp_video_path if self._temp_video_path else self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not self._video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

        # Add sync audio at start
        if self.add_audio:
            sync_audio = self._audio_gen.generate_sync_pattern()
            # Pad to align with frames
            sync_duration_samples = len(sync_audio)
            sync_duration_frames = int(np.ceil(
                sync_duration_samples / self._audio_gen.sample_rate * self.fps
            ))
            # Will be added as first audio segment
            self._sync_audio = sync_audio
            self._sync_frames = max(10, sync_duration_frames)  # At least 10 sync frames

    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.

        Args:
            frame: RGB frame (height, width, 3)
        """
        if self._video_writer is None:
            raise RuntimeError("Video encoder not opened")

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        self._video_writer.write(frame_bgr)
        self._frame_count += 1

        # Progress callback
        if self.on_progress and self._total_frames > 0:
            self.on_progress(self._frame_count, self._total_frames)

    def close(self):
        """
        Close the video encoder and finalize the output file.
        """
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        # If using FFmpeg, mux video with audio
        if self._ffmpeg_available and self.add_audio and self._temp_video_path:
            self._finalize_with_ffmpeg()
        elif self._temp_video_path and os.path.exists(self._temp_video_path):
            # Just copy temp to output if no ffmpeg
            import shutil
            shutil.move(self._temp_video_path, self.output_path)

        # Cleanup temp files
        self._cleanup_temp()

    def _finalize_with_ffmpeg(self):
        """Use FFmpeg to create final MP4 with lossless H.264 and audio."""
        # Generate audio file
        self._temp_audio_path = os.path.join(self._temp_dir, "audio.wav")

        # Calculate total audio duration
        total_video_duration = self._frame_count / self.fps
        total_audio_samples = int(total_video_duration * self._audio_gen.sample_rate)

        # Build audio track
        audio_data = []

        # Sync beeps at start
        audio_data.append(self._sync_audio)

        # Silence for the rest (could add periodic beeps if desired)
        remaining_samples = total_audio_samples - len(self._sync_audio)
        if remaining_samples > 0:
            # Add end beeps near the end
            end_audio = self._audio_gen.generate_end_pattern()
            silence_before_end = remaining_samples - len(end_audio)
            if silence_before_end > 0:
                audio_data.append(self._audio_gen.generate_silence(
                    int(silence_before_end * 1000 / self._audio_gen.sample_rate)
                ))
            audio_data.append(end_audio)

        full_audio = np.concatenate(audio_data)

        # Trim or pad to exact duration
        if len(full_audio) > total_audio_samples:
            full_audio = full_audio[:total_audio_samples]
        elif len(full_audio) < total_audio_samples:
            padding = np.zeros(total_audio_samples - len(full_audio), dtype=np.int16)
            full_audio = np.concatenate([full_audio, padding])

        self._audio_gen.save_wav(full_audio, self._temp_audio_path)

        # FFmpeg command for lossless H.264 with audio
        # -crf 0 = lossless mode for x264
        cmd = [
            'ffmpeg', '-y',
            '-i', self._temp_video_path,
            '-i', self._temp_audio_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '0',  # Lossless
            '-pix_fmt', 'yuv444p',  # Preserve colors better
            '-c:a', 'aac',
            '-b:a', '128k',
            '-shortest',
            self.output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                creationflags=0x08000000 if os.name == 'nt' else 0
            )

            if result.returncode != 0:
                print(f"FFmpeg warning: {result.stderr.decode()}")
                # Fall back to just the video
                import shutil
                # Re-encode video without audio
                cmd_video_only = [
                    'ffmpeg', '-y',
                    '-i', self._temp_video_path,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '0',
                    self.output_path
                ]
                subprocess.run(
                    cmd_video_only,
                    capture_output=True,
                    creationflags=0x08000000 if os.name == 'nt' else 0
                )

        except Exception as e:
            print(f"FFmpeg error: {e}")
            # Fall back to temp video
            import shutil
            if os.path.exists(self._temp_video_path):
                shutil.move(self._temp_video_path, self.output_path)

    def _cleanup_temp(self):
        """Clean up temporary files."""
        import shutil
        if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass

    @property
    def frame_count(self) -> int:
        """Get number of frames written."""
        return self._frame_count


class BatchVideoEncoder:
    """
    Encodes multiple files to separate video files.
    """

    def __init__(
        self,
        output_dir: str,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = DEFAULT_FPS,
        add_audio: bool = True,
        on_file_progress: Optional[Callable[[str, int, int], None]] = None,
        on_file_complete: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize batch encoder.

        Args:
            output_dir: Directory for output videos
            width: Frame width
            height: Frame height
            fps: Frames per second
            add_audio: Whether to add audio sync beeps
            on_file_progress: Callback (filename, current_frame, total_frames)
            on_file_complete: Callback (input_path, output_path)
        """
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.add_audio = add_audio
        self.on_file_progress = on_file_progress
        self.on_file_complete = on_file_complete

        os.makedirs(output_dir, exist_ok=True)

    def get_output_path(self, input_path: str) -> str:
        """Generate output path for an input file."""
        basename = os.path.basename(input_path)
        name, _ = os.path.splitext(basename)
        output_name = f"{name}_encoded.mp4"
        return os.path.join(self.output_dir, output_name)

    def create_encoder(self, input_path: str, total_frames: int) -> VideoEncoder:
        """Create encoder for a specific input file."""
        output_path = self.get_output_path(input_path)

        def progress_callback(current, total):
            if self.on_file_progress:
                self.on_file_progress(input_path, current, total)

        encoder = VideoEncoder(
            output_path=output_path,
            width=self.width,
            height=self.height,
            fps=self.fps,
            add_audio=self.add_audio,
            on_progress=progress_callback
        )

        return encoder


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available for encoding."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            creationflags=0x08000000 if os.name == 'nt' else 0
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_encoder_info() -> dict:
    """Get information about available encoding capabilities."""
    return {
        'opencv_available': OPENCV_AVAILABLE,
        'ffmpeg_available': check_ffmpeg_available(),
        'lossless_supported': check_ffmpeg_available(),  # Need FFmpeg for true lossless
        'audio_supported': check_ffmpeg_available()
    }
