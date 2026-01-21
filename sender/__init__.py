"""
Visual Data Diode - Sender Module

Encodes files as visual frames and outputs to video files.
"""

from .chunker import FileChunker
from .encoder import FrameEncoder
from .video_encoder import VideoEncoder, BatchVideoEncoder, check_ffmpeg_available
from .timing import FrameTimer

__all__ = [
    'FileChunker',
    'FrameEncoder',
    'VideoEncoder',
    'BatchVideoEncoder',
    'check_ffmpeg_available',
    'FrameTimer',
]
