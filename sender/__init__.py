"""
Visual Data Diode - Sender Module

Transmits files by encoding them as visual frames for HDMI output.
"""

from .chunker import FileChunker
from .encoder import FrameEncoder
from .renderer import FrameRenderer
from .timing import FrameTimer

__all__ = [
    'FileChunker',
    'FrameEncoder',
    'FrameRenderer',
    'FrameTimer',
]
