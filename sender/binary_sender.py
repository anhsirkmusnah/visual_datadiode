"""
Visual Data Diode - Binary Pixel Sender

Real-time 1-bit pixel-level encoding. Streams file data as binary frames
directly to an HDMI-connected display via pygame.

Throughput: ~235 KB/frame * 60 fps = ~13.6 MB/s net (1x repeat).

Usage:
    python -m sender.binary_sender <file> [--display 1] [--fps 60] [--repeat 1]
"""

import os
import sys
import time
import struct
import random
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.binary_frame import (
    BinaryFrameHeader, BINARY_MAGIC, BINARY_FRAME_BYTES,
    BINARY_FLAG_FIRST, BINARY_FLAG_LAST, BINARY_FLAG_METADATA,
    BINARY_FLAG_ENCRYPTED,
    calculate_binary_payload_capacity, encode_binary_frame,
)
from shared.fec import SimpleFEC
from shared.crypto import compute_file_hash
from shared.constants import FRAME_WIDTH, FRAME_HEIGHT
from sender.timing import FrameTimer


def setup_display(display_index: int):
    """
    Setup pygame on the specified display (borderless fullscreen).
    Uses DPI-awareness and SDL_VIDEO_WINDOW_POS for correct placement.
    """
    import pygame
    import ctypes

    # DPI awareness
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    pygame.init()

    num = pygame.display.get_num_displays()
    sizes = pygame.display.get_desktop_sizes()

    print(f"  Displays detected: {num}")
    for i, (w, h) in enumerate(sizes):
        tag = " <-- TARGET" if i == display_index else ""
        print(f"    [{i}] {w}x{h}{tag}")

    if display_index >= num:
        display_index = num - 1
        print(f"  Warning: display index clamped to {display_index}")

    # Compute X offset from left edge
    x_offset = sum(sizes[i][0] for i in range(display_index))
    tw, th = sizes[display_index]

    # Also try Win32 monitor coordinates
    try:
        from ctypes import wintypes
        monitors = []
        def _cb(hMon, hdc, lprc, data):
            r = lprc.contents
            monitors.append({'left': r.left, 'top': r.top,
                           'width': r.right - r.left, 'height': r.bottom - r.top})
            return True
        MEPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(wintypes.RECT),
                                     ctypes.POINTER(ctypes.c_int))
        ctypes.windll.user32.EnumDisplayMonitors(None, None, MEPROC(_cb), 0)
        if display_index < len(monitors):
            x_offset = monitors[display_index]['left']
            y_offset = monitors[display_index]['top']
            tw = monitors[display_index]['width']
            th = monitors[display_index]['height']
        else:
            y_offset = 0
    except Exception:
        y_offset = 0

    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},{y_offset}"
    print(f"  SDL_VIDEO_WINDOW_POS = {x_offset},{y_offset}")

    pygame.quit()
    pygame.init()

    screen = pygame.display.set_mode((tw, th), pygame.NOFRAME)
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Binary Data Diode Sender")

    # Verify
    print(f"  Window created: {pygame.display.get_window_size()}")
    return screen, pygame


class BinarySender:
    """
    Real-time binary pixel sender.

    Reads a file, splits into blocks, and streams each block as a
    1-bit-per-pixel frame to the display.
    """

    def __init__(
        self,
        filepath: str = None,
        display_index: int = 1,
        fps: int = 60,
        repeat_count: int = 2,
        fec_ratio: float = 0.10,
        calibration_secs: float = 5.0,
        end_secs: float = 1.0,
        passes: int = 3,
        on_progress=None,
    ):
        self.display_index = display_index
        self.fps = fps
        self.repeat_count = repeat_count
        self.fec_ratio = fec_ratio
        self.calibration_secs = calibration_secs
        self.end_secs = end_secs
        self.passes = passes  # number of complete loops through data (0=infinite)
        self.on_progress = on_progress
        self._running = True

        # FEC setup
        self.fec = SimpleFEC(fec_ratio) if fec_ratio > 0.001 else None

        # Payload capacity
        self.payload_capacity = calculate_binary_payload_capacity(fec_ratio)

        # Prepare file if provided at init time
        if filepath is not None:
            self._prepare_file(filepath)
        else:
            self.filepath = None
            self.file_size = 0
            self.filename = ''
            self.metadata = b''
            self.block0_capacity = 0
            self.total_blocks = 0
            self.session_id = 0

    def _prepare_file(self, filepath: str):
        """Prepare internal state for a single file."""
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.filename = os.path.basename(filepath)

        # Session
        self.session_id = random.randint(0, 0xFFFFFFFF)

        # Block 0 metadata: SHA-256 (32B) + filename_len (2B) + filename
        filename_bytes = self.filename.encode('utf-8')
        self.metadata = struct.pack('<32sH', b'\x00' * 32, len(filename_bytes)) + filename_bytes

        self.block0_capacity = self.payload_capacity - len(self.metadata)
        if self.block0_capacity < 0:
            raise ValueError(f"Metadata ({len(self.metadata)}B) exceeds payload capacity ({self.payload_capacity}B)")

        # Total blocks
        if self.file_size <= self.block0_capacity:
            self.total_blocks = 1
        else:
            remaining = self.file_size - self.block0_capacity
            self.total_blocks = 1 + (remaining + self.payload_capacity - 1) // self.payload_capacity

    def stop(self):
        """Signal the sender to stop after the current frame."""
        self._running = False

    def run(self):
        """Execute the full send sequence for a single file."""
        if self.filepath is None:
            raise ValueError("No file prepared. Pass filepath to __init__ or call _prepare_file() first.")

        self._running = True

        print(f"\n{'='*70}")
        print(f"  BINARY PIXEL SENDER")
        print(f"{'='*70}")
        print(f"  File: {self.filepath}")
        print(f"  Size: {self.file_size:,} bytes ({self.file_size / (1024*1024):.1f} MB)")
        print(f"  Session: 0x{self.session_id:08X}")
        print(f"  FPS: {self.fps}, Repeat: {self.repeat_count}x, Passes: {self.passes}")
        print(f"  FEC ratio: {self.fec_ratio:.0%}")
        print(f"  Payload/frame: {self.payload_capacity:,} bytes")
        print(f"  Total blocks: {self.total_blocks:,}")
        total_frames = self.total_blocks * self.repeat_count * max(self.passes, 1)
        est_time = total_frames / self.fps
        print(f"  Total frames: {total_frames:,} ({est_time:.1f}s per pass)")
        net_mbps = self.payload_capacity * self.fps / self.repeat_count / (1024 * 1024)
        print(f"  Net throughput: {net_mbps:.1f} MB/s")

        # Compute SHA-256
        print(f"\n  Computing SHA-256...")
        file_hash = compute_file_hash(self.filepath)
        print(f"  SHA-256: {file_hash.hex()}")

        # Update metadata with real hash
        filename_bytes = self.filename.encode('utf-8')
        self.metadata = file_hash + struct.pack('<H', len(filename_bytes)) + filename_bytes

        # Setup display
        print(f"\n  Setting up display {self.display_index}...")
        screen, pg = setup_display(self.display_index)

        # Create reusable surface
        surface = pg.Surface((FRAME_WIDTH, FRAME_HEIGHT))

        timer = FrameTimer(self.fps)
        timer.start()

        try:
            # Phase 1: Calibration
            self._send_calibration(screen, pg, surface, timer)

            # Phase 2: Data frames (loop for multiple passes)
            pass_num = 0
            while self._running:
                pass_num += 1
                if self.passes > 0 and pass_num > self.passes:
                    break

                if pass_num > 1:
                    print(f"\n  --- Pass {pass_num}/{self.passes if self.passes > 0 else '∞'} ---")

                self._send_data(screen, pg, surface, timer, file_hash)

            # Phase 3: End frames
            self._send_end(screen, pg, surface, timer)

            print(f"\n  TRANSMISSION COMPLETE ({pass_num} passes)")
            print(f"  Timing: {timer.stats.frames_sent} frames in {timer.stats.total_time:.1f}s")
            print(f"  Actual FPS: {timer.stats.actual_fps:.1f}")
            if timer.stats.late_frames > 0:
                print(f"  Late frames: {timer.stats.late_frames}")

        except KeyboardInterrupt:
            print(f"\n  Interrupted by user.")
        finally:
            pg.quit()

    def _render_frame(self, screen, pg, surface, frame_array: np.ndarray):
        """Blit a (1080, 1920) uint8 array to the display."""
        # pygame expects (width, height, 3) for surfarray
        rgb = np.stack([frame_array] * 3, axis=-1)
        pg.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
        screen.blit(surface, (0, 0))
        pg.display.flip()
        pg.event.pump()

    def _send_calibration(self, screen, pg, surface, timer: FrameTimer):
        """Send calibration frames: white, black, checkerboard, data-density pattern."""
        n_frames = int(self.calibration_secs * self.fps)
        phase_frames = max(1, n_frames // 4)

        print(f"\n  Phase 1: Calibration ({n_frames} frames, {self.calibration_secs:.1f}s)")

        # White
        white = np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8)
        for _ in range(phase_frames):
            if not self._running:
                return
            self._render_frame(screen, pg, surface, white)
            timer.wait_for_next_frame()

        # Black
        black = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        for _ in range(phase_frames):
            if not self._running:
                return
            self._render_frame(screen, pg, surface, black)
            timer.wait_for_next_frame()

        # Checkerboard (alternating pixels)
        checker = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        checker[0::2, 0::2] = 255
        checker[1::2, 1::2] = 255
        for _ in range(phase_frames):
            if not self._running:
                return
            self._render_frame(screen, pg, surface, checker)
            timer.wait_for_next_frame()

        # Data-density pattern: alternating half-black/half-white rows
        # Simulates ~50% bit density of real binary data to let AGC settle
        data_pattern = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        data_pattern[0::2, :] = 255  # even rows white, odd rows black
        for i in range(phase_frames):
            if not self._running:
                return
            # Alternate between two patterns every few frames for variety
            if (i // 4) % 2 == 1:
                data_pattern_inv = 255 - data_pattern
                self._render_frame(screen, pg, surface, data_pattern_inv)
            else:
                self._render_frame(screen, pg, surface, data_pattern)
            timer.wait_for_next_frame()

    def _send_data(self, screen, pg, surface, timer: FrameTimer, file_hash: bytes):
        """Stream all data blocks (single pass)."""
        print(f"\n  Data transmission ({self.total_blocks} blocks)")

        start_time = time.perf_counter()

        with open(self.filepath, 'rb') as f:
            for block_idx in range(self.total_blocks):
                if not self._running:
                    return

                # Build payload
                if block_idx == 0:
                    # Block 0: metadata + file data
                    file_data = f.read(self.block0_capacity)
                    payload = self.metadata + file_data
                else:
                    payload = f.read(self.payload_capacity)

                # Build header
                flags = 0
                if block_idx == 0:
                    flags |= BINARY_FLAG_FIRST | BINARY_FLAG_METADATA
                if block_idx == self.total_blocks - 1:
                    flags |= BINARY_FLAG_LAST

                header = BinaryFrameHeader(
                    magic=BINARY_MAGIC,
                    session_id=self.session_id,
                    block_index=block_idx,
                    total_blocks=self.total_blocks,
                    file_size=self.file_size,
                    payload_size=len(payload),
                    flags=flags,
                    fec_nsym=self.fec.nsym if self.fec else 0,
                )

                # Encode to pixel frame
                frame = encode_binary_frame(header, payload, self.fec)

                # Display with repeat
                for rep in range(self.repeat_count):
                    if not self._running:
                        return
                    self._render_frame(screen, pg, surface, frame)
                    timer.wait_for_next_frame()

                # Progress
                elapsed = time.perf_counter() - start_time
                bytes_sent = min((block_idx + 1) * self.payload_capacity, self.file_size)
                mbps = bytes_sent / elapsed / (1024 * 1024) if elapsed > 0 else 0
                eta = (elapsed / (block_idx + 1)) * (self.total_blocks - block_idx - 1)

                if self.on_progress:
                    self.on_progress(block_idx, self.total_blocks, bytes_sent, elapsed, self.filename)

                if (block_idx + 1) % 100 == 0 or block_idx == self.total_blocks - 1:
                    pct = (block_idx + 1) / self.total_blocks * 100
                    print(f"    Block {block_idx+1}/{self.total_blocks} ({pct:.1f}%) "
                          f"- {mbps:.1f} MB/s - ETA {eta:.0f}s", end='\r')

        print()  # newline after progress

    def _send_end(self, screen, pg, surface, timer: FrameTimer):
        """Send end-of-transmission frames (all black)."""
        n_frames = int(self.end_secs * self.fps)
        print(f"\n  Phase 3: End frames ({n_frames} frames)")

        black = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        for _ in range(n_frames):
            if not self._running:
                return
            self._render_frame(screen, pg, surface, black)
            timer.wait_for_next_frame()

    def run_queue(self, file_paths, display_index=None, fps=None, repeat_count=None,
                  fec_ratio=None, calibration_secs=None):
        """
        Stream multiple files sequentially as separate sessions.

        Each file gets its own session ID and calibration phase.
        The display is set up once and reused across files.
        """
        if display_index is not None:
            self.display_index = display_index
        if fps is not None:
            self.fps = fps
        if repeat_count is not None:
            self.repeat_count = repeat_count
        if fec_ratio is not None:
            self.fec_ratio = fec_ratio
            self.fec = SimpleFEC(fec_ratio) if fec_ratio > 0.001 else None
            self.payload_capacity = calculate_binary_payload_capacity(fec_ratio)
        if calibration_secs is not None:
            self.calibration_secs = calibration_secs

        self._running = True

        # Setup display once
        print(f"\n  Setting up display {self.display_index}...")
        screen, pg = setup_display(self.display_index)
        surface = pg.Surface((FRAME_WIDTH, FRAME_HEIGHT))

        try:
            for file_idx, fpath in enumerate(file_paths):
                if not self._running:
                    break

                print(f"\n{'='*70}")
                print(f"  FILE {file_idx + 1}/{len(file_paths)}: {os.path.basename(fpath)}")
                print(f"{'='*70}")

                self._prepare_file(fpath)

                # Compute SHA-256
                print(f"  Computing SHA-256...")
                file_hash = compute_file_hash(self.filepath)
                print(f"  SHA-256: {file_hash.hex()}")

                # Update metadata with real hash
                filename_bytes = self.filename.encode('utf-8')
                self.metadata = file_hash + struct.pack('<H', len(filename_bytes)) + filename_bytes

                timer = FrameTimer(self.fps)
                timer.start()

                # Calibration
                self._send_calibration(screen, pg, surface, timer)
                if not self._running:
                    break

                # Data (loop for multiple passes)
                pass_num = 0
                while self._running:
                    pass_num += 1
                    if self.passes > 0 and pass_num > self.passes:
                        break
                    if pass_num > 1:
                        print(f"\n  --- Pass {pass_num}/{self.passes if self.passes > 0 else '∞'} ---")
                    self._send_data(screen, pg, surface, timer, file_hash)

                if not self._running:
                    break

                # End
                self._send_end(screen, pg, surface, timer)

                print(f"\n  File complete: {self.filename} ({pass_num} passes)")
                print(f"  Timing: {timer.stats.frames_sent} frames, "
                      f"actual FPS: {timer.stats.actual_fps:.1f}")

            print(f"\n  ALL FILES TRANSMITTED")

        except KeyboardInterrupt:
            print(f"\n  Interrupted by user.")
        finally:
            pg.quit()


def main():
    parser = argparse.ArgumentParser(
        description='Binary Pixel Sender - Real-time 1-bit/pixel HDMI streaming'
    )
    parser.add_argument('file', help='File to send')
    parser.add_argument('--display', type=int, default=1,
                        help='Display index for output (default: 1)')
    parser.add_argument('--fps', type=int, default=60,
                        help='Target FPS (default: 60)')
    parser.add_argument('--repeat', type=int, default=2,
                        help='Repeat count per frame (default: 2)')
    parser.add_argument('--fec', type=float, default=0.10,
                        help='FEC ratio (default: 0.10)')
    parser.add_argument('--calibration', type=float, default=5.0,
                        help='Calibration duration in seconds (default: 5.0)')
    parser.add_argument('--passes', type=int, default=3,
                        help='Number of complete data passes (default: 3, 0=infinite)')
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    sender = BinarySender(
        filepath=args.file,
        display_index=args.display,
        fps=args.fps,
        repeat_count=args.repeat,
        fec_ratio=args.fec,
        calibration_secs=args.calibration,
        passes=args.passes,
    )
    sender.run()


if __name__ == '__main__':
    main()
