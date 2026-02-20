"""
Visual Data Diode - Binary Pixel Receiver

Real-time capture and decode of 1-bit pixel-level binary frames
from a USB HDMI capture device.

Uses:
- Capture thread with MSMF (preferred) or DirectShow @ 1920x1080 @ 60fps
- Disk-backed block store to handle files up to 100GB+
- Deduplication by block index
- SHA-256 verification on assembly

Usage:
    python -m receiver.binary_receiver [--output ./received] [--device 2]
"""

import os
import sys
import time
import struct
import hashlib
import shutil
import tempfile
import threading
import queue
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from shared.binary_frame import (
    BinaryFrameHeader, BINARY_MAGIC, BINARY_FRAME_BYTES,
    BINARY_HEADER_SIZE, BINARY_CRC_SIZE,
    BINARY_FLAG_FIRST, BINARY_FLAG_LAST, BINARY_FLAG_METADATA,
    decode_binary_frame, binary_frame_to_bytes,
)
from shared.fec import SimpleFEC
from shared.constants import FRAME_WIDTH, FRAME_HEIGHT


class BinaryCapture:
    """
    Lightweight capture thread for USB HDMI capture device.

    Tries MSMF backend first (async, faster, doesn't hang), falls back
    to DirectShow. Converts to grayscale and puts frames in a bounded
    queue. Drops oldest on overflow to maintain real-time priority.
    """

    def __init__(self, device_index: int = 2, queue_size: int = 4):
        self.device_index = device_index
        self.queue_size = queue_size
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._thread = None
        self._cap = None

        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0

    def start(self) -> bool:
        """Open capture device and start capture thread."""
        # Try MSMF first (async, faster, doesn't hang on many devices)
        print(f"  Trying MSMF backend for device {self.device_index}...")
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
        if not self._cap.isOpened():
            print(f"  MSMF failed, trying DirectShow...")
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            print(f"  Error: Cannot open capture device {self.device_index} with any backend")
            return False

        # Set MJPG fourcc FIRST — MSMF requires this to negotiate 1080p
        # with many USB capture cards (YUY2/NV12 often fail to set resolution)
        self._cap.set(cv2.CAP_PROP_FOURCC,
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Verify settings
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fmt = ''.join([chr((fourcc >> 8*i) & 0xFF)
                       for i in range(4) if 32 <= ((fourcc >> 8*i) & 0xFF) < 127])
        backend_name = self._cap.getBackendName() if hasattr(self._cap, 'getBackendName') else 'unknown'
        print(f"  Capture: {w}x{h} @ {fps:.0f}fps, format={fmt}, backend={backend_name}")

        if w != FRAME_WIDTH or h != FRAME_HEIGHT:
            print(f"  FATAL: Resolution {w}x{h}, need {FRAME_WIDTH}x{FRAME_HEIGHT}")
            self._cap.release()
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop capture thread and release device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()

    def get_frame(self, timeout: float = 0.1) -> np.ndarray:
        """Get next grayscale frame from queue. Returns None on timeout."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self):
        """Background thread: read frames, convert to gray, enqueue."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue

            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            self.frames_captured += 1

            # Put in queue, drop oldest if full
            try:
                self.frame_queue.put_nowait(gray)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # drop oldest
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(gray)
                except queue.Full:
                    pass


class DiskBackedBlockStore:
    """
    Stores received block payloads on disk to avoid RAM exhaustion
    for large transfers (e.g., 100GB files).

    Each block is written to a separate temp file. Only the index
    set is kept in memory.
    """

    def __init__(self, temp_dir: str = None):
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix='vdd_blocks_')
        else:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)

        self.received_indices = set()
        self._lock = threading.Lock()

    def store(self, block_index: int, payload: bytes):
        """Store a block payload to disk."""
        with self._lock:
            if block_index in self.received_indices:
                return  # Already have it

            path = os.path.join(self.temp_dir, f"blk_{block_index:08d}.bin")
            with open(path, 'wb') as f:
                f.write(payload)
            self.received_indices.add(block_index)

    def get(self, block_index: int) -> bytes:
        """Read a block payload from disk."""
        path = os.path.join(self.temp_dir, f"blk_{block_index:08d}.bin")
        with open(path, 'rb') as f:
            return f.read()

    def has(self, block_index: int) -> bool:
        """Check if block is already stored."""
        return block_index in self.received_indices

    @property
    def count(self) -> int:
        return len(self.received_indices)

    def cleanup(self):
        """Remove temp directory and all block files."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass


class BinaryReceiver:
    """
    Real-time binary pixel receiver.

    Captures frames from USB HDMI device, decodes binary frames,
    deduplicates, assembles, and verifies SHA-256.
    """

    def __init__(
        self,
        output_dir: str = '.',
        device_index: int = 2,
        fec_ratio: float = 0.10,
        idle_timeout: float = 5.0,
        on_progress=None,
        on_file_complete=None,
    ):
        self.output_dir = output_dir
        self.device_index = device_index
        self.fec_ratio = fec_ratio
        self.idle_timeout = idle_timeout
        self.on_progress = on_progress
        self.on_file_complete = on_file_complete
        self._running = True

        os.makedirs(output_dir, exist_ok=True)

        # FEC
        self.fec = SimpleFEC(fec_ratio) if fec_ratio > 0.001 else None

        # State
        self.session_id = None
        self.total_blocks = None
        self.file_size = None
        self.filename = None
        self.file_hash = None
        self.store = None

        # Stats
        self.frames_decoded = 0
        self.frames_failed = 0
        self.fec_corrections = 0
        self.duplicates = 0

    def stop(self):
        """Signal the receiver to stop after the current frame."""
        self._running = False

    def run(self):
        """Main receive loop."""
        self._running = True

        print(f"\n{'='*70}")
        print(f"  BINARY PIXEL RECEIVER")
        print(f"{'='*70}")
        print(f"  Output: {os.path.abspath(self.output_dir)}")
        print(f"  Device: {self.device_index}")
        print(f"  FEC ratio: {self.fec_ratio:.0%}")
        print(f"  Idle timeout: {self.idle_timeout}s")

        # Start capture
        capture = BinaryCapture(device_index=self.device_index)
        if not capture.start():
            print("  FATAL: Cannot start capture.")
            return False

        print(f"\n  Waiting for data frames...")
        print(f"  (Press Ctrl+C to stop)\n")

        self.store = DiskBackedBlockStore()
        last_block_time = None
        start_time = None

        diag_count = 0
        diag_interval = 120  # print diagnostics every N frames

        try:
            while self._running:
                gray = capture.get_frame(timeout=0.5)

                if gray is None:
                    # Check idle timeout
                    if last_block_time is not None:
                        idle = time.time() - last_block_time
                        if idle > self.idle_timeout:
                            print(f"\n  Idle timeout ({self.idle_timeout}s) - assembling...")
                            break
                    # Fire progress callback even when waiting
                    if self.on_progress:
                        self._fire_progress(start_time, 'waiting' if self.session_id is None else 'receiving')
                    continue

                # Decode
                header, payload, stats = decode_binary_frame(gray, self.fec)

                # Periodic diagnostics when no valid frames detected yet
                diag_count += 1
                if self.session_id is None and diag_count % diag_interval == 0:
                    mn, mx = int(gray.min()), int(gray.max())
                    mean = float(gray.mean())
                    # Check how binary the image is (pixels near 0 or 255)
                    near_black = int(np.sum(gray < 64))
                    near_white = int(np.sum(gray > 192))
                    total_px = gray.size
                    binary_pct = (near_black + near_white) / total_px * 100
                    # Check first 32 bytes after thresholding
                    raw_bytes = binary_frame_to_bytes(gray)
                    magic_hex = raw_bytes[:4].hex()
                    print(f"  [diag] frame#{diag_count}: min={mn} max={mx} mean={mean:.0f} "
                          f"binary={binary_pct:.0f}% magic=0x{magic_hex} "
                          f"decoded={self.frames_decoded} failed={self.frames_failed}")

                if header is None or not stats.valid_header:
                    self.frames_failed += 1
                    continue

                if payload is None:
                    # Header OK but CRC/FEC failed
                    self.frames_failed += 1
                    if stats.fec_failed:
                        pass  # FEC couldn't correct
                    continue

                self.frames_decoded += 1
                if stats.fec_corrected > 0:
                    self.fec_corrections += stats.fec_corrected

                # First valid frame - capture session info
                if self.session_id is None:
                    self.session_id = header.session_id
                    self.total_blocks = header.total_blocks
                    self.file_size = header.file_size
                    start_time = time.time()
                    print(f"  Session: 0x{header.session_id:08X}")
                    print(f"  File size: {header.file_size:,} bytes")
                    print(f"  Total blocks: {header.total_blocks:,}")

                    # Update FEC if header specifies different nsym
                    if header.fec_nsym > 0 and self.fec is not None:
                        if header.fec_nsym != self.fec.nsym:
                            self.fec = SimpleFEC(self.fec_ratio)

                # Session mismatch - ignore
                if header.session_id != self.session_id:
                    continue

                # Deduplicate
                if self.store.has(header.block_index):
                    self.duplicates += 1
                    continue

                # Extract metadata from block 0
                if header.block_index == 0 and (header.flags & BINARY_FLAG_METADATA):
                    self._extract_metadata(payload)

                # Store block
                self.store.store(header.block_index, payload)
                last_block_time = time.time()

                # Progress
                received = self.store.count
                if self.on_progress:
                    self._fire_progress(start_time, 'receiving')

                if received % 50 == 0 or received == self.total_blocks:
                    pct = received / self.total_blocks * 100 if self.total_blocks else 0
                    elapsed = time.time() - start_time if start_time else 0
                    print(f"    Blocks: {received}/{self.total_blocks} ({pct:.1f}%) "
                          f"- decoded: {self.frames_decoded}, failed: {self.frames_failed}, "
                          f"dupes: {self.duplicates}, FEC: {self.fec_corrections}",
                          end='\r')

                # All blocks received?
                if self.total_blocks and self.store.count >= self.total_blocks:
                    print(f"\n\n  All {self.total_blocks} blocks received!")
                    break

        except KeyboardInterrupt:
            print(f"\n\n  Interrupted by user.")

        finally:
            capture.stop()

        # Print stats
        elapsed = time.time() - start_time if start_time else 0
        print(f"\n  Capture stats:")
        print(f"    Frames captured: {capture.frames_captured:,}")
        print(f"    Frames dropped: {capture.frames_dropped:,}")
        print(f"    Frames decoded: {self.frames_decoded:,}")
        print(f"    Decode failures: {self.frames_failed:,}")
        print(f"    FEC corrections: {self.fec_corrections:,}")
        print(f"    Duplicates: {self.duplicates:,}")
        print(f"    Blocks received: {self.store.count}/{self.total_blocks or '?'}")
        if elapsed > 0 and self.file_size:
            effective_bytes = min(self.store.count * (BINARY_FRAME_BYTES - 36), self.file_size)
            mbps = effective_bytes / elapsed / (1024 * 1024)
            print(f"    Elapsed: {elapsed:.1f}s")
            print(f"    Effective throughput: {mbps:.1f} MB/s")

        # Assemble if we have enough
        if self.on_progress:
            self._fire_progress(start_time, 'assembling')

        if self.total_blocks and self.store.count > 0:
            result = self._assemble()
            self.store.cleanup()
            return result

        self.store.cleanup()
        return False

    def _fire_progress(self, start_time, state: str):
        """Fire the on_progress callback with current stats."""
        if not self.on_progress:
            return
        elapsed = time.time() - start_time if start_time else 0
        received = self.store.count if self.store else 0
        mbps = 0
        if elapsed > 0 and self.file_size and received > 0:
            effective_bytes = min(received * (BINARY_FRAME_BYTES - 36), self.file_size)
            mbps = effective_bytes / elapsed / (1024 * 1024)
        self.on_progress({
            'blocks_received': received,
            'total_blocks': self.total_blocks or 0,
            'frames_decoded': self.frames_decoded,
            'frames_failed': self.frames_failed,
            'fec_corrections': self.fec_corrections,
            'duplicates': self.duplicates,
            'mbps': mbps,
            'elapsed': elapsed,
            'state': state,
            'filename': self.filename,
            'file_size': self.file_size,
        })

    def _extract_metadata(self, payload: bytes):
        """Extract file hash, filename from block 0 metadata."""
        if len(payload) < 34:  # 32 hash + 2 filename_len
            return

        self.file_hash = payload[:32]
        filename_len = struct.unpack('<H', payload[32:34])[0]
        if len(payload) >= 34 + filename_len:
            self.filename = payload[34:34 + filename_len].decode('utf-8', errors='replace')
            print(f"\n  Filename: {self.filename}")
            print(f"  Expected SHA-256: {self.file_hash.hex()}")

    def _assemble(self) -> bool:
        """Assemble all received blocks into the output file."""
        if self.total_blocks is None:
            print("  Error: No session info available.")
            return False

        missing = []
        for i in range(self.total_blocks):
            if not self.store.has(i):
                missing.append(i)

        if missing:
            print(f"\n  WARNING: Missing {len(missing)} blocks: {missing[:20]}{'...' if len(missing) > 20 else ''}")
            if len(missing) > self.total_blocks * 0.1:
                print(f"  Too many missing blocks ({len(missing)}/{self.total_blocks}). Aborting assembly.")
                return False
            print(f"  Assembling partial file (missing blocks will be zero-filled).")

        # Determine output filename
        if self.filename:
            out_path = os.path.join(self.output_dir, self.filename)
        else:
            out_path = os.path.join(self.output_dir, f"received_0x{self.session_id:08X}.bin")

        # Avoid overwriting
        base, ext = os.path.splitext(out_path)
        counter = 1
        while os.path.exists(out_path):
            out_path = f"{base}_{counter}{ext}"
            counter += 1

        print(f"\n  Assembling to: {out_path}")

        sha256 = hashlib.sha256()
        bytes_written = 0

        with open(out_path, 'wb') as out_f:
            for block_idx in range(self.total_blocks):
                if self.store.has(block_idx):
                    payload = self.store.get(block_idx)
                else:
                    # Missing block - fill with zeros
                    if block_idx == 0:
                        # Can't reconstruct metadata block
                        payload = b'\x00' * (self.file_size if self.file_size < 1024 else 1024)
                    else:
                        payload = b'\x00' * 1024  # placeholder

                if block_idx == 0:
                    # Strip metadata prefix from block 0
                    if self.filename:
                        filename_bytes = self.filename.encode('utf-8')
                        meta_size = 32 + 2 + len(filename_bytes)
                        file_data = payload[meta_size:]
                    else:
                        file_data = payload
                else:
                    file_data = payload

                # Truncate to file_size
                remaining = self.file_size - bytes_written if self.file_size else len(file_data)
                if remaining <= 0:
                    break
                write_data = file_data[:remaining]
                out_f.write(write_data)
                sha256.update(write_data)
                bytes_written += len(write_data)

        print(f"  Wrote {bytes_written:,} bytes")

        # Verify SHA-256
        computed_hash = sha256.digest()
        hash_valid = None
        if self.file_hash:
            if computed_hash == self.file_hash:
                print(f"  SHA-256 VERIFIED: {computed_hash.hex()}")
                hash_valid = True
            else:
                print(f"  SHA-256 MISMATCH!")
                print(f"    Expected: {self.file_hash.hex()}")
                print(f"    Computed: {computed_hash.hex()}")
                hash_valid = False
        else:
            print(f"  SHA-256: {computed_hash.hex()} (no reference to verify against)")
            hash_valid = None

        # Fire file complete callback
        if self.on_file_complete:
            self.on_file_complete({
                'filename': self.filename or os.path.basename(out_path),
                'size': bytes_written,
                'hash_valid': hash_valid,
                'path': out_path,
                'blocks_received': self.store.count,
                'total_blocks': self.total_blocks,
            })

        return hash_valid is not False


def main():
    parser = argparse.ArgumentParser(
        description='Binary Pixel Receiver - Real-time 1-bit/pixel HDMI capture'
    )
    parser.add_argument('--output', '-o', default='./received',
                        help='Output directory (default: ./received)')
    parser.add_argument('--device', '-d', type=int, default=2,
                        help='Capture device index (default: 2)')
    parser.add_argument('--fec', type=float, default=0.10,
                        help='FEC ratio (default: 0.10)')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='Idle timeout before assembly (default: 5.0s)')
    args = parser.parse_args()

    receiver = BinaryReceiver(
        output_dir=args.output,
        device_index=args.device,
        fec_ratio=args.fec,
        idle_timeout=args.timeout,
    )
    success = receiver.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
