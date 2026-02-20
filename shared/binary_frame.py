"""
Visual Data Diode - Binary Frame Protocol

1-bit pixel-level encoding for maximum throughput through HDMI pipeline.
Each frame uses the full 1920x1080 resolution at 1 bit per pixel.

Frame byte layout:
  [header 32B][payload][CRC-32 4B][FEC parity][zero padding to 259,200B]

Header (32 bytes):
  Magic "VDBP" (4B) | Session ID (4B) | Block Index (4B) | Total Blocks (4B)
  File Size (8B, uint64) | Payload Size (4B) | Flags (1B) | FEC nsym (1B)
  Header CRC-16 (2B)
"""

import struct
import zlib
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .block import crc16
from .fec import SimpleFEC
from .constants import FRAME_WIDTH, FRAME_HEIGHT

# Protocol constants
BINARY_MAGIC = b'VDBP'
BINARY_HEADER_SIZE = 32
BINARY_CRC_SIZE = 4  # CRC-32

# Total bytes in a 1920x1080 binary frame (1 bit/pixel -> packbits)
BINARY_FRAME_BYTES = (FRAME_WIDTH * FRAME_HEIGHT) // 8  # 259,200 bytes

# Flags
BINARY_FLAG_FIRST = 0x01
BINARY_FLAG_LAST = 0x02
BINARY_FLAG_ENCRYPTED = 0x04
BINARY_FLAG_METADATA = 0x08  # Block 0 carries file metadata


@dataclass
class BinaryFrameHeader:
    """32-byte header for binary pixel frames."""
    magic: bytes = BINARY_MAGIC      # 4B
    session_id: int = 0              # 4B uint32
    block_index: int = 0             # 4B uint32
    total_blocks: int = 0            # 4B uint32
    file_size: int = 0               # 8B uint64
    payload_size: int = 0            # 4B uint32
    flags: int = 0                   # 1B uint8
    fec_nsym: int = 0                # 1B uint8
    header_crc: int = 0              # 2B uint16

    # struct format: little-endian
    # 4s = magic, I = session_id, I = block_index, I = total_blocks,
    # Q = file_size, I = payload_size, B = flags, B = fec_nsym,
    # H = header_crc
    _FORMAT = '<4sIIIQIBBH'
    _SIZE = 32

    def pack(self) -> bytes:
        """Pack header into 32 bytes. CRC is computed over first 30 bytes."""
        # Pack without CRC first
        data = struct.pack(
            self._FORMAT,
            self.magic,
            self.session_id & 0xFFFFFFFF,
            self.block_index & 0xFFFFFFFF,
            self.total_blocks & 0xFFFFFFFF,
            self.file_size & 0xFFFFFFFFFFFFFFFF,
            self.payload_size & 0xFFFFFFFF,
            self.flags & 0xFF,
            self.fec_nsym & 0xFF,
            0,  # placeholder CRC
        )
        # Compute CRC-16 over first 30 bytes
        crc = crc16(data[:30])
        # Repack with actual CRC
        return data[:30] + struct.pack('<H', crc)

    @classmethod
    def unpack(cls, data: bytes) -> Optional['BinaryFrameHeader']:
        """Unpack header from 32 bytes. Returns None on CRC mismatch."""
        if len(data) < cls._SIZE:
            return None

        fields = struct.unpack(cls._FORMAT, data[:cls._SIZE])
        magic = fields[0]

        if magic != BINARY_MAGIC:
            return None

        # Verify CRC-16
        stored_crc = fields[8]
        computed_crc = crc16(data[:30])
        if stored_crc != computed_crc:
            return None

        return cls(
            magic=magic,
            session_id=fields[1],
            block_index=fields[2],
            total_blocks=fields[3],
            file_size=fields[4],
            payload_size=fields[5],
            flags=fields[6],
            fec_nsym=fields[7],
            header_crc=stored_crc,
        )


def calculate_binary_payload_capacity(fec_ratio: float = 0.10) -> int:
    """
    Calculate net payload bytes per binary frame.

    Layout: [header 32B][payload N][CRC-32 4B][FEC parity M][zero pad]
    Total = 259,200 bytes

    FEC parity is computed over (header + payload + CRC).
    """
    overhead_fixed = BINARY_HEADER_SIZE + BINARY_CRC_SIZE  # 36 bytes
    available = BINARY_FRAME_BYTES - overhead_fixed

    if fec_ratio <= 0.001:
        return available

    # payload + fec_parity = available
    # fec_parity ~ fec_ratio * (header + payload + crc)
    # Approximate: payload = available / (1 + fec_ratio)
    # Then verify with actual FEC calculation
    fec = SimpleFEC(fec_ratio)
    # Iteratively find max payload
    payload = int(available / (1 + fec_ratio))
    while payload > 0:
        data_for_fec = BINARY_HEADER_SIZE + payload + BINARY_CRC_SIZE
        parity_size = fec.parity_size(data_for_fec)
        if payload + parity_size <= available:
            return payload
        payload -= 1

    return 0


def bytes_to_binary_frame(data: bytes) -> np.ndarray:
    """
    Convert packed bytes (259,200) to a 1080x1920 uint8 frame.

    Each bit becomes a pixel: 0 -> 0 (black), 1 -> 255 (white).
    """
    if len(data) < BINARY_FRAME_BYTES:
        # Pad with zeros
        data = data + b'\x00' * (BINARY_FRAME_BYTES - len(data))

    bits = np.unpackbits(np.frombuffer(data[:BINARY_FRAME_BYTES], dtype=np.uint8))
    # bits is (259200*8,) but we only need 1920*1080 = 2,073,600 bits
    pixels = bits[:FRAME_WIDTH * FRAME_HEIGHT]
    frame = (pixels * 255).astype(np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH)
    return frame


def binary_frame_to_bytes(frame: np.ndarray, threshold: float = None) -> bytes:
    """
    Convert a grayscale frame to packed bytes via adaptive threshold.

    If threshold is None (default), computes it from the frame's actual
    min/max pixel values to handle HDMI pipeline distortion (gamma, AGC,
    color space conversion) where black may be 15-40 and white 210-245.

    Returns exactly 259,200 bytes.
    """
    flat = frame.ravel()
    if threshold is None:
        fmin, fmax = int(flat.min()), int(flat.max())
        if fmax - fmin > 50:
            # Enough contrast — use midpoint of actual range
            threshold = (fmin + fmax) / 2.0
        else:
            threshold = 128  # fallback for blank/uniform frames
    binary = (flat >= threshold).astype(np.uint8)
    packed = np.packbits(binary)
    return bytes(packed[:BINARY_FRAME_BYTES])


@dataclass
class BinaryDecodeStats:
    """Statistics from decoding a binary frame."""
    valid_header: bool = False
    crc_ok: bool = False
    fec_corrected: int = 0
    fec_failed: bool = False


def encode_binary_frame(
    header: BinaryFrameHeader,
    payload: bytes,
    fec: Optional[SimpleFEC] = None,
) -> np.ndarray:
    """
    Encode a binary frame: header + payload + CRC-32 + FEC -> pixel frame.

    Returns:
        numpy array (1080, 1920) uint8 with 0/255 values.
    """
    header_bytes = header.pack()

    # CRC-32 over header + payload
    crc_data = header_bytes + payload
    crc32_val = zlib.crc32(crc_data) & 0xFFFFFFFF
    crc_bytes = struct.pack('<I', crc32_val)

    # FEC parity over header + payload + CRC
    fec_parity = b''
    if fec is not None:
        fec_input = header_bytes + payload + crc_bytes
        _, fec_parity = fec.encode(fec_input)

    # Assemble frame data
    frame_data = header_bytes + payload + crc_bytes + fec_parity

    # Pad to exactly BINARY_FRAME_BYTES
    if len(frame_data) < BINARY_FRAME_BYTES:
        frame_data += b'\x00' * (BINARY_FRAME_BYTES - len(frame_data))

    return bytes_to_binary_frame(frame_data)


def decode_binary_frame(
    gray_frame: np.ndarray,
    fec: Optional[SimpleFEC] = None,
) -> Tuple[Optional[BinaryFrameHeader], Optional[bytes], BinaryDecodeStats]:
    """
    Decode a binary frame from a grayscale capture.

    Args:
        gray_frame: (1080, 1920) uint8 grayscale image
        fec: Optional FEC decoder

    Returns:
        (header, payload, stats) or (None, None, stats) on failure
    """
    stats = BinaryDecodeStats()

    # Convert frame to bytes
    frame_bytes = binary_frame_to_bytes(gray_frame)

    # Try to unpack header
    header = BinaryFrameHeader.unpack(frame_bytes[:BINARY_HEADER_SIZE])
    if header is None:
        return None, None, stats
    stats.valid_header = True

    payload_size = header.payload_size
    payload_end = BINARY_HEADER_SIZE + payload_size
    crc_end = payload_end + BINARY_CRC_SIZE

    if crc_end > BINARY_FRAME_BYTES:
        return None, None, stats

    # Extract regions
    payload = frame_bytes[BINARY_HEADER_SIZE:payload_end]
    stored_crc = struct.unpack('<I', frame_bytes[payload_end:crc_end])[0]

    # Verify CRC-32
    crc_data = frame_bytes[:BINARY_HEADER_SIZE] + payload
    computed_crc = zlib.crc32(crc_data) & 0xFFFFFFFF

    if stored_crc == computed_crc:
        stats.crc_ok = True
        return header, payload, stats

    # CRC failed - try FEC correction if available
    if fec is not None and header.fec_nsym > 0:
        data_for_fec = frame_bytes[:crc_end]
        fec_parity_size = fec.parity_size(crc_end)
        fec_parity = frame_bytes[crc_end:crc_end + fec_parity_size]

        if len(fec_parity) == fec_parity_size:
            corrected, n_errors = fec.decode(data_for_fec, fec_parity)
            if corrected is not None:
                stats.fec_corrected = n_errors
                # Re-extract from corrected data
                c_header = BinaryFrameHeader.unpack(corrected[:BINARY_HEADER_SIZE])
                if c_header is not None:
                    c_payload = corrected[BINARY_HEADER_SIZE:BINARY_HEADER_SIZE + c_header.payload_size]
                    c_crc_start = BINARY_HEADER_SIZE + c_header.payload_size
                    c_stored_crc = struct.unpack('<I', corrected[c_crc_start:c_crc_start + 4])[0]
                    c_computed = zlib.crc32(corrected[:c_crc_start]) & 0xFFFFFFFF
                    if c_stored_crc == c_computed:
                        stats.crc_ok = True
                        return c_header, c_payload, stats

            stats.fec_failed = True

    return header, None, stats
