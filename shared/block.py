"""
Visual Data Diode - Block Structure

Defines the Block class for packing/unpacking transfer blocks.
"""

import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

from .constants import (
    HEADER_SIZE, CRC_SIZE, BlockFlags,
    HEADER_SESSION_ID_OFFSET, HEADER_BLOCK_INDEX_OFFSET,
    HEADER_TOTAL_BLOCKS_OFFSET, HEADER_FILE_SIZE_OFFSET,
    HEADER_PAYLOAD_SIZE_OFFSET, HEADER_FLAGS_OFFSET
)


@dataclass
class BlockHeader:
    """Block header structure."""
    session_id: int      # 4 bytes: Random session identifier
    block_index: int     # 4 bytes: 0-indexed block number
    total_blocks: int    # 4 bytes: Total blocks in transfer
    file_size: int       # 4 bytes: Total file size in bytes
    payload_size: int    # 2 bytes: Payload size in this block
    flags: BlockFlags    # 1 byte: Bit flags
    reserved: int = 0    # 1 byte: Reserved

    def pack(self) -> bytes:
        """Pack header into 20 bytes."""
        return struct.pack(
            '<IIIIHBB',
            self.session_id,
            self.block_index,
            self.total_blocks,
            self.file_size,
            self.payload_size,
            self.flags,
            self.reserved
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'BlockHeader':
        """Unpack header from 20 bytes."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header data too short: {len(data)} < {HEADER_SIZE}")

        session_id, block_index, total_blocks, file_size, payload_size, flags, reserved = \
            struct.unpack('<IIIIHBB', data[:HEADER_SIZE])

        return cls(
            session_id=session_id,
            block_index=block_index,
            total_blocks=total_blocks,
            file_size=file_size,
            payload_size=payload_size,
            flags=BlockFlags(flags),
            reserved=reserved
        )

    @property
    def is_first(self) -> bool:
        return bool(self.flags & BlockFlags.FIRST_BLOCK)

    @property
    def is_last(self) -> bool:
        return bool(self.flags & BlockFlags.LAST_BLOCK)

    @property
    def is_encrypted(self) -> bool:
        return bool(self.flags & BlockFlags.ENCRYPTED)

    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & BlockFlags.COMPRESSED)


@dataclass
class Block:
    """Complete block with header, payload, and integrity data."""
    header: BlockHeader
    payload: bytes
    crc32: int = 0
    fec_parity: bytes = b''

    def compute_crc(self) -> int:
        """Compute CRC-32 of header + payload."""
        data = self.header.pack() + self.payload
        return zlib.crc32(data) & 0xFFFFFFFF

    def pack(self, include_fec: bool = True) -> bytes:
        """
        Pack block into bytes.

        Format: [header (20)] [payload (N)] [CRC-32 (4)] [FEC parity (M)]
        """
        header_bytes = self.header.pack()
        crc = self.compute_crc()
        crc_bytes = struct.pack('<I', crc)

        result = header_bytes + self.payload + crc_bytes

        if include_fec and self.fec_parity:
            result += self.fec_parity

        return result

    @classmethod
    def unpack(cls, data: bytes, fec_size: int = 0) -> Tuple['Block', bool]:
        """
        Unpack block from bytes.

        Returns: (block, crc_valid)
        """
        if len(data) < HEADER_SIZE + CRC_SIZE:
            raise ValueError("Block data too short")

        header = BlockHeader.unpack(data[:HEADER_SIZE])
        payload_end = HEADER_SIZE + header.payload_size
        crc_end = payload_end + CRC_SIZE

        if len(data) < crc_end:
            raise ValueError("Block data truncated")

        payload = data[HEADER_SIZE:payload_end]
        crc_stored = struct.unpack('<I', data[payload_end:crc_end])[0]

        fec_parity = b''
        if fec_size > 0 and len(data) >= crc_end + fec_size:
            fec_parity = data[crc_end:crc_end + fec_size]

        block = cls(
            header=header,
            payload=payload,
            crc32=crc_stored,
            fec_parity=fec_parity
        )

        # Verify CRC
        computed_crc = block.compute_crc()
        crc_valid = (crc_stored == computed_crc)

        return block, crc_valid

    @staticmethod
    def calculate_overhead(fec_size: int) -> int:
        """Calculate total overhead bytes (header + CRC + FEC)."""
        return HEADER_SIZE + CRC_SIZE + fec_size


@dataclass
class FileMetadata:
    """
    Metadata stored in block 0's payload prefix.

    Format:
      [file_hash (32)] [filename_len (4)] [filename (N)]
      [aes_nonce (12)] [aes_tag (16)] (if encrypted)
    """
    file_hash: bytes           # SHA-256 hash (32 bytes)
    filename: str              # Original filename
    aes_nonce: Optional[bytes] = None  # AES-GCM nonce (12 bytes)
    aes_tag: Optional[bytes] = None    # AES-GCM tag (16 bytes)

    def pack(self) -> bytes:
        """Pack metadata into bytes."""
        filename_bytes = self.filename.encode('utf-8')
        filename_len = len(filename_bytes)

        result = self.file_hash + struct.pack('<I', filename_len) + filename_bytes

        if self.aes_nonce and self.aes_tag:
            result += self.aes_nonce + self.aes_tag

        return result

    @classmethod
    def unpack(cls, data: bytes, encrypted: bool = False) -> Tuple['FileMetadata', int]:
        """
        Unpack metadata from bytes.

        Returns: (metadata, bytes_consumed)
        """
        if len(data) < 36:  # 32 (hash) + 4 (filename_len)
            raise ValueError("Metadata too short")

        file_hash = data[:32]
        filename_len = struct.unpack('<I', data[32:36])[0]

        if len(data) < 36 + filename_len:
            raise ValueError("Metadata truncated at filename")

        filename = data[36:36 + filename_len].decode('utf-8')
        consumed = 36 + filename_len

        aes_nonce = None
        aes_tag = None

        if encrypted:
            if len(data) < consumed + 28:  # 12 (nonce) + 16 (tag)
                raise ValueError("Metadata truncated at encryption data")
            aes_nonce = data[consumed:consumed + 12]
            aes_tag = data[consumed + 12:consumed + 28]
            consumed += 28

        return cls(
            file_hash=file_hash,
            filename=filename,
            aes_nonce=aes_nonce,
            aes_tag=aes_tag
        ), consumed


def calculate_payload_capacity(profile, fec_ratio: float = 0.10) -> int:
    """
    Calculate net payload capacity per block for a given profile.

    Args:
        profile: EncodingProfile instance
        fec_ratio: FEC overhead ratio (0.0 to 0.5)

    Returns:
        Net payload bytes per block
    """
    raw_bytes = profile.payload_bytes
    fec_bytes = int(raw_bytes * fec_ratio)
    overhead = Block.calculate_overhead(fec_bytes)
    return raw_bytes - overhead


def calculate_total_blocks(file_size: int, payload_capacity: int, metadata_size: int) -> int:
    """
    Calculate total blocks needed for a file.

    Block 0 has reduced capacity due to metadata.
    """
    if file_size == 0:
        return 1

    # Block 0 capacity is reduced by metadata size
    block0_capacity = payload_capacity - metadata_size
    if block0_capacity < 0:
        raise ValueError("Metadata too large for block capacity")

    if file_size <= block0_capacity:
        return 1

    remaining = file_size - block0_capacity
    additional_blocks = (remaining + payload_capacity - 1) // payload_capacity

    return 1 + additional_blocks
