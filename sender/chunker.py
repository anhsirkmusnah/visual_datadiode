"""
Visual Data Diode - File Chunker

Splits files into blocks for transmission.
"""

import os
import random
from typing import Generator, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    Block, BlockHeader, BlockFlags, FileMetadata,
    SimpleFEC, compute_file_hash,
    EncodingProfile, DEFAULT_PROFILE, DEFAULT_FEC_RATIO,
    HEADER_SIZE, CRC_SIZE
)


class FileChunker:
    """
    Splits a file into blocks for visual transmission.

    Handles:
    - File metadata generation (hash, filename)
    - Block header creation
    - FEC parity generation
    - Optional compression/encryption
    """

    def __init__(
        self,
        profile: EncodingProfile = DEFAULT_PROFILE,
        fec_ratio: float = DEFAULT_FEC_RATIO,
        encrypt: bool = False,
        encryption_key: Optional[bytes] = None,
        compress: bool = False
    ):
        """
        Initialize chunker.

        Args:
            profile: Encoding profile (determines capacity)
            fec_ratio: FEC overhead ratio
            encrypt: Whether to encrypt payload
            encryption_key: 32-byte AES-256 key (required if encrypt=True)
            compress: Whether to compress payload
        """
        self.profile = profile
        self.fec_ratio = fec_ratio
        self.encrypt = encrypt
        self.compress = compress

        # Initialize FEC
        self.fec = SimpleFEC(fec_ratio)

        # Calculate capacities
        self._calculate_capacities()

        # Encryption setup
        if encrypt:
            if encryption_key is None:
                raise ValueError("Encryption key required when encrypt=True")
            from shared import Encryptor
            self.encryptor = Encryptor(encryption_key)
            self.encryption_key = encryption_key
        else:
            self.encryptor = None
            self.encryption_key = None

        # Session state
        self.session_id = 0
        self.file_path: Optional[str] = None
        self.file_size = 0
        self.file_hash = b''
        self.total_blocks = 0
        self.aes_nonce: Optional[bytes] = None
        self.aes_tag: Optional[bytes] = None

    def _calculate_capacities(self):
        """Calculate block capacities based on profile and FEC."""
        # Raw bytes available in payload area
        raw_bytes = self.profile.payload_bytes

        # FEC parity size estimate (for max data)
        self.fec_parity_size = self.fec.parity_size(raw_bytes)

        # Net capacity per block (excluding header, CRC, FEC)
        self.block_capacity = raw_bytes - HEADER_SIZE - CRC_SIZE - self.fec_parity_size

        # Metadata size for block 0
        # Base: 32 (hash) + 4 (filename_len) + filename + optional crypto
        self.base_metadata_size = 36  # Without filename
        self.crypto_metadata_size = 28 if self.encrypt else 0  # nonce + tag

    def prepare_file(self, file_path: str) -> Tuple[int, int, bytes]:
        """
        Prepare a file for transmission.

        Args:
            file_path: Path to file

        Returns:
            (total_blocks, file_size, file_hash)
        """
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.file_hash = compute_file_hash(file_path)
        self.session_id = random.randint(1, 0xFFFFFFFF)

        # Get filename
        filename = os.path.basename(file_path)
        filename_bytes = filename.encode('utf-8')
        self.filename = filename

        # Calculate metadata size
        metadata_size = self.base_metadata_size + len(filename_bytes) + self.crypto_metadata_size

        # Block 0 has reduced capacity due to metadata
        block0_capacity = self.block_capacity - metadata_size

        if block0_capacity < 0:
            raise ValueError(
                f"Filename too long. Max: {self.block_capacity - self.base_metadata_size - self.crypto_metadata_size} bytes"
            )

        # Calculate total blocks
        if self.file_size <= block0_capacity:
            self.total_blocks = 1
        else:
            remaining = self.file_size - block0_capacity
            self.total_blocks = 1 + (remaining + self.block_capacity - 1) // self.block_capacity

        # If encrypting, encrypt the entire file first
        if self.encrypt:
            self._encrypt_file()

        return self.total_blocks, self.file_size, self.file_hash

    def _encrypt_file(self):
        """Encrypt the file data."""
        import os as os_module
        self.aes_nonce = os_module.urandom(12)

        # Read and encrypt entire file
        with open(self.file_path, 'rb') as f:
            plaintext = f.read()

        if self.compress:
            import zlib
            plaintext = zlib.compress(plaintext, level=6)

        ciphertext, tag = self.encryptor.encrypt_with_nonce(
            plaintext, self.aes_nonce
        )
        self.aes_tag = tag
        self._encrypted_data = ciphertext
        self.file_size = len(ciphertext)

        # Recalculate total blocks with encrypted size
        filename_bytes = self.filename.encode('utf-8')
        metadata_size = self.base_metadata_size + len(filename_bytes) + self.crypto_metadata_size
        block0_capacity = self.block_capacity - metadata_size

        if self.file_size <= block0_capacity:
            self.total_blocks = 1
        else:
            remaining = self.file_size - block0_capacity
            self.total_blocks = 1 + (remaining + self.block_capacity - 1) // self.block_capacity

    def generate_blocks(self) -> Generator[Block, None, None]:
        """
        Generate all blocks for the file.

        Yields:
            Block objects ready for encoding
        """
        if self.file_path is None:
            raise ValueError("No file prepared. Call prepare_file() first.")

        # Open file for reading
        if self.encrypt and hasattr(self, '_encrypted_data'):
            file_data = self._encrypted_data
            data_reader = _BytesReader(file_data)
        else:
            data_reader = _FileReader(self.file_path, self.compress)

        try:
            for block_index in range(self.total_blocks):
                yield self._generate_block(block_index, data_reader)
        finally:
            data_reader.close()

    def _generate_block(self, block_index: int, data_reader) -> Block:
        """Generate a single block."""
        is_first = (block_index == 0)
        is_last = (block_index == self.total_blocks - 1)

        # Build flags
        flags = BlockFlags.NONE
        if is_first:
            flags |= BlockFlags.FIRST_BLOCK
        if is_last:
            flags |= BlockFlags.LAST_BLOCK
        if self.encrypt:
            flags |= BlockFlags.ENCRYPTED
        if self.compress and not self.encrypt:  # Compression noted only if not encrypted
            flags |= BlockFlags.COMPRESSED

        # Determine payload capacity for this block
        if is_first:
            # Include metadata in block 0
            metadata = FileMetadata(
                file_hash=self.file_hash,
                filename=self.filename,
                aes_nonce=self.aes_nonce,
                aes_tag=self.aes_tag
            )
            metadata_bytes = metadata.pack()
            capacity = self.block_capacity - len(metadata_bytes)
        else:
            metadata_bytes = b''
            capacity = self.block_capacity

        # Read payload data
        payload_data = data_reader.read(capacity)

        # Combine metadata and payload for block 0
        full_payload = metadata_bytes + payload_data

        # Create header
        header = BlockHeader(
            session_id=self.session_id,
            block_index=block_index,
            total_blocks=self.total_blocks,
            file_size=self.file_size,
            payload_size=len(full_payload),
            flags=flags
        )

        # Create block and compute FEC
        block = Block(header=header, payload=full_payload)

        # Generate FEC parity
        block_data = header.pack() + full_payload
        _, parity = self.fec.encode(block_data)
        block.fec_parity = parity

        return block

    def get_block(self, block_index: int) -> Block:
        """
        Get a specific block by index.

        Useful for retransmission.
        """
        if self.file_path is None:
            raise ValueError("No file prepared.")

        if block_index < 0 or block_index >= self.total_blocks:
            raise ValueError(f"Block index out of range: {block_index}")

        # Calculate file offset
        filename_bytes = self.filename.encode('utf-8')
        metadata_size = self.base_metadata_size + len(filename_bytes) + self.crypto_metadata_size
        block0_capacity = self.block_capacity - metadata_size

        if block_index == 0:
            offset = 0
        else:
            offset = block0_capacity + (block_index - 1) * self.block_capacity

        # Read data at offset
        if self.encrypt and hasattr(self, '_encrypted_data'):
            data_reader = _BytesReader(self._encrypted_data)
        else:
            data_reader = _FileReader(self.file_path, self.compress)

        try:
            data_reader.seek(offset)
            return self._generate_block(block_index, data_reader)
        finally:
            data_reader.close()

    @property
    def effective_bitrate(self) -> float:
        """Calculate effective bitrate in bytes/second at default FPS."""
        from shared import DEFAULT_FPS
        return self.block_capacity * DEFAULT_FPS

    @property
    def estimated_transfer_time(self) -> float:
        """Estimate transfer time in seconds."""
        if self.file_size == 0:
            return 0
        return self.file_size / self.effective_bitrate


class _FileReader:
    """Helper class to read file in chunks."""

    def __init__(self, path: str, compress: bool = False):
        self.compress = compress
        if compress:
            import zlib
            with open(path, 'rb') as f:
                self._data = zlib.compress(f.read(), level=6)
            self._pos = 0
        else:
            self._file = open(path, 'rb')

    def read(self, size: int) -> bytes:
        if self.compress:
            data = self._data[self._pos:self._pos + size]
            self._pos += len(data)
            return data
        return self._file.read(size)

    def seek(self, offset: int):
        if self.compress:
            self._pos = offset
        else:
            self._file.seek(offset)

    def close(self):
        if not self.compress:
            self._file.close()


class _BytesReader:
    """Helper class to read from bytes."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, size: int) -> bytes:
        data = self._data[self._pos:self._pos + size]
        self._pos += len(data)
        return data

    def seek(self, offset: int):
        self._pos = offset

    def close(self):
        pass
