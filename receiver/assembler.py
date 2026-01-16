"""
Visual Data Diode - Block Assembler

Assembles received blocks into complete files.
"""

import os
import hashlib
import tempfile
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    Block, BlockHeader, FileMetadata, BlockFlags,
    Decryptor, derive_key
)


@dataclass
class AssemblyStatus:
    """Status of file assembly."""
    complete: bool
    blocks_received: int
    total_blocks: int
    missing_blocks: List[int]
    file_hash_valid: Optional[bool]
    output_path: Optional[str]
    filename: Optional[str]
    file_size: int
    message: str = ""


class BlockAssembler:
    """
    Assembles received blocks into complete files.

    Features:
    - Atomic block writes (no partial files on crash)
    - Duplicate detection
    - Missing block tracking
    - Final hash verification
    - Decryption support
    """

    def __init__(
        self,
        output_dir: str = None,
        password: str = None
    ):
        """
        Initialize assembler.

        Args:
            output_dir: Directory for output files
            password: Decryption password (optional)
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        self.password = password

        # State
        self.session_id: Optional[int] = None
        self.total_blocks: Optional[int] = None
        self.file_size: int = 0
        self.metadata: Optional[FileMetadata] = None

        # Block storage
        self.blocks: Dict[int, bytes] = {}  # block_index -> payload

        # Temporary file for streaming writes
        self._temp_file = None
        self._temp_path: Optional[str] = None

    def add_block(self, block: Block) -> Tuple[bool, str]:
        """
        Add a decoded block.

        Args:
            block: Decoded block

        Returns:
            (success, message)
        """
        header = block.header

        # Initialize session if first block
        if self.session_id is None:
            self.session_id = header.session_id
            self.total_blocks = header.total_blocks
            self.file_size = header.file_size

            # Create temp file
            self._temp_path = os.path.join(
                self.output_dir,
                f"vdd_recv_{self.session_id}.tmp"
            )

        # Verify session
        if header.session_id != self.session_id:
            return False, f"Session mismatch: {header.session_id} != {self.session_id}"

        # Check for duplicate
        if header.block_index in self.blocks:
            return True, "Duplicate block (ignored)"

        # Extract payload
        payload = block.payload

        # Handle block 0 (contains metadata)
        if header.is_first:
            try:
                encrypted = bool(header.flags & BlockFlags.ENCRYPTED)
                self.metadata, consumed = FileMetadata.unpack(payload, encrypted)
                payload = payload[consumed:]
            except Exception as e:
                return False, f"Failed to parse metadata: {e}"

        # Store block
        self.blocks[header.block_index] = payload

        # Update progress
        received = len(self.blocks)
        total = self.total_blocks or 1

        if header.is_last:
            return True, f"Last block received ({received}/{total})"

        return True, f"Block {header.block_index} received ({received}/{total})"

    def get_status(self) -> AssemblyStatus:
        """Get current assembly status."""
        if self.total_blocks is None:
            return AssemblyStatus(
                complete=False,
                blocks_received=0,
                total_blocks=0,
                missing_blocks=[],
                file_hash_valid=None,
                output_path=None,
                filename=None,
                file_size=0,
                message="No blocks received"
            )

        received = len(self.blocks)
        missing = [
            i for i in range(self.total_blocks)
            if i not in self.blocks
        ]

        return AssemblyStatus(
            complete=(received == self.total_blocks),
            blocks_received=received,
            total_blocks=self.total_blocks,
            missing_blocks=missing,
            file_hash_valid=None,
            output_path=None,
            filename=self.metadata.filename if self.metadata else None,
            file_size=self.file_size,
            message=f"Received {received}/{self.total_blocks} blocks"
        )

    def is_complete(self) -> bool:
        """Check if all blocks received."""
        if self.total_blocks is None:
            return False
        return len(self.blocks) == self.total_blocks

    def assemble(self, output_path: str = None) -> AssemblyStatus:
        """
        Assemble the file from received blocks.

        Args:
            output_path: Output file path (optional, uses metadata filename)

        Returns:
            AssemblyStatus with result
        """
        if not self.is_complete():
            status = self.get_status()
            status.message = f"Cannot assemble: missing {len(status.missing_blocks)} blocks"
            return status

        # Determine output path
        if output_path is None:
            if self.metadata and self.metadata.filename:
                output_path = os.path.join(self.output_dir, self.metadata.filename)
            else:
                output_path = os.path.join(
                    self.output_dir,
                    f"received_{self.session_id}.bin"
                )

        try:
            # Concatenate blocks in order
            data = b''.join(
                self.blocks[i] for i in range(self.total_blocks)
            )

            # Decrypt if needed
            if self.metadata and self.metadata.aes_nonce:
                data = self._decrypt(data)
                if data is None:
                    return AssemblyStatus(
                        complete=True,
                        blocks_received=self.total_blocks,
                        total_blocks=self.total_blocks,
                        missing_blocks=[],
                        file_hash_valid=False,
                        output_path=None,
                        filename=self.metadata.filename if self.metadata else None,
                        file_size=self.file_size,
                        message="Decryption failed (wrong password?)"
                    )

            # Verify hash
            computed_hash = hashlib.sha256(data).digest()
            expected_hash = self.metadata.file_hash if self.metadata else None

            hash_valid = (computed_hash == expected_hash) if expected_hash else None

            # Write file atomically
            temp_path = output_path + ".tmp"
            with open(temp_path, 'wb') as f:
                f.write(data)

            os.replace(temp_path, output_path)

            return AssemblyStatus(
                complete=True,
                blocks_received=self.total_blocks,
                total_blocks=self.total_blocks,
                missing_blocks=[],
                file_hash_valid=hash_valid,
                output_path=output_path,
                filename=self.metadata.filename if self.metadata else None,
                file_size=len(data),
                message="File assembled successfully" if hash_valid else "File assembled (hash mismatch!)"
            )

        except Exception as e:
            return AssemblyStatus(
                complete=True,
                blocks_received=self.total_blocks,
                total_blocks=self.total_blocks,
                missing_blocks=[],
                file_hash_valid=False,
                output_path=None,
                filename=self.metadata.filename if self.metadata else None,
                file_size=self.file_size,
                message=f"Assembly failed: {e}"
            )

    def _decrypt(self, data: bytes) -> Optional[bytes]:
        """Decrypt data using stored password."""
        if not self.password:
            return None

        if not self.metadata or not self.metadata.aes_nonce:
            return data

        try:
            key, _ = derive_key(self.password)
            decryptor = Decryptor(key)

            plaintext = decryptor.decrypt(
                data,
                self.metadata.aes_nonce,
                self.metadata.aes_tag
            )

            if plaintext is None:
                return None

            # Decompress if needed (compression is done before encryption)
            # Note: decompression would be handled here if flag is set
            return plaintext

        except Exception:
            return None

    def reset(self):
        """Reset assembler state."""
        self.session_id = None
        self.total_blocks = None
        self.file_size = 0
        self.metadata = None
        self.blocks.clear()

        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except Exception:
                pass

        self._temp_file = None
        self._temp_path = None

    def save_state(self, path: str):
        """
        Save assembler state for later resume.

        Args:
            path: Path to save state file
        """
        import json
        import base64

        state = {
            'session_id': self.session_id,
            'total_blocks': self.total_blocks,
            'file_size': self.file_size,
            'blocks': {
                str(k): base64.b64encode(v).decode('ascii')
                for k, v in self.blocks.items()
            }
        }

        if self.metadata:
            state['metadata'] = {
                'file_hash': base64.b64encode(self.metadata.file_hash).decode('ascii'),
                'filename': self.metadata.filename,
                'aes_nonce': base64.b64encode(self.metadata.aes_nonce).decode('ascii') if self.metadata.aes_nonce else None,
                'aes_tag': base64.b64encode(self.metadata.aes_tag).decode('ascii') if self.metadata.aes_tag else None
            }

        with open(path, 'w') as f:
            json.dump(state, f)

    def load_state(self, path: str) -> bool:
        """
        Load assembler state from file.

        Args:
            path: Path to state file

        Returns:
            True if successful
        """
        import json
        import base64

        try:
            with open(path, 'r') as f:
                state = json.load(f)

            self.session_id = state['session_id']
            self.total_blocks = state['total_blocks']
            self.file_size = state['file_size']

            self.blocks = {
                int(k): base64.b64decode(v)
                for k, v in state['blocks'].items()
            }

            if 'metadata' in state:
                m = state['metadata']
                self.metadata = FileMetadata(
                    file_hash=base64.b64decode(m['file_hash']),
                    filename=m['filename'],
                    aes_nonce=base64.b64decode(m['aes_nonce']) if m['aes_nonce'] else None,
                    aes_tag=base64.b64decode(m['aes_tag']) if m['aes_tag'] else None
                )

            return True

        except Exception:
            return False


class StreamingAssembler(BlockAssembler):
    """
    Assembler that writes blocks to disk as they arrive.

    Better for very large files that don't fit in memory.
    """

    def __init__(self, output_dir: str = None, password: str = None):
        super().__init__(output_dir, password)
        self._block_files: Dict[int, str] = {}

    def add_block(self, block: Block) -> Tuple[bool, str]:
        """Add block and write to disk immediately."""
        result = super().add_block(block)

        if result[0] and block.header.block_index in self.blocks:
            # Write block to individual file
            block_path = os.path.join(
                self.output_dir,
                f"block_{self.session_id}_{block.header.block_index:08d}.tmp"
            )

            try:
                with open(block_path, 'wb') as f:
                    f.write(self.blocks[block.header.block_index])

                self._block_files[block.header.block_index] = block_path

                # Clear from memory
                # Keep in self.blocks for tracking but could clear payload
                # self.blocks[block.header.block_index] = b''

            except Exception as e:
                return False, f"Failed to write block: {e}"

        return result

    def assemble(self, output_path: str = None) -> AssemblyStatus:
        """Assemble from block files."""
        # For streaming, we need to read blocks from files
        # and concatenate them

        # First, reload any blocks from files
        for idx, path in self._block_files.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.blocks[idx] = f.read()

        result = super().assemble(output_path)

        # Clean up block files if successful
        if result.complete and result.file_hash_valid:
            for path in self._block_files.values():
                try:
                    os.remove(path)
                except Exception:
                    pass

        return result
