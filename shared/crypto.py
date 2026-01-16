"""
Visual Data Diode - Cryptography Module

Optional AES-256-GCM encryption for payload data.
"""

import os
import hashlib
from typing import Optional, Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    AESGCM = None


class Encryptor:
    """AES-256-GCM encryptor."""

    def __init__(self, key: bytes):
        """
        Initialize encryptor with a 256-bit key.

        Args:
            key: 32-byte encryption key
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography library not available. "
                "Install with: pip install cryptography"
            )

        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")

        self.aesgcm = AESGCM(key)

    def encrypt(self, plaintext: bytes, associated_data: bytes = b'') -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt plaintext with AES-256-GCM.

        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data (not encrypted)

        Returns:
            (ciphertext, nonce, tag)
            Note: ciphertext includes the tag appended by cryptography library
        """
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, associated_data)
        # cryptography library appends 16-byte tag to ciphertext
        # Split them for our protocol
        tag = ciphertext[-16:]
        ciphertext_only = ciphertext[:-16]
        return ciphertext_only, nonce, tag

    def encrypt_with_nonce(
        self, plaintext: bytes, nonce: bytes, associated_data: bytes = b''
    ) -> Tuple[bytes, bytes]:
        """
        Encrypt with specified nonce.

        Args:
            plaintext: Data to encrypt
            nonce: 12-byte nonce
            associated_data: Additional authenticated data

        Returns:
            (ciphertext, tag)
        """
        if len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes")

        ciphertext = self.aesgcm.encrypt(nonce, plaintext, associated_data)
        tag = ciphertext[-16:]
        ciphertext_only = ciphertext[:-16]
        return ciphertext_only, tag


class Decryptor:
    """AES-256-GCM decryptor."""

    def __init__(self, key: bytes):
        """
        Initialize decryptor with a 256-bit key.

        Args:
            key: 32-byte decryption key
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography library not available. "
                "Install with: pip install cryptography"
            )

        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")

        self.aesgcm = AESGCM(key)

    def decrypt(
        self, ciphertext: bytes, nonce: bytes, tag: bytes,
        associated_data: bytes = b''
    ) -> Optional[bytes]:
        """
        Decrypt ciphertext with AES-256-GCM.

        Args:
            ciphertext: Encrypted data (without tag)
            nonce: 12-byte nonce used for encryption
            tag: 16-byte authentication tag
            associated_data: Additional authenticated data

        Returns:
            Plaintext bytes, or None if authentication fails
        """
        try:
            # Reconstruct the ciphertext+tag format expected by cryptography
            ciphertext_with_tag = ciphertext + tag
            plaintext = self.aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data)
            return plaintext
        except Exception:
            return None


def derive_key(password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """
    Derive a 256-bit key from a password using PBKDF2.

    Args:
        password: User password
        salt: Optional 16-byte salt (generated if not provided)

    Returns:
        (key, salt) - 32-byte key and salt used
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography library not available")

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    if salt is None:
        salt = os.urandom(16)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )

    key = kdf.derive(password.encode('utf-8'))
    return key, salt


def compute_file_hash(filepath: str) -> bytes:
    """
    Compute SHA-256 hash of a file.

    Args:
        filepath: Path to file

    Returns:
        32-byte SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)  # 64KB chunks
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.digest()


def compute_data_hash(data: bytes) -> bytes:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Bytes to hash

    Returns:
        32-byte SHA-256 hash
    """
    return hashlib.sha256(data).digest()


def check_crypto_available() -> bool:
    """Check if cryptography library is available."""
    return CRYPTO_AVAILABLE
