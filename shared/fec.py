"""
Visual Data Diode - Forward Error Correction

Reed-Solomon encoder/decoder using reedsolo library.
Provides interleaved coding for burst error resistance.
"""

from typing import Optional, Tuple, List
import math

try:
    from reedsolo import RSCodec, ReedSolomonError
    REEDSOLO_AVAILABLE = True
except ImportError:
    REEDSOLO_AVAILABLE = False
    RSCodec = None
    ReedSolomonError = Exception


class FECEncoder:
    """
    Reed-Solomon FEC encoder with interleaving.

    Uses multiple RS codewords interleaved to spread burst errors.
    """

    def __init__(self, nsym: int = 32, max_codeword: int = 255):
        """
        Initialize FEC encoder.

        Args:
            nsym: Number of parity symbols per codeword (2t for RS)
            max_codeword: Maximum codeword length (n for RS(n,k))
        """
        if not REEDSOLO_AVAILABLE:
            raise ImportError(
                "reedsolo library not available. Install with: pip install reedsolo"
            )

        self.nsym = nsym
        self.max_codeword = max_codeword
        self.max_data = max_codeword - nsym  # k = n - 2t
        self.codec = RSCodec(nsym)

    def encode(self, data: bytes) -> Tuple[bytes, bytes]:
        """
        Encode data with Reed-Solomon FEC.

        For data larger than max_data, uses interleaved codewords.

        Args:
            data: Input data bytes

        Returns:
            (data, parity) - Original data and parity bytes
        """
        if len(data) == 0:
            return data, b''

        if len(data) <= self.max_data:
            # Single codeword
            encoded = self.codec.encode(data)
            parity = encoded[len(data):]
            return data, bytes(parity)

        # Multiple interleaved codewords
        num_codewords = math.ceil(len(data) / self.max_data)
        parity_parts = []

        for i in range(num_codewords):
            # Interleave: take every num_codewords-th byte starting at i
            interleaved = bytes(data[j] for j in range(i, len(data), num_codewords))

            if len(interleaved) > self.max_data:
                # Split into chunks
                chunks = [interleaved[k:k + self.max_data]
                          for k in range(0, len(interleaved), self.max_data)]
                parity_chunk = b''
                for chunk in chunks:
                    encoded = self.codec.encode(chunk)
                    parity_chunk += encoded[len(chunk):]
                parity_parts.append(parity_chunk)
            else:
                encoded = self.codec.encode(interleaved)
                parity_parts.append(bytes(encoded[len(interleaved):]))

        # Interleave parity bytes
        max_parity_len = max(len(p) for p in parity_parts)
        interleaved_parity = bytearray()

        for j in range(max_parity_len):
            for i in range(num_codewords):
                if j < len(parity_parts[i]):
                    interleaved_parity.append(parity_parts[i][j])

        return data, bytes(interleaved_parity)

    def calculate_parity_size(self, data_size: int) -> int:
        """Calculate parity size for given data size."""
        if data_size == 0:
            return 0

        if data_size <= self.max_data:
            return self.nsym

        num_codewords = math.ceil(data_size / self.max_data)
        return num_codewords * self.nsym


class FECDecoder:
    """
    Reed-Solomon FEC decoder with interleaving support.
    """

    def __init__(self, nsym: int = 32, max_codeword: int = 255):
        """
        Initialize FEC decoder.

        Args:
            nsym: Number of parity symbols per codeword
            max_codeword: Maximum codeword length
        """
        if not REEDSOLO_AVAILABLE:
            raise ImportError(
                "reedsolo library not available. Install with: pip install reedsolo"
            )

        self.nsym = nsym
        self.max_codeword = max_codeword
        self.max_data = max_codeword - nsym
        self.codec = RSCodec(nsym)

    def decode(self, data: bytes, parity: bytes) -> Tuple[Optional[bytes], int]:
        """
        Decode data with FEC error correction.

        Args:
            data: Potentially corrupted data
            parity: Reed-Solomon parity bytes

        Returns:
            (corrected_data, errors_corrected)
            Returns (None, -1) if uncorrectable
        """
        if len(data) == 0:
            return data, 0

        if len(parity) == 0:
            return data, 0

        try:
            if len(data) <= self.max_data:
                # Single codeword
                codeword = bytearray(data) + bytearray(parity)
                decoded, _, errata_pos = self.codec.decode(codeword)
                return bytes(decoded), len(errata_pos)

            # Multiple interleaved codewords
            num_codewords = math.ceil(len(data) / self.max_data)
            parity_per_codeword = len(parity) // num_codewords

            corrected_data = bytearray(len(data))
            total_errors = 0

            for i in range(num_codewords):
                # De-interleave data
                interleaved_data = bytearray(
                    data[j] for j in range(i, len(data), num_codewords)
                )

                # De-interleave parity (same interleaving as encoder)
                interleaved_parity = bytearray(
                    parity[j] for j in range(i, len(parity), num_codewords)
                )

                # Decode codeword
                codeword = interleaved_data + bytearray(interleaved_parity)

                try:
                    decoded, _, errata_pos = self.codec.decode(codeword)
                    total_errors += len(errata_pos)

                    # Re-interleave corrected data
                    for idx, byte_val in enumerate(decoded):
                        original_pos = i + idx * num_codewords
                        if original_pos < len(data):
                            corrected_data[original_pos] = byte_val

                except ReedSolomonError:
                    # This codeword is uncorrectable
                    return None, -1

            return bytes(corrected_data), total_errors

        except ReedSolomonError:
            return None, -1
        except Exception:
            return None, -1


class SimpleFEC:
    """
    Simplified FEC interface for the visual data diode.

    Handles both encoding and decoding with consistent parameters.
    """

    def __init__(self, fec_ratio: float = 0.10):
        """
        Initialize with target FEC ratio.

        Args:
            fec_ratio: Target parity overhead (0.0 to 0.5)
        """
        self.fec_ratio = fec_ratio

        # If fec_ratio is 0 or very small, disable FEC entirely
        if fec_ratio <= 0.001:
            self.nsym = 0
            self.encoder = None
            self.decoder = None
            return

        # Calculate nsym based on ratio
        # For RS(255, k), nsym = 255 - k
        # If we want 10% overhead, and k=223, nsym=32, overhead = 32/223 ≈ 14%
        # Adjust nsym to match target ratio
        self.nsym = max(4, min(64, int(255 * fec_ratio / (1 + fec_ratio))))

        if REEDSOLO_AVAILABLE:
            self.encoder = FECEncoder(nsym=self.nsym)
            self.decoder = FECDecoder(nsym=self.nsym)
        else:
            self.encoder = None
            self.decoder = None

    def encode(self, data: bytes) -> Tuple[bytes, bytes]:
        """Encode data, returning (data, parity)."""
        if self.encoder is None:
            return data, b''
        return self.encoder.encode(data)

    def decode(self, data: bytes, parity: bytes) -> Tuple[Optional[bytes], int]:
        """Decode data, returning (corrected_data, error_count) or (None, -1)."""
        if self.decoder is None:
            return data, 0
        return self.decoder.decode(data, parity)

    def parity_size(self, data_size: int) -> int:
        """Calculate parity size for given data size."""
        if self.encoder is None:
            return 0
        return self.encoder.calculate_parity_size(data_size)

    @property
    def available(self) -> bool:
        """Check if FEC is available."""
        return REEDSOLO_AVAILABLE


def check_fec_available() -> bool:
    """Check if FEC library is available."""
    return REEDSOLO_AVAILABLE
