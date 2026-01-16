"""
Visual Data Diode - Shared Module

Common code used by both sender and receiver.
"""

from .constants import (
    FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_FPS,
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE,
    DEFAULT_PROFILE, EncodingProfile,
    GRAY_LEVELS, GRAY_THRESHOLDS, gray_to_bits, bits_to_gray,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_WHITE, COLOR_BLACK,
    CORNER_TOP_LEFT, CORNER_TOP_RIGHT, CORNER_BOTTOM_LEFT, CORNER_BOTTOM_RIGHT,
    HEADER_SIZE, CRC_SIZE, BlockFlags,
    RS_DEFAULT_NSYM, DEFAULT_FEC_RATIO,
    SYNC_FRAME_COUNT, END_FRAME_COUNT, DEFAULT_REPEAT_COUNT,
    CELL_SAMPLE_RATIO, PROTOCOL_VERSION, PROTOCOL_MAGIC
)

from .block import (
    BlockHeader, Block, FileMetadata,
    calculate_payload_capacity, calculate_total_blocks
)

from .fec import (
    FECEncoder, FECDecoder, SimpleFEC, check_fec_available
)

from .crypto import (
    Encryptor, Decryptor, derive_key,
    compute_file_hash, compute_data_hash, check_crypto_available
)

__all__ = [
    # Constants
    'FRAME_WIDTH', 'FRAME_HEIGHT', 'DEFAULT_FPS',
    'PROFILE_CONSERVATIVE', 'PROFILE_STANDARD', 'PROFILE_AGGRESSIVE',
    'DEFAULT_PROFILE', 'EncodingProfile',
    'GRAY_LEVELS', 'GRAY_THRESHOLDS', 'gray_to_bits', 'bits_to_gray',
    'COLOR_CYAN', 'COLOR_MAGENTA', 'COLOR_WHITE', 'COLOR_BLACK',
    'CORNER_TOP_LEFT', 'CORNER_TOP_RIGHT', 'CORNER_BOTTOM_LEFT', 'CORNER_BOTTOM_RIGHT',
    'HEADER_SIZE', 'CRC_SIZE', 'BlockFlags',
    'RS_DEFAULT_NSYM', 'DEFAULT_FEC_RATIO',
    'SYNC_FRAME_COUNT', 'END_FRAME_COUNT', 'DEFAULT_REPEAT_COUNT',
    'CELL_SAMPLE_RATIO', 'PROTOCOL_VERSION', 'PROTOCOL_MAGIC',
    # Block
    'BlockHeader', 'Block', 'FileMetadata',
    'calculate_payload_capacity', 'calculate_total_blocks',
    # FEC
    'FECEncoder', 'FECDecoder', 'SimpleFEC', 'check_fec_available',
    # Crypto
    'Encryptor', 'Decryptor', 'derive_key',
    'compute_file_hash', 'compute_data_hash', 'check_crypto_available',
]
