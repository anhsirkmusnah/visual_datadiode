"""
Visual Data Diode - Shared Module

Common code used by both sender and receiver.
"""

from .constants import (
    FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_FPS,
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
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
    calculate_payload_capacity, calculate_total_blocks, crc16
)

from .fec import (
    FECEncoder, FECDecoder, SimpleFEC, check_fec_available
)

from .crypto import (
    Encryptor, Decryptor, derive_key,
    compute_file_hash, compute_data_hash, check_crypto_available
)

from .enhanced_encoding import (
    EnhancedEncodingProfile,
    ENHANCED_LUMA_LEVELS, ENHANCED_LUMA_THRESHOLDS,
    ColorState, ENHANCED_COLOR_RGB, ENHANCED_COLOR_YCBCR,
    CalibrationData,
    luma_to_bits_enhanced, bits_to_luma_enhanced,
    get_color_rgb, detect_color_state,
    ENHANCED_PROFILE_CONSERVATIVE, ENHANCED_PROFILE_STANDARD,
    CALIBRATION_MAGIC, is_calibration_frame_marker,
    SYNC_MAGIC, is_sync_frame_marker,
)

from .enhanced_encoder import (
    EnhancedFrameEncoder,
    pack_data_for_enhanced_frame, unpack_data_from_enhanced_frame,
)

from .enhanced_decoder import (
    EnhancedFrameDecoder, EnhancedDecodeResult,
    EnhancedStreamDecoder,
)

from .cuda_decoder import (
    CUDAFrameDecoder, CUDAVideoProcessor,
    check_cuda_available, get_cuda_info,
)

from .binary_frame import (
    BinaryFrameHeader, BinaryDecodeStats,
    BINARY_MAGIC, BINARY_HEADER_SIZE, BINARY_CRC_SIZE, BINARY_FRAME_BYTES,
    BINARY_FLAG_FIRST, BINARY_FLAG_LAST, BINARY_FLAG_ENCRYPTED, BINARY_FLAG_METADATA,
    calculate_binary_payload_capacity,
    bytes_to_binary_frame, binary_frame_to_bytes,
    encode_binary_frame, decode_binary_frame,
)

__all__ = [
    # Constants
    'FRAME_WIDTH', 'FRAME_HEIGHT', 'DEFAULT_FPS',
    'PROFILE_CONSERVATIVE', 'PROFILE_STANDARD', 'PROFILE_AGGRESSIVE', 'PROFILE_ULTRA',
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
    'calculate_payload_capacity', 'calculate_total_blocks', 'crc16',
    # FEC
    'FECEncoder', 'FECDecoder', 'SimpleFEC', 'check_fec_available',
    # Crypto
    'Encryptor', 'Decryptor', 'derive_key',
    'compute_file_hash', 'compute_data_hash', 'check_crypto_available',
    # Enhanced Encoding
    'EnhancedEncodingProfile',
    'ENHANCED_LUMA_LEVELS', 'ENHANCED_LUMA_THRESHOLDS',
    'ColorState', 'ENHANCED_COLOR_RGB', 'ENHANCED_COLOR_YCBCR',
    'CalibrationData',
    'luma_to_bits_enhanced', 'bits_to_luma_enhanced',
    'get_color_rgb', 'detect_color_state',
    'ENHANCED_PROFILE_CONSERVATIVE', 'ENHANCED_PROFILE_STANDARD',
    'CALIBRATION_MAGIC', 'is_calibration_frame_marker',
    'SYNC_MAGIC', 'is_sync_frame_marker',
    # Enhanced Encoder/Decoder
    'EnhancedFrameEncoder',
    'pack_data_for_enhanced_frame', 'unpack_data_from_enhanced_frame',
    'EnhancedFrameDecoder', 'EnhancedDecodeResult',
    'EnhancedStreamDecoder',
    # CUDA Decoder
    'CUDAFrameDecoder', 'CUDAVideoProcessor',
    'check_cuda_available', 'get_cuda_info',
    # Binary Frame Protocol
    'BinaryFrameHeader', 'BinaryDecodeStats',
    'BINARY_MAGIC', 'BINARY_HEADER_SIZE', 'BINARY_CRC_SIZE', 'BINARY_FRAME_BYTES',
    'BINARY_FLAG_FIRST', 'BINARY_FLAG_LAST', 'BINARY_FLAG_ENCRYPTED', 'BINARY_FLAG_METADATA',
    'calculate_binary_payload_capacity',
    'bytes_to_binary_frame', 'binary_frame_to_bytes',
    'encode_binary_frame', 'decode_binary_frame',
]
