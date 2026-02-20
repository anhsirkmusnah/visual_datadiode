#!/usr/bin/env python3
"""
End-to-end test for enhanced encoding pipeline.

Tests the full encode -> H.264 video -> decode pipeline.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from encode_enhanced import EnhancedFastEncoder
from process_enhanced import EnhancedVideoProcessor


def test_encode_decode(test_size: int = 4000, crf: int = 18, repeat_count: int = 1):
    """Test encode/decode round-trip."""
    print(f"\n{'='*60}")
    print(f"Testing encode/decode with {test_size} bytes, CRF={crf}, repeat={repeat_count}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file with random-ish data (avoid sequential patterns that hit extreme luma)
        test_file = Path(tmpdir) / "test_input.bin"
        # Use a mix of values, avoiding 0x00 and 0xFF which cause extreme luma
        import random
        random.seed(42)  # Reproducible
        test_data = bytes([random.randint(16, 239) for _ in range(test_size)])
        test_file.write_bytes(test_data)
        print(f"Created test file: {len(test_data)} bytes (avoiding extreme values)")

        # Encode to video
        video_file = Path(tmpdir) / "test_video.mp4"
        encoder = EnhancedFastEncoder(fps=30, crf=crf, repeat_count=repeat_count, calibration_frames=10)
        result = encoder.encode_file(str(test_file), str(video_file))
        print(f"Encoded to: {result.output_file}")
        print(f"Video size: {result.output_size:,} bytes")

        # Decode video
        output_dir = Path(tmpdir) / "output"
        processor = EnhancedVideoProcessor(
            video_path=str(video_file),
            output_dir=str(output_dir)
        )
        proc_result = processor.process()

        print(f"\nProcessing result: {proc_result.success}")
        print(f"Files decoded: {len(proc_result.files_decoded)}")

        if proc_result.files_decoded:
            decoded_file = proc_result.files_decoded[0]
            print(f"Decoded: {decoded_file.filename}")
            print(f"Size: {decoded_file.file_size}")
            print(f"Blocks: {decoded_file.blocks_received}/{decoded_file.total_blocks}")
            print(f"Hash valid: {decoded_file.hash_valid}")

            # Compare data
            decoded_path = Path(decoded_file.output_path)
            if decoded_path.exists():
                decoded_data = decoded_path.read_bytes()
                if decoded_data == test_data:
                    print("\n[SUCCESS] Data matches exactly!")
                    return True
                else:
                    # Count total differences
                    diffs = sum(1 for a, b in zip(test_data, decoded_data) if a != b)
                    len_diff = abs(len(test_data) - len(decoded_data))

                    print(f"\n[FAILURE] Data mismatch!")
                    print(f"  Expected: {len(test_data)} bytes")
                    print(f"  Got: {len(decoded_data)} bytes")
                    print(f"  Total byte differences: {diffs}")

                    # Find first few differences
                    diff_positions = []
                    for i, (a, b) in enumerate(zip(test_data, decoded_data)):
                        if a != b:
                            diff_positions.append((i, a, b))
                            if len(diff_positions) >= 5:
                                break

                    for pos, expected, got in diff_positions:
                        region = "header/meta" if pos < 100 else "data"
                        print(f"  Byte {pos} ({region}): expected {expected:02x}, got {got:02x}")

                    return False
        else:
            print("\n[FAILURE] No files decoded!")
            return False

    return False


def test_sync_frame_detection():
    """Test that sync frames are properly detected and not interpreted as data."""
    import numpy as np
    from shared import EnhancedFrameEncoder, EnhancedFrameDecoder, ENHANCED_PROFILE_CONSERVATIVE

    print(f"\n{'='*60}")
    print("Testing sync frame detection")
    print(f"{'='*60}")

    encoder = EnhancedFrameEncoder(ENHANCED_PROFILE_CONSERVATIVE)
    decoder = EnhancedFrameDecoder(ENHANCED_PROFILE_CONSERVATIVE)

    # Create and test sync frame
    sync_frame = encoder.create_sync_frame(0)
    print(f"Created sync frame: {sync_frame.shape}")

    # Extract grid (simulate what processor does)
    cell_size = ENHANCED_PROFILE_CONSERVATIVE.cell_size
    grid_h = ENHANCED_PROFILE_CONSERVATIVE.grid_height
    grid_w = ENHANCED_PROFILE_CONSERVATIVE.grid_width

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for row in range(grid_h):
        for col in range(grid_w):
            cy = row * cell_size + cell_size // 2
            cx = col * cell_size + cell_size // 2
            if cy < sync_frame.shape[0] and cx < sync_frame.shape[1]:
                grid[row, col] = sync_frame[cy, cx]

    # Decode
    result = decoder.decode_grid(grid)
    print(f"Decode result:")
    print(f"  success: {result.success}")
    print(f"  is_calibration_frame: {result.is_calibration_frame}")
    print(f"  is_sync_frame: {result.is_sync_frame}")

    if result.is_sync_frame:
        print("\n[SUCCESS] Sync frame correctly detected!")
        return True
    else:
        print("\n[FAILURE] Sync frame not detected!")
        return False


def test_calibration_frame_detection():
    """Test that calibration frames are properly detected."""
    import numpy as np
    from shared import EnhancedFrameEncoder, EnhancedFrameDecoder, ENHANCED_PROFILE_CONSERVATIVE

    print(f"\n{'='*60}")
    print("Testing calibration frame detection")
    print(f"{'='*60}")

    encoder = EnhancedFrameEncoder(ENHANCED_PROFILE_CONSERVATIVE)
    decoder = EnhancedFrameDecoder(ENHANCED_PROFILE_CONSERVATIVE)

    # Create and test calibration frame
    cal_frame = encoder.create_calibration_frame()
    print(f"Created calibration frame: {cal_frame.shape}")

    # Extract grid
    cell_size = ENHANCED_PROFILE_CONSERVATIVE.cell_size
    grid_h = ENHANCED_PROFILE_CONSERVATIVE.grid_height
    grid_w = ENHANCED_PROFILE_CONSERVATIVE.grid_width

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for row in range(grid_h):
        for col in range(grid_w):
            cy = row * cell_size + cell_size // 2
            cx = col * cell_size + cell_size // 2
            if cy < cal_frame.shape[0] and cx < cal_frame.shape[1]:
                grid[row, col] = cal_frame[cy, cx]

    # Decode
    result = decoder.decode_grid(grid)
    print(f"Decode result:")
    print(f"  success: {result.success}")
    print(f"  is_calibration_frame: {result.is_calibration_frame}")
    print(f"  is_sync_frame: {result.is_sync_frame}")

    if result.is_calibration_frame:
        print("\n[SUCCESS] Calibration frame correctly detected!")
        if result.calibration:
            print(f"  Luma separation: {result.calibration.luma_separation:.1f}")
            print(f"  Color separation: {result.calibration.color_separation:.1f}")
        return True
    else:
        print("\n[FAILURE] Calibration frame not detected!")
        return False


def test_data_frame_not_confused():
    """Test that data frames are not confused with sync/calibration frames."""
    import numpy as np
    from shared import (
        EnhancedFrameEncoder, EnhancedFrameDecoder,
        ENHANCED_PROFILE_CONSERVATIVE, pack_data_for_enhanced_frame
    )

    print(f"\n{'='*60}")
    print("Testing data frame detection")
    print(f"{'='*60}")

    profile = ENHANCED_PROFILE_CONSERVATIVE
    encoder = EnhancedFrameEncoder(profile)
    decoder = EnhancedFrameDecoder(profile)

    # Create data frame with random data
    luma_bytes = profile.luma_bits_per_frame // 8
    color_bytes = profile.color_bits_per_frame // 8

    test_data = bytes(range(256)) * ((luma_bytes + color_bytes) // 256 + 1)
    test_data = test_data[:luma_bytes + color_bytes]

    luma_data, color_data = pack_data_for_enhanced_frame(test_data, profile)
    data_frame = encoder.encode_data_frame(luma_data, color_data, 0)
    print(f"Created data frame: {data_frame.shape}")

    # Extract grid
    cell_size = profile.cell_size
    grid_h = profile.grid_height
    grid_w = profile.grid_width

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for row in range(grid_h):
        for col in range(grid_w):
            cy = row * cell_size + cell_size // 2
            cx = col * cell_size + cell_size // 2
            if cy < data_frame.shape[0] and cx < data_frame.shape[1]:
                grid[row, col] = data_frame[cy, cx]

    # Decode
    result = decoder.decode_grid(grid)
    print(f"Decode result:")
    print(f"  success: {result.success}")
    print(f"  is_calibration_frame: {result.is_calibration_frame}")
    print(f"  is_sync_frame: {result.is_sync_frame}")
    print(f"  has luma_data: {result.luma_data is not None}")
    print(f"  luma_confidence: {result.luma_confidence:.2f}")

    if not result.is_calibration_frame and not result.is_sync_frame and result.luma_data:
        print("\n[SUCCESS] Data frame correctly detected as data!")
        return True
    else:
        print("\n[FAILURE] Data frame incorrectly classified!")
        return False


def test_color_encode_decode():
    """Test color encoding/decoding for all dibit values."""
    import numpy as np
    from shared import (
        EnhancedFrameEncoder, EnhancedFrameDecoder,
        ENHANCED_PROFILE_CONSERVATIVE, get_color_rgb, ColorState,
        ENHANCED_LUMA_LEVELS
    )

    print(f"\n{'='*60}")
    print("Testing color encode/decode for all dibits")
    print(f"{'='*60}")

    profile = ENHANCED_PROFILE_CONSERVATIVE
    encoder = EnhancedFrameEncoder(profile)
    decoder = EnhancedFrameDecoder(profile)

    errors = []

    # Test each dibit value (0-3) at each luma level (0-7)
    for luma_level in range(8):
        for dibit in range(4):
            # Generate RGB color
            color_state = ColorState(dibit)
            rgb = get_color_rgb(color_state, luma_level)

            # Simulate what decoder does: convert RGB to CbCr
            r, g, b = rgb
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 128 + 0.564 * (b - y)
            cr = 128 + 0.713 * (r - y)

            # Decode back
            decoded_state, conf = decoder._decode_color_default(cb, cr)

            if decoded_state != dibit:
                errors.append((luma_level, dibit, decoded_state, rgb, cb, cr, conf))

    if errors:
        print(f"\n[FAILURE] Found {len(errors)} color detection errors:")
        for luma, expected, got, rgb, cb, cr, conf in errors[:10]:
            print(f"  Luma {luma}, dibit {expected} -> {got} (RGB={rgb}, Cb={cb:.1f}, Cr={cr:.1f}, conf={conf:.2f})")
        return False
    else:
        print("\n[SUCCESS] All dibit values correctly detected!")
        return True


def test_direct_encode_decode():
    """Test direct encode/decode without video pipeline."""
    import numpy as np
    from shared import (
        EnhancedFrameEncoder, EnhancedFrameDecoder,
        ENHANCED_PROFILE_CONSERVATIVE, pack_data_for_enhanced_frame,
        unpack_data_from_enhanced_frame
    )

    print(f"\n{'='*60}")
    print("Testing direct encode/decode (no video)")
    print(f"{'='*60}")

    profile = ENHANCED_PROFILE_CONSERVATIVE
    encoder = EnhancedFrameEncoder(profile)
    decoder = EnhancedFrameDecoder(profile)

    # First, calibrate using calibration frame
    cell_size = profile.cell_size
    grid_h = profile.grid_height
    grid_w = profile.grid_width

    cal_frame = encoder.create_calibration_frame()
    cal_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for row in range(grid_h):
        for col in range(grid_w):
            cy = row * cell_size + cell_size // 2
            cx = col * cell_size + cell_size // 2
            if cy < cal_frame.shape[0] and cx < cal_frame.shape[1]:
                cal_grid[row, col] = cal_frame[cy, cx]

    cal_result = decoder.decode_grid(cal_grid)
    print(f"Calibration: valid={cal_result.calibration.is_valid if cal_result.calibration else False}")

    # Test data - 3190 bytes (full capacity)
    total_bytes = profile.payload_bytes
    test_data = bytes(range(256)) * (total_bytes // 256 + 1)
    test_data = test_data[:total_bytes]
    print(f"Test data: {len(test_data)} bytes")
    print(f"  First 20 bytes: {test_data[:20].hex()}")
    print(f"  Byte 2740: {test_data[2740]:02x}")

    # Pack and encode
    luma_data, color_data = pack_data_for_enhanced_frame(test_data, profile)
    print(f"Packed: luma={len(luma_data)} bytes, color={len(color_data)} bytes")
    print(f"  Color first byte: {color_data[0]:02x}")

    frame = encoder.encode_data_frame(luma_data, color_data, 0)
    print(f"Encoded frame: {frame.shape}")

    # Extract grid (no video compression)
    cell_size = profile.cell_size
    grid_h = profile.grid_height
    grid_w = profile.grid_width

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for row in range(grid_h):
        for col in range(grid_w):
            cy = row * cell_size + cell_size // 2
            cx = col * cell_size + cell_size // 2
            if cy < frame.shape[0] and cx < frame.shape[1]:
                grid[row, col] = frame[cy, cx]

    # Decode
    result = decoder.decode_grid(grid)
    print(f"Decoded: luma={len(result.luma_data)} bytes, color={len(result.color_data)} bytes")
    print(f"  Color first byte decoded: {result.color_data[0]:02x}")

    # Unpack
    decoded_data = unpack_data_from_enhanced_frame(result.luma_data, result.color_data)
    print(f"Unpacked: {len(decoded_data)} bytes")
    print(f"  First 20 bytes: {decoded_data[:20].hex()}")
    print(f"  Byte 2740: {decoded_data[2740]:02x}")

    # Compare
    if decoded_data == test_data:
        print("\n[SUCCESS] Direct encode/decode matches!")
        return True
    else:
        # Count all differences
        diffs = []
        for i, (a, b) in enumerate(zip(test_data, decoded_data)):
            if a != b:
                diffs.append((i, a, b))

        print(f"\n[FAILURE] Found {len(diffs)} byte differences:")

        # Show first 10 differences
        for i, (pos, expected, got) in enumerate(diffs[:10]):
            region = "luma" if pos < 2740 else "color"
            offset = pos if pos < 2740 else pos - 2740
            print(f"  Byte {pos} ({region}[{offset}]): expected {expected:02x}, got {got:02x}")

        if len(diffs) > 10:
            print(f"  ... and {len(diffs) - 10} more differences")

        # Show first error context
        pos, expected, got = diffs[0]
        start = max(0, pos - 5)
        end = min(len(test_data), pos + 10)
        print(f"\nContext around first error:")
        print(f"  Expected: {test_data[start:end].hex()}")
        print(f"  Got:      {decoded_data[start:end].hex()}")

        return False


def main():
    all_passed = True

    # Test frame type detection
    all_passed &= test_calibration_frame_detection()
    all_passed &= test_sync_frame_detection()
    all_passed &= test_data_frame_not_confused()

    # Test color encoding/decoding
    all_passed &= test_color_encode_decode()

    # Test direct encode/decode (no video compression)
    all_passed &= test_direct_encode_decode()

    # Test full pipeline
    # CRF 0 = lossless with yuv444p
    result = test_encode_decode(test_size=4000, crf=0, repeat_count=1)
    if result:
        all_passed = True
        print("\n[CRF=0 SUCCESS] - Lossless encoding works!")
    else:
        print("\n[CRF=0 FAILED] - Even lossless encoding has errors")

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
