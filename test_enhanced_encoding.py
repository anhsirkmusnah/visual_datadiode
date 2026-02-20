"""
Test script for enhanced encoding with 8 luma + 6 color states.

Creates calibration frame, encodes test data, and verifies round-trip.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    EnhancedEncodingProfile,
    ENHANCED_PROFILE_CONSERVATIVE,
    EnhancedFrameEncoder,
    EnhancedFrameDecoder,
    EnhancedStreamDecoder,
    pack_data_for_enhanced_frame,
    unpack_data_from_enhanced_frame,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def test_calibration_frame():
    """Test calibration frame generation and analysis."""
    print("=" * 60)
    print("TEST: Calibration Frame")
    print("=" * 60)

    profile = ENHANCED_PROFILE_CONSERVATIVE
    encoder = EnhancedFrameEncoder(profile)
    decoder = EnhancedFrameDecoder(profile)

    # Generate calibration frame
    print("\n1. Generating calibration frame...")
    cal_frame = encoder.create_calibration_frame()
    print(f"   Frame shape: {cal_frame.shape}")

    # Save as image if cv2 available
    if HAS_CV2:
        output_path = "test_calibration_frame.png"
        cv2.imwrite(output_path, cv2.cvtColor(cal_frame, cv2.COLOR_RGB2BGR))
        print(f"   Saved to: {output_path}")

    # Extract grid (simulate receiver)
    print("\n2. Extracting grid from frame...")
    # Sample cell centers
    grid = np.zeros((profile.grid_height, profile.grid_width, 3), dtype=np.uint8)
    for row in range(profile.grid_height):
        for col in range(profile.grid_width):
            cy = row * profile.cell_size + profile.cell_size // 2
            cx = col * profile.cell_size + profile.cell_size // 2
            grid[row, col] = cal_frame[cy, cx]

    # Decode calibration frame
    print("\n3. Analyzing calibration frame...")
    result = decoder.decode_grid(grid)

    print(f"   Is calibration frame: {result.is_calibration_frame}")
    if result.calibration:
        cal = result.calibration
        print(f"   Calibration valid: {cal.is_valid}")
        print(f"   Luma separation: {cal.luma_separation:.1f}")
        print(f"   Color separation: {cal.color_separation:.1f}")

        print("\n   Luma levels measured:")
        for level, (mean, std) in sorted(cal.luma_calibration.items()):
            expected = [0, 36, 73, 109, 146, 182, 219, 255][level]
            print(f"     Level {level}: mean={mean:.1f} (expected {expected}), std={std:.1f}")

        print("\n   Color states measured (Cb, Cr):")
        for state, (cb, cr, std) in sorted(cal.color_calibration.items()):
            print(f"     State {state}: Cb={cb:.1f}, Cr={cr:.1f}, std={std:.1f}")

    return result.calibration


def test_data_encoding(calibration=None):
    """Test data frame encoding and decoding."""
    print("\n" + "=" * 60)
    print("TEST: Data Encoding Round-Trip")
    print("=" * 60)

    profile = ENHANCED_PROFILE_CONSERVATIVE
    encoder = EnhancedFrameEncoder(profile)
    decoder = EnhancedFrameDecoder(profile)

    # Apply calibration if provided
    if calibration:
        decoder.set_calibration(calibration)
        print("   Using calibration data")

    # Generate test data
    print("\n1. Generating test data...")
    capacity = encoder.get_capacity()
    print(f"   Capacity: {capacity}")

    luma_size = capacity['luma_bytes_per_frame']
    color_size = capacity['color_bytes_per_frame']

    # Create known test pattern
    luma_data = bytes([i % 256 for i in range(luma_size)])
    color_data = bytes([i % 256 for i in range(color_size)])

    print(f"   Luma data: {len(luma_data)} bytes")
    print(f"   Color data: {len(color_data)} bytes")

    # Encode
    print("\n2. Encoding data frame...")
    frame = encoder.encode_data_frame(luma_data, color_data)
    print(f"   Frame shape: {frame.shape}")

    if HAS_CV2:
        output_path = "test_data_frame.png"
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"   Saved to: {output_path}")

    # Extract grid
    print("\n3. Extracting grid...")
    grid = np.zeros((profile.grid_height, profile.grid_width, 3), dtype=np.uint8)
    for row in range(profile.grid_height):
        for col in range(profile.grid_width):
            cy = row * profile.cell_size + profile.cell_size // 2
            cx = col * profile.cell_size + profile.cell_size // 2
            grid[row, col] = frame[cy, cx]

    # Decode
    cal_status = "with calibration" if calibration else "without calibration"
    print(f"\n4. Decoding ({cal_status})...")
    result = decoder.decode_grid(grid)

    print(f"   Success: {result.success}")
    print(f"   Luma confidence: {result.luma_confidence:.3f}")
    print(f"   Color confidence: {result.color_confidence:.3f}")

    if result.luma_data and result.color_data:
        # Compare
        luma_match = result.luma_data[:100] == luma_data[:100]
        color_match = result.color_data[:100] == color_data[:100]

        print(f"\n   First 100 bytes match (luma): {luma_match}")
        print(f"   First 100 bytes match (color): {color_match}")

        # Count errors
        luma_errors = sum(1 for a, b in zip(result.luma_data, luma_data) if a != b)
        color_errors = sum(1 for a, b in zip(result.color_data, color_data) if a != b)

        print(f"\n   Luma errors: {luma_errors}/{len(luma_data)} ({100*luma_errors/len(luma_data):.2f}%)")
        print(f"   Color errors: {color_errors}/{len(color_data)} ({100*color_errors/len(color_data):.2f}%)")


def test_grayscale_only():
    """Test grayscale-only encoding (no color)."""
    print("\n" + "=" * 60)
    print("TEST: Grayscale-Only Encoding")
    print("=" * 60)

    profile = ENHANCED_PROFILE_CONSERVATIVE
    encoder = EnhancedFrameEncoder(profile)
    decoder = EnhancedFrameDecoder(profile)

    # Generate test data
    luma_size = profile.luma_bits_per_frame // 8
    test_data = bytes([i % 256 for i in range(luma_size)])

    print(f"\n1. Encoding {len(test_data)} bytes in grayscale only...")
    frame = encoder.encode_grayscale_frame(test_data)

    if HAS_CV2:
        output_path = "test_grayscale_frame.png"
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"   Saved to: {output_path}")

    # Extract and decode
    grid = np.zeros((profile.grid_height, profile.grid_width, 3), dtype=np.uint8)
    for row in range(profile.grid_height):
        for col in range(profile.grid_width):
            cy = row * profile.cell_size + profile.cell_size // 2
            cx = col * profile.cell_size + profile.cell_size // 2
            grid[row, col] = frame[cy, cx]

    result = decoder.decode_grid(grid)

    if result.luma_data:
        luma_errors = sum(1 for a, b in zip(result.luma_data, test_data) if a != b)
        print(f"\n   Luma confidence: {result.luma_confidence:.3f}")
        print(f"   Luma errors: {luma_errors}/{len(test_data)} ({100*luma_errors/len(test_data):.2f}%)")


def main():
    print("Enhanced Encoding Test Suite")
    print("8 Luma Levels + 6 Color States")
    print()

    # Run tests
    calibration = test_calibration_frame()

    # Test without calibration
    test_data_encoding(calibration=None)

    # Test with calibration
    test_data_encoding(calibration=calibration)

    test_grayscale_only()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

    if HAS_CV2:
        print("\nGenerated files:")
        print("  - test_calibration_frame.png")
        print("  - test_data_frame.png")
        print("  - test_grayscale_frame.png")


if __name__ == "__main__":
    main()
