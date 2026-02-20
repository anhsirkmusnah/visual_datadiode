"""Debug calibration frame detection."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    ENHANCED_PROFILE_CONSERVATIVE,
    EnhancedFrameEncoder,
    EnhancedFrameDecoder,
    CALIBRATION_MAGIC,
    ENHANCED_LUMA_LEVELS,
    luma_to_bits_enhanced,
)

profile = ENHANCED_PROFILE_CONSERVATIVE
encoder = EnhancedFrameEncoder(profile)
decoder = EnhancedFrameDecoder(profile)

# Generate calibration frame
print("Generating calibration frame...")
cal_frame = encoder.create_calibration_frame()

# Extract grid (sample cell centers)
print("\nExtracting grid...")
grid = np.zeros((profile.grid_height, profile.grid_width, 3), dtype=np.uint8)
for row in range(profile.grid_height):
    for col in range(profile.grid_width):
        cy = row * profile.cell_size + profile.cell_size // 2
        cx = col * profile.cell_size + profile.cell_size // 2
        grid[row, col] = cal_frame[cy, cx]

# Get interior
interior = grid[
    decoder.interior_top:decoder.interior_bottom,
    decoder.interior_left:decoder.interior_right
]

print(f"Grid shape: {grid.shape}")
print(f"Interior shape: {interior.shape}")
print(f"Interior bounds: top={decoder.interior_top}, bottom={decoder.interior_bottom}, left={decoder.interior_left}, right={decoder.interior_right}")

# Check magic marker
interior_h = interior.shape[0]
luma_section_h = interior_h // 2
strip_h = luma_section_h // 8

print(f"\nLuma section height: {luma_section_h} cells")
print(f"Strip height: {strip_h} cells")

print("\nChecking rightmost column for magic marker:")
rightmost_col = interior[:, -1]
print(f"Rightmost column shape: {rightmost_col.shape}")

print("\nExpected magic pattern:", CALIBRATION_MAGIC)
print("Expected luma values:", [ENHANCED_LUMA_LEVELS[l] for l in CALIBRATION_MAGIC])

print("\nDetected at strip centers:")
for i in range(8):
    y = i * strip_h + strip_h // 2
    if y >= len(rightmost_col):
        print(f"  Strip {i}: y={y} OUT OF BOUNDS")
        continue
    r, g, b = rightmost_col[y]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    level = luma_to_bits_enhanced(gray)
    expected = CALIBRATION_MAGIC[i]
    match = "OK" if level == expected else "MISMATCH"
    print(f"  Strip {i}: y={y}, RGB=({r},{g},{b}), luma={gray:.1f}, level={level} (expected {expected}) {match}")

# Also check what's at the very top rows of rightmost column
print("\nFirst 10 rows of rightmost column:")
for y in range(min(10, len(rightmost_col))):
    r, g, b = rightmost_col[y]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    level = luma_to_bits_enhanced(gray)
    print(f"  y={y}: RGB=({r},{g},{b}), luma={gray:.1f}, level={level}")
