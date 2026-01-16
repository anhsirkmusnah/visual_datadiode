# Visual Encoding Specification

## Research Summary: Encoding Strategy Comparison

### 1. QR Codes
**Evaluated:** Standard QR codes (Version 1-40)

**Pros:**
- Built-in Reed-Solomon error correction
- Mature libraries (qrcode, pyzbar, zbar)
- Designed for camera/scanner capture
- Self-synchronizing finder patterns

**Cons:**
- Designed for static images, not streaming video
- Maximum capacity (Version 40, L-level): 2,953 bytes
- Finder patterns waste ~10-15% of frame area
- At 720p with single large QR: ~2.9KB/frame → 58 KB/s at 20 FPS
- Not optimized for controlled HDMI→capture pipeline
- Detection latency impacts throughput

**Verdict:** Suboptimal. Finder patterns waste space; designed for unknown capture conditions we don't have.

---

### 2. Color-Cell Grid (Custom Design) ★ SELECTED
**Evaluated:** Custom grid of colored cells with explicit sync borders

**Pros:**
- Fully optimized for known capture conditions (720p, HDMI, USB capture)
- No wasted space on finder patterns (use border markers instead)
- Direct control over cell size vs. reliability tradeoff
- Can optimize color palette for YUV422 characteristics
- Simple, fast decoding (no complex pattern matching)

**Cons:**
- Must design sync mechanism from scratch
- Must handle color quantization and compression artifacts
- No off-the-shelf libraries

**Analysis:**
With MS2130 chipset in YUV422 mode:
- Chrominance (U/V) is subsampled 2:1 horizontally
- Luminance (Y) preserved at full resolution
- **Grayscale-based encoding is most robust**

**Verdict:** Selected. Optimal for controlled, high-throughput streaming.

---

### 3. DataMatrix / Aztec Codes
**Evaluated:** 2D barcode alternatives to QR

**Pros:**
- DataMatrix has no quiet zone requirement
- Aztec has central finder (more payload at edges)
- Smaller codes possible for small payloads

**Cons:**
- Similar fundamental limitations to QR
- Less library support
- Still designed for static capture

**Verdict:** Not significantly better than QR for streaming use case.

---

### 4. DCT/Frequency-Based Encoding
**Evaluated:** Encoding data in frequency domain to survive JPEG compression

**Pros:**
- Could theoretically survive JPEG compression better
- Works with the compression rather than against it

**Cons:**
- Extremely complex to implement correctly
- HDMI→capture chain introduces unknown transforms
- Synchronization is very difficult
- Low data density in practice

**Verdict:** Too complex, impractical for this application.

---

## Selected Approach: Grayscale Cell Grid

### Design Rationale

1. **Cell Size Selection**
   - Larger cells = more robust, less data
   - Smaller cells = more data, more fragile
   - **Configurable profiles:**
     - CONSERVATIVE: 16x16 pixels (79×45 = 3,555 cells)
     - STANDARD: 10x10 pixels (128×72 = 9,216 cells)
     - AGGRESSIVE: 8x8 pixels (160×90 = 14,400 cells)

2. **Color Palette: 4 Grayscale Levels (2 bits/cell)**

   | Value | Binary | Gray Level | RGB Value    |
   |-------|--------|------------|--------------|
   | 0     | 00     | Black      | (0, 0, 0)    |
   | 1     | 01     | Dark Gray  | (85, 85, 85) |
   | 2     | 10     | Light Gray | (170, 170, 170) |
   | 3     | 11     | White      | (255, 255, 255) |

   **Why 4 levels, not 8 or 16?**
   - 4 levels provide ~40 units separation in 0-255 range
   - Robust to ±15 units noise from compression
   - 8 levels would have only ~32 units separation (fragile)
   - YUV422 subsampling doesn't affect grayscale

3. **Sync Border: Colored Markers**

   For frame boundary detection, we use HIGH-CONTRAST colors:
   - **Cyan (0, 255, 255)** and **Magenta (255, 0, 255)**
   - Alternating pattern around border
   - These survive YUV422 well due to extreme saturation
   - Easy to detect even with color bleeding

4. **Corner Markers: Orientation Detection**

   Each corner has a unique 3x3 pattern for:
   - Frame detection
   - Rotation detection (not expected but defensive)
   - Sub-pixel alignment reference

---

## Capacity Calculations

### STANDARD Profile (10x10 cells at 720p)

```
Frame: 1280 × 720 pixels
Cell:  10 × 10 pixels
Grid:  128 × 72 cells = 9,216 total cells

Border (sync): 2 cells wide on all sides
  - Top/Bottom: 128 × 2 × 2 = 512 cells
  - Left/Right: (72-4) × 2 × 2 = 272 cells
  - Total sync: 784 cells

Interior grid: 124 × 68 = 8,432 cells
  - Header row: 124 cells = 31 bytes
  - Footer row: 124 cells = 31 bytes
  - Payload rows: 66 rows × 124 cells = 8,184 cells

Payload capacity: 8,184 cells × 2 bits ÷ 8 = 2,046 bytes raw

Overhead allocation:
  - Block index:     4 bytes
  - Total blocks:    4 bytes
  - Session ID:      4 bytes
  - Sequence number: 4 bytes
  - Flags:           2 bytes
  - Reserved:        2 bytes
  - CRC-32:          4 bytes
  - FEC parity:    205 bytes (10% RS overhead)
  ─────────────────────────────
  Total overhead:  229 bytes

Net payload per frame: 2,046 - 229 = 1,817 bytes
At 20 FPS: 36,340 bytes/sec ≈ **35.5 KB/s**
```

### Transfer Time Estimates (STANDARD profile)

| File Size | Time      |
|-----------|-----------|
| 100 MB    | 48 min    |
| 500 MB    | 4 hours   |
| 1 GB      | 8 hours   |
| 5 GB      | 40 hours  |

### AGGRESSIVE Profile (8x8 cells)

```
Grid: 160 × 90 cells = 14,400 total
Interior: 156 × 86 = 13,416 cells
Payload cells: ~13,000
Net payload: ~3,000 bytes/frame
At 20 FPS: ~58.5 KB/s
```

| File Size | Time      |
|-----------|-----------|
| 100 MB    | 29 min    |
| 1 GB      | 4.8 hours |

---

## Cell Encoding Details

### Bit Packing Order

Data is packed MSB-first, left-to-right, top-to-bottom:

```
Byte 0:     [b7 b6 b5 b4 b3 b2 b1 b0]
             ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
Cells:      C0  C1  C2  C3 ...
            [3] [2] [1] [0]
```

Each cell encodes 2 bits:
- Bits 7-6 → Cell 0
- Bits 5-4 → Cell 1
- Bits 3-2 → Cell 2
- Bits 1-0 → Cell 3

### Gray Level Detection (Receiver)

Sample center region of each cell (inner 60% to avoid edge effects):
```
For 10×10 cell: sample 6×6 center = 36 pixels
Average luminance value → quantize to nearest level

Thresholds:
  Y < 43   → Level 0 (Black)
  Y < 128  → Level 1 (Dark Gray)
  Y < 213  → Level 2 (Light Gray)
  Y >= 213 → Level 3 (White)
```

---

## Sync Border Specification

### Pattern

```
┌─────────────────────────────────────────┐
│ M C M C M C M C M C M C M C M C M C M C │  ← Row 0: Magenta/Cyan alternating
│ C M C M C M C M C M C M C M C M C M C M │  ← Row 1: Cyan/Magenta alternating
│ M C ┌─────────────────────────────┐ M C │
│ C M │                             │ C M │
│ M C │                             │ M C │
│ C M │        PAYLOAD GRID         │ C M │
│ M C │                             │ M C │
│ C M │                             │ C M │
│ M C └─────────────────────────────┘ M C │
│ C M C M C M C M C M C M C M C M C M C M │
│ M C M C M C M C M C M C M C M C M C M C │
└─────────────────────────────────────────┘
```

### Corner Markers (3×3 cells each)

```
Top-Left:       Top-Right:      Bottom-Left:    Bottom-Right:
┌───────┐       ┌───────┐       ┌───────┐       ┌───────┐
│ W W W │       │ B W W │       │ W W B │       │ B W B │
│ W B W │       │ W B W │       │ W B W │       │ W B W │
│ W W B │       │ W W W │       │ W W W │       │ W W W │
└───────┘       └───────┘       └───────┘       └───────┘
(W=White, B=Black)
```

These unique patterns allow:
- Confirming all 4 corners are visible
- Detecting any rotation/flip
- Sub-cell alignment calibration

---

## Robustness Features

### 1. Temporal Redundancy
Each block is transmitted N times (configurable, default N=2):
- First transmission: block 0, 1, 2, ...
- Second pass: block 0, 1, 2, ... (repeat)

Receiver uses first valid decode, ignores duplicates.

### 2. Spatial Margin
- Cell sampling uses center 60% only
- Border cells are sync-only (never payload)
- 1-pixel padding between logical cells (optional)

### 3. Adaptive Thresholding
Receiver can:
- Use sync border colors to calibrate expected levels
- Adjust thresholds based on observed histogram
- Flag low-confidence cells for FEC recovery

---

## Comparison with QR Alternative

| Metric | QR Code Array | Grayscale Grid |
|--------|---------------|----------------|
| Payload/frame | ~2.5 KB | ~2.0 KB |
| Decode complexity | High (pattern match) | Low (sample grid) |
| Sync overhead | ~15% (finder) | ~8% (border) |
| Robustness | Good | Good |
| Streaming optimized | No | Yes |
| Custom FEC | No (fixed RS) | Yes (tunable) |

**Conclusion:** Grayscale grid is better suited for continuous streaming with known capture conditions.
