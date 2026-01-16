# Frame and Protocol Format Specification

## Overview

This document defines the complete frame structure, block format, and protocol for the Visual Data Diode system.

---

## Frame Structure

### Layer Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RENDERED FRAME                             │
│  (1280×720 pixels, 24-bit RGB, rendered to HDMI output)            │
├─────────────────────────────────────────────────────────────────────┤
│                          CELL GRID                                  │
│  (128×72 cells at 10×10 pixels each in STANDARD profile)           │
├─────────────────────────────────────────────────────────────────────┤
│  SYNC BORDER (2 cells) │ HEADER │ PAYLOAD │ FOOTER │ SYNC BORDER   │
├─────────────────────────────────────────────────────────────────────┤
│                          DATA BLOCK                                 │
│  (Binary data: metadata + payload + CRC + FEC)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Profiles

### Resolution and Cell Size Options

| Profile      | Cell Size | Grid Size  | Payload Cells | Bytes/Frame | @ 20 FPS |
|--------------|-----------|------------|---------------|-------------|----------|
| CONSERVATIVE | 16×16     | 80×45      | 5,016         | 1,254       | 24.5 KB/s |
| STANDARD     | 10×10     | 128×72     | 8,184         | 1,817       | 35.5 KB/s |
| AGGRESSIVE   | 8×8       | 160×90     | 13,072        | 2,998       | 58.5 KB/s |

Default: **STANDARD**

---

## Cell Grid Layout (STANDARD Profile)

```
     0         2                              125      127
   ┌───┬───┬───┬───────────────────────────────┬───┬───┬───┐
 0 │ ● │ S │ S │  S  S  S  S  S  S  S  S  S  S │ S │ S │ ● │  Sync Row 0
 1 │ S │ ● │ S │  S  S  S  S  S  S  S  S  S  S │ S │ ● │ S │  Sync Row 1
   ├───┼───┼───┼───────────────────────────────┼───┼───┼───┤
 2 │ S │ S │ H │  H  H  H  H  H  H  H  H  H  H │ H │ S │ S │  Header
 3 │ S │ S │ D │  D  D  D  D  D  D  D  D  D  D │ D │ S │ S │  Payload
 4 │ S │ S │ D │  D  D  D  D  D  D  D  D  D  D │ D │ S │ S │  Payload
   │   │   │   │              ...              │   │   │   │
69 │ S │ S │ D │  D  D  D  D  D  D  D  D  D  D │ D │ S │ S │  Payload
   ├───┼───┼───┼───────────────────────────────┼───┼───┼───┤
70 │ S │ ● │ S │  S  S  S  S  S  S  S  S  S  S │ S │ ● │ S │  Sync Row -2
71 │ ● │ S │ S │  S  S  S  S  S  S  S  S  S  S │ S │ S │ ● │  Sync Row -1
   └───┴───┴───┴───────────────────────────────┴───┴───┴───┘

Legend:
  ● = Corner marker (unique pattern)
  S = Sync border (cyan/magenta alternating)
  H = Header cells (block metadata)
  D = Data cells (payload + CRC + FEC)
```

---

## Sync Border Specification

### Color Values

| Color   | RGB           | Purpose           |
|---------|---------------|-------------------|
| Cyan    | (0, 255, 255) | Sync pattern A    |
| Magenta | (255, 0, 255) | Sync pattern B    |
| White   | (255, 255, 255) | Corner marker   |
| Black   | (0, 0, 0)     | Corner marker     |

### Alternating Pattern

Row-column parity determines color:
```python
if (row + col) % 2 == 0:
    color = MAGENTA
else:
    color = CYAN
```

### Corner Markers

Each corner contains a unique 3×3 cell pattern for orientation:

```
Top-Left (0,0):     Top-Right (0,125):
┌─────────┐         ┌─────────┐
│ W  W  W │         │ B  W  W │
│ W  B  W │         │ W  B  W │
│ W  W  B │         │ W  W  W │
└─────────┘         └─────────┘

Bottom-Left (69,0): Bottom-Right (69,125):
┌─────────┐         ┌─────────┐
│ W  W  B │         │ B  W  B │
│ W  B  W │         │ W  B  W │
│ W  W  W │         │ W  W  W │
└─────────┘         └─────────┘
```

---

## Data Block Format

### Block Header (20 bytes)

```
Offset  Size  Field           Description
──────────────────────────────────────────────────────────
0       4     session_id      Random 32-bit ID for this transfer
4       4     block_index     0-indexed block number
8       4     total_blocks    Total blocks in file
12      4     file_size       Total file size in bytes
16      2     payload_size    Bytes of payload in this block
18      1     flags           Bit flags (see below)
19      1     reserved        Reserved (0x00)
──────────────────────────────────────────────────────────
```

### Flags Byte

```
Bit  Name           Description
───────────────────────────────────────────
0    FIRST_BLOCK    This is block 0
1    LAST_BLOCK     This is the final block
2    ENCRYPTED      Payload is AES-256-GCM encrypted
3    COMPRESSED     Payload is zlib compressed
4-7  reserved       Reserved (0)
───────────────────────────────────────────
```

### Block Payload

Variable size, determined by `payload_size` field.
Maximum: `(total_data_cells × 2 / 8) - header_size - crc_size - fec_size`

### Block Footer

```
Offset     Size   Field           Description
──────────────────────────────────────────────────────────
-N         N      fec_parity      Reed-Solomon parity bytes
-(4+N)     4      crc32           CRC-32 of header + payload
──────────────────────────────────────────────────────────
```

---

## Complete Block Layout

```
┌────────────────────────────────────────────────────────────┐
│                    BLOCK (serialized)                       │
├─────────────┬───────────────────────┬────────┬─────────────┤
│   HEADER    │       PAYLOAD         │ CRC-32 │ FEC PARITY  │
│  (20 bytes) │    (variable)         │ (4 B)  │  (N bytes)  │
├─────────────┴───────────────────────┴────────┴─────────────┤
│              Encoded into grayscale cells                   │
│              Packed MSB-first, left→right, top→bottom       │
└────────────────────────────────────────────────────────────┘
```

---

## Reed-Solomon FEC Configuration

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Symbol size | 8 bits | GF(2^8) |
| n | 255 | Codeword length |
| k | configurable | Data symbols (default: 223) |
| 2t | 32 | Error correction capability |

### Interleaving

For large blocks, data is interleaved across multiple RS codewords:
```
Block data: [B0 B1 B2 ... Bn]
            ↓ interleave across M codewords
Codeword 0: [B0, Bm, B2m, ...]
Codeword 1: [B1, Bm+1, B2m+1, ...]
...
```

This spreads burst errors across codewords.

---

## File Transfer Protocol

### Session Initialization

1. Sender generates random 32-bit `session_id`
2. Sender calculates total blocks: `ceil(file_size / payload_per_block)`
3. Sender optionally encrypts file with AES-256-GCM
4. Sender computes SHA-256 of original file

### Transmission Sequence

```
┌─────────────────────────────────────────────────────────────┐
│ SYNC FRAMES (10 frames of sync pattern only, no data)       │
├─────────────────────────────────────────────────────────────┤
│ Block 0 (FIRST_BLOCK flag set)                              │
│ Block 1                                                     │
│ Block 2                                                     │
│ ...                                                         │
│ Block N-1 (LAST_BLOCK flag set)                            │
├─────────────────────────────────────────────────────────────┤
│ REPEAT PASS (optional, configurable)                        │
│ Block 0, Block 1, ... Block N-1                            │
├─────────────────────────────────────────────────────────────┤
│ END MARKER (10 frames of all-black)                         │
└─────────────────────────────────────────────────────────────┘
```

### Frame Timing

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target FPS | 20 | Configurable 10-30 |
| Frame duration | 50 ms | At 20 FPS |
| Inter-frame gap | 0 ms | Continuous |
| Sync hold time | 500 ms | Initial sync period |

---

## Receiver State Machine

```
                    ┌───────────────┐
                    │   SEARCHING   │
                    │ (looking for  │
                    │  sync border) │
                    └───────┬───────┘
                            │ sync detected
                            ▼
                    ┌───────────────┐
                    │   SYNCING     │
                    │ (validating   │
                    │  corners)     │
                    └───────┬───────┘
                            │ corners valid
                            ▼
                    ┌───────────────┐
                    │   RECEIVING   │◄─────────┐
                    │ (decoding     │          │
                    │  blocks)      │──────────┤ more blocks
                    └───────┬───────┘          │
                            │ LAST_BLOCK       │
                            ▼                  │
                    ┌───────────────┐          │
                    │   VERIFYING   │          │
                    │ (hash check)  │          │
                    └───────┬───────┘          │
                            │                  │
              ┌─────────────┴─────────────┐    │
              ▼                           ▼    │
      ┌───────────────┐          ┌───────────────┐
      │   COMPLETE    │          │    ERROR      │
      │ (file saved)  │          │ (retry/abort) │
      └───────────────┘          └───────────────┘
```

---

## Block Reception Logic

```python
def receive_block(frame):
    # 1. Detect sync border
    if not detect_sync_border(frame):
        return SYNC_LOST

    # 2. Locate corners, compute perspective transform
    corners = find_corners(frame)
    if not all_corners_valid(corners):
        return CORNER_ERROR

    # 3. Extract cell grid
    grid = extract_cells(frame, corners)

    # 4. Decode grayscale values to bits
    bits = decode_cells(grid)

    # 5. Unpack header
    header = unpack_header(bits[:160])  # 20 bytes × 8 bits

    # 6. Verify CRC
    crc_offset = 160 + header.payload_size * 8
    crc_received = unpack_crc(bits[crc_offset:crc_offset+32])
    crc_computed = crc32(bits[:crc_offset])

    if crc_received != crc_computed:
        # 7. Attempt FEC recovery
        corrected = rs_decode(bits)
        if corrected is None:
            return CRC_ERROR
        bits = corrected

    # 8. Extract payload
    payload = unpack_payload(bits, header.payload_size)

    # 9. Store block
    store_block(header.session_id, header.block_index, payload)

    return SUCCESS
```

---

## Duplicate Block Handling

Receiver maintains:
```python
received_blocks = {}  # block_index → payload
block_attempts = {}   # block_index → decode_attempts

def handle_block(header, payload):
    idx = header.block_index

    if idx in received_blocks:
        # Already have this block, verify consistency
        if received_blocks[idx] == payload:
            return DUPLICATE_OK
        else:
            return DUPLICATE_MISMATCH  # Corruption somewhere

    received_blocks[idx] = payload
    return NEW_BLOCK
```

---

## File Reconstruction

After receiving all blocks or on explicit "finalize" command:

```python
def reconstruct_file(session_id):
    blocks = get_all_blocks(session_id)

    # Check completeness
    total = blocks[0].total_blocks
    missing = [i for i in range(total) if i not in blocks]

    if missing:
        return INCOMPLETE, missing

    # Assemble in order
    data = b''.join(blocks[i].payload for i in range(total))

    # Decrypt if needed
    if blocks[0].flags & ENCRYPTED:
        data = aes_gcm_decrypt(data, key, nonce)

    # Decompress if needed
    if blocks[0].flags & COMPRESSED:
        data = zlib.decompress(data)

    # Verify hash (stored in block 0 metadata or separate)
    if sha256(data) != expected_hash:
        return HASH_MISMATCH

    return SUCCESS, data
```

---

## Error Handling

### Recoverable Errors

| Error | Recovery Action |
|-------|-----------------|
| Single block CRC fail | FEC decode attempt |
| FEC decode fail | Wait for retransmission |
| Sync loss | Re-enter SEARCHING state |
| Duplicate block | Ignore (idempotent) |

### Fatal Errors

| Error | Action |
|-------|--------|
| Session ID mismatch mid-transfer | Warn user, may be new transfer |
| Hash verification fail | Report corruption, request re-send |
| Too many missing blocks | Report incomplete transfer |

---

## Metadata Exchange

### Pre-Transfer Metadata (Block 0 Extended Payload)

Block 0 includes additional metadata before file data:

```
Offset  Size  Field
────────────────────────────────────
0       32    file_hash         SHA-256 of original file
32      4     filename_len      Length of filename
36      N     filename          UTF-8 filename
36+N    16    aes_nonce         (if encrypted) AES-GCM nonce
52+N    16    aes_tag           (if encrypted) AES-GCM auth tag
────────────────────────────────────
```

---

## Performance Tuning Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `cell_size` | 8-20 | 10 | Larger = more robust, slower |
| `fec_ratio` | 0.0-0.5 | 0.1 | Higher = more redundancy |
| `repeat_count` | 1-5 | 2 | Higher = more chances to decode |
| `target_fps` | 10-30 | 20 | Higher = faster, more drops |
| `sync_frames` | 5-20 | 10 | More = easier initial sync |
