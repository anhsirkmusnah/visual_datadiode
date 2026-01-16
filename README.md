# Visual Data Diode

A one-way, high-reliability data transfer system that transmits files from a restricted (locked) laptop to a free laptop using HDMI video output and USB HDMI capture.

## Overview

This system implements a **visual modem** / **optical data diode** that:
- Encodes binary data into grayscale cell grids rendered on HDMI output
- Captures and decodes those frames using a USB capture device
- Provides robust error correction and integrity verification
- Supports encryption for sensitive data

## Features

- **One-way transfer**: No back-channel required; strictly air-gapped
- **Large file support**: Transfer files of any size (GBs)
- **Error resilience**: Reed-Solomon FEC + CRC-32 for corruption detection/correction
- **Optional encryption**: AES-256-GCM for sensitive payloads
- **Configurable profiles**: Trade-off between speed and reliability
- **User-friendly GUI**: Tkinter-based interfaces for both sender and receiver

## System Requirements

### Sender (Locked Laptop)
- Python 3.8+
- HDMI output (1280x720 capable)
- Packages: `pygame`, `numpy`, `reedsolo`, `cryptography` (optional)

### Receiver (Free Laptop)
- Python 3.8+
- USB HDMI capture device (MS2130 chipset recommended)
- Windows OS with DirectShow support
- Packages: `opencv-python`, `numpy`, `reedsolo`, `Pillow` (for preview), `cryptography` (optional)

## Installation

```bash
# Clone or copy the repository
cd visual_datadiode

# Install dependencies (sender)
pip install pygame numpy reedsolo cryptography

# Install dependencies (receiver)
pip install opencv-python numpy reedsolo Pillow cryptography
```

## Usage

### Sender

```bash
# Run the sender application
python -m sender.main
```

1. Click "Browse..." to select a file to transmit
2. Choose encoding profile (Standard recommended for most cases)
3. Configure FPS and repeat count
4. Optionally enable encryption
5. Select output display (secondary HDMI)
6. Click "Start Transmission"

### Receiver

```bash
# Run the receiver application
python -m receiver.main
```

1. Select the capture device from the dropdown
2. Choose the same encoding profile as the sender
3. Set output directory for received files
4. Enter decryption password if the file is encrypted
5. Click "Start Receiving"

## Encoding Profiles

| Profile | Cell Size | Throughput | Reliability | Use Case |
|---------|-----------|------------|-------------|----------|
| Conservative | 16x16 | ~24 KB/s | Highest | Noisy capture, long cable |
| Standard | 10x10 | ~35 KB/s | High | Recommended default |
| Aggressive | 8x8 | ~58 KB/s | Medium | High-quality capture |

## Transfer Time Estimates (Standard Profile @ 20 FPS)

| File Size | Time (1x repeat) | Time (2x repeat) |
|-----------|------------------|------------------|
| 100 MB | 48 min | 96 min |
| 500 MB | 4 hours | 8 hours |
| 1 GB | 8 hours | 16 hours |

## Architecture

```
visual_datadiode/
├── sender/
│   ├── main.py        # Sender application entry point
│   ├── ui.py          # Tkinter UI
│   ├── chunker.py     # File splitting and block creation
│   ├── encoder.py     # Binary-to-visual encoding
│   ├── renderer.py    # HDMI output via pygame
│   └── timing.py      # Frame timing control
│
├── receiver/
│   ├── main.py        # Receiver application entry point
│   ├── ui.py          # Tkinter UI
│   ├── capture.py     # USB capture via OpenCV
│   ├── sync.py        # Frame boundary detection
│   ├── decoder.py     # Visual-to-binary decoding
│   └── assembler.py   # Block reassembly and verification
│
├── shared/
│   ├── constants.py   # Protocol constants
│   ├── block.py       # Block structure definitions
│   ├── fec.py         # Reed-Solomon FEC
│   └── crypto.py      # AES-256-GCM encryption
│
├── protocol/
│   ├── encoding_spec.md   # Encoding strategy documentation
│   └── frame_format.md    # Frame and protocol specification
│
└── tests/
    └── test_patterns.py   # Test pattern generators
```

## Protocol Details

### Encoding Strategy

The system uses a **grayscale cell grid** encoding:
- 4 grayscale levels (0, 85, 170, 255) = 2 bits per cell
- Cyan/magenta sync border for frame detection
- Unique corner markers for orientation verification
- Row-major, MSB-first bit packing

### Block Structure

```
[Header (20 bytes)] [Payload (variable)] [CRC-32 (4 bytes)] [FEC Parity (variable)]
```

Header fields:
- Session ID (4 bytes)
- Block Index (4 bytes)
- Total Blocks (4 bytes)
- File Size (4 bytes)
- Payload Size (2 bytes)
- Flags (1 byte)
- Reserved (1 byte)

### Error Handling

1. **CRC-32**: Detects corrupted blocks
2. **Reed-Solomon FEC**: Corrects up to 10% symbol errors
3. **Repeat transmission**: Each block sent multiple times
4. **Duplicate detection**: Receiver ignores repeated blocks

## Troubleshooting

### Capture device not detected
- Ensure the USB capture device is connected
- Try the DirectShow backend explicitly
- Check Windows Device Manager for driver issues

### Low sync confidence
- Increase cell size (use Conservative profile)
- Check HDMI cable connection
- Ensure no desktop scaling is applied
- Verify capture resolution matches 1280x720

### High CRC error rate
- Use a higher repeat count (3-4)
- Switch to Conservative profile
- Check for interference or signal quality issues
- Try MJPEG mode if YUV422 fails

### Decryption fails
- Verify the password is correct
- Ensure the same password is used on both ends
- Check that encryption was enabled during send

## Known Limitations

- Transfer speeds are limited by visual encoding (~35 KB/s max)
- Requires stable HDMI connection
- Sender display may be unusable during transmission
- Long transfers (GB+) take hours

## Future Improvements

- Adaptive encoding based on measured quality
- Partial file resume after interruption
- Multiple file container support
- Color-based encoding for higher throughput
- Compressed payload option

## License

This project is provided as-is for educational and authorized use.

## Technical References

- [Reed-Solomon Error Correction](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction)
- [AES-GCM Encryption](https://en.wikipedia.org/wiki/Galois/Counter_Mode)
- [MS2130 USB Capture Chipset](https://www.macrosilicon.com/)
