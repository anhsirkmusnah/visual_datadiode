"""
Simple file transfer test for visual data diode.
Tests sending and receiving a small file through the visual channel.
"""
import cv2
import numpy as np
import pygame
import time
import sys
import os
import hashlib

# Configuration - VALIDATED SETTINGS
CAPTURE_DEVICE = 3
DISPLAY_INDEX = 3
CELL_SIZE = 16  # Conservative setting for reliability
FPS = 30

def create_test_file(path, size_bytes=1024):
    """Create a test file with known content."""
    # Create predictable content based on position
    data = bytes([i % 256 for i in range(size_bytes)])
    with open(path, 'wb') as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()

def encode_byte_to_cells(byte_val, cell_size):
    """
    Encode a single byte as a 2x4 grid of grayscale cells.
    Each cell represents 2 bits (4 levels: 0, 85, 170, 255).
    """
    levels = [0, 85, 170, 255]
    cells = []
    for i in range(4):
        bits = (byte_val >> (6 - i*2)) & 0x03
        cells.append(levels[bits])
    return cells

def decode_cells_to_byte(cell_values):
    """
    Decode 4 grayscale cell values back to a byte.
    """
    byte_val = 0
    thresholds = [43, 128, 213]

    for i, val in enumerate(cell_values):
        if val < thresholds[0]:
            bits = 0
        elif val < thresholds[1]:
            bits = 1
        elif val < thresholds[2]:
            bits = 2
        else:
            bits = 3
        byte_val |= (bits << (6 - i*2))

    return byte_val

def create_frame_from_data(data, frame_w, frame_h, cell_size):
    """
    Create a frame image encoding the given data.

    Layout:
    - Cyan/Magenta border for sync
    - Data cells in interior
    """
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Border colors
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)

    border_cells = 2
    border_px = border_cells * cell_size

    # Draw sync border
    # Top border - cyan
    frame[:border_px, :] = CYAN
    # Bottom border - magenta
    frame[-border_px:, :] = MAGENTA
    # Left border - cyan
    frame[:, :border_px] = CYAN
    # Right border - magenta
    frame[:, -border_px:] = MAGENTA

    # Interior dimensions
    int_left = border_px
    int_top = border_px
    int_right = frame_w - border_px
    int_bottom = frame_h - border_px
    int_w = int_right - int_left
    int_h = int_bottom - int_top

    # Cells per row (4 cells per byte)
    cells_per_row = int_w // cell_size
    bytes_per_row = cells_per_row // 4

    # Encode data into cells
    data_idx = 0
    y = int_top

    while y + cell_size <= int_bottom and data_idx < len(data):
        x = int_left
        for _ in range(bytes_per_row):
            if data_idx >= len(data):
                break

            byte_val = data[data_idx]
            cells = encode_byte_to_cells(byte_val, cell_size)

            for cell_val in cells:
                if x + cell_size <= int_right:
                    frame[y:y+cell_size, x:x+cell_size] = [cell_val, cell_val, cell_val]
                    x += cell_size

            data_idx += 1

        y += cell_size

    return frame, data_idx  # Return how many bytes were encoded

def decode_frame_to_data(frame, cell_size, expected_bytes):
    """
    Decode data from a captured frame.
    """
    frame_h, frame_w = frame.shape[:2]

    border_cells = 2
    border_px = border_cells * cell_size

    # Interior dimensions
    int_left = border_px
    int_top = border_px
    int_right = frame_w - border_px
    int_bottom = frame_h - border_px

    # Cells per row
    cells_per_row = (int_right - int_left) // cell_size
    bytes_per_row = cells_per_row // 4

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    data = bytearray()
    y = int_top

    while y + cell_size <= int_bottom and len(data) < expected_bytes:
        x = int_left
        for _ in range(bytes_per_row):
            if len(data) >= expected_bytes:
                break

            cell_values = []
            for _ in range(4):
                if x + cell_size <= int_right:
                    # Sample center of cell
                    cx = x + cell_size // 2
                    cy = y + cell_size // 2
                    sample_r = cell_size // 4
                    region = gray[cy-sample_r:cy+sample_r, cx-sample_r:cx+sample_r]
                    cell_values.append(int(np.mean(region)))
                    x += cell_size

            if len(cell_values) == 4:
                byte_val = decode_cells_to_byte(cell_values)
                data.append(byte_val)

        y += cell_size

    return bytes(data)

def test_transfer(file_size=256):
    """
    Test file transfer through visual channel.
    """
    print("="*60)
    print(f"VISUAL DATA DIODE - FILE TRANSFER TEST ({file_size} bytes)")
    print("="*60)

    # Create test file
    test_file = "test_input.bin"
    expected_hash = create_test_file(test_file, file_size)
    print(f"Created test file: {test_file}")
    print(f"Expected hash: {expected_hash[:16]}...")

    with open(test_file, 'rb') as f:
        original_data = f.read()

    # Initialize pygame
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()

    x_offset = sum(sizes[i][0] for i in range(DISPLAY_INDEX))
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},0"

    disp_w, disp_h = sizes[DISPLAY_INDEX]
    screen = pygame.display.set_mode((disp_w, disp_h), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    print(f"Display: {DISPLAY_INDEX} ({disp_w}x{disp_h})")

    # Open capture
    cap = cv2.VideoCapture(CAPTURE_DEVICE, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Capture: Device {CAPTURE_DEVICE} with MSMF")

    # Flush buffer
    for _ in range(30):
        cap.grab()

    # Calculate how many bytes fit in one frame
    border_px = 2 * CELL_SIZE
    int_w = disp_w - 2 * border_px
    int_h = disp_h - 2 * border_px
    cells_per_row = int_w // CELL_SIZE
    bytes_per_row = cells_per_row // 4
    rows = int_h // CELL_SIZE
    bytes_per_frame = bytes_per_row * rows

    print(f"Capacity: {bytes_per_frame} bytes per frame (cell size: {CELL_SIZE})")

    # Encode and send frames
    total_bytes = len(original_data)
    frames_needed = (total_bytes + bytes_per_frame - 1) // bytes_per_frame
    print(f"Sending {total_bytes} bytes in {frames_needed} frame(s)...")

    received_data = bytearray()
    sent_offset = 0
    frame_num = 0

    while sent_offset < total_bytes:
        # Create frame with next chunk of data
        chunk = original_data[sent_offset:sent_offset + bytes_per_frame]
        frame_img, bytes_encoded = create_frame_from_data(
            chunk, disp_w, disp_h, CELL_SIZE
        )

        # Display frame
        surface = pygame.surfarray.make_surface(np.transpose(frame_img, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Wait for display to update
        time.sleep(0.1)

        # Flush and capture
        for _ in range(5):
            cap.grab()

        ret, captured = cap.read()
        if ret:
            # Decode captured frame
            decoded = decode_frame_to_data(captured, CELL_SIZE, len(chunk))
            received_data.extend(decoded)

            # Check this frame
            if decoded == chunk:
                print(f"  Frame {frame_num}: {len(chunk)} bytes OK")
            else:
                errors = sum(1 for a, b in zip(decoded, chunk) if a != b)
                print(f"  Frame {frame_num}: {len(chunk)} bytes - {errors} errors")
        else:
            print(f"  Frame {frame_num}: capture failed!")

        sent_offset += bytes_encoded
        frame_num += 1

    # Cleanup
    cap.release()
    pygame.quit()

    # Verify
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    received_hash = hashlib.sha256(received_data).hexdigest()
    print(f"Received {len(received_data)} bytes")
    print(f"Received hash: {received_hash[:16]}...")

    if received_data == original_data:
        print("\nSTATUS: SUCCESS - Data transferred correctly!")
        success = True
    else:
        errors = sum(1 for a, b in zip(received_data, original_data) if a != b)
        print(f"\nSTATUS: FAIL - {errors} byte errors out of {total_bytes}")
        success = False

    # Cleanup test file
    os.remove(test_file)

    return success

if __name__ == "__main__":
    # Test with increasing file sizes
    sizes_to_test = [64, 256, 1024]

    if len(sys.argv) > 1:
        sizes_to_test = [int(sys.argv[1])]

    for size in sizes_to_test:
        print("\n" + "="*70)
        success = test_transfer(size)
        if not success:
            print(f"\nTest failed at {size} bytes")
            sys.exit(1)
        time.sleep(1)

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
