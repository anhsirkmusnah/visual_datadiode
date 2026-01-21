"""
End-to-end test for visual data diode.
Tests the complete pipeline: Display -> Capture -> Decode.
"""
import cv2
import numpy as np
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_transfer(capture_device=3, display_index=1):
    """
    Test transferring simple data patterns through the visual channel.
    """
    import pygame

    print("="*60)
    print("VISUAL DATA DIODE - END TO END TEST")
    print("="*60)
    print(f"Capture device: {capture_device}")
    print(f"Display index: {display_index}")
    print()

    # Initialize pygame
    pygame.init()

    # Setup display
    sizes = pygame.display.get_desktop_sizes()
    print(f"Available displays: {sizes}")

    if display_index >= len(sizes):
        print(f"Display {display_index} not found!")
        return False

    # Position on target display
    x_offset = sum(sizes[i][0] for i in range(display_index))
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},0"

    disp_w, disp_h = sizes[display_index]
    print(f"Using display {display_index}: {disp_w}x{disp_h}")

    screen = pygame.display.set_mode((disp_w, disp_h), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    # Open capture with MSMF
    print(f"\nOpening capture device {capture_device} with MSMF...")
    cap = cv2.VideoCapture(capture_device, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("Failed to open capture device!")
        pygame.quit()
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Capture ready: {w}x{h} @ {fps:.0f} FPS")

    # Flush buffer
    for _ in range(30):
        cap.grab()

    # Test 1: Color pattern test
    print("\n--- TEST 1: Color Patterns ---")
    colors = [
        ("BLACK", (0, 0, 0)),
        ("WHITE", (255, 255, 255)),
        ("RED", (255, 0, 0)),
        ("GREEN", (0, 255, 0)),
        ("BLUE", (0, 0, 255)),
        ("CYAN", (0, 255, 255)),
        ("MAGENTA", (255, 0, 255)),
        ("YELLOW", (255, 255, 0)),
    ]

    color_results = []
    for name, color in colors:
        screen.fill(color)
        pygame.display.flip()
        time.sleep(0.3)

        # Flush and capture
        for _ in range(10):
            cap.grab()
        ret, frame = cap.read()

        if ret:
            # Sample center
            cy, cx = frame.shape[0]//2, frame.shape[1]//2
            sample = frame[cy-20:cy+20, cx-20:cx+20]
            avg = np.mean(sample, axis=(0, 1))
            # OpenCV is BGR
            received = (int(avg[2]), int(avg[1]), int(avg[0]))

            diff = sum(abs(a-b) for a, b in zip(color, received))
            ok = diff < 100

            print(f"  {name}: sent {color} -> received {received} {'OK' if ok else 'FAIL'}")
            color_results.append(ok)
        else:
            print(f"  {name}: capture failed")
            color_results.append(False)

    color_success = sum(color_results) / len(color_results)
    print(f"  Color test: {color_success*100:.0f}% success")

    # Test 2: Grid pattern test (simulates data encoding)
    print("\n--- TEST 2: Grid Pattern ---")
    cell_size = 20
    grid_w = disp_w // cell_size
    grid_h = disp_h // cell_size

    # Create a checkerboard pattern
    pattern = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    for y in range(grid_h):
        for x in range(grid_w):
            if (x + y) % 2 == 0:
                pattern[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size] = [255, 255, 255]

    # Display pattern
    surface = pygame.surfarray.make_surface(np.transpose(pattern, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    time.sleep(0.5)

    # Flush and capture
    for _ in range(15):
        cap.grab()
    ret, frame = cap.read()

    if ret:
        # Resize captured frame to match display
        frame_resized = cv2.resize(frame, (disp_w, disp_h))

        # Convert to grayscale
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Sample grid cells
        correct = 0
        total = 0
        for y in range(grid_h):
            for x in range(grid_w):
                cy = y * cell_size + cell_size // 2
                cx = x * cell_size + cell_size // 2
                sample = gray[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
                avg = np.mean(sample)

                expected_white = (x + y) % 2 == 0
                is_white = avg > 128

                if expected_white == is_white:
                    correct += 1
                total += 1

        grid_accuracy = correct / total
        print(f"  Grid cells: {correct}/{total} correct ({grid_accuracy*100:.1f}%)")
    else:
        print("  Grid test: capture failed")
        grid_accuracy = 0

    # Test 3: FPS test
    print("\n--- TEST 3: Transfer FPS ---")

    # Rapidly display and capture
    frames_sent = 0
    frames_received = 0
    start = time.time()
    test_duration = 3

    while time.time() - start < test_duration:
        # Display a frame number as brightness
        brightness = (frames_sent % 256)
        screen.fill((brightness, brightness, brightness))
        pygame.display.flip()
        frames_sent += 1

        # Non-blocking capture
        ret, _ = cap.read()
        if ret:
            frames_received += 1

        # Small delay to not overwhelm
        time.sleep(0.01)

    elapsed = time.time() - start
    send_fps = frames_sent / elapsed
    recv_fps = frames_received / elapsed

    print(f"  Send FPS: {send_fps:.1f}")
    print(f"  Receive FPS: {recv_fps:.1f}")
    print(f"  Effective throughput: {recv_fps:.1f} FPS")

    # Cleanup
    cap.release()
    pygame.quit()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  Color accuracy: {color_success*100:.0f}%")
    print(f"  Grid accuracy: {grid_accuracy*100:.1f}%")
    print(f"  Capture FPS: {recv_fps:.1f}")

    overall = color_success >= 0.8 and grid_accuracy >= 0.9 and recv_fps >= 25

    if overall:
        print("\n  STATUS: PASS - System ready for data transfer")
        print(f"\n  Recommended settings:")
        print(f"    - Capture device: {capture_device}")
        print(f"    - Display: {display_index}")
        print(f"    - Backend: MSMF")
        print(f"    - Expected FPS: {recv_fps:.0f}")
    else:
        print("\n  STATUS: FAIL - Issues detected")
        if color_success < 0.8:
            print("    - Color detection issues - check HDMI connection")
        if grid_accuracy < 0.9:
            print("    - Grid detection issues - may need larger cell size")
        if recv_fps < 25:
            print("    - Low FPS - check USB 3.0 connection")

    return overall


if __name__ == "__main__":
    # Parse arguments
    capture_device = 3  # Default to device 3 (1080p capture card)
    display_index = 1   # Default to display 1

    if len(sys.argv) > 1:
        capture_device = int(sys.argv[1])
    if len(sys.argv) > 2:
        display_index = int(sys.argv[2])

    success = test_simple_transfer(capture_device, display_index)
    sys.exit(0 if success else 1)
