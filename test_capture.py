"""
Diagnostic test for capture device.
Tests raw capture FPS without any processing.
"""
import cv2
import time
import sys

def test_capture_device(device_index=0, duration=5):
    """Test capture device raw FPS."""
    print(f"\n{'='*60}")
    print(f"Testing capture device {device_index}")
    print(f"{'='*60}")

    # Try different backends
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
    ]

    best_result = None

    for backend_name, backend in backends:
        print(f"\n--- Testing {backend_name} backend ---")

        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            print(f"  Failed to open with {backend_name}")
            continue

        # Try MJPEG format for high FPS
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get actual settings
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        try:
            fourcc_str = ''.join([chr((fourcc >> 8*i) & 0xFF) for i in range(4) if 32 <= ((fourcc >> 8*i) & 0xFF) < 127])
        except:
            fourcc_str = "unknown"

        print(f"  Resolution: {w}x{h}")
        print(f"  Reported FPS: {fps:.1f}")
        print(f"  Format: {fourcc_str or 'raw'}")

        # Test raw capture speed
        print(f"  Testing capture for {duration} seconds...")

        # Warm up - flush buffer
        for _ in range(10):
            cap.grab()

        start = time.time()
        frames = 0
        while time.time() - start < duration:
            ret, frame = cap.read()
            if ret and frame is not None:
                frames += 1

        elapsed = time.time() - start
        actual_fps = frames / elapsed if elapsed > 0 else 0

        print(f"  Captured {frames} frames in {elapsed:.2f}s")
        print(f"  ACTUAL FPS: {actual_fps:.1f}")

        cap.release()

        if best_result is None or actual_fps > best_result[1]:
            best_result = (backend_name, actual_fps, w, h)

    return best_result

def list_devices():
    """List available capture devices."""
    print("\n" + "="*60)
    print("Scanning for capture devices...")
    print("="*60)

    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Try to get 1080p
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            devices.append((i, w, h))
            cap.release()

    # Print sorted by resolution (higher = more likely capture card)
    devices.sort(key=lambda x: x[1]*x[2], reverse=True)
    for i, w, h in devices:
        marker = " <-- likely capture card" if w >= 1920 else ""
        print(f"  Device {i}: {w}x{h}{marker}")

    if not devices:
        print("  No devices found!")

    return devices

def test_display():
    """Test pygame display output."""
    print("\n" + "="*60)
    print("Testing display output...")
    print("="*60)

    try:
        import pygame
        pygame.init()

        # Get displays
        num_displays = pygame.display.get_num_displays()
        print(f"  Found {num_displays} displays")

        sizes = pygame.display.get_desktop_sizes()
        for i, (w, h) in enumerate(sizes):
            print(f"  Display {i}: {w}x{h}")

        pygame.quit()
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_roundtrip(device_index, display_index=1):
    """Test sending a pattern and receiving it back."""
    print("\n" + "="*60)
    print(f"Testing roundtrip: Display {display_index} -> Capture Device {device_index}")
    print("="*60)

    import pygame
    import numpy as np

    # Initialize pygame
    pygame.init()

    # Setup display on secondary monitor
    try:
        sizes = pygame.display.get_desktop_sizes()
        if display_index >= len(sizes):
            print(f"  Display {display_index} not found, using 0")
            display_index = 0

        x_offset = sum(sizes[i][0] for i in range(display_index))
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},0"

        screen = pygame.display.set_mode(
            (sizes[display_index][0], sizes[display_index][1]),
            pygame.FULLSCREEN | pygame.HWSURFACE
        )
        pygame.mouse.set_visible(False)
    except Exception as e:
        print(f"  Display setup failed: {e}")
        pygame.quit()
        return False

    # Open capture
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  Failed to open capture device {device_index}")
        pygame.quit()
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Flush capture buffer
    for _ in range(10):
        cap.grab()

    print("  Testing color patterns...")

    # Test patterns: Red, Green, Blue, White
    patterns = [
        ("RED", (255, 0, 0)),
        ("GREEN", (0, 255, 0)),
        ("BLUE", (0, 0, 255)),
        ("WHITE", (255, 255, 255)),
        ("CYAN", (0, 255, 255)),
        ("MAGENTA", (255, 0, 255)),
    ]

    results = []

    for name, color in patterns:
        # Display solid color
        screen.fill(color)
        pygame.display.flip()

        # Wait for display to update and capture to receive
        time.sleep(0.5)

        # Flush and capture
        for _ in range(5):
            cap.grab()

        ret, frame = cap.read()
        if ret and frame is not None:
            # Sample center of frame (BGR format from OpenCV)
            h, w = frame.shape[:2]
            center = frame[h//2-50:h//2+50, w//2-50:w//2+50]
            avg_bgr = np.mean(center, axis=(0, 1))
            avg_rgb = (avg_bgr[2], avg_bgr[1], avg_bgr[0])

            # Check if color matches
            diff = sum(abs(a - b) for a, b in zip(avg_rgb, color))
            match = diff < 150

            print(f"    {name}: sent {color}, received ({avg_rgb[0]:.0f}, {avg_rgb[1]:.0f}, {avg_rgb[2]:.0f}) - {'OK' if match else 'MISMATCH'}")
            results.append(match)
        else:
            print(f"    {name}: capture failed")
            results.append(False)

    cap.release()
    pygame.quit()

    success_rate = sum(results) / len(results)
    print(f"\n  Success rate: {success_rate*100:.0f}%")

    return success_rate > 0.8

if __name__ == "__main__":
    # List devices
    devices = list_devices()

    # Test display
    test_display()

    if devices:
        # Find the highest resolution device (likely capture card)
        capture_device = devices[0][0]  # Highest res first after sorting

        if len(sys.argv) > 1:
            capture_device = int(sys.argv[1])

        print(f"\nUsing device {capture_device} for testing")

        # Test capture FPS
        result = test_capture_device(capture_device)

        if result:
            backend, fps, w, h = result
            print("\n" + "="*60)
            print("CAPTURE TEST RESULTS:")
            print("="*60)
            print(f"  Best backend: {backend}")
            print(f"  Resolution: {w}x{h}")
            print(f"  FPS: {fps:.1f}")

            if fps >= 25:
                print("  Status: GOOD - sufficient for data transfer")
            elif fps >= 15:
                print("  Status: OK - transfer will work but slower")
            else:
                print("  Status: POOR - check USB 3.0 connection")

        # Ask about roundtrip test
        print("\n" + "="*60)
        print("Ready for roundtrip test?")
        print("This will display patterns on Display 1 and capture them.")
        print("Press Enter to continue or Ctrl+C to skip...")
        try:
            input()
            test_roundtrip(capture_device, display_index=1)
        except KeyboardInterrupt:
            print("\nSkipped roundtrip test")
