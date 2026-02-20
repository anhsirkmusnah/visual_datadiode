#!/usr/bin/env python3
"""
MS2131 Raw Capture Validation Test v2

Tests PiBOX VC304D (MS2131) capture card through HDMI pipeline.
Validates pixel-level accuracy to determine maximum encoding throughput.

Usage:
    python test_ms2131_raw.py
    python test_ms2131_raw.py --device 2 --display 1
"""

import cv2
import time
import sys
import os
import argparse
import numpy as np

FRAME_W, FRAME_H = 1920, 1080
CAPTURE_DEVICE = 2  # "USB3 Video" = PiBOX MS2131


def open_capture(device_idx, fmt='YUY2'):
    """Open capture device with specified format."""
    cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None

    if fmt == 'YUY2':
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', '2'))
    elif fmt == 'MJPG':
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def flush_cap(cap, n=10):
    for _ in range(n):
        cap.grab()


def drain_until_fresh(cap, settle_time=0.5, drain_time=2.0):
    """Wait for display to update, then drain ALL buffered frames.
    Returns the very last (freshest) frame."""
    time.sleep(settle_time)
    # Read continuously for drain_time seconds, keeping only the last frame
    last_frame = None
    start = time.time()
    count = 0
    while time.time() - start < drain_time:
        ret, frame = cap.read()
        if ret and frame is not None:
            last_frame = frame
            count += 1
    return last_frame, count


def setup_pygame_display(display_index):
    """Setup pygame on secondary display with verification.
    Tries multiple strategies to ensure the window appears on the correct display."""
    import pygame
    import ctypes

    # Make process DPI-aware so coordinates match Windows virtual desktop
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    pygame.init()

    num = pygame.display.get_num_displays()
    sizes = pygame.display.get_desktop_sizes()

    print(f"  Pygame sees {num} display(s):")
    for i, (w, h) in enumerate(sizes):
        tag = " <-- TARGET" if i == display_index else ""
        print(f"    [{i}] {w}x{h}{tag}")

    if display_index >= num:
        display_index = num - 1

    # Strategy 1: Use pygame's desktop sizes to compute offset
    # This is what the existing sender/renderer.py uses
    pg_x_offset = sum(sizes[i][0] for i in range(display_index))
    tw, th = sizes[display_index]

    # Strategy 2: Also get Win32 monitor coordinates for comparison
    try:
        from ctypes import wintypes
        monitors = []
        def _cb(hMon, hdc, lprc, data):
            r = lprc.contents
            monitors.append({'left': r.left, 'top': r.top,
                           'width': r.right - r.left, 'height': r.bottom - r.top})
            return True
        MEPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(wintypes.RECT),
                                     ctypes.POINTER(ctypes.c_int))
        ctypes.windll.user32.EnumDisplayMonitors(None, None, MEPROC(_cb), 0)
        if display_index < len(monitors):
            win32_x = monitors[display_index]['left']
            win32_y = monitors[display_index]['top']
            win32_w = monitors[display_index]['width']
            win32_h = monitors[display_index]['height']
        else:
            win32_x, win32_y, win32_w, win32_h = pg_x_offset, 0, tw, th
    except Exception:
        win32_x, win32_y, win32_w, win32_h = pg_x_offset, 0, tw, th

    print(f"  Pygame offset: x={pg_x_offset}, size={tw}x{th}")
    print(f"  Win32 offset:  x={win32_x}, y={win32_y}, size={win32_w}x{win32_h}")

    # Try each strategy until we find one that works
    strategies = [
        ("pygame offset", pg_x_offset, 0, tw, th),
        ("win32 offset", win32_x, win32_y, win32_w, win32_h),
        ("pygame fullscreen display=N", None, None, tw, th),  # special case
    ]

    for name, x, y, w, h in strategies:
        print(f"\n  Trying strategy: {name}...")
        pygame.quit()
        pygame.init()

        if x is not None:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
            print(f"    SDL_VIDEO_WINDOW_POS = {x},{y}")
            screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
        else:
            # Try the display= parameter with fullscreen
            try:
                screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN | pygame.NOFRAME,
                                                  display=display_index)
                print(f"    Using display={display_index} parameter")
            except TypeError:
                continue

        pygame.mouse.set_visible(False)

        # Paint it bright magenta to visually confirm
        screen.fill((255, 0, 255))
        pygame.display.flip()
        pygame.event.pump()

        print(f"    Window created: {pygame.display.get_window_size()}")
        print(f"    Displayed MAGENTA - check your HDMI display!")

        # Return this screen - caller will verify via capture
        return screen, pygame

    return None, pygame


def verify_display_link(screen, pg, cap):
    """Verify the HDMI capture is actually seeing the pygame display.
    Uses time-based drain to handle deep DirectShow buffers."""
    print("\n  Verifying display->capture link...")
    print("  (Using 2s drain per color to defeat DirectShow buffering)")

    tests = [
        ("BLACK", (0, 0, 0)),
        ("WHITE", (255, 255, 255)),
        ("RED",   (255, 0, 0)),
        ("GREEN", (0, 255, 0)),
        ("BLUE",  (0, 0, 255)),
    ]

    captured = {}

    for name, color in tests:
        screen.fill(color)
        pg.display.flip()
        pg.event.pump()

        frame, n_drained = drain_until_fresh(cap, settle_time=0.3, drain_time=2.5)
        if frame is None:
            print(f"    {name}: CAPTURE FAILED")
            return False

        h, w = frame.shape[:2]
        center = frame[h//3:2*h//3, w//3:2*w//3]
        avg_bgr = np.mean(center, axis=(0, 1))
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        avg_gray = np.mean(gray)
        captured[name] = avg_gray

        print(f"    {name:6s}: sent RGB{color} -> captured BGR({avg_bgr[0]:.0f},{avg_bgr[1]:.0f},{avg_bgr[2]:.0f}), luma={avg_gray:.1f}  (drained {n_drained} frames)")

    # Check contrast
    bk = captured.get('BLACK', 0)
    wh = captured.get('WHITE', 0)
    contrast = wh - bk
    print(f"\n    BLACK luma={bk:.1f}, WHITE luma={wh:.1f}, CONTRAST={contrast:.1f}")

    if contrast < 50:
        print("    *** LINK BROKEN: Capture is NOT seeing the pygame display! ***")
        print("    Check: Is the HDMI cable plugged in? Is the extended display active?")
        return False
    elif contrast < 150:
        print("    *** WARNING: Low contrast. Possible Limited Range (16-235) output. ***")
        return True
    else:
        print("    LINK VERIFIED: Capture sees the display output correctly.")
        return True


def detect_color_range(screen, pg, cap):
    """Detect if HDMI is outputting Full Range (0-255) or Limited Range (16-235)."""
    print("\n" + "=" * 70)
    print("PHASE 2: COLOR RANGE DETECTION")
    print("=" * 70)

    # Send extreme values and see what arrives
    test_values = [0, 16, 32, 64, 128, 192, 224, 235, 240, 255]
    results = []

    for val in test_values:
        screen.fill((val, val, val))
        pg.display.flip()
        pg.event.pump()

        frame, _ = drain_until_fresh(cap, settle_time=0.2, drain_time=1.5)
        if frame is None:
            results.append((val, -1))
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center = gray[400:700, 600:1300]
        avg = np.mean(center)
        std = np.std(center)
        results.append((val, avg, std))

    print(f"\n  {'Sent':>5s} | {'Received':>9s} | {'Noise':>6s} | Notes")
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*6}-+-------")

    for r in results:
        sent = r[0]
        if r[1] < 0:
            print(f"  {sent:5d} | {'FAIL':>9s} |")
            continue
        recv, std = r[1], r[2]
        notes = ""
        if sent == 0 and recv > 10:
            notes = "Black not reaching 0 -> Limited Range?"
        if sent == 255 and recv < 245:
            notes = "White not reaching 255 -> Limited Range?"
        if sent == 16 and abs(recv - 0) < 5:
            notes = "16->0 = Limited Range mapping"
        if sent == 235 and abs(recv - 255) < 10:
            notes = "235->255 = Limited Range mapping"
        print(f"  {sent:5d} | {recv:9.1f} | {std:6.2f} | {notes}")

    # Determine range type
    sent_0_recv = next((r[1] for r in results if r[0] == 0), -1)
    sent_255_recv = next((r[1] for r in results if r[0] == 255), -1)
    sent_16_recv = next((r[1] for r in results if r[0] == 16), -1)
    sent_235_recv = next((r[1] for r in results if r[0] == 235), -1)

    if sent_0_recv >= 0 and sent_255_recv >= 0:
        dynamic_range = sent_255_recv - sent_0_recv
        print(f"\n  Dynamic range: {sent_0_recv:.0f} to {sent_255_recv:.0f} (span={dynamic_range:.0f})")

        if dynamic_range > 230:
            print("  RESULT: FULL RANGE (0-255) - Excellent for data encoding!")
            return 'full'
        elif dynamic_range > 180:
            print("  RESULT: Likely LIMITED RANGE (16-235) - Some values clipped")
            print("  ACTION: Set AMD Adrenalin -> Display -> Color -> Pixel Format = 'RGB 4:4:4 Full'")
            return 'limited'
        else:
            print("  RESULT: POOR RANGE - Check HDMI settings")
            return 'poor'

    return 'unknown'


def test_256_gray_levels(screen, pg, cap):
    """Test all 256 gray levels for pixel accuracy."""
    print("\n" + "=" * 70)
    print("PHASE 3: ALL 256 GRAY LEVELS")
    print("=" * 70)
    print("  Sweeping 0-255 gray values (drain per sample)...")
    print("  This will take ~6-7 minutes...")

    sent = []
    received = []
    noise = []

    for gray in range(256):
        screen.fill((gray, gray, gray))
        pg.display.flip()
        pg.event.pump()

        # Use shorter drain for the sweep but still time-based
        frame, _ = drain_until_fresh(cap, settle_time=0.1, drain_time=1.2)
        if frame is None:
            sent.append(gray)
            received.append(-1)
            noise.append(0)
            continue

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center = g[400:700, 600:1300]
        avg = np.mean(center)
        std = np.std(center)

        sent.append(gray)
        received.append(avg)
        noise.append(std)

        if gray % 32 == 0 or gray == 255:
            err = abs(gray - avg)
            print(f"    Sent {gray:3d} -> Received {avg:6.1f} (err={err:5.1f}, noise={std:.2f})")

    sent = np.array(sent, dtype=float)
    recv = np.array(received, dtype=float)
    valid = recv >= 0

    errors = np.abs(sent[valid] - recv[valid])
    avg_noise = np.mean(noise)

    print(f"\n  Summary:")
    print(f"    Mean error:   {np.mean(errors):.2f}")
    print(f"    Median error: {np.median(errors):.2f}")
    print(f"    Max error:    {np.max(errors):.2f}")
    print(f"    Avg noise:    {avg_noise:.2f}")
    print(f"    Min received: {np.min(recv[valid]):.1f}")
    print(f"    Max received: {np.max(recv[valid]):.1f}")

    # Transfer function: is it linear?
    if np.sum(valid) > 200:
        slope, intercept = np.polyfit(sent[valid], recv[valid], 1)
        print(f"    Transfer fn:  recv = {slope:.4f} * sent + {intercept:.2f}")
        if abs(slope - 1.0) < 0.05 and abs(intercept) < 10:
            print(f"    -> Near-perfect linear mapping!")
        elif abs(slope - (255/219)) < 0.1 and intercept < -10:
            print(f"    -> Looks like Limited Range expansion (16-235 -> 0-255)")

    return sent, recv, noise


def analyze_bit_depth(sent, recv, noise):
    """Determine maximum reliable bits per pixel."""
    print("\n" + "=" * 70)
    print("PHASE 4: RELIABLE BIT DEPTH ANALYSIS")
    print("=" * 70)

    valid = recv >= 0
    avg_noise = np.mean(np.array(noise)[valid])

    # Build lookup: for each sent value, what's received?
    mapping = {}
    for s, r in zip(sent, recv):
        if r >= 0:
            mapping[int(s)] = r

    print(f"\n  Avg pixel noise: {avg_noise:.2f}")
    print(f"  {'Bits':>4s} | {'Levels':>6s} | {'Step':>6s} | {'Min Sep':>8s} | {'Safety':>7s} | {'Status':>10s} | {'Bytes/Frame':>11s} | {'MB/s @60fps':>11s}")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*11}-+-{'-'*11}")

    best_bits = 0

    for bits in range(1, 9):
        n_levels = 2 ** bits
        step = 255.0 / (n_levels - 1) if n_levels > 1 else 255
        levels = [int(round(i * step)) for i in range(n_levels)]

        # Get received values for each level
        recv_levels = [mapping.get(lv, -1) for lv in levels]
        if any(r < 0 for r in recv_levels):
            print(f"  {bits:4d} | {n_levels:6d} | {step:6.1f} | {'N/A':>8s} | {'N/A':>7s} | {'NO DATA':>10s}")
            continue

        # Min separation between adjacent received levels
        seps = [recv_levels[i] - recv_levels[i-1] for i in range(1, len(recv_levels))]
        min_sep = min(seps)
        safety = min_sep / (2 * avg_noise) if avg_noise > 0.1 else min_sep / 0.2

        # Need safety margin > 3 for reliable decoding
        reliable = min_sep > 2.0 and safety > 3.0
        status = "RELIABLE" if reliable else "MARGINAL" if safety > 1.5 else "NO"

        bpf = FRAME_W * FRAME_H * bits // 8
        mbps = bpf * 60 / (1024 * 1024)

        print(f"  {bits:4d} | {n_levels:6d} | {step:6.1f} | {min_sep:8.2f} | {safety:7.1f}x | {status:>10s} | {bpf:>9,d} B | {mbps:>9.1f}")

        if reliable:
            best_bits = bits

    return best_bits


def test_cell_vs_pixel(screen, pg, cap, best_bits):
    """Test encoding accuracy with cells vs pixel-level at the determined bit depth."""
    print("\n" + "=" * 70)
    print("PHASE 5: CELL SIZE ACCURACY TEST")
    print("=" * 70)

    n_levels = min(2 ** best_bits, 4) if best_bits > 0 else 4
    levels = np.array([int(round(i * 255 / (n_levels - 1))) for i in range(n_levels)], dtype=np.uint8)
    bits_per_cell = int(np.log2(n_levels))

    print(f"  Testing with {n_levels} gray levels ({bits_per_cell} bits/cell): {list(levels)}")

    cell_sizes = [1, 2, 3, 4, 6, 8]

    for cs in cell_sizes:
        cols = FRAME_W // cs
        rows = FRAME_H // cs

        # Generate random data
        np.random.seed(12345)
        data = np.random.randint(0, n_levels, (rows, cols), dtype=np.uint8)
        frame_gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                frame_gray[r*cs:(r+1)*cs, c*cs:(c+1)*cs] = levels[data[r, c]]

        # Display
        rgb = np.stack([frame_gray]*3, axis=-1)
        surf = pg.surfarray.make_surface(rgb.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))
        pg.display.flip()
        pg.event.pump()

        frame, _ = drain_until_fresh(cap, settle_time=0.3, drain_time=2.0)
        if frame is None:
            print(f"  Cell {cs:2d}px: CAPTURE FAILED")
            continue

        captured_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Decode: sample center of each cell, nearest level
        margin_cells = max(3, 50 // cs)
        correct = 0
        total = 0

        for r in range(margin_cells, rows - margin_cells):
            for c in range(margin_cells, cols - margin_cells):
                cy = r * cs + cs // 2
                cx = c * cs + cs // 2
                pad = max(1, cs // 4)
                region = captured_gray[max(0,cy-pad):min(FRAME_H,cy+pad), max(0,cx-pad):min(FRAME_W,cx+pad)]
                val = np.mean(region)

                decoded = np.argmin(np.abs(val - levels.astype(float)))
                expected = data[r, c]
                if decoded == expected:
                    correct += 1
                total += 1

        acc = correct / total * 100 if total > 0 else 0
        bpf = cols * rows * bits_per_cell // 8
        mbps = bpf * 60 / (1024 * 1024)
        net_mbps = mbps * 0.9  # 10% overhead

        print(f"  Cell {cs:2d}px: {cols:4d}x{rows:4d} = {cols*rows:>9,d} cells | Accuracy: {acc:6.2f}% | {bpf:>9,d} B/frame | {net_mbps:.1f} MB/s net")


def test_pixel_level_data(screen, pg, cap, best_bits):
    """Test actual data encoding/decoding at pixel level."""
    print("\n" + "=" * 70)
    print("PHASE 6: PIXEL-LEVEL DATA TRANSFER TEST")
    print("=" * 70)

    for bits in [1, 2, min(best_bits, 4)]:
        if bits < 1:
            continue

        n_levels = 2 ** bits
        levels = np.array([int(round(i * 255 / (n_levels - 1))) for i in range(n_levels)], dtype=np.uint8)

        print(f"\n  {bits}-bit encoding ({n_levels} levels: {list(levels)})")

        # Generate random payload
        np.random.seed(99)
        pixel_data = np.random.randint(0, n_levels, (FRAME_H, FRAME_W), dtype=np.uint8)
        frame_gray = levels[pixel_data]

        rgb = np.stack([frame_gray]*3, axis=-1)
        surf = pg.surfarray.make_surface(rgb.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))
        pg.display.flip()
        pg.event.pump()

        frame, _ = drain_until_fresh(cap, settle_time=0.3, drain_time=2.0)
        if frame is None:
            print(f"    CAPTURE FAILED")
            continue

        captured_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Decode entire frame
        margin = 50
        sent_region = pixel_data[margin:-margin, margin:-margin]
        recv_region = captured_gray[margin:-margin, margin:-margin]

        # Nearest-level decode
        decoded = np.argmin(
            np.abs(recv_region[:,:,None].astype(float) - levels[None,None,:].astype(float)),
            axis=2
        ).astype(np.uint8)

        correct = np.sum(decoded == sent_region)
        total = decoded.size
        accuracy = correct / total * 100

        bpf = FRAME_W * FRAME_H * bits // 8
        mbps_raw = bpf * 60 / (1024 * 1024)

        print(f"    Accuracy: {accuracy:.4f}% ({correct:,}/{total:,})")
        print(f"    Raw capacity: {bpf:,} bytes/frame = {mbps_raw:.1f} MB/s @60fps")

        # Per-level breakdown
        for i, lv in enumerate(levels):
            mask = sent_region == i
            if np.sum(mask) > 0:
                lv_correct = np.sum(decoded[mask] == i)
                lv_total = np.sum(mask)
                lv_acc = lv_correct / lv_total * 100
                lv_recv_mean = np.mean(recv_region[mask])
                lv_recv_std = np.std(recv_region[mask])
                print(f"      Level {i} (gray={lv:3d}): acc={lv_acc:.2f}%, recv_mean={lv_recv_mean:.1f}, std={lv_recv_std:.2f}")


def throughput_summary(best_bits, fps=59.7):
    """Print throughput estimates."""
    print("\n" + "=" * 70)
    print("PHASE 7: THROUGHPUT ESTIMATES")
    print("=" * 70)

    print(f"\n  Measured FPS: {fps:.1f}")
    print(f"  Best reliable bits/pixel: {best_bits}")

    print(f"\n  {'Encoding':40s} | {'Bytes/Frame':>11s} | {'Gross MB/s':>10s} | {'Net MB/s':>8s} | {'100 GB':>8s}")
    print(f"  {'-'*40}-+-{'-'*11}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    configs = []
    for bits in [1, 2, 3, 4]:
        bpf = FRAME_W * FRAME_H * bits // 8
        gross = bpf * fps / (1024**2)
        net = gross * 0.9  # 10% FEC/overhead
        hours = (100 * 1024) / (net * 3600) if net > 0 else 999
        tag = " <-- RECOMMENDED" if bits == best_bits else ""
        time_str = f"{hours:.1f}h" if hours < 24 else f"{hours/24:.1f}d"
        print(f"  Pixel-level {bits}-bit ({2**bits:2d} levels){tag:17s} | {bpf:>9,d} B | {gross:>8.1f} | {net:>6.1f} | {time_str:>8s}")

    # Cell-based options
    for cs, bits in [(4, 2), (8, 2), (4, 1)]:
        cols, rows = FRAME_W // cs, FRAME_H // cs
        bpf = cols * rows * bits // 8
        gross = bpf * fps / (1024**2)
        net = gross * 0.9
        hours = (100 * 1024) / (net * 3600) if net > 0 else 999
        time_str = f"{hours:.1f}h" if hours < 24 else f"{hours/24:.1f}d"
        print(f"  Cell {cs}px, {bits}-bit ({2**bits} levels){' '*17} | {bpf:>9,d} B | {gross:>8.1f} | {net:>6.1f} | {time_str:>8s}")


def main():
    parser = argparse.ArgumentParser(description='MS2131 Raw Capture Validation v2')
    parser.add_argument('--device', type=int, default=CAPTURE_DEVICE, help='Capture device index (default: 2)')
    parser.add_argument('--display', type=int, default=1, help='Pygame display index (default: 1)')
    args = parser.parse_args()

    print("=" * 70)
    print("  MS2131 RAW CAPTURE VALIDATION TEST v2")
    print("  PiBOX VC304D / MacroSilicon MS2131")
    print("  Device: USB3 Video (index {})".format(args.device))
    print("=" * 70)

    # ---- PHASE 1: Setup & verify link ----
    print("\n" + "=" * 70)
    print("PHASE 1: SETUP & LINK VERIFICATION")
    print("=" * 70)

    # Set up display FIRST, then open capture to minimize stale buffer
    screen, pg = setup_pygame_display(args.display)
    if screen is None:
        print("  FATAL: Could not create display!")
        return

    # Display bright magenta and wait for it to appear on HDMI
    screen.fill((255, 0, 255))
    pg.display.flip()
    pg.event.pump()
    print("\n  Displaying MAGENTA, waiting 3s for HDMI pipeline to settle...")
    time.sleep(3.0)

    # NOW open capture (fresh buffer, no stale frames)
    cap = open_capture(args.device)
    if cap is None:
        print("  FATAL: Cannot open capture device!")
        pg.quit()
        return

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fmt_str = ''.join([chr((fourcc >> 8*i) & 0xFF) for i in range(4) if 32 <= ((fourcc >> 8*i) & 0xFF) < 127])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Capture: {w}x{h} @ {fps:.0f}fps, format={fmt_str}")

    # Drain any initial buffer by reading continuously for 2 seconds
    print("  Draining capture buffer (2s)...")
    drain_start = time.time()
    drain_count = 0
    while time.time() - drain_start < 2.0:
        cap.read()
        drain_count += 1
    print(f"  Drained {drain_count} frames")

    # NOW check if we see magenta
    ret, frame = cap.read()
    if ret and frame is not None:
        center = frame[400:700, 600:1300]
        avg = np.mean(center, axis=(0, 1))
        gray = np.mean(cv2.cvtColor(center, cv2.COLOR_BGR2GRAY))
        print(f"\n  Magenta check: captured BGR({avg[0]:.0f},{avg[1]:.0f},{avg[2]:.0f}), luma={gray:.1f}")
        if avg[0] > 150 and avg[1] < 100 and avg[2] > 150:
            print("  >>> MAGENTA DETECTED! Display->Capture link VERIFIED! <<<")
        else:
            print("  Magenta NOT detected yet. Continuing with link test...")

    link_ok = verify_display_link(screen, pg, cap)
    if not link_ok:
        print("\n  ABORTING: Display-capture link not working.")
        print("  Troubleshooting:")
        print("  1. Check HDMI cable is connected: ThinkPad HDMI OUT -> PiBOX HDMI IN")
        print("  2. Check Windows Settings -> Display: should show 2 displays")
        print("  3. The extended display should be 1920x1080")
        print("  4. Try: Win+P -> Extend")
        cap.release()
        pg.quit()
        return

    # ---- PHASE 2: Color range ----
    color_range = detect_color_range(screen, pg, cap)

    # ---- PHASE 3: 256 gray levels ----
    sent, recv, noise_arr = test_256_gray_levels(screen, pg, cap)

    # ---- PHASE 4: Bit depth analysis ----
    best_bits = analyze_bit_depth(sent, recv, noise_arr)
    print(f"\n  >>> BEST RELIABLE BIT DEPTH: {best_bits} bits/pixel ({2**best_bits} levels) <<<")

    # ---- PHASE 5: Cell size test ----
    test_cell_vs_pixel(screen, pg, cap, best_bits)

    # ---- PHASE 6: Pixel-level data test ----
    test_pixel_level_data(screen, pg, cap, best_bits)

    # ---- PHASE 7: Throughput summary ----
    throughput_summary(best_bits, fps=59.7)

    # Cleanup
    cap.release()
    pg.quit()

    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)
    if color_range == 'limited':
        print("\n  IMPORTANT: Switch to RGB Full Range for best throughput!")
        print("  AMD Adrenalin -> Display -> Color -> Pixel Format = 'RGB 4:4:4 PC Standard (Full RGB)'")
    if best_bits >= 2:
        print(f"\n  Your pipeline supports {best_bits} bits/pixel = {2**best_bits} gray levels")
        bpf = FRAME_W * FRAME_H * best_bits // 8
        net = bpf * 59.7 * 0.9 / (1024**2)
        hours = (100 * 1024) / (net * 3600)
        print(f"  Estimated: {net:.1f} MB/s net -> 100 GB in {hours:.1f} hours")


if __name__ == '__main__':
    main()
