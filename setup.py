#!/usr/bin/env python3
"""
Visual Data Diode - Setup Script

Installs dependencies and prepares the encoder environment.

Usage:
    python setup.py          # Install dependencies
    python setup.py --check  # Check installation
    python setup.py --test   # Run basic test
"""

import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check Python version is 3.8+."""
    if sys.version_info < (3, 8):
        print(f"ERROR: Python 3.8+ required, got {sys.version}")
        return False
    print(f"Python: {sys.version}")
    return True


def install_dependencies():
    """Install required Python packages."""
    print("\nInstalling dependencies...")

    packages = [
        "numpy",
        "opencv-python",
        "reedsolo",  # Reed-Solomon FEC
    ]

    for package in packages:
        print(f"  Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr}")
            return False
        print(f"    OK")

    return True


def check_dependencies():
    """Check all dependencies are installed."""
    print("\nChecking dependencies...")

    required = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "reedsolo": "reedsolo",
    }

    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  {package}: OK")
        except ImportError:
            print(f"  {package}: MISSING")
            all_ok = False

    return all_ok


def check_cuda_decoder():
    """Check if CUDA decoder is built."""
    print("\nChecking CUDA decoder...")

    decoder_paths = [
        Path(__file__).parent / "decoder_cuda/build/Release/vdd_decode.exe",
        Path(__file__).parent / "decoder_cuda/build/vdd_decode.exe",
    ]

    for path in decoder_paths:
        if path.exists():
            print(f"  Found: {path}")
            return True

    print("  NOT FOUND - Build with: cd decoder_cuda && build.bat")
    return False


def run_basic_test():
    """Run a basic encode-decode test."""
    print("\nRunning basic test...")

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from shared import PROFILE_STANDARD, Block, BlockHeader, BlockFlags
        from sender.encoder import FrameEncoder
        import numpy as np

        # Create encoder
        encoder = FrameEncoder(PROFILE_STANDARD)
        print(f"  Profile: {PROFILE_STANDARD.name}")
        print(f"  Grid: {encoder.grid_width}x{encoder.grid_height}")

        # Create test block
        test_data = b"Hello Visual Data Diode!" * 100
        header = BlockHeader(
            session_id=12345,
            block_index=0,
            total_blocks=1,
            file_size=len(test_data),
            payload_size=len(test_data),
            flags=BlockFlags.FIRST_BLOCK | BlockFlags.LAST_BLOCK
        )
        block = Block(header=header, payload=test_data)

        # Encode
        frame = encoder.encode_to_frame(block)
        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")

        # Verify frame has sync border colors
        # Top-left corner should be white (corner marker)
        assert frame[0, 0, 0] == 255, "Corner marker not white"

        print("  Basic test: PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_usage():
    """Print usage information."""
    print("""
Visual Data Diode - Encoder Setup Complete

ENCODING FILES:
    python batch_encode.py <input_file_or_folder> <output_folder>
    python batch_encode.py ./files_to_send ./videos --profile standard --fps 60

TESTING INTEGRITY:
    python test_integrity.py --small           # 10MB test
    python test_integrity.py --size 100MB      # Custom size
    python test_integrity.py --all             # Full test suite

DECODING (requires CUDA decoder):
    vdd_decode.exe -i video.avi -o ./output/ -p standard

PROFILES:
    conservative  - 16px cells, ~200 kbps, most reliable
    standard      - 10px cells, ~350 kbps, balanced (default)
    aggressive    - 8px cells, ~550 kbps, faster
    ultra         - 6px cells, ~1000 kbps, fastest

For more help: python batch_encode.py --help
""")


def main():
    parser = argparse.ArgumentParser(description="Visual Data Diode Setup")
    parser.add_argument('--check', action='store_true', help='Check installation')
    parser.add_argument('--test', action='store_true', help='Run basic test')
    parser.add_argument('--install', action='store_true', help='Install dependencies')

    args = parser.parse_args()

    print("=" * 60)
    print("VISUAL DATA DIODE - SETUP")
    print("=" * 60)

    # Check Python
    if not check_python_version():
        sys.exit(1)

    if args.install or (not args.check and not args.test):
        # Install dependencies
        if not install_dependencies():
            print("\nERROR: Failed to install dependencies")
            sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("\nERROR: Missing dependencies. Run: python setup.py --install")
        sys.exit(1)

    # Check CUDA decoder
    check_cuda_decoder()

    if args.test:
        # Run test
        if not run_basic_test():
            sys.exit(1)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    print_usage()


if __name__ == "__main__":
    main()
