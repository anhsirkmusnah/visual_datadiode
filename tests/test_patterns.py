"""
Visual Data Diode - Test Patterns and Validation

Provides test pattern generation and validation tools for
calibrating and debugging the visual transfer system.
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    FRAME_WIDTH, FRAME_HEIGHT,
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE,
    GRAY_LEVELS, DEFAULT_FPS
)
from sender.encoder import FrameEncoder, TestPatternGenerator
from sender.chunker import FileChunker
from shared.block import Block, BlockHeader, BlockFlags


class TestSuite:
    """
    Comprehensive test suite for the visual data diode system.
    """

    def __init__(self, profile=PROFILE_STANDARD):
        self.profile = profile
        self.encoder = FrameEncoder(profile)
        self.pattern_gen = TestPatternGenerator(profile)

    def generate_grayscale_test(self) -> np.ndarray:
        """Generate horizontal grayscale gradient pattern."""
        return self.pattern_gen.grayscale_gradient()

    def generate_level_test(self) -> np.ndarray:
        """Generate 4-level band test pattern."""
        return self.pattern_gen.level_test()

    def generate_checkerboard(self, cell_size: int = 2) -> np.ndarray:
        """Generate checkerboard alignment pattern."""
        return self.pattern_gen.checkerboard(cell_size)

    def generate_sync_frame(self) -> np.ndarray:
        """Generate sync-only frame (no data)."""
        grid = self.encoder.encode_sync_only()
        return self.encoder.grid_to_frame(grid)

    def generate_test_block(self, block_index: int = 0, payload_size: int = 100) -> np.ndarray:
        """Generate a frame with a test block."""
        # Create test payload
        payload = bytes(range(256)) * (payload_size // 256 + 1)
        payload = payload[:payload_size]

        header = BlockHeader(
            session_id=0x12345678,
            block_index=block_index,
            total_blocks=10,
            file_size=1000,
            payload_size=len(payload),
            flags=BlockFlags.NONE
        )

        block = Block(header=header, payload=payload)
        return self.encoder.encode_to_frame(block)

    def save_test_patterns(self, output_dir: str):
        """Save all test patterns as PNG files."""
        try:
            from PIL import Image
        except ImportError:
            print("PIL required for saving images. Install with: pip install Pillow")
            return

        os.makedirs(output_dir, exist_ok=True)

        patterns = [
            ("grayscale_gradient.png", self.generate_grayscale_test()),
            ("level_test.png", self.generate_level_test()),
            ("checkerboard.png", self.generate_checkerboard()),
            ("sync_frame.png", self.generate_sync_frame()),
            ("test_block.png", self.generate_test_block()),
        ]

        for filename, frame in patterns:
            path = os.path.join(output_dir, filename)
            img = Image.fromarray(frame)
            img.save(path)
            print(f"Saved: {path}")


class CorruptionSimulator:
    """
    Simulates various corruption scenarios for testing.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def add_noise(self, frame: np.ndarray, stddev: float = 10.0) -> np.ndarray:
        """Add Gaussian noise to frame."""
        noise = self.rng.normal(0, stddev, frame.shape).astype(np.float32)
        result = frame.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def add_salt_pepper(self, frame: np.ndarray, ratio: float = 0.01) -> np.ndarray:
        """Add salt and pepper noise."""
        result = frame.copy()
        h, w = frame.shape[:2]
        num_pixels = int(h * w * ratio)

        # Salt (white)
        salt_coords = (
            self.rng.integers(0, h, num_pixels),
            self.rng.integers(0, w, num_pixels)
        )
        result[salt_coords] = 255

        # Pepper (black)
        pepper_coords = (
            self.rng.integers(0, h, num_pixels),
            self.rng.integers(0, w, num_pixels)
        )
        result[pepper_coords] = 0

        return result

    def blur_frame(self, frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply Gaussian blur."""
        try:
            import cv2
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        except ImportError:
            # Simple box blur fallback
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            result = np.zeros_like(frame)
            for c in range(3):
                result[:, :, c] = np.convolve(
                    frame[:, :, c].flatten(),
                    kernel.flatten(),
                    mode='same'
                ).reshape(frame.shape[:2])
            return result

    def drop_region(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """Black out a region of the frame."""
        result = frame.copy()
        result[y:y+h, x:x+w] = 0
        return result

    def shift_colors(
        self,
        frame: np.ndarray,
        r_shift: int = 0, g_shift: int = 0, b_shift: int = 0
    ) -> np.ndarray:
        """Shift color channels."""
        result = frame.astype(np.int16)
        result[:, :, 0] = np.clip(result[:, :, 0] + r_shift, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] + g_shift, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] + b_shift, 0, 255)
        return result.astype(np.uint8)


class FrameDropSimulator:
    """
    Simulates frame drops in a transmission.
    """

    def __init__(self, drop_rate: float = 0.1, burst_length: int = 1, seed: int = None):
        """
        Initialize simulator.

        Args:
            drop_rate: Probability of starting a drop
            burst_length: Average number of consecutive frames to drop
            seed: Random seed
        """
        self.drop_rate = drop_rate
        self.burst_length = burst_length
        self.rng = np.random.default_rng(seed)
        self._drops_remaining = 0

    def should_drop(self) -> bool:
        """Check if current frame should be dropped."""
        if self._drops_remaining > 0:
            self._drops_remaining -= 1
            return True

        if self.rng.random() < self.drop_rate:
            self._drops_remaining = max(0, int(self.rng.exponential(self.burst_length)))
            return True

        return False

    def filter_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Filter frames, dropping some randomly."""
        return [f for f in frames if not self.should_drop()]


class EndToEndTester:
    """
    End-to-end testing framework for the visual data diode.
    """

    def __init__(self, profile=PROFILE_STANDARD):
        self.profile = profile
        self.encoder = FrameEncoder(profile)
        self.results = []

    def test_encode_decode_cycle(self, data: bytes) -> Tuple[bool, str]:
        """
        Test encoding and decoding of data.

        Returns:
            (success, message)
        """
        from receiver.decoder import FrameDecoder

        decoder = FrameDecoder(self.profile)

        # Create block
        header = BlockHeader(
            session_id=0xDEADBEEF,
            block_index=0,
            total_blocks=1,
            file_size=len(data),
            payload_size=len(data),
            flags=BlockFlags.FIRST_BLOCK | BlockFlags.LAST_BLOCK
        )

        from shared.fec import SimpleFEC
        fec = SimpleFEC(0.10)

        block = Block(header=header, payload=data)
        block_bytes = header.pack() + data
        _, parity = fec.encode(block_bytes)
        block.fec_parity = parity

        # Encode to frame
        frame = self.encoder.encode_to_frame(block)

        # Simulate capture (add slight noise)
        simulator = CorruptionSimulator(seed=42)
        captured = simulator.add_noise(frame, stddev=5.0)

        # Decode (need to extract grid first)
        from receiver.sync import FrameSync
        sync = FrameSync(self.profile)

        sync_result = sync.detect_sync(captured)

        if not sync_result.is_synced:
            return False, "Sync failed"

        grid = sync.extract_grid(captured, sync_result)

        if grid is None:
            return False, "Grid extraction failed"

        decode_result = decoder.decode_grid(grid)

        if not decode_result.success:
            return False, f"Decode failed: {decode_result.message}"

        if decode_result.block.payload != data:
            return False, "Payload mismatch"

        return True, f"Success (FEC corrected {decode_result.fec_corrected} errors)"

    def run_stress_test(self, num_blocks: int = 100, payload_size: int = 500) -> dict:
        """
        Run stress test with many blocks.

        Returns:
            Test results dictionary
        """
        successes = 0
        failures = 0
        fec_corrections = 0

        for i in range(num_blocks):
            # Random payload
            payload = os.urandom(payload_size)
            success, message = self.test_encode_decode_cycle(payload)

            if success:
                successes += 1
                if "corrected" in message:
                    try:
                        corrections = int(message.split()[-2])
                        fec_corrections += corrections
                    except Exception:
                        pass
            else:
                failures += 1

        return {
            'total': num_blocks,
            'successes': successes,
            'failures': failures,
            'success_rate': successes / num_blocks,
            'total_fec_corrections': fec_corrections
        }


def run_calibration_test():
    """Interactive calibration test."""
    print("Visual Data Diode - Calibration Test")
    print("=" * 50)

    suite = TestSuite()

    print("\nGenerating test patterns...")
    output_dir = "test_output"
    suite.save_test_patterns(output_dir)

    print(f"\nTest patterns saved to: {output_dir}/")
    print("\nDisplay these patterns on the sender and verify:")
    print("1. Sync border is visible (cyan/magenta)")
    print("2. Four gray levels are distinguishable")
    print("3. Corner markers are clear")


def run_unit_tests():
    """Run unit tests for core components."""
    print("Visual Data Diode - Unit Tests")
    print("=" * 50)

    # Test 1: Encoding capacity
    print("\n[Test 1] Encoding Capacity")
    for profile in [PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE]:
        encoder = FrameEncoder(profile)
        cells, bytes_cap = encoder.get_capacity()
        throughput = bytes_cap * DEFAULT_FPS
        print(f"  {profile.name}: {cells} cells, {bytes_cap} bytes/frame, {throughput/1024:.1f} KB/s")

    # Test 2: Block packing/unpacking
    print("\n[Test 2] Block Pack/Unpack")
    header = BlockHeader(
        session_id=0x12345678,
        block_index=42,
        total_blocks=100,
        file_size=50000,
        payload_size=100,
        flags=BlockFlags.ENCRYPTED
    )
    packed = header.pack()
    unpacked = BlockHeader.unpack(packed)

    assert unpacked.session_id == header.session_id
    assert unpacked.block_index == header.block_index
    assert unpacked.total_blocks == header.total_blocks
    assert unpacked.flags == header.flags
    print("  Header pack/unpack: PASS")

    # Test 3: FEC
    print("\n[Test 3] FEC Encode/Decode")
    from shared.fec import SimpleFEC, check_fec_available

    if check_fec_available():
        fec = SimpleFEC(0.10)
        original = b"Hello, World! This is a test of FEC encoding." * 10
        _, parity = fec.encode(original)

        # Corrupt some bytes
        corrupted = bytearray(original)
        corrupted[10] ^= 0xFF
        corrupted[50] ^= 0xFF
        corrupted = bytes(corrupted)

        recovered, errors = fec.decode(corrupted, parity)
        assert recovered == original
        print(f"  FEC correction: PASS (corrected {errors} errors)")
    else:
        print("  FEC not available (reedsolo not installed)")

    # Test 4: Grayscale quantization
    print("\n[Test 4] Grayscale Quantization")
    from shared.constants import gray_to_bits, bits_to_gray

    for level in range(4):
        gray = bits_to_gray(level)
        recovered = gray_to_bits(gray)
        assert recovered == level
    print("  Grayscale levels: PASS")

    print("\n" + "=" * 50)
    print("All unit tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual Data Diode Test Suite")
    parser.add_argument(
        "--mode",
        choices=["calibration", "unit", "stress"],
        default="unit",
        help="Test mode"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=100,
        help="Number of blocks for stress test"
    )

    args = parser.parse_args()

    if args.mode == "calibration":
        run_calibration_test()
    elif args.mode == "unit":
        run_unit_tests()
    elif args.mode == "stress":
        print("Running stress test...")
        tester = EndToEndTester()
        results = tester.run_stress_test(num_blocks=args.blocks)
        print(f"\nResults:")
        print(f"  Total blocks: {results['total']}")
        print(f"  Successes: {results['successes']}")
        print(f"  Failures: {results['failures']}")
        print(f"  Success rate: {results['success_rate']:.1%}")
        print(f"  FEC corrections: {results['total_fec_corrections']}")
