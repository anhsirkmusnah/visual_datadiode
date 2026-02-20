#!/usr/bin/env python3
"""
End-to-end loopback test for the binary pixel streaming pipeline.

Simulates the full sender -> receiver path in-memory (no HDMI hardware):
  1. Generate a large test file (configurable, default 1 GB)
  2. Encode each block into binary frames (sender logic)
  3. Decode each frame back (receiver logic)
  4. Assemble to output file via DiskBackedBlockStore
  5. Verify SHA-256 match

Usage:
    python test_binary_e2e.py                  # 1 GB test
    python test_binary_e2e.py --size 2         # 2 GB test
    python test_binary_e2e.py --size 0.01      # 10 MB quick test
    python test_binary_e2e.py --no-fec         # without FEC
"""

import os
import sys
import time
import struct
import hashlib
import random
import argparse
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.binary_frame import (
    BinaryFrameHeader, BINARY_MAGIC, BINARY_FRAME_BYTES,
    BINARY_HEADER_SIZE, BINARY_CRC_SIZE,
    BINARY_FLAG_FIRST, BINARY_FLAG_LAST, BINARY_FLAG_METADATA,
    calculate_binary_payload_capacity,
    encode_binary_frame, decode_binary_frame,
)
from shared.fec import SimpleFEC
from receiver.binary_receiver import DiskBackedBlockStore


def generate_test_file(path: str, size_bytes: int) -> bytes:
    """Generate a test file with pseudo-random data. Returns SHA-256."""
    sha256 = hashlib.sha256()
    written = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    rng = random.Random(42)  # Deterministic seed

    with open(path, 'wb') as f:
        while written < size_bytes:
            remaining = min(chunk_size, size_bytes - written)
            chunk = rng.randbytes(remaining)
            f.write(chunk)
            sha256.update(chunk)
            written += remaining

    return sha256.digest()


def run_e2e_test(size_gb: float, fec_ratio: float, inject_errors: bool = False):
    size_bytes = int(size_gb * 1024 * 1024 * 1024)
    size_mb = size_bytes / (1024 * 1024)

    print("=" * 70)
    print("  BINARY PIXEL E2E LOOPBACK TEST")
    print("=" * 70)
    print(f"  Test size: {size_mb:.1f} MB ({size_gb:.3f} GB)")
    print(f"  FEC ratio: {fec_ratio:.0%}")
    print(f"  Error injection: {inject_errors}")

    # Setup
    fec = SimpleFEC(fec_ratio) if fec_ratio > 0.001 else None
    payload_capacity = calculate_binary_payload_capacity(fec_ratio)
    print(f"  Payload capacity: {payload_capacity:,} bytes/frame")
    print(f"  Theoretical throughput @60fps: {payload_capacity * 60 / 1024 / 1024:.1f} MB/s")

    # Metadata size (same as sender)
    test_filename = "test_e2e_data.bin"
    filename_bytes = test_filename.encode('utf-8')
    metadata_size = 32 + 2 + len(filename_bytes)  # hash + len + name
    block0_capacity = payload_capacity - metadata_size

    # Total blocks
    if size_bytes <= block0_capacity:
        total_blocks = 1
    else:
        remaining = size_bytes - block0_capacity
        total_blocks = 1 + (remaining + payload_capacity - 1) // payload_capacity

    total_frames = total_blocks
    est_frame_time = total_frames / 60.0
    print(f"  Total blocks: {total_blocks:,}")
    print(f"  Estimated @60fps: {est_frame_time:.1f}s")

    # Phase 1: Generate test file
    print(f"\n  Phase 1: Generating {size_mb:.1f} MB test file...")
    tmp_dir = tempfile.mkdtemp(prefix='vdd_e2e_')
    input_path = os.path.join(tmp_dir, test_filename)
    output_path = os.path.join(tmp_dir, "output_" + test_filename)

    t0 = time.perf_counter()
    expected_hash = generate_test_file(input_path, size_bytes)
    gen_time = time.perf_counter() - t0
    print(f"  Generated in {gen_time:.1f}s ({size_mb / gen_time:.0f} MB/s)")
    print(f"  SHA-256: {expected_hash.hex()}")

    # Build metadata payload
    metadata = expected_hash + struct.pack('<H', len(filename_bytes)) + filename_bytes

    # Phase 2: Encode -> Decode loopback
    print(f"\n  Phase 2: Encode -> Decode loopback ({total_blocks:,} blocks)...")

    session_id = random.randint(0, 0xFFFFFFFF)
    store = DiskBackedBlockStore()
    error_rng = random.Random(99)

    stats = {
        'encode_time': 0.0,
        'decode_time': 0.0,
        'crc_ok': 0,
        'fec_corrected': 0,
        'fec_corrections_total': 0,
        'decode_fail': 0,
    }

    t_start = time.perf_counter()
    last_report = t_start

    with open(input_path, 'rb') as f:
        for block_idx in range(total_blocks):
            # === SENDER SIDE ===

            # Read payload
            if block_idx == 0:
                file_data = f.read(block0_capacity)
                payload = metadata + file_data
            else:
                payload = f.read(payload_capacity)

            # Build header
            flags = 0
            if block_idx == 0:
                flags |= BINARY_FLAG_FIRST | BINARY_FLAG_METADATA
            if block_idx == total_blocks - 1:
                flags |= BINARY_FLAG_LAST

            header = BinaryFrameHeader(
                magic=BINARY_MAGIC,
                session_id=session_id,
                block_index=block_idx,
                total_blocks=total_blocks,
                file_size=size_bytes,
                payload_size=len(payload),
                flags=flags,
                fec_nsym=fec.nsym if fec else 0,
            )

            # Encode to pixel frame
            t_enc = time.perf_counter()
            frame = encode_binary_frame(header, payload, fec)
            stats['encode_time'] += time.perf_counter() - t_enc

            # === SIMULATE HDMI PIPELINE ===
            # (In real use, this goes through HDMI out -> capture card -> grayscale)
            # The frame is already uint8 with 0/255 values — perfect loopback.

            # Optional: inject random bit errors to test FEC
            if inject_errors and fec is not None:
                n_errors = error_rng.randint(0, 5)
                for _ in range(n_errors):
                    y = error_rng.randint(0, frame.shape[0] - 1)
                    x = error_rng.randint(0, frame.shape[1] - 1)
                    frame[y, x] = 255 - frame[y, x]  # flip pixel

            # === RECEIVER SIDE ===

            t_dec = time.perf_counter()
            dec_header, dec_payload, dec_stats = decode_binary_frame(frame, fec)
            stats['decode_time'] += time.perf_counter() - t_dec

            if dec_header is None or dec_payload is None:
                stats['decode_fail'] += 1
                print(f"\n  DECODE FAIL at block {block_idx}: "
                      f"header={'OK' if dec_stats.valid_header else 'FAIL'}, "
                      f"crc={'OK' if dec_stats.crc_ok else 'FAIL'}, "
                      f"fec_failed={dec_stats.fec_failed}")
                continue

            if dec_stats.crc_ok:
                stats['crc_ok'] += 1
            if dec_stats.fec_corrected > 0:
                stats['fec_corrected'] += 1
                stats['fec_corrections_total'] += dec_stats.fec_corrected

            # Store block
            store.store(dec_header.block_index, dec_payload)

            # Progress
            now = time.perf_counter()
            if now - last_report >= 2.0 or block_idx == total_blocks - 1:
                elapsed = now - t_start
                pct = (block_idx + 1) / total_blocks * 100
                bytes_done = min((block_idx + 1) * payload_capacity, size_bytes)
                mbps = bytes_done / elapsed / (1024 * 1024) if elapsed > 0 else 0
                eta = (elapsed / (block_idx + 1)) * (total_blocks - block_idx - 1)
                print(f"    Block {block_idx+1:,}/{total_blocks:,} ({pct:.1f}%) "
                      f"| {mbps:.1f} MB/s | ETA {eta:.0f}s "
                      f"| fail={stats['decode_fail']} fec={stats['fec_corrected']}",
                      end='\r')
                last_report = now

    total_time = time.perf_counter() - t_start
    print()  # newline after progress

    print(f"\n  Loopback complete in {total_time:.1f}s")
    print(f"    Encode time: {stats['encode_time']:.1f}s ({size_mb / stats['encode_time']:.1f} MB/s)")
    print(f"    Decode time: {stats['decode_time']:.1f}s ({size_mb / stats['decode_time']:.1f} MB/s)")
    print(f"    CRC OK: {stats['crc_ok']:,}/{total_blocks:,}")
    print(f"    FEC corrections: {stats['fec_corrected']:,} blocks ({stats['fec_corrections_total']:,} symbols)")
    print(f"    Decode failures: {stats['decode_fail']:,}")
    print(f"    Blocks stored: {store.count:,}/{total_blocks:,}")

    if stats['decode_fail'] > 0:
        print(f"\n  WARNING: {stats['decode_fail']} blocks failed to decode!")

    # Phase 3: Assemble output file
    print(f"\n  Phase 3: Assembling output file...")

    sha256 = hashlib.sha256()
    bytes_written = 0

    with open(output_path, 'wb') as out_f:
        for idx in range(total_blocks):
            if not store.has(idx):
                print(f"  Missing block {idx}!")
                continue

            raw_payload = store.get(idx)

            if idx == 0:
                # Strip metadata
                meta_end = 32 + 2 + len(filename_bytes)
                file_data = raw_payload[meta_end:]
            else:
                file_data = raw_payload

            # Truncate to file_size
            remaining = size_bytes - bytes_written
            if remaining <= 0:
                break
            write_data = file_data[:remaining]
            out_f.write(write_data)
            sha256.update(write_data)
            bytes_written += len(write_data)

    computed_hash = sha256.digest()

    print(f"  Wrote {bytes_written:,} bytes to {output_path}")
    print(f"  Expected SHA-256: {expected_hash.hex()}")
    print(f"  Computed SHA-256: {computed_hash.hex()}")

    # Phase 4: Verify
    print(f"\n  Phase 4: Verification")

    match = computed_hash == expected_hash
    size_match = bytes_written == size_bytes

    if match and size_match:
        print(f"  SHA-256: MATCH")
        print(f"  Size:    MATCH ({bytes_written:,} == {size_bytes:,})")
        print(f"\n  *** E2E TEST PASSED ***")
    else:
        if not match:
            print(f"  SHA-256: MISMATCH!")
        if not size_match:
            print(f"  Size:    MISMATCH ({bytes_written:,} != {size_bytes:,})")
        print(f"\n  *** E2E TEST FAILED ***")

    # Throughput summary
    print(f"\n  Throughput Summary:")
    print(f"    Total loopback: {size_mb / total_time:.1f} MB/s")
    print(f"    Encode only:    {size_mb / stats['encode_time']:.1f} MB/s")
    print(f"    Decode only:    {size_mb / stats['decode_time']:.1f} MB/s")
    sim_time_60fps = total_blocks / 60.0
    print(f"    Simulated @60fps 1x repeat: {size_mb / sim_time_60fps:.1f} MB/s ({sim_time_60fps:.1f}s)")
    sim_time_60fps_2x = total_blocks * 2 / 60.0
    print(f"    Simulated @60fps 2x repeat: {size_mb / sim_time_60fps_2x:.1f} MB/s ({sim_time_60fps_2x:.1f}s)")

    # Cleanup
    store.cleanup()
    try:
        os.remove(input_path)
        os.remove(output_path)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return match and size_match


def main():
    parser = argparse.ArgumentParser(description='Binary Pixel E2E Loopback Test')
    parser.add_argument('--size', type=float, default=1.0,
                        help='Test file size in GB (default: 1.0)')
    parser.add_argument('--fec', type=float, default=0.10,
                        help='FEC ratio (default: 0.10)')
    parser.add_argument('--no-fec', action='store_true',
                        help='Disable FEC')
    parser.add_argument('--errors', action='store_true',
                        help='Inject random bit errors (tests FEC correction)')
    args = parser.parse_args()

    fec_ratio = 0.0 if args.no_fec else args.fec
    success = run_e2e_test(args.size, fec_ratio, inject_errors=args.errors)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
