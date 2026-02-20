#!/usr/bin/env python3
"""Quick test for interleaved redundancy."""

import sys
sys.stdout.reconfigure(line_buffering=True)

import tempfile
import os
import shutil
from test_100mb_hpc import (
    create_test_file, test_hpc_encoding, test_cuda_decoding, verify_output
)

def main():
    # Quick 1 MB test with 3x interleaved redundancy
    temp_dir = tempfile.mkdtemp(prefix='vdd_interleave_')
    print(f'Temp dir: {temp_dir}')

    try:
        # Create 1 MB test file
        input_path = os.path.join(temp_dir, 'test_1mb.bin')
        original_hash = create_test_file(1, input_path)

        # Encode with 3x interleaved redundancy
        video_path = os.path.join(temp_dir, 'encoded.mp4')
        encode_result = test_hpc_encoding(input_path, video_path, workers=4, batch_size=50)

        # Decode
        output_dir = os.path.join(temp_dir, 'decoded')
        os.makedirs(output_dir, exist_ok=True)
        decode_result = test_cuda_decoding(video_path, output_dir, use_cuda=False, batch_size=32)

        # Verify
        if decode_result['files_decoded'] > 0:
            for decoded_file in decode_result['decoded_files']:
                if verify_output(original_hash, decoded_file.output_path):
                    print('\n=== 1 MB INTERLEAVED REDUNDANCY TEST PASSED ===')
                    return 0
            print('\n=== TEST FAILED ===')
            return 1
        else:
            print('\n=== TEST FAILED - No files decoded ===')
            return 1
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    sys.exit(main())
