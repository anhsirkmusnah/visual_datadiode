#!/usr/bin/env python3
"""Test device enumeration."""

import sys
sys.path.insert(0, '.')

from receiver.capture import list_capture_devices, _get_dshow_device_names

print("Getting device names from ffmpeg...")
names = _get_dshow_device_names()
print(f"Found {len(names)} device names:")
for n in names:
    print(f"  - {n}")

print()
print("Listing capture devices...")
devices = list_capture_devices()
print(f"Found {len(devices)} devices:")
for d in devices:
    print(f"  - {d['description']}")
