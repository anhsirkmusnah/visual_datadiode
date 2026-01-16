"""
Visual Data Diode - Receiver UI

Tkinter-based user interface for the receiver application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from typing import Optional, Callable
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE,
    check_fec_available, check_crypto_available
)


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class ReceiverUI:
    """
    Tkinter UI for the Visual Data Diode receiver.

    Provides:
    - Device selection
    - Live preview
    - Reception progress
    - File save controls
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visual Data Diode - Receiver")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)

        # State
        self.is_receiving = False
        self.is_synced = False
        self.output_dir = str(Path.home() / "Downloads")

        # Callbacks
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None
        self.on_save: Optional[Callable] = None

        # Variables
        self.profile_var = tk.StringVar(value="standard")
        self.device_var = tk.IntVar(value=0)
        self.password_var = tk.StringVar(value="")
        self.show_preview_var = tk.BooleanVar(value=True)

        # Preview image
        self._preview_photo = None

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Device Selection Section
        device_frame = ttk.LabelFrame(main_frame, text="Capture Device", padding="5")
        device_frame.pack(fill=tk.X, pady=(0, 10))

        device_row = ttk.Frame(device_frame)
        device_row.pack(fill=tk.X)

        ttk.Label(device_row, text="Device:").pack(side=tk.LEFT)

        self.device_combo = ttk.Combobox(device_row, width=40, state="readonly")
        self.device_combo.pack(side=tk.LEFT, padx=5)

        self.refresh_btn = ttk.Button(
            device_row, text="Refresh", command=self._refresh_devices
        )
        self.refresh_btn.pack(side=tk.LEFT)

        self._refresh_devices()

        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Profile selection
        profile_row = ttk.Frame(settings_frame)
        profile_row.pack(fill=tk.X, pady=2)

        ttk.Label(profile_row, text="Profile:").pack(side=tk.LEFT)
        profiles = [
            ("Conservative", "conservative"),
            ("Standard", "standard"),
            ("Aggressive", "aggressive")
        ]
        for text, value in profiles:
            ttk.Radiobutton(
                profile_row, text=text, value=value,
                variable=self.profile_var
            ).pack(side=tk.LEFT, padx=5)

        # Output directory
        output_row = ttk.Frame(settings_frame)
        output_row.pack(fill=tk.X, pady=2)

        ttk.Label(output_row, text="Output:").pack(side=tk.LEFT)
        self.output_label = ttk.Label(
            output_row, text=self.output_dir, foreground="gray"
        )
        self.output_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(
            output_row, text="Browse...", command=self._browse_output
        ).pack(side=tk.RIGHT)

        # Password for decryption
        password_row = ttk.Frame(settings_frame)
        password_row.pack(fill=tk.X, pady=2)

        ttk.Label(password_row, text="Password:").pack(side=tk.LEFT)
        self.password_entry = ttk.Entry(
            password_row, textvariable=self.password_var, show="*", width=30
        )
        self.password_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(
            password_row, text="(leave empty if not encrypted)",
            foreground="gray"
        ).pack(side=tk.LEFT)

        # Preview Section
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        preview_controls = ttk.Frame(preview_frame)
        preview_controls.pack(fill=tk.X)

        self.preview_check = ttk.Checkbutton(
            preview_controls, text="Show live preview",
            variable=self.show_preview_var
        )
        self.preview_check.pack(side=tk.LEFT)

        self.sync_label = ttk.Label(
            preview_controls, text="Sync: Not synced", foreground="gray"
        )
        self.sync_label.pack(side=tk.RIGHT)

        # Preview canvas
        self.preview_canvas = tk.Canvas(
            preview_frame, width=320, height=180, bg='black'
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, pady=5)

        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(
            progress_frame, text="Ready to receive"
        )
        self.progress_label.pack(fill=tk.X)

        # Stats row
        stats_row = ttk.Frame(progress_frame)
        stats_row.pack(fill=tk.X)

        self.stats_label = ttk.Label(
            stats_row, text="", foreground="gray"
        )
        self.stats_label.pack(side=tk.LEFT)

        self.fps_label = ttk.Label(
            stats_row, text="FPS: --", foreground="gray"
        )
        self.fps_label.pack(side=tk.RIGHT)

        # Reception Details
        details_row = ttk.Frame(progress_frame)
        details_row.pack(fill=tk.X, pady=2)

        self.blocks_label = ttk.Label(details_row, text="Blocks: 0/0")
        self.blocks_label.pack(side=tk.LEFT)

        self.crc_label = ttk.Label(details_row, text="CRC Errors: 0")
        self.crc_label.pack(side=tk.LEFT, padx=20)

        self.fec_label = ttk.Label(details_row, text="FEC Corrections: 0")
        self.fec_label.pack(side=tk.LEFT)

        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_btn = ttk.Button(
            control_frame, text="Start Receiving",
            command=self._on_start_clicked
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame, text="Stop",
            command=self._on_stop_clicked, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            control_frame, text="Save File",
            command=self._on_save_clicked, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # File info
        self.file_info_label = ttk.Label(
            control_frame, text="", foreground="blue"
        )
        self.file_info_label.pack(side=tk.RIGHT)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        fec_status = "FEC: Available" if check_fec_available() else "FEC: Not available"
        crypto_status = "Crypto: Available" if check_crypto_available() else "Crypto: Not available"

        ttk.Label(
            status_frame, text=f"{fec_status} | {crypto_status}",
            foreground="gray", font=('TkDefaultFont', 8)
        ).pack(side=tk.LEFT)

    def _refresh_devices(self):
        """Refresh capture device list."""
        try:
            from .capture import list_capture_devices
            devices = list_capture_devices()

            if devices:
                options = [d['description'] for d in devices]
                self.device_combo['values'] = options
                self.device_combo.current(0)
            else:
                self.device_combo['values'] = ["No devices found"]
                self.device_combo.current(0)
        except Exception as e:
            self.device_combo['values'] = [f"Error: {e}"]
            self.device_combo.current(0)

    def _browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select output directory",
            initialdir=self.output_dir
        )

        if directory:
            self.output_dir = directory
            # Truncate for display
            display = directory
            if len(display) > 50:
                display = "..." + display[-47:]
            self.output_label.config(text=display)

    def _on_start_clicked(self):
        """Handle start button click."""
        self.is_receiving = True
        self._update_control_buttons()

        if self.on_start:
            settings = self._get_settings()
            self.on_start(settings)

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.is_receiving = False
        self._update_control_buttons()

        if self.on_stop:
            self.on_stop()

    def _on_save_clicked(self):
        """Handle save button click."""
        if self.on_save:
            self.on_save()

    def _get_settings(self) -> dict:
        """Get current settings."""
        profile_name = self.profile_var.get()
        if profile_name == "conservative":
            profile = PROFILE_CONSERVATIVE
        elif profile_name == "aggressive":
            profile = PROFILE_AGGRESSIVE
        else:
            profile = PROFILE_STANDARD

        device_idx = self.device_combo.current()

        return {
            'profile': profile,
            'device_index': device_idx,
            'output_dir': self.output_dir,
            'password': self.password_var.get() or None,
            'show_preview': self.show_preview_var.get()
        }

    def _update_control_buttons(self):
        """Update button states."""
        if self.is_receiving:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.device_combo.config(state=tk.DISABLED)
            self.refresh_btn.config(state=tk.DISABLED)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.device_combo.config(state="readonly")
            self.refresh_btn.config(state=tk.NORMAL)

    def update_preview(self, frame: np.ndarray):
        """Update preview canvas with a frame."""
        if not self.show_preview_var.get():
            return

        try:
            from PIL import Image, ImageTk

            # Resize for preview
            h, w = frame.shape[:2]
            preview_w = 320
            preview_h = int(h * preview_w / w)

            img = Image.fromarray(frame)
            img = img.resize((preview_w, preview_h), Image.Resampling.LANCZOS)

            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                preview_w // 2, preview_h // 2,
                image=self._preview_photo
            )

        except ImportError:
            # PIL not available
            pass
        except Exception:
            pass

    def update_sync_status(self, synced: bool, confidence: float):
        """Update sync status display."""
        self.is_synced = synced

        if synced:
            self.sync_label.config(
                text=f"Sync: OK ({confidence:.0%})",
                foreground="green"
            )
        else:
            self.sync_label.config(
                text="Sync: Searching...",
                foreground="orange"
            )

    def update_progress(
        self,
        blocks_received: int,
        total_blocks: int,
        crc_errors: int,
        fec_corrections: int,
        elapsed_time: float,
        fps: float
    ):
        """Update progress display."""
        if total_blocks > 0:
            progress = (blocks_received / total_blocks) * 100
            self.progress_bar['value'] = progress
            self.progress_label.config(
                text=f"Receiving: {blocks_received:,} / {total_blocks:,} blocks ({progress:.1f}%)"
            )

            remaining = total_blocks - blocks_received
            if fps > 0:
                eta = remaining / fps
                self.stats_label.config(
                    text=f"Elapsed: {format_duration(elapsed_time)} | ETA: {format_duration(eta)}"
                )
        else:
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Waiting for first block...")

        self.blocks_label.config(text=f"Blocks: {blocks_received}/{total_blocks}")
        self.crc_label.config(text=f"CRC Errors: {crc_errors}")
        self.fec_label.config(text=f"FEC Corrections: {fec_corrections}")
        self.fps_label.config(text=f"FPS: {fps:.1f}")

    def show_complete(
        self,
        success: bool,
        filename: str = "",
        file_size: int = 0,
        hash_valid: bool = None,
        message: str = ""
    ):
        """Show completion status."""
        self.is_receiving = False
        self._update_control_buttons()

        if success:
            self.progress_bar['value'] = 100
            self.progress_label.config(text="Reception complete!")

            if hash_valid:
                hash_status = "Hash: Verified"
                hash_color = "green"
            elif hash_valid is False:
                hash_status = "Hash: MISMATCH"
                hash_color = "red"
            else:
                hash_status = "Hash: Not verified"
                hash_color = "gray"

            self.file_info_label.config(
                text=f"{filename} ({self._format_size(file_size)}) - {hash_status}",
                foreground=hash_color
            )

            self.save_btn.config(state=tk.NORMAL)

            if hash_valid:
                messagebox.showinfo(
                    "Complete",
                    f"File received successfully!\n\n"
                    f"Filename: {filename}\n"
                    f"Size: {self._format_size(file_size)}\n"
                    f"Hash verified: Yes"
                )
            else:
                messagebox.showwarning(
                    "Complete with Warning",
                    f"File received but hash verification {'failed' if hash_valid is False else 'not performed'}!\n\n"
                    f"Filename: {filename}\n"
                    f"Size: {self._format_size(file_size)}"
                )
        else:
            self.progress_label.config(text=f"Reception failed: {message}")
            messagebox.showerror("Error", message or "Reception failed")

    def _format_size(self, size: int) -> str:
        """Format file size."""
        if size < 1024:
            return f"{size} bytes"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"

    def run(self):
        """Start the UI main loop."""
        self.root.mainloop()

    def close(self):
        """Close the UI."""
        self.root.destroy()


def run_receiver_ui():
    """Run the receiver UI as standalone."""
    ui = ReceiverUI()
    ui.run()


if __name__ == "__main__":
    run_receiver_ui()
