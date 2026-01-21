"""
Visual Data Diode - Sender UI

Tkinter-based user interface for the sender application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
from typing import Optional, Callable
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    DEFAULT_FPS, check_fec_available, check_crypto_available
)
from .timing import format_duration


class SenderUI:
    """
    Tkinter UI for the Visual Data Diode sender.

    Provides:
    - File selection
    - Profile and options configuration
    - Transmission controls
    - Progress display
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visual Data Diode - Sender")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        self.root.minsize(500, 400)

        # State
        self.file_path: Optional[str] = None
        self.file_size: int = 0
        self.is_transmitting = False
        self.is_paused = False

        # Callbacks
        self.on_start: Optional[Callable] = None
        self.on_pause: Optional[Callable] = None
        self.on_resume: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None
        self.on_restart: Optional[Callable] = None

        # Variables
        self.profile_var = tk.StringVar(value="conservative")  # Use conservative for reliability
        self.fps_var = tk.IntVar(value=DEFAULT_FPS)
        self.repeat_var = tk.IntVar(value=2)
        self.encrypt_var = tk.BooleanVar(value=False)
        self.display_var = tk.IntVar(value=3)  # Default to Display 3 (validated)
        self.resolution_var = tk.StringVar(value="1920x1080")
        self._displays = []

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File Selection Section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_btn = ttk.Button(
            file_frame, text="Browse...", command=self._browse_file
        )
        self.browse_btn.pack(side=tk.RIGHT)

        # File Info
        self.file_info_label = ttk.Label(
            main_frame, text="", foreground="gray"
        )
        self.file_info_label.pack(fill=tk.X)

        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(10, 10))

        # Profile selection
        profile_frame = ttk.Frame(settings_frame)
        profile_frame.pack(fill=tk.X, pady=2)

        ttk.Label(profile_frame, text="Profile:").pack(side=tk.LEFT)
        profiles = [
            ("Conservative", "conservative"),
            ("Standard", "standard"),
            ("Aggressive", "aggressive"),
            ("Ultra (fastest)", "ultra")
        ]
        for text, value in profiles:
            ttk.Radiobutton(
                profile_frame, text=text, value=value,
                variable=self.profile_var, command=self._update_estimates
            ).pack(side=tk.LEFT, padx=5)

        # FPS selection
        fps_frame = ttk.Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=2)

        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_spinbox = ttk.Spinbox(
            fps_frame, from_=10, to=30, width=5,
            textvariable=self.fps_var, command=self._update_estimates
        )
        self.fps_spinbox.pack(side=tk.LEFT, padx=5)

        ttk.Label(fps_frame, text="Repeat count:").pack(side=tk.LEFT, padx=(20, 0))
        self.repeat_spinbox = ttk.Spinbox(
            fps_frame, from_=1, to=5, width=5,
            textvariable=self.repeat_var, command=self._update_estimates
        )
        self.repeat_spinbox.pack(side=tk.LEFT, padx=5)

        # Encryption checkbox
        encrypt_frame = ttk.Frame(settings_frame)
        encrypt_frame.pack(fill=tk.X, pady=2)

        self.encrypt_check = ttk.Checkbutton(
            encrypt_frame, text="Encrypt payload (AES-256-GCM)",
            variable=self.encrypt_var
        )
        self.encrypt_check.pack(side=tk.LEFT)

        if not check_crypto_available():
            self.encrypt_check.config(state=tk.DISABLED)
            ttk.Label(
                encrypt_frame, text="(install cryptography)",
                foreground="gray"
            ).pack(side=tk.LEFT)

        # Display selection
        display_frame = ttk.Frame(settings_frame)
        display_frame.pack(fill=tk.X, pady=2)

        ttk.Label(display_frame, text="Display:").pack(side=tk.LEFT)
        self.display_combo = ttk.Combobox(
            display_frame, width=35, state="readonly"
        )
        self.display_combo.pack(side=tk.LEFT, padx=5)
        self.display_combo.bind('<<ComboboxSelected>>', self._on_display_selected)

        # Resolution selection
        resolution_frame = ttk.Frame(settings_frame)
        resolution_frame.pack(fill=tk.X, pady=2)

        ttk.Label(resolution_frame, text="Resolution:").pack(side=tk.LEFT)
        self.resolution_combo = ttk.Combobox(
            resolution_frame, width=15, textvariable=self.resolution_var, state="readonly"
        )
        resolutions = [
            "1920x1080",
            "1280x720",
            "2560x1440",
            "3840x2160",
            "1600x900",
            "1366x768"
        ]
        self.resolution_combo['values'] = resolutions
        self.resolution_combo.pack(side=tk.LEFT, padx=5)
        self.resolution_combo.bind('<<ComboboxSelected>>', self._on_resolution_changed)

        ttk.Label(resolution_frame, text="(output to display)", foreground="gray").pack(side=tk.LEFT)

        # Populate displays now that resolution_combo exists
        self._populate_displays()

        # Estimates Section
        estimates_frame = ttk.LabelFrame(main_frame, text="Estimates", padding="5")
        estimates_frame.pack(fill=tk.X, pady=(0, 10))

        self.estimates_label = ttk.Label(
            estimates_frame,
            text="Select a file to see estimates",
            justify=tk.LEFT
        )
        self.estimates_label.pack(fill=tk.X)

        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(
            progress_frame, text="Ready"
        )
        self.progress_label.pack(fill=tk.X)

        self.stats_label = ttk.Label(
            progress_frame, text="", foreground="gray"
        )
        self.stats_label.pack(fill=tk.X)

        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_btn = ttk.Button(
            control_frame, text="Start Transmission",
            command=self._on_start_clicked
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = ttk.Button(
            control_frame, text="Pause",
            command=self._on_pause_clicked, state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame, text="Stop",
            command=self._on_stop_clicked, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.restart_btn = ttk.Button(
            control_frame, text="Restart",
            command=self._on_restart_clicked, state=tk.DISABLED
        )
        self.restart_btn.pack(side=tk.LEFT, padx=5)

        # FEC status
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        fec_status = "FEC: Available" if check_fec_available() else "FEC: Not available (install reedsolo)"
        crypto_status = "Crypto: Available" if check_crypto_available() else "Crypto: Not available"

        ttk.Label(
            status_frame, text=f"{fec_status} | {crypto_status}",
            foreground="gray", font=('TkDefaultFont', 8)
        ).pack(side=tk.LEFT)

    def _populate_displays(self):
        """Populate display selection dropdown with actual display names."""
        try:
            from .renderer import list_displays
            self._displays = list_displays()

            if self._displays:
                options = ["Default (auto)"] + [d['description'] for d in self._displays]
                self.display_combo['values'] = options
                self.display_combo.current(0)

                # Set default resolution from primary display
                default_res = f"{self._displays[0]['width']}x{self._displays[0]['height']}"
                self.resolution_var.set(default_res)

                # Add to resolutions if not present (only if combo exists)
                if hasattr(self, 'resolution_combo'):
                    current_vals = list(self.resolution_combo['values'])
                    if default_res not in current_vals:
                        current_vals.insert(0, default_res)
                        self.resolution_combo['values'] = current_vals
            else:
                self._displays = []
                self.display_combo['values'] = ["Default"]
                self.display_combo.current(0)
        except Exception as e:
            print(f"Error populating displays: {e}")
            self._displays = []
            self.display_combo['values'] = ["Default"]
            self.display_combo.current(0)

    def _on_display_selected(self, event=None):
        """Handle display selection - update resolution to match display."""
        idx = self.display_combo.current()
        if idx > 0 and hasattr(self, '_displays') and idx - 1 < len(self._displays):
            display = self._displays[idx - 1]
            res = f"{display['width']}x{display['height']}"
            self.resolution_var.set(res)
            # Add to resolutions if not present
            current_vals = list(self.resolution_combo['values'])
            if res not in current_vals:
                current_vals.insert(0, res)
                self.resolution_combo['values'] = current_vals

    def _on_resolution_changed(self, event=None):
        """Handle resolution change."""
        self._update_estimates()

    def _browse_file(self):
        """Open file browser dialog."""
        file_path = filedialog.askopenfilename(
            title="Select file to transmit",
            filetypes=[("All files", "*.*")]
        )

        if file_path:
            self.file_path = file_path
            self.file_size = os.path.getsize(file_path)

            # Update labels
            filename = os.path.basename(file_path)
            if len(filename) > 50:
                display_name = filename[:47] + "..."
            else:
                display_name = filename

            self.file_label.config(text=display_name)
            self.file_info_label.config(
                text=f"Path: {file_path}\nSize: {self._format_size(self.file_size)}"
            )

            self._update_estimates()

    def _format_size(self, size: int) -> str:
        """Format file size as human-readable string."""
        if size < 1024:
            return f"{size} bytes"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"

    def _update_estimates(self):
        """Update transfer estimates based on current settings."""
        if not self.file_path or self.file_size == 0:
            return

        # Get profile
        profile_name = self.profile_var.get()
        if profile_name == "conservative":
            profile = PROFILE_CONSERVATIVE
        elif profile_name == "aggressive":
            profile = PROFILE_AGGRESSIVE
        elif profile_name == "ultra":
            profile = PROFILE_ULTRA
        else:
            profile = PROFILE_STANDARD

        fps = self.fps_var.get()
        repeat = self.repeat_var.get()

        # Calculate estimates
        from shared import calculate_payload_capacity

        payload_per_block = calculate_payload_capacity(profile, 0.10)
        total_blocks = (self.file_size + payload_per_block - 1) // payload_per_block
        total_frames = total_blocks * repeat

        bytes_per_second = payload_per_block * fps / repeat
        transfer_time = self.file_size / bytes_per_second

        estimates_text = (
            f"Profile: {profile.name} ({profile.cell_size}x{profile.cell_size} cells)\n"
            f"Payload per block: {payload_per_block} bytes\n"
            f"Total blocks: {total_blocks:,}\n"
            f"Total frames: {total_frames:,} (with {repeat}x repeat)\n"
            f"Effective rate: {bytes_per_second / 1024:.1f} KB/s\n"
            f"Estimated time: {format_duration(transfer_time)}"
        )

        self.estimates_label.config(text=estimates_text)

    def _on_start_clicked(self):
        """Handle start button click."""
        if not self.file_path:
            messagebox.showwarning("No File", "Please select a file first.")
            return

        if self.encrypt_var.get():
            # Get password
            password = self._get_password()
            if password is None:
                return  # Cancelled
            self._encryption_password = password
        else:
            self._encryption_password = None

        self.is_transmitting = True
        self.is_paused = False
        self._update_control_buttons()

        if self.on_start:
            # Get settings
            settings = self._get_settings()
            self.on_start(settings)

    def _get_password(self) -> Optional[str]:
        """Show password dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Encryption Password")
        dialog.geometry("300x120")
        dialog.transient(self.root)
        dialog.grab_set()

        result = [None]

        ttk.Label(dialog, text="Enter encryption password:").pack(pady=10)

        password_entry = ttk.Entry(dialog, show="*", width=30)
        password_entry.pack(pady=5)
        password_entry.focus()

        def on_ok():
            result[0] = password_entry.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT)

        password_entry.bind('<Return>', lambda e: on_ok())

        dialog.wait_window()
        return result[0]

    def _get_settings(self) -> dict:
        """Get current settings as dictionary."""
        profile_name = self.profile_var.get()
        if profile_name == "conservative":
            profile = PROFILE_CONSERVATIVE
        elif profile_name == "aggressive":
            profile = PROFILE_AGGRESSIVE
        elif profile_name == "ultra":
            profile = PROFILE_ULTRA
        else:
            profile = PROFILE_STANDARD

        display_idx = self.display_combo.current() - 1  # -1 for "Default"

        # Parse resolution
        res_str = self.resolution_var.get()
        try:
            res_width, res_height = map(int, res_str.split('x'))
        except:
            res_width, res_height = 1920, 1080

        return {
            'file_path': self.file_path,
            'profile': profile,
            'fps': self.fps_var.get(),
            'repeat_count': self.repeat_var.get(),
            'encrypt': self.encrypt_var.get(),
            'password': getattr(self, '_encryption_password', None),
            'display_index': display_idx,
            'resolution_width': res_width,
            'resolution_height': res_height
        }

    def _on_pause_clicked(self):
        """Handle pause/resume button click."""
        if self.is_paused:
            self.is_paused = False
            self.pause_btn.config(text="Pause")
            if self.on_resume:
                self.on_resume()
        else:
            self.is_paused = True
            self.pause_btn.config(text="Resume")
            if self.on_pause:
                self.on_pause()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if messagebox.askyesno("Confirm Stop", "Stop transmission?"):
            self.is_transmitting = False
            self.is_paused = False
            self._update_control_buttons()
            if self.on_stop:
                self.on_stop()

    def _on_restart_clicked(self):
        """Handle restart button click."""
        if messagebox.askyesno("Confirm Restart", "Restart transmission from beginning?"):
            if self.on_restart:
                self.on_restart()

    def _update_control_buttons(self):
        """Update button states based on transmission state."""
        if self.is_transmitting:
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.restart_btn.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.DISABLED)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED, text="Pause")
            self.stop_btn.config(state=tk.DISABLED)
            self.restart_btn.config(state=tk.DISABLED)
            self.browse_btn.config(state=tk.NORMAL)

    def update_progress(
        self,
        current_block: int,
        total_blocks: int,
        bytes_sent: int,
        elapsed_time: float,
        actual_fps: float
    ):
        """Update progress display."""
        progress = (current_block / total_blocks) * 100 if total_blocks > 0 else 0
        self.progress_bar['value'] = progress

        remaining_blocks = total_blocks - current_block
        remaining_time = remaining_blocks / actual_fps if actual_fps > 0 else 0

        self.progress_label.config(
            text=f"Block {current_block:,} / {total_blocks:,} ({progress:.1f}%)"
        )

        bytes_per_sec = bytes_sent / elapsed_time if elapsed_time > 0 else 0
        self.stats_label.config(
            text=f"Rate: {bytes_per_sec / 1024:.1f} KB/s | "
                 f"FPS: {actual_fps:.1f} | "
                 f"Elapsed: {format_duration(elapsed_time)} | "
                 f"Remaining: {format_duration(remaining_time)}"
        )

    def show_complete(self, success: bool, message: str = ""):
        """Show completion message."""
        self.is_transmitting = False
        self._update_control_buttons()

        if success:
            self.progress_bar['value'] = 100
            self.progress_label.config(text="Transmission complete!")
            messagebox.showinfo("Complete", message or "File transmission completed successfully!")
        else:
            self.progress_label.config(text="Transmission failed")
            messagebox.showerror("Error", message or "Transmission failed.")

    def run(self):
        """Start the UI main loop."""
        self.root.mainloop()

    def close(self):
        """Close the UI."""
        self.root.destroy()


def run_sender_ui():
    """Run the sender UI as standalone."""
    ui = SenderUI()
    ui.run()


if __name__ == "__main__":
    run_sender_ui()
