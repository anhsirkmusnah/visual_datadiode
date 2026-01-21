"""
Visual Data Diode - Sender UI

Tkinter-based user interface for encoding files to video.
Supports multiple file selection with separate video output for each.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
from typing import Optional, Callable, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    DEFAULT_FPS, check_fec_available, check_crypto_available
)
from .timing import format_duration
from .video_encoder import check_ffmpeg_available, get_encoder_info


class SenderUI:
    """
    Tkinter UI for the Visual Data Diode sender.

    Provides:
    - Multiple file selection
    - Video output configuration
    - Profile and encoding options
    - Progress display for batch encoding
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visual Data Diode - Video Encoder")
        self.root.geometry("700x650")
        self.root.resizable(True, True)
        self.root.minsize(600, 550)

        # State
        self.file_paths: List[str] = []
        self.total_size: int = 0
        self.output_dir: str = str(Path.home() / "Videos")
        self.is_encoding = False
        self.current_file_index = 0

        # Callbacks
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None

        # Variables
        self.profile_var = tk.StringVar(value="conservative")
        self.fps_var = tk.IntVar(value=DEFAULT_FPS)
        self.repeat_var = tk.IntVar(value=2)
        self.encrypt_var = tk.BooleanVar(value=False)
        self.resolution_var = tk.StringVar(value="1920x1080")
        self.add_audio_var = tk.BooleanVar(value=True)

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File Selection Section
        file_frame = ttk.LabelFrame(main_frame, text="Files to Encode", padding="5")
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # File list with scrollbar
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, height=6
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # File buttons
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        self.add_files_btn = ttk.Button(
            btn_frame, text="Add Files...", command=self._add_files
        )
        self.add_files_btn.pack(side=tk.LEFT, padx=2)

        self.remove_btn = ttk.Button(
            btn_frame, text="Remove Selected", command=self._remove_selected
        )
        self.remove_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = ttk.Button(
            btn_frame, text="Clear All", command=self._clear_files
        )
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        # File count and size info
        self.files_info_label = ttk.Label(
            file_frame, text="No files selected", foreground="gray"
        )
        self.files_info_label.pack(fill=tk.X, pady=(5, 0))

        # Output Directory Section
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding="5")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        self.output_label = ttk.Label(
            output_frame, text=self.output_dir, foreground="blue"
        )
        self.output_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_output_btn = ttk.Button(
            output_frame, text="Browse...", command=self._browse_output
        )
        self.browse_output_btn.pack(side=tk.RIGHT)

        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="Encoding Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Profile selection
        profile_frame = ttk.Frame(settings_frame)
        profile_frame.pack(fill=tk.X, pady=2)

        ttk.Label(profile_frame, text="Profile:").pack(side=tk.LEFT)
        profiles = [
            ("Conservative", "conservative"),
            ("Standard", "standard"),
            ("Aggressive", "aggressive"),
            ("Ultra", "ultra")
        ]
        for text, value in profiles:
            ttk.Radiobutton(
                profile_frame, text=text, value=value,
                variable=self.profile_var, command=self._update_estimates
            ).pack(side=tk.LEFT, padx=5)

        # FPS and Repeat
        fps_frame = ttk.Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=2)

        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_spinbox = ttk.Spinbox(
            fps_frame, from_=10, to=60, width=5,
            textvariable=self.fps_var, command=self._update_estimates
        )
        self.fps_spinbox.pack(side=tk.LEFT, padx=5)

        ttk.Label(fps_frame, text="Repeat count:").pack(side=tk.LEFT, padx=(20, 0))
        self.repeat_spinbox = ttk.Spinbox(
            fps_frame, from_=1, to=5, width=5,
            textvariable=self.repeat_var, command=self._update_estimates
        )
        self.repeat_spinbox.pack(side=tk.LEFT, padx=5)

        # Resolution selection
        resolution_frame = ttk.Frame(settings_frame)
        resolution_frame.pack(fill=tk.X, pady=2)

        ttk.Label(resolution_frame, text="Resolution:").pack(side=tk.LEFT)
        self.resolution_combo = ttk.Combobox(
            resolution_frame, width=12, textvariable=self.resolution_var, state="readonly"
        )
        resolutions = [
            "1920x1080",
            "2560x1440",
            "3840x2160",
            "1280x720",
            "1600x900"
        ]
        self.resolution_combo['values'] = resolutions
        self.resolution_combo.pack(side=tk.LEFT, padx=5)
        self.resolution_combo.bind('<<ComboboxSelected>>', self._update_estimates)

        # Audio option
        audio_frame = ttk.Frame(settings_frame)
        audio_frame.pack(fill=tk.X, pady=2)

        self.audio_check = ttk.Checkbutton(
            audio_frame, text="Add audio sync beeps",
            variable=self.add_audio_var
        )
        self.audio_check.pack(side=tk.LEFT)

        ffmpeg_available = check_ffmpeg_available()
        if not ffmpeg_available:
            self.audio_check.config(state=tk.DISABLED)
            self.add_audio_var.set(False)
            ttk.Label(
                audio_frame, text="(requires FFmpeg)",
                foreground="gray"
            ).pack(side=tk.LEFT)

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

        # Estimates Section
        estimates_frame = ttk.LabelFrame(main_frame, text="Estimates", padding="5")
        estimates_frame.pack(fill=tk.X, pady=(0, 10))

        self.estimates_label = ttk.Label(
            estimates_frame,
            text="Add files to see encoding estimates",
            justify=tk.LEFT
        )
        self.estimates_label.pack(fill=tk.X)

        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        # Overall progress
        ttk.Label(progress_frame, text="Overall:").pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate'
        )
        self.overall_progress.pack(fill=tk.X, pady=2)

        # Current file progress
        ttk.Label(progress_frame, text="Current file:").pack(anchor=tk.W)
        self.file_progress = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate'
        )
        self.file_progress.pack(fill=tk.X, pady=2)

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
            control_frame, text="Start Encoding",
            command=self._on_start_clicked
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame, text="Stop",
            command=self._on_stop_clicked, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        encoder_info = get_encoder_info()
        fec_status = "FEC: OK" if check_fec_available() else "FEC: N/A"
        ffmpeg_status = "FFmpeg: OK" if encoder_info['ffmpeg_available'] else "FFmpeg: N/A"
        crypto_status = "Crypto: OK" if check_crypto_available() else "Crypto: N/A"

        ttk.Label(
            status_frame,
            text=f"{fec_status} | {ffmpeg_status} | {crypto_status}",
            foreground="gray", font=('TkDefaultFont', 8)
        ).pack(side=tk.LEFT)

    def _add_files(self):
        """Add files to the list."""
        file_paths = filedialog.askopenfilenames(
            title="Select files to encode",
            filetypes=[("All files", "*.*")]
        )

        if file_paths:
            for path in file_paths:
                if path not in self.file_paths:
                    self.file_paths.append(path)
                    filename = os.path.basename(path)
                    size = os.path.getsize(path)
                    self.file_listbox.insert(tk.END, f"{filename} ({self._format_size(size)})")

            self._update_file_info()
            self._update_estimates()

    def _remove_selected(self):
        """Remove selected files from the list."""
        selected = list(self.file_listbox.curselection())
        selected.reverse()  # Remove from end to preserve indices

        for idx in selected:
            self.file_listbox.delete(idx)
            del self.file_paths[idx]

        self._update_file_info()
        self._update_estimates()

    def _clear_files(self):
        """Clear all files from the list."""
        self.file_listbox.delete(0, tk.END)
        self.file_paths.clear()
        self._update_file_info()
        self._update_estimates()

    def _update_file_info(self):
        """Update file count and total size display."""
        count = len(self.file_paths)
        self.total_size = sum(os.path.getsize(p) for p in self.file_paths)

        if count == 0:
            self.files_info_label.config(text="No files selected")
        else:
            self.files_info_label.config(
                text=f"{count} file(s) selected, total: {self._format_size(self.total_size)}"
            )

    def _browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select output directory",
            initialdir=self.output_dir
        )

        if directory:
            self.output_dir = directory
            display = directory
            if len(display) > 50:
                display = "..." + display[-47:]
            self.output_label.config(text=display)

    def _format_size(self, size: int) -> str:
        """Format file size as human-readable string."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"

    def _get_profile(self):
        """Get the selected encoding profile."""
        profile_name = self.profile_var.get()
        if profile_name == "conservative":
            return PROFILE_CONSERVATIVE
        elif profile_name == "aggressive":
            return PROFILE_AGGRESSIVE
        elif profile_name == "ultra":
            return PROFILE_ULTRA
        else:
            return PROFILE_STANDARD

    def _update_estimates(self, event=None):
        """Update encoding estimates based on current settings."""
        if not self.file_paths:
            self.estimates_label.config(text="Add files to see encoding estimates")
            return

        profile = self._get_profile()
        fps = self.fps_var.get()
        repeat = self.repeat_var.get()

        # Parse resolution
        res_str = self.resolution_var.get()
        try:
            res_width, res_height = map(int, res_str.split('x'))
        except:
            res_width, res_height = 1920, 1080

        from shared import calculate_payload_capacity, SYNC_FRAME_COUNT, END_FRAME_COUNT

        payload_per_block = calculate_payload_capacity(profile, 0.10)
        total_blocks = (self.total_size + payload_per_block - 1) // payload_per_block
        total_data_frames = total_blocks * repeat
        total_frames = total_data_frames + SYNC_FRAME_COUNT + END_FRAME_COUNT

        video_duration = total_frames / fps
        bytes_per_second = payload_per_block * fps / repeat

        # Estimate video file size (lossless H.264 is roughly 0.5-2 bytes/pixel depending on content)
        # For grayscale grid patterns, expect lower compression
        pixels_per_frame = res_width * res_height
        estimated_video_bytes = int(pixels_per_frame * 1.5 * total_frames / fps * fps)  # rough estimate

        estimates_text = (
            f"Profile: {profile.name} ({profile.cell_size}x{profile.cell_size} cells)\n"
            f"Resolution: {res_width}x{res_height} @ {fps} FPS\n"
            f"Payload per block: {payload_per_block} bytes\n"
            f"Total blocks: {total_blocks:,} ({len(self.file_paths)} file(s))\n"
            f"Total frames: {total_frames:,} (with {repeat}x repeat)\n"
            f"Video duration: {format_duration(video_duration)}\n"
            f"Effective data rate: {bytes_per_second / 1024:.1f} KB/s"
        )

        self.estimates_label.config(text=estimates_text)

    def _on_start_clicked(self):
        """Handle start button click."""
        if not self.file_paths:
            messagebox.showwarning("No Files", "Please add files to encode.")
            return

        if self.encrypt_var.get():
            password = self._get_password()
            if password is None:
                return
            self._encryption_password = password
        else:
            self._encryption_password = None

        self.is_encoding = True
        self.current_file_index = 0
        self._update_control_buttons()

        if self.on_start:
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
        profile = self._get_profile()

        res_str = self.resolution_var.get()
        try:
            res_width, res_height = map(int, res_str.split('x'))
        except:
            res_width, res_height = 1920, 1080

        return {
            'file_paths': self.file_paths.copy(),
            'output_dir': self.output_dir,
            'profile': profile,
            'fps': self.fps_var.get(),
            'repeat_count': self.repeat_var.get(),
            'encrypt': self.encrypt_var.get(),
            'password': getattr(self, '_encryption_password', None),
            'resolution_width': res_width,
            'resolution_height': res_height,
            'add_audio': self.add_audio_var.get()
        }

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if messagebox.askyesno("Confirm Stop", "Stop encoding?"):
            self.is_encoding = False
            self._update_control_buttons()
            if self.on_stop:
                self.on_stop()

    def _update_control_buttons(self):
        """Update button states based on encoding state."""
        if self.is_encoding:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.add_files_btn.config(state=tk.DISABLED)
            self.remove_btn.config(state=tk.DISABLED)
            self.clear_btn.config(state=tk.DISABLED)
            self.browse_output_btn.config(state=tk.DISABLED)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.add_files_btn.config(state=tk.NORMAL)
            self.remove_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            self.browse_output_btn.config(state=tk.NORMAL)

    def update_progress(
        self,
        current_file_index: int,
        total_files: int,
        current_file_name: str,
        current_frame: int,
        total_frames: int,
        elapsed_time: float
    ):
        """Update progress display."""
        self.current_file_index = current_file_index

        # Overall progress
        if total_files > 0:
            overall = ((current_file_index + current_frame / max(total_frames, 1)) / total_files) * 100
            self.overall_progress['value'] = overall

        # Current file progress
        if total_frames > 0:
            file_progress = (current_frame / total_frames) * 100
            self.file_progress['value'] = file_progress

        self.progress_label.config(
            text=f"File {current_file_index + 1}/{total_files}: {current_file_name}\n"
                 f"Frame {current_frame:,} / {total_frames:,}"
        )

        if elapsed_time > 0 and current_frame > 0:
            fps = current_frame / elapsed_time
            remaining_frames = total_frames - current_frame
            remaining_time = remaining_frames / fps if fps > 0 else 0

            self.stats_label.config(
                text=f"Encoding: {fps:.1f} FPS | "
                     f"Elapsed: {format_duration(elapsed_time)} | "
                     f"Remaining: {format_duration(remaining_time)}"
            )

    def show_file_complete(self, file_path: str, output_path: str):
        """Show completion for a single file."""
        filename = os.path.basename(file_path)
        output_name = os.path.basename(output_path)
        self.progress_label.config(text=f"Completed: {filename} -> {output_name}")

    def show_complete(self, success: bool, message: str = ""):
        """Show completion message."""
        self.is_encoding = False
        self._update_control_buttons()

        if success:
            self.overall_progress['value'] = 100
            self.file_progress['value'] = 100
            self.progress_label.config(text="Encoding complete!")
            messagebox.showinfo("Complete", message or "All files encoded successfully!")
        else:
            self.progress_label.config(text="Encoding stopped")
            if message:
                messagebox.showerror("Error", message)

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
