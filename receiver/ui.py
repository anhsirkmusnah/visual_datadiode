"""
Visual Data Diode - Receiver UI

Tkinter-based user interface for decoding video files.
Supports processing recorded videos containing multiple encoded files.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
from typing import Optional, Callable, List
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    check_fec_available, check_crypto_available
)
from .video_processor import get_video_info, ProcessorProgress, ProcessorState


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


def format_size(size: int) -> str:
    """Format file size as human-readable string."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


class ReceiverUI:
    """
    Tkinter UI for the Visual Data Diode receiver.

    Provides:
    - Video file selection
    - Output directory selection
    - Processing progress display
    - Decoded files list
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visual Data Diode - Video Decoder")
        self.root.geometry("750x700")
        self.root.resizable(True, True)
        self.root.minsize(650, 600)

        # State
        self.is_processing = False
        self.video_path: Optional[str] = None
        self.output_dir = str(Path.home() / "Downloads")
        self.video_info: Optional[dict] = None

        # Callbacks
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None

        # Variables
        self.profile_var = tk.StringVar(value="conservative")
        self.password_var = tk.StringVar(value="")

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Video File Section
        video_frame = ttk.LabelFrame(main_frame, text="Input Video", padding="5")
        video_frame.pack(fill=tk.X, pady=(0, 10))

        # Video file selection row
        file_row = ttk.Frame(video_frame)
        file_row.pack(fill=tk.X, pady=2)

        self.video_label = ttk.Label(
            file_row, text="No video selected", foreground="gray"
        )
        self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_btn = ttk.Button(
            file_row, text="Browse...", command=self._browse_video
        )
        self.browse_btn.pack(side=tk.RIGHT)

        # Video info display
        self.video_info_label = ttk.Label(
            video_frame, text="", foreground="gray", font=('TkDefaultFont', 9)
        )
        self.video_info_label.pack(fill=tk.X, pady=(5, 0))

        # Output Directory Section
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding="5")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        output_row = ttk.Frame(output_frame)
        output_row.pack(fill=tk.X)

        self.output_label = ttk.Label(
            output_row, text=self.output_dir, foreground="blue"
        )
        self.output_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(
            output_row, text="Browse...", command=self._browse_output
        ).pack(side=tk.RIGHT)

        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="Decoding Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Profile selection
        profile_row = ttk.Frame(settings_frame)
        profile_row.pack(fill=tk.X, pady=2)

        ttk.Label(profile_row, text="Profile:").pack(side=tk.LEFT)
        profiles = [
            ("Conservative", "conservative"),
            ("Standard", "standard"),
            ("Aggressive", "aggressive"),
            ("Ultra", "ultra")
        ]
        for text, value in profiles:
            ttk.Radiobutton(
                profile_row, text=text, value=value,
                variable=self.profile_var
            ).pack(side=tk.LEFT, padx=5)

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

        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Processing Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        # Overall progress bar
        ttk.Label(progress_frame, text="Video progress:").pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=2)

        # Current file progress bar
        ttk.Label(progress_frame, text="Current file:").pack(anchor=tk.W)
        self.file_progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate'
        )
        self.file_progress_bar.pack(fill=tk.X, pady=2)

        # Progress labels
        self.progress_label = ttk.Label(
            progress_frame, text="Ready to process"
        )
        self.progress_label.pack(fill=tk.X)

        self.stats_label = ttk.Label(
            progress_frame, text="", foreground="gray"
        )
        self.stats_label.pack(fill=tk.X)

        # State indicator
        self.state_label = ttk.Label(
            progress_frame, text="State: Idle", foreground="gray"
        )
        self.state_label.pack(fill=tk.X)

        # Decoded Files Section
        files_frame = ttk.LabelFrame(main_frame, text="Decoded Files", padding="5")
        files_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Files list with scrollbar
        list_frame = ttk.Frame(files_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview for files
        columns = ('filename', 'size', 'status', 'hash')
        self.files_tree = ttk.Treeview(
            list_frame, columns=columns, show='headings', height=8
        )

        self.files_tree.heading('filename', text='Filename')
        self.files_tree.heading('size', text='Size')
        self.files_tree.heading('status', text='Status')
        self.files_tree.heading('hash', text='Hash')

        self.files_tree.column('filename', width=250)
        self.files_tree.column('size', width=80)
        self.files_tree.column('status', width=100)
        self.files_tree.column('hash', width=100)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=scrollbar.set)

        self.files_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Files summary
        self.files_summary_label = ttk.Label(
            files_frame, text="No files decoded yet", foreground="gray"
        )
        self.files_summary_label.pack(fill=tk.X, pady=(5, 0))

        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_btn = ttk.Button(
            control_frame, text="Start Processing",
            command=self._on_start_clicked
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame, text="Stop",
            command=self._on_stop_clicked, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.open_folder_btn = ttk.Button(
            control_frame, text="Open Output Folder",
            command=self._open_output_folder
        )
        self.open_folder_btn.pack(side=tk.LEFT, padx=5)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        fec_status = "FEC: OK" if check_fec_available() else "FEC: N/A"
        crypto_status = "Crypto: OK" if check_crypto_available() else "Crypto: N/A"

        ttk.Label(
            status_frame, text=f"{fec_status} | {crypto_status}",
            foreground="gray", font=('TkDefaultFont', 8)
        ).pack(side=tk.LEFT)

    def _browse_video(self):
        """Browse for video file."""
        file_path = filedialog.askopenfilename(
            title="Select recorded video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.webm"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.video_path = file_path

            # Display truncated path
            display = file_path
            if len(display) > 60:
                display = "..." + display[-57:]
            self.video_label.config(text=display, foreground="black")

            # Get video info
            self._update_video_info(file_path)

    def _update_video_info(self, video_path: str):
        """Update video info display."""
        def load_info():
            info = get_video_info(video_path)
            self.root.after(0, lambda: self._display_video_info(info))

        # Load in background to avoid UI freeze
        threading.Thread(target=load_info, daemon=True).start()

    def _display_video_info(self, info: Optional[dict]):
        """Display video info in UI."""
        if info:
            self.video_info = info
            duration = format_duration(info['duration'])
            text = (
                f"{info['width']}x{info['height']} @ {info['fps']:.1f} FPS | "
                f"{info['frames']:,} frames | Duration: {duration}"
            )
            self.video_info_label.config(text=text)
        else:
            self.video_info = None
            self.video_info_label.config(text="Could not read video info")

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

    def _on_start_clicked(self):
        """Handle start button click."""
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file to process.")
            return

        if not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return

        self.is_processing = True
        self._update_control_buttons()

        # Clear previous files
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        self.files_summary_label.config(text="Processing...")

        if self.on_start:
            settings = self._get_settings()
            self.on_start(settings)

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if messagebox.askyesno("Confirm Stop", "Stop processing?"):
            self.is_processing = False
            self._update_control_buttons()
            if self.on_stop:
                self.on_stop()

    def _open_output_folder(self):
        """Open output folder in file explorer."""
        if os.path.exists(self.output_dir):
            import subprocess
            if os.name == 'nt':
                subprocess.Popen(['explorer', self.output_dir])
            else:
                subprocess.Popen(['xdg-open', self.output_dir])

    def _get_settings(self) -> dict:
        """Get current settings."""
        profile_name = self.profile_var.get()
        if profile_name == "conservative":
            profile = PROFILE_CONSERVATIVE
        elif profile_name == "aggressive":
            profile = PROFILE_AGGRESSIVE
        elif profile_name == "ultra":
            profile = PROFILE_ULTRA
        else:
            profile = PROFILE_STANDARD

        return {
            'video_path': self.video_path,
            'output_dir': self.output_dir,
            'profile': profile,
            'password': self.password_var.get() or None
        }

    def _update_control_buttons(self):
        """Update button states."""
        if self.is_processing:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.DISABLED)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.browse_btn.config(state=tk.NORMAL)

    def update_progress(self, progress: ProcessorProgress):
        """Update progress display."""
        # Video progress
        if progress.total_frames > 0:
            video_percent = (progress.current_frame / progress.total_frames) * 100
            self.progress_bar['value'] = video_percent

        # Current file progress
        if progress.current_file_total_blocks > 0:
            file_percent = (progress.current_file_blocks / progress.current_file_total_blocks) * 100
            self.file_progress_bar['value'] = file_percent
        else:
            self.file_progress_bar['value'] = 0

        # Progress text
        self.progress_label.config(
            text=f"Frame {progress.current_frame:,} / {progress.total_frames:,} | "
                 f"Files found: {progress.files_found}"
        )

        # Stats
        self.stats_label.config(
            text=f"Files decoded: {progress.files_decoded} | "
                 f"Gap frames skipped: {progress.gap_frames:,}"
        )

        # State
        state_colors = {
            ProcessorState.SCANNING: ("Scanning for sync...", "orange"),
            ProcessorState.RECEIVING: ("Receiving data", "green"),
            ProcessorState.END_DETECTED: ("End detected", "blue"),
            ProcessorState.GAP: ("In gap between files", "gray")
        }
        text, color = state_colors.get(progress.state, ("Unknown", "gray"))
        self.state_label.config(text=f"State: {text}", foreground=color)

    def add_decoded_file(self, file_info: dict):
        """Add a decoded file to the list."""
        filename = file_info.get('filename', 'Unknown')
        size = format_size(file_info.get('file_size', 0))

        blocks = file_info.get('blocks_received', 0)
        total = file_info.get('total_blocks', 0)
        if blocks == total:
            status = "Complete"
        else:
            status = f"Partial ({blocks}/{total})"

        hash_valid = file_info.get('hash_valid')
        if hash_valid is True:
            hash_text = "Verified"
        elif hash_valid is False:
            hash_text = "MISMATCH"
        else:
            hash_text = "N/A"

        self.files_tree.insert('', tk.END, values=(filename, size, status, hash_text))

        # Update summary
        children = self.files_tree.get_children()
        self.files_summary_label.config(
            text=f"{len(children)} file(s) decoded"
        )

    def show_complete(self, success: bool, message: str = "", files_decoded: int = 0, elapsed: float = 0):
        """Show completion status."""
        self.is_processing = False
        self._update_control_buttons()

        if success:
            self.progress_bar['value'] = 100
            self.progress_label.config(text="Processing complete!")
            self.state_label.config(text="State: Complete", foreground="green")

            messagebox.showinfo(
                "Complete",
                f"Processing complete!\n\n"
                f"Files decoded: {files_decoded}\n"
                f"Time: {format_duration(elapsed)}\n\n"
                f"Output directory:\n{self.output_dir}"
            )
        else:
            self.progress_label.config(text=f"Processing stopped: {message}")
            self.state_label.config(text="State: Stopped", foreground="red")

            if message:
                messagebox.showerror("Error", message)

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
