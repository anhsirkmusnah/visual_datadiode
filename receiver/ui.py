"""
Visual Data Diode - Receiver UI

Tkinter-based user interface for decoding video files and
real-time binary HDMI capture.
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
    ENHANCED_PROFILE_CONSERVATIVE, ENHANCED_PROFILE_STANDARD,
    check_fec_available, check_crypto_available, check_cuda_available, get_cuda_info
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

    Provides two tabs:
    - Video Decode: Decode files from recorded video
    - Binary Stream: Real-time capture from HDMI capture device
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visual Data Diode - Receiver")
        self.root.geometry("750x750")
        self.root.resizable(True, True)
        self.root.minsize(650, 650)

        # State - Video Decode
        self.is_processing = False
        self.video_path: Optional[str] = None
        self.output_dir = str(Path.home() / "Downloads")
        self.video_info: Optional[dict] = None

        # State - Binary Stream
        self.is_capturing = False
        self.bs_output_dir = str(Path.home() / "Downloads")

        # Callbacks - Video Decode
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None

        # Callbacks - Binary Stream
        self.on_capture_start: Optional[Callable] = None
        self.on_capture_stop: Optional[Callable] = None

        # Variables - Video Decode
        self.profile_var = tk.StringVar(value="enhanced_conservative")
        self.password_var = tk.StringVar(value="")
        self.use_enhanced_var = tk.BooleanVar(value=True)
        self.use_cuda_var = tk.BooleanVar(value=check_cuda_available())
        self.batch_size_var = tk.IntVar(value=32)

        # Variables - Binary Stream
        self.bs_device_var = tk.StringVar(value="")
        self.bs_fec_var = tk.StringVar(value="10%")
        self.bs_timeout_var = tk.IntVar(value=5)

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create notebook with two tabs."""
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Video Decode
        video_tab = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(video_tab, text="Video Decode")
        self._create_video_tab(video_tab)

        # Tab 2: Binary Stream
        binary_tab = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(binary_tab, text="Binary Stream")
        self._create_binary_tab(binary_tab)

        # Status bar (shared)
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        fec_status = "FEC: OK" if check_fec_available() else "FEC: N/A"
        crypto_status = "Crypto: OK" if check_crypto_available() else "Crypto: N/A"
        cuda_status = "CUDA: OK" if check_cuda_available() else "CUDA: N/A"

        ttk.Label(
            status_frame, text=f"{fec_status} | {crypto_status} | {cuda_status}",
            foreground="gray", font=('TkDefaultFont', 8)
        ).pack(side=tk.LEFT)

    # ── Video Decode Tab ─────────────────────────────────────────────

    def _create_video_tab(self, parent):
        """Create all widgets for the Video Decode tab."""
        # Video File
        video_frame = ttk.LabelFrame(parent, text="Input Video", padding="5")
        video_frame.pack(fill=tk.X, pady=(0, 5))

        file_row = ttk.Frame(video_frame)
        file_row.pack(fill=tk.X, pady=2)

        self.video_label = ttk.Label(file_row, text="No video selected", foreground="gray")
        self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.browse_btn = ttk.Button(file_row, text="Browse...", command=self._browse_video)
        self.browse_btn.pack(side=tk.RIGHT)

        self.video_info_label = ttk.Label(video_frame, text="", foreground="gray", font=('TkDefaultFont', 9))
        self.video_info_label.pack(fill=tk.X, pady=(5, 0))

        # Output Directory
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding="5")
        output_frame.pack(fill=tk.X, pady=(0, 5))

        output_row = ttk.Frame(output_frame)
        output_row.pack(fill=tk.X)
        self.output_label = ttk.Label(output_row, text=self.output_dir, foreground="blue")
        self.output_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_row, text="Browse...", command=self._browse_output).pack(side=tk.RIGHT)

        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Decoding Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        # Enhanced encoding
        encoding_row = ttk.Frame(settings_frame)
        encoding_row.pack(fill=tk.X, pady=2)
        self.enhanced_check = ttk.Checkbutton(
            encoding_row, text="Enhanced encoding (8 luma + 4 color)",
            variable=self.use_enhanced_var, command=self._on_enhanced_changed
        )
        self.enhanced_check.pack(side=tk.LEFT)

        # Profile
        profile_row = ttk.Frame(settings_frame)
        profile_row.pack(fill=tk.X, pady=2)
        ttk.Label(profile_row, text="Profile:").pack(side=tk.LEFT)

        self.std_profiles_frame = ttk.Frame(profile_row)
        for text, value in [("Conservative", "conservative"), ("Standard", "standard"),
                            ("Aggressive", "aggressive"), ("Ultra", "ultra")]:
            ttk.Radiobutton(self.std_profiles_frame, text=text, value=value,
                            variable=self.profile_var).pack(side=tk.LEFT, padx=5)

        self.enhanced_profiles_frame = ttk.Frame(profile_row)
        for text, value in [("Enhanced Conservative", "enhanced_conservative"),
                            ("Enhanced Standard", "enhanced_standard")]:
            ttk.Radiobutton(self.enhanced_profiles_frame, text=text, value=value,
                            variable=self.profile_var).pack(side=tk.LEFT, padx=5)

        self._on_enhanced_changed()

        # Password
        password_row = ttk.Frame(settings_frame)
        password_row.pack(fill=tk.X, pady=2)
        ttk.Label(password_row, text="Password:").pack(side=tk.LEFT)
        self.password_entry = ttk.Entry(password_row, textvariable=self.password_var, show="*", width=30)
        self.password_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(password_row, text="(leave empty if not encrypted)", foreground="gray").pack(side=tk.LEFT)

        # CUDA
        cuda_frame = ttk.LabelFrame(parent, text="GPU Acceleration", padding="5")
        cuda_frame.pack(fill=tk.X, pady=(0, 5))

        cuda_enable_row = ttk.Frame(cuda_frame)
        cuda_enable_row.pack(fill=tk.X, pady=2)

        cuda_available = check_cuda_available()
        self.cuda_check = ttk.Checkbutton(cuda_enable_row, text="Enable CUDA GPU acceleration",
                                          variable=self.use_cuda_var, command=self._on_cuda_changed)
        self.cuda_check.pack(side=tk.LEFT)
        if not cuda_available:
            self.cuda_check.config(state=tk.DISABLED)
            self.use_cuda_var.set(False)
            ttk.Label(cuda_enable_row, text="(CUDA not available)", foreground="gray").pack(side=tk.LEFT, padx=5)

        cuda_params_row = ttk.Frame(cuda_frame)
        cuda_params_row.pack(fill=tk.X, pady=2)
        ttk.Label(cuda_params_row, text="Batch size:").pack(side=tk.LEFT)
        self.batch_spinbox = ttk.Spinbox(cuda_params_row, from_=1, to=128, width=5,
                                         textvariable=self.batch_size_var)
        self.batch_spinbox.pack(side=tk.LEFT, padx=5)
        if not cuda_available:
            self.batch_spinbox.config(state=tk.DISABLED)

        cuda_info = get_cuda_info()
        if cuda_info['available']:
            cuda_text = f"GPU: {cuda_info['device_name']} | Memory: {cuda_info['memory_free'] / 1024**3:.1f} GB free"
        else:
            cuda_text = "GPU: Not available (install CuPy for CUDA support)"
        self.cuda_info_label = ttk.Label(cuda_frame, text=cuda_text, foreground="gray")
        self.cuda_info_label.pack(fill=tk.X)

        # Progress
        progress_frame = ttk.LabelFrame(parent, text="Processing Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(progress_frame, text="Video progress:").pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=2)

        ttk.Label(progress_frame, text="Current file:").pack(anchor=tk.W)
        self.file_progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.file_progress_bar.pack(fill=tk.X, pady=2)

        self.progress_label = ttk.Label(progress_frame, text="Ready to process")
        self.progress_label.pack(fill=tk.X)
        self.stats_label = ttk.Label(progress_frame, text="", foreground="gray")
        self.stats_label.pack(fill=tk.X)
        self.state_label = ttk.Label(progress_frame, text="State: Idle", foreground="gray")
        self.state_label.pack(fill=tk.X)

        # Decoded Files
        files_frame = ttk.LabelFrame(parent, text="Decoded Files", padding="5")
        files_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        list_frame = ttk.Frame(files_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        columns = ('filename', 'size', 'status', 'hash')
        self.files_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=6)
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

        self.files_summary_label = ttk.Label(files_frame, text="No files decoded yet", foreground="gray")
        self.files_summary_label.pack(fill=tk.X, pady=(5, 0))

        # Controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(5, 0))

        self.start_btn = ttk.Button(control_frame, text="Start Processing", command=self._on_start_clicked)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._on_stop_clicked, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.open_folder_btn = ttk.Button(control_frame, text="Open Output Folder", command=self._open_output_folder)
        self.open_folder_btn.pack(side=tk.LEFT, padx=5)

    # ── Binary Stream Tab ────────────────────────────────────────────

    def _create_binary_tab(self, parent):
        """Create all widgets for the Binary Stream capture tab."""
        # Capture Device
        device_frame = ttk.LabelFrame(parent, text="Capture Device", padding="5")
        device_frame.pack(fill=tk.X, pady=(0, 5))

        dev_row = ttk.Frame(device_frame)
        dev_row.pack(fill=tk.X)

        ttk.Label(dev_row, text="Device:").pack(side=tk.LEFT)
        self.bs_device_combo = ttk.Combobox(dev_row, textvariable=self.bs_device_var,
                                            state="readonly", width=40)
        self.bs_device_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.bs_refresh_btn = ttk.Button(dev_row, text="Refresh", command=self._bs_refresh_devices)
        self.bs_refresh_btn.pack(side=tk.RIGHT)

        self._bs_refresh_devices()

        # Output Directory
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding="5")
        output_frame.pack(fill=tk.X, pady=(0, 5))

        out_row = ttk.Frame(output_frame)
        out_row.pack(fill=tk.X)
        self.bs_output_label = ttk.Label(out_row, text=self.bs_output_dir, foreground="blue")
        self.bs_output_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse...", command=self._bs_browse_output).pack(side=tk.RIGHT)

        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="FEC:").pack(side=tk.LEFT)
        self.bs_fec_combo = ttk.Combobox(row1, textvariable=self.bs_fec_var, state="readonly",
                                         width=5, values=["0%", "5%", "10%"])
        self.bs_fec_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Idle timeout (s):").pack(side=tk.LEFT, padx=(20, 0))
        self.bs_timeout_spin = ttk.Spinbox(row1, from_=2, to=30, width=5, textvariable=self.bs_timeout_var)
        self.bs_timeout_spin.pack(side=tk.LEFT, padx=5)

        # Progress
        prog_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        prog_frame.pack(fill=tk.X, pady=(0, 5))

        self.bs_progress_bar = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.bs_progress_bar.pack(fill=tk.X, pady=2)

        self.bs_progress_label = ttk.Label(prog_frame, text="Ready")
        self.bs_progress_label.pack(fill=tk.X)

        self.bs_state_label = ttk.Label(prog_frame, text="State: Idle", foreground="gray")
        self.bs_state_label.pack(fill=tk.X)

        # Stats
        stats_frame = ttk.LabelFrame(parent, text="Stats", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.bs_stats_label = ttk.Label(stats_frame, text="No data yet", foreground="gray")
        self.bs_stats_label.pack(fill=tk.X)

        # Decoded Files
        files_frame = ttk.LabelFrame(parent, text="Decoded Files", padding="5")
        files_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        list_frame = ttk.Frame(files_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        columns = ('filename', 'size', 'status', 'hash')
        self.bs_files_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=5)
        self.bs_files_tree.heading('filename', text='Filename')
        self.bs_files_tree.heading('size', text='Size')
        self.bs_files_tree.heading('status', text='Status')
        self.bs_files_tree.heading('hash', text='Hash')
        self.bs_files_tree.column('filename', width=250)
        self.bs_files_tree.column('size', width=80)
        self.bs_files_tree.column('status', width=100)
        self.bs_files_tree.column('hash', width=100)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.bs_files_tree.yview)
        self.bs_files_tree.configure(yscrollcommand=scrollbar.set)
        self.bs_files_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.bs_files_summary = ttk.Label(files_frame, text="No files decoded yet", foreground="gray")
        self.bs_files_summary.pack(fill=tk.X, pady=(5, 0))

        # Controls
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, pady=(5, 0))

        self.bs_start_btn = ttk.Button(ctrl_frame, text="Start Capture", command=self._bs_on_start)
        self.bs_start_btn.pack(side=tk.LEFT, padx=5)
        self.bs_stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self._bs_on_stop, state=tk.DISABLED)
        self.bs_stop_btn.pack(side=tk.LEFT, padx=5)
        self.bs_open_btn = ttk.Button(ctrl_frame, text="Open Output Folder", command=self._bs_open_output)
        self.bs_open_btn.pack(side=tk.LEFT, padx=5)

    # ── Binary Stream: device management ─────────────────────────────

    def _bs_refresh_devices(self):
        """Refresh capture device list."""
        try:
            from .capture import list_capture_devices
            devices = list_capture_devices()
        except Exception:
            devices = [{'index': 0, 'description': 'Device 0'}]

        self._bs_devices = devices
        descriptions = [d.get('description', f"Device {d['index']}") for d in devices]
        self.bs_device_combo['values'] = descriptions

        # Default to first 1080p device or first device
        selected = 0
        for i, d in enumerate(devices):
            if d.get('is_capture_card', False):
                selected = i
                break
        if devices:
            self.bs_device_combo.current(selected)

    def _bs_get_device_index(self) -> int:
        idx = self.bs_device_combo.current()
        if idx >= 0 and idx < len(self._bs_devices):
            return self._bs_devices[idx]['index']
        return 0

    def _bs_get_fec_ratio(self) -> float:
        fec_str = self.bs_fec_var.get()
        return int(fec_str.replace('%', '')) / 100.0

    # ── Binary Stream: output directory ──────────────────────────────

    def _bs_browse_output(self):
        directory = filedialog.askdirectory(title="Select output directory", initialdir=self.bs_output_dir)
        if directory:
            self.bs_output_dir = directory
            display = directory if len(directory) <= 50 else "..." + directory[-47:]
            self.bs_output_label.config(text=display)

    def _bs_open_output(self):
        if os.path.exists(self.bs_output_dir):
            import subprocess
            if os.name == 'nt':
                subprocess.Popen(['explorer', self.bs_output_dir])
            else:
                subprocess.Popen(['xdg-open', self.bs_output_dir])

    # ── Binary Stream: start / stop ──────────────────────────────────

    def _bs_on_start(self):
        self.is_capturing = True
        self._bs_update_controls()

        # Clear previous files
        for item in self.bs_files_tree.get_children():
            self.bs_files_tree.delete(item)
        self.bs_files_summary.config(text="Capturing...")

        if self.on_capture_start:
            self.on_capture_start(self._bs_get_settings())

    def _bs_on_stop(self):
        if messagebox.askyesno("Confirm Stop", "Stop capture?"):
            self.is_capturing = False
            self._bs_update_controls()
            if self.on_capture_stop:
                self.on_capture_stop()

    def _bs_get_settings(self) -> dict:
        return {
            'device_index': self._bs_get_device_index(),
            'output_dir': self.bs_output_dir,
            'fec_ratio': self._bs_get_fec_ratio(),
            'idle_timeout': self.bs_timeout_var.get(),
        }

    def _bs_update_controls(self):
        if self.is_capturing:
            self.bs_start_btn.config(state=tk.DISABLED)
            self.bs_stop_btn.config(state=tk.NORMAL)
        else:
            self.bs_start_btn.config(state=tk.NORMAL)
            self.bs_stop_btn.config(state=tk.DISABLED)

    # ── Binary Stream: progress updates from main.py ─────────────────

    def update_capture_progress(self, stats: dict):
        """Update binary stream capture progress (called from main thread)."""
        total = stats.get('total_blocks', 0)
        received = stats.get('blocks_received', 0)

        if total > 0:
            pct = (received / total) * 100
            self.bs_progress_bar['value'] = pct
            self.bs_progress_label.config(
                text=f"Blocks: {received:,}/{total:,}  |  {stats.get('mbps', 0):.1f} MB/s"
            )
        else:
            self.bs_progress_label.config(text=f"Blocks: {received:,}  |  Waiting for session info...")

        # State
        state = stats.get('state', 'idle')
        state_colors = {
            'waiting': ("Waiting for data...", "orange"),
            'receiving': ("Receiving", "green"),
            'assembling': ("Assembling file...", "blue"),
        }
        text, color = state_colors.get(state, (state.capitalize(), "gray"))
        self.bs_state_label.config(text=f"State: {text}", foreground=color)

        # Stats
        self.bs_stats_label.config(text=(
            f"Decoded: {stats.get('frames_decoded', 0):,}  |  "
            f"Failed: {stats.get('frames_failed', 0):,}  |  "
            f"FEC: {stats.get('fec_corrections', 0):,}  |  "
            f"Dupes: {stats.get('duplicates', 0):,}"
        ))

    def add_capture_file(self, file_info: dict):
        """Add a decoded file to the binary stream files list."""
        filename = file_info.get('filename', 'Unknown')
        size = format_size(file_info.get('size', 0))

        blocks = file_info.get('blocks_received', 0)
        total = file_info.get('total_blocks', 0)
        if total > 0 and blocks >= total:
            status = "Complete"
        elif total > 0:
            status = f"Partial ({blocks}/{total})"
        else:
            status = "Unknown"

        hash_valid = file_info.get('hash_valid')
        if hash_valid is True:
            hash_text = "Verified"
        elif hash_valid is False:
            hash_text = "MISMATCH"
        else:
            hash_text = "N/A"

        self.bs_files_tree.insert('', tk.END, values=(filename, size, status, hash_text))

        children = self.bs_files_tree.get_children()
        self.bs_files_summary.config(text=f"{len(children)} file(s) decoded")

    def show_capture_complete(self, success: bool, message: str = ""):
        """Show binary stream capture completion."""
        self.is_capturing = False
        self._bs_update_controls()

        if success:
            self.bs_progress_bar['value'] = 100
            self.bs_state_label.config(text="State: Complete", foreground="green")
            messagebox.showinfo("Complete", message or "Capture complete!")
        else:
            self.bs_state_label.config(text="State: Stopped", foreground="red")
            if message:
                messagebox.showerror("Error", message)

    # ── Video Decode: existing methods (unchanged) ───────────────────

    def _browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select recorded video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            display = file_path if len(file_path) <= 60 else "..." + file_path[-57:]
            self.video_label.config(text=display, foreground="black")
            self._update_video_info(file_path)

    def _update_video_info(self, video_path: str):
        def load_info():
            info = get_video_info(video_path)
            self.root.after(0, lambda: self._display_video_info(info))
        threading.Thread(target=load_info, daemon=True).start()

    def _display_video_info(self, info: Optional[dict]):
        if info:
            self.video_info = info
            duration = format_duration(info['duration'])
            text = (f"{info['width']}x{info['height']} @ {info['fps']:.1f} FPS | "
                    f"{info['frames']:,} frames | Duration: {duration}")
            self.video_info_label.config(text=text)
        else:
            self.video_info = None
            self.video_info_label.config(text="Could not read video info")

    def _browse_output(self):
        directory = filedialog.askdirectory(title="Select output directory", initialdir=self.output_dir)
        if directory:
            self.output_dir = directory
            display = directory if len(directory) <= 50 else "..." + directory[-47:]
            self.output_label.config(text=display)

    def _on_start_clicked(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file to process.")
            return
        if not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return

        self.is_processing = True
        self._update_control_buttons()

        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        self.files_summary_label.config(text="Processing...")

        if self.on_start:
            settings = self._get_settings()
            self.on_start(settings)

    def _on_stop_clicked(self):
        if messagebox.askyesno("Confirm Stop", "Stop processing?"):
            self.is_processing = False
            self._update_control_buttons()
            if self.on_stop:
                self.on_stop()

    def _open_output_folder(self):
        if os.path.exists(self.output_dir):
            import subprocess
            if os.name == 'nt':
                subprocess.Popen(['explorer', self.output_dir])
            else:
                subprocess.Popen(['xdg-open', self.output_dir])

    def _on_enhanced_changed(self):
        if self.use_enhanced_var.get():
            self.std_profiles_frame.pack_forget()
            self.enhanced_profiles_frame.pack(side=tk.LEFT)
            if self.profile_var.get() not in ["enhanced_conservative", "enhanced_standard"]:
                self.profile_var.set("enhanced_conservative")
        else:
            self.enhanced_profiles_frame.pack_forget()
            self.std_profiles_frame.pack(side=tk.LEFT)
            if self.profile_var.get() not in ["conservative", "standard", "aggressive", "ultra"]:
                self.profile_var.set("conservative")

    def _on_cuda_changed(self):
        state = tk.NORMAL if self.use_cuda_var.get() else tk.DISABLED
        self.batch_spinbox.config(state=state)

    def _get_settings(self) -> dict:
        profile_name = self.profile_var.get()
        if profile_name == "conservative":
            profile = PROFILE_CONSERVATIVE
        elif profile_name == "aggressive":
            profile = PROFILE_AGGRESSIVE
        elif profile_name == "ultra":
            profile = PROFILE_ULTRA
        elif profile_name == "enhanced_conservative":
            profile = ENHANCED_PROFILE_CONSERVATIVE
        elif profile_name == "enhanced_standard":
            profile = ENHANCED_PROFILE_STANDARD
        else:
            profile = PROFILE_STANDARD

        return {
            'video_path': self.video_path,
            'output_dir': self.output_dir,
            'profile': profile,
            'password': self.password_var.get() or None,
            'use_enhanced': self.use_enhanced_var.get(),
            'use_cuda': self.use_cuda_var.get(),
            'batch_size': self.batch_size_var.get(),
        }

    def _update_control_buttons(self):
        if self.is_processing:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.DISABLED)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.browse_btn.config(state=tk.NORMAL)

    def update_progress(self, progress: ProcessorProgress):
        if progress.total_frames > 0:
            video_percent = (progress.current_frame / progress.total_frames) * 100
            self.progress_bar['value'] = video_percent

        if progress.current_file_total_blocks > 0:
            file_percent = (progress.current_file_blocks / progress.current_file_total_blocks) * 100
            self.file_progress_bar['value'] = file_percent
        else:
            self.file_progress_bar['value'] = 0

        self.progress_label.config(
            text=f"Frame {progress.current_frame:,} / {progress.total_frames:,} | "
                 f"Files found: {progress.files_found}")

        self.stats_label.config(
            text=f"Files decoded: {progress.files_decoded} | "
                 f"Gap frames skipped: {progress.gap_frames:,}")

        state_colors = {
            ProcessorState.SCANNING: ("Scanning for sync...", "orange"),
            ProcessorState.RECEIVING: ("Receiving data", "green"),
            ProcessorState.END_DETECTED: ("End detected", "blue"),
            ProcessorState.GAP: ("In gap between files", "gray")
        }
        text, color = state_colors.get(progress.state, ("Unknown", "gray"))
        self.state_label.config(text=f"State: {text}", foreground=color)

    def add_decoded_file(self, file_info: dict):
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

        children = self.files_tree.get_children()
        self.files_summary_label.config(text=f"{len(children)} file(s) decoded")

    def show_complete(self, success: bool, message: str = "", files_decoded: int = 0, elapsed: float = 0):
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
                f"Output directory:\n{self.output_dir}")
        else:
            self.progress_label.config(text=f"Processing stopped: {message}")
            self.state_label.config(text="State: Stopped", foreground="red")
            if message:
                messagebox.showerror("Error", message)

    def run(self):
        self.root.mainloop()

    def close(self):
        self.root.destroy()


def run_receiver_ui():
    """Run the receiver UI as standalone."""
    ui = ReceiverUI()
    ui.run()


if __name__ == "__main__":
    run_receiver_ui()
