"""
Visual Data Diode - Sender UI

Tkinter-based user interface for encoding files to video and
real-time binary HDMI streaming.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
from typing import Optional, Callable, List
from pathlib import Path
from multiprocessing import cpu_count

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    PROFILE_CONSERVATIVE, PROFILE_STANDARD, PROFILE_AGGRESSIVE, PROFILE_ULTRA,
    ENHANCED_PROFILE_CONSERVATIVE, ENHANCED_PROFILE_STANDARD,
    DEFAULT_FPS, check_fec_available, check_crypto_available
)
from shared.binary_frame import calculate_binary_payload_capacity
from .timing import format_duration
from .video_encoder import check_ffmpeg_available, get_encoder_info


class SenderUI:
    """
    Tkinter UI for the Visual Data Diode sender.

    Provides two tabs:
    - Video Encode: Encode files to video (offline)
    - Binary Stream: Real-time HDMI pixel streaming
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visual Data Diode - Sender")
        self.root.geometry("720x700")
        self.root.resizable(True, True)
        self.root.minsize(620, 600)

        # State - Video Encode tab
        self.file_paths: List[str] = []
        self.total_size: int = 0
        self.output_dir: str = str(Path.home() / "Videos")
        self.is_encoding = False
        self.current_file_index = 0

        # State - Binary Stream tab
        self.bs_file_paths: List[str] = []
        self.bs_total_size: int = 0
        self.is_streaming = False

        # Callbacks - Video Encode
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None

        # Callbacks - Binary Stream
        self.on_stream_start: Optional[Callable] = None
        self.on_stream_stop: Optional[Callable] = None

        # Variables - Video Encode
        self.profile_var = tk.StringVar(value="conservative")
        self.fps_var = tk.IntVar(value=DEFAULT_FPS)
        self.repeat_var = tk.IntVar(value=2)
        self.encrypt_var = tk.BooleanVar(value=False)
        self.resolution_var = tk.StringVar(value="1920x1080")
        self.add_audio_var = tk.BooleanVar(value=True)

        # HPC options
        self.use_enhanced_var = tk.BooleanVar(value=True)
        self.use_hpc_var = tk.BooleanVar(value=True)
        self.workers_var = tk.IntVar(value=max(1, cpu_count() - 1))
        self.batch_size_var = tk.IntVar(value=100)
        self.crf_var = tk.IntVar(value=18)

        # Variables - Binary Stream
        self.bs_display_var = tk.StringVar(value="")
        self.bs_fps_var = tk.IntVar(value=60)
        self.bs_repeat_var = tk.IntVar(value=1)
        self.bs_fec_var = tk.StringVar(value="10%")
        self.bs_calibration_var = tk.DoubleVar(value=2.0)

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create notebook with two tabs."""
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Video Encode
        video_tab = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(video_tab, text="Video Encode")
        self._create_video_tab(video_tab)

        # Tab 2: Binary Stream
        binary_tab = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(binary_tab, text="Binary Stream")
        self._create_binary_tab(binary_tab)

        # Status bar (shared)
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        encoder_info = get_encoder_info()
        fec_status = "FEC: OK" if check_fec_available() else "FEC: N/A"
        ffmpeg_status = "FFmpeg: OK" if encoder_info['ffmpeg_available'] else "FFmpeg: N/A"
        crypto_status = "Crypto: OK" if check_crypto_available() else "Crypto: N/A"

        ttk.Label(
            status_frame,
            text=f"{fec_status} | {ffmpeg_status} | {crypto_status}",
            foreground="gray", font=('TkDefaultFont', 8)
        ).pack(side=tk.LEFT)

    # ── Video Encode Tab ─────────────────────────────────────────────

    def _create_video_tab(self, parent):
        """Create all widgets for the Video Encode tab."""
        # File Selection
        file_frame = ttk.LabelFrame(parent, text="Files to Encode", padding="5")
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=5)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        self.add_files_btn = ttk.Button(btn_frame, text="Add Files...", command=self._add_files)
        self.add_files_btn.pack(side=tk.LEFT, padx=2)
        self.remove_btn = ttk.Button(btn_frame, text="Remove Selected", command=self._remove_selected)
        self.remove_btn.pack(side=tk.LEFT, padx=2)
        self.clear_btn = ttk.Button(btn_frame, text="Clear All", command=self._clear_files)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        self.files_info_label = ttk.Label(file_frame, text="No files selected", foreground="gray")
        self.files_info_label.pack(fill=tk.X, pady=(5, 0))

        # Output Directory
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding="5")
        output_frame.pack(fill=tk.X, pady=(0, 5))

        self.output_label = ttk.Label(output_frame, text=self.output_dir, foreground="blue")
        self.output_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.browse_output_btn = ttk.Button(output_frame, text="Browse...", command=self._browse_output)
        self.browse_output_btn.pack(side=tk.RIGHT)

        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Encoding Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        # Enhanced encoding
        encoding_type_frame = ttk.Frame(settings_frame)
        encoding_type_frame.pack(fill=tk.X, pady=2)
        self.enhanced_check = ttk.Checkbutton(
            encoding_type_frame, text="Enhanced encoding (8 luma + 4 color, +75% capacity)",
            variable=self.use_enhanced_var, command=self._on_enhanced_changed
        )
        self.enhanced_check.pack(side=tk.LEFT)

        # Profile
        profile_frame = ttk.Frame(settings_frame)
        profile_frame.pack(fill=tk.X, pady=2)
        ttk.Label(profile_frame, text="Profile:").pack(side=tk.LEFT)

        self.std_profiles_frame = ttk.Frame(profile_frame)
        for text, value in [("Conservative", "conservative"), ("Standard", "standard"),
                            ("Aggressive", "aggressive"), ("Ultra", "ultra")]:
            ttk.Radiobutton(self.std_profiles_frame, text=text, value=value,
                            variable=self.profile_var, command=self._update_estimates
                            ).pack(side=tk.LEFT, padx=5)

        self.enhanced_profiles_frame = ttk.Frame(profile_frame)
        for text, value in [("Enhanced Conservative", "enhanced_conservative"),
                            ("Enhanced Standard", "enhanced_standard")]:
            ttk.Radiobutton(self.enhanced_profiles_frame, text=text, value=value,
                            variable=self.profile_var, command=self._update_estimates
                            ).pack(side=tk.LEFT, padx=5)

        self._on_enhanced_changed()

        # FPS / Repeat
        fps_frame = ttk.Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_spinbox = ttk.Spinbox(fps_frame, from_=10, to=60, width=5,
                                       textvariable=self.fps_var, command=self._update_estimates)
        self.fps_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(fps_frame, text="Repeat count:").pack(side=tk.LEFT, padx=(20, 0))
        self.repeat_spinbox = ttk.Spinbox(fps_frame, from_=1, to=5, width=5,
                                          textvariable=self.repeat_var, command=self._update_estimates)
        self.repeat_spinbox.pack(side=tk.LEFT, padx=5)

        # Resolution
        resolution_frame = ttk.Frame(settings_frame)
        resolution_frame.pack(fill=tk.X, pady=2)
        ttk.Label(resolution_frame, text="Resolution:").pack(side=tk.LEFT)
        self.resolution_combo = ttk.Combobox(resolution_frame, width=12,
                                             textvariable=self.resolution_var, state="readonly")
        self.resolution_combo['values'] = ["1920x1080", "2560x1440", "3840x2160", "1280x720", "1600x900"]
        self.resolution_combo.pack(side=tk.LEFT, padx=5)
        self.resolution_combo.bind('<<ComboboxSelected>>', self._update_estimates)

        # Audio
        audio_frame = ttk.Frame(settings_frame)
        audio_frame.pack(fill=tk.X, pady=2)
        self.audio_check = ttk.Checkbutton(audio_frame, text="Add audio sync beeps",
                                           variable=self.add_audio_var)
        self.audio_check.pack(side=tk.LEFT)
        ffmpeg_available = check_ffmpeg_available()
        if not ffmpeg_available:
            self.audio_check.config(state=tk.DISABLED)
            self.add_audio_var.set(False)
            ttk.Label(audio_frame, text="(requires FFmpeg)", foreground="gray").pack(side=tk.LEFT)

        # Encryption
        encrypt_frame = ttk.Frame(settings_frame)
        encrypt_frame.pack(fill=tk.X, pady=2)
        self.encrypt_check = ttk.Checkbutton(encrypt_frame, text="Encrypt payload (AES-256-GCM)",
                                             variable=self.encrypt_var)
        self.encrypt_check.pack(side=tk.LEFT)
        if not check_crypto_available():
            self.encrypt_check.config(state=tk.DISABLED)
            ttk.Label(encrypt_frame, text="(install cryptography)", foreground="gray").pack(side=tk.LEFT)

        # HPC
        hpc_frame = ttk.LabelFrame(parent, text="HPC Parallel Encoding", padding="5")
        hpc_frame.pack(fill=tk.X, pady=(0, 5))

        hpc_enable_frame = ttk.Frame(hpc_frame)
        hpc_enable_frame.pack(fill=tk.X, pady=2)
        self.hpc_check = ttk.Checkbutton(hpc_enable_frame, text="Enable HPC parallel encoding (multiprocessing)",
                                         variable=self.use_hpc_var, command=self._on_hpc_changed)
        self.hpc_check.pack(side=tk.LEFT)

        hpc_params_frame = ttk.Frame(hpc_frame)
        hpc_params_frame.pack(fill=tk.X, pady=2)
        ttk.Label(hpc_params_frame, text="Workers:").pack(side=tk.LEFT)
        self.workers_spinbox = ttk.Spinbox(hpc_params_frame, from_=1, to=cpu_count() * 2, width=5,
                                           textvariable=self.workers_var)
        self.workers_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(hpc_params_frame, text="Batch size:").pack(side=tk.LEFT, padx=(20, 0))
        self.batch_spinbox = ttk.Spinbox(hpc_params_frame, from_=10, to=500, width=5,
                                         textvariable=self.batch_size_var)
        self.batch_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(hpc_params_frame, text="CRF (quality):").pack(side=tk.LEFT, padx=(20, 0))
        self.crf_spinbox = ttk.Spinbox(hpc_params_frame, from_=0, to=51, width=5,
                                       textvariable=self.crf_var)
        self.crf_spinbox.pack(side=tk.LEFT, padx=5)

        self.hpc_info_label = ttk.Label(hpc_frame, text=f"Available CPU cores: {cpu_count()}", foreground="gray")
        self.hpc_info_label.pack(fill=tk.X)

        # Estimates
        estimates_frame = ttk.LabelFrame(parent, text="Estimates", padding="5")
        estimates_frame.pack(fill=tk.X, pady=(0, 5))
        self.estimates_label = ttk.Label(estimates_frame, text="Add files to see encoding estimates", justify=tk.LEFT)
        self.estimates_label.pack(fill=tk.X)

        # Progress
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(progress_frame, text="Overall:").pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.overall_progress.pack(fill=tk.X, pady=2)

        ttk.Label(progress_frame, text="Current file:").pack(anchor=tk.W)
        self.file_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.file_progress.pack(fill=tk.X, pady=2)

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(fill=tk.X)
        self.stats_label = ttk.Label(progress_frame, text="", foreground="gray")
        self.stats_label.pack(fill=tk.X)

        # Controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(5, 0))
        self.start_btn = ttk.Button(control_frame, text="Start Encoding", command=self._on_start_clicked)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._on_stop_clicked, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    # ── Binary Stream Tab ────────────────────────────────────────────

    def _create_binary_tab(self, parent):
        """Create all widgets for the Binary Stream tab."""
        # File Queue
        queue_frame = ttk.LabelFrame(parent, text="File Queue", padding="5")
        queue_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        list_frame = ttk.Frame(queue_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.bs_file_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=5)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.bs_file_listbox.yview)
        self.bs_file_listbox.config(yscrollcommand=scrollbar.set)
        self.bs_file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(queue_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        self.bs_add_btn = ttk.Button(btn_frame, text="Add Files...", command=self._bs_add_files)
        self.bs_add_btn.pack(side=tk.LEFT, padx=2)
        self.bs_remove_btn = ttk.Button(btn_frame, text="Remove", command=self._bs_remove_selected)
        self.bs_remove_btn.pack(side=tk.LEFT, padx=2)
        self.bs_clear_btn = ttk.Button(btn_frame, text="Clear", command=self._bs_clear_files)
        self.bs_clear_btn.pack(side=tk.LEFT, padx=2)

        self.bs_files_info = ttk.Label(queue_frame, text="No files queued", foreground="gray")
        self.bs_files_info.pack(fill=tk.X, pady=(5, 0))

        # Display Selection
        display_frame = ttk.LabelFrame(parent, text="Display Selection", padding="5")
        display_frame.pack(fill=tk.X, pady=(0, 5))

        disp_row = ttk.Frame(display_frame)
        disp_row.pack(fill=tk.X)

        ttk.Label(disp_row, text="Display:").pack(side=tk.LEFT)
        self.bs_display_combo = ttk.Combobox(disp_row, textvariable=self.bs_display_var,
                                             state="readonly", width=40)
        self.bs_display_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.bs_refresh_displays_btn = ttk.Button(disp_row, text="Refresh", command=self._bs_refresh_displays)
        self.bs_refresh_displays_btn.pack(side=tk.RIGHT)

        # Load display list
        self._bs_refresh_displays()

        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="FPS:").pack(side=tk.LEFT)
        self.bs_fps_spin = ttk.Spinbox(row1, from_=30, to=60, width=5, textvariable=self.bs_fps_var,
                                       values=(30, 60), command=self._bs_update_estimates)
        self.bs_fps_spin.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Repeat:").pack(side=tk.LEFT, padx=(20, 0))
        self.bs_repeat_spin = ttk.Spinbox(row1, from_=1, to=3, width=5, textvariable=self.bs_repeat_var,
                                          command=self._bs_update_estimates)
        self.bs_repeat_spin.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="FEC:").pack(side=tk.LEFT, padx=(20, 0))
        self.bs_fec_combo = ttk.Combobox(row1, textvariable=self.bs_fec_var, state="readonly", width=5,
                                         values=["0%", "5%", "10%"])
        self.bs_fec_combo.pack(side=tk.LEFT, padx=5)
        self.bs_fec_combo.bind('<<ComboboxSelected>>', self._bs_update_estimates)

        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="Calibration (s):").pack(side=tk.LEFT)
        self.bs_cal_spin = ttk.Spinbox(row2, from_=1.0, to=5.0, increment=0.5, width=5,
                                       textvariable=self.bs_calibration_var, format="%.1f")
        self.bs_cal_spin.pack(side=tk.LEFT, padx=5)

        # Estimates
        est_frame = ttk.LabelFrame(parent, text="Estimates", padding="5")
        est_frame.pack(fill=tk.X, pady=(0, 5))
        self.bs_estimates_label = ttk.Label(est_frame, text="Add files to see streaming estimates", justify=tk.LEFT)
        self.bs_estimates_label.pack(fill=tk.X)

        # Progress
        prog_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        prog_frame.pack(fill=tk.X, pady=(0, 5))

        self.bs_progress_bar = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.bs_progress_bar.pack(fill=tk.X, pady=2)

        self.bs_progress_label = ttk.Label(prog_frame, text="Ready")
        self.bs_progress_label.pack(fill=tk.X)
        self.bs_stats_label = ttk.Label(prog_frame, text="", foreground="gray")
        self.bs_stats_label.pack(fill=tk.X)

        # Controls
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, pady=(5, 0))

        self.bs_start_btn = ttk.Button(ctrl_frame, text="Start Streaming", command=self._bs_on_start)
        self.bs_start_btn.pack(side=tk.LEFT, padx=5)
        self.bs_stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self._bs_on_stop, state=tk.DISABLED)
        self.bs_stop_btn.pack(side=tk.LEFT, padx=5)

    # ── Binary Stream: file queue management ─────────────────────────

    def _bs_add_files(self):
        """Add files to the binary stream queue."""
        file_paths = filedialog.askopenfilenames(
            title="Select files to stream",
            filetypes=[("All files", "*.*")]
        )
        if file_paths:
            for path in file_paths:
                if path not in self.bs_file_paths:
                    self.bs_file_paths.append(path)
                    size = os.path.getsize(path)
                    self.bs_file_listbox.insert(tk.END, f"{os.path.basename(path)} ({self._format_size(size)})")
            self._bs_update_file_info()
            self._bs_update_estimates()

    def _bs_remove_selected(self):
        selected = list(self.bs_file_listbox.curselection())
        selected.reverse()
        for idx in selected:
            self.bs_file_listbox.delete(idx)
            del self.bs_file_paths[idx]
        self._bs_update_file_info()
        self._bs_update_estimates()

    def _bs_clear_files(self):
        self.bs_file_listbox.delete(0, tk.END)
        self.bs_file_paths.clear()
        self._bs_update_file_info()
        self._bs_update_estimates()

    def _bs_update_file_info(self):
        count = len(self.bs_file_paths)
        self.bs_total_size = sum(os.path.getsize(p) for p in self.bs_file_paths) if count else 0
        if count == 0:
            self.bs_files_info.config(text="No files queued")
        else:
            self.bs_files_info.config(
                text=f"{count} file(s), total: {self._format_size(self.bs_total_size)}")

    # ── Binary Stream: display management ────────────────────────────

    def _bs_refresh_displays(self):
        """Refresh the display list from renderer."""
        try:
            from .renderer import list_displays
            displays = list_displays()
        except Exception:
            displays = [{'index': 0, 'description': 'Default (1920x1080)'}]

        self._bs_displays = displays
        descriptions = [d['description'] for d in displays]
        self.bs_display_combo['values'] = descriptions

        # Default to second display if available
        if len(displays) > 1:
            self.bs_display_combo.current(1)
        elif displays:
            self.bs_display_combo.current(0)

    def _bs_get_display_index(self) -> int:
        """Get selected display index."""
        idx = self.bs_display_combo.current()
        if idx >= 0 and idx < len(self._bs_displays):
            return self._bs_displays[idx]['index']
        return 1

    # ── Binary Stream: settings / estimates ──────────────────────────

    def _bs_get_fec_ratio(self) -> float:
        fec_str = self.bs_fec_var.get()
        return int(fec_str.replace('%', '')) / 100.0

    def _bs_update_estimates(self, event=None):
        """Update binary stream estimates."""
        if not self.bs_file_paths:
            self.bs_estimates_label.config(text="Add files to see streaming estimates")
            return

        fec_ratio = self._bs_get_fec_ratio()
        fps = self.bs_fps_var.get()
        repeat = self.bs_repeat_var.get()
        payload_cap = calculate_binary_payload_capacity(fec_ratio)

        total_blocks = 0
        for fpath in self.bs_file_paths:
            fsize = os.path.getsize(fpath)
            # Rough block count per file (metadata overhead is small, ignore for estimate)
            total_blocks += max(1, (fsize + payload_cap - 1) // payload_cap)

        total_frames = total_blocks * repeat
        calibration_frames = int(self.bs_calibration_var.get() * fps) * len(self.bs_file_paths)
        total_frames += calibration_frames
        est_time = total_frames / fps if fps > 0 else 0
        throughput = payload_cap * fps / repeat

        self.bs_estimates_label.config(text=(
            f"Payload/frame: {payload_cap:,} bytes\n"
            f"Total blocks: {total_blocks:,} ({len(self.bs_file_paths)} file(s))\n"
            f"Total frames: {total_frames:,} (with {repeat}x repeat)\n"
            f"Est. time: {format_duration(est_time)}\n"
            f"Net throughput: {throughput / (1024 * 1024):.1f} MB/s"
        ))

    # ── Binary Stream: start / stop ──────────────────────────────────

    def _bs_on_start(self):
        """Handle Start Streaming button."""
        if not self.bs_file_paths:
            messagebox.showwarning("No Files", "Please add files to the queue.")
            return

        self.is_streaming = True
        self._bs_update_controls()

        if self.on_stream_start:
            self.on_stream_start(self._bs_get_settings())

    def _bs_on_stop(self):
        """Handle Stop button for binary stream."""
        if messagebox.askyesno("Confirm Stop", "Stop streaming?"):
            self.is_streaming = False
            self._bs_update_controls()
            if self.on_stream_stop:
                self.on_stream_stop()

    def _bs_get_settings(self) -> dict:
        """Get binary stream settings."""
        return {
            'file_paths': self.bs_file_paths.copy(),
            'display_index': self._bs_get_display_index(),
            'fps': self.bs_fps_var.get(),
            'repeat_count': self.bs_repeat_var.get(),
            'fec_ratio': self._bs_get_fec_ratio(),
            'calibration_secs': self.bs_calibration_var.get(),
            'passes': 3,  # loop data 3 times so receiver catches all blocks
        }

    def _bs_update_controls(self):
        """Update binary stream button states."""
        if self.is_streaming:
            self.bs_start_btn.config(state=tk.DISABLED)
            self.bs_stop_btn.config(state=tk.NORMAL)
            self.bs_add_btn.config(state=tk.DISABLED)
            self.bs_remove_btn.config(state=tk.DISABLED)
            self.bs_clear_btn.config(state=tk.DISABLED)
        else:
            self.bs_start_btn.config(state=tk.NORMAL)
            self.bs_stop_btn.config(state=tk.DISABLED)
            self.bs_add_btn.config(state=tk.NORMAL)
            self.bs_remove_btn.config(state=tk.NORMAL)
            self.bs_clear_btn.config(state=tk.NORMAL)

    # ── Binary Stream: progress updates from main.py ─────────────────

    def update_stream_progress(self, block_idx, total_blocks, bytes_sent, elapsed, current_file):
        """Update binary stream progress display (called from main thread)."""
        if total_blocks > 0:
            pct = (block_idx + 1) / total_blocks * 100
            self.bs_progress_bar['value'] = pct

        self.bs_progress_label.config(
            text=f"File: {current_file}  |  Block {block_idx + 1:,}/{total_blocks:,}"
        )

        mbps = bytes_sent / elapsed / (1024 * 1024) if elapsed > 0 else 0
        eta = (elapsed / (block_idx + 1)) * (total_blocks - block_idx - 1) if block_idx > 0 else 0
        self.bs_stats_label.config(
            text=f"{mbps:.1f} MB/s  |  Elapsed: {format_duration(elapsed)}  |  ETA: {format_duration(eta)}"
        )

    def show_stream_complete(self, success: bool, message: str = ""):
        """Show binary stream completion."""
        self.is_streaming = False
        self._bs_update_controls()

        if success:
            self.bs_progress_bar['value'] = 100
            self.bs_progress_label.config(text="Streaming complete!")
            self.bs_stats_label.config(text="")
            messagebox.showinfo("Complete", message or "All files streamed successfully!")
        else:
            self.bs_progress_label.config(text="Streaming stopped")
            if message:
                messagebox.showerror("Error", message)

    # ── Video Encode: existing methods (unchanged) ───────────────────

    def _add_files(self):
        """Add files to the encode list."""
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
        selected = list(self.file_listbox.curselection())
        selected.reverse()
        for idx in selected:
            self.file_listbox.delete(idx)
            del self.file_paths[idx]
        self._update_file_info()
        self._update_estimates()

    def _clear_files(self):
        self.file_listbox.delete(0, tk.END)
        self.file_paths.clear()
        self._update_file_info()
        self._update_estimates()

    def _update_file_info(self):
        count = len(self.file_paths)
        self.total_size = sum(os.path.getsize(p) for p in self.file_paths)
        if count == 0:
            self.files_info_label.config(text="No files selected")
        else:
            self.files_info_label.config(
                text=f"{count} file(s) selected, total: {self._format_size(self.total_size)}")

    def _browse_output(self):
        directory = filedialog.askdirectory(title="Select output directory", initialdir=self.output_dir)
        if directory:
            self.output_dir = directory
            display = directory
            if len(display) > 50:
                display = "..." + display[-47:]
            self.output_label.config(text=display)

    def _format_size(self, size: int) -> str:
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"

    def _on_enhanced_changed(self):
        if self.use_enhanced_var.get():
            self.std_profiles_frame.pack_forget()
            self.enhanced_profiles_frame.pack(side=tk.LEFT)
            self.profile_var.set("enhanced_conservative")
        else:
            self.enhanced_profiles_frame.pack_forget()
            self.std_profiles_frame.pack(side=tk.LEFT)
            self.profile_var.set("conservative")
        self._update_estimates()

    def _on_hpc_changed(self):
        state = tk.NORMAL if self.use_hpc_var.get() else tk.DISABLED
        self.workers_spinbox.config(state=state)
        self.batch_spinbox.config(state=state)

    def _get_profile(self):
        profile_name = self.profile_var.get()
        if profile_name == "conservative":
            return PROFILE_CONSERVATIVE
        elif profile_name == "aggressive":
            return PROFILE_AGGRESSIVE
        elif profile_name == "ultra":
            return PROFILE_ULTRA
        elif profile_name == "enhanced_conservative":
            return ENHANCED_PROFILE_CONSERVATIVE
        elif profile_name == "enhanced_standard":
            return ENHANCED_PROFILE_STANDARD
        else:
            return PROFILE_STANDARD

    def _update_estimates(self, event=None):
        if not hasattr(self, 'estimates_label'):
            return
        if not self.file_paths:
            self.estimates_label.config(text="Add files to see encoding estimates")
            return

        profile = self._get_profile()
        fps = self.fps_var.get()
        repeat = self.repeat_var.get()

        res_str = self.resolution_var.get()
        try:
            res_width, res_height = map(int, res_str.split('x'))
        except:
            res_width, res_height = 1920, 1080

        from shared import calculate_payload_capacity, SYNC_FRAME_COUNT, END_FRAME_COUNT

        if self.use_enhanced_var.get():
            payload_per_block = profile.payload_bytes - 28 - 2 - 32
            encoding_type = "Enhanced (8 luma + 4 color)"
        else:
            payload_per_block = calculate_payload_capacity(profile, 0.10)
            encoding_type = "Standard (4 grayscale)"

        total_blocks = (self.total_size + payload_per_block - 1) // payload_per_block
        total_data_frames = total_blocks * repeat
        calibration_frames = 5 if self.use_enhanced_var.get() else 0
        total_frames = total_data_frames + calibration_frames + SYNC_FRAME_COUNT + END_FRAME_COUNT

        video_duration = total_frames / fps
        bytes_per_second = payload_per_block * fps / repeat

        hpc_info = ""
        if self.use_hpc_var.get():
            hpc_info = f"HPC: {self.workers_var.get()} workers, batch {self.batch_size_var.get()}\n"

        estimates_text = (
            f"Encoding: {encoding_type}\n"
            f"Profile: {profile.name} ({profile.cell_size}x{profile.cell_size} cells)\n"
            f"Resolution: {res_width}x{res_height} @ {fps} FPS\n"
            f"Payload per block: {payload_per_block} bytes\n"
            f"Total blocks: {total_blocks:,} ({len(self.file_paths)} file(s))\n"
            f"Total frames: {total_frames:,} (with {repeat}x repeat)\n"
            f"Video duration: {format_duration(video_duration)}\n"
            f"{hpc_info}"
            f"Effective data rate: {bytes_per_second / 1024:.1f} KB/s"
        )
        self.estimates_label.config(text=estimates_text)

    def _on_start_clicked(self):
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
            'add_audio': self.add_audio_var.get(),
            'use_enhanced': self.use_enhanced_var.get(),
            'use_hpc': self.use_hpc_var.get(),
            'workers': self.workers_var.get(),
            'batch_size': self.batch_size_var.get(),
            'crf': self.crf_var.get(),
        }

    def _on_stop_clicked(self):
        if messagebox.askyesno("Confirm Stop", "Stop encoding?"):
            self.is_encoding = False
            self._update_control_buttons()
            if self.on_stop:
                self.on_stop()

    def _update_control_buttons(self):
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

    def update_progress(self, current_file_index, total_files, current_file_name,
                        current_frame, total_frames, elapsed_time):
        self.current_file_index = current_file_index

        if total_files > 0:
            overall = ((current_file_index + current_frame / max(total_frames, 1)) / total_files) * 100
            self.overall_progress['value'] = overall

        if total_frames > 0:
            file_progress = (current_frame / total_frames) * 100
            self.file_progress['value'] = file_progress

        self.progress_label.config(
            text=f"File {current_file_index + 1}/{total_files}: {current_file_name}\n"
                 f"Frame {current_frame:,} / {total_frames:,}")

        if elapsed_time > 0 and current_frame > 0:
            fps = current_frame / elapsed_time
            remaining_frames = total_frames - current_frame
            remaining_time = remaining_frames / fps if fps > 0 else 0
            self.stats_label.config(
                text=f"Encoding: {fps:.1f} FPS | "
                     f"Elapsed: {format_duration(elapsed_time)} | "
                     f"Remaining: {format_duration(remaining_time)}")

    def show_file_complete(self, file_path: str, output_path: str):
        filename = os.path.basename(file_path)
        output_name = os.path.basename(output_path)
        self.progress_label.config(text=f"Completed: {filename} -> {output_name}")

    def show_complete(self, success: bool, message: str = ""):
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
        self.root.mainloop()

    def close(self):
        self.root.destroy()


def run_sender_ui():
    """Run the sender UI as standalone."""
    ui = SenderUI()
    ui.run()


if __name__ == "__main__":
    run_sender_ui()
