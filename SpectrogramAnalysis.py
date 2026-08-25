"""Interactive spectral analysis for signals stored in an HDF5 file.
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import h5py
import matplotlib
import numpy as np

# Use a Tk-compatible Matplotlib backend for the desktop interface.
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


EPSILON = np.finfo(np.float64).tiny
MAX_SPECTROGRAM_COLUMNS = 4000
# Used only when an HDF5 file does not contain a usable time dataset.
FALLBACK_SAMPLE_RATE_HZ = 1_000_000.0
# Conversion factors for timestamps stored in different units.
TIME_FACTORS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


class SpectrogramAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Configure the main application window and its initial state.
        self.title("Plasma Diagnostic Spectral Analysis")
        self.geometry("1450x900")
        self.minsize(1100, 700)

        self.h5_file = None
        self.current_dataset_path = ""
        self.data = np.array([], dtype=np.float64)
        self.time_seconds = np.array([], dtype=np.float64)
        self.spectrogram_freqs = np.array([], dtype=np.float64)
        self.spectrogram_times = np.array([], dtype=np.float64)
        self.spectrogram_data = np.empty((0, 0), dtype=np.float64)
        self.slice_marker = None

        # Tk variables connect the analysis settings to the input controls.
        self.n_fft_var = tk.IntVar(value=1024)
        self.time_min_var = tk.DoubleVar(value=0.0)
        self.time_max_var = tk.DoubleVar(value=1.0)
        self.freq_min_var = tk.DoubleVar(value=0.0)
        self.freq_max_var = tk.DoubleVar(value=500_000.0)
        self.power_min_var = tk.DoubleVar(value=-120.0)
        self.power_max_var = tk.DoubleVar(value=20.0)
        self.raw_units_var = tk.StringVar(value="Amplitude")
        self.time_unit_var = tk.StringVar(value="s")
        self.slice_index_var = tk.IntVar(value=0)
        self.log_freq_var = tk.BooleanVar(value=False)
        self.log_power_var = tk.BooleanVar(value=True)

        # Build the interface after state exists, then handle window closing.
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # Create the main layout, file controls, and HDF5 navigation tree.
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 8))
        ttk.Button(top, text="Open HDF5 File", command=self.open_file).pack(side="left")
        self.file_label = ttk.Label(top, text="No file selected")
        self.file_label.pack(side="left", padx=10)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(left, text="HDF5 structure").pack(anchor="w", pady=(0, 5))
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, show="tree")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewOpen>>", self.on_tree_open)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Create the plot area: raw signal, spectrogram, colorbar, and slice.
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)
        self.channel_label = ttk.Label(right, text="Selected channel: none")
        self.channel_label.pack(anchor="w", fill="x", pady=(0, 8))

        self.figure = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        grid = self.figure.add_gridspec(
            2, 3, width_ratios=(4.5, 0.18, 2.4), height_ratios=(1.2, 3.0),
            hspace=0.08, wspace=0.12,
        )
        self.raw_ax = self.figure.add_subplot(grid[0, 0])
        self.spec_ax = self.figure.add_subplot(grid[1, 0], sharex=self.raw_ax)
        self.cax = self.figure.add_subplot(grid[1, 1])
        self.slice_ax = self.figure.add_subplot(grid[1, 2], sharey=self.spec_ax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Add numeric, unit, time-axis, and display controls below the plots.
        controls = ttk.LabelFrame(right, text="Analysis controls")
        controls.pack(fill="x", pady=(8, 0))
        grid_controls = ttk.Frame(controls)
        grid_controls.pack(fill="x", padx=10, pady=8)

        fields = (
            ("Time min (s)", self.time_min_var),
            ("Time max (s)", self.time_max_var),
            ("Frequency min (Hz)", self.freq_min_var),
            ("Frequency max (Hz)", self.freq_max_var),
            ("Power min", self.power_min_var),
            ("Power max", self.power_max_var),
            ("N FFT", self.n_fft_var),
            ("Raw signal units", self.raw_units_var),
        )
        for index, (label, variable) in enumerate(fields):
            row, col = divmod(index, 4)
            base = col * 2
            ttk.Label(grid_controls, text=label + ":").grid(
                row=row, column=base, sticky="w", padx=(0 if col == 0 else 18, 6), pady=3
            )
            ttk.Entry(grid_controls, textvariable=variable, width=12).grid(
                row=row, column=base + 1, sticky="w", pady=3
            )

        options = ttk.Frame(controls)
        options.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(options, text="Stored time unit:").pack(side="left")
        ttk.Combobox(
            options, textvariable=self.time_unit_var, values=tuple(TIME_FACTORS),
            width=5, state="readonly",
        ).pack(side="left", padx=(5, 18))
        ttk.Checkbutton(options, text="Log frequency axis", variable=self.log_freq_var).pack(side="left")
        ttk.Checkbutton(options, text="Log power (dB/Hz)", variable=self.log_power_var).pack(side="left", padx=18)
        ttk.Button(options, text="Apply", command=self.apply_settings).pack(side="left")
        ttk.Button(options, text="Clear", command=self.clear_plot).pack(side="left", padx=8)

        slider_row = ttk.Frame(controls)
        slider_row.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(slider_row, text="Selected spectrum time:").pack(side="left")
        self.slice_time_label = ttk.Label(slider_row, text="—")
        self.slice_time_label.pack(side="left")

        # Start with empty plots until a dataset is selected.
        self.clear_plot()

    def on_close(self):
        # Close the HDF5 handle before destroying the Tk window.
        if self.h5_file is not None:
            self.h5_file.close()
        self.destroy()

    def open_file(self):
        # Open a file read-only and reset the tree and plots for the new file.
        path = filedialog.askopenfilename(
            title="Select an HDF5 file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5 *.hdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            new_file = h5py.File(path, "r")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Open HDF5", f"Unable to open file:\n{exc}")
            return
        if self.h5_file is not None:
            self.h5_file.close()
        self.h5_file = new_file
        self.file_label.config(text=os.path.basename(path))
        self.tree.delete(*self.tree.get_children())
        self._insert_group_children(self.h5_file, "")
        self.current_dataset_path = ""
        self.data = np.array([], dtype=np.float64)
        self.time_seconds = np.array([], dtype=np.float64)
        self.channel_label.config(text="Selected channel: none")
        self.clear_plot()

    def _insert_group_children(self, group, parent):
        """Insert one metadata level; groups are populated lazily when expanded."""
        # Populate only one level so opening a large file stays responsive.
        try:
            names = sorted(group.keys())
        except (KeyError, OSError) as exc:
            messagebox.showerror("HDF5 structure", str(exc))
            return
        for name in names:
            obj = group.get(name, getlink=False)
            node_path = f"{parent}/{name}" if parent else name
            node = self.tree.insert(parent, "end", iid=node_path, text=name)
            if isinstance(obj, h5py.Group):
                self.tree.insert(node, "end", iid=node_path + "/__placeholder__", text="Loading…")

    def on_tree_open(self, _event):
        # Replace a group's placeholder with its real children when expanded.
        selected = self.tree.focus()
        if not selected or self.h5_file is None:
            return
        children = self.tree.get_children(selected)
        if len(children) != 1 or not children[0].endswith("/__placeholder__"):
            return
        self.tree.delete(children[0])
        try:
            group = self.h5_file[self._tree_path(selected)]
            self._insert_group_children(group, selected)
        except (KeyError, OSError) as exc:
            messagebox.showerror("HDF5 structure", str(exc))

    def _tree_path(self, item):
        # Convert a Treeview item path into an HDF5 absolute path.
        return "/" + item.strip("/")

    def on_tree_select(self, _event):
        # Load datasets selected in the tree; identify groups without loading data.
        selected = self.tree.selection()
        if not selected or self.h5_file is None:
            return
        path = self._tree_path(selected[0])
        try:
            obj = self.h5_file[path]
        except (KeyError, OSError):
            return
        if isinstance(obj, h5py.Dataset):
            self.load_dataset(path)
        else:
            self.channel_label.config(text=f"Selected channel: {path} (group)")

    def load_dataset(self, path):
        # Validate and load a one-dimensional numeric signal from HDF5.
        dataset = self.h5_file[path]
        if dataset.ndim == 0:
            messagebox.showwarning("Dataset", "A scalar dataset cannot be spectrally analyzed.")
            return
        if dataset.ndim > 2 or (dataset.ndim == 2 and min(dataset.shape) != 1):
            messagebox.showwarning(
                "Dataset", f"Select a one-dimensional signal. This dataset has shape {dataset.shape}."
            )
            return
        try:
            values = np.asarray(dataset[...]).reshape(-1)
            if np.iscomplexobj(values):
                messagebox.showwarning("Dataset", "Complex signals are not supported by this one-sided PSD view.")
                return
            values = values.astype(np.float64, copy=False)
            times, time_source = self._resolve_time_array(path, values.size)
        except (OSError, ValueError, TypeError, MemoryError, tk.TclError) as exc:
            messagebox.showerror("Dataset", f"Unable to load dataset:\n{exc}")
            return

        count = min(values.size, times.size)
        # Align signal and timestamps, discard invalid samples, and sort in time.
        values, times = values[:count], times[:count]
        finite = np.isfinite(values) & np.isfinite(times)
        values, times = values[finite], times[finite]
        if values.size < 2:
            messagebox.showwarning("Dataset", "At least two finite samples are required.")
            return

        factor = TIME_FACTORS[self.time_unit_var.get()]
        times = times * factor if time_source != "generated" else times
        order = np.argsort(times, kind="stable")
        values, times = values[order], times[order]
        unique = np.concatenate(([True], np.diff(times) > 0))
        values, times = values[unique], times[unique]
        if values.size < 2:
            messagebox.showwarning("Dataset", "Timestamps must contain at least two distinct values.")
            return

        self.current_dataset_path = path
        self.data, self.time_seconds = values, times
        self.time_min_var.set(times[0])
        self.time_max_var.set(times[-1])
        self.channel_label.config(
            text=f"Selected channel: {path} — {values.size:,} samples; time: {time_source}"
        )
        self.apply_settings()

    def _resolve_time_array(self, dataset_path, length):
        # Search nearby conventional time-dataset names before using the fallback rate.
        group_path, dataset_name = dataset_path.rsplit("/", 1)
        group_path = group_path or "/"
        candidates = ["time", "Time", f"time_{dataset_name}"]
        chord = re.search(r"([tv]\d{1,2})", dataset_name.lower())
        if chord:
            candidates.insert(0, f"time_{chord.group(1)}")
        system = re.search(r"(core|tangential)", dataset_name.lower())
        if system:
            candidates.insert(0, f"time_{system.group(1)}")
        for name in dict.fromkeys(candidates):
            candidate = f"/{name}" if group_path == "/" else f"{group_path}/{name}"
            if candidate == dataset_path or candidate not in self.h5_file:
                continue
            obj = self.h5_file[candidate]
            if isinstance(obj, h5py.Dataset) and obj.size:
                return np.asarray(obj[...], dtype=np.float64).reshape(-1)[:length], candidate
        return np.arange(length, dtype=np.float64) / FALLBACK_SAMPLE_RATE_HZ, "generated"

    def apply_settings(self):
        # Validate settings, derive the sampling interval, and refresh all plots.
        if self.data.size < 2:
            return
        try:
            n_fft = int(self.n_fft_var.get())
            tmin, tmax = float(self.time_min_var.get()), float(self.time_max_var.get())
            fmin, fmax = float(self.freq_min_var.get()), float(self.freq_max_var.get())
            pmin, pmax = float(self.power_min_var.get()), float(self.power_max_var.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning("Settings", "All numerical controls must contain valid numbers.")
            return
        dt = float(np.median(np.diff(self.time_seconds)))
        if not np.isfinite(dt) or dt <= 0:
            messagebox.showwarning("Settings", "Unable to determine a positive sample interval.")
            return
        nyquist = 0.5 / dt
        if not np.isfinite(tmin + tmax) or tmax <= tmin:
            messagebox.showwarning("Settings", "Time maximum must exceed time minimum.")
            return
        n_fft = min(max(n_fft, 2), self.data.size)
        if fmin < 0 or fmax <= fmin or fmin >= nyquist:
            messagebox.showwarning("Settings", f"Frequency limits must overlap 0–{nyquist:g} Hz.")
            return
        fmax = min(fmax, nyquist)
        if self.log_freq_var.get() and fmin <= 0:
            fmin = max(1.0 / (n_fft * dt), np.finfo(float).eps)
            self.freq_min_var.set(fmin)
        if not np.isfinite(pmin + pmax) or pmax <= pmin:
            messagebox.showwarning("Settings", "Power maximum must exceed power minimum.")
            return
        self._update_plots(n_fft, fmin, fmax, pmin, pmax, tmin, tmax, dt)

    def _calculate_psd(self, n_fft, dt):
        # Calculate overlapping, windowed one-sided PSD columns for the signal.
        hop = max(1, n_fft // 2)
        starts = np.arange(0, self.data.size - n_fft + 1, hop, dtype=np.int64)
        if starts.size == 0:
            starts = np.array([0], dtype=np.int64)
        if starts.size > MAX_SPECTROGRAM_COLUMNS:
            starts = starts[np.linspace(0, starts.size - 1, MAX_SPECTROGRAM_COLUMNS, dtype=int)]
        window = np.hanning(n_fft) if n_fft > 2 else np.ones(n_fft)
        scale = (1.0 / dt) * np.sum(window**2)
        psd = np.empty((n_fft // 2 + 1, starts.size), dtype=np.float64)
        for column, start in enumerate(starts):
            # Center and pad each segment so every FFT has the same length.
            segment = self.data[start:start + n_fft]
            if segment.size < n_fft:
                segment = np.pad(segment, (0, n_fft - segment.size))
            segment = segment - np.mean(segment)
            power = np.abs(np.fft.rfft(segment * window)) ** 2 / scale
            if power.size > 2:
                # Double non-edge bins to represent the one-sided spectrum.
                power[1:-1] *= 2.0
            psd[:, column] = power
        frequencies = np.fft.rfftfreq(n_fft, d=dt)
        centers = np.minimum(starts + n_fft // 2, self.time_seconds.size - 1)
        return frequencies, self.time_seconds[centers], psd

    def _power_label(self):
        # Format the PSD label using either dB or the selected linear units.
        if self.log_power_var.get():
            return "PSD (dB/Hz)"
        units = self.raw_units_var.get().strip() or "unit"
        return f"PSD ({units}²/Hz)"

    def _update_plots(self, n_fft, fmin, fmax, pmin, pmax, tmin, tmax, dt):
        # Compute the spectrogram, apply frequency limits, and draw the main views.
        frequencies, times, power = self._calculate_psd(n_fft, dt)
        mask = (frequencies >= fmin) & (frequencies <= fmax)
        if not np.any(mask):
            messagebox.showwarning("Settings", "The selected range contains no FFT frequency bins.")
            return
        frequencies, power = frequencies[mask], power[mask, :]
        if self.log_power_var.get():
            power = 10.0 * np.log10(np.maximum(power, EPSILON))

        self.spectrogram_freqs = frequencies
        self.spectrogram_times = times
        self.spectrogram_data = power
        self.slice_index_var.set(0)
        # Initialize the selected-slice display before the first slice is drawn.
        self.slice_time_label.config(text="—")

        # Clear stale artists before rendering the newly selected dataset/settings.
        self.raw_ax.clear()
        self.spec_ax.clear()
        self.slice_ax.clear()
        self.cax.clear()
        self.cax.set_axis_on()
        self.slice_marker = None
        self.raw_ax.plot(self.time_seconds, self.data, color="tab:blue", linewidth=0.8)
        self.raw_ax.set(title="Time-series signal", ylabel="Amplitude")
        self.raw_ax.grid(True, alpha=0.2)
        self.raw_ax.tick_params(labelbottom=False)
        mesh = self.spec_ax.pcolormesh(times, frequencies, power, shading="auto", cmap="viridis", vmin=pmin, vmax=pmax)
        self.spec_ax.set(title="Auto-power spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
        self.spec_ax.set_yscale("log" if self.log_freq_var.get() else "linear")
        self.spec_ax.set_ylim(fmin, fmax)
        limits = (tmin, tmax)
        self.raw_ax.set_xlim(limits)
        self.spec_ax.set_xlim(limits)
        colorbar = self.figure.colorbar(mesh, cax=self.cax)
        colorbar.set_label(self._power_label())
        self._update_slice(0, draw=False)
        self.canvas.draw_idle()

    def _update_slice(self, index, draw=True):
        # Draw the PSD column corresponding to the selected spectrogram time.
        if self.spectrogram_data.size == 0:
            return
        index = int(np.clip(index, 0, self.spectrogram_data.shape[1] - 1))
        self.slice_index_var.set(index)
        self.slice_ax.clear()
        self.slice_ax.plot(self.spectrogram_data[:, index], self.spectrogram_freqs, color="tab:orange")
        self.slice_ax.set(
            title=f"Power spectrum at t={self.spectrogram_times[index]:.6g} s",
            xlabel=self._power_label(),
            ylabel="Frequency (Hz)",
        )
        self.slice_ax.set_yscale("log" if self.log_freq_var.get() else "linear")
        self.slice_ax.set_ylim(self.spec_ax.get_ylim())
        self.slice_ax.set_xlim(self.power_min_var.get(), self.power_max_var.get())
        self.slice_ax.yaxis.tick_right()
        self.slice_ax.yaxis.set_label_position("right")
        self.slice_ax.grid(True, alpha=0.25)
        self.slice_time_label.config(text=f"{self.spectrogram_times[index]:.6g} s")
        if self.slice_marker is not None:
            self.slice_marker.remove()
        self.slice_marker = self.spec_ax.axvline(self.spectrogram_times[index], color="white", alpha=0.8)
        if draw:
            self.canvas.draw_idle()

    def on_click(self, event):
        # Map a click in the spectrogram to the nearest available time slice.
        if event.inaxes is not self.spec_ax or event.xdata is None or self.spectrogram_times.size == 0:
            return
        index = int(np.argmin(np.abs(self.spectrogram_times - event.xdata)))
        self._update_slice(index)

    def clear_plot(self):
        # Reset stored spectrogram data and restore the initial empty-plot labels.
        self.spectrogram_freqs = np.array([], dtype=np.float64)
        self.spectrogram_times = np.array([], dtype=np.float64)
        self.spectrogram_data = np.empty((0, 0), dtype=np.float64)
        self.slice_marker = None
        for axis in (self.raw_ax, self.spec_ax, self.slice_ax, self.cax):
            axis.clear()
        self.raw_ax.set(title="Time-series signal", ylabel="Amplitude")
        self.spec_ax.set(title="Auto-power spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
        self.slice_ax.set(title="Power spectrum", xlabel="PSD", ylabel="Frequency (Hz)")
        self.slice_ax.yaxis.tick_right()
        self.slice_ax.yaxis.set_label_position("right")
        self.cax.set_axis_off()
        # Reset the selected-slice state because no spectrogram is displayed.
        self.slice_index_var.set(0)
        self.slice_time_label.config(text="—")
        self.canvas.draw_idle()


if __name__ == "__main__":
    SpectrogramAnalysisApp().mainloop()
