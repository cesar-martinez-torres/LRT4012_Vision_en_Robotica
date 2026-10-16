"""Display the camera parameters stored in an OpenCV calibration NPZ file."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np


APP_TITLE = "Camera Calibration Viewer"


def default_directory() -> Path:
    images = Path(__file__).resolve().parent / "imagenes"
    return images if images.is_dir() else Path(__file__).resolve().parent


def scalar(data, key: str, default="N/A"):
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.size == 1 else value.tolist()


def load_parameters(filename: Path) -> dict:
    with np.load(filename, allow_pickle=False) as data:
        required = {"K", "distortion", "image_size"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("Missing fields: " + ", ".join(sorted(missing)))

        K = np.asarray(data["K"], dtype=float)
        distortion = np.asarray(data["distortion"], dtype=float).ravel()
        image_size = np.asarray(data["image_size"], dtype=int).ravel()
        if K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3); received {K.shape}.")
        if image_size.size != 2 or np.any(image_size <= 0):
            raise ValueError("image_size must contain a positive width and height.")

        width, height = (int(image_size[0]), int(image_size[1]))
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if fx <= 0 or fy <= 0:
            raise ValueError("The focal lengths fx and fy must be positive.")

        return {
            "filename": filename.resolve(),
            "keys": list(data.files),
            "K": K,
            "distortion": distortion,
            "image_size": (width, height),
            "fx": fx,
            "fy": fy,
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "skew": float(K[0, 1]),
            "fov_x": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
            "fov_y": math.degrees(2.0 * math.atan(height / (2.0 * fy))),
            "camera": scalar(data, "camera"),
            "rectified": scalar(data, "rectified"),
            "source": scalar(data, "source"),
            "source_image_size": scalar(data, "source_image_size"),
            "board_shape": scalar(data, "board_shape"),
            "square_size": scalar(data, "square_size"),
            "valid_views": scalar(data, "valid_views"),
            "rms": scalar(data, "rms"),
            "mean_reprojection_error": scalar(data, "mean_reprojection_error"),
        }


def number(value, digits=4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def format_report(parameters: dict) -> str:
    K = parameters["K"]
    distortion = parameters["distortion"]
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    coefficients = "\n".join(
        f"  {names[index] if index < len(names) else f'd{index + 1}'} = {value:.8f}"
        for index, value in enumerate(distortion)
    ) or "  No coefficients"
    width, height = parameters["image_size"]

    optional = []
    fields = [
        ("Camera", "camera"),
        ("Rectified", "rectified"),
        ("Source", "source"),
        ("Source image size", "source_image_size"),
        ("Checkerboard inner corners", "board_shape"),
        ("Square size", "square_size"),
        ("Valid views", "valid_views"),
        ("Calibration RMS", "rms"),
        ("Mean reprojection error [px]", "mean_reprojection_error"),
    ]
    for label, key in fields:
        if parameters[key] != "N/A":
            optional.append(f"{label}: {number(parameters[key], 6)}")

    return "\n".join([
        f"File: {parameters['filename']}",
        f"Image size: {width} x {height} pixels",
        *optional,
        "",
        "Intrinsic matrix K",
        np.array2string(K, precision=6, suppress_small=True),
        "",
        "Intrinsic parameters",
        f"  fx = {parameters['fx']:.6f} px",
        f"  fy = {parameters['fy']:.6f} px",
        f"  cx = {parameters['cx']:.6f} px",
        f"  cy = {parameters['cy']:.6f} px",
        f"  skew = {parameters['skew']:.6f}",
        f"  horizontal field of view = {parameters['fov_x']:.3f} deg",
        f"  vertical field of view = {parameters['fov_y']:.3f} deg",
        "",
        "Distortion coefficients",
        coefficients,
        "",
        "Stored fields",
        "  " + ", ".join(parameters["keys"]),
    ])


class CalibrationViewer:
    def __init__(self, root: tk.Tk, initial_file: Path | None = None):
        self.root = root
        root.title(APP_TITLE)
        root.minsize(760, 620)
        root.geometry("900x720")

        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Path.TLabel", foreground="#555b63")

        header = ttk.Frame(root, padding=(24, 20, 24, 12))
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="OPEN NPZ", command=self.open_file).pack(side="right")

        self.path_label = ttk.Label(root, text="Select a calibration file.", style="Path.TLabel")
        self.path_label.pack(fill="x", padx=24, pady=(0, 10))

        self.report = ScrolledText(
            root, wrap="none", font=("Consolas", 11), padx=14, pady=14,
            background="#fbfbfb", relief="solid", borderwidth=1
        )
        self.report.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self.report.configure(state="disabled")

        if initial_file is not None:
            self.show_file(initial_file)
        else:
            root.after(100, self.open_file)

    def open_file(self):
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Open camera calibration",
            initialdir=default_directory(),
            filetypes=[("NumPy calibration", "*.npz"), ("All files", "*.*")],
        )
        if selected:
            self.show_file(Path(selected))

    def show_file(self, filename: Path):
        try:
            text = format_report(load_parameters(filename))
        except Exception as error:
            messagebox.showerror(APP_TITLE, str(error), parent=self.root)
            return
        self.path_label.configure(text=str(filename.resolve()))
        self.root.title(f"{APP_TITLE} - {filename.name}")
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="Calibration NPZ file")
    parser.add_argument("--text", action="store_true", help="Print the report without opening a window")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.text:
        if args.file is None:
            raise SystemExit("--text requires a calibration NPZ file")
        print(format_report(load_parameters(args.file)))
        return 0

    root = tk.Tk()
    CalibrationViewer(root, args.file)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
