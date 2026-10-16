"""Camera calibration utility for the visual odometry activity.

Run without arguments to open the graphical interface:

    python calibrate.py

The checkerboard dimensions are the number of inner corners, not squares.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


APP_TITLE = "Camera calibration"
SETUP_SIZE = (1180, 720)
CAPTURE_SIZE = (1280, 720)


@dataclass
class CalibrationConfig:
    camera_index: int = 0
    columns: int = 9
    rows: int = 6
    target_views: int = 15
    square_size: float = 1.0

    @property
    def board_shape(self) -> tuple[int, int]:
        return self.columns, self.rows


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("VISION_ROBOTICA_REPO")
    if configured:
        roots.append(Path(configured).expanduser())
    script = Path(__file__).resolve()
    current = Path.cwd().resolve()
    roots.extend([script.parent, *script.parents, current, *current.parents])
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def course_root() -> Path:
    for root in candidate_roots():
        if (root / "imagenes").is_dir() or (root / "requirements.txt").is_file():
            return root
    return Path(__file__).resolve().parent


def work_root() -> Path:
    return course_root() / "student_work" / "visual_odometry"


def default_output() -> Path:
    return work_root() / "camera_calibration.npz"


def choose_output_file(suggested_file: Path) -> Path | None:
    suggested_file = suggested_file.expanduser().resolve()
    suggested_file.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        from ctypes import wintypes

        class OPENFILENAMEW(ctypes.Structure):
            _fields_ = [
                ("lStructSize", wintypes.DWORD),
                ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HINSTANCE),
                ("lpstrFilter", wintypes.LPCWSTR),
                ("lpstrCustomFilter", wintypes.LPWSTR),
                ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPWSTR),
                ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPWSTR),
                ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCWSTR),
                ("lpstrTitle", wintypes.LPCWSTR),
                ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD),
                ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR),
                ("lCustData", wintypes.LPARAM),
                ("lpfnHook", ctypes.c_void_p),
                ("lpTemplateName", wintypes.LPCWSTR),
                ("pvReserved", ctypes.c_void_p),
                ("dwReserved", wintypes.DWORD),
                ("FlagsEx", wintypes.DWORD),
            ]

        buffer = ctypes.create_unicode_buffer(32768)
        buffer.value = suggested_file.name
        dialog = OPENFILENAMEW()
        dialog.lStructSize = ctypes.sizeof(OPENFILENAMEW)
        dialog.lpstrFilter = "NumPy calibration files\0*.npz\0All files\0*.*\0\0"
        dialog.nFilterIndex = 1
        dialog.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
        dialog.nMaxFile = len(buffer)
        dialog.lpstrInitialDir = str(suggested_file.parent)
        dialog.lpstrTitle = "Save camera calibration"
        dialog.lpstrDefExt = "npz"
        dialog.Flags = 0x00000002 | 0x00000800 | 0x00000008
        if ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(dialog)):
            selected = Path(buffer.value).resolve()
            return selected if selected.suffix.lower() == ".npz" else selected.with_suffix(".npz")
        return None

    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        selected = filedialog.asksaveasfilename(
            title="Save camera calibration",
            initialdir=str(suggested_file.parent),
            initialfile=suggested_file.name,
            defaultextension=".npz",
            filetypes=[("NumPy calibration files", "*.npz"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(selected).resolve() if selected else None
    except Exception:
        return None


def detect_board(gray: np.ndarray, board_shape: tuple[int, int]):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_shape, flags)
    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            1e-3,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def calibration_files(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir)
    return sorted([
        *directory.glob("*.png"),
        *directory.glob("*.jpg"),
        *directory.glob("*.jpeg"),
    ])


def calibrate_from_images(
    image_dir: str | Path,
    output_file: str | Path,
    board_shape: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> dict:
    files = calibration_files(image_dir)
    if not files:
        raise FileNotFoundError(f"No calibration images were found in {image_dir}.")

    object_template = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
    object_template[:, :2] = np.mgrid[
        0:board_shape[0], 0:board_shape[1]
    ].T.reshape(-1, 2)
    object_template *= float(square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is not None and current_size != image_size:
            raise ValueError("All calibration images must have the same resolution.")
        image_size = current_size
        found, corners = detect_board(gray, board_shape)
        if found:
            object_points.append(object_template.copy())
            image_points.append(corners)

    if len(object_points) < 8:
        raise RuntimeError(
            f"Only {len(object_points)} valid views were found; at least 8 are required."
        )
    assert image_size is not None
    rms, K, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors = []
    for object_pts, image_pts, rotation, translation in zip(
        object_points, image_points, rotations, translations
    ):
        projected, _ = cv2.projectPoints(
            object_pts, rotation, translation, K, distortion
        )
        point_errors = np.linalg.norm(
            image_pts.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
        )
        errors.append(float(point_errors.mean()))

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mean_error = float(np.mean(errors))
    np.savez(
        output_file,
        K=K,
        distortion=distortion,
        image_size=np.asarray(image_size),
        board_shape=np.asarray(board_shape),
        square_size=float(square_size),
        rms=float(rms),
        mean_reprojection_error=mean_error,
        valid_views=len(object_points),
    )
    return {
        "file": str(output_file.resolve()),
        "valid_views": len(object_points),
        "rms": float(rms),
        "mean_reprojection_error_px": mean_error,
        "image_size": list(image_size),
        "K": K.tolist(),
        "distortion": distortion.ravel().tolist(),
    }


def put_line(image, text, x, y, scale=0.54,
             color=(230, 232, 235), thickness=1):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def fit_image(image: np.ndarray, width: int, height: int):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))),
         max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((height, width, 3), np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas, scale, x, y


def capture_dashboard(
    frame: np.ndarray,
    corners,
    found: bool,
    accepted: int,
    config: CalibrationConfig,
    output_size: tuple[int, int],
) -> np.ndarray:
    width, height = output_size
    width, height = max(width, 960), max(height, 540)
    sx, sy = width / CAPTURE_SIZE[0], height / CAPTURE_SIZE[1]
    ui_scale = min(sx, sy)
    header = max(58, int(round(66 * sy)))
    footer = max(54, int(round(62 * sy)))
    canvas, image_scale, offset_x, offset_y = fit_image(
        frame, width, height - header - footer
    )
    dashboard = np.full((height, width, 3), (24, 27, 32), np.uint8)
    dashboard[header:height - footer] = canvas
    if found and corners is not None:
        display_corners = corners.copy()
        display_corners[:, :, 0] = display_corners[:, :, 0] * image_scale + offset_x
        display_corners[:, :, 1] = (
            display_corners[:, :, 1] * image_scale + offset_y + header
        )
        cv2.drawChessboardCorners(
            dashboard, config.board_shape, display_corners, True
        )

    line_width = max(1, int(round(ui_scale)))
    put_line(dashboard, "CAMERA CALIBRATION", int(22 * sx), int(42 * sy),
             0.76 * ui_scale, (248, 249, 250), max(1, 2 * line_width))
    put_line(dashboard, f"{frame.shape[1]} x {frame.shape[0]}",
             int(430 * sx), int(41 * sy),
             0.52 * ui_scale, (175, 181, 190), line_width)
    state = "BOARD DETECTED" if found else "BOARD NOT DETECTED"
    state_color = (90, 220, 110) if found else (80, 180, 245)
    put_line(dashboard, state, int(800 * sx), int(41 * sy),
             0.55 * ui_scale, state_color, line_width)
    footer_y = height - footer
    cv2.rectangle(dashboard, (0, footer_y), (width, height), (17, 19, 23), -1)
    put_line(dashboard, f"Accepted views: {accepted}/{config.target_views}",
             int(22 * sx), footer_y + int(38 * sy),
             0.56 * ui_scale, (245, 246, 248), line_width)
    put_line(dashboard, "SPACE accept view   Q finish   ESC cancel",
             int(590 * sx), footer_y + int(38 * sy),
             0.48 * ui_scale, (205, 209, 214), line_width)
    return dashboard


def capture_views(config: CalibrationConfig, output_dir: Path) -> tuple[int, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {config.camera_index} could not be opened.")

    accepted = 0
    cancelled = False
    cv2.namedWindow(APP_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(APP_TITLE, *CAPTURE_SIZE)
    try:
        while accepted < config.target_views:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("A camera frame could not be read.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = detect_board(gray, config.board_shape)
            try:
                _, _, window_width, window_height = cv2.getWindowImageRect(APP_TITLE)
                output_size = (window_width, window_height)
            except cv2.error:
                output_size = CAPTURE_SIZE
            dashboard = capture_dashboard(
                frame, corners, found, accepted, config, output_size
            )
            cv2.imshow(APP_TITLE, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(APP_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                cancelled = True
                break
            if key == 27:
                cancelled = True
                break
            if key in (ord("q"), ord("Q")):
                break
            if key == 32 and found:
                path = output_dir / f"view_{accepted:02d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"The image could not be saved: {path}")
                accepted += 1
                time.sleep(0.12)
    finally:
        cap.release()
        try:
            cv2.destroyWindow(APP_TITLE)
        except cv2.error:
            pass
    return accepted, cancelled


class SetupScreen:
    def __init__(self, output_file: Path, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.output_file = output_file
        self.actions: list[tuple[tuple[int, int, int, int], Callable[[], None]]] = []
        self.layout_scale = 1.0
        self.layout_offset = (0, 0)
        self.start_requested = False
        self.message = "Configure the checkerboard and camera."

    def change(self, name: str, amount: int, minimum: int, maximum: int):
        value = int(np.clip(getattr(self.config, name) + amount, minimum, maximum))
        setattr(self.config, name, value)

    def add_button(self, image, rect, text, action, selected=False, enabled=True):
        scale = self.layout_scale
        ox, oy = self.layout_offset
        x1, y1, x2, y2 = (
            int(round(rect[0] * scale + ox)), int(round(rect[1] * scale + oy)),
            int(round(rect[2] * scale + ox)), int(round(rect[3] * scale + oy)),
        )
        fill = (183, 105, 47) if selected else (62, 67, 76)
        if not enabled:
            fill = (49, 52, 58)
        thickness = max(1, int(round(scale)))
        font_scale = max(0.35, 0.55 * scale)
        cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (100, 106, 116), thickness)
        size = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
        put_line(image, text, x1 + (x2 - x1 - size[0]) // 2,
                 y1 + (y2 - y1 + size[1]) // 2, font_scale,
                 (245, 246, 248) if enabled else (125, 129, 136), thickness)
        if enabled:
            self.actions.append(((x1, y1, x2, y2), action))

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        width, height = size or SETUP_SIZE
        width, height = max(width, 700), max(height, 430)
        scale = min(width / SETUP_SIZE[0], height / SETUP_SIZE[1])
        ox = int(round((width - SETUP_SIZE[0] * scale) / 2))
        oy = int(round((height - SETUP_SIZE[1] * scale) / 2))
        self.layout_scale = scale
        self.layout_offset = (ox, oy)
        point = lambda x, y: (int(round(x * scale + ox)), int(round(y * scale + oy)))
        line_width = lambda value=1: max(1, int(round(value * scale)))
        image = np.full((height, width, 3), (242, 244, 247), np.uint8)

        def text(label, x, y, font_scale=0.54,
                 color=(230, 232, 235), thickness=1):
            px, py = point(x, y)
            put_line(image, label, px, py, max(0.30, font_scale * scale),
                     color, line_width(thickness))

        cv2.rectangle(image, point(0, 0), point(SETUP_SIZE[0], 78),
                      (34, 38, 45), -1)
        text("CAMERA CALIBRATION", 34, 47, 0.88, (248, 249, 250), 2)
        self.actions = []

        text("CAMERA", 45, 145, 0.53, (70, 76, 85), 2)
        self.add_button(image, (250, 112, 310, 160), "-",
                        lambda: self.change("camera_index", -1, 0, 9))
        text(str(self.config.camera_index), 347, 147, 0.86, (40, 44, 50), 2)
        self.add_button(image, (405, 112, 465, 160), "+",
                        lambda: self.change("camera_index", 1, 0, 9))

        text("INNER CORNERS", 45, 225, 0.53, (70, 76, 85), 2)
        self.add_button(image, (250, 192, 310, 240), "-",
                        lambda: self.change("columns", -1, 3, 20))
        text(str(self.config.columns), 345, 227, 0.78, (40, 44, 50), 2)
        self.add_button(image, (405, 192, 465, 240), "+",
                        lambda: self.change("columns", 1, 3, 20))
        text("x", 505, 225, 0.60, (90, 96, 105), 1)
        self.add_button(image, (555, 192, 615, 240), "-",
                        lambda: self.change("rows", -1, 3, 20))
        text(str(self.config.rows), 650, 227, 0.78, (40, 44, 50), 2)
        self.add_button(image, (710, 192, 770, 240), "+",
                        lambda: self.change("rows", 1, 3, 20))

        text("TARGET VIEWS", 45, 305, 0.53, (70, 76, 85), 2)
        self.add_button(image, (250, 272, 310, 320), "-",
                        lambda: self.change("target_views", -1, 8, 40))
        text(str(self.config.target_views), 340, 307, 0.78, (40, 44, 50), 2)
        self.add_button(image, (405, 272, 465, 320), "+",
                        lambda: self.change("target_views", 1, 8, 40))

        board_columns = self.config.columns + 1
        board_rows = self.config.rows + 1
        tile = min(300 / board_columns, 210 / board_rows)
        board_width = tile * board_columns
        board_height = tile * board_rows
        board_x = 810 + (300 - board_width) / 2
        board_y = 105 + (210 - board_height) / 2
        for row in range(board_rows):
            for column in range(board_columns):
                color = (35, 35, 35) if (row + column) % 2 == 0 else (255, 255, 255)
                cv2.rectangle(
                    image,
                    point(board_x + column * tile, board_y + row * tile),
                    point(board_x + (column + 1) * tile,
                          board_y + (row + 1) * tile),
                    color,
                    -1,
                )
        cv2.rectangle(
            image, point(board_x, board_y),
            point(board_x + board_width, board_y + board_height),
            (90, 96, 105), line_width()
        )
        corner_radius = max(2, int(round(3.2 * scale)))
        for row in range(1, board_rows):
            for column in range(1, board_columns):
                center = point(board_x + column * tile, board_y + row * tile)
                cv2.circle(image, center, corner_radius + line_width(),
                           (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, center, corner_radius,
                           (40, 95, 225), -1, cv2.LINE_AA)
        label = f"{self.config.columns} x {self.config.rows} INNER CORNERS"
        label_scale = max(0.30, 0.46 * scale)
        label_width = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, line_width()
        )[0][0]
        put_line(image, label,
                 point(960, 340)[0] - label_width // 2,
                 point(0, 340)[1], label_scale, (70, 76, 85), line_width())

        cv2.rectangle(image, point(45, 360), point(1135, 505),
                      (255, 255, 255), -1)
        cv2.rectangle(image, point(45, 360), point(1135, 505),
                      (205, 210, 217), line_width())
        text("CAPTURE", 68, 398, 0.53, (70, 76, 85), 2)
        text("Camera resolution is detected automatically when capture starts.",
             68, 435, 0.52, (50, 55, 62), 1)
        text("Move the checkerboard through different positions, angles, and distances.",
             68, 470, 0.49, (90, 96, 105), 1)

        text("Output file is selected after capture.", 45, 565,
             0.52, (75, 80, 88), 1)
        text(self.message, 45, 615, 0.50, (75, 80, 88), 1)
        self.add_button(image, (760, 585, 1112, 646), "START CAPTURE",
                        self.request_start, selected=True)
        text("Q or ESC: close", 45, 684, 0.48, (105, 111, 120), 1)
        return image

    def request_start(self):
        self.start_requested = True

    def mouse(self, event, x, y, flags, parameter):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for (x1, y1, x2, y2), action in self.actions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                action()
                break


def run_gui(output_file: Path, config: CalibrationConfig | None = None) -> int:
    setup = SetupScreen(output_file, config)
    cv2.namedWindow(APP_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(APP_TITLE, *SETUP_SIZE)
    cv2.setMouseCallback(APP_TITLE, setup.mouse)
    while True:
        try:
            _, _, window_width, window_height = cv2.getWindowImageRect(APP_TITLE)
            setup_size = (window_width, window_height)
        except cv2.error:
            setup_size = SETUP_SIZE
        cv2.imshow(APP_TITLE, setup.render(setup_size))
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(APP_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (27, ord("q"), ord("Q")):
            break
        if not setup.start_requested:
            continue

        setup.start_requested = False
        views_dir = work_root() / "calibration_views" / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        try:
            accepted, cancelled = capture_views(setup.config, views_dir)
            if cancelled:
                setup.message = "Capture cancelled."
            elif accepted < 8:
                setup.message = f"Only {accepted} views captured; at least 8 are required."
            else:
                selected_output = choose_output_file(output_file)
                if selected_output is None:
                    setup.message = "Save cancelled; calibration images were preserved."
                    continue
                output_file = selected_output
                setup.output_file = selected_output
                report = calibrate_from_images(
                    views_dir,
                    selected_output,
                    setup.config.board_shape,
                    setup.config.square_size,
                )
                setup.message = (
                    f"Saved {Path(report['file']).name} | "
                    f"{report['valid_views']} views | "
                    f"mean error {report['mean_reprojection_error_px']:.3f} px"
                )
        except Exception as error:
            setup.message = f"Error: {error}"
        finally:
            cv2.namedWindow(APP_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(APP_TITLE, *SETUP_SIZE)
            cv2.setMouseCallback(APP_TITLE, setup.mouse)
    cv2.destroyAllWindows()
    return 0


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path,
                        help="Calibrate directly from an existing image directory.")
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--columns", type=int, default=9,
                        help="Checkerboard inner-corner columns.")
    parser.add_argument("--rows", type=int, default=6,
                        help="Checkerboard inner-corner rows.")
    parser.add_argument("--views", type=int, default=15)
    parser.add_argument("--square-size", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.images is not None:
        report = calibrate_from_images(
            args.images,
            args.output,
            (args.columns, args.rows),
            args.square_size,
        )
        print(json.dumps(report, indent=2))
        return 0
    config = CalibrationConfig(
        camera_index=args.camera,
        columns=args.columns,
        rows=args.rows,
        target_views=args.views,
        square_size=args.square_size,
    )
    return run_gui(args.output, config)


if __name__ == "__main__":
    raise SystemExit(main())
