"""Interactive monocular visual odometry with ORB or SIFT.

Run without arguments to open the graphical setup screen:

    python visual_odometry.py

The trajectory is relative because monocular visual odometry does not recover
metric scale without an additional source of information.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


APP_TITLE = "Visual odometry"
SETUP_SIZE = (1180, 720)
DASHBOARD_SIZE = (1280, 720)


@dataclass
class VOConfig:
    source_mode: str = "video"
    algorithm: str = "ORB"
    geometry: str = "approximate"
    camera_index: int = 0
    calibration_file: str | None = None
    record_live: bool = True
    save_results: bool = True
    max_features: int = 1500
    ratio: float | None = None
    ransac_threshold_px: float = 1.0
    min_matches: int = 12
    min_pose_inliers: int = 10
    min_inlier_ratio: float = 0.30
    min_parallax_px: float = 1.25
    max_reference_gap: int = 12
    max_display_tracks: int = 180


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


def default_video_directory() -> Path:
    for root in candidate_roots():
        for folder_name in ("imagenes", "images"):
            folder = root / folder_name
            if folder.is_dir():
                return folder.resolve()
    return Path.cwd().resolve()


def choose_file(
    initial_directory: Path,
    title: str,
    windows_filter: str,
    tkinter_filetypes: list[tuple[str, str]],
) -> Path | None:
    """Open a native file selector without adding GUI dependencies."""
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
        dialog = OPENFILENAMEW()
        dialog.lStructSize = ctypes.sizeof(OPENFILENAMEW)
        dialog.lpstrFilter = windows_filter
        dialog.nFilterIndex = 1
        dialog.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
        dialog.nMaxFile = len(buffer)
        dialog.lpstrInitialDir = str(initial_directory)
        dialog.lpstrTitle = title
        dialog.Flags = 0x00080000 | 0x00001000 | 0x00000800
        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(dialog)):
            return Path(buffer.value).resolve()
        return None

    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=str(initial_directory),
            filetypes=tkinter_filetypes,
        )
        root.destroy()
        return Path(selected).resolve() if selected else None
    except Exception:
        return None


def choose_video_file(initial_directory: Path) -> Path | None:
    return choose_file(
        initial_directory,
        "Open a video for visual odometry",
        "Video files\0*.mp4;*.avi;*.mov;*.mkv;*.m4v\0All files\0*.*\0\0",
        [("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
    )


def choose_calibration_file(initial_directory: Path) -> Path | None:
    return choose_file(
        initial_directory,
        "Load camera calibration",
        "NumPy calibration files\0*.npz\0All files\0*.*\0\0",
        [("NumPy calibration files", "*.npz"), ("All files", "*.*")],
    )


def find_default_calibration() -> Path:
    return work_root() / "camera_calibration.npz"


def make_detector(config: VOConfig):
    if config.algorithm == "ORB":
        detector = cv2.ORB_create(nfeatures=config.max_features, fastThreshold=20)
        return detector, cv2.NORM_HAMMING, 0.80 if config.ratio is None else config.ratio
    if config.algorithm == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV installation does not provide SIFT.")
        detector = cv2.SIFT_create(nfeatures=config.max_features, contrastThreshold=0.04)
        return detector, cv2.NORM_L2, 0.75 if config.ratio is None else config.ratio
    raise ValueError("algorithm must be ORB or SIFT")


def approximate_intrinsics(width: int, height: int) -> np.ndarray:
    focal = 0.9 * width
    return np.array(
        [[focal, 0.0, (width - 1) / 2],
         [0.0, focal, (height - 1) / 2],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_calibration(path: str | Path, frame_size: tuple[int, int]):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = np.load(path)
    required = {"K", "distortion", "image_size"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Calibration file is missing: {', '.join(sorted(missing))}")
    K = np.asarray(data["K"], dtype=np.float64).copy()
    distortion = np.asarray(data["distortion"], dtype=np.float64)
    original_width, original_height = np.asarray(data["image_size"], dtype=float)
    width, height = frame_size
    if not np.isclose(original_width / original_height, width / height, rtol=0.01):
        raise ValueError("Video and calibration must use the same aspect ratio.")
    K[0, :] *= width / original_width
    K[1, :] *= height / original_height
    K[2, 2] = 1.0
    return K, distortion


def ratio_matches(descriptors_1, descriptors_2, norm: int, ratio: float):
    if (
        descriptors_1 is None
        or descriptors_2 is None
        or len(descriptors_1) < 2
        or len(descriptors_2) < 2
    ):
        return []
    pairs = cv2.BFMatcher(norm).knnMatch(descriptors_1, descriptors_2, k=2)
    return [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio * second.distance
    ]


def estimate_motion(keypoints_1, keypoints_2, matches, K, config: VOConfig):
    if len(matches) < config.min_matches:
        return None, "insufficient matches"
    points_1 = np.float32([keypoints_1[m.queryIdx].pt for m in matches])
    points_2 = np.float32([keypoints_2[m.trainIdx].pt for m in matches])
    E, ransac_mask = cv2.findEssentialMat(
        points_1,
        points_2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=config.ransac_threshold_px,
    )
    if E is None or ransac_mask is None:
        return None, "essential matrix failed"
    if E.shape != (3, 3):
        E = E[:3, :3]
    ransac_inliers = int(np.count_nonzero(ransac_mask))
    if ransac_inliers < config.min_pose_inliers:
        return None, "few RANSAC inliers"
    pose_inliers, R, translation, pose_mask = cv2.recoverPose(
        E, points_1, points_2, K, mask=ransac_mask.copy()
    )
    pose_inliers = int(pose_inliers)
    if pose_inliers < config.min_pose_inliers:
        return None, "pose rejected"
    pose_mask_bool = pose_mask.ravel().astype(bool)
    inlier_ratio = pose_inliers / max(len(matches), 1)
    if inlier_ratio < config.min_inlier_ratio:
        return None, "low inlier ratio"
    parallax = np.linalg.norm(points_2 - points_1, axis=1)
    median_parallax = float(np.median(parallax[pose_mask_bool]))
    if median_parallax < config.min_parallax_px:
        return None, "low parallax"

    homography_ratio = 0.0
    homography, homography_mask = cv2.findHomography(
        points_1, points_2, cv2.RANSAC, 3.0
    )
    if homography is not None and homography_mask is not None:
        homography_ratio = float(np.count_nonzero(homography_mask) / len(matches))
    quality_warning = ""
    if homography_ratio >= 0.80:
        quality_warning = "planar or rotation dominant"
    return {
        "R": R,
        "translation": translation,
        "points_1": points_1,
        "points_2": points_2,
        "pose_mask": pose_mask_bool,
        "ransac_inliers": ransac_inliers,
        "pose_inliers": pose_inliers,
        "inlier_ratio": inlier_ratio,
        "median_parallax_px": median_parallax,
        "homography_ratio": homography_ratio,
        "quality_warning": quality_warning,
    }, "accepted"


def fit_image(image: np.ndarray, width: int, height: int, color=(20, 23, 28)) -> np.ndarray:
    canvas = np.full((height, width, 3), color, np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def draw_tracks(frame: np.ndarray, motion, max_tracks: int = 180) -> np.ndarray:
    result = frame.copy()
    if motion is None:
        return result
    mask = motion["pose_mask"]
    accepted = np.flatnonzero(mask)
    rejected = np.flatnonzero(~mask)
    accepted_limit = max(1, int(max_tracks * 0.78))
    rejected_limit = max(1, max_tracks - accepted_limit)
    if len(accepted) > accepted_limit:
        accepted = accepted[np.linspace(0, len(accepted) - 1, accepted_limit).astype(int)]
    if len(rejected) > rejected_limit:
        rejected = rejected[np.linspace(0, len(rejected) - 1, rejected_limit).astype(int)]
    indices = np.concatenate((accepted, rejected))
    for index in indices:
        previous = motion["points_1"][index]
        current = motion["points_2"][index]
        inlier = bool(mask[index])
        p0 = tuple(np.round(previous).astype(int))
        p1 = tuple(np.round(current).astype(int))
        color = (60, 220, 80) if inlier else (50, 70, 230)
        cv2.line(result, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(result, p1, 2, color, -1, cv2.LINE_AA)
    return result


def trajectory_canvas(trajectory: list[np.ndarray], width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (246, 247, 249), np.uint8)
    ui_scale = max(0.72, min(width / 430, height / 370))
    px = lambda value: max(1, int(round(value * ui_scale)))
    cv2.putText(canvas, "TOP VIEW: RELATIVE X-Z TRAJECTORY", (px(18), px(27)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56 * ui_scale, (50, 55, 62),
                px(1), cv2.LINE_AA)
    cv2.putText(canvas, "Keep the camera approximately level", (px(18), px(48)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40 * ui_scale, (100, 105, 112),
                px(1), cv2.LINE_AA)
    plot_top, plot_bottom = px(60), height - px(34)
    cv2.rectangle(canvas, (px(15), plot_top), (width - px(15), plot_bottom),
                  (205, 210, 216), px(1))
    points = np.asarray(trajectory, dtype=float)[:, [0, 2]]
    if len(points) > 1:
        window = min(7, len(points))
        smoothed = np.vstack([
            points[max(0, index - window + 1):index + 1].mean(axis=0)
            for index in range(len(points))
        ])
        combined = np.vstack((points, smoothed))
        minimum, maximum = combined.min(axis=0), combined.max(axis=0)
        span = max(float(np.max(maximum - minimum)), 1.0)
        scale = min((width - px(60)) / span,
                    (plot_bottom - plot_top - px(40)) / span)
        center = (minimum + maximum) / 2
        raw_pixels = np.column_stack((
            width / 2 + (points[:, 0] - center[0]) * scale,
            (plot_top + plot_bottom) / 2 - (points[:, 1] - center[1]) * scale,
        )).astype(int)
        smooth_pixels = np.column_stack((
            width / 2 + (smoothed[:, 0] - center[0]) * scale,
            (plot_top + plot_bottom) / 2 - (smoothed[:, 1] - center[1]) * scale,
        )).astype(int)
        cv2.polylines(canvas, [raw_pixels], False, (185, 188, 193), px(1), cv2.LINE_AA)
        cv2.polylines(canvas, [smooth_pixels], False, (210, 100, 45), px(2), cv2.LINE_AA)
        cv2.circle(canvas, tuple(smooth_pixels[0]), px(5), (45, 165, 65), -1)
        cv2.circle(canvas, tuple(smooth_pixels[-1]), px(5), (45, 45, 220), -1)
    cv2.putText(canvas, "gray raw | blue smoothed | green start | red current",
                (px(18), height - px(10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.38 * ui_scale, (90, 95, 102), px(1), cv2.LINE_AA)
    return canvas


def put_line(image, text: str, x: int, y: int, scale=0.54, color=(230, 232, 235), thickness=1):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


class VOEngine:
    def __init__(self, config: VOConfig):
        self.config = config
        self.detector, self.norm, self.ratio = make_detector(config)
        self.reset()

    def reset(self):
        self.K = None
        self.distortion = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.R_world_camera = np.eye(3)
        self.camera_center = np.zeros((3, 1))
        self.trajectory = [np.zeros(3)]
        self.metrics: list[dict] = []
        self.last_motion = None
        self.last_status = "initial frame"
        self.reference_gap = 0

    def process(self, raw_frame: np.ndarray, frame_index: int, source_fps: float):
        started = time.perf_counter()
        height, width = raw_frame.shape[:2]
        if self.K is None:
            if self.config.geometry == "calibrated":
                if not self.config.calibration_file:
                    raise ValueError("A calibration file is required.")
                self.K, self.distortion = load_calibration(
                    self.config.calibration_file, (width, height)
                )
            else:
                self.K = approximate_intrinsics(width, height)
                self.distortion = None
        frame = (
            cv2.undistort(raw_frame, self.K, self.distortion)
            if self.distortion is not None
            else raw_frame
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        matches = []
        motion = None
        status = "initial frame"
        if self.previous_keypoints is not None:
            matches = ratio_matches(
                self.previous_descriptors, descriptors, self.norm, self.ratio
            )
            motion, status = estimate_motion(
                self.previous_keypoints, keypoints, matches, self.K, self.config
            )
            if motion is not None:
                self.R_world_camera = self.R_world_camera @ motion["R"].T
                self.camera_center = (
                    self.camera_center
                    - self.R_world_camera @ motion["translation"]
                )
                self.previous_keypoints = keypoints
                self.previous_descriptors = descriptors
                self.reference_gap = 0
            else:
                self.reference_gap += 1
                if self.reference_gap >= self.config.max_reference_gap:
                    self.previous_keypoints = keypoints
                    self.previous_descriptors = descriptors
                    self.reference_gap = 0
                    status += "; reference reset"
            self.trajectory.append(self.camera_center.ravel().copy())
            elapsed_ms = 1000 * (time.perf_counter() - started)
            pose_inliers = 0 if motion is None else motion["pose_inliers"]
            ransac_inliers = 0 if motion is None else motion["ransac_inliers"]
            self.metrics.append({
                "frame": frame_index,
                "time_s": frame_index / max(source_fps, 1e-9),
                "keypoints": len(keypoints),
                "matches": len(matches),
                "ransac_inliers": ransac_inliers,
                "pose_inliers": pose_inliers,
                "inlier_ratio": pose_inliers / max(len(matches), 1),
                "median_parallax_px": 0.0 if motion is None else motion["median_parallax_px"],
                "homography_ratio": 0.0 if motion is None else motion["homography_ratio"],
                "quality_warning": "" if motion is None else motion["quality_warning"],
                "processing_ms": elapsed_ms,
                "processing_fps": 1000 / max(elapsed_ms, 1e-9),
                "status": status,
                "x": float(self.camera_center[0, 0]),
                "y": float(self.camera_center[1, 0]),
                "z": float(self.camera_center[2, 0]),
            })
        else:
            self.previous_keypoints = keypoints
            self.previous_descriptors = descriptors
        self.last_motion = motion
        self.last_status = status
        return frame, keypoints, matches, motion, status


def make_dashboard(
    frame: np.ndarray,
    engine: VOEngine,
    keypoints,
    matches,
    motion,
    status: str,
    paused: bool,
    source_label: str,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    width, height = output_size or DASHBOARD_SIZE
    width = max(width, 960)
    height = max(height, 540)
    sx, sy = width / DASHBOARD_SIZE[0], height / DASHBOARD_SIZE[1]
    ui_scale = min(sx, sy)
    X = lambda value: int(round(value * sx))
    Y = lambda value: int(round(value * sy))
    T = lambda value: max(0.30, value * ui_scale)
    L = lambda value: max(1, int(round(value * ui_scale)))
    dashboard = np.full((height, width, 3), (24, 27, 32), np.uint8)
    left_width = X(850)
    header_bottom = Y(48)
    footer_top = Y(686)
    video = fit_image(
        draw_tracks(frame, motion, engine.config.max_display_tracks),
        left_width, footer_top - header_bottom
    )
    dashboard[header_bottom:footer_top, :left_width] = video
    cv2.rectangle(dashboard, (0, 0), (width, header_bottom), (34, 38, 45), -1)
    put_line(dashboard, f"{engine.config.algorithm} | {engine.config.geometry}",
             X(18), Y(31), T(0.64), (245, 246, 248), L(2))
    put_line(dashboard, source_label, X(330), Y(30), T(0.52),
             (175, 181, 190), L(1))
    warning = "" if motion is None else motion["quality_warning"]
    display_status = status if not warning else f"{status} | {warning}"
    state_color = (
        (70, 185, 245) if warning
        else (90, 220, 110) if status == "accepted"
        else (80, 180, 245)
    )
    put_line(dashboard, display_status.upper(), X(610), Y(30), T(0.42),
             state_color, L(1))
    if paused:
        put_line(dashboard, "PAUSED", X(760), Y(30), T(0.58),
                 (70, 210, 245), L(2))

    right_width = width - left_width
    trajectory_bottom = Y(418)
    trajectory = trajectory_canvas(engine.trajectory, right_width,
                                   trajectory_bottom - header_bottom)
    dashboard[header_bottom:trajectory_bottom, left_width:] = trajectory
    cv2.rectangle(dashboard, (left_width, trajectory_bottom),
                  (width - 1, height - 1), (34, 38, 45), -1)
    latest = engine.metrics[-1] if engine.metrics else {}
    y = Y(445)
    lines = [
        ("Keypoints", str(len(keypoints))),
        ("Ratio matches", str(len(matches))),
        ("Pose inliers", str(0 if motion is None else motion["pose_inliers"])),
        ("Inlier ratio", f"{latest.get('inlier_ratio', 0):.2f}"),
        ("Median parallax", f"{latest.get('median_parallax_px', 0):.1f} px"),
        ("Homography support", f"{latest.get('homography_ratio', 0):.2f}"),
        ("Processing", f"{latest.get('processing_ms', 0):.1f} ms"),
        ("Processing rate", f"{latest.get('processing_fps', 0):.1f} FPS"),
        ("Accepted poses", str(sum(m["status"] == "accepted" for m in engine.metrics))),
    ]
    for name, value in lines:
        put_line(dashboard, name, left_width + X(22), y, T(0.50),
                 (175, 181, 190), L(1))
        text_width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX,
                                     T(0.54), L(1))[0][0]
        put_line(dashboard, value, width - X(20) - text_width, y,
                 T(0.54), (242, 243, 245), L(1))
        y += Y(27)
    cv2.rectangle(dashboard, (0, footer_top), (width, height), (17, 19, 23), -1)
    put_line(dashboard,
             "SPACE pause/resume   N next frame   R reset   S snapshot   Q stop   ESC exit",
             X(20), Y(709), T(0.48), (205, 209, 214), L(1))
    return dashboard


def summarize(engine: VOEngine) -> dict:
    if not engine.metrics:
        return {
            "frame_pairs": 0,
            "accepted_poses": 0,
            "accepted_fraction": 0.0,
        }
    accepted = np.array([m["status"] == "accepted" for m in engine.metrics])
    processing = np.array([m["processing_ms"] for m in engine.metrics])
    accepted_metrics = [m for m in engine.metrics if m["status"] == "accepted"]
    inliers = np.array([m["pose_inliers"] for m in accepted_metrics])
    ratios = np.array([m["inlier_ratio"] for m in accepted_metrics])
    trajectory = np.asarray(engine.trajectory)
    path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    closure = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    return {
        "algorithm": engine.config.algorithm,
        "geometry": engine.config.geometry,
        "frame_pairs": len(engine.metrics),
        "accepted_poses": int(accepted.sum()),
        "accepted_fraction": float(accepted.mean()),
        "median_pose_inliers": float(np.median(inliers)) if len(inliers) else 0.0,
        "median_inlier_ratio": float(np.median(ratios)) if len(ratios) else 0.0,
        "median_processing_ms": float(np.median(processing)),
        "relative_path_length": path_length,
        "relative_closure_error": closure,
        "normalized_closure_error": closure / max(path_length, 1e-9),
    }


def save_run(engine: VOEngine, dashboard: np.ndarray | None, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    fields = list(engine.metrics[0].keys()) if engine.metrics else ["frame", "status"]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(engine.metrics)
    with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z"])
        writer.writerows(np.asarray(engine.trajectory).tolist())
    (run_dir / "summary.json").write_text(
        json.dumps(summarize(engine), indent=2), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(engine.config), indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(run_dir / "trajectory.png"), trajectory_canvas(engine.trajectory, 720, 720))
    if dashboard is not None:
        cv2.imwrite(str(run_dir / "final_dashboard.png"), dashboard)
    return run_dir


def open_source(source, config: VOConfig):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    return cap, fps


def run_session(
    source,
    config: VOConfig,
    display: bool = True,
    max_frames: int | None = None,
    source_label: str = "",
) -> tuple[VOEngine, np.ndarray | None, bool]:
    cap, source_fps = open_source(source, config)
    engine = VOEngine(config)
    writer = None
    dashboard = None
    paused = False
    step_once = False
    frame_index = -1
    exit_application = False
    record_path = None
    if config.source_mode == "camera" and config.record_live:
        record_path = work_root() / "recordings" / f"route_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        record_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            if display and paused and not step_once:
                try:
                    _, _, window_width, window_height = cv2.getWindowImageRect(APP_TITLE)
                    paused_size = (window_width, window_height)
                except cv2.error:
                    paused_size = DASHBOARD_SIZE
                dashboard = make_dashboard(
                    frame, engine, keypoints, matches, motion, status,
                    True, source_label, paused_size
                )
                cv2.imshow(APP_TITLE, dashboard)
                key = cv2.waitKey(30) & 0xFF
                if cv2.getWindowProperty(APP_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    exit_application = True
                    break
                if key in (ord("q"), ord("Q")):
                    break
                if key == 27:
                    exit_application = True
                    break
                if key == 32:
                    paused = False
                elif key in (ord("n"), ord("N")):
                    step_once = True
                elif key in (ord("r"), ord("R")):
                    engine.reset()
                elif key in (ord("s"), ord("S")) and dashboard is not None:
                    snapshots = work_root() / "snapshots"
                    snapshots.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snapshots / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"), dashboard)
                continue

            cycle_started = time.perf_counter()
            ok, raw_frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if writer is None and record_path is not None:
                height, width = raw_frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create recording: {record_path}")
            if writer is not None:
                writer.write(raw_frame)

            frame, keypoints, matches, motion, status = engine.process(
                raw_frame, frame_index, source_fps
            )
            if display:
                try:
                    _, _, window_width, window_height = cv2.getWindowImageRect(APP_TITLE)
                    output_size = (window_width, window_height)
                except cv2.error:
                    output_size = DASHBOARD_SIZE
                dashboard = make_dashboard(
                    frame, engine, keypoints, matches, motion, status,
                    paused, source_label, output_size
                )
                cv2.imshow(APP_TITLE, dashboard)
                if config.source_mode == "video":
                    elapsed = 1000 * (time.perf_counter() - cycle_started)
                    delay = max(1, int(1000 / source_fps - elapsed))
                else:
                    delay = 1
                key = cv2.waitKey(delay) & 0xFF
                if cv2.getWindowProperty(APP_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    exit_application = True
                    break
                if key in (ord("q"), ord("Q")):
                    break
                if key == 27:
                    exit_application = True
                    break
                if key == 32:
                    paused = True
                    dashboard = make_dashboard(
                        frame, engine, keypoints, matches, motion, status,
                        True, source_label, output_size
                    )
                    cv2.imshow(APP_TITLE, dashboard)
                elif key in (ord("r"), ord("R")):
                    engine.reset()
                elif key in (ord("s"), ord("S")):
                    snapshots = work_root() / "snapshots"
                    snapshots.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snapshots / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"), dashboard)
            step_once = False
            if max_frames is not None and len(engine.metrics) >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            try:
                cv2.destroyWindow(APP_TITLE)
            except cv2.error:
                pass
    return engine, dashboard, exit_application


class SetupScreen:
    def __init__(self):
        self.video_directory = default_video_directory()
        self.video_path: Path | None = None
        default_calibration = find_default_calibration()
        self.calibration_directory = default_calibration.parent
        self.config = VOConfig(calibration_file=None)
        if default_calibration.is_file():
            self.config.calibration_file = str(default_calibration)
        self.actions: list[tuple[tuple[int, int, int, int], Callable[[], None]]] = []
        self.message = "Select a source and configuration."
        self.start_requested = False
        self.layout_scale = 1.0
        self.layout_offset = (0, 0)

    def selected_video(self) -> Path | None:
        return self.video_path

    def add_button(self, image, rect, text, selected, action, enabled=True):
        scale = self.layout_scale
        offset_x, offset_y = self.layout_offset
        x1, y1, x2, y2 = (
            int(round(rect[0] * scale + offset_x)),
            int(round(rect[1] * scale + offset_y)),
            int(round(rect[2] * scale + offset_x)),
            int(round(rect[3] * scale + offset_y)),
        )
        fill = (183, 105, 47) if selected else (62, 67, 76)
        if not enabled:
            fill = (49, 52, 58)
        cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
        line_width = max(1, int(round(scale)))
        font_scale = max(0.35, 0.55 * scale)
        cv2.rectangle(image, (x1, y1), (x2, y2), (100, 106, 116), line_width)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale, line_width)[0]
        put_line(image, text, x1 + (x2 - x1 - size[0]) // 2,
                 y1 + (y2 - y1 + size[1]) // 2, font_scale,
                 (245, 246, 248) if enabled else (125, 129, 136), line_width)
        if enabled:
            self.actions.append(((x1, y1, x2, y2), action))

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        width, height = size or SETUP_SIZE
        width = max(width, 700)
        height = max(height, 430)
        scale = min(width / SETUP_SIZE[0], height / SETUP_SIZE[1])
        offset_x = int(round((width - SETUP_SIZE[0] * scale) / 2))
        offset_y = int(round((height - SETUP_SIZE[1] * scale) / 2))
        self.layout_scale = scale
        self.layout_offset = (offset_x, offset_y)
        point = lambda x, y: (
            int(round(x * scale + offset_x)), int(round(y * scale + offset_y))
        )
        line_width = lambda value=1: max(1, int(round(value * scale)))
        image = np.full((height, width, 3), (242, 244, 247), np.uint8)
        def text(label, x, y, font_scale=0.54,
                 color=(230, 232, 235), thickness=1):
            px, py = point(x, y)
            put_line(image, label, px, py, max(0.30, font_scale * scale),
                     color, line_width(thickness))

        cv2.rectangle(image, point(0, 0), point(SETUP_SIZE[0], 78),
                      (34, 38, 45), -1)
        text("MONOCULAR VISUAL ODOMETRY", 34, 47, 0.88, (248, 249, 250), 2)
        self.actions = []
        text("SOURCE", 45, 125, 0.53, (70, 76, 85), 2)
        self.add_button(image, (210, 96, 390, 142), "RECORDED VIDEO",
                        self.config.source_mode == "video",
                        lambda: self.set_value("source_mode", "video"))
        self.add_button(image, (402, 96, 582, 142), "LIVE CAMERA",
                        self.config.source_mode == "camera",
                        lambda: self.set_value("source_mode", "camera"))

        text("ALGORITHM", 45, 195, 0.53, (70, 76, 85), 2)
        self.add_button(image, (210, 166, 390, 212), "ORB",
                        self.config.algorithm == "ORB",
                        lambda: self.set_value("algorithm", "ORB"))
        self.add_button(image, (402, 166, 582, 212), "SIFT",
                        self.config.algorithm == "SIFT",
                        lambda: self.set_value("algorithm", "SIFT"))

        text("GEOMETRY", 45, 265, 0.53, (70, 76, 85), 2)
        self.add_button(image, (210, 236, 390, 282), "APPROXIMATE",
                        self.config.geometry == "approximate",
                        lambda: self.set_value("geometry", "approximate"))
        calibrated_available = Path(self.config.calibration_file or "").is_file()
        self.add_button(image, (402, 236, 582, 282), "CALIBRATE",
                        self.config.geometry == "calibrated",
                        self.open_calibration)
        if calibrated_available:
            text(Path(self.config.calibration_file).name,
                 620, 265, 0.45, (90, 96, 105))

        cv2.rectangle(image, point(45, 315), point(1135, 482),
                      (255, 255, 255), -1)
        cv2.rectangle(image, point(45, 315), point(1135, 482),
                      (205, 210, 217), line_width())
        if self.config.source_mode == "video":
            text("VIDEO", 68, 350, 0.53, (70, 76, 85), 2)
            self.add_button(image, (68, 375, 262, 424), "OPEN VIDEO...", False,
                            self.open_video)
            video = self.selected_video()
            name = "No video selected" if video is None else video.name
            text(name, 300, 393, 0.62, (40, 44, 50), 2)
        else:
            text("CAMERA INDEX", 68, 355, 0.53, (70, 76, 85), 2)
            self.add_button(image, (210, 337, 264, 384), "-", False, self.camera_down)
            text(str(self.config.camera_index), 300, 369, 0.86, (40, 44, 50), 2)
            self.add_button(image, (355, 337, 409, 384), "+", False, self.camera_up)
            self.add_button(image, (68, 411, 318, 457),
                            "RECORD RAW VIDEO: " + ("ON" if self.config.record_live else "OFF"),
                            self.config.record_live, self.toggle_recording)

        text("OUTPUT", 45, 540, 0.53, (70, 76, 85), 2)
        self.add_button(image, (210, 511, 462, 557),
                        "SAVE RESULTS: " + ("ON" if self.config.save_results else "OFF"),
                        self.config.save_results, self.toggle_saving)

        can_start = (
            (self.config.source_mode == "camera" or self.selected_video() is not None)
            and (self.config.geometry != "calibrated" or calibrated_available)
        )
        self.add_button(image, (760, 585, 1112, 646), "START", can_start,
                        self.request_start, can_start)
        text(self.message, 45, 615, 0.52, (75, 80, 88))
        text("Q or ESC: close", 45, 684, 0.48, (105, 111, 120))
        return image

    def set_value(self, name, value):
        setattr(self.config, name, value)
        self.message = f"{name.replace('_', ' ').title()}: {value}"

    def open_video(self):
        selected = choose_video_file(self.video_directory)
        if selected is not None:
            self.video_path = selected
            self.video_directory = selected.parent
            self.message = f"Video selected: {selected.name}"

    def open_calibration(self):
        selected = choose_calibration_file(self.calibration_directory)
        if selected is not None:
            try:
                with np.load(selected) as data:
                    required = {"K", "distortion", "image_size"}
                    missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"missing {', '.join(sorted(missing))}")
            except Exception as error:
                self.message = f"Invalid calibration file: {error}"
                return
            self.config.calibration_file = str(selected)
            self.config.geometry = "calibrated"
            self.calibration_directory = selected.parent
            self.message = f"Calibration loaded: {selected.name}"

    def camera_down(self):
        self.config.camera_index = max(0, self.config.camera_index - 1)

    def camera_up(self):
        self.config.camera_index += 1

    def toggle_recording(self):
        self.config.record_live = not self.config.record_live

    def toggle_saving(self):
        self.config.save_results = not self.config.save_results

    def request_start(self):
        self.start_requested = True

    def mouse(self, event, x, y, flags, parameter):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for (x1, y1, x2, y2), action in self.actions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                action()
                break


def run_gui() -> int:
    setup = SetupScreen()
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
        source = (
            str(setup.selected_video())
            if setup.config.source_mode == "video"
            else setup.config.camera_index
        )
        source_label = (
            setup.selected_video().name
            if setup.config.source_mode == "video" and setup.selected_video()
            else f"camera {setup.config.camera_index}"
        )
        try:
            cv2.resizeWindow(APP_TITLE, *DASHBOARD_SIZE)
            engine, dashboard, exit_application = run_session(
                source, setup.config, display=True, source_label=source_label
            )
            summary = summarize(engine)
            setup.message = (
                f"Finished: {summary['accepted_poses']}/{summary['frame_pairs']} "
                f"accepted poses, {summary.get('median_processing_ms', 0):.1f} ms median."
            )
            if setup.config.save_results:
                label = f"{setup.config.algorithm.lower()}_{setup.config.geometry}"
                directory = work_root() / datetime.now().strftime("%Y%m%d_%H%M%S") / label
                save_run(engine, dashboard, directory)
                setup.message += f" Saved in {directory}"
            if exit_application:
                break
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
    parser.add_argument("--video", type=Path, help="Process one recorded video.")
    parser.add_argument("--algorithm", choices=["ORB", "SIFT"], default="ORB")
    parser.add_argument("--geometry", choices=["approximate", "calibrated"], default="approximate")
    parser.add_argument("--calibration", type=Path, default=find_default_calibration())
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Do not open OpenCV windows.")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.video is None:
        if args.headless:
            raise SystemExit("--headless requires --video")
        return run_gui()
    config = VOConfig(
        source_mode="video",
        algorithm=args.algorithm,
        geometry=args.geometry,
        calibration_file=str(args.calibration) if args.geometry == "calibrated" else None,
        save_results=not args.no_save,
    )
    engine, dashboard, _ = run_session(
        str(args.video), config, display=not args.headless,
        max_frames=args.max_frames, source_label=args.video.name
    )
    report = summarize(engine)
    print(json.dumps(report, indent=2))
    if config.save_results:
        label = f"{config.algorithm.lower()}_{config.geometry}"
        directory = work_root() / datetime.now().strftime("%Y%m%d_%H%M%S") / label
        print(f"Saved in: {save_run(engine, dashboard, directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
