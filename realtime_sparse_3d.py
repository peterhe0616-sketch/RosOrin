"""Real-time sparse forward 3D model from monocular depth and a 2D LD19 scan."""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import openvino as ov
import roslibpy
import yaml
from open3d.visualization import gui, rendering


@dataclass
class SensorState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    ranges: list[float] = field(default_factory=list)
    angle_min: float = 0.0
    angle_increment: float = 0.0
    range_min: float = 0.02
    range_max: float = 25.0
    camera_matrix: list[float] = field(default_factory=list)
    distortion: list[float] = field(default_factory=list)
    calibration_size: tuple[int, int] = (0, 0)
    odom_from_base: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    scan_count: int = 0
    odom_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rolling sparse 3D view from ROSOrin camera and LD19 data."
    )
    parser.add_argument("--host", default="192.168.3.17")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    parser.add_argument("--video-port", type=int, default=8080)
    parser.add_argument("--config", default="fusion_config.yaml")
    parser.add_argument("--device", help="Override OpenVINO device, e.g. GPU, CPU, AUTO")
    parser.add_argument("--preview", action="store_true", help="Show RGB/depth debug view")
    parser.add_argument("--no-window", action="store_true", help="Run without Open3D window")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model_path = Path(config["model"]["path"])
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    config["model"]["path"] = str(model_path.resolve())
    config["camera_to_base"] = np.asarray(config["camera_to_base"], dtype=np.float64)
    config["lidar_to_camera"] = np.asarray(config["lidar_to_camera"], dtype=np.float64)
    return config


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


class RosSensors:
    def __init__(self, host: str, port: int, state: SensorState) -> None:
        self.state = state
        self.ros = roslibpy.Ros(host=host, port=port)
        self.scan = roslibpy.Topic(
            self.ros, "/scan_raw", "sensor_msgs/msg/LaserScan", queue_length=1
        )
        self.camera_info = roslibpy.Topic(
            self.ros,
            "/ascamera/camera_publisher/rgb0/camera_info",
            "sensor_msgs/msg/CameraInfo",
            throttle_rate=500,
            queue_length=1,
        )
        self.odom = roslibpy.Topic(
            self.ros, "/odom", "nav_msgs/msg/Odometry", queue_length=1
        )

    def connect(self) -> None:
        self.ros.run(timeout=10)
        if not self.ros.is_connected:
            raise RuntimeError("rosbridge connection timed out")
        self.scan.subscribe(self.on_scan)
        self.camera_info.subscribe(self.on_camera_info)
        self.odom.subscribe(self.on_odom)

    def on_scan(self, message: dict) -> None:
        with self.state.lock:
            self.state.ranges = message.get("ranges", [])
            self.state.angle_min = float(message.get("angle_min", 0.0))
            self.state.angle_increment = float(message.get("angle_increment", 0.0))
            self.state.range_min = float(message.get("range_min", 0.02))
            self.state.range_max = float(message.get("range_max", 25.0))
            self.state.scan_count += 1

    def on_camera_info(self, message: dict) -> None:
        with self.state.lock:
            self.state.camera_matrix = [float(value) for value in message.get("k", [])]
            self.state.distortion = [float(value) for value in message.get("d", [])]
            self.state.calibration_size = (
                int(message.get("width", 0)),
                int(message.get("height", 0)),
            )

    def on_odom(self, message: dict) -> None:
        pose = message.get("pose", {}).get("pose", {})
        position = pose.get("position", {})
        orientation = pose.get("orientation", {})
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quaternion_matrix(
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
            float(orientation.get("w", 1.0)),
        )
        transform[:3, 3] = [
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        ]
        with self.state.lock:
            self.state.odom_from_base = transform
            self.state.odom_count += 1

    def snapshot(self) -> dict:
        with self.state.lock:
            return {
                "ranges": list(self.state.ranges),
                "angle_min": self.state.angle_min,
                "angle_increment": self.state.angle_increment,
                "range_min": self.state.range_min,
                "range_max": self.state.range_max,
                "camera_matrix": list(self.state.camera_matrix),
                "distortion": list(self.state.distortion),
                "calibration_size": self.state.calibration_size,
                "odom_from_base": self.state.odom_from_base.copy(),
            }

    def close(self) -> None:
        self.scan.unsubscribe()
        self.camera_info.unsubscribe()
        self.odom.unsubscribe()
        self.ros.terminate()


class DepthModel:
    def __init__(self, model_config: dict, device_override: str | None = None) -> None:
        path = model_config["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Depth model not found: {path}")
        self.width = int(model_config["input_width"])
        self.height = int(model_config["input_height"])
        self.core = ov.Core()
        self.device = device_override or model_config.get("device", "AUTO")
        print(f"OpenVINO devices: {', '.join(self.core.available_devices)}")
        print(f"Compiling depth model on {self.device} ...")
        compile_options = {}
        performance_mode = model_config.get("performance_mode")
        if performance_mode:
            compile_options["PERFORMANCE_HINT"] = str(performance_mode)
        num_streams = model_config.get("num_streams")
        if num_streams:
            compile_options["NUM_STREAMS"] = str(int(num_streams))
        self.compiled = self.core.compile_model(path, self.device, compile_options)
        self.input = self.compiled.input(0)
        self.output = self.compiled.output(0)
        self._thread_local = threading.local()
        optimal_requests = self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        print(
            f"Model input: {self.input.partial_shape}; output: {self.output.partial_shape}; "
            f"optimal requests: {optimal_requests}"
        )

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        tensor = np.transpose(rgb, (2, 0, 1))[None]
        request = getattr(self._thread_local, "request", None)
        if request is None:
            request = self.compiled.create_infer_request()
            self._thread_local.request = request
        prediction = np.asarray(request.infer({self.input: tensor})[self.output]).squeeze()
        return cv2.resize(prediction, (bgr.shape[1], bgr.shape[0]), cv2.INTER_CUBIC)


class CameraRectifier:
    def __init__(self) -> None:
        self.key: tuple | None = None
        self.map_x: np.ndarray | None = None
        self.map_y: np.ndarray | None = None
        self.new_matrix: np.ndarray | None = None

    def apply(
        self,
        frame: np.ndarray,
        matrix_values: list[float],
        distortion_values: list[float],
        calibration_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(matrix_values) != 9 or len(distortion_values) < 4:
            return None
        height, width = frame.shape[:2]
        source_width = calibration_size[0] or width
        source_height = calibration_size[1] or height
        key = (
            width,
            height,
            source_width,
            source_height,
            tuple(matrix_values),
            tuple(distortion_values),
        )
        if key != self.key:
            matrix = np.asarray(matrix_values, dtype=np.float64).reshape(3, 3)
            if (source_width, source_height) != (width, height):
                matrix = matrix.copy()
                matrix[0, :] *= width / source_width
                matrix[1, :] *= height / source_height
                matrix[2, 2] = 1.0
            distortion = np.asarray(distortion_values, dtype=np.float64)
            self.new_matrix, _ = cv2.getOptimalNewCameraMatrix(
                matrix, distortion, (width, height), 0.0, (width, height)
            )
            self.map_x, self.map_y = cv2.initUndistortRectifyMap(
                matrix,
                distortion,
                None,
                self.new_matrix,
                (width, height),
                cv2.CV_32FC1,
            )
            self.key = key
        corrected = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
        return corrected, self.new_matrix.copy()


def project_lidar(snapshot: dict, lidar_to_camera: np.ndarray, camera_matrix: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.asarray(snapshot["ranges"], dtype=np.float64)
    if not len(ranges) or not snapshot["angle_increment"]:
        return np.empty((0, 2)), np.empty(0), np.empty((0, 3))
    angles = snapshot["angle_min"] + np.arange(len(ranges)) * snapshot["angle_increment"]
    valid = (
        np.isfinite(ranges)
        & (ranges >= snapshot["range_min"])
        & (ranges <= snapshot["range_max"])
    )
    ranges = ranges[valid]
    angles = angles[valid]
    lidar_points = np.column_stack(
        (ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges))
    )
    camera_points = transform_points(lidar_to_camera, lidar_points)
    forward = camera_points[:, 2] > 0.1
    camera_points = camera_points[forward]
    lidar_points = lidar_points[forward]
    pixels_h = camera_points @ camera_matrix.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    height, width = shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[inside], camera_points[inside, 2], lidar_points[inside]


def robust_affine(features: np.ndarray, targets: np.ndarray) -> tuple[float, float, float] | None:
    mask = np.isfinite(features) & np.isfinite(targets)
    if np.count_nonzero(mask) < 6:
        return None
    x, y = features[mask], targets[mask]
    for _ in range(4):
        design = np.column_stack((x, np.ones_like(x)))
        scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - (scale * x + shift)
        median = np.median(residual)
        mad = np.median(np.abs(residual - median)) + 1e-6
        keep = np.abs(residual - median) < 2.8 * 1.4826 * mad
        if np.count_nonzero(keep) < 6 or np.all(keep):
            break
        x, y = x[keep], y[keep]
    score = float(np.median(np.abs(y - (scale * x + shift))))
    return float(scale), float(shift), score


class DepthAligner:
    def __init__(self, config: dict) -> None:
        self.minimum_matches = int(config["min_lidar_matches"])
        self.radius = int(config["lidar_match_radius_px"])
        self.ema = float(config["scale_ema"])
        self.mode: str | None = None
        self.scale: float | None = None
        self.shift: float | None = None
        self.matches = 0
        self.residual = math.nan

    def align(self, relative: np.ndarray, pixels: np.ndarray, metric_depth: np.ndarray) -> np.ndarray | None:
        samples: list[float] = []
        targets: list[float] = []
        height, width = relative.shape
        for (u, v), target in zip(pixels, metric_depth):
            x, y = int(round(u)), int(round(v))
            x0, x1 = max(0, x - self.radius), min(width, x + self.radius + 1)
            y0, y1 = max(0, y - self.radius), min(height, y + self.radius + 1)
            patch = relative[y0:y1, x0:x1]
            values = patch[np.isfinite(patch)]
            if len(values):
                samples.append(float(np.median(values)))
                targets.append(float(target))
        if len(samples) < self.minimum_matches:
            return self.metric(relative)

        raw = np.asarray(samples, dtype=np.float64)
        targets_array = np.asarray(targets, dtype=np.float64)
        positive_floor = max(float(np.percentile(raw, 1)) * 0.05, 1e-6)
        candidates = {
            "raw": raw,
            "inverse": 1.0 / np.maximum(raw, positive_floor),
        }
        fits = []
        for mode, features in candidates.items():
            fit = robust_affine(features, targets_array)
            if fit is not None:
                fits.append((fit[2], mode, fit[0], fit[1]))
        if fits:
            score, mode, scale, shift = min(fits, key=lambda item: item[0])
            if self.mode != mode or self.scale is None:
                self.mode, self.scale, self.shift = mode, scale, shift
            else:
                self.scale = (1 - self.ema) * self.scale + self.ema * scale
                self.shift = (1 - self.ema) * self.shift + self.ema * shift
            self.matches = len(samples)
            self.residual = score
        return self.metric(relative)

    def metric(self, relative: np.ndarray) -> np.ndarray | None:
        if self.scale is None or self.shift is None or self.mode is None:
            return None
        if self.mode == "inverse":
            floor = max(float(np.percentile(relative, 1)) * 0.05, 1e-6)
            feature = 1.0 / np.maximum(relative, floor)
        else:
            feature = relative
        return self.scale * feature + self.shift


def sparse_cloud(
    frame: np.ndarray,
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    stride = int(config["pixel_stride"])
    grid = np.zeros((height, width), dtype=bool)
    grid[stride // 2 :: stride, stride // 2 :: stride] = True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(dx, dy)
    threshold = np.percentile(edges, float(config["edge_percentile"]))
    edge_sample = (edges >= threshold) & (np.indices(edges.shape).sum(axis=0) % stride == 0)
    valid = (
        np.isfinite(depth)
        & (depth >= float(config["min_m"]))
        & (depth <= float(config["max_m"]))
        & (grid | edge_sample)
    )
    v, u = np.nonzero(valid)
    limit = int(config["max_points_per_frame"])
    if len(u) > limit:
        indices = np.linspace(0, len(u) - 1, limit).astype(np.int32)
        u, v = u[indices], v[indices]
    z = depth[v, u]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    colors = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[v, u].astype(np.float64) / 255.0
    return points, colors


def dense_pointmap(
    frame: np.ndarray,
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a dense RGB frame into a colored current-frame pointmap."""
    height, width = depth.shape
    stride = max(1, int(config.get("pointmap_stride", 3)))
    v, u = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    u = u.reshape(-1)
    v = v.reshape(-1)
    z = depth[v, u]
    valid = (
        np.isfinite(z)
        & (z >= float(config["min_m"]))
        & (z <= float(config["max_m"]))
    )
    u, v, z = u[valid], v[valid], z[valid]
    limit = max(1, int(config.get("max_current_points", 100000)))
    if len(u) > limit:
        indices = np.linspace(0, len(u) - 1, limit).astype(np.int32)
        u, v, z = u[indices], v[indices], z[indices]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    colors = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[v, u].astype(np.float64) / 255.0
    return points, colors


def voxel_first(points: np.ndarray, colors: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return points, colors
    keys = np.floor(points / size).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[indices], colors[indices]


class CloudHistory:
    def __init__(self, seconds: float, voxel_size: float) -> None:
        self.seconds = seconds
        self.voxel_size = voxel_size
        self.frames: deque[tuple[float, np.ndarray, np.ndarray]] = deque()

    def add(self, timestamp: float, points_odom: np.ndarray, colors: np.ndarray) -> None:
        self.frames.append((timestamp, points_odom, colors))
        while self.frames and timestamp - self.frames[0][0] > self.seconds:
            self.frames.popleft()

    def local_cloud(self, odom_from_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.frames:
            return np.empty((0, 3)), np.empty((0, 3))
        points = np.concatenate([frame[1] for frame in self.frames])
        colors = np.concatenate([frame[2] for frame in self.frames])
        base_from_odom = np.linalg.inv(odom_from_base)
        points = transform_points(base_from_odom, points)
        return voxel_first(points, colors, self.voxel_size)


class IncrementalVoxelMap:
    """Keyframe-based colored voxel map in a fixed odom coordinate frame."""

    def __init__(self, config: dict) -> None:
        self.voxel_size = float(config.get("voxel_size_m", 0.08))
        self.max_points = int(config.get("max_points", 250000))
        self.min_translation = float(config.get("min_translation_m", 0.06))
        self.min_rotation = math.radians(float(config.get("min_rotation_deg", 3.0)))
        self.blend = float(np.clip(config.get("observation_blend", 0.25), 0.0, 1.0))
        self.voxels: dict[tuple[int, int, int], np.ndarray] = {}
        self.last_pose: np.ndarray | None = None
        self.keyframes = 0

    def _is_keyframe(self, pose: np.ndarray) -> bool:
        if self.last_pose is None:
            return True
        translation = float(np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3]))
        relative_rotation = self.last_pose[:3, :3].T @ pose[:3, :3]
        cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        rotation = float(math.acos(cosine))
        return translation >= self.min_translation or rotation >= self.min_rotation

    def add(
        self, points_world: np.ndarray, colors: np.ndarray, pose: np.ndarray
    ) -> bool:
        if not len(points_world) or not self._is_keyframe(pose):
            return False
        keys = np.floor(points_world / self.voxel_size).astype(np.int32)
        _, indices = np.unique(keys, axis=0, return_index=True)
        for index in indices:
            key = tuple(int(value) for value in keys[index])
            observation = np.concatenate((points_world[index], colors[index])).astype(
                np.float32, copy=False
            )
            previous = self.voxels.get(key)
            if previous is None:
                self.voxels[key] = observation.copy()
            else:
                previous *= 1.0 - self.blend
                previous += observation * self.blend
        while len(self.voxels) > self.max_points:
            self.voxels.pop(next(iter(self.voxels)))
        self.last_pose = pose.copy()
        self.keyframes += 1
        return True

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.voxels:
            return np.empty((0, 3)), np.empty((0, 3))
        values = np.asarray(list(self.voxels.values()), dtype=np.float32)
        return values[:, :3], values[:, 3:]


def ground_grid() -> o3d.geometry.LineSet:
    points = []
    lines = []
    colors = []
    for y in np.arange(-10.0, 10.01, 0.5):
        index = len(points)
        points.extend([[-10.0, y, 0.0], [10.0, y, 0.0]])
        lines.append([index, index + 1])
        colors.append([0.25, 0.25, 0.25])
    for x in np.arange(-10.0, 10.01, 0.5):
        index = len(points)
        points.extend([[x, -10.0, 0.0], [x, 10.0, 0.0]])
        lines.append([index, index + 1])
        colors.append([0.25, 0.25, 0.25])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


class CloudViewer:
    def __init__(self, enabled: bool, point_size_px: float) -> None:
        self.enabled = enabled
        self.app = None
        self.window = None
        self.widget = None
        self.scene = None
        self.cloud = None
        self.alive = True
        self.drag_position: tuple[int, int] | None = None
        self.target = np.asarray([2.3, 0.0, 0.45], dtype=np.float32)
        self.eye = np.asarray([-1.8, -3.2, 2.6], dtype=np.float32)
        self.up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if not enabled:
            return

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("ROSOrin Fixed-Frame RGB 3D Map", 1280, 800)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene = self.widget.scene
        self.scene.set_background([0.04, 0.05, 0.06, 1.0])
        self.window.add_child(self.widget)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget.set_on_mouse(self._on_mouse)

        self.point_material = rendering.MaterialRecord()
        self.point_material.shader = "defaultUnlit"
        self.point_material.point_size = max(1.0, float(point_size_px))
        self.cloud = o3d.geometry.PointCloud()

        line_material = rendering.MaterialRecord()
        line_material.shader = "unlitLine"
        line_material.line_width = 1.0
        self.scene.add_geometry("ground", ground_grid(), line_material)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        axes_material = rendering.MaterialRecord()
        axes_material.shader = "defaultUnlit"
        self.scene.add_geometry("world_axes", axes, axes_material)

        robot = o3d.geometry.TriangleMesh.create_box(width=0.42, height=0.30, depth=0.12)
        robot.translate([-0.21, -0.15, 0.025])
        robot.compute_vertex_normals()
        robot.paint_uniform_color([0.10, 0.55, 0.95])
        robot_material = rendering.MaterialRecord()
        robot_material.shader = "defaultLit"
        self.scene.add_geometry("robot", robot, robot_material)

        arrow = o3d.geometry.LineSet()
        arrow.points = o3d.utility.Vector3dVector(
            [[0.0, 0.0, 0.18], [0.48, 0.0, 0.18], [0.34, 0.10, 0.18], [0.34, -0.10, 0.18]]
        )
        arrow.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [1, 3]])
        arrow.colors = o3d.utility.Vector3dVector([[1.0, 0.85, 0.10]] * 3)
        arrow_material = rendering.MaterialRecord()
        arrow_material.shader = "unlitLine"
        arrow_material.line_width = 4.0
        self.scene.add_geometry("robot_forward", arrow, arrow_material)
        self.robot_label = self.widget.add_3d_label(
            np.asarray([0.0, 0.0, 0.30]), "ROBOT"
        )

        bounds = o3d.geometry.AxisAlignedBoundingBox([-2.0, -4.0, -0.1], [8.0, 4.0, 3.0])
        self.widget.setup_camera(60.0, bounds, self.target)
        self._apply_camera()
        self.app.run_one_tick()

    def _on_layout(self, _context: gui.LayoutContext) -> None:
        self.widget.frame = self.window.content_rect

    def _on_close(self) -> bool:
        self.alive = False
        return True

    def _apply_camera(self) -> None:
        self.scene.camera.look_at(self.target, self.eye, self.up)

    def _on_mouse(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        result = gui.Widget.EventCallbackResult
        if event.type == gui.MouseEvent.BUTTON_DOWN:
            self.drag_position = (event.x, event.y)
            return result.CONSUMED
        if event.type == gui.MouseEvent.BUTTON_UP:
            self.drag_position = None
            return result.CONSUMED
        if event.type == gui.MouseEvent.DRAG and self.drag_position is not None:
            previous_x, previous_y = self.drag_position
            dx, dy = event.x - previous_x, event.y - previous_y
            self.drag_position = (event.x, event.y)
            view = self.target - self.eye
            distance = max(float(np.linalg.norm(view)), 0.5)
            forward = np.asarray([view[0], view[1], 0.0], dtype=np.float32)
            forward /= max(float(np.linalg.norm(forward)), 1e-6)
            right = np.cross(forward, self.up)
            metres_per_pixel = distance * 1.25 / max(self.widget.frame.height, 1)
            offset = (right * dx - forward * dy) * metres_per_pixel
            self.target += offset
            self.eye += offset
            self._apply_camera()
            return result.CONSUMED
        if event.type == gui.MouseEvent.WHEEL:
            view = self.eye - self.target
            factor = float(np.exp(np.clip(event.wheel_dy, -3.0, 3.0) * 0.12))
            new_distance = np.clip(np.linalg.norm(view) * factor, 1.0, 20.0)
            self.eye = self.target + view / max(np.linalg.norm(view), 1e-6) * new_distance
            self._apply_camera()
            return result.CONSUMED
        return result.IGNORED

    def update(self, points: np.ndarray, colors: np.ndarray) -> bool:
        if not self.enabled:
            return True
        if not self.alive:
            return False
        self.cloud.points = o3d.utility.Vector3dVector(points)
        self.cloud.colors = o3d.utility.Vector3dVector(colors)
        if self.scene.has_geometry("cloud"):
            self.scene.remove_geometry("cloud")
        self.scene.add_geometry("cloud", self.cloud, self.point_material)
        return bool(self.app.run_one_tick()) and self.alive

    def update_robot_pose(self, world_from_base: np.ndarray) -> None:
        if not self.enabled or not self.alive:
            return
        transform = np.asarray(world_from_base, dtype=np.float64)
        self.scene.set_geometry_transform("robot", transform)
        self.scene.set_geometry_transform("robot_forward", transform)
        label_position = transform_points(
            transform, np.asarray([[0.0, 0.0, 0.30]], dtype=np.float64)
        )[0]
        self.robot_label.position = label_position

    def poll(self) -> bool:
        if not self.enabled:
            return True
        if not self.alive:
            return False
        return bool(self.app.run_one_tick()) and self.alive

    def close(self) -> None:
        if self.enabled and self.window is not None and self.alive:
            self.alive = False
            self.window.close()
            self.app.run_one_tick()


def depth_preview(frame: np.ndarray, metric: np.ndarray | None, lidar_pixels: np.ndarray, aligner: DepthAligner) -> np.ndarray:
    rgb = frame.copy()
    for u, v in lidar_pixels:
        cv2.circle(rgb, (int(round(u)), int(round(v))), 2, (0, 0, 255), -1)
    if metric is None:
        colored = np.zeros_like(frame)
    else:
        normalized = np.clip(metric / 8.0, 0.0, 1.0)
        colored = cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    preview = np.hstack((rgb, colored))
    text = f"mode={aligner.mode or 'waiting'} matches={aligner.matches} residual={aligner.residual:.3f}m"
    cv2.putText(preview, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    depth_model = DepthModel(config["model"], args.device)
    sensors = RosSensors(args.host, args.rosbridge_port, SensorState())
    print(f"Connecting to rosbridge ws://{args.host}:{args.rosbridge_port} ...")
    sensors.connect()
    video_url = (
        f"http://{args.host}:{args.video_port}/stream"
        "?topic=/ascamera/camera_publisher/rgb0/image&type=mjpeg&quality=70"
    )
    capture = cv2.VideoCapture(video_url)
    if not capture.isOpened():
        sensors.close()
        raise RuntimeError(f"Camera stream unavailable: {video_url}")

    rectifier = CameraRectifier()
    aligner = DepthAligner(config["depth"])
    history = CloudHistory(
        float(config["cloud"]["history_seconds"]),
        float(config["cloud"]["voxel_size_m"]),
    )
    viewer = CloudViewer(
        not args.no_window,
        float(config["cloud"].get("point_size_px", 5.0)),
    )
    camera_to_base = config["camera_to_base"]
    lidar_to_camera = config["lidar_to_camera"]
    started = time.monotonic()
    last_report = started
    frames = 0

    try:
        while not stop.is_set():
            if args.duration and time.monotonic() - started >= args.duration:
                break
            if args.max_frames and frames >= args.max_frames:
                break
            ok, raw_frame = capture.read()
            if not ok:
                print("Camera stream disconnected")
                break
            snapshot = sensors.snapshot()
            rectified = rectifier.apply(
                raw_frame,
                snapshot["camera_matrix"],
                snapshot["distortion"],
                snapshot["calibration_size"],
            )
            if rectified is None or not snapshot["ranges"]:
                time.sleep(0.02)
                continue
            frame, camera_matrix = rectified

            inference_start = time.monotonic()
            relative_depth = depth_model.infer(frame)
            inference_ms = (time.monotonic() - inference_start) * 1000.0
            lidar_pixels, lidar_depths, lidar_points = project_lidar(
                snapshot, lidar_to_camera, camera_matrix, frame.shape[:2]
            )
            metric_depth = aligner.align(relative_depth, lidar_pixels, lidar_depths)
            if metric_depth is not None:
                camera_points, colors = sparse_cloud(
                    frame, metric_depth, camera_matrix, config["depth"]
                )
                base_points = transform_points(camera_to_base, camera_points)
                if len(lidar_points):
                    lidar_base = transform_points(camera_to_base @ lidar_to_camera, lidar_points)
                    lidar_colors = np.tile([1.0, 0.15, 0.05], (len(lidar_base), 1))
                    base_points = np.vstack((base_points, lidar_base))
                    colors = np.vstack((colors, lidar_colors))
                odom_points = transform_points(snapshot["odom_from_base"], base_points)
                history.add(time.monotonic(), odom_points, colors)

            local_points, local_colors = history.local_cloud(snapshot["odom_from_base"])
            if not viewer.update(local_points, local_colors):
                break
            if args.preview:
                cv2.imshow(
                    "RGB + metric depth (red = LD19 projection)",
                    depth_preview(frame, metric_depth, lidar_pixels, aligner),
                )
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frames += 1
            now = time.monotonic()
            if now - last_report >= 1.0:
                fps = frames / max(now - started, 1e-6)
                print(
                    f"fps={fps:.1f} inference={inference_ms:.0f}ms "
                    f"lidar_in_view={len(lidar_pixels)} matches={aligner.matches} "
                    f"residual={aligner.residual:.3f}m points={len(local_points)}"
                )
                last_report = now
    finally:
        capture.release()
        sensors.close()
        viewer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
