"""Receive camera, LiDAR and IMU data from the ROSOrin robot on Windows."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import math
import os
import signal
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import roslibpy

WINDOW_NAME = "ROSOrin Camera + LiDAR - ESC to quit"
DISTORTION_TRACKBAR = "Distortion: -10x < 0 > +10x"
DISTORTION_LIMIT = 10.0
DISTORTION_SCALE = 100


@dataclass
class SensorState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    lidar_count: int = 0
    imu_count: int = 0
    camera_count: int = 0
    odom_count: int = 0
    lidar_points: int = 0
    lidar_nearest: float = math.nan
    lidar_ranges: list[float] = field(default_factory=list)
    lidar_angle_min: float = 0.0
    lidar_angle_increment: float = 0.0
    lidar_range_min: float = 0.02
    lidar_range_max: float = 25.0
    camera_matrix: list[float] = field(default_factory=list)
    distortion: list[float] = field(default_factory=list)
    calibration_width: int = 0
    calibration_height: int = 0
    control_keys: str = "IDLE"
    control_linear: float = 0.0
    control_angular: float = 0.0
    accel: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    gyro: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    odom_from_base: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive ROSOrin camera, LiDAR and IMU streams."
    )
    parser.add_argument("--host", default="192.168.3.17", help="Robot IP address")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    parser.add_argument("--video-port", type=int, default=8080)
    parser.add_argument("--quality", type=int, default=60, choices=range(1, 101))
    parser.add_argument("--no-video", action="store_true", help="Do not open camera")
    parser.add_argument(
        "--lidar-range",
        type=float,
        default=6.0,
        help="Displayed LiDAR radius in metres (default: 6)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Receive data without opening windows"
    )
    parser.add_argument(
        "--no-control", action="store_true", help="Disable global WASD control"
    )
    parser.add_argument(
        "--linear-speed", type=float, default=0.2, help="W/S speed in m/s"
    )
    parser.add_argument(
        "--angular-speed", type=float, default=0.69, help="A/D yaw rate in rad/s"
    )
    parser.add_argument(
        "--no-undistort", action="store_true", help="Show the original camera image"
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help="Undistortion crop: 0 crops invalid edges, 1 keeps the full view",
    )
    parser.add_argument(
        "--distortion-strength",
        type=float,
        default=1.0,
        help="Initial correction multiplier from -10.0 to +10.0 (default: +1.0)",
    )
    parser.add_argument(
        "--no-3d", action="store_true", help="Disable the live colored 3D pointmap"
    )
    parser.add_argument("--config", default="fusion_config.yaml")
    parser.add_argument("--device", help="OpenVINO device override, e.g. GPU or CPU")
    parser.add_argument(
        "--duration", type=float, default=0, help="Exit after N seconds; 0 runs forever"
    )
    return parser.parse_args()


def finite_min(values: list[float]) -> float:
    valid = (float(value) for value in values if value is not None)
    return min((value for value in valid if math.isfinite(value)), default=math.nan)


def infer_depth_timed(depth_model: object, frame: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.monotonic()
    depth = depth_model.infer(frame)
    return depth, (time.monotonic() - started) * 1000.0


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (220, 230, 235),
    scale: float = 0.55,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


class CameraUndistorter:
    def __init__(self, alpha: float) -> None:
        self.alpha = min(1.0, max(0.0, alpha))
        self.cache_key: tuple | None = None
        self.map_x: np.ndarray | None = None
        self.map_y: np.ndarray | None = None
        self.new_matrix: np.ndarray | None = None

    def apply(
        self,
        frame: np.ndarray,
        matrix_values: list[float],
        distortion_values: list[float],
        calibration_size: tuple[int, int],
        strength: float = 1.0,
    ) -> tuple[np.ndarray, bool]:
        if len(matrix_values) != 9 or len(distortion_values) < 4:
            return frame, False

        height, width = frame.shape[:2]
        source_width, source_height = calibration_size
        source_width = source_width or width
        source_height = source_height or height
        cache_key = (
            width,
            height,
            source_width,
            source_height,
            tuple(matrix_values),
            tuple(distortion_values),
            self.alpha,
            strength,
        )
        if cache_key != self.cache_key:
            camera_matrix = np.asarray(matrix_values, dtype=np.float64).reshape(3, 3)
            if (source_width, source_height) != (width, height):
                camera_matrix = camera_matrix.copy()
                camera_matrix[0, :] *= width / source_width
                camera_matrix[1, :] *= height / source_height
                camera_matrix[2, 2] = 1.0
            distortion = np.asarray(distortion_values, dtype=np.float64) * strength
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, distortion, (width, height), self.alpha, (width, height)
            )
            self.new_matrix = new_matrix
            self.map_x, self.map_y = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                None,
                new_matrix,
                (width, height),
                cv2.CV_32FC1,
            )
            self.cache_key = cache_key

        corrected = cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return corrected, True


def scaled_camera_matrix(
    matrix_values: list[float],
    calibration_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> np.ndarray | None:
    if len(matrix_values) != 9:
        return None
    width, height = frame_size
    source_width = calibration_size[0] or width
    source_height = calibration_size[1] or height
    matrix = np.asarray(matrix_values, dtype=np.float64).reshape(3, 3)
    if (source_width, source_height) != (width, height):
        matrix = matrix.copy()
        matrix[0, :] *= width / source_width
        matrix[1, :] *= height / source_height
        matrix[2, 2] = 1.0
    return matrix


class WasdController:
    """Poll global W/A/S/D key state and publish Ackermann Twist commands."""

    def __init__(
        self,
        topic: roslibpy.Topic,
        state: SensorState,
        stop: threading.Event,
        linear_speed: float,
        angular_speed: float,
    ) -> None:
        self.topic = topic
        self.state = state
        self.stop = stop
        self.linear_speed = abs(linear_speed)
        self.angular_speed = abs(angular_speed)
        self.thread: threading.Thread | None = None
        self.last_command = (0.0, 0.0)

    @staticmethod
    def pressed(key: str) -> bool:
        if os.name != "nt":
            return False
        return bool(ctypes.windll.user32.GetAsyncKeyState(ord(key.upper())) & 0x8000)

    def start(self) -> None:
        self.topic.advertise()
        self.thread = threading.Thread(target=self.run, name="wasd-control", daemon=True)
        self.thread.start()

    def run(self) -> None:
        while not self.stop.is_set():
            keys = {key for key in "WASD" if self.pressed(key)}
            linear = (float("W" in keys) - float("S" in keys)) * self.linear_speed
            angular = (float("A" in keys) - float("D" in keys)) * self.angular_speed
            command = (linear, angular)

            # Publish continuously while a key is held, and publish zero once on release.
            if keys or command != self.last_command:
                self.topic.publish(
                    roslibpy.Message(
                        {
                            "linear": {"x": linear, "y": 0.0, "z": 0.0},
                            "angular": {"x": 0.0, "y": 0.0, "z": angular},
                        }
                    )
                )
            self.last_command = command
            with self.state.lock:
                self.state.control_keys = "+".join(key for key in "WASD" if key in keys) or "IDLE"
                self.state.control_linear = linear
                self.state.control_angular = angular
            time.sleep(0.05)

    def close(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.topic.unadvertise()


def render_lidar(
    ranges: list[float],
    angle_min: float,
    angle_increment: float,
    sensor_range_min: float,
    sensor_range_max: float,
    display_range: float,
    size: int = 640,
) -> np.ndarray:
    """Render a ROS LaserScan as a forward-up, top-down OpenCV view."""
    canvas = np.full((size, size, 3), (18, 23, 28), dtype=np.uint8)
    margin = 54
    center = (size // 2, size // 2 + 18)
    radius_px = min(center[0] - margin, center[1] - margin, size - center[1] - margin)
    display_range = max(0.5, display_range)
    pixels_per_metre = radius_px / display_range

    # Metre rings and cross axes.
    ring_step = 1.0 if display_range <= 10 else 2.0
    ring = ring_step
    while ring <= display_range + 1e-6:
        ring_px = int(round(ring * pixels_per_metre))
        cv2.circle(canvas, center, ring_px, (58, 67, 73), 1, cv2.LINE_AA)
        draw_text(canvas, f"{ring:g} m", (center[0] + 5, center[1] - ring_px + 17), (125, 139, 148), 0.42)
        ring += ring_step
    cv2.line(canvas, (center[0], center[1] - radius_px), (center[0], center[1] + radius_px), (63, 74, 81), 1)
    cv2.line(canvas, (center[0] - radius_px, center[1]), (center[0] + radius_px, center[1]), (63, 74, 81), 1)

    nearest_point: tuple[int, int] | None = None
    nearest_distance = math.inf
    valid_count = 0
    if ranges and angle_increment:
        values = np.asarray(ranges, dtype=np.float32)
        angles = angle_min + np.arange(values.size, dtype=np.float32) * angle_increment
        valid = (
            np.isfinite(values)
            & (values >= max(0.0, sensor_range_min))
            & (values <= min(display_range, sensor_range_max))
        )
        valid_values = values[valid]
        valid_angles = angles[valid]
        valid_count = int(valid_values.size)

        if valid_count:
            # ROS axes: +x forward, +y left. Screen: forward up, left left.
            px = center[0] - valid_values * np.sin(valid_angles) * pixels_per_metre
            py = center[1] - valid_values * np.cos(valid_angles) * pixels_per_metre
            points = np.column_stack((px, py)).astype(np.int32)

            # Distance bands make nearby obstacles visually prominent.
            for point, distance in zip(points, valid_values):
                ratio = float(distance / display_range)
                if ratio < 0.25:
                    color = (64, 90, 255)
                elif ratio < 0.55:
                    color = (40, 205, 255)
                else:
                    color = (90, 235, 150)
                cv2.circle(canvas, tuple(point), 2, color, -1, cv2.LINE_AA)

            nearest_index = int(np.argmin(valid_values))
            nearest_distance = float(valid_values[nearest_index])
            nearest_point = tuple(int(value) for value in points[nearest_index])

    # Robot footprint and forward direction.
    robot = np.array(
        [
            (center[0], center[1] - 15),
            (center[0] - 10, center[1] + 12),
            (center[0] + 10, center[1] + 12),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, robot, (235, 235, 235), cv2.LINE_AA)
    draw_text(canvas, "FORWARD", (center[0] - 38, center[1] - radius_px - 13), (190, 205, 212), 0.45)
    draw_text(canvas, "LEFT", (center[0] - radius_px - 3, center[1] - 8), (125, 139, 148), 0.42)
    draw_text(canvas, "RIGHT", (center[0] + radius_px - 42, center[1] - 8), (125, 139, 148), 0.42)

    if nearest_point is not None:
        cv2.line(canvas, center, nearest_point, (70, 80, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, nearest_point, 7, (70, 80, 255), 2, cv2.LINE_AA)
        label_pos = (min(nearest_point[0] + 10, size - 105), max(nearest_point[1] - 10, 30))
        draw_text(canvas, f"{nearest_distance:.2f} m", label_pos, (100, 130, 255), 0.52)

    draw_text(canvas, "LiDAR /scan_raw", (18, 28), (235, 240, 242), 0.62)
    nearest_label = f"nearest {nearest_distance:.2f} m" if math.isfinite(nearest_distance) else "waiting for scan"
    draw_text(canvas, f"{valid_count} visible points | {nearest_label}", (18, size - 18), (155, 170, 178), 0.48)
    return canvas


def compose_dashboard(
    camera_frame: np.ndarray | None,
    lidar_frame: np.ndarray,
    control_keys: str,
    linear: float,
    angular: float,
    undistorted: bool,
    distortion_strength: float,
) -> np.ndarray:
    if camera_frame is None:
        content = lidar_frame
    else:
        target_height = lidar_frame.shape[0]
        camera_width = int(camera_frame.shape[1] * target_height / camera_frame.shape[0])
        camera = cv2.resize(camera_frame, (camera_width, target_height), interpolation=cv2.INTER_AREA)
        camera_label = (
            f"Camera / correction {distortion_strength:+.2f}x"
            if undistorted
            else "Camera / raw (0.00x)"
        )
        draw_text(camera, camera_label, (18, 28), (235, 240, 242), 0.62)
        content = np.hstack((camera, lidar_frame))

    header = np.full((48, content.shape[1], 3), (28, 34, 39), dtype=np.uint8)
    active = control_keys != "IDLE"
    color = (80, 225, 150) if active else (155, 170, 178)
    draw_text(header, f"WASD: {control_keys}", (18, 30), color, 0.62)
    draw_text(
        header,
        f"linear.x {linear:+.2f} m/s   angular.z {angular:+.2f} rad/s",
        (190, 30),
        (220, 230, 235),
        0.55,
    )
    return np.vstack((header, content))


def main() -> int:
    args = parse_args()
    state = SensorState()
    stop = threading.Event()

    def request_stop(*_args: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    ros = roslibpy.Ros(host=args.host, port=args.rosbridge_port)
    print(f"Connecting to rosbridge ws://{args.host}:{args.rosbridge_port} ...")
    try:
        ros.run(timeout=10)
    except Exception as exc:
        print(f"rosbridge connection failed: {exc}")
        return 1

    if not ros.is_connected:
        print("rosbridge connection timed out")
        return 1
    print("rosbridge connected")

    lidar = roslibpy.Topic(
        ros,
        "/scan_raw",
        "sensor_msgs/msg/LaserScan",
        throttle_rate=100,
        queue_length=1,
    )
    imu = roslibpy.Topic(
        ros,
        "/imu",
        "sensor_msgs/msg/Imu",
        throttle_rate=20,
        queue_length=1,
    )
    camera_info = roslibpy.Topic(
        ros,
        "/ascamera/camera_publisher/rgb0/camera_info",
        "sensor_msgs/msg/CameraInfo",
        throttle_rate=500,
        queue_length=1,
    )
    odom = roslibpy.Topic(
        ros,
        "/odom",
        "nav_msgs/msg/Odometry",
        throttle_rate=20,
        queue_length=1,
    )
    cmd_vel = roslibpy.Topic(
        ros,
        "/controller/cmd_vel",
        "geometry_msgs/msg/Twist",
        queue_length=1,
    )

    def on_lidar(message: dict) -> None:
        ranges = message.get("ranges", [])
        with state.lock:
            state.lidar_count += 1
            state.lidar_points = len(ranges)
            state.lidar_nearest = finite_min(ranges)
            state.lidar_ranges = ranges
            state.lidar_angle_min = float(message.get("angle_min", 0.0))
            state.lidar_angle_increment = float(message.get("angle_increment", 0.0))
            state.lidar_range_min = float(message.get("range_min", 0.02))
            state.lidar_range_max = float(message.get("range_max", 25.0))

    def on_imu(message: dict) -> None:
        acceleration = message.get("linear_acceleration", {})
        angular_velocity = message.get("angular_velocity", {})
        with state.lock:
            state.imu_count += 1
            state.accel = tuple(float(acceleration.get(axis, math.nan)) for axis in "xyz")
            state.gyro = tuple(float(angular_velocity.get(axis, math.nan)) for axis in "xyz")

    def on_camera_info(message: dict) -> None:
        with state.lock:
            state.camera_matrix = [float(value) for value in message.get("k", [])]
            state.distortion = [float(value) for value in message.get("d", [])]
            state.calibration_width = int(message.get("width", 0))
            state.calibration_height = int(message.get("height", 0))

    def on_odom(message: dict) -> None:
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
        with state.lock:
            state.odom_from_base = transform
            state.odom_count += 1

    lidar.subscribe(on_lidar)
    imu.subscribe(on_imu)
    camera_info.subscribe(on_camera_info)
    odom.subscribe(on_odom)
    print("Subscribed: /scan_raw, /imu, /odom and camera_info")

    controller = None
    if not args.no_control:
        controller = WasdController(
            cmd_vel,
            state,
            stop,
            args.linear_speed,
            args.angular_speed,
        )
        controller.start()
        print("WASD control enabled: /controller/cmd_vel")

    capture = None
    if not args.no_video:
        video_url = (
            f"http://{args.host}:{args.video_port}/stream"
            "?topic=/ascamera/camera_publisher/rgb0/image"
            f"&type=mjpeg&quality={args.quality}"
        )
        print(f"Opening camera: {video_url}")
        capture = cv2.VideoCapture(video_url)
        if not capture.isOpened():
            print("Camera could not be opened; LiDAR and IMU will continue.")
            capture.release()
            capture = None

    depth_model = None
    depth_aligner = None
    pointmap_viewer = None
    fusion_config = None
    project_lidar_3d = None
    dense_pointmap_3d = None
    transform_points_3d = None
    map_builder = None
    pointmap_inference_ms = math.nan
    pointmap_points = 0
    pointmap_executor = None
    pointmap_pending: list[tuple] = []
    pointmap_sequence = 0
    pointmap_async_requests = 1
    if not args.no_3d and not args.headless and capture is not None:
        try:
            from realtime_sparse_3d import (
                CloudViewer,
                DepthAligner,
                DepthModel,
                IncrementalVoxelMap,
                dense_pointmap,
                load_config,
                project_lidar,
                transform_points,
            )

            fusion_config = load_config(args.config)
            depth_model = DepthModel(fusion_config["model"], args.device)
            depth_aligner = DepthAligner(fusion_config["depth"])
            map_builder = IncrementalVoxelMap(fusion_config.get("mapping", {}))
            pointmap_viewer = CloudViewer(
                True, float(fusion_config["cloud"].get("point_size_px", 5.0))
            )
            project_lidar_3d = project_lidar
            dense_pointmap_3d = dense_pointmap
            transform_points_3d = transform_points
            pointmap_async_requests = max(
                1, int(fusion_config["model"].get("async_requests", 4))
            )
            pointmap_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=pointmap_async_requests,
                thread_name_prefix="depth-infer",
            )
            print("Live dense RGB 3D pointmap enabled")
            print(f"Asynchronous depth pipeline: {pointmap_async_requests} requests")
        except Exception as exc:
            print(f"3D pointmap disabled: {exc}")
            print(r"Run with .\.venv3d\Scripts\python.exe to enable OpenVINO/Open3D.")

    started = time.monotonic()
    undistorter = CameraUndistorter(args.undistort_alpha)
    distortion_strength = min(
        DISTORTION_LIMIT, max(-DISTORTION_LIMIT, args.distortion_strength)
    )
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if capture is not None and not args.no_undistort:
            slider_center = int(DISTORTION_LIMIT * DISTORTION_SCALE)
            slider_position = int(
                round(distortion_strength * DISTORTION_SCALE + slider_center)
            )
            cv2.createTrackbar(
                DISTORTION_TRACKBAR,
                WINDOW_NAME,
                slider_position,
                slider_center * 2,
                lambda _position: None,
            )
    last_report = started
    last_counts = (0, 0, 0)

    try:
        while not stop.is_set():
            now = time.monotonic()
            if args.duration > 0 and now - started >= args.duration:
                break

            camera_frame = None
            pointmap_camera_matrix = None
            if capture is not None:
                ok, frame = capture.read()
                if ok:
                    with state.lock:
                        state.camera_count += 1
                        matrix_values = list(state.camera_matrix)
                        distortion_values = list(state.distortion)
                        calibration_size = (
                            state.calibration_width,
                            state.calibration_height,
                        )
                    if args.no_undistort:
                        camera_frame = frame
                        camera_is_undistorted = False
                        distortion_strength = 0.0
                        pointmap_camera_matrix = scaled_camera_matrix(
                            matrix_values,
                            calibration_size,
                            (frame.shape[1], frame.shape[0]),
                        )
                    else:
                        if not args.headless:
                            slider_position = cv2.getTrackbarPos(
                                DISTORTION_TRACKBAR, WINDOW_NAME
                            )
                            slider_center = int(DISTORTION_LIMIT * DISTORTION_SCALE)
                            distortion_strength = (
                                slider_position - slider_center
                            ) / DISTORTION_SCALE
                        camera_frame, camera_is_undistorted = undistorter.apply(
                            frame,
                            matrix_values,
                            distortion_values,
                            calibration_size,
                            distortion_strength,
                        )
                        camera_is_undistorted = (
                            camera_is_undistorted and abs(distortion_strength) > 1e-6
                        )
                        if undistorter.new_matrix is not None:
                            pointmap_camera_matrix = undistorter.new_matrix.copy()
                else:
                    print("Camera stream disconnected; LiDAR and IMU will continue.")
                    capture.release()
                    capture = None
            else:
                time.sleep(0.02)

            with state.lock:
                scan_ranges = list(state.lidar_ranges)
                scan_angle_min = state.lidar_angle_min
                scan_angle_increment = state.lidar_angle_increment
                scan_range_min = state.lidar_range_min
                scan_range_max = state.lidar_range_max
                control_keys = state.control_keys
                control_linear = state.control_linear
                control_angular = state.control_angular
                odom_from_base = state.odom_from_base.copy()

            pointmap_updated = False
            if pointmap_viewer is not None and pointmap_pending:
                completed = [item for item in pointmap_pending if item[1].done()]
                pointmap_pending = [item for item in pointmap_pending if not item[1].done()]
                if completed:
                    sequence, future, inference_frame, inference_matrix, inference_snapshot = max(
                        completed, key=lambda item: item[0]
                    )
                    try:
                        relative_depth, pointmap_inference_ms = future.result()
                        lidar_pixels, lidar_depths, _ = project_lidar_3d(
                            inference_snapshot,
                            fusion_config["lidar_to_camera"],
                            inference_matrix,
                            inference_frame.shape[:2],
                        )
                        metric_depth = depth_aligner.align(
                            relative_depth, lidar_pixels, lidar_depths
                        )
                        if metric_depth is not None:
                            depth_config = {
                                **fusion_config["depth"],
                                **{
                                    key: fusion_config["cloud"][key]
                                    for key in ("pointmap_stride", "max_current_points")
                                    if key in fusion_config["cloud"]
                                },
                            }
                            camera_points, colors = dense_pointmap_3d(
                                inference_frame,
                                metric_depth,
                                inference_matrix,
                                depth_config,
                            )
                            base_points = transform_points_3d(
                                fusion_config["camera_to_base"], camera_points
                            )
                            world_points = transform_points_3d(
                                inference_snapshot["odom_from_base"], base_points
                            )
                            if map_builder.add(
                                world_points,
                                colors,
                                inference_snapshot["odom_from_base"],
                            ):
                                map_points, map_colors = map_builder.arrays()
                                pointmap_points = len(map_points)
                                pointmap_updated = True
                                if not pointmap_viewer.update(map_points, map_colors):
                                    pointmap_viewer.close()
                                    pointmap_viewer = None
                    except Exception as exc:
                        print(f"Asynchronous depth inference failed: {exc}")

            if (
                pointmap_viewer is not None
                and camera_frame is not None
                and pointmap_camera_matrix is not None
                and scan_ranges
                and pointmap_executor is not None
                and len(pointmap_pending) < pointmap_async_requests
            ):
                snapshot = {
                    "ranges": scan_ranges,
                    "angle_min": scan_angle_min,
                    "angle_increment": scan_angle_increment,
                    "range_min": scan_range_min,
                    "range_max": scan_range_max,
                    "odom_from_base": odom_from_base.copy(),
                }
                inference_frame = camera_frame.copy()
                future = pointmap_executor.submit(
                    infer_depth_timed, depth_model, inference_frame
                )
                pointmap_pending.append(
                    (
                        pointmap_sequence,
                        future,
                        inference_frame,
                        pointmap_camera_matrix.copy(),
                        snapshot,
                    )
                )
                pointmap_sequence += 1

            if pointmap_viewer is not None:
                pointmap_viewer.update_robot_pose(odom_from_base)
            if pointmap_viewer is not None and not pointmap_updated:
                if not pointmap_viewer.poll():
                    pointmap_viewer.close()
                    pointmap_viewer = None

            if not args.headless:
                lidar_frame = render_lidar(
                    scan_ranges,
                    scan_angle_min,
                    scan_angle_increment,
                    scan_range_min,
                    scan_range_max,
                    args.lidar_range,
                )
                dashboard = compose_dashboard(
                    camera_frame,
                    lidar_frame,
                    control_keys,
                    control_linear,
                    control_angular,
                    camera_is_undistorted if camera_frame is not None else False,
                    distortion_strength,
                )
                cv2.imshow(WINDOW_NAME, dashboard)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if now - last_report >= 1.0:
                with state.lock:
                    counts = (state.camera_count, state.lidar_count, state.imu_count)
                    points = state.lidar_points
                    nearest = state.lidar_nearest
                    accel = state.accel
                    gyro = state.gyro
                elapsed = now - last_report
                rates = tuple((new - old) / elapsed for new, old in zip(counts, last_counts))
                print(
                    f"camera={rates[0]:5.1f}Hz  lidar={rates[1]:5.1f}Hz/{points}pts "
                    f"nearest={nearest:5.2f}m  imu={rates[2]:5.1f}Hz  "
                    f"accel=({accel[0]:.2f},{accel[1]:.2f},{accel[2]:.2f})  "
                    f"gyro=({gyro[0]:.3f},{gyro[1]:.3f},{gyro[2]:.3f})  "
                    f"map={pointmap_points}pts/{map_builder.keyframes if map_builder else 0}kf "
                    f"infer={pointmap_inference_ms:.0f}ms"
                )
                last_counts = counts
                last_report = now
    finally:
        lidar.unsubscribe()
        imu.unsubscribe()
        camera_info.unsubscribe()
        odom.unsubscribe()
        if controller is not None:
            controller.close()
        if capture is not None:
            capture.release()
        if pointmap_viewer is not None:
            pointmap_viewer.close()
        if pointmap_executor is not None:
            pointmap_executor.shutdown(wait=True, cancel_futures=True)
        cv2.destroyAllWindows()
        ros.terminate()
        print("Stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
